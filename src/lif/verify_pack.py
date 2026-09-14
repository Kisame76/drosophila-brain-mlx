#!/usr/bin/env python3
"""Independent verification of a compiled CSR pack against its raw source files.

Deliberately does NOT reuse the compilers' build path: rows are reconstructed
per source neuron with dictionary aggregation over boolean masks, so a bug in
the vectorised lexsort/reduceat pipeline cannot hide behind itself.

The same independence applies per dataset. The MaleCNS reader below re-derives
the node selection and the transmitter signs from the published Feather tables
using a plain dict mapping, not the searchsorted pipeline in
compile_pack_malecns.py, so the two cannot agree by sharing a mistake.

Handles both packs:

    python -m lif.verify_pack                                  # FlyWire v630
    python -m lif.verify_pack --pack data/pack/male_cns_v1     # MaleCNS v1.0

Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

COL_PRE = "Presynaptic_Index"
COL_POST = "Postsynaptic_Index"
COL_W = "Excitatory x Connectivity"

# MaleCNS. Restated here rather than imported: this file must be able to
# contradict compile_pack_malecns.py, which it cannot do if it shares its
# constants.
MC_SIGNS = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
MC_ALIASES = {"ach": "acetylcholine", "glu": "glutamate"}


def mc_sign(label) -> int:
    if not isinstance(label, str):
        return 0
    t = label.strip().casefold()
    return MC_SIGNS.get(MC_ALIASES.get(t, t), 0)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


class Report:
    def __init__(self) -> None:
        self.failed = 0
        self.section = None

    def __call__(self, section: str, name: str, ok: bool, detail: str) -> bool:
        if section != self.section:
            print(f"\n[{section}]")
            self.section = section
        ok = bool(ok)
        if not ok:
            self.failed += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<38} {detail}")
        return ok



# ---------------------------------------------------------------- datasets
# Each reader checks neuron identity against the raw source and returns the raw
# edge list already mapped into pack index space, so sections 4 and 5 below are
# dataset-agnostic.


def read_flywire(args, r, neuron_ids):
    n = neuron_ids.size

    lines = args.completeness.read_text().splitlines()
    csv_ids = [ln.split(",", 1)[0] for ln in lines[1:] if ln.strip()]
    r("neurons", "row count matches CSV", len(csv_ids) == n, f"{len(csv_ids)} CSV rows vs {n} in pack")
    r("neurons", "ids match CSV text exactly",
      all(str(int(v)) == s for v, s in zip(neuron_ids.tolist(), csv_ids)),
      f"all {n} ids equal their source text, in file order")
    r("neurons", "pack order == CSV order",
      [str(int(v)) for v in neuron_ids.tolist()] == csv_ids,
      "no reordering applied")
    r("neurons", "ids unique", len(set(csv_ids)) == n, f"{len(set(csv_ids))} distinct")

    import pyarrow.parquet as pq

    table = pq.read_table(args.connectivity)
    pre = table.column(COL_PRE).to_numpy(zero_copy_only=False).astype(np.int64)
    post = table.column(COL_POST).to_numpy(zero_copy_only=False).astype(np.int64)
    w = np.rint(table.column(COL_W).to_numpy(zero_copy_only=False).astype(np.float64)).astype(np.int64)
    r("raw", "parquet rows", pre.size == table.num_rows, f"{table.num_rows} rows")
    return pre, post, w


def read_malecns(args, r, neuron_ids):
    """Re-derive selection, signs and edges from the published Feather tables.

    Mapping is a Python dict keyed on body ID, not a sorted searchsorted, and
    the tables are read column-wise rather than batch-wise. Both differ from the
    compiler on purpose.
    """
    from pyarrow import feather

    n = neuron_ids.size

    annot = feather.read_table(args.annotations, columns=["bodyId", "superclass"])
    body = annot["bodyId"].to_numpy(zero_copy_only=False).astype(np.int64).tolist()
    sup = annot["superclass"].to_pylist()
    selected = sorted(b for b, c in zip(body, sup) if c is not None)

    r("neurons", "selection count matches pack", len(selected) == n,
      f"{len(selected)} bodies with a superclass vs {n} in pack")
    r("neurons", "ids match the selection exactly", neuron_ids.tolist() == selected,
      f"all {n} ids equal sorted(bodyId where superclass is not null)")
    r("neurons", "ids unique", len(set(selected)) == n, f"{len(set(selected))} distinct")

    idx = {b: i for i, b in enumerate(selected)}
    r("neurons", "id -> index map is bijective", len(idx) == n, f"{len(idx)} mappings")

    nt = feather.read_table(args.neurotransmitters, columns=["body", "consensus_nt"])
    nt_body = nt["body"].to_numpy(zero_copy_only=False).astype(np.int64).tolist()
    sign_of_body = {b: mc_sign(l) for b, l in zip(nt_body, nt["consensus_nt"].to_pylist())}
    sign = np.zeros(n, dtype=np.int64)
    for b, i in idx.items():
        sign[i] = sign_of_body.get(b, 0)
    r("neurons", "signed sources present", int((sign != 0).sum()) > 0,
      f"excitatory {int((sign > 0).sum())}, inhibitory {int((sign < 0).sum())}, "
      f"unsigned {int((sign == 0).sum())} of {n}")

    reader = feather.read_table(args.connectivity, columns=["body_pre", "body_post", "weight"])
    b_pre = reader["body_pre"].to_numpy(zero_copy_only=False).astype(np.int64)
    b_post = reader["body_post"].to_numpy(zero_copy_only=False).astype(np.int64)
    weight = reader["weight"].to_numpy(zero_copy_only=False).astype(np.int64)
    r("raw", "feather rows", b_pre.size == reader.num_rows, f"{reader.num_rows} rows")

    get = idx.get
    pre_i = np.fromiter((get(b, -1) for b in b_pre.tolist()), dtype=np.int64, count=b_pre.size)
    post_i = np.fromiter((get(b, -1) for b in b_post.tolist()), dtype=np.int64, count=b_post.size)
    keep = (pre_i >= 0) & (post_i >= 0) & (weight > 0)
    keep &= np.where(pre_i >= 0, sign[np.maximum(pre_i, 0)], 0) != 0
    r("raw", "edges surviving the published filter", bool(keep.any()),
      f"{int(keep.sum())} of {b_pre.size} rows: both endpoints selected, "
      f"presynaptic transmitter signed, weight > 0")

    pre_i = pre_i[keep]
    post_i = post_i[keep]
    w = sign[pre_i] * weight[keep]
    return pre_i, post_i, w


READERS = {"flywire": read_flywire, "male-cns": read_malecns}


def reader_for(dataset: str):
    for prefix, fn in READERS.items():
        if dataset.startswith(prefix):
            return fn
    raise SystemExit(f"no verifier for dataset {dataset!r}; known prefixes: {sorted(READERS)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--pack", type=Path, default=root / "data/pack/v630")
    ap.add_argument("--completeness", type=Path, default=root / "data/raw/completeness_630.csv")
    ap.add_argument("--connectivity", type=Path, default=None,
                    help="connectivity table; defaults to the one the pack's dataset implies")
    mc = root / "data/raw/male_cns"
    ap.add_argument("--annotations", type=Path, default=mc / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    ap.add_argument("--neurotransmitters", type=Path, default=mc / "body-neurotransmitters-male-cns-v1.0.feather")
    ap.add_argument("--samples", type=int, default=400, help="source neurons to reconstruct row-by-row")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    r = Report()
    manifest = json.loads((args.pack / "manifest.json").read_text())
    dataset = manifest["dataset"]
    read_raw = reader_for(dataset)
    if args.connectivity is None:
        args.connectivity = (root / "data/raw/connectivity_630.parquet" if read_raw is read_flywire
                             else mc / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    print(f"pack     : {args.pack}")
    print(f"format   : {manifest['pack_format']}  dataset {dataset}")

    # ---- 1. manifest integrity -----------------------------------------
    arrays = {}
    for name, meta in manifest["arrays"].items():
        path = args.pack / meta["file"]
        r("manifest", f"{name} file sha256", sha256_file(path) == meta["file_sha256"], meta["file_sha256"][:24] + "...")
        a = np.load(path, allow_pickle=False)
        arrays[name] = a
        r("manifest", f"{name} buffer sha256", sha256_array(a) == meta["sha256"], f"{a.dtype} {a.shape}")
        r("manifest", f"{name} dtype/shape", str(a.dtype) == meta["dtype"] and list(a.shape) == meta["shape"],
          f"declared {meta['dtype']} {meta['shape']}")

    source_paths = {
        "completeness_csv": args.completeness, "connectivity_parquet": args.connectivity,
        "annotations": args.annotations, "neurotransmitters": args.neurotransmitters,
        "connectivity": args.connectivity,
    }
    for key, meta in manifest["sources"].items():
        path = source_paths.get(key)
        if path is None:
            r("manifest", f"source {key}", False, "manifest names a source this verifier cannot locate")
            continue
        r("manifest", f"source {key}", sha256_file(path) == meta["sha256"],
          f"{meta['bytes']} bytes, {meta['sha256'][:24]}...")

    neuron_ids = arrays["neuron_ids"]
    row_ptr = arrays["row_ptr"]
    dest = arrays["destinations"]
    cnt = arrays["signed_counts"]
    n, e = neuron_ids.size, dest.size

    # ---- 2. neuron identity and the raw edge list, per dataset ----------
    pre, post, w = read_raw(args, r, neuron_ids)

    # ---- 3. structural invariants ---------------------------------------
    r("structure", "row_ptr length N+1", row_ptr.size == n + 1, f"{row_ptr.size}")
    r("structure", "row_ptr[0] == 0", int(row_ptr[0]) == 0, f"{int(row_ptr[0])}")
    r("structure", "row_ptr[N] == E", int(row_ptr[-1]) == e, f"{int(row_ptr[-1])} == {e}")
    r("structure", "row_ptr non-decreasing", bool((np.diff(row_ptr) >= 0).all()), f"min delta {int(np.diff(row_ptr).min())}")
    r("structure", "destinations in [0, N)", bool((dest >= 0).all() and (dest < n).all()),
      f"[{int(dest.min())}, {int(dest.max())}]")
    r("structure", "destinations/counts same length", dest.size == cnt.size, f"{dest.size} / {cnt.size}")
    r("structure", "dtypes are int32", dest.dtype == np.int32 and cnt.dtype == np.int32 and row_ptr.dtype == np.int32,
      f"{row_ptr.dtype} / {dest.dtype} / {cnt.dtype}")
    r("structure", "neuron_ids dtype int64", neuron_ids.dtype == np.int64, str(neuron_ids.dtype))

    src_of_edge = np.repeat(np.arange(n, dtype=np.int64), np.diff(row_ptr.astype(np.int64)))
    r("structure", "row expansion length == E", src_of_edge.size == e, f"{src_of_edge.size}")
    row_change = np.diff(src_of_edge) != 0
    r("structure", "destinations ascend within rows",
      bool((row_change | (np.diff(dest.astype(np.int64)) > 0)).all()),
      "strict ascent inside every row")

    # ---- 4. independent per-row reconstruction from the raw edges -------
    r("raw", "pack edges <= raw rows", e <= pre.size, f"{e} packed vs {pre.size} raw (difference = merged duplicates)")
    r("raw", "total signed count conserved", int(cnt.sum()) == int(w.sum()),
      f"pack {int(cnt.sum())} == raw {int(w.sum())}")

    rng = random.Random(args.seed)
    # Bias the sample towards neurons that actually have outgoing edges, but
    # keep some empty rows in to exercise the row_ptr[i] == row_ptr[i+1] case.
    out_deg = np.diff(row_ptr.astype(np.int64))
    with_out = np.flatnonzero(out_deg > 0)
    without_out = np.flatnonzero(out_deg == 0)
    sample = [int(x) for x in rng.sample(list(with_out), min(args.samples, with_out.size))]
    if without_out.size:
        sample += [int(x) for x in rng.sample(list(without_out), min(20, without_out.size))]
    sample.append(int(with_out[int(np.argmax(out_deg[with_out]))]))  # the hub with the most outputs

    mismatches, checked_edges = [], 0
    for i in sample:
        mask = pre == i
        agg: dict[int, int] = {}
        for d, c in zip(post[mask].tolist(), w[mask].tolist()):
            agg[d] = agg.get(d, 0) + c
        expect = sorted(agg.items())
        lo, hi = int(row_ptr[i]), int(row_ptr[i + 1])
        got = list(zip(dest[lo:hi].tolist(), cnt[lo:hi].tolist()))
        checked_edges += len(got)
        if got != expect:
            mismatches.append((i, len(expect), len(got)))

    r("rows", f"{len(sample)} rows reconstructed independently", not mismatches,
      f"{checked_edges} edges compared; {len(mismatches)} mismatched rows"
      + (f" -> {mismatches[:5]}" if mismatches else ""))

    # ---- 5. global aggregate cross-check --------------------------------
    in_raw = np.bincount(post, minlength=n, weights=w.astype(np.float64))
    in_pack = np.bincount(dest.astype(np.int64), minlength=n, weights=cnt.astype(np.float64))
    r("aggregate", "per-neuron in-weight matches raw", np.array_equal(in_raw, in_pack),
      f"max |delta| {float(np.abs(in_raw - in_pack).max())} over {n} neurons")
    out_raw = np.bincount(pre, minlength=n, weights=w.astype(np.float64))
    out_pack = np.bincount(src_of_edge, minlength=n, weights=cnt.astype(np.float64))
    r("aggregate", "per-neuron out-weight matches raw", np.array_equal(out_raw, out_pack),
      f"max |delta| {float(np.abs(out_raw - out_pack).max())} over {n} neurons")

    pack_pairs = src_of_edge * np.int64(n) + dest.astype(np.int64)
    r("aggregate", "pack (pre,post) pairs unique", np.unique(pack_pairs).size == e,
      f"{np.unique(pack_pairs).size} distinct of {e}")
    r("aggregate", "pack pair set == raw pair set",
      np.array_equal(np.unique(pack_pairs), np.unique(pre * np.int64(n) + post)),
      f"{np.unique(pre * np.int64(n) + post).size} distinct raw pairs")

    # ---- 6. declared stats are truthful ---------------------------------
    st = manifest["stats"]
    r("stats", "edges_out", st["edges_out"] == e, f"{st['edges_out']}")
    r("stats", "excitatory_edges", st["excitatory_edges"] == int((cnt > 0).sum()), f"{st['excitatory_edges']}")
    r("stats", "inhibitory_edges", st["inhibitory_edges"] == int((cnt < 0).sum()), f"{st['inhibitory_edges']}")
    r("stats", "self_loops", st["self_loops"] == int((src_of_edge == dest).sum()), f"{st['self_loops']}")
    r("stats", "max_out_degree", st["max_out_degree"] == int(out_deg.max()), f"{st['max_out_degree']}")
    r("stats", "total_signed_count", st["total_signed_count"] == int(cnt.sum()), f"{st['total_signed_count']}")

    print(f"\n>>> {'VERIFIED' if not r.failed else str(r.failed) + ' CHECK(S) FAILED'}: "
          f"{n} neurons, {e} edges, {sum(m['bytes'] for m in manifest['arrays'].values())} bytes")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
