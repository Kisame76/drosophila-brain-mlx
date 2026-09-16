"""CSR-Compiler: FlyWire connectome CSV/Parquet -> source-major CSR pack.

Emits neuron_ids / row_ptr / destinations / signed_counts plus a manifest.json
carrying a SHA-256 per array. Every ID/index mapping, duplicate, bound, sign,
count and dtype range is checked BEFORE anything is written; a failed check
aborts the compile without touching the output directory.

Semantics mirror philshiu/Drosophila_brain_model model.py:create_model():
  * neuron i == row i of the completeness CSV, in file order (NOT sorted)
  * edge weight == df_con['Excitatory x Connectivity'], a signed synaptic
    contact count; the 0.275 mV factor is applied at runtime, not here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

COL_PRE = "Presynaptic_Index"
COL_POST = "Postsynaptic_Index"
COL_W = "Excitatory x Connectivity"
COL_PRE_ID = "Presynaptic_ID"
COL_POST_ID = "Postsynaptic_ID"
COL_CONN = "Connectivity"
COL_EXC = "Excitatory"

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


class CheckLog:
    """Collects named checks; any failure aborts before the write phase."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []
        self.failed = 0

    def check(self, section: str, name: str, ok: bool, detail: str) -> bool:
        self.rows.append((section, name, bool(ok), detail))
        if not ok:
            self.failed += 1
        return bool(ok)

    def note(self, section: str, name: str, detail: str) -> None:
        """Informational: recorded and printed, never fails the compile."""
        self.rows.append((section, name, None, detail))

    def render(self) -> str:
        out, cur = [], None
        for section, name, ok, detail in self.rows:
            if section != cur:
                out.append(f"\n[{section}]")
                cur = section
            mark = "INFO" if ok is None else ("PASS" if ok else "FAIL")
            out.append(f"  {mark}  {name:<34} {detail}")
        return "\n".join(out)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    """Hash of the array's raw little-endian buffer; dtype/shape hashed separately."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- load


def load_neurons(csv_path: Path, log: CheckLog) -> np.ndarray:
    """Read FlyWire root IDs in file order, parsed from raw text as int64.

    Deliberately does not go through pandas' dtype inference: these IDs exceed
    2**53, so any float64 round-trip silently corrupts them.
    """
    raw_lines = csv_path.read_text().splitlines()
    header = raw_lines[0]
    body = [ln for ln in raw_lines[1:] if ln.strip()]

    log.note("neurons", "header", repr(header))
    id_strings = [ln.split(",", 1)[0] for ln in body]

    ids = np.array([int(s) for s in id_strings], dtype=np.int64)
    n = ids.size

    log.check("neurons", "row count > 0", n > 0, f"N = {n}")
    log.check(
        "neurons",
        "ids exceed float64 mantissa",
        bool(ids.max() > 2**53),
        f"max id {ids.max()} vs 2**53 = {2**53} -> int64 parse is mandatory",
    )
    # Hard proof that no precision was lost anywhere in the parse path.
    roundtrip_ok = all(str(int(v)) == s for v, s in zip(ids.tolist(), id_strings))
    log.check("neurons", "int64 parse is exact", roundtrip_ok, f"{n}/{n} ids round-trip to their source text")

    log.check("neurons", "ids unique", len(set(id_strings)) == n, f"{len(set(id_strings))} distinct of {n}")
    log.check("neurons", "ids positive", bool((ids > 0).all()), f"min id {ids.min()}")
    log.check("neurons", "ids fit int64", bool(ids.max() < 2**63 - 1), f"max id {ids.max()}")

    inversions = int((np.diff(ids) <= 0).sum())
    log.note(
        "neurons",
        "file order",
        f"{inversions} non-ascending steps"
        + (" -> ids happen to be ascending, but row order (not id order) defines the model index"
           if inversions == 0 else
           " -> ids are NOT sorted; CSV row order defines the model index"),
    )

    # Bijectivity of the flyid<->index mapping the engine will rely on.
    flyid2i = {int(v): i for i, v in enumerate(ids)}
    bijective = len(flyid2i) == n and all(flyid2i[int(ids[i])] == i for i in range(n))
    log.check("neurons", "flyid<->index bijective", bijective, f"{len(flyid2i)} mappings, all round-trip")

    return ids


def load_edges(parquet_path: Path, log: CheckLog) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    schema = {f.name: str(f.type) for f in table.schema}
    log.note("edges", "parquet schema", json.dumps(schema))
    log.note("edges", "row count", f"{table.num_rows}")

    for col in (COL_PRE, COL_POST, COL_W):
        if not log.check("edges", f"column {col!r} present", col in schema, schema.get(col, "MISSING")):
            raise SystemExit("edge schema mismatch")

    pre = table.column(COL_PRE).to_numpy(zero_copy_only=False)
    post = table.column(COL_POST).to_numpy(zero_copy_only=False)
    w = table.column(COL_W).to_numpy(zero_copy_only=False)

    # Optional provenance columns: the raw FlyWire IDs alongside the indices.
    # They let the index mapping be proven against the data itself rather than
    # only against the compiler's own bookkeeping.
    ids_pre = table.column(COL_PRE_ID).to_numpy(zero_copy_only=False) if COL_PRE_ID in schema else None
    ids_post = table.column(COL_POST_ID).to_numpy(zero_copy_only=False) if COL_POST_ID in schema else None

    # 'Excitatory x Connectivity' should be the product of its two factors.
    if COL_CONN in schema and COL_EXC in schema:
        conn = table.column(COL_CONN).to_numpy(zero_copy_only=False).astype(np.int64)
        exc = table.column(COL_EXC).to_numpy(zero_copy_only=False).astype(np.int64)
        log.check(
            "edges",
            "signed count == Excitatory * Connectivity",
            np.array_equal(exc * conn, w.astype(np.int64)),
            f"factorisation holds for all {w.size} edges; "
            f"Connectivity in [{int(conn.min())}, {int(conn.max())}], "
            f"Excitatory values {sorted(set(np.unique(exc).tolist()))}",
        )
    return pre, post, w, ids_pre, ids_post


# ---------------------------------------------------------------- checks


def check_edges(
    pre: np.ndarray, post: np.ndarray, w: np.ndarray, n: int, log: CheckLog
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = pre.size
    log.check("edges", "arrays equal length", pre.size == post.size == w.size, f"{pre.size}/{post.size}/{w.size}")

    finite = np.isfinite(w.astype(np.float64)) if w.dtype.kind == "f" else np.ones(e, bool)
    log.check("edges", "no NaN/Inf weights", bool(finite.all()), f"{int((~finite).sum())} non-finite")

    wf = w.astype(np.float64)
    integral = np.array_equal(wf, np.rint(wf))
    log.check(
        "edges",
        "weights are exact integers",
        integral,
        f"dtype {w.dtype}; max |w - round(w)| = {float(np.abs(wf - np.rint(wf)).max())}",
    )
    if not integral:
        raise SystemExit("non-integer synaptic counts - int32 contact-count model is invalid")

    counts = np.rint(wf).astype(np.int64)
    log.check(
        "edges",
        "counts fit int32",
        bool(counts.min() >= INT32_MIN and counts.max() <= INT32_MAX),
        f"range [{counts.min()}, {counts.max()}]",
    )

    # Index bounds: these become direct memory offsets in the engine.
    pre_i = pre.astype(np.int64)
    post_i = post.astype(np.int64)
    log.check(
        "edges",
        "pre index in [0, N)",
        bool((pre_i >= 0).all() and (pre_i < n).all()),
        f"range [{pre_i.min()}, {pre_i.max()}], N = {n}",
    )
    log.check(
        "edges",
        "post index in [0, N)",
        bool((post_i >= 0).all() and (post_i < n).all()),
        f"range [{post_i.min()}, {post_i.max()}], N = {n}",
    )
    log.check(
        "edges",
        "index space saturated",
        max(int(pre_i.max()), int(post_i.max())) == n - 1,
        f"max index {max(int(pre_i.max()), int(post_i.max()))} == N-1 = {n - 1}",
    )

    n_pos = int((counts > 0).sum())
    n_neg = int((counts < 0).sum())
    n_zero = int((counts == 0).sum())
    log.check("edges", "sign partition covers all", n_pos + n_neg + n_zero == e, f"+{n_pos} / -{n_neg} / 0:{n_zero}")
    log.note(
        "edges",
        "sign distribution",
        f"excitatory {n_pos} ({100*n_pos/e:.2f}%), inhibitory {n_neg} ({100*n_neg/e:.2f}%), zero {n_zero}",
    )
    log.note(
        "edges",
        "count magnitude",
        f"|count| min {int(np.abs(counts).min())}, max {int(np.abs(counts).max())}, "
        f"mean {float(np.abs(counts).mean()):.3f}, sum {int(counts.sum())}",
    )

    self_loops = int((pre_i == post_i).sum())
    log.note("edges", "self-loops", f"{self_loops} edges with pre == post (kept; Brian2 keeps them too)")

    # Duplicate (pre, post) pairs: Brian2's syn.connect() would build parallel
    # synapses that each deliver g += w, so summing their counts is equivalent.
    key = pre_i * np.int64(n) + post_i
    uniq_keys = np.unique(key)
    n_dup = e - uniq_keys.size
    log.note(
        "edges",
        "duplicate (pre,post) pairs",
        f"{n_dup} duplicate rows ({uniq_keys.size} distinct pairs of {e} rows)",
    )

    return pre_i, post_i, counts


def check_id_mapping(
    ids: np.ndarray, pre: np.ndarray, post: np.ndarray,
    ids_pre, ids_post, log: CheckLog
) -> None:
    """Prove the index mapping against the connectivity table's own ID columns.

    The engine will address neurons purely by index; if the parquet's indices
    ever disagreed with its IDs, every downstream result would be silently
    wired to the wrong cells. This is the only check that can catch that.
    """
    if ids_pre is None or ids_post is None:
        log.note("mapping", "id columns", "absent from parquet - mapping provable only internally")
        return
    ok_pre = np.array_equal(ids[pre], ids_pre.astype(np.int64))
    ok_post = np.array_equal(ids[post], ids_post.astype(np.int64))
    log.check("mapping", "neuron_ids[pre_index] == Presynaptic_ID", ok_pre,
              f"{pre.size} edges; completeness row order agrees with the connectivity table")
    log.check("mapping", "neuron_ids[post_index] == Postsynaptic_ID", ok_post,
              f"{post.size} edges")
    if not (ok_pre and ok_post):
        bad = int((ids[pre] != ids_pre.astype(np.int64)).sum() + (ids[post] != ids_post.astype(np.int64)).sum())
        log.note("mapping", "mismatch count", f"{bad} edge endpoints disagree")


# ---------------------------------------------------------------- build


def build_csr(
    pre: np.ndarray, post: np.ndarray, counts: np.ndarray, n: int, log: CheckLog
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Source-major CSR, rows ordered by source, destinations ascending within a row."""
    order = np.lexsort((post, pre))  # stable and fully determined by the data
    s_pre, s_post, s_cnt = pre[order], post[order], counts[order]

    # Merge parallel edges: identical (pre, post) rows deliver additively at the
    # same 1.8 ms delay, so summing their signed counts preserves semantics.
    key_sorted = s_pre * np.int64(n) + s_post
    first = np.empty(key_sorted.size, dtype=bool)
    first[0] = True
    np.not_equal(key_sorted[1:], key_sorted[:-1], out=first[1:])
    group_start = np.flatnonzero(first)

    n_out = group_start.size
    n_merged = key_sorted.size - n_out
    src = s_pre[group_start]
    dst = s_post[group_start]
    cnt = np.add.reduceat(s_cnt, group_start) if n_merged else s_cnt
    log.note(
        "csr",
        "parallel edges merged",
        f"{key_sorted.size} rows -> {n_out} unique (pre,post) edges"
        + (f", {n_merged} duplicate rows summed" if n_merged else ", no duplicates present"),
    )

    row_counts = np.bincount(src, minlength=n)
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(row_counts, out=row_ptr[1:])

    # --- structural invariants -------------------------------------------
    log.check("csr", "row_ptr[0] == 0", int(row_ptr[0]) == 0, f"{int(row_ptr[0])}")
    log.check("csr", "row_ptr[N] == E", int(row_ptr[-1]) == n_out, f"{int(row_ptr[-1])} == {n_out}")
    log.check("csr", "row_ptr non-decreasing", bool((np.diff(row_ptr) >= 0).all()),
              f"min delta {int(np.diff(row_ptr).min())}")
    log.check("csr", "row_ptr fits int32", int(row_ptr[-1]) <= INT32_MAX, f"E = {n_out} <= {INT32_MAX}")
    log.check("csr", "destinations in [0, N)", bool((dst >= 0).all() and (dst < n).all()),
              f"range [{int(dst.min())}, {int(dst.max())}], N = {n}")

    # Within one row the destinations must strictly ascend; across a row
    # boundary the source index changes instead.
    row_change = np.diff(src) != 0
    dst_ascends = np.diff(dst) > 0
    log.check("csr", "destinations strictly ascending per row",
              bool((row_change | dst_ascends).all()),
              f"{int((~(row_change | dst_ascends)).sum())} violations over {max(n_out - 1, 0)} adjacent pairs")

    # --- round-trip: expand CSR back to COO, compare against aggregated input
    expanded_src = np.repeat(np.arange(n, dtype=np.int64), row_counts)
    log.check("csr", "CSR->COO reproduces sources", np.array_equal(expanded_src, src),
              f"{expanded_src.size} entries compared elementwise")

    ref_order = np.lexsort((post, pre))
    ref_key = pre[ref_order] * np.int64(n) + post[ref_order]
    ref_first = np.empty(ref_key.size, dtype=bool)
    ref_first[0] = True
    np.not_equal(ref_key[1:], ref_key[:-1], out=ref_first[1:])
    ref_gs = np.flatnonzero(ref_first)
    ref_uniq_key = ref_key[ref_gs]
    ref_sum = np.add.reduceat(counts[ref_order], ref_gs)
    pack_key = expanded_src * np.int64(n) + dst
    log.check("csr", "CSR->COO reproduces (pre,post) set", np.array_equal(pack_key, ref_uniq_key),
              f"{pack_key.size} (pre,post) keys identical to the aggregated source table")
    log.check("csr", "CSR->COO reproduces counts", np.array_equal(cnt, ref_sum),
              f"{cnt.size} signed counts identical to the aggregated source table")

    # --- conservation against the untouched input -------------------------
    # All values are integers well below 2**53, so float64 bincount sums are exact.
    abs_total = int(np.abs(counts).sum())
    log.check("csr", "sums exact in float64", abs_total < 2**53,
              f"total |count| = {abs_total} < 2**53 = {2**53}")
    log.check("csr", "total signed count conserved", int(cnt.sum()) == int(counts.sum()),
              f"pack {int(cnt.sum())} == source {int(counts.sum())}")
    in_src = np.bincount(post, minlength=n, weights=counts.astype(np.float64))
    in_pack = np.bincount(dst, minlength=n, weights=cnt.astype(np.float64))
    log.check("csr", "per-neuron in-weight conserved", np.array_equal(in_src, in_pack),
              f"max |delta| = {float(np.abs(in_src - in_pack).max())} over {n} neurons")
    out_src = np.bincount(pre, minlength=n, weights=counts.astype(np.float64))
    out_pack = np.bincount(src, minlength=n, weights=cnt.astype(np.float64))
    log.check("csr", "per-neuron out-weight conserved", np.array_equal(out_src, out_pack),
              f"max |delta| = {float(np.abs(out_src - out_pack).max())} over {n} neurons")

    # --- determinism: rebuild from a shuffled input, demand bit-identity ---
    perm = np.random.default_rng(0).permutation(pre.size)
    p_pre, p_post, p_cnt = pre[perm], post[perm], counts[perm]
    o2 = np.lexsort((p_post, p_pre))
    k2 = p_pre[o2] * np.int64(n) + p_post[o2]
    f2 = np.empty(k2.size, dtype=bool)
    f2[0] = True
    np.not_equal(k2[1:], k2[:-1], out=f2[1:])
    g2 = np.flatnonzero(f2)
    log.check("csr", "deterministic under input shuffle",
              np.array_equal(p_post[o2][g2], dst) and np.array_equal(np.add.reduceat(p_cnt[o2], g2), cnt),
              "rebuilt from a seed-0 row permutation; destinations and counts bit-identical")

    row_ptr32, dst32, cnt32 = row_ptr.astype(np.int32), dst.astype(np.int32), cnt.astype(np.int32)
    log.check("csr", "int32 downcast lossless",
              np.array_equal(row_ptr32.astype(np.int64), row_ptr)
              and np.array_equal(dst32.astype(np.int64), dst)
              and np.array_equal(cnt32.astype(np.int64), cnt),
              "row_ptr / destinations / signed_counts all survive int64 -> int32")

    stats = {
        "edges_in": int(pre.size),
        "edges_out": int(n_out),
        "parallel_edges_merged": int(n_merged),
        "self_loops": int((src == dst).sum()),
        "max_out_degree": int(row_counts.max()),
        "mean_out_degree": float(row_counts.mean()),
        "max_in_degree": int(np.bincount(dst, minlength=n).max()),
        "neurons_without_outputs": int((row_counts == 0).sum()),
        "neurons_without_inputs": int((np.bincount(dst, minlength=n) == 0).sum()),
        "excitatory_edges": int((cnt > 0).sum()),
        "inhibitory_edges": int((cnt < 0).sum()),
        "zero_count_edges": int((cnt == 0).sum()),
        "total_signed_count": int(cnt.sum()),
        "total_abs_count": int(np.abs(cnt).sum()),
        "min_count": int(cnt.min()),
        "max_count": int(cnt.max()),
    }
    return row_ptr32, dst32, cnt32, stats


# ---------------------------------------------------------------- write


# Dataset-specific manifest text. The FlyWire entry is the original and stays
# the default, so the v630 pack keeps producing a byte-identical manifest.
FLYWIRE_SEMANTICS = {
    "index_mapping": "model index i == row i of the completeness CSV, in file order",
    "signed_counts": "signed synaptic contact count == 'Excitatory x Connectivity'; "
                     "sign is the neurotransmitter polarity, magnitude the contact count",
    "weight_application": "runtime multiplies accumulated int32 counts by w_syn = 0.275 mV; "
                          "never accumulate in float",
    "row_ptr": "int32[N+1], row_ptr[i]..row_ptr[i+1] are the outgoing edges of source neuron i",
    "destinations": "int32[E], ascending within each row",
}


def write_pack(
    out_dir: Path,
    arrays: dict[str, np.ndarray],
    stats: dict,
    sources: dict[str, dict],
    log: CheckLog,
    dataset: str = "flywire-v630",
    semantics: dict | None = None,
) -> dict:
    if out_dir.exists():
        # This replaces the output directory wholesale, and both compilers pass
        # --out through unexamined: `--out data/pack` rather than data/pack/v630
        # is one keystroke away and would take every compiled pack under it with
        # no way back. Only something that is already a pack may be replaced.
        # shuffle_pack.write_shuffled_pack keeps its own, stricter form of this
        # check -- there the target must be a *shuffled* pack.
        try:
            existing = json.loads((out_dir / "manifest.json").read_text())
        except (OSError, ValueError):
            existing = None
        if not (isinstance(existing, dict)
                and existing.get("pack_format") == "source-major-csr/1"):
            raise ValueError(f"{out_dir} exists and is not a pack, refusing to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    entries = {}
    for name, arr in arrays.items():
        path = out_dir / f"{name}.npy"
        np.save(path, arr, allow_pickle=False)
        reloaded = np.load(path, allow_pickle=False)
        if not (np.array_equal(reloaded, arr) and reloaded.dtype == arr.dtype):
            raise SystemExit(f"write verification failed for {name}")
        entries[name] = {
            "file": path.name,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "bytes": int(arr.nbytes),
            "sha256": sha256_array(arr),
            "file_sha256": sha256_file(path),
        }
        log.check("write", f"{name} reload identical", True, f"{arr.dtype} {arr.shape}, sha256 {entries[name]['sha256'][:16]}...")

    manifest = {
        "pack_format": "source-major-csr/1",
        "dataset": dataset,
        "neurons": int(arrays["neuron_ids"].size),
        "edges": int(arrays["destinations"].size),
        "arrays": entries,
        "stats": stats,
        "sources": sources,
        "semantics": semantics if semantics is not None else FLYWIRE_SEMANTICS,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--completeness", type=Path, default=root / "data/raw/completeness_630.csv")
    ap.add_argument("--connectivity", type=Path, default=root / "data/raw/connectivity_630.parquet")
    ap.add_argument("--out", type=Path, default=root / "data/pack/v630")
    args = ap.parse_args()

    log = CheckLog()
    print(f"completeness : {args.completeness}")
    print(f"connectivity : {args.connectivity}")
    print(f"out          : {args.out}")

    src_hashes = {
        "completeness_csv": {"path": args.completeness.name, "sha256": sha256_file(args.completeness),
                             "bytes": args.completeness.stat().st_size},
        "connectivity_parquet": {"path": args.connectivity.name, "sha256": sha256_file(args.connectivity),
                                 "bytes": args.connectivity.stat().st_size},
    }

    ids = load_neurons(args.completeness, log)
    pre_raw, post_raw, w_raw, ids_pre, ids_post = load_edges(args.connectivity, log)
    pre, post, counts = check_edges(pre_raw, post_raw, w_raw, ids.size, log)
    check_id_mapping(ids, pre, post, ids_pre, ids_post, log)
    row_ptr, dst, cnt, stats = build_csr(pre, post, counts, ids.size, log)

    print(log.render())
    if log.failed:
        print(f"\n>>> {log.failed} CHECK(S) FAILED - nothing written.")
        return 1

    manifest = write_pack(
        args.out,
        {"neuron_ids": ids, "row_ptr": row_ptr, "destinations": dst, "signed_counts": cnt},
        stats,
        src_hashes,
        log,
    )
    total = sum(e["bytes"] for e in manifest["arrays"].values())
    print(f"\n>>> ALL CHECKS PASSED. pack written: {manifest['neurons']} neurons, "
          f"{manifest['edges']} edges, {total} bytes ({total/2**20:.1f} MiB)")
    print(f"    manifest: {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
