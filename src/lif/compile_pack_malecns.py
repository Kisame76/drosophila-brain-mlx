"""Compile the published MaleCNS v1.0 tables into the same source-major CSR pack.

Why a second front-end rather than a flag on compile_pack.py: the two datasets
agree on nothing at the input layer. FlyWire ships one CSV of neurons plus one
parquet whose rows already carry model indices and a pre-signed
'Excitatory x Connectivity' column. MaleCNS ships three Feather tables of raw
body IDs, and the sign has to be derived from a separate neurotransmitter table.
Everything downstream of `(ids, pre, post, counts)` is shared: the same
check_edges, the same build_csr, the same write_pack, the same manifest hashes.

The selection rules are taken from the MaleCNS release and match the
materialization flyBrain publishes as `male-cns-v1.0-superclass-non-null-known-nt`,
so the two packs are comparable:

  nodes  annotation rows whose `superclass` is non-null, ordered by ascending bodyId
  edges  kept when both endpoints are selected nodes AND the presynaptic body's
         `consensus_nt` is acetylcholine (+1), GABA (-1) or glutamate (-1)

A neuron with an unknown, modulatory or missing transmitter keeps its node -- it
still receives and still integrates -- but contributes no outgoing edges. That is
a property of the published data, not a modelling choice, and the count of
dropped edges is recorded in the manifest.

Note this is a different animal from a different specimen: MaleCNS is male and
covers brain plus ventral nerve cord, FlyWire is female and brain-only. Spike
counts from the two packs are not comparable and the parity gate does not span
them. What carries over is the engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from lif.compile_pack import (
    CheckLog,
    build_csr,
    check_edges,
    sha256_file,
    write_pack,
)

BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
)
SOURCE_FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "connectivity": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
DOWNLOAD_PAGE = "https://male-cns.janelia.org/download/"
LICENSE = "CC BY 4.0"

COL_ANNOT = ("bodyId", "superclass")
COL_NT = ("body", "consensus_nt")
COL_CONN = ("body_pre", "body_post", "weight")

# Polarity of the three transmitters the release calls known. Everything else
# (octopamine, dopamine, serotonin, unclear, null) has no sign in this model.
KNOWN_SIGNS = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
NT_ALIASES = {"ach": "acetylcholine", "glu": "glutamate"}


def nt_sign(label: str | None) -> int:
    if label is None:
        return 0
    norm = label.strip().casefold()
    return KNOWN_SIGNS.get(NT_ALIASES.get(norm, norm), 0)


# ---------------------------------------------------------------- nodes
def load_nodes(annot_path: Path, log: CheckLog) -> np.ndarray:
    """Selected body IDs, ascending. Position in this array is the model index."""
    from pyarrow import feather, ipc

    schema = ipc.open_file(annot_path).schema
    log.note("nodes", "annotation schema", json.dumps([f.name for f in schema]))
    missing = [c for c in COL_ANNOT if c not in schema.names]
    if not log.check("nodes", "annotation columns present", not missing, f"missing {missing}"):
        raise SystemExit("annotation schema mismatch")

    table = feather.read_table(annot_path, columns=list(COL_ANNOT))
    body = table["bodyId"].to_numpy(zero_copy_only=False)
    superclass = table["superclass"].to_pylist()

    log.check(
        "nodes",
        "bodyId is an integer column",
        body.dtype.kind in "iu",
        f"dtype {body.dtype}, {body.size} rows",
    )
    body = body.astype(np.int64, copy=False)
    log.check("nodes", "bodyId unique over all rows", np.unique(body).size == body.size,
              f"{np.unique(body).size} distinct of {body.size}")
    log.check("nodes", "bodyId positive", bool((body > 0).all()), f"min {int(body.min())}")

    keep = np.fromiter((v is not None for v in superclass), dtype=bool, count=body.size)
    ids = np.sort(body[keep])

    counts = Counter(v for v in superclass if v is not None)
    log.note("nodes", "superclass distribution",
             ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    log.check("nodes", "selection non-empty", ids.size > 0, f"N = {ids.size} of {body.size} annotated bodies")
    log.check("nodes", "selected ids strictly ascending", bool((np.diff(ids) > 0).all()),
              "ascending order defines the model index")
    log.check("nodes", "ids fit int64", bool(ids.max() < 2**63 - 1), f"max id {int(ids.max())}")
    return ids


def load_signs(nt_path: Path, ids: np.ndarray, log: CheckLog) -> np.ndarray:
    """int8[N]: transmitter polarity per selected node, 0 where unusable."""
    from pyarrow import feather, ipc

    schema = ipc.open_file(nt_path).schema
    missing = [c for c in COL_NT if c not in schema.names]
    if not log.check("nt", "neurotransmitter columns present", not missing, f"missing {missing}"):
        raise SystemExit("neurotransmitter schema mismatch")

    table = feather.read_table(nt_path, columns=list(COL_NT))
    nt_body = table["body"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    nt_label = table["consensus_nt"].to_pylist()
    log.check("nt", "body unique", np.unique(nt_body).size == nt_body.size,
              f"{np.unique(nt_body).size} distinct of {nt_body.size}")

    order = np.argsort(nt_body, kind="stable")
    nt_body = nt_body[order]
    label_sorted = [nt_label[i] for i in order]
    sign_all = np.fromiter((nt_sign(v) for v in label_sorted), dtype=np.int8, count=len(label_sorted))

    pos = np.searchsorted(nt_body, ids)
    safe = np.minimum(pos, max(nt_body.size - 1, 0))
    present = (pos < nt_body.size) & (nt_body[safe] == ids)

    signs = np.zeros(ids.size, dtype=np.int8)
    signs[present] = sign_all[safe[present]]

    label_counts = Counter(label_sorted)
    log.note("nt", "label distribution",
             ", ".join(f"{k} {v}" for k, v in sorted(label_counts.items(), key=lambda kv: str(kv[0]))))
    log.note("nt", "nodes without an nt row", f"{int((~present).sum())} of {ids.size}")
    log.check(
        "nt",
        "signs are exactly -1/0/+1",
        set(np.unique(signs).tolist()) <= {-1, 0, 1},
        f"excitatory {int((signs > 0).sum())}, inhibitory {int((signs < 0).sum())}, "
        f"unsigned {int((signs == 0).sum())} of {ids.size}",
    )
    log.check("nt", "some node carries each polarity", bool((signs > 0).any() and (signs < 0).any()),
              "both signs present")
    return signs


# ---------------------------------------------------------------- edges
def load_edges(
    conn_path: Path, ids: np.ndarray, signs: np.ndarray, log: CheckLog
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Stream the 1 GB connectivity table, keep what the rules allow.

    Streamed batch by batch rather than read whole: the raw table is far larger
    than the surviving edge set, and materialising it costs several GB for rows
    that are then thrown away.
    """
    from pyarrow import ipc

    reader = ipc.open_file(conn_path)
    schema = reader.schema
    log.note("edges", "connectivity schema", json.dumps({f.name: str(f.type) for f in schema}))
    missing = [c for c in COL_CONN if c not in schema.names]
    if not log.check("edges", "connectivity columns present", not missing, f"missing {missing}"):
        raise SystemExit("connectivity schema mismatch")

    pre_parts: list[np.ndarray] = []
    post_parts: list[np.ndarray] = []
    cnt_parts: list[np.ndarray] = []

    n_rows = 0
    drop_endpoint = 0
    drop_unsigned = 0
    drop_zero_weight = 0
    weight_max = 0

    for b in range(reader.num_record_batches):
        batch = reader.get_batch(b)
        pre_id = batch.column(schema.get_field_index("body_pre")).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        post_id = batch.column(schema.get_field_index("body_post")).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        weight = batch.column(schema.get_field_index("weight")).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        n_rows += pre_id.size
        if weight.size:
            weight_max = max(weight_max, int(weight.max()))

        pi = np.searchsorted(ids, pre_id)
        qi = np.searchsorted(ids, post_id)
        pi_safe = np.minimum(pi, ids.size - 1)
        qi_safe = np.minimum(qi, ids.size - 1)
        pre_ok = (pi < ids.size) & (ids[pi_safe] == pre_id)
        post_ok = (qi < ids.size) & (ids[qi_safe] == post_id)

        endpoints = pre_ok & post_ok
        drop_endpoint += int((~endpoints).sum())

        s = np.where(endpoints, signs[pi_safe], 0).astype(np.int64)
        signed = endpoints & (s != 0)
        drop_unsigned += int((endpoints & (s == 0)).sum())

        nonzero = signed & (weight > 0)
        drop_zero_weight += int((signed & (weight <= 0)).sum())
        if not nonzero.any():
            continue

        pre_parts.append(pi_safe[nonzero].astype(np.int32))
        post_parts.append(qi_safe[nonzero].astype(np.int32))
        cnt_parts.append((s[nonzero] * weight[nonzero]).astype(np.int64))

    if not log.check("edges", "surviving edge set non-empty", bool(pre_parts),
                     f"{len(pre_parts)} non-empty batches survived the filter"):
        raise SystemExit("empty edge set: no row had both endpoints selected and a signed source")

    pre = np.concatenate(pre_parts)
    post = np.concatenate(post_parts)
    cnt = np.concatenate(cnt_parts)
    del pre_parts, post_parts, cnt_parts

    kept = pre.size
    log.note("edges", "raw row count", f"{n_rows} rows in {reader.num_record_batches} record batches")
    log.note(
        "edges",
        "filter accounting",
        f"kept {kept}; dropped {drop_endpoint} (endpoint not a selected node), "
        f"{drop_unsigned} (presynaptic transmitter has no sign), "
        f"{drop_zero_weight} (non-positive weight)",
    )
    log.check(
        "edges",
        "filter accounting is exhaustive",
        kept + drop_endpoint + drop_unsigned + drop_zero_weight == n_rows,
        f"{kept} + {drop_endpoint} + {drop_unsigned} + {drop_zero_weight} == {n_rows}",
    )
    log.note("edges", "raw weight range", f"max weight {weight_max}")

    stats = {
        "raw_rows": int(n_rows),
        "kept": int(kept),
        "dropped_endpoint_not_selected": int(drop_endpoint),
        "dropped_unsigned_presynaptic": int(drop_unsigned),
        "dropped_non_positive_weight": int(drop_zero_weight),
        "max_raw_weight": int(weight_max),
    }
    return pre, post, cnt, stats


MALECNS_SEMANTICS = {
    "index_mapping": "model index i == position of bodyId in the ascending array of "
                     "annotation rows whose superclass is non-null",
    "signed_counts": "signed synaptic contact count == nt_sign(presynaptic consensus_nt) * weight; "
                     "acetylcholine +1, GABA -1, glutamate -1, everything else drops the edge",
    "weight_application": "runtime multiplies accumulated int32 counts by w_syn = 0.275 mV; "
                          "never accumulate in float",
    "row_ptr": "int32[N+1], row_ptr[i]..row_ptr[i+1] are the outgoing edges of source neuron i",
    "destinations": "int32[E], ascending within each row",
    "caveat": "male specimen, brain plus ventral nerve cord; not comparable to the FlyWire "
              "female brain-only pack. The Shiu et al. constants are reused unchanged and "
              "have not been refitted to this dataset.",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    raw = root / "data/raw/male_cns"
    ap.add_argument("--annotations", type=Path, default=raw / SOURCE_FILES["annotations"])
    ap.add_argument("--neurotransmitters", type=Path, default=raw / SOURCE_FILES["neurotransmitters"])
    ap.add_argument("--connectivity", type=Path, default=raw / SOURCE_FILES["connectivity"])
    ap.add_argument("--out", type=Path, default=root / "data/pack/male_cns_v1")
    args = ap.parse_args()

    for path in (args.annotations, args.neurotransmitters, args.connectivity):
        if not path.is_file():
            print(f"missing source: {path}\ndownload page: {DOWNLOAD_PAGE}", file=sys.stderr)
            return 2

    log = CheckLog()
    print(f"annotations  : {args.annotations}")
    print(f"transmitters : {args.neurotransmitters}")
    print(f"connectivity : {args.connectivity}")
    print(f"out          : {args.out}")

    src_hashes = {
        key: {
            "path": path.name,
            "url": BASE_URL + SOURCE_FILES[key],
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for key, path in (
            ("annotations", args.annotations),
            ("neurotransmitters", args.neurotransmitters),
            ("connectivity", args.connectivity),
        )
    }

    ids = load_nodes(args.annotations, log)
    signs = load_signs(args.neurotransmitters, ids, log)
    pre_raw, post_raw, cnt_raw, filter_stats = load_edges(args.connectivity, ids, signs, log)

    # Shared with the FlyWire path from here down. check_edges re-derives the
    # bounds and integrality checks against N rather than trusting this module.
    pre, post, counts = check_edges(pre_raw, post_raw, cnt_raw, ids.size, log)
    row_ptr, dst, cnt, stats = build_csr(pre, post, counts, ids.size, log)
    stats["source_filter"] = filter_stats

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
        dataset="male-cns-v1.0-superclass-non-null-known-nt",
        semantics=MALECNS_SEMANTICS,
    )
    manifest["license"] = LICENSE
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(e["bytes"] for e in manifest["arrays"].values())
    print(f"\n>>> ALL CHECKS PASSED. pack written: {manifest['neurons']} neurons, "
          f"{manifest['edges']} edges, {total} bytes ({total/2**20:.1f} MiB)")
    print(f"    manifest: {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
