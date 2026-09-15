"""Degree-preserving shuffle of a pack: the control for whether the wiring matters.

Every source neuron keeps its CSR row, so its out-degree and its signed counts,
and with them its sign and its total output, stay the source pack's. The
destinations are permuted across all edges, so every neuron keeps its in-degree.
A pack merges parallel edges, so a destination that lands twice in one row is
swapped with a random edge's until no row repeats one. Which neuron reaches which
is random, and with it each neuron's incoming weight and its mix of excitatory
and inhibitory inputs.

The result is its own pack. Its manifest says it is not the connectome, records
the source pack's array hashes under derived_from and no sources, so run_exp
refuses a path_comp or path_con with it.

    python -m lif.shuffle_pack --pack data/pack/v630 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lif.compile_pack import CheckLog, sha256_array, write_pack


def shuffle_edges(
    row_ptr: np.ndarray,
    destinations: np.ndarray,
    signed_counts: np.ndarray,
    seed: int,
    max_rounds: int = 1000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """New destinations and signed counts for the same row_ptr, each row ascending
    by destination, and stats: repair_rounds, edges_kept, self_loops."""
    row_ptr = np.asarray(row_ptr, dtype=np.int64)
    destinations, signed_counts = np.asarray(destinations), np.asarray(signed_counts)
    e = destinations.size
    if (row_ptr.size == 0 or row_ptr[0] != 0 or row_ptr[-1] != e
            or signed_counts.size != e or (np.diff(row_ptr) < 0).any()):
        raise ValueError(
            f"row_ptr must rise from 0 to the number of edges, got {row_ptr.size} entries "
            f"ending at {int(row_ptr[-1]) if row_ptr.size else None} for {e} destinations "
            f"and {signed_counts.size} signed counts")
    n = row_ptr.size - 1
    if e and (destinations.min() < 0 or destinations.max() >= n):
        raise ValueError(f"destinations must be neuron indices below {n}, got "
                         f"{int(destinations.min())} to {int(destinations.max())}")
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(row_ptr))
    real = src * n + destinations
    if (np.diff(real) <= 0).any():
        raise ValueError("destinations must ascend strictly within each row, as in a pack")

    rng = np.random.default_rng(seed)
    dst = destinations.astype(np.int64)[rng.permutation(e)]
    for rounds in range(max_rounds + 1):
        flat = src * n + dst
        order = np.argsort(flat, kind="stable")
        key = flat[order]
        repeat = order[1:][key[1:] == key[:-1]]
        if repeat.size == 0:
            break
        if rounds == max_rounds:
            raise ValueError(f"{repeat.size} parallel edges left after {max_rounds} repair rounds")
        # Swap each repeat's destination with a random edge's. No edge takes part in
        # two swaps, so together they permute the destinations and in-degrees stay.
        partner = rng.integers(0, e, repeat.size)
        free = ~np.isin(partner, repeat)
        partner, first = np.unique(partner[free], return_index=True)
        repeat = repeat[free][first]
        swap = np.arange(e)
        swap[repeat], swap[partner] = partner, repeat
        dst = dst[swap]

    # src ascends, so the last round's order sorts each row by destination and
    # leaves every edge in its row, with its signed count.
    at = np.minimum(np.searchsorted(real, key), max(e - 1, 0))
    stats = {
        "repair_rounds": rounds,
        "edges_kept": int((real[at] == key).sum()) if e else 0,
        "self_loops": int((src == dst[order]).sum()),
    }
    return dst[order].astype(destinations.dtype), signed_counts[order], stats


def write_shuffled_pack(source: Path, out: Path, seed: int) -> dict:
    source, out = Path(source), Path(out)
    manifest = json.loads((source / "manifest.json").read_text())
    if "shuffle" in manifest:
        raise ValueError(f"{source} is already a shuffled pack")
    # write_pack deletes its output directory first, so that may only be a shuffled pack.
    existing = out / "manifest.json"
    if out.exists() and not (existing.is_file() and "shuffle" in json.loads(existing.read_text())):
        raise ValueError(f"{out} exists and is not a shuffled pack, refusing to replace it")

    arrays = {}
    for name, meta in manifest["arrays"].items():
        path = source / f"{name}.npy"
        arr = np.load(path, allow_pickle=False)
        if sha256_array(arr) != meta["sha256"]:
            raise ValueError(f"{path} does not match the sha256 in its manifest")
        arrays[name] = arr
    destinations, signed_counts, stats = shuffle_edges(
        arrays["row_ptr"], arrays["destinations"], arrays["signed_counts"], seed)
    arrays.update(destinations=destinations, signed_counts=signed_counts)

    written = write_pack(
        out, arrays, dict(manifest["stats"], self_loops=stats["self_loops"]), {}, CheckLog(),
        dataset=f"{manifest['dataset']}-degree-shuffled-seed{seed}", semantics=manifest["semantics"])
    written["shuffle"] = {
        "connectome": False,
        "note": "Not the connectome: a control whose wiring is random and whose degrees "
                "and signed counts are the source pack's.",
        "seed": seed,
        "method": "destinations permuted across all edges; a destination repeated within a row "
                  "swapped with a uniformly chosen edge's until no row repeats one; each row "
                  "then sorted by destination, every signed count staying in its row",
        "kept": ["every neuron's ID, out-degree and in-degree",
                 "every source neuron's signed counts, and so its sign and its total output"],
        "changed": ["which neuron each edge reaches",
                    "every neuron's incoming weight and its mix of excitatory and inhibitory inputs"],
        **stats,
    }
    written["derived_from"] = {
        "dataset": manifest["dataset"],
        "pack": source.name,
        "arrays": {name: meta["sha256"] for name, meta in manifest["arrays"].items()},
    }
    (out / "manifest.json").write_text(json.dumps(written, indent=2) + "\n")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--pack", type=Path, default=root / "data/pack/v630")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <pack>-shuffled-seed<seed>, beside the source pack")
    args = ap.parse_args()
    out = args.out or args.pack.with_name(f"{args.pack.name}-shuffled-seed{args.seed}")
    manifest = write_shuffled_pack(args.pack, out, args.seed)
    s = manifest["shuffle"]
    print(f"{out}: {manifest['dataset']}, {manifest['edges']} edges, {s['repair_rounds']} repair "
          f"rounds, {s['edges_kept']} edges the source pack has too, {s['self_loops']} self-loops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
