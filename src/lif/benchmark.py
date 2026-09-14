"""Reproduce the numbers in README.md.

    python -m lif.benchmark              # all MLX lanes, sugar-GRN stimulus
    python -m lif.benchmark --repeat 5   # with variance
    python -m lif.benchmark --quick      # 2000 ticks instead of 10000

Every lane is checked against the naive baseline before its timing is reported;
a lane that fails parity is reported as FAIL and its time is meaningless.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from lif import core, stimulus_flybrain


def _chip() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return platform.machine()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", type=int, default=10_000)
    ap.add_argument("--quick", action="store_true", help="2000 ticks")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--rate-hz", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--out", type=Path, default=Path("bench/results.json"))
    ap.add_argument("--pack", type=Path, default=core.PACK_DIR,
                    help="pack directory; defaults to the FlyWire v630 pack")
    ap.add_argument("--edge-split", type=int, default=None,
                    help="threads per source neuron for both kernel lanes; "
                         "default 2 (metal) and 1 (fused) for the sugar drive, 8 for the "
                         "hub drive")
    ap.add_argument("--stimulus", choices=("auto", "sugar", "hubs"), default="auto",
                    help="auto uses the sugar GRNs on a FlyWire pack, top-out-degree "
                         "hubs on any other dataset")
    args = ap.parse_args()
    ticks = 2000 if args.quick else args.ticks

    from lif import engine_chunked, engine_fused, engine_metal, engine_naive

    pack = core.load_pack(args.pack)

    # The 21 sugar GRNs are FlyWire root IDs. They do not exist in any other
    # specimen, so a non-FlyWire pack falls back to the deterministic hub drive.
    mode = args.stimulus
    if mode == "auto":
        mode = "sugar" if pack.manifest.get("dataset", "").startswith("flywire") else "hubs"

    if mode == "sugar":
        targets = stimulus_flybrain.targets(pack.neuron_ids)
        draws = stimulus_flybrain.bernoulli(len(targets), ticks, args.rate_hz,
                                            core.DT, args.seed)
        stim = core.Stimulus(targets=targets, draws=mx.array(draws), n_ticks=ticks,
                             rate_hz=args.rate_hz, seed=args.seed)
        drive = f"{len(targets)} sugar GRNs"
    else:
        stim = core.make_stimulus(pack, ticks, args.seed, n_targets=100,
                                  rate_hz=args.rate_hz)
        targets = stim.targets
        draws = np.asarray(stim.draws)
        drive = f"{len(targets)} top-out-degree hubs"

    # edge_split has no runtime-choosable value (see engine_metal.py), so each
    # kernel lane runs at the value measured best for the drive: 1 for the fused
    # lane and 2 for the sparse one on the sugar drive, 8 for both on the hub
    # drive (table in README.md, "Use"). What makes a comparison between the lanes
    # meaningless is a value wrong for the drive, not a different value per lane:
    # with the fused lane hardcoded to 1 and the sparse one at the old module
    # default of 16, the fused lane came out slower on a hub stimulus than the
    # lane it replaces. One shared value would misreport a lane on the sugar
    # drive, where the fused lane at 2 takes 1.16x its time at 1. --edge-split
    # sets both anyway, for comparing the lanes at one setting rather than each at
    # its best.
    if mode == "sugar":
        split_metal, split_fused = engine_metal.EDGE_SPLIT_SPARSE, 1
    else:
        split_metal = split_fused = engine_metal.EDGE_SPLIT
    if args.edge_split is not None:
        split_metal = split_fused = args.edge_split

    print(f"{_chip()}  |  {pack.n_neurons} neurons, {pack.n_edges} edges")
    print(f"dataset {pack.manifest.get('dataset', 'unknown')}")
    print(f"{ticks} ticks = {ticks * core.DT / 1000:.1f} biological s, "
          f"{drive} @ {args.rate_hz:g} Hz, "
          f"{int(draws.sum())} input spikes, "
          f"edge_split {split_metal} (metal) / {split_fused} (fused)\n")

    lanes = [
        ("naive dense (eval/tick)", lambda: engine_naive.run(pack, stim, warmup=50)),
        ("chunked dense", lambda: engine_chunked.run(pack, stim, chunk=64, warmup=50)),
        ("sparse metal kernel", lambda: engine_metal.run(pack, stim, chunk=32, warmup=50,
                                                          split=split_metal)),
        ("fused, 2 dispatches", lambda: engine_fused.run(pack, stim, chunk=32,
                                                         warmup=50, edge_split=split_fused)),
    ]

    print(f"{'lane':<26}{'s/biol.s':>10}{'ms/tick':>10}{'peak MB':>9}  parity")
    ref = None
    results = {}
    for name, fn in lanes:
        times = []
        for _ in range(args.repeat):
            r = fn()
            times.append(r.seconds)
        if ref is None:
            ref = r
            parity = "baseline"
        else:
            ok = (r.counts_sha256() == ref.counts_sha256()
                  and np.array_equal(r.v_final, ref.v_final)
                  and np.array_equal(r.g_final, ref.g_final))
            parity = "PASS" if ok else "FAIL"
        best = min(times)
        sec_per_bio = best / ticks * 10_000
        spread = f" ±{(max(times) - min(times)) / 2 / ticks * 10_000:.4f}" if args.repeat > 1 else ""
        print(f"{name:<26}{sec_per_bio:>10.4f}{best / ticks * 1000:>10.4f}"
              f"{r.peak_bytes / 1e6:>9.0f}  {parity}{spread}")
        results[name] = {
            "seconds_per_biological_second": sec_per_bio,
            "ms_per_tick": best / ticks * 1000,
            "peak_bytes": r.peak_bytes,
            "parity": parity,
            "runs": times,
        }

    print(f"\nspikes {ref.total_spikes()}, {int((ref.spike_counts > 0).sum())} neurons fired")
    print(f"spike-count sha256 {ref.counts_sha256()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "device": _chip(),
        "neurons": pack.n_neurons,
        "edges": pack.n_edges,
        "ticks": ticks,
        "stimulus": "right_sugar_grns",
        "rate_hz": args.rate_hz,
        "seed": args.seed,
        "total_spikes": int(ref.total_spikes()),
        "spike_counts_sha256": ref.counts_sha256(),
        "lanes": results,
    }, indent=2) + "\n")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
