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
    args = ap.parse_args()
    ticks = 2000 if args.quick else args.ticks

    from lif import engine_chunked, engine_fused, engine_metal, engine_naive

    pack = core.load_pack()
    targets = stimulus_flybrain.targets(pack.neuron_ids)
    draws = stimulus_flybrain.bernoulli(len(targets), ticks, args.rate_hz,
                                        core.DT, args.seed)
    stim = core.Stimulus(targets=targets, draws=mx.array(draws), n_ticks=ticks,
                         rate_hz=args.rate_hz, seed=args.seed)

    print(f"{_chip()}  |  {pack.n_neurons} neurons, {pack.n_edges} edges")
    print(f"{ticks} ticks = {ticks * core.DT / 1000:.1f} biological s, "
          f"{len(targets)} sugar GRNs @ {args.rate_hz:g} Hz, "
          f"{int(draws.sum())} input spikes\n")

    lanes = [
        ("naive dense (eval/tick)", lambda: engine_naive.run(pack, stim, warmup=50)),
        ("chunked dense", lambda: engine_chunked.run(pack, stim, chunk=64, warmup=50)),
        ("sparse metal kernel", lambda: engine_metal.run(pack, stim, chunk=32, warmup=50)),
        ("fused, 2 dispatches", lambda: engine_fused.run(pack, stim, chunk=32,
                                                         warmup=50, edge_split=1)),
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
