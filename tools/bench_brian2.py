"""Time Brian2 on the whole FlyWire v630 brain: the figure this repository is
compared against.

The README's speed claim is against the published Brian2 implementation, and that
number has to come from somewhere reproducible. This runs upstream's own
`model.py` -- the same `create_model` and `poi` its `run_exp` calls -- on the same
127,400 neurons and 14,687,178 connections, driven by the same 21 right sugar GRNs
at 150 Hz that `lif.benchmark` uses, and times `net.run` alone.

Two runs of the same duration are timed. The first pays Brian2's code generation,
the second does not, the way `lif.benchmark` excludes pack loading and shader
compilation from its lanes. The second is the comparable number; both are printed,
because hiding the first would flatter Brian2's competitor, not Brian2.

Spikes are reported as well. Brian2 slows down as activity rises, so a figure
measured on a barely active network is a lower bound on the cost of a busy one,
and a number quoted without its activity level means little.

A whole-experiment total is deliberately not measured here. Upstream's `run_trial`
rebuilds the network inside every trial and joblib gives each trial its own
process, so its memory starts clean each time. Looping in one process instead
accumulates 14.7 M-synapse networks: tried on 2026-09-15 it reached 5.9 GB
resident and was still climbing after six minutes, with the timings degrading
along with it, and it would not have measured upstream's cost anyway. The
30-trial figure in the README is therefore 30 x (build + run) from the numbers
this prints, and is labelled an extrapolation.

    python tools/bench_brian2.py --ticks 200
    python tools/bench_brian2.py --ticks 2000

Needs the reference extra: pip install -e '.[reference]', plus upstream's files
from ./tools/fetch_upstream.sh.
"""

from __future__ import annotations

import argparse
import importlib.util
import resource
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def peak_mb() -> float:
    """Peak resident size of this process in MB; ru_maxrss is bytes on macOS."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def resolved_target() -> str:
    """The codegen target Brian2 will actually use, for the provenance line.

    prefs.codegen.target keeps whatever it was set to: Brian2 resolves 'auto'
    inside auto_target() at code-object build time and never writes the answer
    back. Printing the pref would therefore record the literal string 'auto' for
    every default run, and cython and numpy differ by about 4x on this very
    benchmark -- a figure whose target is unknown is not reproducible.
    """
    from brian2 import prefs
    from brian2.devices.device import auto_target

    target = prefs.codegen.target
    return target if target != "auto" else f"auto -> {auto_target().class_name}"


def stub_joblib() -> bool:
    """Stand in for joblib unless a real one is there, and say whether it did.

    Upstream's model.py imports joblib at module level for the parallel driver
    inside its own run_exp. Nothing used here -- create_model, poi,
    default_params -- touches it, so a stand-in keeps upstream's file unmodified
    and spares a timing run a dependency it never calls. An installed joblib wins:
    testing sys.modules alone would shadow one that simply had not been imported
    yet.
    """
    if "joblib" in sys.modules:
        return False
    if importlib.util.find_spec("joblib") is not None:
        return False
    stub = types.ModuleType("joblib")
    stub.Parallel = stub.delayed = stub.parallel_backend = None
    sys.modules["joblib"] = stub
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", type=int, default=200,
                    help="ticks of 0.1 ms to run, twice (default: 200, as the published figure)")
    ap.add_argument("--rate", type=float, default=150.0, help="Poisson input on each GRN in Hz")
    ap.add_argument("--target", choices=("auto", "numpy", "cython"), default="auto",
                    help="Brian2's code generation target. 'auto' lets Brian2 choose, which is "
                         "cython where a compiler is available and numpy otherwise. The two "
                         "differ by several times, so a quoted figure without its target is "
                         "not reproducible")
    ap.add_argument("--completeness", type=Path, default=ROOT / "data/raw/completeness_630.csv")
    ap.add_argument("--connectivity", type=Path, default=ROOT / "data/raw/connectivity_630.parquet")
    ap.add_argument("--ref", type=Path, default=ROOT / "data/ref",
                    help="directory holding upstream's model.py")
    args = ap.parse_args()
    if args.ticks < 1:
        raise SystemExit("--ticks must be 1 or more")
    for path in (args.completeness, args.connectivity, args.ref / "model.py"):
        if not path.exists():
            raise SystemExit(f"missing {path}; run ./tools/fetch_upstream.sh")

    sys.path.insert(0, str(args.ref))
    import brian2
    import pandas as pd
    from brian2 import Hz, Network, defaultclock, ms, prefs

    # Must be set before any code object is built, so before create_model.
    if args.target != "auto":
        prefs.codegen.target = args.target

    stub_joblib()

    import model  # upstream's, unchanged

    defaultclock.dt = 0.1 * ms   # the engine's tick, and upstream's default
    duration = args.ticks * defaultclock.dt
    biological_s = float(duration / (1000 * ms))

    params = dict(model.default_params)
    params["t_run"] = duration
    params["r_poi"] = args.rate * Hz

    from lif import stimulus_flybrain

    print(f"Brian2 {brian2.__version__}, codegen target {resolved_target()}, "
          f"dt {defaultclock.dt!s}")
    print(f"{args.ticks} ticks = {biological_s:g} biological s, "
          f"21 sugar GRNs at {args.rate:g} Hz, timed twice")

    t0 = time.perf_counter()
    neu, syn, spk_mon = model.create_model(args.completeness, args.connectivity, params)
    df_comp = pd.read_csv(args.completeness, index_col=0)
    flyid2i = {flyid: i for i, flyid in enumerate(df_comp.index)}
    missing = [i for i in stimulus_flybrain.RIGHT_SUGAR_GRN_IDS if i not in flyid2i]
    if missing:
        raise SystemExit(f"sugar GRNs not in the completeness table: {missing}")
    exc = [flyid2i[i] for i in stimulus_flybrain.RIGHT_SUGAR_GRN_IDS]
    pois, neu = model.poi(neu, exc, [], params)
    net = Network(neu, syn, spk_mon, *pois)
    build_s = time.perf_counter() - t0
    print(f"\nnetwork built in {build_s:.1f} s: {len(neu)} neurons, {len(syn)} synapses, "
          f"{len(exc)} driven")

    print(f"{'run':<6}{'wall s':>10}{'s / biological s':>20}{'spikes':>12}{'peak MB':>10}")
    before = 0
    for n in (1, 2):
        start = time.perf_counter()
        net.run(duration)
        wall = time.perf_counter() - start
        spikes = int(spk_mon.num_spikes) - before
        before += spikes
        print(f"{n:<6}{wall:>10.2f}{wall / biological_s:>20.1f}{spikes:>12}{peak_mb():>10.0f}")

    print("\nRun 2 is the comparable figure: run 1 includes Brian2's code generation, "
          "as\nlif.benchmark excludes shader compilation from its lanes. Brian2 slows as\n"
          "activity rises, so this is a lower bound for a busier network.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
