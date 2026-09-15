"""How activity develops over a trial on the MaleCNS pack, by region, stimulus by stimulus.

The activity film's run, the right labellum's LB3b and LB3c neurons at 100 Hz,
fires more and more of its spikes in the ventral nerve cord (VNC) as the trial
goes on. This runs that stimulus and seven others, of other senses and of the
VNC's own sensory neurons, and reports for every 100 ms of the trial the spikes
per trial of the brain, the VNC, and ascending and descending neurons, the VNC's
share of them, and the neurons above 100 Hz in the last 100 ms with how many of
them sit in abdominal neuromeres. Three more runs of the film's stimulus look for
where the rise lives: with its input switched off after 200 ms and after 500 ms,
and with the outgoing synapses of the VNC's abdominal neurons silenced.

Regions follow the annotation table's superclass, and abdominal means a
somaNeuromere of A1 to A10. Trial n draws its input with seed + n, as run_exp
does, so the film's stimulus gives the spikes of the film's run.

    python -m lif.stimulus_survey
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
from pyarrow import feather

from lif import core, engine_fused, experiment
from lif import names as lif_names
from lif.activity_film import DRIVE
from lif.compile_pack import sha256_file
from lif.compile_pack_malecns import SOURCE_FILES

REGIONS = ("brain", "VNC", "ascending", "descending", "other")


def region_of(superclass: str | None) -> int:
    """The index into REGIONS of an annotation table superclass."""
    if superclass is None:
        return REGIONS.index("other")
    if superclass.startswith("vnc_"):
        return REGIONS.index("VNC")
    if superclass.startswith(("cb_", "ol_", "visual_")):
        return REGIONS.index("brain")
    if "ascending" in superclass:
        return REGIONS.index("ascending")
    if "descending" in superclass:
        return REGIONS.index("descending")
    return REGIONS.index("other")


def is_abdominal(neuromere: str | None) -> bool:
    """Whether a somaNeuromere is one of the abdominal ones, A1 to A10."""
    return neuromere is not None and neuromere[:1] == "A" and neuromere[1:].isdigit()


def switch_off(stim: core.Stimulus, after_ticks: int) -> core.Stimulus:
    """The same stimulus without input from tick after_ticks on."""
    if not 0 <= after_ticks <= stim.n_ticks:
        raise ValueError(f"after_ticks must be from 0 to {stim.n_ticks}, got {after_ticks}")
    if after_ticks == stim.n_ticks:
        return stim
    draws = np.array(stim.draws)
    draws[after_ticks:] = False
    return core.Stimulus(targets=stim.targets, draws=mx.array(draws), n_ticks=stim.n_ticks,
                         rate_hz=float("nan"), seed=stim.seed)


def drive_targets(select: Callable[[str], Sequence[int]], drive: Sequence[str]) -> list[int]:
    """Model indices of the neurons a drive names, in run_exp's order: name by name,
    each name's neurons as select gives them. make_stimulus_for draws one column per
    target in the order given, so this order is what gives a trial run_exp's input."""
    missing = [name for name in drive if not select(name)]
    if missing:
        raise ValueError(f"no neurons named {missing}")
    targets = [int(i) for name in drive for i in select(name)]
    values, times = np.unique(targets, return_counts=True)
    if (times > 1).any():
        raise ValueError(f"the drive names neurons more than once: {values[times > 1].tolist()}")
    return targets


class Course:
    """Spikes per region per window over a run's trials, and every neuron's spikes
    in the last window."""

    def __init__(self, region: np.ndarray, n_ticks: int, window_ticks: int):
        if window_ticks < 1 or n_ticks % window_ticks:
            raise ValueError(f"windows of {window_ticks} ticks do not tile a run of {n_ticks}")
        self.region = np.asarray(region, dtype=np.int64)
        self.n_ticks, self.window_ticks = n_ticks, window_ticks
        self.n_windows = n_ticks // window_ticks
        self.counts = np.zeros((len(REGIONS), self.n_windows), dtype=np.int64)
        self.last = np.zeros(self.region.size, dtype=np.int64)
        self.last_by_trial: list[np.ndarray] = []   # int64[R] per trial
        self.trials = 0

    def add(self, events: np.ndarray) -> None:
        """One trial's spike events, int[E, 2] of (tick, neuron)."""
        tick = events[:, 0].astype(np.int64)
        neuron = events[:, 1].astype(np.int64)
        if tick.size and (tick.min() < 0 or tick.max() >= self.n_ticks):
            raise ValueError(f"spike events at ticks outside 0 to {self.n_ticks - 1}")
        window = tick // self.window_ticks
        self.counts += np.bincount(self.region[neuron] * self.n_windows + window,
                                   minlength=self.counts.size).reshape(self.counts.shape)
        last = neuron[window == self.n_windows - 1]
        self.last += np.bincount(last, minlength=self.region.size)
        self.last_by_trial.append(np.bincount(self.region[last], minlength=len(REGIONS)))
        self.trials += 1

    def spikes_per_trial(self) -> np.ndarray:
        """float64[W]: spikes of all regions per trial in each window."""
        return self.counts.sum(axis=0) / self.trials

    def share(self, region: int) -> np.ndarray:
        """float64[W]: one region's fraction of the spikes in each window."""
        total = self.counts.sum(axis=0)
        return np.divide(self.counts[region], total, out=np.zeros(self.n_windows), where=total > 0)

    def last_window_hz(self) -> np.ndarray:
        """float64[N]: every neuron's rate in the last window."""
        return self.last / (self.trials * self.window_ticks * core.DT / 1000)

    def last_window_per_trial(self, region: int) -> np.ndarray:
        """int64[T]: one region's spikes in the last window, trial by trial, which a
        mean over the trials would hide when trials settle differently."""
        return np.array([counts[region] for counts in self.last_by_trial], dtype=np.int64)


@dataclass(frozen=True)
class Stimulus:
    label: str
    drive: tuple[str, ...]   # instances or types in the pack's names sidecar


STIMULI = (
    Stimulus("taste, sweet: the film's", DRIVE),
    Stimulus("taste, water", ("LB3a_R",)),
    Stimulus("smell, glomerulus DM1", ("ORN_DM1_R",)),
    Stimulus("Johnston's organ", ("JO-CM_R",)),
    Stimulus("vision, LC4", ("LC4_R",)),
    Stimulus("giant fibers", ("DNp01",)),
    Stimulus("taste, VNC", ("LgLG1a_R",)),
    Stimulus("proprioception, VNC", ("SNpp50",)),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--pack", type=Path, default=root / "data/pack/male_cns_v1")
    ap.add_argument("--annotations", type=Path,
                    default=root / "data/raw/male_cns" / SOURCE_FILES["annotations"])
    ap.add_argument("--rate", type=float, default=100.0, help="Poisson input on each in Hz")
    ap.add_argument("--trials", type=int, default=experiment.default_params["n_run"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window-ms", type=float, default=100.0)
    args = ap.parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be 1 or more")

    pack = core.load_pack(args.pack)
    dataset = pack.manifest["dataset"]
    recorded = pack.manifest.get("sources", {}).get("annotations")
    if recorded is None or sha256_file(args.annotations) != recorded["sha256"]:
        raise SystemExit(f"{args.annotations} is not the annotation table the {dataset} pack "
                         "was compiled from")
    names = lif_names.load(pack)
    if names is None:
        raise SystemExit(f"the {dataset} pack has no names sidecar; "
                         "python -m lif.compile_pack_malecns --names-only")
    table = feather.read_table(args.annotations, columns=["bodyId", "superclass", "somaNeuromere"])
    rows = pa.array(lif_names.annotation_rows(table, pack.neuron_ids))
    region = np.array([region_of(s) for s in table["superclass"].take(rows).to_pylist()])
    neuromere = table["somaNeuromere"].take(rows).to_pylist()
    abdominal = np.array([is_abdominal(n) for n in neuromere])
    n_ticks = round(experiment.default_params["t_run"] * 1000 / core.DT)
    window_ticks = round(args.window_ms / core.DT)
    vnc = REGIONS.index("VNC")
    print(f"{dataset}: {int((region == vnc).sum())} VNC neurons, "
          f"{int((abdominal & (region == vnc)).sum())} of them abdominal; {args.trials} trials of "
          f"{n_ticks * core.DT / 1000:g} s per run, seed {args.seed}, fused lane")

    summary = []

    def run(label, drive, off_after_ms=None, silenced=None):
        targets = np.array(drive_targets(names.select, drive), dtype=np.int32)
        course = Course(region, n_ticks, window_ticks)
        for n in range(args.trials):
            stim = core.make_stimulus_for(pack, targets, args.rate, n_ticks, args.seed + n)
            if off_after_ms is not None:
                stim = switch_off(stim, round(off_after_ms / core.DT))
            course.add(engine_fused.run(pack, stim, silenced=silenced, warmup=0, edge_split=8,
                                        record=True).events)

        fast = course.last_window_hz() > 100
        fast_abdominal = Counter(neuromere[i] for i in np.flatnonzero(fast & abdominal))
        spikes, share = course.spikes_per_trial(), course.share(vnc)
        print(f"\n{label}: {', '.join(drive)}, {targets.size} neurons at {args.rate:g} Hz")
        print(f"  {'from ms':<14}" + "".join(f"{k * args.window_ms:>8g}" for k in range(course.n_windows)))
        print(f"  {'spikes/trial':<14}" + "".join(f"{x:>8.0f}" for x in spikes))
        for r, name in enumerate(REGIONS):
            print(f"    {name:<12}" + "".join(f"{x / course.trials:>8.0f}" for x in course.counts[r]))
        print(f"  {'VNC share %':<14}" + "".join(f"{100 * x:>8.1f}" for x in share))
        print(f"  last {args.window_ms:g} ms: {int(fast.sum())} neurons above 100 Hz, "
              f"{int((fast & abdominal).sum())} of them abdominal {dict(fast_abdominal.most_common())}")
        print(f"  VNC spikes in the last {args.window_ms:g} ms, trial by trial, ascending: "
              f"{sorted(course.last_window_per_trial(vnc).tolist())}")
        summary.append((label, spikes[0], spikes[-1], share[0], share[-1], int(fast.sum()),
                        int((fast & abdominal).sum())))

    for stimulus in STIMULI:
        run(stimulus.label, stimulus.drive)
    film = STIMULI[0]
    for ms in (200.0, 500.0):
        run(f"{film.label}, input off after {ms:g} ms", film.drive, off_after_ms=ms)
    silenced = (region == vnc) & abdominal
    run(f"{film.label}, {int(silenced.sum())} abdominal VNC neurons silenced", film.drive,
        silenced=silenced)

    w = args.window_ms
    width = max(len(row[0]) for row in summary) + 2
    print(f"\n{'':<{width}}{'spikes per trial':^28}{'VNC share %':^16}"
          f"{f'above 100 Hz, last {w:g} ms':^28}")
    print(f"{'run':<{width}}{f'first {w:g} ms':>14}{f'last {w:g} ms':>14}{'first':>8}{'last':>8}"
          f"{'all':>14}{'abdominal':>14}")
    for label, s0, s1, v0, v1, n_fast, n_abdominal in summary:
        print(f"{label:<{width}}{s0:>14.0f}{s1:>14.0f}{100 * v0:>8.1f}{100 * v1:>8.1f}"
              f"{n_fast:>14}{n_abdominal:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
