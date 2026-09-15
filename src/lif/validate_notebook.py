"""Upstream's example notebook on this engine, against the files upstream published.

philshiu/Drosophila_brain_model ships example.ipynb and, in results/example, the
parquet files its Brian2 run_exp wrote for five of that notebook's experiments.
This runs the notebook's code cells as they are, in a directory where `model` is
this engine's run_exp and default_params and `utils` is upstream's utils.py, and
checks that every file the notebook wrote came from this engine. Then it runs the
five published experiments again, with the rate their files were written with,
once with each of two seeds, and compares every neuron's rate over the trials
with the published file's, and the two seeds' runs with each other. Those two
differ only in their random input, so they show how far apart two runs of the
same model fall.

It exits non-zero when a comparison fails the gate in `problems`, which catches a
gross error, such as a wrong rate or a wrong silenced neuron, and is not a test
of significance. The notebook imports Brian2 and upstream's utils.py imports
pandas: pip install -e '.[reference]'.

    ./tools/fetch_upstream.sh
    python -m lif.validate_notebook
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lif import experiment

# Upstream's names for its two model files, and the names tools/fetch_upstream.sh
# gives them in data/raw.
DATA_FILES = {
    "2023_03_23_completeness_630_final.csv": "completeness_630.csv",
    "2023_03_23_connectivity_630_final.parquet": "connectivity_630.parquet",
}

# The experiments upstream published: (r_poi in Hz, silenced IDs), each with the
# notebook's sugar GRNs as neu_exc. Upstream's sugarR file was written at 200 Hz,
# the default the notebook's text gives, while model.py's default_params now say
# 150 Hz, which is what the notebook run here uses; the command prints the sugar
# GRNs' rates, which show the difference.
PUBLISHED = {
    "sugarR": (200.0, ()),
    "sugarR_100Hz": (100.0, ()),
    "sugarR-720575940617937543": (100.0, (720575940617937543,)),
    "sugarR-720575940621754367": (100.0, (720575940621754367,)),
    "sugarR-720575940622695448": (100.0, (720575940622695448,)),
}

STAND_IN = '''"""Stands in for upstream's model.py, so that the notebook runs this engine."""
from lif.experiment import default_params, run_exp  # noqa: F401
'''

# The gate: spikes per trial or the readout neuron's rate with a |z| not below
# Z_LIMIT fails, and so do more than OUTLIER_SHARE of the neurons above it.
Z_LIMIT = 4.0
OUTLIER_SHARE = 0.01

# Marks a working directory this command made, which it may therefore replace.
MARKER = ".validate_notebook"


def code_cells(notebook: Mapping) -> tuple[list[tuple[int, str]], list[int]]:
    """(index, source) of every code cell plain Python can run, and the indices of
    the code cells left out: those with a line starting with ! or %, IPython's shell
    commands and magics, such as upstream's first cell, which sets up Colab."""
    cells, skipped = [], []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            skipped.append(i)
        else:
            cells.append((i, source))
    return cells, skipped


def prepare(workdir: Path, ref: Path, raw: Path) -> None:
    """Lay workdir out as the notebook expects its directory: a model.py standing in
    for upstream's, upstream's utils.py, and the two model files under upstream's
    names, the last two as links."""
    links = {"utils.py": ref / "utils.py",
             **{upstream: raw / name for upstream, name in DATA_FILES.items()}}
    missing = [str(target) for target in links.values() if not target.is_file()]
    if missing:
        raise FileNotFoundError(f"no {missing}; run tools/fetch_upstream.sh")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "model.py").write_text(STAND_IN)
    for name, target in links.items():
        (workdir / name).symlink_to(target.resolve())


def run_cells(cells: Sequence[tuple[int, str]], workdir: Path) -> dict:
    """Run the cells in order in one namespace, as a notebook's kernel does, with
    workdir as the working directory and first on sys.path; returns the namespace.

    Modules named like workdir's .py files that were imported before are set aside
    for the run and put back after it, as are the working directory and sys.path.
    An exception from a cell carries a note naming the cell.
    """
    workdir = Path(workdir).resolve()
    own = {p.stem for p in workdir.glob("*.py")}
    set_aside = {name: sys.modules.pop(name) for name in own if name in sys.modules}
    cwd, path = Path.cwd(), list(sys.path)
    namespace = {"__name__": "__main__"}
    os.chdir(workdir)
    sys.path.insert(0, str(workdir))
    importlib.invalidate_caches()
    try:
        for index, source in cells:
            try:
                # Running the notebook's code is what this function is for.
                exec(compile(source, f"<code cell {index}>", "exec"), namespace)  # noqa: S102
            except Exception as e:
                e.add_note(f"in code cell {index} of the notebook")
                raise
    finally:
        os.chdir(cwd)
        sys.path[:] = path
        for name in own:
            sys.modules.pop(name, None)
        sys.modules.update(set_aside)
    return namespace


def trial_counts(path: str | Path, n_run: int) -> tuple[np.ndarray, np.ndarray]:
    """The IDs of the neurons that fired, ascending, and each one's number of spikes
    in each trial, int64[K, n_run]."""
    table = pq.read_table(path, columns=["trial", "flywire_id"])
    trial = table["trial"].to_numpy()
    if trial.size and (trial.min() < 0 or trial.max() >= n_run):
        raise ValueError(f"{path} has trials outside 0 to {n_run - 1}")
    ids, neuron = np.unique(table["flywire_id"].to_numpy(), return_inverse=True)
    counts = np.bincount(neuron * n_run + trial, minlength=ids.size * n_run)
    return ids, counts.reshape(ids.size, n_run)


@dataclass(frozen=True, eq=False)
class Comparison:
    """Two experiment files, over every neuron that fired in either."""

    ids: np.ndarray      # int64[K], ascending
    rate_a: np.ndarray   # float64[K], Hz, mean over the trials
    rate_b: np.ndarray
    z: np.ndarray        # float64[K], rate_b - rate_a over its standard error
    spikes_a: float      # spikes of all neurons per trial, mean over the trials
    spikes_b: float
    z_spikes: float

    def of(self, neuron: int) -> tuple[float, float, float]:
        """rate_a, rate_b and z of one neuron, zeros if it fired in neither file."""
        k = int(np.searchsorted(self.ids, neuron))
        if k == self.ids.size or self.ids[k] != neuron:
            return 0.0, 0.0, 0.0
        return float(self.rate_a[k]), float(self.rate_b[k]), float(self.z[k])


def _z(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The difference of the means over the last axis in standard errors: 0 where
    both are constant and equal, infinite where both are constant and differ."""
    diff = b.mean(axis=-1) - a.mean(axis=-1)
    se = np.sqrt((a.var(axis=-1, ddof=1) + b.var(axis=-1, ddof=1)) / a.shape[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, diff / se, np.where(diff == 0, 0.0, np.copysign(np.inf, diff)))


def compare(a: str | Path, b: str | Path, n_run: int, t_run: float) -> Comparison:
    """Rates, z per neuron and spikes per trial of two files of n_run trials of t_run
    seconds; a neuron that fired in one file only has 0 spikes in the other."""
    if n_run < 2:
        raise ValueError(f"a z over trials needs 2 or more trials, got {n_run}")
    ids_a, counts_a = trial_counts(a, n_run)
    ids_b, counts_b = trial_counts(b, n_run)
    ids = np.union1d(ids_a, ids_b)
    ca = np.zeros((ids.size, n_run))
    cb = np.zeros((ids.size, n_run))
    ca[np.searchsorted(ids, ids_a)] = counts_a
    cb[np.searchsorted(ids, ids_b)] = counts_b
    return Comparison(
        ids=ids, rate_a=ca.mean(axis=1) / t_run, rate_b=cb.mean(axis=1) / t_run, z=_z(ca, cb),
        spikes_a=float(ca.sum(axis=0).mean()), spikes_b=float(cb.sum(axis=0).mean()),
        z_spikes=float(_z(ca.sum(axis=0), cb.sum(axis=0))))


def problems(c: Comparison, neuron: int) -> list[str]:
    """Why the comparison fails the gate, empty if it passes: spikes per trial or
    the rate of `neuron` with a |z| not below Z_LIMIT, or more than OUTLIER_SHARE of
    the neurons with a |z| above it."""
    out = []
    if not abs(c.z_spikes) < Z_LIMIT:
        out.append(f"spikes per trial {c.spikes_a:.1f} and {c.spikes_b:.1f}, z {c.z_spikes:+.2f}")
    a, b, z = c.of(neuron)
    if not abs(z) < Z_LIMIT:
        out.append(f"neuron {neuron} at {a:.2f} and {b:.2f} Hz, z {z:+.2f}")
    beyond = int((np.abs(c.z) > Z_LIMIT).sum())
    if beyond > OUTLIER_SHARE * c.z.size:
        out.append(f"{beyond} of {c.z.size} neurons differ by |z| > {Z_LIMIT:g}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--ref", type=Path, default=root / "data/ref",
                    help="upstream's example.ipynb, utils.py and results/example")
    ap.add_argument("--raw", type=Path, default=root / "data/raw", help="upstream's model files")
    ap.add_argument("--out", type=Path, default=root / "outputs/validate_notebook")
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 1000], metavar="SEED",
                    help="this engine's two seeds; trial n draws its input with seed + n")
    args = ap.parse_args()

    published = args.ref / "results" / "example"
    needed = [args.ref / "example.ipynb", args.ref / "utils.py",
              *(published / f"{stem}.parquet" for stem in PUBLISHED),
              *(args.raw / name for name in DATA_FILES.values())]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise SystemExit(f"missing {missing}; run tools/fetch_upstream.sh")
    for module in ("brian2", "pandas"):
        if importlib.util.find_spec(module) is None:
            raise SystemExit(f"the notebook needs {module}: pip install -e '.[reference]'")
    base = dict(experiment.default_params)
    n_run, t_run = base["n_run"], base["t_run"]
    if abs(args.seeds[1] - args.seeds[0]) < n_run:
        raise SystemExit(f"seeds {args.seeds} would share trial inputs; "
                         f"they must be {n_run} or more apart")

    workdir = args.out / "notebook"
    if workdir.exists():
        if not (workdir / MARKER).is_file():
            raise SystemExit(f"{workdir} exists, and this command did not make it")
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    (workdir / MARKER).write_text("")
    prepare(workdir, args.ref, args.raw)
    cells, skipped = code_cells(json.loads((args.ref / "example.ipynb").read_text()))
    try:
        ns = run_cells(cells, workdir)
    finally:
        experiment.default_params.update(base)   # the notebook changes r_poi in place
    written = sorted((workdir / "results").rglob("*.parquet"))
    if not written:
        raise SystemExit("the notebook wrote no experiment file")
    lanes = {experiment.metadata(p)["engine"] for p in written}   # raises for a file not ours
    try:
        sugar, mn9 = [int(i) for i in ns["neu_sugar"]], int(ns["id_mn9"])
    except KeyError as e:
        raise SystemExit(f"example.ipynb no longer defines {e}") from e
    print(f"\nexample.ipynb: ran code cells {[i for i, _ in cells]} as they are and left out "
          f"{skipped}, which run shell commands; its {len(written)} files were written by "
          f"this engine's {', '.join(sorted(lanes))} lane")

    path_comp, path_con = (args.raw / name for name in DATA_FILES.values())
    s0, s1 = args.seeds
    failed = []
    for stem, (rate, silenced) in PUBLISHED.items():
        brian2 = published / f"{stem}.parquet"
        exp_names = set(pq.read_table(brian2, columns=["exp_name"])["exp_name"].to_pylist())
        if exp_names != {stem}:
            raise SystemExit(f"{brian2} holds the experiments {sorted(exp_names)}")
        with contextlib.redirect_stdout(io.StringIO()):   # run_exp's progress lines
            run0, run1 = (experiment.run_exp(
                stem, sugar, args.out / f"seed{seed}", path_comp, path_con,
                params={**base, "r_poi": rate}, neu_slnc=list(silenced), force_overwrite=True,
                seed=seed) for seed in args.seeds)
        pairs = [("Brian2", f"seed {s0}", compare(brian2, run0, n_run, t_run)),
                 ("Brian2", f"seed {s1}", compare(brian2, run1, n_run, t_run)),
                 (f"seed {s0}", f"seed {s1}", compare(run0, run1, n_run, t_run))]
        b0, b1, own = (c for _, _, c in pairs)
        m0, m1, mo = (c.of(mn9) for c in (b0, b1, own))
        grn = np.mean([(b0.of(i)[0], b0.of(i)[1], b1.of(i)[1]) for i in sugar], axis=0)

        print(f"\n{stem}: the sugar GRNs at {rate:g} Hz, silenced: "
              f"{', '.join(map(str, silenced)) or 'none'}")
        print(f"  {'':<17}{'Brian2':>10}{f'seed {s0}':>11}{'z':>7}{f'seed {s1}':>11}{'z':>7}"
              f"{f'{s0} vs {s1}, z':>16}")
        print(f"  {'spikes per trial':<17}{b0.spikes_a:>10.1f}{b0.spikes_b:>11.1f}"
              f"{b0.z_spikes:>+7.2f}{b1.spikes_b:>11.1f}{b1.z_spikes:>+7.2f}{own.z_spikes:>+16.2f}")
        print(f"  {'MN9 (Hz)':<17}{m0[0]:>10.2f}{m0[1]:>11.2f}{m0[2]:>+7.2f}"
              f"{m1[1]:>11.2f}{m1[2]:>+7.2f}{mo[2]:>+16.2f}")
        print(f"  {'sugar GRNs (Hz)':<17}{grn[0]:>10.2f}{grn[1]:>11.2f}{'':>7}{grn[2]:>11.2f}")
        print(f"  {'neurons |z| > 3':<17}{'':>10}" + "".join(
            f"{f'{int((np.abs(c.z) > 3).sum())} of {c.z.size}':>{width}}"
            for c, width in ((b0, 18), (b1, 18), (own, 16))))
        failed += [f"{stem}, {a} against {b}: {p}" for a, b, c in pairs for p in problems(c, mn9)]

    if failed:
        print("\ngate FAIL\n  " + "\n  ".join(failed))
        return 1
    print(f"\ngate PASS: in all {3 * len(PUBLISHED)} comparisons, spikes per trial and MN9 "
          f"within |z| < {Z_LIMIT:g}, and at most {OUTLIER_SHARE:.0%} of the neurons beyond it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
