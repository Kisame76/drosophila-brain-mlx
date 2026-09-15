"""The control demo: does the wiring matter?

Drives the 21 right-hemisphere sugar GRNs of upstream's example notebook at
100 Hz, as its MN9 cells do, through run_exp twice: on the v630 pack and on its
degree-shuffled copy, with the same seed and so the same input spikes on both.
From the recorded spikes of both runs it draws one SVG: MN9's rate over time in
the top row, and in the bottom row the sugar GRNs' own mean rate, which shows the
input arriving in both.

    python -m lif.shuffle_pack --pack data/pack/v630 --seed 0
    python -m lif.control_demo
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lif import core, experiment, stimulus_flybrain

# The proboscis motor neuron upstream's example notebook reads out.
MN9 = 720575940660219265


def rate_over_time(
    t: np.ndarray, n_run: int, t_run: float, bin_s: float, n_neurons: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges in seconds and each bin's rate in spikes per neuron per trial per
    second, from the spike times of n_neurons neurons over all trials."""
    n_bins = round(t_run / bin_s)
    if n_bins < 1 or not math.isclose(n_bins * bin_s, t_run, rel_tol=1e-9):
        raise ValueError(f"a bin of {bin_s} s does not tile a run of {t_run} s")
    t = np.asarray(t, dtype=np.float64)
    outside = (t < 0) | (t >= t_run)
    if outside.any():
        raise ValueError(f"spike times outside 0 to {t_run} s: {t[outside].tolist()}")
    edges = np.linspace(0.0, t_run, n_bins + 1)
    counts, _ = np.histogram(t, bins=edges)
    return edges, counts / (n_neurons * n_run * (t_run / n_bins))


@dataclass
class Panel:
    title: str           # above the plot, "" for none
    subtitle: str
    edges: np.ndarray    # float[B+1], seconds
    rate_hz: np.ndarray  # float[B]
    note: str = ""       # written across the plot, for one with nothing in it


PLOT_W, PLOT_H, LEFT, GAP = 330, 170, 56, 40
TITLE_H, HEAD_H, AXIS_H = 40, 48, 48
ROW_H = HEAD_H + PLOT_H + AXIS_H


def _axis_top(peak: float) -> float:
    """The smallest 1, 2 or 5 times a power of ten at or above peak, 10 when silent."""
    if peak <= 0:
        return 10.0
    step = 10.0 ** math.floor(math.log10(peak))
    return next(m * step for m in (1, 2, 5, 10) if m * step >= peak)


def figure(rows: list[list[Panel]], ylabels: list[str], title: str, caption: str) -> str:
    """Rows of bar charts of rate over time as SVG text, the panels of a row side by
    side on one y axis."""
    lines = caption.splitlines()
    columns = max(len(row) for row in rows)
    width = LEFT + columns * PLOT_W + (columns - 1) * GAP + 28
    height = TITLE_H + len(rows) * ROW_H + 16 * len(lines) + 12
    out = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" '
         f'font-size="12" fill="#222222">'),
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{LEFT}" y="24" font-size="15" font-weight="bold">{escape(title)}</text>',
    ]
    for r, (row, ylabel) in enumerate(zip(rows, ylabels, strict=True)):
        top = _axis_top(max(float(p.rate_hz.max(initial=0.0)) for p in row))
        y0 = TITLE_H + r * ROW_H + HEAD_H
        out.append(f'<text transform="translate(16 {y0 + PLOT_H / 2:.2f}) rotate(-90)" '
                   f'text-anchor="middle">{escape(ylabel)}</text>')
        for k, p in enumerate(row):
            out.extend(_panel(p, LEFT + k * (PLOT_W + GAP), y0, top))
    for i, line in enumerate(lines):
        out.append(f'<text x="{LEFT}" y="{TITLE_H + len(rows) * ROW_H + 16 * i + 4}" '
                   f'fill="#555555">{escape(line)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _panel(p: Panel, x0: int, y0: int, top: float) -> list[str]:
    bottom = y0 + PLOT_H
    t0, span = float(p.edges[0]), float(p.edges[-1] - p.edges[0])
    out = []
    if p.title:
        out.append(f'<text x="{x0}" y="{y0 - 30}" font-weight="bold">{escape(p.title)}</text>')
    out.append(f'<text x="{x0}" y="{y0 - 12}" fill="#555555">{escape(p.subtitle)}</text>')
    for v in (0.0, top / 2, top):
        y = bottom - PLOT_H * v / top
        out.append(f'<line x1="{x0}" x2="{x0 + PLOT_W}" y1="{y:.2f}" y2="{y:.2f}" '
                   f'stroke="#e5e5e5"/>')
        out.append(f'<text x="{x0 - 6}" y="{y + 4:.2f}" text-anchor="end">{v:g}</text>')
    for a, b, r in zip(p.edges[:-1], p.edges[1:], p.rate_hz, strict=True):
        h = PLOT_H * float(r) / top
        out.append(f'<rect class="bar" x="{x0 + PLOT_W * (a - t0) / span:.2f}" '
                   f'y="{bottom - h:.2f}" width="{PLOT_W * (b - a) / span:.2f}" '
                   f'height="{h:.2f}" fill="#3b6ea5"/>')
    if p.note:
        # Just above the middle grid line, which would otherwise run through the text.
        out.append(f'<text x="{x0 + PLOT_W / 2:.2f}" y="{y0 + PLOT_H / 2 - 8:.2f}" '
                   f'text-anchor="middle" fill="#555555">{escape(p.note)}</text>')
    out.append(f'<path d="M{x0} {y0}V{bottom}H{x0 + PLOT_W}" fill="none" stroke="#222222"/>')
    for frac in (0.0, 0.5, 1.0):
        out.append(f'<text x="{x0 + PLOT_W * frac:.2f}" y="{bottom + 16}" '
                   f'text-anchor="middle">{(t0 + span * frac) * 1000:g}</text>')
    out.append(f'<text x="{x0 + PLOT_W / 2:.2f}" y="{bottom + 32}" '
               f'text-anchor="middle">time (ms)</text>')
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--real", type=Path, default=root / "data/pack/v630")
    ap.add_argument("--shuffled", type=Path, default=root / "data/pack/v630-shuffled-seed0")
    ap.add_argument("--rate", type=float, default=100.0,
                    help="Poisson input on each sugar GRN in Hz; the notebook's MN9 cells use 100")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin-ms", type=float, default=20.0)
    ap.add_argument("--results", type=Path, default=root / "outputs/control_demo")
    ap.add_argument("--out", type=Path, default=root / "docs/figures/control-demo.svg")
    args = ap.parse_args()

    if not (args.shuffled / "manifest.json").is_file():
        raise SystemExit(f"no pack at {args.shuffled}; build it with "
                         f"python -m lif.shuffle_pack --pack {args.real}")
    real, shuffled = core.load_pack(args.real), core.load_pack(args.shuffled)
    if (shuffled.manifest.get("derived_from", {}).get("arrays")
            != {k: v["sha256"] for k, v in real.manifest["arrays"].items()}):
        raise SystemExit(f"{args.shuffled} is not a shuffled copy of {args.real}")

    params = dict(experiment.default_params, r_poi=args.rate)
    n_run, t_run, bin_s = params["n_run"], params["t_run"], args.bin_ms / 1000
    sugar = [int(i) for i in stimulus_flybrain.RIGHT_SUGAR_GRN_IDS]
    mn9_row, sugar_row = [], []
    for label, pack, title in (
        ("real", real, f"Real wiring: {real.manifest['dataset']}"),
        ("shuffled", shuffled,
         f"Shuffled wiring, same degrees: seed {shuffled.manifest['shuffle']['seed']}"),
    ):
        path = experiment.run_exp(f"sugarR_{args.rate:g}Hz", sugar, args.results / label,
                                  params=params, force_overwrite=True, pack=pack, seed=args.seed)
        spikes = pq.read_table(path, columns=["t", "flywire_id"])
        ids, t = spikes["flywire_id"].to_numpy(), spikes["t"].to_numpy()
        edges, mn9 = rate_over_time(t[ids == MN9], n_run, t_run, bin_s)
        _, grn = rate_over_time(t[np.isin(ids, sugar)], n_run, t_run, bin_s, n_neurons=len(sugar))

        table = experiment.rates(path)
        rate_of = dict(zip(table["flywire_id"].to_pylist(), table["rate_hz"].to_pylist(), strict=True))
        mn9_mean = rate_of.get(MN9, 0.0)
        grn_mean = sum(rate_of.get(i, 0.0) for i in sugar) / len(sugar)
        for what, binned, mean in (("MN9", mn9, mn9_mean), ("sugar GRN", grn, grn_mean)):
            if not math.isclose(float(binned.mean()), mean, rel_tol=1e-9, abs_tol=1e-12):
                raise SystemExit(f"{label}: binned {what} rate {binned.mean()} Hz, rates() {mean} Hz")

        n_neurons = int(np.unique(ids).size)
        print(f"{label:<9} {pack.manifest['dataset']:<36} MN9 {mn9_mean:6.2f} Hz  sugar GRNs "
              f"{grn_mean:6.2f} Hz  {ids.size:>9} spikes  {n_neurons:>6} neurons spiked")
        mn9_row.append(Panel(
            title, f"MN9 {mn9_mean:.1f} Hz; brain: {ids.size:,} spikes, {n_neurons:,} neurons",
            edges, mn9, note="" if mn9_mean else f"no MN9 spike in {n_run} trials"))
        sugar_row.append(Panel("", f"the {len(sugar)} sugar GRNs: {grn_mean:.1f} Hz each on average",
                               edges, grn))

    caption = (f"{n_run} trials of {t_run:g} s on each pack, the same input spikes on both "
               f"(seed {args.seed}), fused lane.\nBars: spikes per {args.bin_ms:g} ms bin, divided "
               "by neurons, trials and bin width.\nRegenerate with python -m lif.control_demo.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(figure(
        [mn9_row, sugar_row], ["MN9 rate (Hz)", "sugar GRN rate (Hz)"],
        f"MN9 while the 21 right sugar GRNs receive Poisson input at {args.rate:g} Hz", caption))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
