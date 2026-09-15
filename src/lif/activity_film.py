"""The activity film: where in the male CNS the spikes of one experiment happen.

Drives the right labellum's LB3b and LB3c gustatory neurons in the MaleCNS pack at
100 Hz through run_exp, 30 trials of 1 s. Tastekin et al. (2025, bioRxiv
10.1101/2025.08.25.671814) match both types to Gr64f-GAL4 neurons, which makes them
likely sweet-sensing. Every recorded spike is drawn at its neuron's soma position
from the MaleCNS annotation table, in an animated PNG with one frame per 10 ms of
biological time, the spikes of all trials summed, while the view turns once about
the CNS's long axis. Neurons the table gives no soma position, the driven ones among
them, are counted but not drawn. The film shows where activity is, not why: it is a
visualization, not evidence.

    python -m lif.activity_film
"""

from __future__ import annotations

import argparse
import collections
import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow import feather

from lif import core, experiment
from lif import names as lif_names
from lif.compile_pack import sha256_file
from lif.compile_pack_malecns import SOURCE_FILES

# The right labellum's LB3b and LB3c neurons, as MaleCNS instances.
DRIVE = ("LB3b_R", "LB3c_R")

MARGIN, STRIP = 10, 40            # pixels around the scene, and below it for the time
FOG_LEVELS, ACT_LEVELS = 31, 8    # the palette holds every fog level with every activity level
FOG_RGB = (60, 82, 120)
FOG_GAIN, FOG_BLUR = 5.0, 3       # blurred neuron weight per pixel that makes 63 % fog; passes
FOG_GAMMA = 0.5                   # lifts thin fog, so the ventral nerve cord's outline shows
RATE_HZ = 40.0                    # a neuron firing at this rate in a bin is lit to 63 %
GLOW, GLOW_BLUR = 3.0, 6
FIRE_AT = (0.0, 0.35, 0.7, 1.0)
FIRE_RGB = ((0, 0, 0), (230, 90, 20), (255, 190, 70), (255, 255, 225))


def _scene_rgb(fog: np.ndarray, act: np.ndarray) -> np.ndarray:
    fire = np.stack([np.interp(act, FIRE_AT, [c[k] for c in FIRE_RGB]) for k in range(3)], axis=-1)
    return fog[..., None] * np.array(FOG_RGB) * (1 - act[..., None]) + fire


_FOG, _ACT = np.meshgrid(np.linspace(0, 1, FOG_LEVELS), np.linspace(0, 1, ACT_LEVELS), indexing="ij")
# Index f * ACT_LEVELS + a is fog level f with activity level a; three colours follow.
TEXT, TRACK, BAR = FOG_LEVELS * ACT_LEVELS, FOG_LEVELS * ACT_LEVELS + 1, FOG_LEVELS * ACT_LEVELS + 2
PALETTE = np.vstack([np.rint(np.clip(_scene_rgb(_FOG.ravel(), _ACT.ravel()), 0, 255)),
                     [(200, 200, 200), (60, 60, 60), (230, 150, 60)]]).astype(np.uint8)

# 5 by 7 pixel glyphs for the time below the scene.
FONT = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "m": ("00000", "00000", "11010", "10101", "10101", "10101", "10101"),
    "s": ("00000", "00000", "01111", "10000", "01110", "00001", "11110"),
    " ": ("00000",) * 7,
}


def soma_positions(annotations: pa.Table, neuron_ids: np.ndarray) -> np.ndarray:
    """float64[N, 3]: each neuron's somaLocation in pack order, NaN where there is none."""
    if "somaLocation" not in annotations.column_names:
        raise ValueError("annotations lack the columns ['somaLocation']")
    ids = np.asarray(neuron_ids, dtype=np.int64)
    column = annotations["somaLocation"].combine_chunks().take(
        pa.array(lif_names.annotation_rows(annotations, ids)))
    placed = column.is_valid().to_numpy(zero_copy_only=False)
    length = pc.list_value_length(column).fill_null(3).to_numpy(zero_copy_only=False)
    if (length != 3).any():
        raise ValueError(f"somaLocation is not three numbers for bodies {ids[length != 3].tolist()}")
    positions = np.full((ids.size, 3), np.nan)
    positions[placed] = column.flatten().to_numpy(zero_copy_only=False).reshape(-1, 3)
    return positions


def spike_bins(t: np.ndarray, neuron: np.ndarray, t_run: float,
               bin_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """The model indices that spiked, and int64[A, B] spikes of each in each bin of
    bin_ms, over all trials, from spike times in seconds on the engine's tick grid."""
    per_bin = round(bin_ms / core.DT)
    if per_bin < 1 or not math.isclose(per_bin * core.DT, bin_ms, rel_tol=1e-9):
        raise ValueError(f"a bin of {bin_ms} ms is not a whole number of {core.DT} ms ticks")
    n_ticks = round(t_run * 1000 / core.DT)
    if n_ticks % per_bin:
        raise ValueError(f"bins of {bin_ms} ms do not tile a run of {t_run} s")
    t = np.asarray(t, dtype=np.float64)
    tick = np.rint(t * 1000 / core.DT)
    off = np.abs(tick * core.DT / 1000 - t) > 1e-9
    if off.any():
        raise ValueError(f"spike times between ticks: {t[off].tolist()}")
    outside = (tick < 0) | (tick >= n_ticks)
    if outside.any():
        raise ValueError(f"spike times outside 0 to {t_run} s: {t[outside].tolist()}")
    n_bins = n_ticks // per_bin
    active, inverse = np.unique(np.asarray(neuron), return_inverse=True)
    counts = np.bincount(inverse * n_bins + tick.astype(np.int64) // per_bin,
                         minlength=active.size * n_bins)
    return active, counts.reshape(active.size, n_bins)


@dataclass(frozen=True, eq=False)
class View:
    """An orthographic camera that turns about the z axis, the CNS's long axis."""

    width: int
    scene_height: int
    height: int
    center: np.ndarray   # float64[3], the middle of the positions' bounding box
    radius: float        # the largest distance of a position from the z axis through center
    scale: float         # pixels per unit of the annotation table

    @classmethod
    def fit(cls, positions: np.ndarray, width: int) -> View:
        """The view in which every one of positions stays in the picture at every angle."""
        lo, hi = positions.min(axis=0), positions.max(axis=0)
        center = (lo + hi) / 2
        radius = float(np.hypot(*(positions[:, :2] - center[:2]).T).max())
        scale = (width - 1 - 2 * MARGIN) / (2 * radius)
        scene_height = math.ceil(scale * (hi[2] - lo[2])) + 1 + 2 * MARGIN
        return cls(width, scene_height, scene_height + STRIP, center, radius, scale)

    def project(self, positions: np.ndarray, theta: float):
        """Pixel column, pixel row and depth from -1, nearest, to 1 of each position.

        At theta 0 screen right is x and screen down z, so a larger y is nearer: a
        right-handed view, not its mirror image."""
        x, y, z = (np.asarray(positions, dtype=np.float64) - self.center).T
        c, s = math.cos(theta), math.sin(theta)
        return ((self.width - 1) / 2 + self.scale * (x * c + y * s),
                (self.scene_height - 1) / 2 + self.scale * z,
                (x * s - y * c) / self.radius)


def _splat(view: View, u: np.ndarray, v: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """The weights summed per pixel of the scene, each on its nearest pixel."""
    col, row = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    if col.size and (col.min() < 0 or col.max() >= view.width
                     or row.min() < 0 or row.max() >= view.scene_height):
        raise ValueError("positions outside the view; fit the view to them")
    return np.bincount(row * view.width + col, weights=np.broadcast_to(weight, col.shape),
                       minlength=view.scene_height * view.width).reshape(view.scene_height, view.width)


def _blur(img: np.ndarray, passes: int) -> np.ndarray:
    """Passes of a [1, 2, 1] / 4 kernel along rows, then columns; borders stay."""
    img = img.copy()
    for _ in range(passes):
        img[:, 1:-1] = 0.25 * img[:, :-2] + 0.5 * img[:, 1:-1] + 0.25 * img[:, 2:]
        img[1:-1] = 0.25 * img[:-2] + 0.5 * img[1:-1] + 0.25 * img[2:]
    return img


def draw_text(img: np.ndarray, text: str, x: int, y: int, index: int, scale: int = 2) -> None:
    """Write text into an image of palette indices, its top left corner at (x, y)."""
    for ch in text:
        if ch not in FONT:
            raise ValueError(f"the font has no glyph for {ch!r}")
        glyph = np.kron(np.array([[bit == "1" for bit in row] for row in FONT[ch]]),
                        np.ones((scale, scale), dtype=bool))
        region = img[y:y + glyph.shape[0], x:x + glyph.shape[1]]
        region[glyph[:region.shape[0], :region.shape[1]]] = index
        x += 6 * scale


def frame(view: View, fog_positions: np.ndarray, active_positions: np.ndarray,
          rate_hz: np.ndarray, theta: float, label: str, progress: float) -> np.ndarray:
    """One picture as uint8 palette indices: every one of fog_positions as fog, denser
    where nearer, each of active_positions lit by its rate on its pixel and less on the
    eight around it, with a glow, and below them label and a bar filled to progress."""
    u, v, depth = view.project(fog_positions, theta)
    fog = 1 - np.exp(-_blur(_splat(view, u, v, 1.15 - 0.5 * depth), FOG_BLUR) / FOG_GAIN)
    u, v, _ = view.project(active_positions, theta)
    lit = _splat(view, u, v, 1 - np.exp(-np.asarray(rate_hz, dtype=np.float64) / RATE_HZ))
    # One blur pass times 4 keeps the neuron's own pixel at its value.
    act = np.clip(4 * _blur(lit, 1) + GLOW * _blur(lit, GLOW_BLUR), 0, 1)
    fog **= FOG_GAMMA
    img = np.zeros((view.height, view.width), dtype=np.uint8)
    img[:view.scene_height] = (np.rint(fog * (FOG_LEVELS - 1)) * ACT_LEVELS
                               + np.rint(act * (ACT_LEVELS - 1)))
    y = view.scene_height + (STRIP - 14) // 2
    draw_text(img, label, 14, y, TEXT)
    x0, x1 = 26 + 12 * len(label), view.width - 14
    img[y + 6:y + 8, x0:x1] = TRACK
    img[y + 4:y + 10, x0:x0 + round((x1 - x0) * progress)] = BAR
    return img


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _paeth_scanlines(pixels: np.ndarray) -> bytes:
    """PNG scanlines of one byte per pixel, each with filter type 4, Paeth."""
    x = pixels.astype(np.int16)
    a = np.zeros_like(x)
    a[:, 1:] = x[:, :-1]
    b = np.zeros_like(x)
    b[1:] = x[:-1]
    c = np.zeros_like(x)
    c[1:, 1:] = x[:-1, :-1]
    p = a + b - c
    pa, pb, pc_ = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    predicted = np.where((pa <= pb) & (pa <= pc_), a, np.where(pb <= pc_, b, c))
    rows = ((x - predicted) % 256).astype(np.uint8)
    return np.hstack([np.full((rows.shape[0], 1), 4, dtype=np.uint8), rows]).tobytes()


def apng(frames: Sequence[np.ndarray], palette: np.ndarray, delay_ms: int) -> bytes:
    """An animated PNG of palette-indexed frames, each shown delay_ms, looping."""
    palette = np.asarray(palette, dtype=np.uint8)
    if not 1 <= len(palette) <= 256:
        raise ValueError(f"a PNG palette holds 1 to 256 colours, not {len(palette)}")
    shape = frames[0].shape
    for f in frames:
        if f.ndim != 2 or f.shape != shape:
            raise ValueError(f"frames differ in shape: {shape} and {f.shape}")
        if int(f.max(initial=0)) >= len(palette):
            raise ValueError(f"a frame uses index {int(f.max())}, beyond a palette of "
                             f"{len(palette)} colours")
    height, width = shape
    out = [b"\x89PNG\r\n\x1a\n",
           _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)),
           _chunk(b"PLTE", palette.tobytes()),
           _chunk(b"acTL", struct.pack(">II", len(frames), 0))]
    sequence = 0
    for i, f in enumerate(frames):
        out.append(_chunk(b"fcTL", struct.pack(">IIIIIHHBB", sequence, width, height, 0, 0,
                                               delay_ms, 1000, 0, 0)))
        sequence += 1
        data = zlib.compress(_paeth_scanlines(f), 9)
        if i == 0:
            out.append(_chunk(b"IDAT", data))
        else:
            out.append(_chunk(b"fdAT", struct.pack(">I", sequence) + data))
            sequence += 1
    out.append(_chunk(b"IEND", b""))
    return b"".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--pack", type=Path, default=root / "data/pack/male_cns_v1")
    ap.add_argument("--annotations", type=Path,
                    default=root / "data/raw/male_cns" / SOURCE_FILES["annotations"])
    ap.add_argument("--drive", nargs="+", default=list(DRIVE),
                    help="instances or types from the pack's names sidecar")
    ap.add_argument("--rate", type=float, default=100.0, help="Poisson input on each in Hz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin-ms", type=float, default=10.0)
    ap.add_argument("--slowdown", type=float, default=8.0,
                    help="seconds of film per second of biological time")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--report", nargs="*", default=["MN9"],
                    help="names whose rates are printed")
    ap.add_argument("--results", type=Path, default=root / "outputs/activity_film")
    ap.add_argument("--out", type=Path, default=root / "docs/figures/activity-film.png")
    args = ap.parse_args()

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

    params = dict(experiment.default_params, r_poi=args.rate)
    n_run, t_run = params["n_run"], params["t_run"]
    path = experiment.run_exp(f"{'+'.join(args.drive)}_{args.rate:g}Hz", args.drive, args.results,
                              params=params, force_overwrite=True, pack=pack, seed=args.seed)
    spikes = pq.read_table(path, columns=["t", "flywire_id"])
    ids = spikes["flywire_id"].to_numpy()
    sorter = np.argsort(pack.neuron_ids)
    neuron = sorter[np.searchsorted(pack.neuron_ids, ids, sorter=sorter)]
    active, counts = spike_bins(spikes["t"].to_numpy(), neuron, t_run, args.bin_ms)

    table = experiment.rates(path)
    rate_of = dict(zip(table["flywire_id"].to_pylist(), table["rate_hz"].to_pylist(), strict=True))
    mean = np.array([rate_of[int(i)] for i in pack.neuron_ids[active]])
    if len(rate_of) != active.size or not np.allclose(counts.sum(axis=1) / (n_run * t_run), mean,
                                                      rtol=1e-12, atol=0):
        raise SystemExit("the binned spikes do not add up to the rates rates() reads from the file")

    annotations = feather.read_table(args.annotations, columns=["bodyId", "somaLocation", "superclass"])
    positions = soma_positions(annotations, pack.neuron_ids)
    superclass = np.array(annotations["superclass"].take(
        pa.array(lif_names.annotation_rows(annotations, pack.neuron_ids))).to_pylist(), dtype=object)
    placed = ~np.isnan(positions[:, 0])
    lit = placed[active]
    view = View.fit(positions[placed], args.width)
    n_bins = counts.shape[1]
    rate = counts[lit] / (n_run * args.bin_ms / 1000)
    frames = [frame(view, positions[placed], positions[active[lit]], rate[:, k],
                    2 * math.pi * k / n_bins, f"{(k + 1) * args.bin_ms:>4g} ms", (k + 1) / n_bins)
              for k in range(n_bins)]
    delay = round(args.bin_ms * args.slowdown)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(apng(frames, PALETTE, delay))

    driven = experiment.metadata(path)["neu_exc"]
    print(f"{dataset}: {len(driven)} neurons ({', '.join(args.drive)}) at {args.rate:g} Hz, "
          f"{n_run} trials of {t_run:g} s, seed {args.seed}")
    print(f"{placed.sum()} of {pack.n_neurons} neurons have a soma position")
    print(f"{ids.size} spikes from {active.size} neurons: {lit.sum()} drawn, {(~lit).sum()} "
          f"without a soma position, by superclass "
          f"{dict(collections.Counter(superclass[active[~lit]]).most_common())}")
    for name in args.report:
        for i in names.select(name):
            label = names.labels().get(int(pack.neuron_ids[i]), name)
            print(f"{label}: {rate_of.get(int(pack.neuron_ids[i]), 0.0):.2f} Hz")
    print(f"wrote {args.out}: {n_bins} frames of {delay} ms, {view.width}x{view.height}, "
          f"{args.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
