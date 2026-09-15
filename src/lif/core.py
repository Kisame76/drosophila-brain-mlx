"""Shared model definition: constants, pack loading, stimulus.

Everything in here is lane-agnostic. The naive (phase 1) and chunked (phase 2)
engines must both build on exactly these numbers and this state layout, or the
parity gate in phase 3 is meaningless.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

# Reuse the compiler's hash function rather than restating it: this loader
# already drifted once by hashing the bare buffer while the compiler hashes
# dtype and shape alongside it. verify_pack.py keeps its own copy on purpose --
# an independent verifier that shares code with the thing it checks is worthless.
from lif.compile_pack import sha256_array

PACK_DIR = Path(__file__).resolve().parents[2] / "data" / "pack" / "v630"

# ---------------------------------------------------------------- constants
# Shiu et al., as encoded in data/ref/model.py. Times in ms, voltages in mV.
DT = 0.1
V_0 = -52.0          # resting potential, also the reset potential
V_TH = -45.0         # threshold; the comparison is strict (v > V_TH)
T_MBR = 20.0         # membrane time constant
TAU = 5.0            # synaptic time constant
T_RFC = 2.2          # refractory period
T_DLY = 1.8          # axonal delay
W_SYN = 0.275        # mV per synaptic contact
F_POI = 250.0        # scaling factor for the external Poisson input
W_EXT = W_SYN * F_POI  # 68.75 mV, applied directly to v

# Straight time/dt quotients, confirmed against Brian2 2.10.1 tick by tick.
#
# Both were briefly "corrected" to 23 / 19 on the strength of a StateMonitor
# trace. That was wrong: StateMonitor records with when='start', i.e. BEFORE the
# state update, so monitor row t holds the state at the END of tick t-1. Reading
# it as same-tick state fakes a one-tick lag in both the delay and the
# refractory release. Compare monitor row t+1 against engine tick t.
RFC_TICKS = round(T_RFC / DT)               # 22
DELAY_TICKS = round(T_DLY / DT)             # 18

# Exact closed-form update over one dt for
#     dv/dt = (V_0 - v + g) / T_MBR
#     dg/dt = -g / TAU
#
# These coefficients are transcribed from the code Brian2 2.10.1 actually
# generates for method='linear' (inspected via neu.state_updater.codeobj.code),
# NOT re-derived. The algebraically obvious form
#     v <- V_0 + (v - V_0)*A + g * (TAU/(TAU-T_MBR)) * (B - A)
# is mathematically identical but rounds differently, and that difference alone
# was enough to flip borderline spikes: 2951 vs Brian2's 2973 on the validation
# subnetwork. Brian2 emits
#     _lio_8  = v_0 - v_0*exp(-dt/t_mbr)
#     _lio_9  = ((tau/(t_mbr-tau)) * (-exp(dt/t_mbr) + exp(dt/tau))) * exp(-dt/t_mbr) * exp(-dt/tau)
#     _lio_10 = exp(-dt/t_mbr)
#     _v = _lio_8 + (_lio_9 * g + _lio_10 * v)
# and the association order below is preserved deliberately. Do not "simplify".
DECAY_V = math.exp(-DT / T_MBR)                                     # _lio_10
DECAY_G = math.exp(-DT / TAU)                                       # _lio_2
V0_TERM = V_0 - V_0 * math.exp(-DT / T_MBR)                         # _lio_8
COUPLE_G = (((TAU / (T_MBR - TAU)) * (-math.exp(DT / T_MBR) + math.exp(DT / TAU)))
            * math.exp(-DT / T_MBR)) * math.exp(-DT / TAU)          # _lio_9


def constants_f32() -> dict:
    """Model constants rounded to the precision the engines actually compute in.

    MLX's Metal backend has no float64, so both lanes run in float32. Deriving
    the coefficients in float64 and then casting once, here, keeps the two lanes
    bit-identical instead of letting each one round on its own.
    """
    return {
        "decay_v": np.float32(DECAY_V),
        "decay_g": np.float32(DECAY_G),
        "couple_g": np.float32(COUPLE_G),
        "v0_term": np.float32(V0_TERM),
        "v_0": np.float32(V_0),
        "v_th": np.float32(V_TH),
        "w_syn": np.float32(W_SYN),
        "w_ext": np.float32(W_EXT),
    }


# ---------------------------------------------------------------- pack
@dataclass
class Pack:
    """Source-major CSR connectome, on the MLX device."""

    n_neurons: int
    n_edges: int
    row_ptr: mx.array        # int32[N+1]
    destinations: mx.array   # int32[E]
    signed_counts: mx.array  # int32[E]
    edge_src: mx.array       # int32[E], expanded from row_ptr
    neuron_ids: np.ndarray   # int64[N], host-side only
    out_degree: np.ndarray   # int32[N], host-side only
    manifest: dict
    path: Path | None = None  # the directory it was loaded from, where its sidecars are


def load_pack(pack_dir: Path = PACK_DIR, verify_hashes: bool = True) -> Pack:
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    arrays = {}
    for name, meta in manifest["arrays"].items():
        arr = np.load(pack_dir / f"{name}.npy")
        if verify_hashes:
            digest = sha256_array(arr)
            if digest != meta["sha256"]:
                raise SystemExit(f"pack corrupt: {name} sha256 {digest[:16]} != {meta['sha256'][:16]}")
        arrays[name] = arr

    row_ptr = arrays["row_ptr"]
    n = int(manifest["neurons"])
    e = int(manifest["edges"])

    # The dense propagation needs a source index per edge. It is derivable from
    # row_ptr, so it is not stored in the pack -- but it costs another 4*E bytes
    # of device memory at runtime. This is the price of source-major CSR under a
    # scatter-add formulation; see PROJECT.md.
    out_degree = np.diff(row_ptr).astype(np.int32)
    edge_src = np.repeat(np.arange(n, dtype=np.int32), out_degree)
    assert edge_src.size == e

    return Pack(
        n_neurons=n,
        n_edges=e,
        row_ptr=mx.array(row_ptr),
        destinations=mx.array(arrays["destinations"]),
        signed_counts=mx.array(arrays["signed_counts"]),
        edge_src=mx.array(edge_src),
        neuron_ids=arrays["neuron_ids"],
        out_degree=out_degree,
        manifest=manifest,
        path=Path(pack_dir),
    )


# ---------------------------------------------------------------- stimulus
@dataclass
class Stimulus:
    """A fully materialised external drive.

    Pre-generating every draw on the host removes the RNG from the parity
    question entirely: both lanes consume the identical bit pattern, so a
    spike-count mismatch can only come from the tick body.
    """

    targets: np.ndarray      # int32[K], neuron indices receiving Poisson input, each once
    draws: mx.array          # bool[T, K]
    n_ticks: int
    rate_hz: float           # nan when the draws are not all at one rate
    seed: int

    def __post_init__(self):
        # What every lane assumes. A neuron listed twice gets the input of both its
        # columns in the naive, chunked and metal lanes and of the last one only in
        # the fused lane, which maps each neuron to one column. A draw that is not
        # bool is scaled by the first three and truncated to uint8 by the fused lane.
        t = self.targets
        if not (isinstance(t, np.ndarray) and t.ndim == 1
                and np.issubdtype(t.dtype, np.integer)):
            raise ValueError(
                f"targets must be a 1-D numpy integer array, got {type(t).__name__} "
                f"dtype={getattr(t, 'dtype', None)} shape={getattr(t, 'shape', None)}")
        if (t < 0).any():
            raise ValueError(f"targets must be neuron indices, got {t[t < 0].tolist()}")
        values, times = np.unique(t, return_counts=True)
        if (times > 1).any():
            raise ValueError(f"targets listed more than once: {values[times > 1].tolist()}")
        d = self.draws
        if (not isinstance(d, mx.array) or d.dtype != mx.bool_
                or tuple(d.shape) != (self.n_ticks, t.size)):
            raise ValueError(
                f"draws must be a bool mx.array of shape ({self.n_ticks}, {t.size}), got "
                f"{type(d).__name__} dtype={getattr(d, 'dtype', None)} "
                f"shape={tuple(getattr(d, 'shape', ()))}")

    def sha256(self) -> str:
        return hashlib.sha256(np.asarray(self.draws, dtype=np.uint8).tobytes()).hexdigest()


def make_stimulus(
    pack: Pack, n_ticks: int, seed: int, n_targets: int = 100, rate_hz: float = 150.0
) -> Stimulus:
    """Pick target neurons and draw their Poisson input for every tick.

    Targets are the `n_targets` highest out-degree neurons, tie-broken by index.
    That is a deterministic, biologically meaningless choice -- it exists so the
    drive actually propagates through the network instead of dying in a corner,
    and so the selection is reproducible without shipping a neuron list.
    """
    order = np.lexsort((np.arange(pack.n_neurons), -pack.out_degree))
    targets = np.sort(order[:n_targets]).astype(np.int32)
    return make_stimulus_for(pack, targets, rate_hz, n_ticks, seed)


def make_stimulus_for(
    pack: Pack, targets, rate_hz: float, n_ticks: int, seed: int,
    targets2=(), rate2_hz: float = 0.0,
) -> Stimulus:
    """Poisson input on chosen neurons: targets at rate_hz, targets2 at rate2_hz.

    What upstream's run_exp gives neu_exc at r_poi and neu_exc2 at r_poi2, here as
    neuron indices. Its poi() also takes the refractory period away from every
    neuron in either list, at any rate, 0 Hz included. initial_state does the same
    for all targets, so a neuron of targets2 at 0 Hz receives no input and still has
    no refractory period, which changes a run once that neuron fires.

    The first set is drawn before the second from one generator, so adding a second
    set leaves the first set's draws as they were. A neuron in both sets would get
    two inputs upstream; this refuses it rather than guess.
    """
    sets = []
    for name, idx, rate in (("targets", targets, rate_hz), ("targets2", targets2, rate2_hz)):
        idx = np.asarray(idx)
        if idx.size == 0:
            idx = idx.astype(np.int32)  # np.asarray([]) is float64
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"{name} must be a list of neuron indices, got "
                             f"dtype={idx.dtype} shape={idx.shape}")
        outside = idx[(idx < 0) | (idx >= pack.n_neurons)]
        if outside.size:
            raise ValueError(f"{name} outside 0 to {pack.n_neurons - 1}: {outside.tolist()}")
        p = rate * (DT / 1000.0)  # 150 Hz * 0.1 ms = 0.015
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} at {rate} Hz would be drawn with probability {p} per "
                             f"tick; the rate must be from 0 to {1000.0 / DT:g} Hz")
        sets.append((idx.astype(np.int32), p))
    (a, p_a), (b, p_b) = sets
    both = np.intersect1d(a, b)
    if both.size:
        raise ValueError(f"neurons in both targets and targets2: {both.tolist()}")

    rng = np.random.default_rng(seed)
    draws = np.hstack([rng.random((n_ticks, a.size)) < p_a,
                       rng.random((n_ticks, b.size)) < p_b])
    return Stimulus(
        targets=np.concatenate([a, b]),
        draws=mx.array(draws),
        n_ticks=n_ticks,
        rate_hz=rate_hz if b.size == 0 else float("nan"),
        seed=seed,
    )


# ---------------------------------------------------------------- silencing
def silenced_mask(pack: Pack, silenced: np.ndarray | None) -> mx.array | None:
    """Validate a per-run silencing mask and put it on the device; None stays None.

    Silencing neuron i sets every synapse FROM i to zero weight, which is what
    data/ref/model.py's `silence` does (`syn.w['<i> == i'] = 0*mV`). The upstream
    README says synapses "to and from" the neuron; the code, which produced the
    published results, says from, and this follows the code. A silenced neuron
    still integrates, still spikes and still counts; it delivers nothing. It may
    also be a stimulus target.

    Strict on purpose. Without this check an index array in place of the mask
    would be read out of bounds by the dense lanes, without any error, and a
    mask of length 1 would broadcast to every neuron in the kernel lanes.
    """
    if silenced is None:
        return None
    if (not isinstance(silenced, np.ndarray) or silenced.dtype != np.bool_
            or silenced.shape != (pack.n_neurons,)):
        raise ValueError(
            f"silenced must be a numpy bool array of shape ({pack.n_neurons},), got "
            f"{type(silenced).__name__} dtype={getattr(silenced, 'dtype', None)} "
            f"shape={getattr(silenced, 'shape', None)}")
    return mx.array(silenced)


# ---------------------------------------------------------------- state
def initial_state(pack: Pack, stim: Stimulus) -> dict:
    """Initial v, g, refractory counters and delay ring.

    model.py sets v = v_0, g = 0, rfc = 2.2 ms for all neurons, then overrides
    rfc = 0 for every Poisson target -- so the driven neurons are never
    refractory. That override is reproduced here via a per-neuron reload value.
    """
    n = pack.n_neurons
    rfc_reload = np.full(n, RFC_TICKS, dtype=np.int32)
    rfc_reload[stim.targets] = 0

    return {
        "v": mx.full((n,), V_0, dtype=mx.float32),
        "g": mx.zeros((n,), dtype=mx.float32),
        "rfc": mx.zeros((n,), dtype=mx.int32),
        "rfc_reload": mx.array(rfc_reload),
        "ring": mx.zeros((DELAY_TICKS, n), dtype=mx.bool_),
        "counts": mx.zeros((n,), dtype=mx.int32),
    }
