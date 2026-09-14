"""Reimplementation of flyBrain's splitmix64 counter stimulus.

Used only to compare the two engines under a bit-identical input. Transcribed
from work/flyBrain/rust/src/stimulus.rs (EventSchedule::bernoulli + splitmix64).
Lane order follows RIGHT_SUGAR_GRN_IDS as written, not sorted -- protocol.rs
pushes indices in array order.
"""

from __future__ import annotations

import numpy as np

RIGHT_SUGAR_GRN_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def _splitmix64(value: np.ndarray) -> np.ndarray:
    v = value + _GOLDEN
    v = (v ^ (v >> np.uint64(30))) * _M1
    v = (v ^ (v >> np.uint64(27))) * _M2
    return v ^ (v >> np.uint64(31))


def bernoulli(n_targets: int, steps: int, rate_hz: float, dt_ms: float,
              seed: int) -> np.ndarray:
    """Return a bool[steps, n_targets] draw matrix identical to the Rust engine."""
    probability = rate_hz * dt_ms / 1000.0
    if probability > 1.0:
        raise ValueError("rate x timestep cannot exceed one for N=1 input")
    tick = np.arange(steps, dtype=np.uint64)[:, None]
    lane = np.arange(n_targets, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        key = np.uint64(seed) ^ (tick * _GOLDEN) ^ (lane * _M1)
        sample = _splitmix64(key)
    uniform = (sample >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))
    return uniform < probability


def targets(neuron_ids: np.ndarray) -> np.ndarray:
    """Model indices of the sugar GRNs, in the engine's lane order."""
    pos = {int(v): i for i, v in enumerate(neuron_ids)}
    return np.array([pos[i] for i in RIGHT_SUGAR_GRN_IDS if i in pos], dtype=np.int32)
