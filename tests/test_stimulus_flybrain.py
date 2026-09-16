"""The sugar drive: its targets, and that its draws are flyBrain's draws.

No pack needed: the lookup is checked against synthetic neuron ID arrays, and the
draws against a scalar transcription of the Rust that produced them.

`stimulus_flybrain.bernoulli` exists so the two engines can be compared under the
same input, which is only worth anything if it really is the same input. The
README asserts that; these tests gate it. The algorithm itself is small and was
transcribed once; what can drift silently is the vectorised numpy form of it --
broadcasting, dtypes, uint64 wraparound -- so the reference below is written with
plain Python integers and explicit 64-bit masking, loop for loop from
work/flyBrain/rust/src/stimulus.rs (EventSchedule::bernoulli and splitmix64).
"""

from __future__ import annotations

import numpy as np
import pytest

from lif import core, stimulus_flybrain

GRNS = stimulus_flybrain.RIGHT_SUGAR_GRN_IDS
_MASK64 = (1 << 64) - 1


def test_targets_follow_the_id_list_not_the_pack_order():
    ids = np.array([5, *reversed(GRNS), 7], dtype=np.int64)
    assert ids[stimulus_flybrain.targets(ids)].tolist() == GRNS


def test_targets_refuses_a_pack_missing_a_sugar_grn():
    ids = np.array([5, *GRNS[:-1], 7], dtype=np.int64)
    with pytest.raises(ValueError, match=str(GRNS[-1])):
        stimulus_flybrain.targets(ids)


def _splitmix64(value: int) -> int:
    """stimulus.rs's splitmix64, in plain integers."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _bernoulli(n_targets: int, steps: int, rate_hz: float, dt_ms: float,
               seed: int) -> np.ndarray:
    """EventSchedule::bernoulli from the same file, one draw at a time."""
    probability = rate_hz * dt_ms / 1000.0
    out = np.zeros((steps, n_targets), dtype=bool)
    for tick in range(steps):
        for lane in range(n_targets):
            key = (seed
                   ^ ((tick * 0x9E3779B97F4A7C15) & _MASK64)
                   ^ ((lane * 0xBF58476D1CE4E5B9) & _MASK64)) & _MASK64
            uniform = (_splitmix64(key) >> 11) * (1.0 / float(1 << 53))
            out[tick, lane] = uniform < probability
    return out


@pytest.mark.parametrize(("n_targets", "steps", "rate_hz", "seed"), [
    (2, 100, 150.0, 7),           # the configuration flyBrain's own test uses
    (21, 200, 150.0, 20260816),   # the sugar drive, at the seed lif.benchmark runs
    (3, 50, 800.0, 0),            # a rate that fires often, and seed 0
    (1, 32, 150.0, 2**63 + 5),    # a seed above the signed range, to catch a cast
])
def test_the_vectorised_draws_match_the_scalar_rust_transcription(n_targets, steps,
                                                                  rate_hz, seed):
    """The vectorised form is the one that can drift; the scalar one is the spec."""
    fast = stimulus_flybrain.bernoulli(n_targets, steps, rate_hz, 0.1, seed)
    slow = _bernoulli(n_targets, steps, rate_hz, 0.1, seed)
    assert fast.shape == slow.shape == (steps, n_targets)
    assert fast.dtype == np.bool_
    np.testing.assert_array_equal(fast, slow)


def test_flybrains_own_test_configuration_draws_where_it_does():
    """bernoulli(vec![1, 3], 100, 150.0, 0.1, 7) in stimulus.rs. Its assertions are
    only determinism and 'something fired', so the positions are pinned here."""
    draws = stimulus_flybrain.bernoulli(2, 100, 150.0, 0.1, 7)
    assert [(int(t), int(lane)) for t, lane in zip(*np.nonzero(draws))] == [
        (12, 0), (16, 1), (77, 1)]


def test_the_sugar_drive_makes_the_input_spike_count_the_benchmark_reports():
    """lif.benchmark prints '3109 input spikes' for this configuration and the
    README quotes runs made with it, so a change that moves this number changes
    published results. Same function, so this pins rather than derives it."""
    assert core.DT == 0.1, "the benchmark's draws are at dt = 0.1 ms"
    draws = stimulus_flybrain.bernoulli(21, 10_000, 150.0, core.DT, 20260816)
    assert int(draws.sum()) == 3109


def test_a_rate_that_would_need_more_than_one_draw_per_tick_is_refused():
    with pytest.raises(ValueError, match="cannot exceed one"):
        stimulus_flybrain.bernoulli(2, 10, 20_000.0, 0.1, 0)


@pytest.mark.parametrize("rate_hz", [-150.0, float("nan"), float("inf")])
def test_a_rate_that_is_not_a_finite_non_negative_number_is_refused(rate_hz):
    """`uniform < probability` is False for all three, so an unguarded draw
    matrix is silently all-False and the run looks like a run with no input.
    stimulus.rs bails on exactly these, and so does core.make_stimulus_for."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        stimulus_flybrain.bernoulli(2, 10, rate_hz, 0.1, 0)
