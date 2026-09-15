"""What a drive must be before a lane runs it, and the two-rate drive run_exp builds
from upstream's neu_exc and neu_exc2.

No pack needed: building a stimulus reads only the neuron count and the
out-degrees. Runs with a two-rate drive are in test_engines.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from lif import core

N, T, SEED = 1_000, 2_000, 20260914
PACK = SimpleNamespace(
    n_neurons=N, out_degree=np.random.default_rng(1).integers(0, 50, N).astype(np.int32))
# Unsorted, as a list of neurons from a notebook may be.
A = np.random.default_rng(2).permutation(400)[:40].astype(np.int32)
B = np.arange(600, 650, dtype=np.int32)


def stimulus(targets, draws=None):
    targets = np.asarray(targets)
    if draws is None:
        draws = mx.zeros((T, targets.size), dtype=mx.bool_)
    return core.Stimulus(targets=targets, draws=draws, n_ticks=T, rate_hz=float("nan"),
                         seed=-1)


# ---------------------------------------------------------------- Stimulus
def test_a_stimulus_refuses_a_target_listed_twice():
    """The fused lane maps each neuron to one draw column and keeps the last; the
    other lanes add every column. One draw in the first of two columns for the
    same neuron fired it in naive, chunked and metal, and not in fused."""
    with pytest.raises(ValueError, match=r"\[3\]"):
        stimulus([3, 7, 3])


@pytest.mark.parametrize("targets", [[-1, 2], [1.0, 2.0], [True, False], [[1, 2]]])
def test_a_stimulus_needs_non_negative_integer_targets_in_one_dimension(targets):
    with pytest.raises(ValueError):
        stimulus(targets, draws=mx.zeros((T, 2), dtype=mx.bool_))


@pytest.mark.parametrize("draws", [mx.zeros((T, 3), dtype=mx.bool_),
                                   mx.zeros((T - 1, 2), dtype=mx.bool_),
                                   mx.zeros((T, 2), dtype=mx.float32)])
def test_a_stimulus_needs_one_bool_draw_per_tick_and_target(draws):
    """A draw that is not bool reads differently per lane: 0.5 is half an input
    where the lane multiplies and none where it casts to uint8."""
    with pytest.raises(ValueError):
        stimulus([1, 2], draws=draws)


# ---------------------------------------------------------------- two rates
def test_one_set_is_drawn_as_make_stimulus_draws_it():
    ref = core.make_stimulus(PACK, T, SEED, n_targets=100, rate_hz=150.0)
    got = core.make_stimulus_for(PACK, ref.targets, 150.0, T, SEED)
    assert np.array_equal(got.targets, ref.targets)
    assert np.array_equal(np.asarray(got.draws), np.asarray(ref.draws))


@pytest.mark.parametrize("rate2_hz", [0.0, 300.0])
def test_a_second_set_is_appended_and_leaves_the_first_sets_draws_alone(rate2_hz):
    one = core.make_stimulus_for(PACK, A, 150.0, T, SEED)
    two = core.make_stimulus_for(PACK, A, 150.0, T, SEED, targets2=B, rate2_hz=rate2_hz)

    assert np.array_equal(two.targets, np.concatenate([A, B]))
    draws = np.asarray(two.draws)
    assert np.array_equal(draws[:, :A.size], np.asarray(one.draws))
    # p = rate * dt per tick. 80,000 draws at 0.015 and 100,000 at 0.03, so 10 %
    # is more than three standard deviations either way.
    assert draws[:, :A.size].mean() == pytest.approx(150.0 * core.DT / 1000, rel=0.1)
    assert draws[:, A.size:].mean() == pytest.approx(rate2_hz * core.DT / 1000, rel=0.1)


def test_an_empty_set_draws_nothing():
    """run_exp passes upstream's default neu_exc2=[] through; np.asarray([]) is float."""
    got = core.make_stimulus_for(PACK, A, 150.0, T, SEED, targets2=[], rate2_hz=0.0)
    assert np.array_equal(got.targets, A)
    none = core.make_stimulus_for(PACK, [], 150.0, T, SEED)
    assert none.targets.size == 0 and none.draws.shape == (T, 0)


def test_the_second_set_is_never_refractory_even_at_zero_hz():
    """Upstream's poi() sets rfc = 0 ms on every neuron of neu_exc2 whatever r_poi2
    is, so at 0 Hz such a neuron receives no input and still has no refractory
    period. initial_state gives every target that, and both sets are targets."""
    two = core.make_stimulus_for(PACK, A, 150.0, T, SEED, targets2=B, rate2_hz=0.0)
    reload = np.asarray(core.initial_state(PACK, two)["rfc_reload"])
    driven = np.zeros(N, dtype=bool)
    driven[A] = driven[B] = True
    assert (reload[driven] == 0).all()
    assert (reload[~driven] == core.RFC_TICKS).all()


def test_make_stimulus_for_refuses_a_neuron_in_both_sets():
    """Upstream would give it two Poisson inputs; this refuses rather than guess."""
    with pytest.raises(ValueError, match=r"both.*\[3\]"):
        core.make_stimulus_for(PACK, [1, 2, 3], 150.0, T, SEED, targets2=[3, 4],
                               rate2_hz=150.0)


def test_make_stimulus_for_refuses_a_neuron_listed_twice_in_one_set():
    with pytest.raises(ValueError, match=r"\[2\]"):
        core.make_stimulus_for(PACK, [1, 2, 2], 150.0, T, SEED)


@pytest.mark.parametrize("bad", [N, -1])
def test_make_stimulus_for_refuses_targets_outside_the_pack(bad):
    with pytest.raises(ValueError, match=rf"\[{bad}\]"):
        core.make_stimulus_for(PACK, [1, bad], 150.0, T, SEED)
    with pytest.raises(ValueError, match=rf"\[{bad}\]"):
        core.make_stimulus_for(PACK, [1], 150.0, T, SEED, targets2=[bad], rate2_hz=150.0)


@pytest.mark.parametrize("rate", [-1.0, float("nan"), 20_000.0])
def test_make_stimulus_for_refuses_a_rate_that_is_no_probability_per_tick(rate):
    """rate * dt must be from 0 to 1. A negative or NaN rate would otherwise draw no
    input at all, without an error."""
    with pytest.raises(ValueError, match="Hz"):
        core.make_stimulus_for(PACK, [1], rate, T, SEED)
    with pytest.raises(ValueError, match="Hz"):
        core.make_stimulus_for(PACK, [1], 150.0, T, SEED, targets2=[2], rate2_hz=rate)
