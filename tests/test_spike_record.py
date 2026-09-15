"""The spike-event kernel on its own, against np.nonzero. Needs no pack."""

from __future__ import annotations

import pickle

import mlx.core as mx
import numpy as np
import pytest

from lif import core, spike_record


def _nonzero_events(stack: np.ndarray, t0: int = 0) -> np.ndarray:
    ticks, neurons = np.nonzero(stack)
    return np.column_stack([ticks + t0, neurons]).astype(np.int32)


def _submit(rec, stack, t0, dtype=mx.bool_):
    out = rec.submit([mx.array(row).astype(dtype) for row in stack], t0)
    mx.eval(*out)


@pytest.mark.parametrize("dtype", [mx.bool_, mx.uint8])
def test_events_match_nonzero(dtype):
    """bool is what the dense and sparse lanes emit per tick, uint8 the fused lane."""
    stack = np.random.default_rng(1).random((32, 5003)) < 0.01
    rec = spike_record.Recorder(5003)
    _submit(rec, stack, 0, dtype)
    events = rec.events()

    assert events.dtype == np.int32 and events.shape == (int(stack.sum()), 2)
    assert np.array_equal(events, _nonzero_events(stack))


def test_chunks_are_offset_by_their_first_tick_and_sorted_across_chunks():
    """Three chunks, the last one short, drained with the one-chunk lag the lanes use."""
    n = 997
    stack = np.random.default_rng(2).random((71, n)) < 0.02
    rec = spike_record.Recorder(n)
    for t0, k in ((0, 32), (32, 32), (64, 7)):
        _submit(rec, stack[t0:t0 + k], t0)
        rec.drain(keep=1)

    assert np.array_equal(rec.events(), _nonzero_events(stack))


def test_no_spikes_gives_an_empty_event_array():
    rec = spike_record.Recorder(100)
    _submit(rec, np.zeros((32, 100), dtype=bool), 0)
    events = rec.events()
    assert events.shape == (0, 2) and events.dtype == np.int32
    assert spike_record.Recorder(100).events().shape == (0, 2)


def test_sorted_events_are_deterministic():
    """The slot a thread gets is a race; the sorted event set must not be."""
    stack = np.random.default_rng(3).random((32, 20011)) < 0.05
    runs = []
    for _ in range(5):
        rec = spike_record.Recorder(20011)
        _submit(rec, stack, 0)
        runs.append(rec.events())
    for other in runs[1:]:
        assert np.array_equal(runs[0], other)


def test_overflow_names_the_chunk_and_a_full_cap_does_not_raise():
    n = 50
    stack = np.zeros((8, n), dtype=bool)
    stack[5, [4, 9, 20, 41]] = True       # 4 spikes, in the chunk of ticks 4 to 7

    exact = spike_record.Recorder(n, cap=4)
    _submit(exact, stack[:4], 0)
    _submit(exact, stack[4:], 4)
    assert np.array_equal(exact.events(), _nonzero_events(stack))

    rec = spike_record.Recorder(n, cap=3)
    _submit(rec, stack[:4], 0)             # no spikes, drained fine
    _submit(rec, stack[4:], 4)
    with pytest.raises(spike_record.RecordOverflow) as exc:
        rec.drain()
    assert exc.value.tick_range == range(4, 8)
    assert (exc.value.count, exc.value.cap) == (4, 3)
    assert "ticks 4 to 7" in str(exc.value)
    with pytest.raises(spike_record.RecordOverflow):
        rec.events()                       # caught once, never truncated afterwards


def test_record_overflow_survives_pickling():
    """So a run in a worker process can hand the overflow back to its parent."""
    err = spike_record.RecordOverflow(range(32, 64), 10, 5)
    back = pickle.loads(pickle.dumps(err))
    assert (back.tick_range, back.count, back.cap) == (range(32, 64), 10, 5)
    assert str(back) == str(err)


def test_cap_must_fit_an_mlx_shape():
    """MLX shapes are int32; a larger cap would abort the process instead of raising."""
    for cap in (0, 2**31):
        with pytest.raises(ValueError):
            spike_record.Recorder(10, cap=cap)
    assert spike_record.Recorder(10, cap=2**31 - 1).cap == 2**31 - 1


def test_tick_to_seconds_has_no_offset():
    """Engine tick k is Brian2's clock time k * dt; validate_brian2 checks it."""
    got = spike_record.tick_to_seconds(np.array([0, 1, 37, 10_000], dtype=np.int32))
    assert got.dtype == np.float64
    assert np.array_equal(got, np.array([0, 1, 37, 10_000], dtype=np.float64) * (core.DT / 1000.0))
