"""The lanes' performance knobs must refuse a value they cannot honour.

No pack needed: a three-neuron chain is enough, and every check here raises
before the first kernel dispatch, so nothing runs on the device.

These are the two knobs README hands the reader -- `edge_split` ("the one knob
that matters") and `chunk` -- and both used to accept a value that produced a
wrong answer instead of an error: `edge_split=0` dispatched no threads at all and
simulated a network with no synapses, `chunk=0` never advanced the tick cursor
and hung. `warmup` is here for the same reason: negative, with record=True, it
reached mx.stack with nothing to stack.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from lif import core, engine_chunked, engine_fused, engine_metal, engine_naive

CHUNKED_LANES = [engine_chunked, engine_metal, engine_fused]
ALL_LANES = [engine_naive, *CHUNKED_LANES]


@pytest.fixture(scope="module")
def pack():
    """0 -> 1 -> 2, one contact each, as core.load_pack would hand it over."""
    row_ptr = np.array([0, 1, 2, 2], dtype=np.int32)
    return core.Pack(
        n_neurons=3,
        n_edges=2,
        row_ptr=mx.array(row_ptr),
        destinations=mx.array(np.array([1, 2], dtype=np.int32)),
        signed_counts=mx.array(np.array([60, 60], dtype=np.int32)),
        edge_src=mx.array(np.array([0, 1], dtype=np.int32)),
        neuron_ids=np.array([10, 11, 12], dtype=np.int64),
        out_degree=np.diff(row_ptr).astype(np.int32),
        manifest={"dataset": "synthetic"},
    )


@pytest.fixture(scope="module")
def stim(pack):
    return core.make_stimulus_for(pack, [0], 150.0, n_ticks=8, seed=1)


@pytest.mark.parametrize("split", [0, -1])
def test_a_kernel_lane_refuses_a_non_positive_edge_split(split):
    """0 used to dispatch zero threads and return the zero init_value in silence."""
    with pytest.raises(ValueError, match="edge_split"):
        engine_metal._kernel_for(split)


@pytest.mark.parametrize("lane", CHUNKED_LANES, ids=lambda m: m.__name__)
@pytest.mark.parametrize("chunk", [0, -1])
def test_a_chunked_lane_refuses_a_non_positive_chunk(lane, pack, stim, chunk):
    """min(chunk, n - t) is 0 for these, so the tick cursor never advances."""
    with pytest.raises(ValueError, match="chunk"):
        lane.run(pack, stim, chunk=chunk)


@pytest.mark.parametrize("lane", ALL_LANES, ids=lambda m: m.__name__)
def test_every_lane_refuses_a_negative_warmup(lane, pack, stim):
    with pytest.raises(ValueError, match="warmup"):
        lane.run(pack, stim, warmup=-1)
