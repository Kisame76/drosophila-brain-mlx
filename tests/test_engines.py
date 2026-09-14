"""End-to-end checks: every lane must agree, bit for bit.

Needs a compiled pack (tools/fetch_upstream.sh, then python -m lif.compile_pack);
the tests skip cleanly if it is absent. Ticks are kept small so the whole file
runs in well under a minute.
"""

from __future__ import annotations

import numpy as np
import pytest

from lif import core

pytestmark = pytest.mark.skipif(
    not (core.PACK_DIR / "manifest.json").exists(),
    reason="no compiled pack; run tools/fetch_upstream.sh then python -m lif.compile_pack",
)

TICKS = 400
SEED = 20260913


@pytest.fixture(scope="module")
def pack():
    return core.load_pack()


@pytest.fixture(scope="module")
def stim(pack):
    return core.make_stimulus(pack, n_ticks=TICKS, seed=SEED)


def test_pack_is_consistent(pack):
    rp = np.asarray(pack.row_ptr)
    assert rp[0] == 0
    assert rp[-1] == pack.n_edges
    assert np.all(np.diff(rp) >= 0)
    dst = np.asarray(pack.destinations)
    assert dst.min() >= 0 and dst.max() < pack.n_neurons
    assert np.asarray(pack.signed_counts).all(), "no edge may carry a zero count"


@pytest.mark.parametrize("lane", ["chunked", "metal", "fused"])
def test_lane_matches_naive(pack, stim, lane):
    """Every optimisation must reproduce the baseline exactly -- the phase-3 gate."""
    from lif import engine_chunked, engine_fused, engine_metal, engine_naive

    ref = engine_naive.run(pack, stim, warmup=0)
    got = {
        "chunked": lambda: engine_chunked.run(pack, stim, chunk=32, warmup=0),
        "metal": lambda: engine_metal.run(pack, stim, chunk=32, warmup=0),
        "fused": lambda: engine_fused.run(pack, stim, chunk=32, warmup=0, edge_split=1),
    }[lane]()

    assert got.counts_sha256() == ref.counts_sha256()
    assert np.array_equal(got.v_final, ref.v_final)
    assert np.array_equal(got.g_final, ref.g_final)


def test_metal_kernel_is_deterministic(pack):
    """int32 atomics must be order-independent, or the parity gate is luck."""
    import mlx.core as mx

    from lif import engine_metal

    rng = np.random.default_rng(7)
    spike = mx.array(rng.random(pack.n_neurons) < 0.01)
    n_src = mx.array([pack.n_neurons], dtype=mx.uint32)
    runs = [np.asarray(engine_metal.propagate(spike, pack, n_src)) for _ in range(5)]
    for other in runs[1:]:
        assert np.array_equal(runs[0], other)


def test_metal_kernel_matches_dense_formulation(pack):
    """The hand-written kernel must equal the pure-MLX scatter it replaces."""
    import mlx.core as mx

    from lif import engine_metal

    rng = np.random.default_rng(3)
    spike = mx.array(rng.random(pack.n_neurons) < 0.002)
    n_src = mx.array([pack.n_neurons], dtype=mx.uint32)
    got = engine_metal.propagate(spike, pack, n_src)

    active = spike[pack.edge_src]
    vals = mx.where(active, pack.signed_counts, mx.array(0, dtype=mx.int32))
    want = mx.zeros((pack.n_neurons,), dtype=mx.int32).at[pack.destinations].add(vals)
    mx.eval(got, want)
    assert np.array_equal(np.asarray(got), np.asarray(want))


def test_edge_split_does_not_change_results(pack):
    """EDGE_SPLIT is a performance knob; it must never alter the outcome."""
    import mlx.core as mx

    from lif import engine_metal

    rng = np.random.default_rng(5)
    spike = mx.array(rng.random(pack.n_neurons) < 0.005)
    n_src = mx.array([pack.n_neurons], dtype=mx.uint32)
    base = np.asarray(engine_metal.propagate(spike, pack, n_src, 1))
    for split in (2, 4, 16):
        assert np.array_equal(
            np.asarray(engine_metal.propagate(spike, pack, n_src, split)), base
        )
