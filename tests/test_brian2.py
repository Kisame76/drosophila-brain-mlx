"""Spike times against Brian2: the correctness gate that can fail when all lanes agree.

Skips without Brian2 (pip install -e '.[reference]') or without the FlyWire pack.
The table in README.md, "Correctness", comes from python -m lif.validate_brian2.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from lif import core

pytest.importorskip("brian2")
pytestmark = pytest.mark.skipif(
    not (core.PACK_DIR / "manifest.json").exists(),
    reason="no compiled pack; run tools/fetch_upstream.sh then python -m lif.compile_pack",
)

# (ticks, seeds, rate_hz): the configurations of the README table
CONFIGS = [(500, 4, 150.0), (2000, 20, 400.0), (2000, 40, 800.0), (3000, 60, 1200.0)]
# The first 350 ticks of the last one, before float32 first moves a spike (at
# 35.1 ms). 549 of its 2,795 spikes are fired by neurons that are not driven, so
# comparing it checks synaptic delay and propagation; the shortest configuration
# has only driven neurons' spikes, one tick after each input.
PREFIX = (350, 60, 1200.0)


def _id(config):
    return f"{config[1]}seeds-{config[2]:g}Hz-{config[0]}ticks"


@functools.cache
def _compare(n_ticks, n_seeds, rate_hz):
    from lif import validate_brian2

    return validate_brian2.compare(n_ticks, n_seeds, rate_hz)


@pytest.mark.parametrize("config", CONFIGS, ids=_id)
def test_ref64_spike_times_equal_brian2(config):
    """Every (neuron, time) Brian2's SpikeMonitor records, exactly, with engine
    tick k converted as k * dt and no offset."""
    c = _compare(*config)

    assert len(c.brian2_events) == int(c.brian2_counts.sum()) > 0
    assert c.ref64_exact()


@pytest.mark.parametrize("config", CONFIGS, ids=_id)
def test_fused_spike_times_are_within_float32_rounding(config):
    assert _compare(*config).mlx_within_float32_rounding()


@pytest.mark.parametrize("config", [CONFIGS[0], PREFIX], ids=_id)
def test_fused_spike_times_equal_brian2_until_float32_moves_one(config):
    c = _compare(*config)
    driven = set(c.seeds.tolist())
    propagated = sum(1 for neuron, _ in c.brian2_events if neuron not in driven)
    assert propagated > 0 if config == PREFIX else propagated == 0

    assert np.array_equal(c.mlx_counts, c.brian2_counts)
    assert c.mlx_events == c.brian2_events
