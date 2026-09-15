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
RELAY_TICKS = 100
CHUNK = 32
SEED = 20260913
LANES = ["naive", "chunked", "metal", "fused"]


@pytest.fixture(scope="module")
def pack():
    return core.load_pack()


@pytest.fixture(scope="module")
def stim(pack):
    return core.make_stimulus(pack, n_ticks=TICKS, seed=SEED)


def run_lane(lane, pack, stim, silenced=None, **kw):
    """kw goes to the lane: record for any, cap only for the kernel lanes."""
    from lif import engine_chunked, engine_fused, engine_metal, engine_naive

    return {
        "naive": lambda: engine_naive.run(pack, stim, silenced=silenced, warmup=0, **kw),
        "chunked": lambda: engine_chunked.run(pack, stim, silenced=silenced, chunk=CHUNK,
                                              warmup=0, **kw),
        "metal": lambda: engine_metal.run(pack, stim, silenced=silenced, chunk=CHUNK,
                                          warmup=0, **kw),
        "fused": lambda: engine_fused.run(pack, stim, silenced=silenced, chunk=CHUNK,
                                          warmup=0, edge_split=1, **kw),
    }[lane]()


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
    ref = run_lane("naive", pack, stim)
    got = run_lane(lane, pack, stim)

    assert got.counts_sha256() == ref.counts_sha256()
    assert np.array_equal(got.v_final, ref.v_final)
    assert np.array_equal(got.g_final, ref.g_final)


# ---------------------------------------------------------------- silencing
def _top_out_degree(pack, candidates, n=10):
    """The n highest out-degree neurons among candidates, tie-broken by index."""
    candidates = np.asarray(candidates)
    return candidates[np.lexsort((candidates, -pack.out_degree[candidates]))][:n]


@pytest.fixture(scope="module", params=["sugar", "hubs"])
def silencing_case(request, pack, stim):
    """A stimulus, a silencing mask, and the naive runs without and with it.

    The design asked for the 10 highest out-degree neurons under the sugar
    stimulus. Those neurons never fire under it (0 spikes at 400 and at 10,000
    ticks), so silencing them changes nothing and parity would hold vacuously.
    Two cases instead:
      sugar  the 10 highest out-degree neurons among those that do fire under
             the sugar drive
      hubs   the 10 highest out-degree neurons overall under the 100-hub drive,
             which excites them directly, so excitation and silencing overlap
    """
    import mlx.core as mx

    from lif import engine_naive, stimulus_flybrain

    if request.param == "sugar":
        targets = stimulus_flybrain.targets(pack.neuron_ids)
        draws = stimulus_flybrain.bernoulli(len(targets), TICKS, 150.0, core.DT, SEED)
        case_stim = core.Stimulus(targets=targets, draws=mx.array(draws), n_ticks=TICKS,
                                  rate_hz=150.0, seed=SEED)
        base = engine_naive.run(pack, case_stim, warmup=0)
        chosen = _top_out_degree(pack, np.nonzero(base.spike_counts > 0)[0])
    else:
        case_stim = stim
        base = engine_naive.run(pack, case_stim, warmup=0)
        chosen = _top_out_degree(pack, np.arange(pack.n_neurons))

    mask = np.zeros(pack.n_neurons, dtype=bool)
    mask[chosen] = True
    ref = engine_naive.run(pack, case_stim, silenced=mask, warmup=0, record=True)
    return case_stim, mask, base, ref


@pytest.mark.parametrize("lane", ["chunked", "metal", "fused"])
def test_silencing_matches_naive(pack, silencing_case, lane):
    """Two implementations of one semantics must agree: a masked copy of the
    edge counts (naive, chunked) and empty edge ranges in the kernel (metal, fused).
    Recorded, so the spike events are compared too."""
    case_stim, mask, _, ref = silencing_case
    got = run_lane(lane, pack, case_stim, silenced=mask, record=True)

    assert got.counts_sha256() == ref.counts_sha256()
    assert np.array_equal(got.v_final, ref.v_final)
    assert np.array_equal(got.g_final, ref.g_final)
    assert np.array_equal(got.events, ref.events)


def test_silencing_changes_the_result(silencing_case):
    """Guards the parity test above against passing because nothing was silenced."""
    _, mask, base, ref = silencing_case
    assert base.spike_counts[mask].sum() > 0, "the silenced neurons must fire"
    assert ref.total_spikes() != base.total_spikes()


@pytest.fixture(scope="module")
def relay(pack):
    """A source S whose single target T has no other input, and a drive on S alone.

    The design placed this on the 800-neuron validation subnetwork, where it
    does not exist: no neuron there has out-degree 1 into a single-input
    target, and all 114 single-input neurons are fed by an inhibitory seed hub,
    so they could never fire. The full pack has 56 excitatory pairs of this
    shape; this takes the one with the most contacts.
    """
    import mlx.core as mx

    rp = np.asarray(pack.row_ptr)
    dst = np.asarray(pack.destinations)
    cnt = np.asarray(pack.signed_counts)
    in_degree = np.bincount(dst, minlength=pack.n_neurons)
    src = np.nonzero(pack.out_degree == 1)[0]
    tgt, contacts = dst[rp[src]], cnt[rp[src]]
    ok = (in_degree[tgt] == 1) & (tgt != src) & (contacts > 0)
    src, tgt, contacts = src[ok], tgt[ok], contacts[ok]
    best = np.lexsort((src, -contacts))[0]

    # One input every other tick. Each lifts S by 68.75 mV, S fires on the next
    # tick, so S alone produces exactly one spike per draw.
    draws = np.zeros((RELAY_TICKS, 1), dtype=bool)
    draws[::2, 0] = True
    relay_stim = core.Stimulus(targets=np.array([src[best]], dtype=np.int32),
                               draws=mx.array(draws), n_ticks=RELAY_TICKS,
                               rate_hz=float("nan"), seed=-1)
    return int(src[best]), int(tgt[best]), relay_stim, int(draws.sum())


@pytest.mark.parametrize("lane", LANES)
def test_silenced_source_still_fires_but_delivers_nothing(pack, relay, lane):
    s, t, relay_stim, n_draws = relay
    base = run_lane(lane, pack, relay_stim)
    assert base.spike_counts[t] > 0, "T must fire from S alone, or silencing S proves nothing"

    mask = np.zeros(pack.n_neurons, dtype=bool)
    mask[s] = True
    got = run_lane(lane, pack, relay_stim, silenced=mask)

    assert got.spike_counts[s] == n_draws
    assert got.spike_counts[t] == 0
    assert got.total_spikes() == n_draws, "no other neuron may receive anything"


@pytest.mark.parametrize("lane", LANES)
def test_silenced_neuron_still_receives_input(pack, relay, lane):
    """Silencing is outgoing only, as upstream's model.py does it; its README
    says "to and from". A silenced T still integrates S's input and fires."""
    _, t, relay_stim, _ = relay
    base = run_lane(lane, pack, relay_stim)

    mask = np.zeros(pack.n_neurons, dtype=bool)
    mask[t] = True
    got = run_lane(lane, pack, relay_stim, silenced=mask)

    assert base.spike_counts[t] > 0
    assert got.spike_counts[t] == base.spike_counts[t]


@pytest.mark.parametrize("lane", LANES)
def test_silenced_must_be_a_bool_mask_over_all_neurons(pack, stim, lane):
    """An index list passed where the mask belongs would otherwise be read out
    of bounds, silently, by the kernel lanes."""
    with pytest.raises(ValueError):
        run_lane(lane, pack, stim, silenced=np.array([3, 7]))
    with pytest.raises(ValueError):
        run_lane(lane, pack, stim, silenced=np.zeros(pack.n_neurons - 1, dtype=bool))


# ---------------------------------------------------------------- recording
@pytest.fixture(scope="module")
def plain_naive(pack, stim):
    return run_lane("naive", pack, stim)


@pytest.fixture(scope="module")
def recorded(pack, stim):
    """Every lane on the shared stimulus with spike-event recording on."""
    return {lane: run_lane(lane, pack, stim, record=True) for lane in LANES}


@pytest.mark.parametrize("lane", LANES)
def test_recording_is_off_by_default(pack, relay, lane):
    assert run_lane(lane, pack, relay[2]).events is None


@pytest.mark.parametrize("lane", ["chunked", "metal", "fused"])
def test_events_match_naive(recorded, lane):
    """The kernel lanes extract events in one Metal dispatch per chunk; the naive
    lane calls np.nonzero on each tick, a code path they share nothing with."""
    assert np.array_equal(recorded[lane].events, recorded["naive"].events)


@pytest.mark.parametrize("lane", LANES)
def test_events_are_sorted_unique_and_agree_with_counts(pack, recorded, lane):
    r = recorded[lane]
    ev = r.events
    assert ev.dtype == np.int32 and ev.ndim == 2 and ev.shape[1] == 2
    assert len(ev) == r.total_spikes() > 0
    assert ev[:, 0].min() >= 0 and ev[:, 0].max() < TICKS
    key = ev[:, 0].astype(np.int64) * pack.n_neurons + ev[:, 1]
    assert np.all(np.diff(key) > 0), "sorted by (tick, neuron), and no event twice"
    assert np.array_equal(np.bincount(ev[:, 1], minlength=pack.n_neurons), r.spike_counts)


@pytest.mark.parametrize("lane", LANES)
def test_recording_does_not_change_the_run(plain_naive, recorded, lane):
    got = recorded[lane]
    assert got.counts_sha256() == plain_naive.counts_sha256()
    assert np.array_equal(got.v_final, plain_naive.v_final)
    assert np.array_equal(got.g_final, plain_naive.g_final)


@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.parametrize("lane", ["chunked", "metal", "fused"])
def test_recording_with_warmup_and_default_chunks(pack, stim, recorded, lane, use_async):
    """The default path: warmup runs the event kernel as well and none of its events
    may reach the result, at each lane's own chunk length (256, 64, 32), draining
    after async_eval and after a blocking eval."""
    from lif import engine_chunked, engine_fused, engine_metal

    run = {"chunked": engine_chunked.run, "metal": engine_metal.run,
           "fused": engine_fused.run}[lane]
    got = run(pack, stim, record=True, warmup=50, use_async=use_async)
    assert np.array_equal(got.events, recorded["naive"].events)


def test_recording_one_chunk_longer_than_the_run(pack, stim, recorded):
    from lif import engine_fused

    got = engine_fused.run(pack, stim, chunk=4 * TICKS, warmup=0, edge_split=1, record=True)
    assert np.array_equal(got.events, recorded["naive"].events)


def test_recorded_fused_runs_are_deterministic(pack, stim, recorded):
    """Three runs in all. Threads race for event slots; the sorted events must not vary."""
    for _ in range(2):
        again = run_lane("fused", pack, stim, record=True)
        assert np.array_equal(again.events, recorded["fused"].events)


def _spikes_per_chunk(recorded):
    ticks = recorded["naive"].events[:, 0]
    return np.bincount(ticks // CHUNK, minlength=-(-TICKS // CHUNK))


@pytest.mark.parametrize("lane", ["chunked", "metal", "fused"])
def test_record_overflow_names_the_first_overfull_chunk(pack, stim, recorded, lane):
    """cap is the first chunk's own spike count: that chunk fits exactly, and the
    overflow must name a later chunk by that chunk's ticks."""
    from lif.spike_record import RecordOverflow

    per_chunk = _spikes_per_chunk(recorded)
    cap = int(per_chunk[0])
    first = int(np.argmax(per_chunk > cap))
    assert cap >= 1 and first > 0 and per_chunk[first] > cap, \
        "a later chunk must overfill the first chunk's count"

    with pytest.raises(RecordOverflow) as exc:
        run_lane(lane, pack, stim, record=True, cap=cap)
    assert exc.value.tick_range == range(first * CHUNK, min((first + 1) * CHUNK, TICKS))
    assert exc.value.count == per_chunk[first]
    assert exc.value.cap == cap


def test_a_cap_equal_to_the_fullest_chunk_does_not_raise(pack, stim, recorded):
    cap = int(_spikes_per_chunk(recorded).max())
    got = run_lane("fused", pack, stim, record=True, cap=cap)
    assert np.array_equal(got.events, recorded["fused"].events)


# ---------------------------------------------------------------- kernel
def test_metal_kernel_is_deterministic(pack):
    """int32 atomics must be order-independent, or the parity gate is luck."""
    import mlx.core as mx

    from lif import engine_metal

    rng = np.random.default_rng(7)
    spike = mx.array(rng.random(pack.n_neurons) < 0.01)
    n_src = mx.array([pack.n_neurons], dtype=mx.uint32)
    row_end = engine_metal.silenced_row_end(pack, None)
    runs = [np.asarray(engine_metal.propagate(spike, pack, n_src, row_end=row_end))
            for _ in range(5)]
    for other in runs[1:]:
        assert np.array_equal(runs[0], other)


def test_silenced_row_end_empties_only_silenced_edge_ranges(pack):
    """The kernel lanes silence a source by ending its edge range where it starts.
    Every other range is untouched, and without a mask none is."""
    import mlx.core as mx

    from lif import engine_metal

    rp = np.asarray(pack.row_ptr)
    assert np.array_equal(np.asarray(engine_metal.silenced_row_end(pack, None)), rp[1:])

    mask = np.random.default_rng(11).random(pack.n_neurons) < 0.5
    got = np.asarray(engine_metal.silenced_row_end(pack, mx.array(mask)))
    assert np.array_equal(got, np.where(mask, rp[:-1], rp[1:]))


@pytest.mark.parametrize("masked", [False, True])
def test_metal_kernel_matches_dense_formulation(pack, masked):
    """The hand-written kernel must equal the pure-MLX scatter it replaces,
    without a silencing mask and with one."""
    import mlx.core as mx

    from lif import engine_metal

    rng = np.random.default_rng(3)
    spike = mx.array(rng.random(pack.n_neurons) < 0.002)
    silenced = mx.array(rng.random(pack.n_neurons) < 0.5) if masked else None
    n_src = mx.array([pack.n_neurons], dtype=mx.uint32)
    got = engine_metal.propagate(spike, pack, n_src,
                                 row_end=engine_metal.silenced_row_end(pack, silenced))

    active = spike[pack.edge_src]
    if masked:
        active = mx.logical_and(active, mx.logical_not(silenced[pack.edge_src]))
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
    row_end = engine_metal.silenced_row_end(pack, None)
    base = np.asarray(engine_metal.propagate(spike, pack, n_src, 1, row_end=row_end))
    for split in (2, 4, 16):
        assert np.array_equal(
            np.asarray(engine_metal.propagate(spike, pack, n_src, split, row_end=row_end)), base
        )
