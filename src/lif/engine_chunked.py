"""Phase 2: chunked evaluation.

Same semantics as engine_naive, three changes:
  - N ticks are encoded before a single mx.eval()
  - the tick body can be fused with mx.compile (OFF by default: it breaks parity)
  - mx.async_eval lets the next chunk encode while the current one commits

No Python control flow, no .item(), no host readback inside a chunk.

The delay ring is 18 separate arrays held in a Python list, rotated by index.
Rotating a list slot is trace-time bookkeeping -- the arrays themselves are
never copied, concatenated or reallocated, and the compiled body never sees the
ring at all: it takes the one delayed mask it needs and returns the new spike
mask. That keeps a single compiled variant with stable shapes, instead of the
18 variants (or a full-ring masked reduction) that indexing inside the body
would have forced.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from lif import core, spike_record
from lif.engine_naive import RunResult


def make_step(pack: core.Pack, signed_counts: mx.array, c: dict, targets: mx.array):
    """One tick. Identical arithmetic to engine_naive.tick, same order."""

    def step(v, g, rfc, counts, rfc_reload, delayed, stim_row):
        # --- 1. refractory refresh, then the exact closed-form update.
        rfc = mx.maximum(rfc - 1, 0)
        not_ref = rfc == 0
        # Association order transcribed from Brian2's generated code; see core.py.
        v_upd = c["v0_term"] + (g * c["couple_g"] + v * c["decay_v"])
        g_upd = g * c["decay_g"]
        v = mx.where(not_ref, v_upd, v)
        g = mx.where(not_ref, g_upd, g)

        # --- 2. strict threshold, gated by not-refractory.
        spike = mx.logical_and(not_ref, v > c["v_th"])

        # --- 3/4. propagate the mask emitted DELAY_TICKS ago, then external drive.
        active = delayed[pack.edge_src]
        vals = mx.where(active, signed_counts, mx.array(0, dtype=mx.int32))
        contrib = mx.zeros((pack.n_neurons,), dtype=mx.int32).at[pack.destinations].add(vals)
        # See engine_naive.tick: "(unless refractory)" shields v and g from
        # synaptic writes too, so arrivals at a refractory neuron are dropped.
        g = g + mx.where(not_ref, contrib.astype(mx.float32) * c["w_syn"], 0.0)
        # Gate and scatter on the ~100 driven neurons only. Materialising a full
        # zeros(N) buffer and masking it cost 20% of the tick in the sparse lane
        # for 100 values. core.Stimulus refuses a target listed twice, so the
        # scatter cannot collide and the arithmetic is unchanged.
        v = v.at[targets].add(
            mx.where(not_ref[targets], stim_row.astype(mx.float32) * c["w_ext"], 0.0))

        # --- 5. reset and refractory reload. The ring store happens outside.
        v = mx.where(spike, c["v_0"], v)
        g = mx.where(spike, mx.array(0.0, dtype=mx.float32), g)
        rfc = mx.where(spike, rfc_reload, rfc)

        return v, g, rfc, counts + spike.astype(mx.int32), spike

    return step


def run(pack: core.Pack, stim: core.Stimulus, silenced: np.ndarray | None = None,
        chunk: int = 256, compile_body: bool = False, use_async: bool = True,
        warmup: int = 50, record: bool = False, cap: int = spike_record.CAP) -> RunResult:
    """Run the chunked lane.

    compile_body defaults to False because mx.compile BREAKS the parity gate.
    It fuses the elementwise ops into one kernel where Metal applies FMA
    contraction: mathematically equivalent, differently rounded in float32.
    The drift is invisible for thousands of ticks, then tips a neuron across the
    strict v > V_TH threshold and the trajectories diverge chaotically -- 54270
    vs 52472 spikes at 10k ticks. MLX exposes no way to forbid the contraction.
    It bought 4%; it is not worth an invalid result. Kept as an opt-in flag so
    the effect stays reproducible.

    record=True also returns the spike events, extracted on the device once per
    chunk (lif.spike_record); cap is the most events one chunk may produce.
    """
    # min(chunk, n - t) is the tick loop's step. At a chunk of 0 or less it is 0,
    # the cursor never advances and the run hangs instead of failing. A negative
    # warmup passes `if w:` below and then encodes an empty range, which reaches
    # mx.stack with nothing to stack once record is on.
    if chunk < 1:
        raise ValueError(f"chunk must be at least 1, got {chunk}")
    if warmup < 0:
        raise ValueError(f"warmup must be 0 or more, got {warmup}")

    # Edges of silenced sources carry a zero count, once per run; see
    # engine_naive.run.
    signed_counts = pack.signed_counts
    mask = core.silenced_mask(pack, silenced)
    if mask is not None:
        signed_counts = mx.where(mask[pack.edge_src], mx.array(0, dtype=mx.int32),
                                 pack.signed_counts)
        mx.eval(signed_counts)

    c = {k: mx.array(v) for k, v in core.constants_f32().items()}
    targets = mx.array(stim.targets)
    step = make_step(pack, signed_counts, c, targets)
    if compile_body:
        step = mx.compile(step)

    n = stim.n_ticks
    N = pack.n_neurons

    def fresh():
        st = core.initial_state(pack, stim)
        ring = [mx.zeros((N,), dtype=mx.bool_) for _ in range(core.DELAY_TICKS)]
        return st, ring

    def encode(st, ring, t0, k, spikes=None):
        v, g, rfc, counts = st["v"], st["g"], st["rfc"], st["counts"]
        rl = st["rfc_reload"]
        for t in range(t0, t0 + k):
            s = t % core.DELAY_TICKS
            v, g, rfc, counts, spike = step(v, g, rfc, counts, rl, ring[s], stim.draws[t])
            ring[s] = spike  # read-then-overwrite: same slot, DELAY_TICKS apart
            if spikes is not None:
                spikes.append(spike)
        st = {"v": v, "g": g, "rfc": rfc, "counts": counts, "rfc_reload": rl}
        return st, ring

    # Warm up outside the measured window: kernel compilation, the mx.compile
    # trace, and first-touch allocation all happen here and are then discarded.
    # With recording on, the event kernel compiles here too; its events are
    # never read.
    w = min(warmup, n)
    if w:
        st, ring = fresh()
        spikes = [] if record else None
        st, ring = encode(st, ring, 0, w, spikes)
        out = spike_record.Recorder(N, cap).submit(spikes, 0) if record else []
        mx.eval(*st.values(), *ring, *out)

    state, ring = fresh()
    mx.eval(*state.values(), *ring)
    rec = spike_record.Recorder(N, cap) if record else None

    mx.reset_peak_memory()
    start = time.perf_counter()
    t = 0
    while t < n:
        k = min(chunk, n - t)
        spikes = [] if rec is not None else None
        state, ring = encode(state, ring, t, k, spikes)
        out = rec.submit(spikes, t) if rec is not None else []
        if use_async:
            mx.async_eval(*state.values(), *ring, *out)
        else:
            mx.eval(*state.values(), *ring, *out)
        if rec is not None:
            rec.drain(keep=1)  # the previous chunk; see spike_record.Recorder
        t += k
    mx.eval(*state.values(), *ring)
    events = rec.events() if rec is not None else None
    elapsed = time.perf_counter() - start

    return RunResult(
        spike_counts=np.asarray(state["counts"]),
        v_final=np.asarray(state["v"]),
        g_final=np.asarray(state["g"]),
        n_ticks=n,
        seconds=elapsed,
        peak_bytes=mx.get_peak_memory(),
        events=events,
    )
