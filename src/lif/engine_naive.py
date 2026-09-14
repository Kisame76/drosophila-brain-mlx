"""Phase 1: the deliberately slow lane.

One mx.eval() per tick. This is the shape of implementation that flyBrain
reports having abandoned, and it is the baseline every phase-2 claim is measured
against -- on this machine, not against someone else's M3 Max.

The tick body follows the Brian2 ordering exactly:
  1. refractory refresh, then the exact linear update of v and g (non-refractory only)
  2. strict threshold, v > V_TH, gated by not-refractory
  3. delayed spikes read out of the ring (emitted DELAY_TICKS ago)
  4. signed contact counts scattered onto destinations, then external drive on v
  5. reset, refractory reload, ring store
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from lif import core


@dataclass
class RunResult:
    spike_counts: np.ndarray   # int32[N]
    v_final: np.ndarray        # float32[N]
    g_final: np.ndarray        # float32[N]
    n_ticks: int
    seconds: float             # measured wall clock, warmup excluded
    peak_bytes: int

    def counts_sha256(self) -> str:
        import hashlib
        a = np.ascontiguousarray(self.spike_counts.astype(np.int32))
        return hashlib.sha256(a.tobytes()).hexdigest()

    def total_spikes(self) -> int:
        return int(self.spike_counts.sum())


def tick(state: dict, pack: core.Pack, signed_counts: mx.array, c: dict,
         stim_row: mx.array, targets: mx.array, slot: int) -> dict:
    """One dt. Pure function of state -> state; no host readback, no branching.

    signed_counts is pack.signed_counts, with the edges of silenced sources
    zeroed when run() was given a mask.
    """
    v, g, rfc, ring = state["v"], state["g"], state["rfc"], state["ring"]

    # --- 1. refractory refresh, then the exact closed-form update.
    # Brian2 prepends the not_refractory computation to the state updater, so the
    # counter is decremented before it gates this tick's integration.
    rfc = mx.maximum(rfc - 1, 0)
    not_ref = rfc == 0
    # The g term uses the PRE-decay g: that is what the analytic solution over
    # the interval gives, not a sequential substitution of the decayed value.
    # Association order transcribed from Brian2's generated code; see core.py.
    v_upd = c["v0_term"] + (g * c["couple_g"] + v * c["decay_v"])
    g_upd = g * c["decay_g"]
    v = mx.where(not_ref, v_upd, v)
    g = mx.where(not_ref, g_upd, g)

    # --- 2. strict threshold, refractory neurons cannot fire.
    spike = mx.logical_and(not_ref, v > c["v_th"])

    # --- 3. spikes emitted DELAY_TICKS ago. Read before overwrite: this slot
    # still holds tick (t - DELAY_TICKS) until step 5 replaces it.
    delayed = ring[slot]

    # --- 4. propagate. Dense over all E edges: MLX 0.32.2 has no segment_sum /
    # bincount / index_add, and arr[idx] = v is last-write-wins rather than
    # accumulating, so arr.at[idx].add() is the only vectorised accumulation
    # available. Counts stay int32 until the very end -- integer adds are
    # order-independent, so the scatter is deterministic even if it uses atomics.
    active = delayed[pack.edge_src]
    vals = mx.where(active, signed_counts, mx.array(0, dtype=mx.int32))
    contrib = mx.zeros((pack.n_neurons,), dtype=mx.int32).at[pack.destinations].add(vals)
    # Both v and g are declared "(unless refractory)", and in Brian2 that shields
    # them from EVERY write while refractory, synaptic input included -- not just
    # from the integration. Arrivals at a refractory neuron are dropped, not
    # queued. Gating on not_ref (computed in step 1, before this tick's reset)
    # reproduces that; without it this engine fires strictly more often.
    g = g + mx.where(not_ref, contrib.astype(mx.float32) * c["w_syn"], 0.0)
    # External Poisson drive lands directly on v, per PoissonInput(target_var='v'),
    # and is shielded the same way.
    # Gate and scatter on the ~100 driven neurons only. Materialising a full
    # zeros(N) buffer and masking it cost 20% of the tick in the sparse lane
    # for 100 values. Targets are unique, so the scatter cannot collide and
    # the arithmetic is unchanged.
    v = v.at[targets].add(
        mx.where(not_ref[targets], stim_row.astype(mx.float32) * c["w_ext"], 0.0))

    # --- 5. reset, refractory reload, ring store.
    v = mx.where(spike, c["v_0"], v)
    g = mx.where(spike, mx.array(0.0, dtype=mx.float32), g)
    rfc = mx.where(spike, state["rfc_reload"], rfc)
    ring[slot] = spike

    return {
        "v": v, "g": g, "rfc": rfc, "ring": ring,
        "rfc_reload": state["rfc_reload"],
        "counts": state["counts"] + spike.astype(mx.int32),
    }


def run(pack: core.Pack, stim: core.Stimulus, silenced: np.ndarray | None = None,
        warmup: int = 50) -> RunResult:
    # Silencing as data: every edge of a silenced source carries a zero count,
    # computed once per run. The kernel lanes implement the same semantics as an
    # early exit instead; that the two mechanisms agree is what the silencing
    # parity test checks. Without a mask the pack's counts are used directly, so
    # an unsilenced run allocates nothing extra.
    signed_counts = pack.signed_counts
    mask = core.silenced_mask(pack, silenced)
    if mask is not None:
        signed_counts = mx.where(mask[pack.edge_src], mx.array(0, dtype=mx.int32),
                                 pack.signed_counts)
        mx.eval(signed_counts)

    c = {k: mx.array(v) for k, v in core.constants_f32().items()}
    state = core.initial_state(pack, stim)
    targets = mx.array(stim.targets)
    n = stim.n_ticks

    # Warm up so kernel compilation and first-touch allocation land outside the
    # measured window. State is rebuilt afterwards, so warmup cannot leak in.
    for t in range(min(warmup, n)):
        state = tick(state, pack, signed_counts, c, stim.draws[t], targets, t % core.DELAY_TICKS)
        mx.eval(state["v"], state["g"], state["rfc"], state["ring"], state["counts"])
    state = core.initial_state(pack, stim)
    mx.eval(*state.values())

    mx.reset_peak_memory()
    start = time.perf_counter()
    for t in range(n):
        state = tick(state, pack, signed_counts, c, stim.draws[t], targets, t % core.DELAY_TICKS)
        # The defining property of this lane: a full host synchronisation here,
        # every single tick.
        mx.eval(state["v"], state["g"], state["rfc"], state["ring"], state["counts"])
    elapsed = time.perf_counter() - start

    return RunResult(
        spike_counts=np.asarray(state["counts"]),
        v_final=np.asarray(state["v"]),
        g_final=np.asarray(state["g"]),
        n_ticks=n,
        seconds=elapsed,
        peak_bytes=mx.get_peak_memory(),
    )
