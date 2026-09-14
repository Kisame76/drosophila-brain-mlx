"""Phase 5b: the whole tick in two Metal dispatches.

Phase 5 made the propagation sparse and moved the bottleneck: at 0.153 ms/tick
the cost no longer depends on activity at all (1.524 s/biol.s at 3,204 spikes vs
1.553 s at 51,338 -- a factor 16 in work, 2% in time). What remains is the elementwise
half, and its cost is memory traffic, not dispatch count: MLX materialises every
intermediate, so a chain of ~16 ops moves 15.6 MiB per tick through DRAM while
the values fit in registers. (16 dispatches inside one eval cost what one costs,
0.1117 vs 0.1121 ms; the ~0.11 ms floor is per eval. Measured separately: the
same 16 ops chained are 0.1333 ms/tick, fused 0.0225 ms/tick.)

flyBrain's README describes the fix for their Rust/Metal engine: fuse the
decay/threshold work with the CSR propagation so there is no full-neuron
dispatch per internal tick. This module does the same from MLX -- one kernel for
propagation (atomics), one for everything else (no atomics), two dispatches
total instead of ~16.

FMA contraction is disabled explicitly. `mx.compile` broke bit-parity in phase 2
precisely because Metal contracted `a*b + c` into a fused multiply-add, which is
mathematically equivalent but rounds differently, and a strict `v > v_th`
threshold turns that into a different spike train after enough ticks. The pragma
below is what keeps this lane bit-identical to the reference lanes.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from lif import core
from lif.engine_metal import EDGE_SPLIT, propagate, silenced_row_end
from lif.engine_naive import RunResult

_HEADER = """
#pragma clang fp contract(off)
"""

# One thread per neuron. Mirrors engine_naive.tick exactly, in the same order:
# refractory refresh, exact linear update, strict threshold, arrivals + external
# drive (both shielded while refractory), then reset / refractory reload.
_STATE_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= n_neurons[0]) return;

    int rfc_in = rfc[i];
    int rfc_dec = rfc_in - 1;
    if (rfc_dec < 0) rfc_dec = 0;
    bool not_ref = (rfc_dec == 0);

    float v_i = v[i];
    float g_i = g[i];

    if (not_ref) {
        float v_upd = c_v0_term[0] + (g_i * c_couple_g[0] + v_i * c_decay_v[0]);
        g_i = g_i * c_decay_g[0];
        v_i = v_upd;
    }

    bool sp = not_ref && (v_i > c_v_th[0]);

    if (not_ref) {
        g_i = g_i + float(contrib[i]) * c_w_syn[0];
        int slot = target_slot[i];
        if (slot >= 0 && stim_row[slot] != 0) {
            v_i = v_i + c_w_ext[0];
        }
    }

    if (sp) {
        v_i = c_v_0[0];
        g_i = 0.0f;
        rfc_dec = rfc_reload[i];
    }

    v_out[i] = v_i;
    g_out[i] = g_i;
    rfc_out[i] = rfc_dec;
    counts_out[i] = counts[i] + (sp ? 1 : 0);
    spike_out[i] = sp ? 1 : 0;
"""

_state_kernel = mx.fast.metal_kernel(
    name="lif_tick_state",
    input_names=["v", "g", "rfc", "counts", "contrib", "stim_row", "target_slot",
                 "rfc_reload", "n_neurons", "c_v0_term", "c_couple_g", "c_decay_v",
                 "c_decay_g", "c_v_th", "c_w_syn", "c_w_ext", "c_v_0"],
    output_names=["v_out", "g_out", "rfc_out", "counts_out", "spike_out"],
    source=_STATE_SRC,
    header=_HEADER,
    atomic_outputs=False,
)


def run(pack: core.Pack, stim: core.Stimulus, silenced: np.ndarray | None = None,
        chunk: int = 32, use_async: bool = True, warmup: int = 50,
        edge_split: int = EDGE_SPLIT) -> RunResult:
    # Read by the propagation kernel only, as empty edge ranges, so a silenced
    # neuron still spikes and counts here.
    row_end = silenced_row_end(pack, core.silenced_mask(pack, silenced))
    cf = core.constants_f32()
    k = {f"c_{name}": mx.array([float(cf[name])], dtype=mx.float32)
         for name in ("v0_term", "couple_g", "decay_v", "decay_g", "v_th",
                      "w_syn", "w_ext", "v_0")}
    N = pack.n_neurons
    n_neurons = mx.array([N], dtype=mx.uint32)
    n_src = mx.array([N], dtype=mx.uint32)

    # Dense neuron -> stimulus-slot map, so the kernel needs no gather logic.
    slot = np.full(N, -1, dtype=np.int32)
    slot[np.asarray(stim.targets)] = np.arange(stim.targets.size, dtype=np.int32)
    target_slot = mx.array(slot)

    draws_u8 = stim.draws.astype(mx.uint8)
    n = stim.n_ticks

    def fresh():
        st = core.initial_state(pack, stim)
        ring = [mx.zeros((N,), dtype=mx.uint8) for _ in range(core.DELAY_TICKS)]
        return st, ring

    def encode(st, ring, t0, kk):
        v, g, rfc, counts = st["v"], st["g"], st["rfc"], st["counts"]
        rl = st["rfc_reload"]
        for t in range(t0, t0 + kk):
            s = t % core.DELAY_TICKS
            contrib = propagate(ring[s], pack, n_src, edge_split, row_end=row_end)
            v, g, rfc, counts, spike = _state_kernel(
                inputs=[v, g, rfc, counts, contrib, draws_u8[t], target_slot,
                        rl, n_neurons, k["c_v0_term"], k["c_couple_g"],
                        k["c_decay_v"], k["c_decay_g"], k["c_v_th"],
                        k["c_w_syn"], k["c_w_ext"], k["c_v_0"]],
                output_shapes=[(N,), (N,), (N,), (N,), (N,)],
                output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.uint8],
                grid=(N, 1, 1),
                threadgroup=(256, 1, 1),
            )
            ring[s] = spike
        return {"v": v, "g": g, "rfc": rfc, "counts": counts, "rfc_reload": rl}, ring

    w = min(warmup, n)
    if w:
        st, ring = fresh()
        st, ring = encode(st, ring, 0, w)
        mx.eval(*st.values(), *ring)

    state, ring = fresh()
    mx.eval(*state.values(), *ring)

    mx.reset_peak_memory()
    start = time.perf_counter()
    t = 0
    while t < n:
        kk = min(chunk, n - t)
        state, ring = encode(state, ring, t, kk)
        if use_async:
            mx.async_eval(*state.values(), *ring)
        else:
            mx.eval(*state.values(), *ring)
        t += kk
    mx.eval(*state.values(), *ring)
    elapsed = time.perf_counter() - start

    return RunResult(
        spike_counts=np.asarray(state["counts"]), v_final=np.asarray(state["v"]),
        g_final=np.asarray(state["g"]), n_ticks=n, seconds=elapsed,
        peak_bytes=mx.get_peak_memory(),
    )
