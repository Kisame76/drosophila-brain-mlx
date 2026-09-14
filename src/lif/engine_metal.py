"""Phase 5: sparse CSR propagation via mx.fast.metal_kernel.

Phases 1-4 established that the cost is not host synchronisation but the dense
sweep: every tick touches all 14,687,178 edges regardless of how few neurons
fired. Pure MLX cannot avoid that, because compacting the spike list has a
data-dependent output size and MLX has no way to express one in a lazy graph.

A hand-written Metal kernel can. One thread per neuron, each thread returning
immediately unless its neuron fired. Only the edges of neurons that actually
spiked are ever loaded. The dispatch stays dense (127,400 threads, statically
sized, no host readback), while the memory traffic becomes sparse.

Accumulation is int32 atomic_fetch_add. Integer addition is associative and
commutative and cannot overflow in this range, so the result is independent of
thread completion order -- deterministic, unlike float atomics.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from lif import core
from lif.engine_naive import RunResult

# EDGE_SPLIT threads cooperate on each source neuron's edge list, striding
# through it. One thread per neuron (EDGE_SPLIT = 1) collapses on load
# imbalance: out-degree runs from 0 to 9615 with a mean of 115, and the driven
# neurons are precisely the biggest hubs, so a single thread would serialise
# ~9600 atomics while 127,399 others idle. Measured on a realistic mask
# (107 spiking neurons, 188,263 active edges):
#     K=1: 0.406 ms   K=4: 0.130   K=16: 0.098   K=64: 0.217   K=256: 0.801
# K=16 wins; beyond it the cost of dispatching N*K threads dominates.
# How many threads cooperate on one source neuron's edge list. The right value
# depends on how much of the network is firing, and the spread is large:
#
#   load                      K=1     K=2     K=4     K=16    K=32
#   2 spikes / 184 edges     0.0297  0.0142  0.0255  0.0684  0.0988   ms
#   107 spikes / 188k edges  0.4058  0.1303* 0.0983  0.0983  0.8006   ms   (*K=4)
#
# High load needs the split to break up hub serialisation (out-degree reaches
# 9615); low load is dominated by dispatching N*K threads that immediately exit.
# There is no single best value, and the engine cannot pick one at runtime
# without reading the spike count back to the host -- which is exactly the
# synchronisation the chunked design exists to avoid. So it is a parameter.
EDGE_SPLIT = 16          # good for dense drive (hub stimulus)
EDGE_SPLIT_SPARSE = 2    # good for physiological drive (e.g. 21 sugar GRNs)

_SRC_TEMPLATE = """
    uint gid = thread_position_in_grid.x;
    uint i = gid / EDGE_SPLIT;
    uint k = gid % EDGE_SPLIT;
    if (i >= n_src[0]) return;
    if (!spike[i]) return;
    int lo = row_ptr[i];
    int hi = row_ptr[i + 1];
    for (int e = lo + int(k); e < hi; e += EDGE_SPLIT) {
        atomic_fetch_add_explicit(&contrib[dst[e]], cnt[e], memory_order_relaxed);
    }
"""

_kernels: dict = {}


def _kernel_for(split: int):
    if split not in _kernels:
        _kernels[split] = mx.fast.metal_kernel(
            name=f"csr_propagate_sparse_k{split}",
            input_names=["spike", "row_ptr", "dst", "cnt", "n_src"],
            output_names=["contrib"],
            source=_SRC_TEMPLATE.replace("EDGE_SPLIT", str(split)),
            atomic_outputs=True,
        )
    return _kernels[split]


def propagate(spike, pack: core.Pack, n_src, split: int = EDGE_SPLIT):
    """Scatter signed contact counts from spiking sources onto destinations."""
    return _kernel_for(split)(
        inputs=[spike, pack.row_ptr, pack.destinations, pack.signed_counts, n_src],
        output_shapes=[(pack.n_neurons,)],
        output_dtypes=[mx.int32],
        grid=(pack.n_neurons * split, 1, 1),
        threadgroup=(256, 1, 1),
        init_value=0,
    )[0]


def make_step(pack: core.Pack, c: dict, targets: mx.array, n_src):
    def step(v, g, rfc, counts, rfc_reload, delayed, stim_row):
        rfc = mx.maximum(rfc - 1, 0)
        not_ref = rfc == 0
        v_upd = c["v0_term"] + (g * c["couple_g"] + v * c["decay_v"])
        g_upd = g * c["decay_g"]
        v = mx.where(not_ref, v_upd, v)
        g = mx.where(not_ref, g_upd, g)

        spike = mx.logical_and(not_ref, v > c["v_th"])

        contrib = propagate(delayed, pack, n_src)
        g = g + mx.where(not_ref, contrib.astype(mx.float32) * c["w_syn"], 0.0)
        # Gate and scatter on the ~100 driven neurons only. Materialising a full
        # zeros(N) buffer and masking it cost 20% of the tick in the sparse lane
        # for 100 values. Targets are unique, so the scatter cannot collide and
        # the arithmetic is unchanged.
        v = v.at[targets].add(
            mx.where(not_ref[targets], stim_row.astype(mx.float32) * c["w_ext"], 0.0))

        v = mx.where(spike, c["v_0"], v)
        g = mx.where(spike, mx.array(0.0, dtype=mx.float32), g)
        rfc = mx.where(spike, rfc_reload, rfc)

        return v, g, rfc, counts + spike.astype(mx.int32), spike

    return step


def run(pack: core.Pack, stim: core.Stimulus, chunk: int = 64,
        use_async: bool = True, warmup: int = 50) -> RunResult:
    c = {k: mx.array(v) for k, v in core.constants_f32().items()}
    targets = mx.array(stim.targets)
    n_src = mx.array([pack.n_neurons], dtype=mx.uint32)
    step = make_step(pack, c, targets, n_src)
    n = stim.n_ticks
    N = pack.n_neurons

    def fresh():
        st = core.initial_state(pack, stim)
        ring = [mx.zeros((N,), dtype=mx.bool_) for _ in range(core.DELAY_TICKS)]
        return st, ring

    def encode(st, ring, t0, k):
        v, g, rfc, counts = st["v"], st["g"], st["rfc"], st["counts"]
        rl = st["rfc_reload"]
        for t in range(t0, t0 + k):
            s = t % core.DELAY_TICKS
            v, g, rfc, counts, spike = step(v, g, rfc, counts, rl, ring[s], stim.draws[t])
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
        k = min(chunk, n - t)
        state, ring = encode(state, ring, t, k)
        if use_async:
            mx.async_eval(*state.values(), *ring)
        else:
            mx.eval(*state.values(), *ring)
        t += k
    mx.eval(*state.values(), *ring)
    elapsed = time.perf_counter() - start

    return RunResult(
        spike_counts=np.asarray(state["counts"]), v_final=np.asarray(state["v"]),
        g_final=np.asarray(state["g"]), n_ticks=n, seconds=elapsed,
        peak_bytes=mx.get_peak_memory(),
    )
