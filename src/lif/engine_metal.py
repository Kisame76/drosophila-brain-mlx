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
# through it. One thread per neuron (K = 1) collapses on load imbalance:
# out-degree runs from 0 to 9615 with a mean of 115, and the driven neurons are
# precisely the biggest hubs, so a single thread would serialise ~9600 atomics
# while 127,399 others idle.
#
# Measured 2026-09-14, M4 Pro, ms per propagate() call, 200 reps after warmup:
#
#   active  edges     K=1     K=2     K=4     K=8     K=16    K=32    K=64
#        2     159  0.1957  0.1771* 0.1982  0.1815  0.2228  0.3188  0.5313
#       21    1547  0.1442  0.1245* 0.1379  0.1608  0.2114  0.3117  0.5106
#      107  379545  0.8866  0.5755  0.4329  0.3612* 0.3640  0.3840  0.5619
#
# End-to-end over 2000 ticks, s per biological second, per kernel lane:
#
#   FlyWire + 21 sugar GRNs    metal  K=2 best (0.8627)   fused  K=1 best (0.3353)
#   MaleCNS + 100 hubs         metal  K=8 best (2.0118)   fused  K=8 best (1.3519)
#
# Two things that cost real time before they were measured. K=16 was this
# module's default and is optimal in none of the four cases; on the sugar drive
# it made the sparse lane twice as slow as it needed to be. And the two kernel
# lanes do not share an optimum on a sparse drive (1 vs 2), because the fused
# kernel dispatches different non-propagation work alongside it, so a benchmark
# that forces one value on both misreports one of them.
#
# There is still no runtime-choosable value: picking K from the live spike count
# needs a host readback, which is exactly the synchronisation the chunked design
# exists to avoid. So it stays a parameter, now with measured defaults.
EDGE_SPLIT = 8           # dense drive (hub stimulus); best for both lanes
EDGE_SPLIT_SPARSE = 2    # physiological drive, this lane (the fused lane wants 1)

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


def make_step(pack: core.Pack, c: dict, targets: mx.array, n_src, split: int = EDGE_SPLIT):
    def step(v, g, rfc, counts, rfc_reload, delayed, stim_row):
        rfc = mx.maximum(rfc - 1, 0)
        not_ref = rfc == 0
        v_upd = c["v0_term"] + (g * c["couple_g"] + v * c["decay_v"])
        g_upd = g * c["decay_g"]
        v = mx.where(not_ref, v_upd, v)
        g = mx.where(not_ref, g_upd, g)

        spike = mx.logical_and(not_ref, v > c["v_th"])

        contrib = propagate(delayed, pack, n_src, split)
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
        use_async: bool = True, warmup: int = 50,
        split: int = EDGE_SPLIT) -> RunResult:
    c = {k: mx.array(v) for k, v in core.constants_f32().items()}
    targets = mx.array(stim.targets)
    n_src = mx.array([pack.n_neurons], dtype=mx.uint32)
    step = make_step(pack, c, targets, n_src, split)
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
