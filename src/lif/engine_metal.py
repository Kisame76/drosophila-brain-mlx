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
# End to end, s per biological second at the best K, median of 7 runs interleaved
# in one process per pack, measured 2026-09-14; every K in README.md, "Use":
#
#                                ticks   metal         fused
#   FlyWire + 21 sugar GRNs     10,000   K=2 0.7457    K=1 0.3010
#   FlyWire + 100 hubs           2,000   K=8 1.6228    K=8 1.1400
#   MaleCNS + 100 hubs           2,000   K=8 1.7096    K=8 1.1680
#
# Two things that cost real time before they were measured. K=16 was this
# module's default and is fastest in none of the six; on the sugar drive it makes
# the sparse lane 2.06x as slow as it needs to be. And on the sugar drive the
# fused lane at the sparse lane's K=2 takes 1.16x its time at K=1, while the
# sparse lane at K=1 is within 0.1 % of K=2, so a benchmark that forces 2 on both
# misreports the fused lane.
#
# There is still no runtime-choosable value: picking K from the live spike count
# needs a host readback, which is exactly the synchronisation the chunked design
# exists to avoid. So it stays a parameter, now with measured defaults.
EDGE_SPLIT = 8           # dense drive (hub stimulus); best for both lanes
EDGE_SPLIT_SPARSE = 2    # physiological drive, this lane (the fused lane wants 1)

# Silencing empties a source's edge range. The kernel reads where each range ends
# from a per-run array: row_ptr[i + 1], except for a silenced source, whose range
# ends where it starts, so its loop never runs. Whether a source spikes is decided
# elsewhere, so a silenced neuron still fires and still counts (see
# core.silenced_mask). engine_naive and engine_chunked zero the same edges'
# counts instead. Without a mask the array is a view of row_ptr.
#
# Not in the early exit, where the first version tested it
# (`if (!spike[i] || silenced[i]) return;`): all N * K threads run that line, and
# the cost of the extra read grew with K. Measured 2026-09-14, fused lane,
# MaleCNS + 100 hubs, 10 silenced neurons that never fire, median of 7 runs
# interleaved in one process, against the kernel without a mask in that process:
#
#                 K=1       K=2       K=4       K=8        K=16
#   exit test     +0.08 %   +1.48 %   +5.54 %   +36.11 %   +74.54 %
#   range end     -0.01 %   +0.62 %   +0.36 %   +0.31 %    +0.20 %
#
# At K=8, a nested `if` or a ternary on the range end cost what `||` did. The
# range end is read after the exit, in place of row_ptr[i + 1], and only by
# threads whose source spiked. Details: docs/design/2026-09-14-experiment-layer.md,
# "Silencing".
_SRC = """
    uint gid = thread_position_in_grid.x;
    uint i = gid / EDGE_SPLIT;
    uint k = gid % EDGE_SPLIT;
    if (i >= n_src[0]) return;
    if (!spike[i]) return;
    int lo = row_ptr[i];
    int hi = row_end[i];
    for (int e = lo + int(k); e < hi; e += EDGE_SPLIT) {
        atomic_fetch_add_explicit(&contrib[dst[e]], cnt[e], memory_order_relaxed);
    }
"""

_kernels: dict = {}


def _kernel_for(split: int):
    if split not in _kernels:
        _kernels[split] = mx.fast.metal_kernel(
            name=f"csr_propagate_sparse_k{split}",
            input_names=["spike", "row_ptr", "row_end", "dst", "cnt", "n_src"],
            output_names=["contrib"],
            source=_SRC.replace("EDGE_SPLIT", str(split)),
            atomic_outputs=True,
        )
    return _kernels[split]


def silenced_row_end(pack: core.Pack, silenced: mx.array | None) -> mx.array:
    """End of each source's edge range for one run, as propagate() reads it.

    row_ptr[i + 1], except where silenced[i]: that range ends at row_ptr[i] and
    is empty. silenced is a mask from core.silenced_mask, or None, which gives a
    view of row_ptr rather than a copy.
    """
    if silenced is None:
        row_end = pack.row_ptr[1:]
    else:
        row_end = mx.where(silenced, pack.row_ptr[:-1], pack.row_ptr[1:])
    mx.eval(row_end)
    return row_end


def propagate(spike, pack: core.Pack, n_src, split: int = EDGE_SPLIT, *, row_end):
    """Scatter signed contact counts from spiking sources onto destinations.

    row_end comes from silenced_row_end(), once per run. Keyword-only, so it
    cannot be mistaken for split.
    """
    return _kernel_for(split)(
        inputs=[spike, pack.row_ptr, row_end, pack.destinations, pack.signed_counts, n_src],
        output_shapes=[(pack.n_neurons,)],
        output_dtypes=[mx.int32],
        grid=(pack.n_neurons * split, 1, 1),
        threadgroup=(256, 1, 1),
        init_value=0,
    )[0]


def make_step(pack: core.Pack, c: dict, targets: mx.array, n_src, row_end,
              split: int = EDGE_SPLIT):
    def step(v, g, rfc, counts, rfc_reload, delayed, stim_row):
        rfc = mx.maximum(rfc - 1, 0)
        not_ref = rfc == 0
        v_upd = c["v0_term"] + (g * c["couple_g"] + v * c["decay_v"])
        g_upd = g * c["decay_g"]
        v = mx.where(not_ref, v_upd, v)
        g = mx.where(not_ref, g_upd, g)

        spike = mx.logical_and(not_ref, v > c["v_th"])

        contrib = propagate(delayed, pack, n_src, split, row_end=row_end)
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


def run(pack: core.Pack, stim: core.Stimulus, silenced: np.ndarray | None = None,
        chunk: int = 64, use_async: bool = True, warmup: int = 50,
        split: int = EDGE_SPLIT) -> RunResult:
    row_end = silenced_row_end(pack, core.silenced_mask(pack, silenced))
    c = {k: mx.array(v) for k, v in core.constants_f32().items()}
    targets = mx.array(stim.targets)
    n_src = mx.array([pack.n_neurons], dtype=mx.uint32)
    step = make_step(pack, c, targets, n_src, row_end, split)
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
