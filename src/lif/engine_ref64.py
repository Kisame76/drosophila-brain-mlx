"""Float64 NumPy reference with exactly the MLX engines' tick semantics.

MLX's Metal backend has no float64, so the MLX lanes cannot be run in double
precision to separate "wrong semantics" from "float32 rounding". This module
does that separation: same tick order, same gating, same closed form, but in
float64 on the host. If this matches Brian2 exactly while the MLX lanes differ
by a handful of borderline spikes, the semantics are right and the residue is
precision. It is a correctness oracle, not a performance lane.
"""

from __future__ import annotations

import numpy as np

from lif import core


def run(sub_row_ptr, destinations, signed_counts, n, targets, draws, n_ticks,
        rfc_reload=None):
    A, B, C, V0T = core.DECAY_V, core.DECAY_G, core.COUPLE_G, core.V0_TERM

    edge_src = np.repeat(np.arange(n), np.diff(sub_row_ptr))
    dst = np.asarray(destinations, dtype=np.int64)
    cnt = np.asarray(signed_counts, dtype=np.int64)

    v = np.full(n, core.V_0, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    rfc = np.zeros(n, dtype=np.int64)
    if rfc_reload is None:
        rfc_reload = np.full(n, core.RFC_TICKS, dtype=np.int64)
        rfc_reload[targets] = 0
    ring = np.zeros((core.DELAY_TICKS, n), dtype=bool)
    counts = np.zeros(n, dtype=np.int64)

    for t in range(n_ticks):
        # 1. refractory refresh, exact linear update for non-refractory neurons
        rfc = np.maximum(rfc - 1, 0)
        not_ref = rfc == 0
        v_upd = V0T + (g * C + v * A)
        g_upd = g * B
        v = np.where(not_ref, v_upd, v)
        g = np.where(not_ref, g_upd, g)

        # 2. strict threshold, gated by not-refractory
        spike = not_ref & (v > core.V_TH)

        # 3. spikes emitted DELAY_TICKS ago
        slot = t % core.DELAY_TICKS
        delayed = ring[slot]

        # 4. signed contact counts, then external drive; both shielded while refractory
        contrib = np.zeros(n, dtype=np.int64)
        active = delayed[edge_src]
        if active.any():
            np.add.at(contrib, dst[active], cnt[active])
        g = g + np.where(not_ref, contrib * core.W_SYN, 0.0)
        ext = np.zeros(n, dtype=np.float64)
        np.add.at(ext, targets, draws[t].astype(np.float64) * core.W_EXT)
        v = v + np.where(not_ref, ext, 0.0)

        # 5. reset, refractory reload, ring store
        v = np.where(spike, core.V_0, v)
        g = np.where(spike, 0.0, g)
        rfc = np.where(spike, rfc_reload, rfc)
        ring[slot] = spike
        counts += spike

    return counts, v, g
