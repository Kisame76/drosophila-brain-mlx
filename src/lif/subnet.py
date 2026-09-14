"""Carve a small connected subnetwork out of the full pack.

Brian2 cannot run 127k neurons x 14.7M edges in reasonable time, so semantic
validation happens on a subnetwork. It must be *connected* -- a random sample of
1000 neurons out of 127400 would contain almost no edges (density is ~0.09%) and
would validate nothing but the leak term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lif import core


@dataclass
class SubNet:
    """A subnetwork in its own 0..n-1 index space."""

    n: int
    parent_idx: np.ndarray   # int32[n], index into the full pack
    pre: np.ndarray          # int32[e], local
    post: np.ndarray         # int32[e], local
    counts: np.ndarray       # int32[e]
    row_ptr: np.ndarray      # int32[n+1]
    destinations: np.ndarray # int32[e]
    signed_counts: np.ndarray# int32[e]
    seeds: np.ndarray        # int32[k], local indices of the driven neurons


def build(pack: core.Pack, n_target: int = 800, n_seeds: int = 4) -> SubNet:
    """Breadth-first from the highest out-degree hubs until n_target neurons.

    Deterministic: seeds are the top-out-degree neurons, and each frontier is
    expanded in ascending destination order.
    """
    rp = np.asarray(pack.row_ptr)
    dst = np.asarray(pack.destinations)
    cnt = np.asarray(pack.signed_counts)

    order = np.lexsort((np.arange(pack.n_neurons), -pack.out_degree))
    seeds = np.sort(order[:n_seeds]).astype(np.int64)

    chosen: list[int] = []
    seen = set()
    frontier = list(seeds)
    for s in frontier:
        seen.add(int(s)); chosen.append(int(s))
    while frontier and len(chosen) < n_target:
        nxt = []
        for src in frontier:
            targets = np.sort(dst[rp[src]:rp[src + 1]])
            for t in targets:
                t = int(t)
                if t not in seen:
                    seen.add(t); chosen.append(t); nxt.append(t)
                    if len(chosen) >= n_target:
                        break
            if len(chosen) >= n_target:
                break
        frontier = nxt

    parent = np.sort(np.array(chosen, dtype=np.int64))
    n = parent.size
    # Map parent index -> local index; -1 for everything outside.
    lut = np.full(pack.n_neurons, -1, dtype=np.int64)
    lut[parent] = np.arange(n)

    # Keep every edge whose BOTH endpoints are inside the subnetwork.
    pre_l, post_l, cnt_l = [], [], []
    for li, gi in enumerate(parent):
        lo, hi = rp[gi], rp[gi + 1]
        d, c = dst[lo:hi], cnt[lo:hi]
        m = lut[d] >= 0
        if m.any():
            pre_l.append(np.full(int(m.sum()), li, dtype=np.int32))
            post_l.append(lut[d[m]].astype(np.int32))
            cnt_l.append(c[m].astype(np.int32))
    pre = np.concatenate(pre_l) if pre_l else np.zeros(0, np.int32)
    post = np.concatenate(post_l) if post_l else np.zeros(0, np.int32)
    counts = np.concatenate(cnt_l) if cnt_l else np.zeros(0, np.int32)

    # Already source-major with ascending destinations, by construction.
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(np.bincount(pre, minlength=n), out=row_ptr[1:])

    return SubNet(
        n=n, parent_idx=parent.astype(np.int32),
        pre=pre, post=post, counts=counts,
        row_ptr=row_ptr, destinations=post.copy(), signed_counts=counts.copy(),
        seeds=lut[seeds].astype(np.int32),
    )


def as_pack(sub: SubNet) -> core.Pack:
    """Wrap the subnetwork in a Pack so the engines can run it unchanged."""
    import mlx.core as mx

    out_degree = np.diff(sub.row_ptr).astype(np.int32)
    edge_src = np.repeat(np.arange(sub.n, dtype=np.int32), out_degree)
    return core.Pack(
        n_neurons=sub.n, n_edges=int(sub.destinations.size),
        row_ptr=mx.array(sub.row_ptr), destinations=mx.array(sub.destinations),
        signed_counts=mx.array(sub.signed_counts), edge_src=mx.array(edge_src),
        neuron_ids=np.zeros(sub.n, dtype=np.int64), out_degree=out_degree,
        manifest={"synthetic": "subnet"},
    )
