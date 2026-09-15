"""What the refractory gate on arriving conductance is worth, in spikes.

Brian2 declares `g` as `(unless refractory)`, which shields it from every write
while a neuron is refractory, synaptic input included: an arrival at a refractory
neuron is dropped, not queued. flyBrain's README states the opposite for its
engine. This measures the difference instead of arguing it, by running the dense
lane twice on the whole brain -- once as shipped, once with that one gate removed
-- and comparing both against flyBrain at the same seed.

    python tools/refractory_gate.py

The engine is never modified. engine_naive.tick is copied here with the single
line changed and patched in for the second run, so nothing in src/lif/ differs
between the two runs except that line.

The gate-on run is the control: it must reproduce the spike count in
bench/results.json. If it does not, this harness is wrong and its gate-off number
means nothing, so it says so and stops.
"""

from __future__ import annotations

import sys

import mlx.core as mx

from lif import core, engine_naive, stimulus_flybrain

TICKS, SEED, RATE = 10_000, 20260816, 150.0
BENCHMARK_SPIKES = 13594     # bench/results.json, same pack, seed, ticks and drive
FLYBRAIN_SPIKES = 16796      # flybrain-rs simulate at this seed, 2026-09-15, n=5


def tick_without_the_gate(state, pack, signed_counts, c, stim_row, targets, slot):
    """engine_naive.tick with the synaptic refractory gate removed.

    Transcribed from it unchanged apart from the line marked below, so the two
    runs differ in that and nothing else.
    """
    v, g, rfc, ring = state["v"], state["g"], state["rfc"], state["ring"]

    rfc = mx.maximum(rfc - 1, 0)
    not_ref = rfc == 0
    v_upd = c["v0_term"] + (g * c["couple_g"] + v * c["decay_v"])
    g_upd = g * c["decay_g"]
    v = mx.where(not_ref, v_upd, v)
    g = mx.where(not_ref, g_upd, g)

    spike = mx.logical_and(not_ref, v > c["v_th"])
    delayed = ring[slot]

    active = delayed[pack.edge_src]
    vals = mx.where(active, signed_counts, mx.array(0, dtype=mx.int32))
    contrib = mx.zeros((pack.n_neurons,), dtype=mx.int32).at[pack.destinations].add(vals)

    # THE ONE CHANGE: arrivals are no longer dropped at a refractory neuron.
    g = g + contrib.astype(mx.float32) * c["w_syn"]

    # Left gated, as shipped. Driven neurons are never refractory, so this is a
    # no-op here, and changing two things at once would measure neither.
    v = v.at[targets].add(
        mx.where(not_ref[targets], stim_row.astype(mx.float32) * c["w_ext"], 0.0))

    v = mx.where(spike, c["v_0"], v)
    g = mx.where(spike, mx.array(0.0, dtype=mx.float32), g)
    rfc = mx.where(spike, state["rfc_reload"], rfc)
    ring[slot] = spike

    return {
        "v": v, "g": g, "rfc": rfc, "ring": ring,
        "rfc_reload": state["rfc_reload"],
        "counts": state["counts"] + spike.astype(mx.int32),
        "spike": spike,
    }


def main() -> int:
    pack = core.load_pack()
    targets = stimulus_flybrain.targets(pack.neuron_ids)
    draws = stimulus_flybrain.bernoulli(len(targets), TICKS, RATE, core.DT, SEED)
    stim = core.Stimulus(targets=targets, draws=mx.array(draws), n_ticks=TICKS,
                         rate_hz=RATE, seed=SEED)
    print(f"{pack.n_neurons} neurons, {pack.n_edges} edges, {TICKS} ticks, "
          f"{len(targets)} sugar GRNs at {RATE:g} Hz, seed {SEED}, dense lane")

    on = engine_naive.run(pack, stim, warmup=50).total_spikes()
    print(f"\ngate on, as shipped : {on} spikes")
    if on != BENCHMARK_SPIKES:
        print(f"  expected {BENCHMARK_SPIKES} from bench/results.json; this harness does "
              f"not reproduce the shipped engine, so its gate-off number means nothing",
              file=sys.stderr)
        return 1
    print(f"  matches bench/results.json ({BENCHMARK_SPIKES}), so the control holds")

    engine_naive.tick = tick_without_the_gate
    off = engine_naive.run(pack, stim, warmup=50).total_spikes()
    closed = (off - on) / (FLYBRAIN_SPIKES - on)
    print(f"\ngate off, experiment: {off} spikes")
    print(f"  flyBrain, same seed : {FLYBRAIN_SPIKES}")
    print(f"  gap closed          : {100 * closed:.0f} % "
          f"({FLYBRAIN_SPIKES - on} spikes -> {FLYBRAIN_SPIKES - off})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
