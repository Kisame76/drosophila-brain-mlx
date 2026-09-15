"""Validate the MLX engine against Brian2 on a subnetwork.

The phase-3 gate only proves the MLX lanes agree with each other. If the tick
semantics are wrong, all of them are wrong identically and the gate stays green.
This is the check that can actually fail.

One configuration runs three times: in Brian2, in the float64 oracle
engine_ref64, and in the fused lane in float32. They are compared per neuron by
spike count and spike by spike as (neuron, time) pairs, with an engine's tick k
converted by spike_record.tick_to_seconds.

The stimulus is fed to every side as a fixed spike train rather than as
PoissonInput, so the comparison is not contaminated by different RNGs.

Exit status 0 when ref64 equals Brian2 exactly, counts and spike times, and the
fused lane leaves out and adds at most MLX_MOVED_LIMIT of Brian2's spikes each.
float32 cannot do better than that; tests/test_brian2.py asserts the same gate.

    python -m lif.validate_brian2 [ticks] [seeds] [rate_hz]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from lif import core, spike_record, subnet

SEED = 20260913

# In the four configurations of the README table, float32 rounding leaves out at
# most 23 and adds at most 24 of Brian2's 2,973 spikes (0.8 %). A timing error in
# the MLX lanes moves far more: one tick more of axonal delay leaves out 620 of
# the 2,795 spikes in the first 350 ticks of 60 seeds at 1,200 Hz.
MLX_MOVED_LIMIT = 0.05


@dataclass
class Comparison:
    """Per-neuron counts, and spikes as {(neuron, seconds)}, for each side."""

    n_neurons: int
    n_edges: int
    seeds: np.ndarray        # local indices of the driven neurons
    input_spikes: int
    brian2_counts: np.ndarray
    ref64_counts: np.ndarray
    mlx_counts: np.ndarray
    brian2_events: set
    ref64_events: set
    mlx_events: set
    brian2_v: np.ndarray
    brian2_g: np.ndarray
    mlx_v: np.ndarray
    mlx_g: np.ndarray

    def ref64_exact(self) -> bool:
        return (bool(np.array_equal(self.ref64_counts, self.brian2_counts))
                and self.ref64_events == self.brian2_events)

    def mlx_within_float32_rounding(self) -> bool:
        limit = MLX_MOVED_LIMIT * len(self.brian2_events)
        return (len(self.brian2_events - self.mlx_events) <= limit
                and len(self.mlx_events - self.brian2_events) <= limit)


def _spike_set(events: np.ndarray) -> set:
    seconds = spike_record.tick_to_seconds(events[:, 0])
    return set(zip(events[:, 1].tolist(), seconds.tolist()))


def run_brian2(sub, draws, n_ticks, dt_ms=core.DT):
    from brian2 import (
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        Synapses,
        defaultclock,
        ms,
        mV,
        prefs,
    )
    prefs.codegen.target = "numpy"
    defaultclock.dt = dt_ms * ms

    params = {
        "v_0": core.V_0 * mV, "v_rst": core.V_0 * mV, "v_th": core.V_TH * mV,
        "t_mbr": core.T_MBR * ms, "tau": core.TAU * ms,
    }
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """
    neu = NeuronGroup(sub.n, eqs, method="linear", threshold="v > v_th",
                      reset="v = v_rst; g = 0*mV", refractory="rfc",
                      namespace=params, name="neu")
    neu.v = core.V_0 * mV
    neu.g = 0 * mV
    neu.rfc = core.T_RFC * ms
    # model.py zeroes the refractory period for every driven neuron.
    neu.rfc[sub.seeds] = 0 * ms

    syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                   delay=core.T_DLY * ms, name="syn")
    syn.connect(i=sub.pre, j=sub.post)
    syn.w = sub.counts * core.W_SYN * mV

    # External drive as an explicit spike train, identical to the engines' side.
    tick_idx, seed_idx = np.nonzero(draws[:n_ticks])
    gen = SpikeGeneratorGroup(len(sub.seeds), seed_idx,
                              tick_idx * dt_ms * ms, name="gen")
    drive = Synapses(gen, neu, on_pre=f"v_post += {core.W_EXT}*mV", name="drive")
    drive.connect(i=np.arange(len(sub.seeds)), j=sub.seeds)

    mon = SpikeMonitor(neu)
    net = Network(neu, syn, gen, drive, mon)
    net.run(n_ticks * dt_ms * ms)

    neurons = np.asarray(mon.i)
    counts = np.bincount(neurons, minlength=sub.n).astype(np.int64)
    # t_ is the time in seconds without units, as the clock computed it.
    events = set(zip(neurons.tolist(), np.asarray(mon.t_).tolist()))
    return counts, events, np.asarray(neu.v / mV), np.asarray(neu.g / mV)


def run_ref64(sub, draws, n_ticks):
    from lif import engine_ref64

    counts, _, _, events = engine_ref64.run(sub.row_ptr, sub.destinations, sub.signed_counts,
                                            sub.n, sub.seeds, draws[:n_ticks], n_ticks,
                                            record=True)
    return counts.astype(np.int64), _spike_set(events)


def run_mlx(sub, draws, n_ticks, lane="fused"):
    import mlx.core as mx

    from lif import engine_chunked, engine_fused, engine_metal, engine_naive

    pack = subnet.as_pack(sub)
    stim = core.Stimulus(targets=sub.seeds.astype(np.int32),
                         draws=mx.array(draws[:n_ticks]),
                         n_ticks=n_ticks, rate_hz=float("nan"), seed=-1)
    mod = {"naive": engine_naive, "chunked": engine_chunked, "metal": engine_metal,
           "fused": engine_fused}[lane]
    r = mod.run(pack, stim, warmup=0, record=True)
    return r.spike_counts.astype(np.int64), _spike_set(r.events), r.v_final, r.g_final


def compare(n_ticks: int, n_seeds: int, rate_hz: float, seed: int = SEED,
            lane: str = "fused") -> Comparison:
    pack = core.load_pack()
    sub = subnet.build(pack, n_target=800, n_seeds=n_seeds)
    rng = np.random.default_rng(seed)
    draws = rng.random((n_ticks, len(sub.seeds))) < rate_hz * (core.DT / 1000.0)

    b_counts, b_events, b_v, b_g = run_brian2(sub, draws, n_ticks)
    r_counts, r_events = run_ref64(sub, draws, n_ticks)
    m_counts, m_events, m_v, m_g = run_mlx(sub, draws, n_ticks, lane)
    return Comparison(
        n_neurons=sub.n, n_edges=int(sub.destinations.size), seeds=sub.seeds,
        input_spikes=int(draws.sum()),
        brian2_counts=b_counts, ref64_counts=r_counts, mlx_counts=m_counts,
        brian2_events=b_events, ref64_events=r_events, mlx_events=m_events,
        brian2_v=b_v, brian2_g=b_g, mlx_v=m_v, mlx_g=m_g,
    )


def main() -> int:
    n_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rate_hz = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0

    c = compare(n_ticks, n_seeds, rate_hz)
    print(f"subnet {c.n_neurons} neurons, {c.n_edges} edges, {len(c.seeds)} driven, "
          f"{n_ticks} ticks ({n_ticks * core.DT:g} ms)")
    print(f"stimulus: {c.input_spikes} input spikes at {rate_hz:g} Hz\n")

    sides = (("brian2", c.brian2_counts, c.brian2_events),
             ("ref64 (f64)", c.ref64_counts, c.ref64_events),
             ("mlx fused (f32)", c.mlx_counts, c.mlx_events))
    for name, counts, _ in sides:
        print(f"{name:<16} total spikes {counts.sum():>7}  neurons fired {int((counts > 0).sum()):>4}")

    for name, counts, events in sides[1:]:
        same_counts = bool(np.array_equal(counts, c.brian2_counts))
        missing = len(c.brian2_events - events)
        extra = len(events - c.brian2_events)
        print(f"\n{name}")
        print(f"  per-neuron spike counts identical: {'yes' if same_counts else 'no'}")
        print(f"  spike times identical:             {'yes' if not (missing or extra) else 'no '}"
              f"  ({missing} of Brian2's {len(c.brian2_events)} spikes missing, {extra} extra)")
        if not same_counts:
            d = c.brian2_counts - counts
            bad = np.nonzero(d)[0]
            print(f"  {bad.size} neurons differ, total delta {int(np.abs(d).sum())}, "
                  f"max |delta| {int(np.abs(d).max())}")
            for i in bad[:10]:
                first = min((t for j, t in c.brian2_events if j == i), default=None)
                print(f"  neuron {i:>4}: brian2 {c.brian2_counts[i]:>4}  {name} {counts[i]:>4}"
                      + (f"   first brian2 spike @ {first * 1000:.1f} ms" if first is not None else ""))

    print(f"\nfinal state, brian2 vs mlx fused  max |dv| {np.abs(c.brian2_v - c.mlx_v).max():.4e} mV"
          f"   max |dg| {np.abs(c.brian2_g - c.mlx_g).max():.4e} mV")
    ok_ref64, ok_mlx = c.ref64_exact(), c.mlx_within_float32_rounding()
    print(f"\ngate  ref64 equals Brian2 exactly: {'PASS' if ok_ref64 else 'FAIL'}")
    print(f"gate  mlx fused leaves out and adds at most {MLX_MOVED_LIMIT:.0%} of Brian2's spikes: "
          f"{'PASS' if ok_mlx else 'FAIL'}")
    return 0 if ok_ref64 and ok_mlx else 1


if __name__ == "__main__":
    sys.exit(main())
