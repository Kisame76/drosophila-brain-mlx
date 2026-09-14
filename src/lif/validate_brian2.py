"""Validate the MLX engine against Brian2 on a subnetwork.

The phase-3 gate only proves the two MLX lanes agree with each other. If the
tick semantics are wrong, both are wrong identically and the gate stays green.
This is the check that can actually fail.

The stimulus is fed to both sides as a fixed spike train rather than as
PoissonInput, so the comparison is not contaminated by two different RNGs.
"""

from __future__ import annotations

import sys

import numpy as np

from lif import core, subnet


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

    # External drive as an explicit spike train, identical to the MLX side.
    tick_idx, seed_idx = np.nonzero(draws[:n_ticks])
    gen = SpikeGeneratorGroup(len(sub.seeds), seed_idx,
                              tick_idx * dt_ms * ms, name="gen")
    drive = Synapses(gen, neu, on_pre=f"v_post += {core.W_EXT}*mV", name="drive")
    drive.connect(i=np.arange(len(sub.seeds)), j=sub.seeds)

    mon = SpikeMonitor(neu)
    net = Network(neu, syn, gen, drive, mon)
    net.run(n_ticks * dt_ms * ms)

    counts = np.zeros(sub.n, dtype=np.int64)
    idx, cts = np.unique(np.asarray(mon.i), return_counts=True)
    counts[idx] = cts
    times = {int(i): np.sort(np.asarray(t / ms)) for i, t in mon.spike_trains().items()}
    return counts, times, np.asarray(neu.v / mV), np.asarray(neu.g / mV)


def run_mlx(sub, draws, n_ticks, lane="naive"):
    import mlx.core as mx

    from lif import engine_chunked, engine_metal, engine_naive

    pack = subnet.as_pack(sub)
    stim = core.Stimulus(targets=sub.seeds.astype(np.int32),
                         draws=mx.array(draws[:n_ticks]),
                         n_ticks=n_ticks, rate_hz=float("nan"), seed=-1)
    mod = {"naive": engine_naive, "chunked": engine_chunked, "metal": engine_metal}[lane]
    r = mod.run(pack, stim, warmup=0)
    return r.spike_counts.astype(np.int64), r.v_final, r.g_final


def main() -> int:
    n_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rate_hz = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
    seed = 20260913

    pack = core.load_pack()
    sub = subnet.build(pack, n_target=800, n_seeds=n_seeds)
    print(f"subnet {sub.n} neurons, {sub.destinations.size} edges, "
          f"{len(sub.seeds)} driven, {n_ticks} ticks ({n_ticks*core.DT} ms)")

    rng = np.random.default_rng(seed)
    p = rate_hz * (core.DT / 1000.0)
    draws = rng.random((n_ticks, len(sub.seeds))) < p
    print(f"stimulus: {int(draws.sum())} input spikes at {rate_hz} Hz\n")

    b_counts, b_times, b_v, b_g = run_brian2(sub, draws, n_ticks)
    m_counts, m_v, m_g = run_mlx(sub, draws, n_ticks)

    print(f"brian2 total spikes {b_counts.sum():>7}  neurons fired {int((b_counts>0).sum()):>4}")
    print(f"mlx    total spikes {m_counts.sum():>7}  neurons fired {int((m_counts>0).sum()):>4}")
    same = bool((b_counts == m_counts).all())
    print(f"\nper-neuron spike counts identical: {'PASS' if same else 'FAIL'}")
    if not same:
        d = b_counts - m_counts
        bad = np.nonzero(d)[0]
        print(f"  {bad.size} neurons differ, total delta {int(np.abs(d).sum())}, "
              f"max |delta| {int(np.abs(d).max())}")
        for i in bad[:10]:
            bt = b_times.get(int(i), np.array([]))
            print(f"  neuron {i:>4}: brian2 {b_counts[i]:>4}  mlx {m_counts[i]:>4}"
                  + (f"   first brian2 spike @ {bt[0]:.1f} ms" if bt.size else ""))
    print(f"\nfinal state  max |dv| {np.abs(b_v-m_v).max():.4e} mV"
          f"   max |dg| {np.abs(b_g-m_g).max():.4e} mV")
    print(f"  brian2 v [{b_v.min():.2f}, {b_v.max():.2f}]  mlx v [{m_v.min():.2f}, {m_v.max():.2f}]")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
