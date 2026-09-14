# mlx-lif-engine

A leaky integrate-and-fire simulation of the *Drosophila* connectome, written in
Apple MLX and accelerated with a custom Metal kernel. It runs the published
Shiu et al. model — all 127,400 FlyWire v630 neurons and 14,687,178 directed
edges — at **0.29 seconds per biological second** on an M4 Pro, against
**62.6 s** for the reference Brian2 implementation on the same machine.

Same model, same parameters, same tick ordering. Only faster.

## What this is

The [published model](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1)
is a Brian2 program. Brian2 is a fine simulator, but a one-second run of the full
brain takes about a minute, which makes sweeps and interactive work painful.
This repository is the same model reimplemented so that a second of brain time
costs less than a second of wall clock.

Nothing here is trained or learned. The 14.7 million synapses come from FlyWire —
real fly brains, sectioned, imaged under an electron microscope, every connection
traced. The engine integrates membrane voltage through that fixed wiring.

## What this is not

- **Not a new model.** The equations, constants and connectivity are Shiu et al.'s.
- **Not a body simulation.** For an embodied fly with learned control, see
  [TuragaLab/flybody](https://github.com/TuragaLab/flybody) — a different project
  with a different goal.
- **Not portable.** Apple silicon only; it depends on Metal.

## Results

M4 Pro, 24 GB. FlyWire v630, 127,400 neurons, 14,687,178 edges, 10,000 ticks
(1 biological second at dt = 0.1 ms), 21 right sugar GRNs driven at 150 Hz.
Pack loading and one-time shader compilation excluded.

| Engine | s / biological s | Peak memory |
|---|---|---|
| **this, fused — 2 Metal dispatches** | **0.2947** (±0.0003, n=3) | 665 MB |
| [flyBrain](https://github.com/mehrantsi/flyBrain) Rust/Metal | 0.3648 (n=5) | 94 MB |
| this, sparse Metal kernel | 1.5163 | 275 MB |
| flyBrain MLX + metal_kernel | 3.241 | — |
| this, dense MLX (chunked) | 19.45 | 834 MB |
| this, dense MLX (eval per tick) | 22.11 | 324 MB |
| flyBrain MLX (dense scatter) | 40.62 | — |
| **Brian2 2.10.1 (the published model)** | **62.6** | — |

Reproduce the rows for this repository with:

```bash
python -m lif.benchmark --repeat 3
```

It checks every lane against the naive baseline before reporting a time, so a
lane that lost parity is marked FAIL rather than being credited with a number.
All four produce a bit-identical spike-count SHA-256.

Read the comparisons carefully:

- **vs. Brian2 (~213×)** is the number that matters. Same model, same machine.
  It is a lower bound: the Brian2 figure was measured over 200 ticks with the
  network barely active, and Brian2 slows down as activity rises while these
  lanes do not.
- **vs. flyBrain (~20 %)** is a narrow win over a small, young project, and it
  costs 7× the memory. Their engine also processes more spikes in this run
  because its refractory semantics differ (see below); at equal spike counts the
  margin is closer to 16 %.

## Install

Requires macOS on Apple silicon and Python ≥ 3.11.

```bash
uv venv --python 3.13 && uv pip install -e .
./tools/fetch_upstream.sh          # ~90 MB from philshiu/Drosophila_brain_model
python -m lif.compile_pack         # builds data/pack/v630, ~114 MB
```

`compile_pack` runs 40 validation checks and refuses to write anything if one
fails. `python -m lif.verify_pack` audits the result independently afterwards,
through a different code path so a bug cannot confirm itself.

```bash
pytest                      # parity and determinism gates
python -m lif.benchmark     # the table above
```

## Use

```python
from lif import core, engine_fused

pack = core.load_pack()
stim = core.make_stimulus(pack, n_ticks=10_000, seed=20260913)
result = engine_fused.run(pack, stim, chunk=32, edge_split=1)

print(result.total_spikes(), result.counts_sha256())
```

`edge_split` is the one knob that matters. It sets how many GPU threads share one
neuron's edge list, and the best value depends on how much of the network is
firing: **1** for physiological drive (a handful of spikes per tick), **16** for
dense drive. Wrong choice costs up to 4×. The engine cannot pick it at runtime
without reading the spike count back to the host, which is exactly the
synchronisation the design avoids.

## Correctness

Speed claims are worthless without a correctness gate, so there are two.

**Between lanes.** All four engines must produce identical per-neuron spike
counts from the same stimulus and seed, compared by SHA-256 over the full
127,400-element vector. They do, and final `v`/`g` match bitwise.

**Against Brian2.** `src/lif/validate_brian2.py` runs Brian2 2.10.1 and this
engine on a connected 800-neuron subnetwork with a fixed spike train, so no RNG
difference can contaminate the comparison. `src/lif/engine_ref64.py` is a float64
NumPy oracle with the same tick semantics — necessary because Metal has no
float64, so without it "wrong semantics" cannot be told apart from "float32
rounding".

| Configuration | Brian2 | ref64 (f64) | MLX (f32) |
|---|---|---|---|
| 4 seeds @ 150 Hz, 500 ticks | 38 | 38 ✅ | 38 ✅ |
| 20 seeds @ 400 Hz, 2,000 ticks | 2,973 | 2,973 ✅ | 2,974 |
| 40 seeds @ 800 Hz, 2,000 ticks | 9,035 | 9,035 ✅ | 9,035 ✅ |
| 60 seeds @ 1,200 Hz, 3,000 ticks | 24,700 | 24,700 ✅ | 24,700 ✅ |

The float64 oracle matches Brian2 exactly in every configuration. The float32
lane differs by one spike in 2,973 in one case — a neuron sitting exactly on the
threshold, unavoidable without float64 on the GPU.

## How it works

Each tick preserves Brian2 ordering: exact closed-form update of `v` and `g` for
non-refractory neurons, strict `v > -45 mV` threshold, propagation of spikes
delayed by 1.8 ms through source-major CSR, application of signed synaptic counts
and external input, then reset, refractory update and delay-ring store.

Synaptic arrivals accumulate as `int32` contact counts and the 0.275 mV factor is
applied afterwards. Integer addition is associative and cannot overflow here, so
the result does not depend on thread completion order — deterministic, unlike
float atomics.

Two Metal dispatches per tick: one for propagation (atomics), one for everything
else. Getting from 20 s to 0.29 s was three findings, in order of size:

1. **Dense is the problem, not synchronisation.** Every tick touched all 14.7 M
   edges regardless of how few neurons fired — dense runtime is flat across a
   5,000× range in activity. Pure MLX cannot avoid this: compacting the spike
   list has a data-dependent output size, which the lazy graph cannot express.
   `mx.fast.metal_kernel` is the only way out, and it works by letting each
   thread exit early rather than by removing a sync.
2. **Load imbalance.** Out-degree runs from 0 to 9,615 with a mean of 115, and
   the driven neurons are the biggest hubs. One thread per neuron serialises
   ~9,600 atomics while 127,399 idle.
3. **Dispatch overhead.** Once propagation is sparse, runtime is entirely fixed
   cost — ~16 kernel dispatches for the elementwise half. Fusing them into one
   brought 1.51 s → 0.29 s.

## Two MLX limitations worth knowing

**`mx.compile` breaks bit-reproducibility.** It fuses elementwise ops into a
kernel where Metal contracts `a*b + c` into a fused multiply-add — mathematically
equivalent, differently rounded. With a strict threshold that eventually flips a
spike: invisible for 4,000 ticks, then 54,270 vs 52,472 spikes at 10,000. MLX
offers no switch to disable contraction, so `mx.compile` is unusable for any
model with a strict threshold decision if you need reproducibility. Inside a
hand-written kernel it *is* controllable — `#pragma clang fp contract(off)`.

**There is no accumulating scatter.** MLX 0.32.2 has no `bincount`,
`segment_sum`, `index_add` or public `scatter_add`, and `arr[idx] = vals` is
last-write-wins, not accumulation. `arr.at[idx].add()` and `mx.fast.metal_kernel`
are the only options.

## Credit where it is due: flyBrain

[`mehrantsi/flyBrain`](https://github.com/mehrantsi/flyBrain) (MIT) is a
Rust/Metal engine for the same model. Two of the three findings above came from
reading it, and it is only fair to say so plainly:

- **It got to `mx.fast.metal_kernel` first.** Its MLX lane already propagates CSR
  through a hand-written Metal kernel with one thread per source neuron and an
  early exit — the same shape as the kernel here. Its README's remark about
  per-tick host synchronisation is what prompted this project in the first place.
- **Kernel fusion came straight from its README**, which describes fusing
  decay/threshold work with CSR propagation to remove a full-neuron dispatch per
  tick. Applying that idea took this engine from 1.51 s to 0.29 s — the single
  largest step here, and not my idea.
- Its benchmark setup (sugar-GRN stimulus, chunked steps, excluding pack load and
  shader compilation) is what made a fair comparison possible at all.

What this engine adds on top is the edge-split for load imbalance, which its
kernel does not do, and a stricter parity gate.

`tools/setup_flybrain_reference.sh` builds it locally so anyone can re-run the
comparison instead of taking the numbers on trust.

### One open discrepancy

Its README states that incoming conductance can accumulate while a neuron is
refractory, "matching the upstream Brian2 equations". As far as I can measure,
Brian2 2.10.1 with this model formulation does the opposite: `(unless
refractory)` shields the variable from *every* write, synaptic input included. A
spike arriving at a refractory neuron leaves its `g` at exactly `0.00000`, before
and after the refractory period ends — dropped, not queued.

This matters because it changes results: with a bit-identical stimulus, disabling
the gate here moves 13,354 spikes to 17,315, close to their 16,796. A residual of
819 spikes stays unexplained, so there is likely a second difference I have not
found, and I may simply be wrong about how their engine handles this — I have not
read their kernel closely enough to claim otherwise.

Reproduce the Brian2 side with `python -m lif.validate_brian2` and the two-neuron
case described in the source. Corrections welcome.

## Repository

```
src/lif/
  compile_pack.py      FlyWire CSV/parquet -> source-major CSR pack, 40 checks
  verify_pack.py       independent audit of a written pack
  core.py              constants, pack loader, stimulus, initial state
  engine_naive.py      eval() per tick — the deliberately slow baseline
  engine_chunked.py    N ticks per eval, async_eval
  engine_metal.py      sparse CSR propagation via mx.fast.metal_kernel
  engine_fused.py      whole tick in two Metal dispatches — the fast lane
  engine_ref64.py      float64 correctness oracle, not a performance lane
  subnet.py            connected subnetwork for Brian2 validation
  validate_brian2.py   Brian2 vs ref64 vs MLX
  benchmark.py         reproduces the results table
tests/                 parity and determinism gates
bench/results.json     measured numbers, written by the benchmark
tools/                 upstream fetch, reference engine build
```

### Things that did not work

Recorded so nobody repeats them:

- **Two-kernel spike compaction.** Writing the firing neurons into a list with an
  atomic counter, then dispatching over `MAXS·K` threads instead of `N·K`, is the
  obvious next optimisation. It is slower: 0.0572 ms against 0.0550 ms for the
  single kernel. The second dispatch costs more than the smaller grid saves.
- **Trusting `StateMonitor` timestamps.** Brian2 records with `when='start'`,
  so monitor row `t` holds the state at the *end of tick t-1*. Read as same-tick
  state it fakes a one-tick lag, which cost me two "fixes" to the refractory and
  delay constants before I caught it. Compare monitor row `t+1` against engine
  tick `t`. The correct values are the plain quotients: 22 and 18 ticks.

## License and attribution

MIT, see [LICENSE](LICENSE). The model, its parameters and the connectivity data
are Shiu et al.'s — see [ATTRIBUTION.md](ATTRIBUTION.md) for what belongs to whom
and how to cite FlyWire.
