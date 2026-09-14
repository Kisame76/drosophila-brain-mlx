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
(1 biological second at dt = 0.1 ms), 21 right sugar GRNs driven at 150 Hz,
`edge_split` 1 for the fused lane and 2 for the sparse Metal lane (see [Use](#use)).
Pack loading and one-time shader compilation excluded.

All rows marked *same session* were measured on 2026-09-14 within minutes of
each other, including the flyBrain comparison. That matters more than it sounds;
see "How much to trust these".

| Engine | s / biological s | Peak memory | |
|---|---|---|---|
| **this, fused — 2 Metal dispatches** | **0.2933** (±0.0001, n=3) | 665 MB | same session |
| [flyBrain](https://github.com/mehrantsi/flyBrain) Rust/Metal | 0.3769 (±0.0107, n=5) | 94 MB | same session |
| this, sparse Metal kernel | 0.7491 (±0.0182, n=3) | 265 MB | same session |
| this, dense MLX (chunked) | 19.55 (±0.518, n=3) | 834 MB | same session |
| this, dense MLX (eval per tick) | 22.02 (±0.356, n=3) | 326 MB | same session |
| flyBrain MLX + metal_kernel | 3.241 | — | earlier session |
| flyBrain MLX (dense scatter) | 40.62 | — | earlier session |
| **Brian2 2.10.1 (the published model)** | **62.6** | — | earlier session |

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
- **vs. flyBrain (~22 %)** is a narrow win over a small, young project, and it
  costs 7× the memory. Their engine also processes more spikes in this run
  because its refractory semantics differ (see below); at equal spike counts the
  margin is smaller, and under CPU load it falls to ~10 % (below).

### How much to trust these

Two things were found on 2026-09-14 while re-measuring. Both are recorded
because a benchmark nobody can falsify is not a benchmark.

**The sparse Metal lane was mistuned, and the table overstated what fusion
bought.** That lane ran at `EDGE_SPLIT = 16`, this module's old default, which
is fastest in none of the six lane and drive combinations swept under [Use](#use).
At the correct `K = 2` it is 2.0× faster than previously
published: 1.5163 → 0.7491. The fused lane was already at its optimum and did
not move. But the fusion step was being credited against that bad baseline, so
its gain drops from a published 5.2× to a measured **2.55×**. The fastest
number in this repository is unchanged; the story of how it was reached is not.

**Wall-clock here is partly host-bound, and flyBrain's is not.** The same
measurement on a machine under CPU load (load average 2.8, an Electron app at
~57 % of a core) moves the two engines very differently:

| | idle-ish | under load | |
|---|---|---|---|
| this, fused | 0.2933 | 0.3341 | +13.9 % |
| flyBrain Rust/Metal | 0.3769 | 0.3726 | −1.1 %, i.e. noise |

The GPU work is identical in both cases; the spike-count SHA-256 never changes.
The difference is that this engine rebuilds its MLX graph on the host every
chunk, and that host work competes for CPU, while a native Rust loop does not.

This lane is faster than flyBrain in **both** states measured, but by very
different amounts: **22 % on a quiet machine, 10 % under the load above.** Two
points is not a curve, so no claim is made about where it ends up on a busier
host than was tested. If you benchmark this yourself and get a worse number than
the table, check your CPU load before suspecting your build.

## Datasets

Two connectome packs are supported. They are different animals from different
laboratories; **spike counts are not comparable across them**, and the Shiu et
al. constants were fitted to FlyWire only.

| | FlyWire v630 | MaleCNS v1.0 |
|---|---|---|
| specimen | female, brain only | male, brain **and** ventral nerve cord |
| neurons | 127,400 | 166,700 |
| edges | 14,687,178 | 24,469,412 |
| leg / wing motor neurons | absent | 708 VNC + 107 brain |
| licence | non-commercial | CC BY 4.0 |

MaleCNS, 2,000 ticks, 100 top-out-degree hubs at 150 Hz, `K = 8` for both kernel
lanes, all four parity-gated against the naive baseline. Measured 2026-09-14,
best of 3 runs, load average 1.79 before and 1.33 after; written to
`bench/results_malecns.json` by the MaleCNS command under [Install](#install):

| Engine | s / biological s | Peak memory |
|---|---|---|
| **this, fused — 2 Metal dispatches** | **1.1676** (±0.0010, n=3) | 434 MB |
| this, sparse Metal kernel | 1.7055 (±0.0013, n=3) | 412 MB |
| this, dense MLX (chunked) | 31.86 (±0.015, n=3) | 1204 MB |
| this, dense MLX (eval per tick) | 34.21 (±0.024, n=3) | 434 MB |

The surviving edge count, 24,469,412, is the same figure flyBrain publishes for
the same published materialization, reached through an independently written
filter. That agreement is the strongest evidence available that the selection
rules are right.

## Install

Requires macOS on Apple silicon and Python ≥ 3.11.

```bash
uv venv --python 3.13 && uv pip install -e .
./tools/fetch_upstream.sh          # ~90 MB from philshiu/Drosophila_brain_model
python -m lif.compile_pack         # builds data/pack/v630, ~114 MB
```

A second dataset is supported: MaleCNS v1.0, a male specimen covering brain
**and** ventral nerve cord, 166,700 neurons, so it contains the leg and wing
motor neurons the FlyWire brain-only pack does not. Optional and much larger to
fetch:

```bash
./tools/fetch_male_cns.sh          # ~1.06 GB, CC BY 4.0
python -m lif.compile_pack_malecns # builds data/pack/male_cns_v1
python -m lif.benchmark --quick --repeat 3 --pack data/pack/male_cns_v1 --out bench/results_malecns.json
```

Spike counts from the two packs are not comparable: different specimen,
different laboratory, and the Shiu et al. constants were fitted to FlyWire.

Both compilers run their validation checks and refuse to write anything if one
fails. `python -m lif.verify_pack [--pack <dir>]` audits the result
independently afterwards, through a different code path so a bug cannot confirm
itself; it re-derives the MaleCNS node selection and transmitter signs from the
raw tables rather than importing them.

```bash
pytest                      # parity and determinism gates
python -m lif.benchmark     # the table above
```

## Use

```python
from lif import core, engine_fused

pack = core.load_pack()
# the hub drive: the 100 highest out-degree neurons at 150 Hz
stim = core.make_stimulus(pack, n_ticks=10_000, seed=20260913)
result = engine_fused.run(pack, stim, chunk=32, edge_split=8)

print(result.total_spikes(), result.counts_sha256())
```

`edge_split` is the one knob that matters. It sets how many GPU threads share one
neuron's edge list. There are that many threads for every neuron in every tick,
and nearly all of them exit at once because their neuron did not fire, so a
larger value helps when the neurons that do fire carry long edge lists and costs
when they do not. The best value follows the drive, and on a sparse drive also
the lane:

- **Sparse drive**, such as the 21 sugar GRNs (FlyWire: 1.36 spikes per tick in
  the whole network, with 209 outgoing edges): **1** for the fused lane, **2**
  for the sparse Metal lane, where 1 measured within 0.1 %.
- **Hub drive**, which is what `make_stimulus` produces (FlyWire: 3.93 spikes
  and 6,534 outgoing edges per tick; MaleCNS: 3.69 and 7,346): **8** for both
  lanes, the default.

Measured end to end on 2026-09-14, M4 Pro, one process per pack with the runs
interleaved, median of 7, load average 2.47 to 3.05. Bold is the fastest value
in s per biological second, every other cell the time relative to it:

| pack, drive, ticks | lane | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|---|
| FlyWire, 21 sugar GRNs, 10,000 | fused | **0.301** | 1.16× | 1.52× | 2.29× | 3.80× |
| | sparse Metal | 1.00× | **0.746** | 1.12× | 1.42× | 2.06× |
| FlyWire, 100 hubs, 2,000 | fused | 2.23× | 1.42× | 1.07× | **1.140** | 1.21× |
| | sparse Metal | 1.89× | 1.31× | 1.06× | **1.623** | 1.14× |
| MaleCNS, 100 hubs, 2,000 | fused | 2.52× | 1.58× | 1.14× | **1.168** | 1.32× |
| | sparse Metal | 2.09× | 1.45× | 1.12× | **1.710** | 1.18× |

The other drive's value costs 2.23× to 2.52× in the fused lane and 1.31× to
1.45× in the sparse Metal lane; 16 is fastest in no row. The engine cannot pick
the value at runtime without reading the spike count back to the host, which is
exactly the synchronisation the design avoids.

To silence neurons, pass a boolean mask over the pack's neuron indices:

```python
import numpy as np

index = {int(n): i for i, n in enumerate(pack.neuron_ids)}
silenced = np.zeros(pack.n_neurons, dtype=bool)
silenced[index[720575940624963786]] = True
result = engine_fused.run(pack, stim, silenced=silenced, chunk=32, edge_split=8)
```

Silencing sets every synapse *from* a neuron to zero weight, which is what
`silence` in the published model's code does. That repository's README says
"to and from"; its code produced the published results, and this follows the
code. A silenced neuron still integrates, spikes and counts, and may also be
driven.

Measured with 10 silenced neurons that never fire, against the same run without
a mask, the fused lane moved −1.16 % on FlyWire with the sugar drive and +0.01 %
on MaleCNS with the hub drive. The mask empties those neurons' edge ranges for
the run rather than being tested in the kernel's early exit, which cost +36 % on
MaleCNS; details in
[docs/design/2026-09-14-experiment-layer.md](docs/design/2026-09-14-experiment-layer.md).

## Correctness

Speed claims are worthless without a correctness gate, so there are two.

**Between lanes.** All four engines must produce identical per-neuron spike
counts from the same stimulus and seed, compared by SHA-256 over the full
127,400-element vector. They do, and final `v`/`g` match bitwise. The same holds
with a silencing mask, which the kernel lanes implement as empty edge ranges and
the dense lanes as zeroed edge counts, so that gate compares two unrelated
mechanisms.

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
else. Getting from 22 s to 0.29 s was three findings, in order of size:

1. **Dense is the problem, not synchronisation.** Every tick touched all 14.7 M
   edges regardless of how few neurons fired — dense runtime is flat across a
   5,000× range in activity. Pure MLX cannot avoid this: compacting the spike
   list has a data-dependent output size, which the lazy graph cannot express.
   `mx.fast.metal_kernel` is the only way out, and it works by letting each
   thread exit early rather than by removing a sync. 19.6 s → 0.75 s, by far
   the largest step.
2. **Load imbalance.** Out-degree runs from 0 to 9,615 with a mean of 115, and
   the driven neurons are the biggest hubs. One thread per neuron serialises
   ~9,600 atomics while 127,399 idle.
3. **Intermediate materialisation.** Once propagation is sparse, the cost is the
   elementwise half: MLX writes every intermediate to memory, so a chain of ~16
   ops moves 15.6 MiB per tick through DRAM at 114 GiB/s while the values
   themselves fit in registers. Fusing the chain into one kernel moves 1.0 MiB
   and brought 0.75 s → 0.29 s.

   That gain was published as 5.2× (1.51 s → 0.29 s) until 2026-09-14. The
   1.51 s baseline was the sparse lane running at a mistuned `EDGE_SPLIT`;
   against a correctly tuned baseline the fusion is worth **2.55×**, not 5.2×.
   The isolated microbenchmark above still measures 5.9× because it fuses a pure
   16-op chain, whereas the engine's elementwise half is a smaller share of a
   tick that also does propagation.

   This was also written up as *dispatch overhead* until it was measured properly.
   It is not: 16 kernel dispatches inside one `mx.eval` cost what one costs
   (0.1117 ms vs 0.1121 ms). The ~0.11 ms floor is per `eval`, not per dispatch.
   A standalone microbenchmark of the same 16 ops gives 0.1333 ms/tick chained
   against 0.0225 ms fused, a 5.9x that matches the 5.2x seen in the engine.

## MLX notes

The transferable findings, with the measurements behind them, are in
[docs/mlx-notes.md](docs/mlx-notes.md): what to measure first, why fusing
elementwise chains is the largest single win, why `eval` and not the dispatch is
the unit of overhead, and the two limitations below in more detail.

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
  tick. Applying that idea took this engine from 0.75 s to 0.29 s, a 2.55×
  that is the second-largest step here, and not my idea.
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
  compile_pack.py          FlyWire CSV/parquet -> source-major CSR pack, 40 checks
  compile_pack_malecns.py  MaleCNS v1.0 tables -> the same pack format
  verify_pack.py           independent audit of a written pack
  core.py                  constants, pack loader, stimulus, initial state
  stimulus_flybrain.py     flyBrain's splitmix64 sugar-GRN stimulus
  engine_naive.py          eval() per tick — the deliberately slow baseline
  engine_chunked.py        N ticks per eval, async_eval
  engine_metal.py          sparse CSR propagation via mx.fast.metal_kernel
  engine_fused.py          whole tick in two Metal dispatches — the fast lane
  engine_ref64.py          float64 correctness oracle, not a performance lane
  subnet.py                connected subnetwork for Brian2 validation
  validate_brian2.py       Brian2 vs ref64 vs MLX
  benchmark.py             reproduces the results tables
tests/                     parity and determinism gates, benchmark and stimulus checks
bench/results*.json        measured numbers, written by the benchmark
tools/                     upstream fetch, reference engine build
```

### Things that did not work

Recorded so nobody repeats them:

- **Two-kernel spike compaction.** Writing the firing neurons into a list with an
  atomic counter, then dispatching over `MAXS·K` threads instead of `N·K`, is the
  obvious next optimisation. It is slower: 0.0572 ms against 0.0550 ms for the
  single kernel. The second dispatch costs more than the smaller grid saves.
- **Testing the silencing mask in the kernel's early exit.**
  `if (!spike[i] || silenced[i]) return;` is one extra read, but all `N·K`
  threads run it, including the ones that exit: +36 % of the fused tick on
  MaleCNS at `edge_split` 8, +74.5 % at 16, nothing at 1. A nested `if` or a
  ternary costs the same. Silencing now empties the neuron's edge range, which
  only threads whose neuron spiked read.
- **Trusting `StateMonitor` timestamps.** Brian2 records with `when='start'`,
  so monitor row `t` holds the state at the *end of tick t-1*. Read as same-tick
  state it fakes a one-tick lag, which cost me two "fixes" to the refractory and
  delay constants before I caught it. Compare monitor row `t+1` against engine
  tick `t`. The correct values are the plain quotients: 22 and 18 ticks.

## Roadmap

The engine is the simulation half of the published model. The experiment half
(activate a set of neurons, silence another, 30 trials, rates) is what comes
next, with the same `run_exp` signature and parquet output as the original so
its notebooks run unchanged. Order, conditions and what is deliberately left
out: [ROADMAP.md](ROADMAP.md).

## License and attribution

MIT, see [LICENSE](LICENSE). The model, its parameters and the connectivity data
are Shiu et al.'s — see [ATTRIBUTION.md](ATTRIBUTION.md) for what belongs to whom
and how to cite FlyWire.
