# Draft: follow-up in the MLX discussion

**Status: draft. Nothing here has been posted.** A human posts it or does not.

**Where:** the existing thread, not a new one —
[ml-explore/mlx#4512](https://github.com/ml-explore/mlx/discussions/4512), which
already presents [docs/mlx-notes.md](../mlx-notes.md) as a case study. Opening a
second thread on the same material would be noise.

**What is new since that post:** the four lanes were re-measured on 2026-09-15 in
one session, and a second dataset and a one-command demo now exist. Nothing below
contradicts the earlier post; it adds measurements.

**What to check before posting:** that the thread is still open, and that the
numbers below still match the README, which is the source for all of them.

---

## Draft text

Follow-up to the case study above, with the same four lanes re-measured today on
the same M4 Pro, 127,400 FlyWire v630 neurons and 14,687,178 directed
connections, 10,000 ticks at dt = 0.1 ms, three repeats, parity-gated against the
naive baseline:

| lane | s per biological second |
|---|---|
| naive dense, one `eval` per tick | 21.9046 (±0.0936) |
| chunked dense, `async_eval` | 19.5416 (±0.0360) |
| sparse CSR via `mx.fast.metal_kernel` | 0.7230 (±0.0047) |
| fused, two Metal dispatches per tick | 0.2937 (±0.0004) |

All four produce a bit-identical per-neuron spike-count SHA-256, which is the
gate that makes the comparison mean anything: a lane that loses parity is
reported as FAIL rather than credited with a time.

The three findings that generalise beyond this model are unchanged, and the
second one is the one I got wrong at first:

1. **Dense is the problem, not synchronisation.** Every tick touched all 14.7 M
   edges however few neurons fired, and dense runtime is flat across a 5,000×
   range in activity. Pure MLX cannot avoid this — compacting a spike list has a
   data-dependent output size, which the lazy graph cannot express — so
   `mx.fast.metal_kernel` is the way out, and it works by letting threads exit
   early, not by removing a sync.
2. **`eval` is the unit of overhead, not the dispatch.** Sixteen kernel
   dispatches inside one `mx.eval` cost what one costs: 0.1117 ms against
   0.1121 ms. I published this as "dispatch overhead" until I measured it. What
   fusing actually saves is memory traffic — a chain of ~16 elementwise ops
   writes every intermediate to memory, 15.6 MiB per tick at 114 GiB/s, while
   the values themselves would fit in registers. Fused, the same tick moves
   1.0 MiB.
3. **`mx.compile` breaks bit-reproducibility.** It fuses elementwise ops into a
   kernel where Metal contracts `a*b + c` into an FMA — mathematically equal,
   differently rounded. With a strict threshold that eventually flips a spike:
   invisible for 4,000 ticks, then 54,270 against 52,472 spikes at 10,000. MLX
   exposes no switch to disable contraction, so `mx.compile` is unusable for a
   model with a strict threshold decision if you need reproducibility. Inside a
   hand-written kernel it is controllable: `#pragma clang fp contract(off)`.

Also still true, and still the sharpest edge for anyone doing graph work in MLX:
there is no accumulating scatter. MLX 0.32.2 has no `bincount`, `segment_sum`,
`index_add` or public `scatter_add`, and `arr[idx] = vals` is last-write-wins.
`arr.at[idx].add()` and `mx.fast.metal_kernel` are the only options.

Repository, with the measurements and the parity gates:
https://github.com/Kisame76/drosophila-brain-mlx

---

## Provenance of every number above

| number | where it comes from |
|---|---|
| 21.9046 / 19.5416 / 0.7230 / 0.2937 | `python -m lif.benchmark --repeat 3`, run 2026-09-15 |
| 0.1117 vs 0.1121 ms, 16 dispatches per `eval` | README, "How it works" |
| 15.6 MiB vs 1.0 MiB per tick, 114 GiB/s | README, "How it works" |
| 54,270 vs 52,472 spikes at 10,000 ticks | README, "Two MLX limitations worth knowing" |
| 127,400 neurons, 14,687,178 edges | pack manifest, README |

The README's own table reports 0.2933 for the fused lane, from a set measured on
2026-09-14 together with the flyBrain comparison. Today's 0.2937 is a separate
run; the two are not spliced together, and the README table is left as the
same-session set it says it is.
