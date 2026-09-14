# Making MLX fast: notes from one real workload

Measured while building [this repository](../README.md), an irregular sparse
simulation: 127,400 nodes, 14.7 M directed edges, 10,000 sequential steps with a
data-dependent branch in each one. Every number below was measured on an Apple
M4 Pro (24 GB) with MLX 0.32.2 on macOS 26.6.2, and the scripts that produced
them are described where they are not part of the repository.

The workload is a neural simulation, but nothing here depends on that. If you
have a loop that steps a few large arrays many times and it is slower than you
expect, the diagnostics apply.

Headline: the same model went from 22.0 s to 0.29 s per simulated second. None
of that came from removing host synchronisation, which is where I started
looking.

## 1. Find out what you are actually bound by

Three things get blamed for slow MLX loops. They need different fixes, and
guessing wrong costs days. Each has a cheap test.

**Is it host synchronisation?** Stop calling `mx.eval()` every step. Queue N
steps, evaluate once, and use `mx.async_eval` so the host runs ahead of the GPU:

```python
t = 0
while t < n_steps:
    k = min(chunk, n_steps - t)
    state = queue_k_steps(state, t, k)   # no .item(), no np.asarray, no branch
    mx.async_eval(*state.values())       # do not block on the result
    t += k
mx.eval(*state.values())
```

Chunking cost me 12 % (22.1 → 19.4 s). If that is all you get, sync was not your
problem. The constraint is strict: anything inside the chunk that reads a value
back to the host collapses it, so control flow has to be expressed as
arithmetic.

**Is it fixed per-step overhead?** Scale the *work* without changing the number
of steps. I varied activity by 5,000x and runtime moved 2 %. Flat runtime means
you are paying a fixed cost per step regardless of the data, and no amount of
tuning the data path will help.

**Is it memory traffic?** Count the bytes your intermediates move. This turned
out to be the answer, and §2 is about it.

## 2. MLX writes every intermediate to memory

This is the single most useful thing I learned.

`a * b + c` over an N-element array is not one pass. MLX evaluates `a * b` into a
full N-element buffer in memory, then reads it back to add `c`. A chain of 16
elementwise ops makes 16 round trips through DRAM.

Measured, 127,400 float32 elements, 16 ops, 32 steps per `eval`:

| | ms per step | bytes moved per step |
|---|---|---|
| chain of 16 MLX elementwise ops | 0.1333 | 15.6 MiB |
| one `mx.fast.metal_kernel` doing the same arithmetic | 0.0225 | 1.0 MiB |

5.9x, and the chain's implied bandwidth is 114 GiB/s, which is roughly what this
machine sustains. **The chain is bandwidth-bound and the bandwidth is spent on
intermediates nobody reads.** The fused kernel keeps the value in a register.

In the real engine the same change was 0.75 s → 0.29 s, a 2.55x. The
microbenchmark's 5.9x is the ceiling, not the engine's number: it fuses a pure
16-op chain, while a real tick spends part of its time on propagation that
fusion does not touch. Expect a fraction of the isolated figure, in proportion
to how much of your step is actually the elementwise chain.

A warning attached to that 2.6x. It was 5.2x in an earlier version of this
document, because the unfused baseline was accidentally running at a bad tuning
constant. Fixing the baseline halved the apparent win. If you are measuring a
speedup against your own "before", make sure the before is tuned; otherwise you
are measuring your own earlier mistake.

I originally wrote this up as *dispatch overhead*, which was wrong. See §3.

**What to do.** If a step of your loop is a long chain of elementwise ops on the
same-shaped arrays, that chain is one kernel. Write it. The arithmetic is
usually a direct transcription and the win is large.

## 3. `eval()` is the synchronisation unit, not the dispatch

Chaining a trivial kernel and evaluating once:

| dispatches inside one eval | total ms |
|---|---|
| 1 | 0.1121 |
| 2 | 0.1147 |
| 4 | 0.1053 |
| 8 | 0.1076 |
| 16 | 0.1117 |

Sixteen dispatches cost what one costs. The ~0.11 ms is the round trip of
submitting a command buffer and waiting for it, paid **per `mx.eval`**, not per
kernel.

Two consequences:

- **Do not count dispatches.** Count bytes. Splitting work into more kernels
  inside one eval is close to free; what costs is what each kernel reads and
  writes.
- **`mx.eval` in a loop is expensive** at ~0.11 ms each. At 10,000 steps that is
  1.1 s of pure floor before any arithmetic. This is the real reason to chunk,
  and it is a much smaller effect than §2.

## 4. There is no accumulating scatter

In MLX 0.32.2 there is no `bincount`, no `segment_sum`, no `index_add`, no
public `scatter_add`, and no `mx.scatter`. `arr[idx] = vals` with repeated
indices is **last-write-wins**, which silently gives you a wrong answer rather
than an error.

What exists:

- `arr.at[idx].add(vals)` accumulates correctly. Use it when `idx` is small.
  In this workload, scattering ~100 values via a full `zeros(N)` buffer and a
  mask cost 20 % of the step; `.at[].add()` on the 100 indices removed that.
- `mx.fast.metal_kernel` with atomics, for anything large or irregular.

**The deeper limitation:** compaction has a data-dependent output size, and a
lazy graph cannot express one. You cannot write "the indices where `x` is true"
in MLX and get an array sized to the answer. Any algorithm whose shape depends
on values has to become a hand-written kernel, or be done densely with early
exits.

Densely with early exits is often fine. My propagation kernel dispatches all
127,400 threads every step and 127,300 of them return on their second
instruction. The *dispatch* stays dense and statically sized; only the memory
traffic becomes sparse. That was enough to go from 19.6 s to 0.75 s, by far the
largest single step in the project.

## 5. `mx.compile` can break bit-reproducibility

`mx.compile` fuses elementwise ops, which is exactly the fix in §2, and it gave
+4 % here. It is also unusable in this project, and the reason is worth knowing.

The fused kernel it generates lets Metal contract `a*b + c` into a fused
multiply-add. FMA is more accurate and mathematically equivalent, and it rounds
differently. With a strict comparison downstream (`if v > threshold`) that
difference eventually flips a decision. Here: identical for 4,000 steps, then
54,270 against 52,472 events at 10,000.

MLX exposes no switch to disable contraction. Inside a hand-written kernel you
control it:

```python
mx.fast.metal_kernel(..., header="\n#pragma clang fp contract(off)\n")
```

So: if your result feeds a strict comparison, an equality, a hash, or anything
where "close enough" is not enough, `mx.compile` is off the table and a
hand-written kernel is the way to get fusion. If you are producing floats a
human will look at, use it.

Related: **Metal has no float64.** Not slow, absent. If you need to distinguish
"my semantics are wrong" from "float32 rounded differently", you need a float64
reference on the CPU. I wrote a NumPy oracle for exactly this, and it earned its
keep: it proved a one-event difference was rounding and not a bug.

## 6. `mx.fast.metal_kernel`, and its traps

The API, as used here:

```python
kernel = mx.fast.metal_kernel(
    name="csr_propagate",
    input_names=["spike", "row_ptr", "dst", "cnt", "n_src"],
    output_names=["contrib"],
    source=body,               # the body only: no signature, no braces
    header="\n#pragma clang fp contract(off)\n",
    atomic_outputs=True,
)
out = kernel(
    inputs=[spike, row_ptr, dst, cnt, n_src],
    output_shapes=[(N,)], output_dtypes=[mx.int32],
    grid=(N * split, 1, 1), threadgroup=(256, 1, 1),
    init_value=0,
)[0]
```

Traps, in the order I hit them:

- **`source` is a body, not a function.** You get `thread_position_in_grid` and
  your named inputs as pointers. Do not write a signature.
- **`atomic_outputs=True` makes *every* output atomic.** There is no per-output
  setting. A kernel with one atomic accumulator and four ordinary outputs has to
  write the ordinary ones with `atomic_store_explicit` too. This produces a
  compile error that does not obviously point at the cause.
- **`grid` is threads, not threadgroups.** Unlike CUDA's block count.
- **Scalars must be arrays.** Pass `mx.array([float(x)], dtype=mx.float32)` and
  index `[0]` in the kernel. `mx.array([np.float32(x)])` raises; the `float()`
  is required.
- **Cache the kernel object.** Building it per call recompiles. Key a dict on
  whatever you template into the source.
- **Templating works.** I generate one kernel variant per unroll factor by
  string substitution, which lets a constant be a compile-time literal.

## 7. Integer atomics are deterministic; float atomics are not

Float addition is not associative, so `atomic_fetch_add` on floats gives you a
result that depends on thread completion order: different every run, and
different again on another machine.

Integer addition is associative and commutative. If your accumulation can be
expressed in integers without overflow, `atomic_fetch_add_explicit` on `int32`
is bit-reproducible regardless of scheduling.

Here the accumulation was a count of synaptic contacts, so it is naturally an
integer; the scale factor is applied once afterwards, in float, after the
non-deterministic part is over. **If your accumulation has a fixed-point
structure, find it.** It buys reproducibility for free.

If you need compaction with an atomic counter, the *slot order* is
non-deterministic but the *set* written is not. Sort on the host and the result
is reproducible again.

## 8. Skewed data needs a tunable unroll

One thread per row is the obvious mapping for a CSR-like structure and it
collapses when row lengths are skewed. Out-degree here runs 0 to 9,615 with a
mean of 115, and the rows that are actually touched are the longest ones. One
thread serialises 9,600 atomics while the rest of the GPU idles.

The fix is to let `K` threads share a row and stride through it:

```
uint i = gid / K;            // row
uint k = gid % K;            // lane within the row
if (!active[i]) return;
for (int e = row_ptr[i] + int(k); e < row_ptr[i+1]; e += K) { ... }
```

Measured, ms per propagation:

| active rows | edges touched | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | K=64 |
|---|---|---|---|---|---|---|---|---|
| 2 | 159 | 0.1957 | **0.1771** | 0.1982 | 0.1815 | 0.2228 | 0.3188 | 0.5313 |
| 21 | 1,547 | 0.1442 | **0.1245** | 0.1379 | 0.1608 | 0.2114 | 0.3117 | 0.5106 |
| 107 | 379,545 | 0.8866 | 0.5755 | 0.4329 | **0.3612** | 0.3640 | 0.3840 | 0.5619 |

Two things to take from it. **There is no single best K**: per propagation, 2
when the work is tiny and 8 to 16 when it is not, with K=1 at 107 active rows
taking 2.5x as long as K=8. End to end the spread is wider and 16 drops out:
swept over K = 1 to 16 in both kernel lanes on both packs, 16 was fastest in
none of six lane and drive combinations, and the slowest K took up to 3.80x as
long as the fastest (README, "Use"). And
**the engine cannot choose K at runtime** without reading the active count back
to the host, which is the synchronisation the whole design exists to avoid. So
it stays a parameter the caller sets, and that is an honest limitation rather
than a missing feature.

Note how flat the small-load rows are: at 2 and 21 active rows the cost barely
differs from K=1 to K=8, because the whole thing is floor (§3), not work.

**Keep the exit to the one read that decides it.** Whatever is tested before
`return` is paid by all `N * K` threads, including the ones that exit. A second
per-row flag in the exit, `if (!active[i] || masked[i]) return;`, measured with
no row active and the kernel dispatched on its own (127,400 rows): +0.9 % at
K=1, +41 % at K=2, +59 % at K=4, +73 % at K=8, against the same kernel without
the flag. In the engine at K=8 that was +29 % or +66 % of the whole step,
depending on the stimulus. A nested `if` or a ternary on the row's end cost the
same. Moving the flag into data read only after the exit, an end-of-row array in
which a masked row ends where it starts, cost at most +2.5 % in the same runs,
about what binding the flag without reading it cost (+2.0 %).

## 9. What did not work

- **Compacting the active list into a dense array with an atomic counter, then
  dispatching over the compacted list.** Two dispatches instead of one, with a
  much smaller grid for the second. Measured 0.0572 ms against 0.0550 ms for the
  single dense-grid kernel with early exits. The extra kernel plus its
  dependency cost more than the smaller grid saved. Dense-with-early-exit is
  hard to beat when the early exit is two instructions.
- **`mx.compile`.** §5.
- **Believing an explanation that was never measured.** I attributed the largest
  win in this project to dispatch count for weeks. It was memory traffic. The
  microbenchmark in §2 and §3 took twenty minutes and would have said so on day
  one.

## 10. Checklist

1. Chunk your steps and use `mx.async_eval`. If that alone fixes it, stop.
2. Vary the work by 100x without changing the step count. Flat runtime means
   fixed cost per step; tuning the data path will not help.
3. Count the bytes your intermediates move. A chain of k elementwise ops on an
   N-element array moves about `2kN * itemsize` per step, and almost all of it
   is avoidable.
4. Fuse the chain. `mx.compile` if approximate floats are fine, a hand-written
   kernel if they are not.
5. Anything with a data-dependent output shape needs a kernel, or needs to be
   done densely with an early exit. Try dense first; it is often enough.
6. Accumulate in integers if the structure allows it, and you get determinism
   for free.
7. Keep a CPU float64 reference. Without it you cannot tell a bug from rounding.
