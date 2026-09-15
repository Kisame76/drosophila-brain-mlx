# Experiment layer: `run_exp` compatible with the published model

Date: 2026-09-14. Status: silencing implemented ("Silencing", tests 5 and 6);
spike-event recording implemented and its overhead measured ("Spike-event
recording", tests 1 to 4 and 7); the two-rate stimulus implemented ("Stimulus
with two rates", test 10); `run_exp` and `rates` implemented (tests 8 and 9);
measured end to end, 10.05 s for 30 trials of 1 s; upstream's example notebook
run against the Brian2 files upstream published for it (test 11, 2026-09-15).
Implements phase 1 of [ROADMAP.md](../../ROADMAP.md), which is done.

## Goal

Make this engine a drop-in replacement for the simulation half of
`philshiu/Drosophila_brain_model`. A notebook that calls

```python
run_exp(exp_name='sugarR', neu_exc=neu_sugar, **config)
df = utl.load_exps(['./results/example/sugarR.parquet'])
df_rate, df_std = utl.get_rate(df, t_run=1, n_run=30, flyid2name=flyid2name)
```

against Brian2 today must produce the same kind of parquet from this engine,
with the upstream analysis functions consuming it unchanged.

## Non-goals

- Reproducing Brian2's random numbers. Trials are statistically equivalent, not
  RNG-identical. The Brian2 validation uses fixed spike trains for that reason.
- Per-run overrides of model constants (`v_th`, `tau`, ...). See "params".
- Dense raster output.

## Upstream contract, as read from the code

`data/ref/model.py` and `data/ref/utils.py`, fetched by `tools/fetch_upstream.sh`.

| Item | Upstream behaviour |
|---|---|
| `run_exp(exp_name, neu_exc, path_res, path_comp, path_con, params=default_params, neu_slnc=[], neu_exc2=[], n_proc=-1, force_overwrite=False)` | writes `{path_res}/{exp_name}.parquet`; skips with a message if it exists and `force_overwrite` is false |
| `default_params` | `t_run = 1000 ms`, `n_run = 30`, `r_poi = 150 Hz`, `r_poi2 = 0 Hz`, plus the model constants |
| activation | `PoissonInput` at `r_poi` on `neu_exc`, at `r_poi2` on `neu_exc2`; weight `w_syn * f_poi = 68.75 mV` |
| silencing | `syn.w['<i> == i'] = 0 mV` for every `i` in `neu_slnc`: **outgoing** synapses only. The upstream README says "to and from"; the code says from. This design follows the code, which produced the published results, and records the discrepancy. |
| trial | fresh network per trial (`create_model` inside `run_trial`), `net.run(t_run)`, `SpikeMonitor.spike_trains()` |
| parquet | one row per spike: `t` float64 seconds, `trial` int64, `flywire_id` int64, `exp_name` string |
| `get_rate` | `len(spikes in trial) / t_run`, mean and std over trials; the spike times themselves are never used |

## Architecture

```
src/lif/experiment.py     run_exp(), default_params, rates(); resolves IDs, builds
                          the stimulus, loops trials, writes parquet
src/lif/spike_record.py   one Metal kernel: [K, N] spike stack -> (tick, neuron)
                          event pairs; host-side sort and overflow check;
                          tick_to_seconds
engines (all four)        run(pack, stim, silenced=None, record=False)
core.py                   make_stimulus_for(): two target sets, two rates
engine_ref64.py           record=True support (numpy), for the Brian2 time test
```

`core.Pack` stays immutable and hash-verified. Silencing is a per-run mask, not
a pack edit.

## Spike-event recording

### Representation

An event is `(tick, neuron)`, both `int32`. `tick` is the 0-based index of the
tick in which `v > v_th` was true. Conversion to seconds happens in
`spike_record.tick_to_seconds` only, which `experiment.py` will use:

```python
def tick_to_seconds(tick: np.ndarray) -> np.ndarray:
    return np.asarray(tick).astype(np.float64) * (core.DT / 1000.0)
```

Correspondence to Brian2: engine tick `k` performs the same state update,
threshold and reset sequence that Brian2 performs at clock time `k * dt`, and
`SpikeMonitor` records the threshold time. So `t = k * dt` with no offset. This
was written as an expectation, because of the `StateMonitor` mistake recorded in
the README; test 7 has since confirmed it, and if it ever fails the offset lives
in this one function.

### Kernel

Runs **once per chunk**, not per tick. Every kernel lane already produces one
spike mask per tick (`bool[N]` in the chunked and sparse lanes, `uint8[N]` in
the fused lane) and keeps the chunk's masks alive until `async_eval`; the lane
stacks them and the kernel consumes the stack.

```
inputs   spikes[K, N] bool or uint8, cap[1] uint32
outputs  events[cap, 2] int32, count[1] uint32, both atomic, zero-initialised
grid     (N, K, 1)

uint i = thread_position_in_grid.x;
uint k = thread_position_in_grid.y;
uint n_ticks = uint(spikes_shape[0]);
uint n = uint(spikes_shape[1]);
if (i >= n || k >= n_ticks) return;
if (!spikes[k * n + i]) return;
uint slot = atomic_fetch_add_explicit(&count[0], 1u, memory_order_relaxed);
if (slot >= cap[0]) return;                      // overflow: detected on the host
atomic_store_explicit(&events[2 * slot], int(k), memory_order_relaxed);
atomic_store_explicit(&events[2 * slot + 1], int(i), memory_order_relaxed);
```

As implemented, three things differ from the first sketch here: the tick is
written relative to the chunk and the host adds the chunk's first tick, so no
per-chunk input array is built; tick and neuron share one output, so one buffer
is allocated and zero-filled instead of two; and `K` and `N` come from the
stack's shape rather than from inputs. `atomic_outputs=True` makes every output
atomic in `mx.fast.metal_kernel`, hence `atomic_store_explicit` (same pattern
`engine_metal.py` already uses). `#pragma clang fp contract(off)` is irrelevant
here; there is no float arithmetic.

Cost per chunk: stacking copies the chunk's `K * N` spike bytes, the dispatch
reads them, the event buffer is zero-filled, and the host reads back `count` and
the events used. The estimate that stood here counted only the read, and put it
at single-digit percent, down from a first 28 % that came from a per-dispatch
cost model measurement later disproved (docs/mlx-notes.md §3). The copy and the
fill were not in it, and on the sparse drive the result is not single-digit.
Measured 2026-09-14, fused lane, recording on and off interleaved in one process
per pack, median of 7 (table in README, "Use"): +13.7 % on FlyWire with the
sugar drive, +4.3 % with the hub drive, +3.8 % on MaleCNS with the hub drive.
The added time is 0.046 to 0.053 s per biological second in all three although
the hub drives fire 2.7 to 2.9 times as many spikes per tick, so it is a cost per
chunk, not per spike. Recording also lowers the FlyWire peak (669 → 380 MB on
the sugar drive): the lane waits for the previous chunk before encoding the
next, so less scheduled work is in flight.

### Host side

Reading a chunk's `count` waits for that chunk, so reading it right after
`async_eval` would turn the pipeline into a sync per chunk. The lanes instead
read back the chunk *before* the one just scheduled (`Recorder.drain(keep=1)`),
and the last one after the loop, so the host never waits on the chunk it has
just handed over, and at most two chunks' event buffers are alive at a time.
For each chunk read: if `count > cap`, raise `RecordOverflow(tick_range, count,
cap)`, which names the chunk's ticks and advises a larger `cap`; otherwise copy
`events[:count]` and add the chunk's first tick. At the end of the run,
`np.lexsort((neuron, tick))`. The atomic slot order is nondeterministic; the
sorted event set is not, and a test asserts it.

Default `cap = 262,144` events per chunk (2 MB), regardless of the lane's
chunk length. The densest run measured so far, 100 hub neurons at 150 Hz,
produced ~52,000 spikes per 10,000 ticks, i.e. ~166 per 32-tick chunk on
average; the cap is ~1,500x that. Burst peaks are not measured; the overflow
check exists for them. Warmup runs the kernel as well, so it compiles outside
the measured window; the warmup's events are never read.

### Other lanes

- `engine_metal.py` and `engine_chunked.py`: same kernel on their chunk stacks.
- `engine_naive.py`: `np.nonzero` on the host per tick. Deliberately a
  different code path, so the kernel is checked against something it does not
  share code with.
- `engine_ref64.py`: `np.nonzero` per tick, float64. Needed for the Brian2
  time comparison.

`RunResult` gains `events: np.ndarray | None` of shape `[E, 2]` (`tick`,
`neuron`), sorted. `spike_counts` stays and must equal
`np.bincount(events[:, 1], minlength=N)` when recording is on.

## Silencing

`silenced: np.ndarray[bool, N]` or `None`, passed to `run()`. Anything else
raises `ValueError` (`core.silenced_mask`). Without that check an index array in
its place would be read out of bounds by the dense lanes, and a length-1 mask
would broadcast to every neuron in the kernel lanes, neither with an error.

- `engine_metal.py` / `engine_fused.py`: a silenced source gets an empty edge
  range. `engine_metal.silenced_row_end` builds, once per run, where each
  source's range ends: `row_ptr[i + 1]`, or `row_ptr[i]` for a silenced source.
  The propagation kernel reads it after its early exit, in place of
  `row_ptr[i + 1]`, so only threads whose source spiked touch it, and the spike
  arrays stay unmasked for recording. Without a mask the array is a view of
  `row_ptr`; one kernel serves both cases.
- `engine_naive.py` / `engine_chunked.py`: `signed_counts_eff =
  mx.where(silenced[edge_src], 0, signed_counts)` once per run. A 59 MB copy on
  FlyWire; acceptable in the slow lanes, and a different implementation of the
  same semantics, which is what the parity test wants. With `silenced is None`
  they use the pack's counts directly, so an unsilenced run pays neither the
  copy nor its memory.

Semantics, matching upstream code: a silenced neuron still integrates and still
spikes (its spikes are recorded); it delivers nothing. Excitation and silencing
may overlap.

### Why not a mask test in the early exit

The first implementation (87e9afc) did what this design originally proposed:
`if (!spike[i] || silenced[i]) return;`. An all-false mask cost +36 % of the
fused lane on MaleCNS, so `silenced=None` got a second kernel without the mask,
and a run that did silence kept paying +36.50 % for reasons unknown at the time.
What was measured on 2026-09-14 to find them (M4 Pro, `powermode 0`,
measurement scripts not in the repository):

**The cost grows with K and is absent at K=1, on both datasets.** Fused lane,
10 silenced neurons that never fire, so the spike train is unchanged (spike-count
SHA-256 asserted equal across arms); median of 7 runs interleaved in one
process, against the kernel without a mask in the same process:

| MaleCNS, 100 hubs, 2,000 ticks | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| no mask, s / biological s | 2.9479 | 1.8436 | 1.3332 | 1.1676 | 1.5455 |
| mask in the exit | +0.08 % | +1.48 % | +5.54 % | +36.11 % | +74.54 % |
| mask as range end | −0.01 % | +0.62 % | +0.36 % | +0.31 % | +0.20 % |

FlyWire at K=8: +65.83 % with the sugar drive, +29.32 % with the hub drive. At
K=1 with the sugar drive −1.92 %, noise, which is why the first measurement,
taken at K=1 only, saw nothing on FlyWire. Not specific to fusion or chunking:
the sparse lane paid +21.66 % on MaleCNS at K=8, and the fused lane with one
tick per eval +25.63 %.

**It is the read in the exit, however it is written.** On MaleCNS at K=8 a
nested second `if` (+36.10 %) and `hi = silenced[i] ? lo : row_ptr[i + 1]`
(+35.93 %) cost what `||` did; the mask bound as a kernel input but never read
cost −0.03 %.

**The kernel alone reproduces it when nothing spikes.** Propagation dispatched
on its own, 32 calls per eval, no source spiking, µs per call:

| | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| FlyWire, no mask | 15.3 | 23.3 | 32.2 | 51.6 |
| FlyWire, mask in the exit | +0.9 % | +41.3 % | +58.5 % | +72.6 % |
| MaleCNS, no mask | 17.5 | 26.8 | 39.0 | 64.1 |
| MaleCNS, mask in the exit | −1.0 % | +41.7 % | +57.5 % | +69.1 % |
| MaleCNS, mask as range end | −1.4 % | −1.8 % | −2.1 % | −3.3 % |

So the cost is paid by the threads that exit, nearly all 1.33 M of them at K=8
on MaleCNS. The earlier isolated measurement missed it because the four highest
out-degree neurons spiked in every call: on MaleCNS that dispatch took 127 µs
without the mask, and the extra read did not show (−1.7 %). Replaying the spike
arrays the fused lane actually produced gives +18.3 µs per call at K=8
(+29.3 %), less than half of the +42.2 µs per tick measured end to end; that
difference is not explained.

Why a read at index `gid / K` costs this much and one at `gid` costs nothing was
not established. The compiled GPU code was not inspected (`xcrun metal` needs
the Metal toolchain, which is not installed here).

### Measured

Before (87e9afc) against after, one session, M4 Pro, `powermode 0`, load average
3.2 to 6.4 from desktop applications, s per biological second, median of 7 runs
interleaved in one process; within each drive every arm produced the same
spike-count SHA-256:

| | before, no mask | before, 10 silenced | after, no mask | after, 10 silenced |
|---|---|---|---|---|
| FlyWire, 21 sugar GRNs, fused K=1, 10,000 ticks | 0.3013 | 0.2973 (−1.33 %) | 0.2999 (−0.46 %) | 0.2964 (−1.16 %) |
| FlyWire, 100 hubs, fused K=8, 2,000 ticks | 1.1182 | 1.4520 (+29.85 %) | 1.1222 (+0.36 %) | 1.1227 (+0.04 %) |
| FlyWire, 100 hubs, sparse K=8, 2,000 ticks | 1.5956 | 1.9153 (+20.03 %) | 1.5973 (+0.11 %) | 1.6014 (+0.25 %) |
| MaleCNS, 100 hubs, fused K=8, 2,000 ticks | 1.1666 | 1.5890 (+36.21 %) | 1.1701 (+0.30 %) | 1.1702 (+0.01 %) |
| MaleCNS, 100 hubs, sparse K=8, 2,000 ticks | 1.7101 | 2.0795 (+21.60 %) | 1.7134 (+0.19 %) | 1.7120 (−0.08 %) |

"Before, 10 silenced" and "after, no mask" are relative to "before, no mask";
"after, 10 silenced" is relative to "after, no mask". The sparse lane with the
sugar drive at K=2 could not be measured at that load: two runs, of 7 and 15
repetitions, gave +4.06 % and −0.09 % for "after, no mask", with the runs of a
single arm spread by up to 23 %.

## Stimulus with two rates

`core.make_stimulus_for(pack, targets, rate_hz, n_ticks, seed, targets2=(),
rate2_hz=0.0)` returns a `Stimulus` whose `draws[T, K1 + K2]` use
`p = rate * dt / 1000` per column, from `numpy.random.default_rng(seed)` as
before. As implemented, the second set is keyword-only with upstream's defaults
(`neu_exc2=[]`, `r_poi2 = 0 Hz`); the first set is drawn before the second, so
adding a second set leaves the first set's draws bit for bit as they were; and
`make_stimulus` calls it, so one function draws. A neuron listed in both sets is
an error (upstream would give it two inputs; this design refuses rather than
guess), and so are an index outside the pack and a rate whose probability per
tick is not from 0 to 1; a negative or NaN rate would otherwise draw no input,
without an error.

`rfc_reload = 0` for every driven neuron, as `initial_state` already does. That
includes a second set at 0 Hz: upstream's `poi()` sets `rfc = 0 ms` on every
neuron of `neu_exc2`, whatever `r_poi2` is. Such a neuron receives no input and
is never refractory, so this design's first expectation, that `neu_exc2` at
`r_poi2 = 0` reproduces the single-set run bit for bit, holds only for neurons
that never fire. Measured 2026-09-14 in the configuration of
`tests/test_engines.py` (FlyWire, hub drive, 400 ticks, fused lane), each set
added at 0 Hz:

| second set | its spikes | all spikes |
|---|---|---|
| none | | 1,400 |
| the 10 highest out-degree neurons that never fire | 0 → 0 | 1,400, bit-identical |
| the 10 highest out-degree neurons that fire | 27 → 34 | 1,397 |
| the 10 that fire most often | 55 → 74 | 1,418 |
| all 462 non-driven neurons that fire | 749 → 831 | 1,512 |

`Stimulus` now also checks what the lanes assume of any drive, however it was
built: every target once, no negative or non-integer target, and draws a bool
`mx.array` of shape `(n_ticks, K)`. A target listed twice broke lane parity
without an error: the fused lane maps each neuron to one draw column and keeps
the last, the other three add every column. With one neuron listed twice and a
single draw in its first column, it fired in the naive, chunked and metal lanes
and not in the fused lane.

Trial `n` uses `seed = base_seed + n`.

## `run_exp`

```python
def run_exp(
    exp_name: str,
    neu_exc: Sequence[int | str],
    path_res: str | Path,
    *,
    pack: core.Pack | None = None,        # default core.load_pack()
    params: Mapping | None = None,        # see below
    neu_slnc: Sequence[int | str] = (),
    neu_exc2: Sequence[int | str] = (),
    names: Mapping[int, str] | None = None,   # id -> name, upstream's flyid2name
    seed: int = 0,
    engine: str = "fused",                # any lane name, for the parity tests
    cap: int = 262_144,
    force_overwrite: bool = False,
    path_comp: str | Path | None = None,  # accepted for **config compatibility
    path_con: str | Path | None = None,
    n_proc: int | None = None,            # accepted and ignored: one GPU
) -> Path
```

- **IDs.** `int` entries are dataset IDs (`pack.neuron_ids`); `str` entries
  are looked up in `names`, which is upstream's `flyid2name` dict (id -> name)
  inverted internally; duplicate names in it raise. Unknown IDs or names raise
  with the full list of offenders. Lookup is a dict built from
  `pack.neuron_ids` once.
- **Return value.** The written `Path`. Upstream returns `None`; notebooks that
  ignore the return are unaffected.
- **`path_comp` / `path_con`.** If given, their SHA-256 is compared with the
  pack manifest's recorded source hashes and a mismatch raises. This turns the
  two upstream arguments into a guard against running against the wrong data.
- **`params`.** Plain floats: `t_run` in ms, `n_run`, `r_poi` and `r_poi2` in
  Hz. `experiment.default_params` provides upstream's values. Any other key
  whose value differs from the compiled constant raises `NotImplementedError`
  naming the key; identical values pass, so a notebook that forwards upstream's
  dict with unit-free numbers keeps working. Brian2 `Quantity` values are not
  accepted (converting them needs Brian2).
- **`t_run`** must be a whole number of ticks.
- **Output.** `{path_res}/{exp_name}.parquet`, written with pyarrow. Columns and
  dtypes exactly as upstream. Rows sorted by `(trial, flywire_id, t)`. Parquet
  file metadata: `dataset`, `engine`, `seed`, `pack_sha256` (hash of the
  manifest's array hashes), `mlx_lif_engine_version`, and the experiment
  parameters `t_run_ms`, `n_run`, `r_poi_hz`, `r_poi2_hz`, `dt_ms`, which
  `rates()` reads back.
- **`flywire_id` on a MaleCNS pack** holds a MaleCNS body ID. The column name
  is the upstream contract and stays; the `dataset` metadata says what it
  holds. Documented as a wart inherited from compatibility.
- **Skip semantics.** Same as upstream: existing file and
  `force_overwrite=False` prints the upstream-style skip message and returns.

`rates(parquet_path, *, names: Mapping[int, str] | None = None) -> pyarrow.Table`
with columns
`flywire_id, name, rate_hz, std_hz`, computed exactly as upstream's `get_rate`
(per-trial count / `t_run`, then mean and population std over `n_run`).
`t_run` and `n_run` come from the file metadata. No pandas dependency; users
with pandas call `.to_pandas()` or use upstream's function directly.

As implemented (`src/lif/experiment.py`, `tests/test_experiment.py`), it
differs from the above in these points:

- **Units.** Upstream's example notebook sets `params['r_poi'] = 100 * Hz` and
  passes `params['t_run']` to `get_rate`. With `t_run` in milliseconds that
  notebook's rates would come out 1,000 times too low, and its quantity would
  be refused. So `default_params` holds plain numbers in seconds, volts and
  hertz, which is what `float()` gives for a Brian2 quantity, and a quantity is
  accepted for any value, its unit checked with Brian2.
- **Signature.** Upstream's positional order is kept, with `path_comp` and
  `path_con` optional; only the additions are keyword-only. `edge_split`
  defaults to 1, the value for a sparse drive.
- **Names.** A name given to two neurons raises only when it is used. With no
  `names` given, a name is looked up in the pack's names sidecar (`lif.names`,
  ROADMAP phase 2), where it selects every neuron whose instance it is and, if
  there is none, every neuron whose type it is. The FlyWire pack has no sidecar,
  so there a name without `names` still raises.
- **Rows** are sorted by trial, model index and time, which is upstream's order;
  both packs list their IDs in ascending order, so it is also by ID.
- **Metadata** is one JSON value under the key `mlx_lif_engine`, which also
  records the three neuron lists and `edge_split`; `t_run` is `t_run_s`.
- **Tests 8 and 9** run on the full FlyWire pack, three trials of 100 ms. `rates`
  equals upstream's `get_rate` exactly, and the `default_params` literal of
  upstream's `model.py`, evaluated with Brian2's units, equals `default_params`.

## Testing

All parity tests run on the 800-neuron validation subnetwork plus, where cheap,
the full pack.

1. **Event parity across lanes.** Same stim, `record=True`: the sorted event
   arrays of naive, chunked, metal and fused are identical. Implemented on the
   full pack with 400 ticks of the 100-hub drive in chunks of 32, so 13 chunks
   with a short last one; recording must also leave counts, `v` and `g`
   unchanged.
2. **Counts agree with events.** `np.bincount(events[:,1], minlength=N) ==
   spike_counts` for every lane.
3. **Determinism.** Three fused runs with recording yield identical sorted
   events, and the same `spike_counts` SHA-256 as `record=False`.
4. **Overflow.** `cap=8` on a run with more than 8 spikes in a chunk raises
   `RecordOverflow` and the exception names the tick range.
5. **Silencing parity across lanes.** All four lanes agree, and the total
   differs from the unsilenced run. As first written here (the 10 highest
   out-degree neurons under the sugar stimulus) the test was vacuous: those
   neurons fire 0 spikes under that drive, measured at 400 and at 10,000
   ticks, so silencing them changes nothing. Implemented as two cases instead:
   the 10 highest out-degree neurons among those that do fire under the sugar
   drive, and the 10 highest overall under the 100-hub drive, which excites
   them directly, so excitation and silencing overlap.
6. **Silencing semantics.** Silence a single source with exactly one downstream
   target that has no other input; that target records zero spikes, the
   silenced source still records its own, and nothing else fires. Planned on
   the subnetwork, where no such pair exists: no neuron there has out-degree 1
   into a single-input target, and all 114 single-input neurons are fed by an
   inhibitory seed hub, so they could never fire. Implemented on the full pack,
   which has 56 excitatory pairs of this shape. Silencing the target instead
   leaves its spike count unchanged, which pins "outgoing only". A `silenced`
   that is not a bool array of shape `(N,)` raises in every lane.
7. **Brian2 spike times.** `validate_brian2.py` extended: for each
   configuration in the existing table, the set of `(neuron, time)` from
   Brian2's `SpikeMonitor` against ref64's event set converted with
   `tick_to_seconds`, and against the fused lane's. Runs only when Brian2 is
   installed, as today. `tests/test_brian2.py` asserts ref64 exactly in all
   four configurations; the fused lane within 5 % of Brian2's spikes left out
   and added in all four; and the fused lane exactly in the shortest
   configuration and in the first 350 ticks of the densest, where 549 of 2,795
   spikes are fired by neurons that are not driven. The shortest configuration
   alone would not do: all its spikes are driven neurons', one tick after each
   input, so an MLX-only change of the axonal delay passed it, and fails the
   350-tick prefix (620 spikes left out, 634 added).

   Measured 2026-09-14: ref64 equals Brian2 exactly in all four, every spike
   time on the tick grid, so there is no offset. This design expected the fused
   lane to differ only by the borderline spike its counts already showed. It
   differs more: in the four configurations 0, 23, 28 and 38 of Brian2's spikes
   are missing from its events and 0, 24, 28 and 38 other spikes are in them,
   in 13, 21 and 20 neurons, the first difference at 19.6, 18.1 and 35.1 ms.
   Pairing each neuron's missing and extra spikes in time order, where their
   numbers are equal, 55 of 81 pairs are one tick early. Its counts still match
   in three configurations, because a spike moved to another tick counts the
   same. All four MLX lanes record identical events
   on the subnetwork in every configuration, so this is float32 rounding, not a
   lane.
8. **Upstream compatibility.** Write a parquet with `run_exp` on the
   subnetwork, load it with `data/ref/utils.py`'s `load_exps` and `get_rate`,
   compare with `experiment.rates`. Requires pandas, which is in the
   `reference` extra, and skipped when absent.
9. **`**config` compatibility.** `run_exp(exp_name, neu_exc, **config)` with
   upstream's exact `config` dict shape (including `n_proc`) runs; a
   `path_comp` with a wrong hash raises.
10. **Two rates.** Written as `neu_exc2` at `r_poi2 = 0` reproducing the
    single-set result bit for bit, which holds only for neurons that never fire
    ("Stimulus with two rates"). Implemented on `make_stimulus_for`, as `run_exp`
    does not exist yet, on the full pack with the hub drive as the first set: a
    second set at 0 Hz of the 10 highest out-degree neurons that never fire
    leaves counts, `v` and `g` bit-identical; one of the 10 highest that do fire
    changes their spike count; at 150 Hz the second set's neurons fire. Without a
    pack, `tests/test_stimulus.py` checks that the first set's draws do not
    depend on the second, that neither set is refractory, and each refusal.
11. **The published notebook.** `python -m lif.validate_notebook` runs
    upstream's `example.ipynb` with its code cells unchanged, `model` resolving
    to `lif.experiment`, and checks that every file it writes carries this
    engine's metadata. It then runs the five experiments whose Brian2 files
    upstream publishes in `results/example` with seeds 0 and 1000, and compares
    every neuron's rate over the 30 trials with the file's as a z score, and the
    two seeds' runs with each other. It exits non-zero if spikes per trial or
    MN9 differ by |z| 4 or more, or more than 1 % of the neurons by more than 4.
    Measured 2026-09-15: all 15 comparisons pass, and between Brian2 and this
    engine spikes per trial differ by at most z 1.90 and MN9 by at most z 1.70
    (README, "Correctness"). Upstream's `sugarR` file was written at 200 Hz, not
    at the 150 Hz of `model.py`'s `default_params`. Without the pack,
    `tests/test_validate_notebook.py` checks the parts: which cells run, the
    stand-in `model`, spike counts per trial, the z and the gate.

Test 7 settled the spike-time offset: there is none. Every measurement this
document called for is in the README, and the Brian2 figure it compared against
was corrected on 2026-09-15 (README, "Results").
