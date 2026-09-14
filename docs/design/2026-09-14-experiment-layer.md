# Experiment layer: `run_exp` compatible with the published model

Date: 2026-09-14. Status: design, awaiting review. Implements phase 1 of
[ROADMAP.md](../../ROADMAP.md).

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
src/lif/spike_record.py   one Metal kernel: [K, N] uint8 spike stack -> (tick, neuron)
                          event pairs; host-side sort and overflow check
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
`experiment.py` only:

```python
def tick_to_seconds(tick: np.ndarray) -> np.ndarray:
    return tick.astype(np.float64) * (core.DT / 1000.0)
```

Expected correspondence to Brian2: engine tick `k` performs the same state
update, threshold and reset sequence that Brian2 performs at clock time
`k * dt`, and `SpikeMonitor` records the threshold time. So `t = k * dt` with no
offset. This is an expectation, not a fact; the `StateMonitor` mistake recorded
in the README is why. The Brian2 validation (below) asserts it, and if it fails
the offset lives in this one function.

### Kernel

Runs **once per chunk**, not per tick. The fused lane already produces one
`uint8[N]` spike array per tick and keeps the chunk's arrays alive until
`async_eval`; the kernel consumes that stack.

```
inputs   spikes[K, N] uint8, t0 uint32, cap uint32
outputs  ev_tick[cap] int32, ev_neuron[cap] int32, count[1] uint32 (atomic)
grid     (K * N, 1, 1)

uint gid = thread_position_in_grid.x;
uint k = gid / N, i = gid % N;
if (k >= K || i >= N) return;
if (!spikes[k * N + i]) return;
uint slot = atomic_fetch_add_explicit(&count[0], 1u, memory_order_relaxed);
if (slot >= cap) return;                      // overflow: detected on the host
atomic_store_explicit(&ev_tick[slot], int(t0 + k), memory_order_relaxed);
atomic_store_explicit(&ev_neuron[slot], int(i), memory_order_relaxed);
```

`atomic_outputs=True` makes every output atomic in `mx.fast.metal_kernel`, hence
`atomic_store_explicit` for the two arrays (same pattern `engine_metal.py`
already uses). `#pragma clang fp contract(off)` is irrelevant here; there is no
float arithmetic.

Cost: one extra dispatch per chunk of 32 ticks, and one extra read of the
`K * N` spike bytes that already exist, i.e. 4 MiB per chunk or 0.125 MiB per
tick. Against the fused tick's ~0.0295 ms and this machine's ~114 GiB/s that is
on the order of 1 ms per 1,000 ticks, so single-digit percent rather than the
28 % first estimated here (that estimate came from a per-dispatch cost model
that measurement later disproved; see docs/mlx-notes.md §3). Measured after
implementation; the number goes into the README next to the fused-lane row as
`record=True` overhead.

### Host side

Per chunk: read `count`; if `count > cap`, raise `RecordOverflow(tick_range,
count, cap)` with the advice to raise `cap`. Otherwise slice `[:count]`,
append. At the end of the run, `np.lexsort((neuron, tick))`. The atomic slot
order is nondeterministic; the sorted event set is not, and a test asserts it.

Default `cap = 262,144` events per chunk (2 MB), regardless of the lane's
chunk length. The densest run measured so far, 100 hub neurons at 150 Hz,
produced ~52,000 spikes per 10,000 ticks, i.e. ~166 per 32-tick chunk on
average; the cap is ~1,500x that. Burst peaks are not measured; the overflow
check exists for them.

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
raises `ValueError` (`core.silenced_mask`): an index array in its place would be
read out of bounds by the kernel, without an error.

- `engine_metal.py` / `engine_fused.py`: the propagation kernel gains a
  `silenced` input and its early exit becomes
  `if (!spike[i] || silenced[i]) return;`. The plan was to pass an all-false
  array when `silenced is None` and to template the kernel into two variants
  only if that cost more than 1 % of the fused lane. It did (below), so there
  are two, and `silenced=None` runs the kernel from before silencing existed,
  source-identical.
- `engine_naive.py` / `engine_chunked.py`: `signed_counts_eff =
  mx.where(silenced[edge_src], 0, signed_counts)` once per run. A 59 MB copy on
  FlyWire; acceptable in the slow lanes, and a different implementation of the
  same semantics, which is what the parity test wants. With `silenced is None`
  they use the pack's counts directly, so an unsilenced run pays neither the
  copy nor its memory.

Measured 2026-09-14, M4 Pro, `powermode 2`, fused lane, s per biological second,
median of 7 runs interleaved in one process; every arm without an effective mask
produced the same spike-count SHA-256:

| | before silencing | one kernel, all-false mask | two variants, `None` | two variants, 10 silenced neurons that never fire |
|---|---|---|---|---|
| FlyWire, 21 sugar GRNs, K=1, 10,000 ticks | 0.2975 | 0.2987 (+0.39 %) | 0.2983 (+0.25 %) | 0.2985 (+0.09 % vs `None`) |
| MaleCNS, 100 hubs, K=8, 2,000 ticks | 1.1642 | 1.5853 (+36.17 %) | 1.1638 (−0.03 %) | 1.5886 (+36.50 % vs `None`) |

The FlyWire row is inside that session's noise (load average 3.2 to 4.1). An
earlier run of the same comparison, at load average 2.1 to 2.5, measured the
one-kernel version at +1.06 % there and +36.18 % on MaleCNS, so the decision
does not rest on a single run.

Open: a run that does silence pays the MaleCNS cost, and why is not understood.
On MaleCNS a second `if` instead of `||` measured −0.32 %, and a uint8 mask the
same as bool. The propagation kernel dispatched on its own (4 hubs spiking,
nothing silenced, 1,000 calls) pays +10.8 % for reading the mask on FlyWire at
K=8, nothing measurable on MaleCNS at K=8, and nothing at K=1 on either; passing
the mask without reading it costs nothing. So the cost is the read, and it
depends on something the isolated dispatch does not reproduce.

Semantics, matching upstream code: a silenced neuron still integrates and still
spikes (its spikes are recorded); it delivers nothing. Excitation and silencing
may overlap.

## Stimulus with two rates

`core.make_stimulus_for(pack, targets_a, rate_a_hz, targets_b, rate_b_hz,
n_ticks, seed)` returns a `Stimulus` whose `draws[T, Ka + Kb]` use
`p = rate * dt / 1000` per column. `numpy.random.default_rng(seed)` as today.
A neuron listed in both sets is an error (upstream would give it two inputs;
this design refuses rather than guess).

Trial `n` uses `seed = base_seed + n`. `rfc_reload = 0` for every driven neuron,
as `initial_state` already does.

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

## Testing

All parity tests run on the 800-neuron validation subnetwork plus, where cheap,
the full pack.

1. **Event parity across lanes.** Same stim, `record=True`: the sorted event
   arrays of naive, chunked, metal and fused are identical.
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
   Brian2's `SpikeMonitor` equals ref64's event set converted with
   `tick_to_seconds`, exactly; and equals the fused lane's except for the
   already-documented single borderline spike. Runs only when Brian2 is
   installed, as today.
8. **Upstream compatibility.** Write a parquet with `run_exp` on the
   subnetwork, load it with `data/ref/utils.py`'s `load_exps` and `get_rate`,
   compare with `experiment.rates`. Requires pandas; added to the `dev` extra
   and skipped when absent.
9. **`**config` compatibility.** `run_exp(exp_name, neu_exc, **config)` with
   upstream's exact `config` dict shape (including `n_proc`) runs; a
   `path_comp` with a wrong hash raises.
10. **Two rates.** `neu_exc2` at `r_poi2 = 0` reproduces the single-set
    result bit for bit; at `r_poi2 > 0` the second set's neurons fire.

## Measurements to add to the README

- fused lane with `record=True`, s per biological second and peak memory
- fused lane with a non-empty `silenced` mask
- `run_exp` end to end, 30 trials, sugar stimulus: total wall clock, and the
  Brian2 figure it replaces (31 min, extrapolated from the measured 62.6 s per
  trial and stated as such)

## Open points

None that block implementation. The spike-time offset is decided by test 7 and
has exactly one place to change.
