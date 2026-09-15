# Roadmap

Where this repository is going, in what order, and what is deliberately not on
it. Numbers here are measured unless marked otherwise.

## Who this is for, in priority order

The order below is the reason the phases are numbered the way they are.

1. **Researchers running the published model.** People who want to activate a
   set of neurons, silence another, and compare firing rates, but do not want to
   wait 31 minutes per experiment. This is what phases 1 and 2 serve, and it is
   the only group the repository can serve without new unverifiable code.
2. **Anyone coupling a brain to a body.** flybody, FlyGym, flyBrain's MuJoCo
   scene. Phase 4, and conditional: the interesting part is also the part this
   repository would not own. It matters because it is the only way the work
   becomes visible to someone who does not read spike-count tables.
3. **MLX developers.** Served already by
   [docs/mlx-notes.md](docs/mlx-notes.md), which needs no further phase.

## Where it stands

The engine runs the published Shiu et al. model at 0.29 s per biological second
on an M4 Pro, parity-gated against Brian2, on FlyWire v630; MaleCNS v1.0 runs
through the same four lanes. A run can drive chosen neurons at one or two rates
(`core.make_stimulus_for`), silence others (`silenced=`) and record which neuron
fired in which tick (`record=True`), and `lif.experiment.run_exp` runs the
published model's experiments with upstream's arguments and parquet format, which
upstream's `get_rate` reads unchanged. Upstream's example notebook runs on it
with its code cells unchanged, and its results agree with the Brian2 files
upstream published for it within trial-to-trial noise (phase 1). On the MaleCNS
pack it takes cell types and instances as names, such as `"MN9"`. One standard
experiment, 30
trials of 1 s, takes 10.05 s end to end. With the sugar GRNs driven at 100 Hz, MN9
fires at 67.30 Hz on the real connectome and not once on any of five copies whose wiring
is shuffled with every neuron's degrees kept (phase 3, the control demo). On the MaleCNS
pack the spikes of such an experiment can be watched at their neurons' soma positions
(phase 3, the activity film).

## Phase 0: MaleCNS pack (done)

A second dataset through the same pack compiler. MaleCNS v1.0 is a male
specimen covering brain **and** ventral nerve cord: 166,700 neurons, including
the leg and wing motor neurons the FlyWire brain-only pack does not contain.

Done when

- `python -m lif.compile_pack_malecns` passes every check and `verify_pack`
  agrees with it through its own code path
- all four lanes pass parity on the new pack
- the README carries a benchmark row for it, with the caveat that spike counts
  are not comparable across specimens
- `ATTRIBUTION.md` credits the MaleCNS collaboration and notes CC BY 4.0

All four hold, re-checked on 2026-09-14: a rebuild passes every compiler check
and is byte-identical to the pack in use, `verify_pack` reports it verified, all
four lanes pass parity in `lif.benchmark`, and the README table and
`ATTRIBUTION.md` are in place.

Known property of the published materialization: 11,609 of 166,700 neurons
have no signed transmitter and therefore no outgoing edges. The compiler's `nt`
check prints that count; the manifest records 11,793 neurons without outputs,
which also counts 184 neurons that have a signed transmitter but no kept
outgoing edge. 7,891 of the 11,609 are histaminergic, and histamine is the
photoreceptor transmitter in the fly, so most of the retina's output is absent
from the signed graph. The model constants are reused unchanged from Shiu et
al.; they were fitted to FlyWire, not to this dataset.

## Phase 1: experiment layer (done)

`run_exp(exp_name, neu_exc, path_res, neu_slnc=..., neu_exc2=..., ...)` with
the same signature and the same parquet output as
`philshiu/Drosophila_brain_model`, so the analysis notebooks published with the
model run unchanged against this engine.

Upstream's example notebook runs against this engine with its code cells
unchanged, and the five of its experiments for which upstream published the
files Brian2 wrote agree with this engine within their trial-to-trial noise.
Checked 2026-09-15 with `python -m lif.validate_notebook`, which runs each of
those experiments with seeds 0 and 1000 and compares every neuron's rate over
the 30 trials as a z score: between Brian2 and this engine, spikes per trial
differ by at most z 1.90 and MN9's rate by at most z 1.70, and 0 to 4 neurons of
409 to 463 differ by more than |z| 3, as between the two seeds' runs of this
engine, except in one seed-0 run with 13 (README, "Correctness"). Upstream's
second notebook, `figures.ipynb`, was not run.

One standard experiment is 30 trials of 1 biological second:

| | per trial | 30 trials |
|---|---|---|
| Brian2 2.10.1, one core | 62.6 s | 31 min |
| this engine, `run_exp`, fused lane | 0.335 s | 10.05 s |

Brian2's 31 min are extrapolated from one measured trial. This engine's row is
`run_exp` end to end, measured 2026-09-14 in High Power mode, load average 1.1
to 1.4: 21 sugar GRNs at 150 Hz, pack already loaded, three repetitions of
10.21, 10.05 and 10.05 s, the first including kernel compilation, each writing
409,437 spikes.

### What you get out of it

1. **A rate table.** One row per neuron: id, name, spikes per second averaged
   over the 30 trials, and the standard deviation across them. Identical in
   content to what upstream's `get_rate` produces, so published analyses can be
   reproduced directly. This is the primary deliverable; everything else in this
   phase exists to produce it.
2. **A before-and-after comparison.** Run once normally, once with a chosen
   neuron silenced, and diff the two rate tables. This is the actual scientific
   question the model is built to answer ("what does this neuron do?"), and it
   is why `silence` is in this phase rather than a later one. Implemented: a
   silenced neuron's edge range is emptied for the run. With a mask the fused
   lane measured +0.01 % on the MaleCNS hub drive and −1.16 % on the FlyWire
   sugar drive against the same run without one; the first version, which
   tested the mask in the kernel's early exit, cost +36 % (design doc,
   "Silencing").

### What it costs to build

Two engine additions, both parity-gated across all four lanes and both in:
**silencing**, and **spike-event recording** (which neuron fired in which tick).
Recording let the Brian2 validation compare spike *times*, not only counts, and
that comparison is stricter: the float64 oracle matches every Brian2 spike time
in all four validation configurations, while the float32 lanes, whose counts
match in three of them, put up to 38 of 24,700 spikes on a different tick
(README, "Correctness"). Recording costs the fused lane +3.8 % to +13.7 % in run
time, the most on the sparse sugar drive (README, "Use"). The two-rate stimulus
(`core.make_stimulus_for`) is in as well. Upstream never makes its second input
set refractory, at 0 Hz included, so a second set at 0 Hz changes a run as soon
as one of its neurons fires (design doc, "Stimulus with two rates"). `run_exp`
and `rates` are in as well (`lif.experiment`): `rates` equals upstream's
`get_rate` exactly on a file `run_exp` wrote, and `default_params` equals
upstream's, evaluated with Brian2's units. Measured end to end,
30 trials of the sugar drive take 10.05 s (table above).

Design: [docs/design/2026-09-14-experiment-layer.md](docs/design/2026-09-14-experiment-layer.md)

## Phase 2: names (done)

FlyWire ships no cell-type names; the published model takes a user-supplied
`flyid2name` dict, and on the FlyWire pack `run_exp` still does. MaleCNS ships
`type`, `instance`, `class` and a `flywireType` cross-reference per neuron.
`python -m lif.compile_pack_malecns` writes those next to the pack as
`names.parquet`, one row per model index, and records a hash of the table's
content in the manifest; `--names-only` adds the sidecar to a pack already
compiled, once the annotation table's sha256 matches the one the pack was
compiled from. With no `names` given, `run_exp` resolves a name from the sidecar
(`lif.names`): a name selects every neuron whose instance it is and, if there is
none, every neuron whose type it is, so `"MN9_R"` is body 16949 and `"MN9"` is
bodies 10331 and 16949. `lif.names.load(pack).labels()` gives `rates` a name per
neuron. `verify_pack` re-derives the sidecar from the annotation table with a
dict on body ID and compares it row by row; run on 2026-09-15, it found all
166,700 rows equal and verified the rest of the pack, 24,469,412 edges, as
before.

Of the 166,700 neurons in the MaleCNS pack, counted 2026-09-15, 164,506 have a
type, 160,061 an instance, 143,154 a flywireType and 26,513 a class.

## Phase 3: seeing it

Ordered by how much each shows per unit of work.

1. **The control demo: does the wiring matter?** Drive the 21 sugar GRNs as
   upstream's example notebook does and read out MN9, the proboscis motor neuron that notebook reports, twice: once on the
   real connectome, once on a shuffled one that keeps every neuron's number of
   incoming and outgoing connections but wires them at random. Both shown side by
   side as rate over time, a histogram of the recorded spike events, so every
   point on the plot is a measured spike, and below them the sugar GRNs' own rate,
   so a silent MN9 cannot be mistaken for a run in which nothing happened.
   Measured 2026-09-15 with
   `python -m lif.control_demo`, the sugar GRNs driven at 100 Hz for 30 trials of
   1 s, the same input spikes on both: MN9 fires at 67.30 Hz on the real
   connectome and not once on the shuffled one (seed 0), where 96 neurons fire at
   all against 408 on the real one, while the sugar GRNs fire at 99.38 Hz there
   and 99.40 Hz on the real one. Measured the same way on 2026-09-15, four
   more shuffles (seeds 1 to 4) leave MN9 silent too, with 89 to 99 neurons
   firing and 73,735 to 74,342 spikes in all, against 290,963 on the real one.

   Why a control and not a fly playing a game: after the MaleCNS v1.0 paper
   (Cell, 2026-09-03), dozens of projects wired the connectome to games and
   videos went viral ([awesome-fly](https://github.com/cobanov/awesome-fly)
   lists them). Their performance comes from hand-made mappings between screen,
   neurons and keys, from trained decoders or from trained weights, and hardly
   any compares against a baseline that would show the connectome contributes
   anything. This repository can run that comparison on a simulation that is
   parity-gated against the published model.

   What it took: `python -m lif.shuffle_pack` builds the shuffled pack, marked
   in its manifest as not the real connectome (v630, seed 0: 6 repair rounds,
   289 self-loops, and 90,491 of its 14,687,178 edges where the real pack has one
   too; seeds 1 to 4: 6 repair rounds each, 257 to 288 self-loops, 90,402 to
   90,720 shared edges); `python -m lif.control_demo` runs both packs through `run_exp` and
   writes [docs/figures/control-demo.svg](docs/figures/control-demo.svg); MN9's
   ID, 720575940660219265, is the one in upstream's notebook and is in the v630
   pack. The plot is **in the README**, near the top, because a table of
   milliseconds persuades someone who already knows what the model is and nobody
   else. The SVG is checked in next to the script that regenerates it, so it can
   be re-derived rather than trusted. A 3D body scene like flyBrain's can follow
   later, labelled with what in it is measured and what is engineered.

   With the plot in the README, on 2026-09-15 the repository was proposed for
   [awesome-fly](https://github.com/cobanov/awesome-fly) under Brain Models in
   [cobanov/awesome-fly#5](https://github.com/cobanov/awesome-fly/pull/5), and a
   case study of [docs/mlx-notes.md](docs/mlx-notes.md) was posted in the GitHub
   Discussions of ml-explore/mlx as
   [ml-explore/mlx#4512](https://github.com/ml-explore/mlx/discussions/4512). Both
   say what the repository is, what is measured and where its limits are, with no
   comparative claims. Independent of
   the demo, reported as
   [philshiu/Drosophila_brain_model#10](https://github.com/philshiu/Drosophila_brain_model/issues/10):
   its README says silencing removes synapses "to and from" a neuron while
   model.py removes only the outgoing ones.
2. **Whole-brain activity film.** The coordinate source to check was the MaleCNS
   annotation table's `somaLocation`, and it serves: it places 139,662 of the
   pack's 166,700 neurons, and of the 27,038 it does not, 17,907 are sensory
   neurons, 8,348 are intrinsic to the optic lobe and 783 are others.
   `python -m lif.activity_film` drives the 17 LB3b and LB3c neurons of the right
   labellum, which Tastekin et al. (2025) match to Gr64f-GAL4 neurons, at 100 Hz
   for 30 trials of 1 s, and draws every recorded spike at its neuron's soma
   position in [docs/figures/activity-film.png](docs/figures/activity-film.png), an
   animated PNG of 100 frames of 10 ms during which the view turns once about the
   CNS's long axis. Measured 2026-09-15, seed 0: 1,036,265 spikes from 5,021
   neurons, 4,957 of them drawn; neurons of the ventral nerve cord's superclasses
   fire 28.6 % of the spikes in the first 100 ms and 66.9 % in the last; MN9_L
   fires at 52.97 Hz and MN9_R at 0.20 Hz. The README shows it under "Where the
   activity goes", labelled as visualization, not evidence, because the constants
   were fitted to the FlyWire brain.

   What it took: no new pack data. The positions are read from the annotation
   table the pack was compiled from, once its sha256 matches the manifest, and the
   film is written by an APNG encoder of numpy and zlib, so it adds no dependency.

   Checked 2026-09-15 with `python -m lif.stimulus_survey`, 30 trials of each of
   eight stimuli at 100 Hz: only the film's makes the ventral nerve cord's share of
   the spikes rise, from 28.6 % to 66.9 %, and 69 of the 85 neurons above 100 Hz at
   its end are abdominal. Silencing the 2,156 abdominal VNC neurons keeps the share
   at 25.1 to 29.9 %, and with the input switched off after 500 ms, 25 of the 30
   trials go on firing between 2,518 and 3,514 VNC spikes in the last 100 ms, where
   all 30 do with the input left on. So the film shows a self-sustained abdominal
   state of this model that one stimulus of the eight sets off, and that a trial
   either settles into or does not (README, "Where the activity goes").

## Phase 4: a body (conditional)

Coupling to flybody, FlyGym or flyBrain's MuJoCo scene. What this repository
would provide: per-tick rates of named motor populations, and a realtime brake
so one biological second takes one wall-clock second. What it would not own:
gait generators, odour decoders, landing gates and the rest of the engineered
layer between motor neurons and joints. flyBrain's README says plainly that
those are "engineering interfaces, not recovered circuits", and the same would
be true here. Starts only after phases 1 to 3, and only with a partner body.

Why it is on the list at all despite that caveat: a rate table convinces a
neuroscientist and nobody else. A fly that walks is the only output of this
work that a person with no background can look at and understand. That is a
legitimate goal, and keeping it explicitly separate from the measured claims is
how it stays honest.

## Decisions already taken

Recorded so they are not re-argued, with what was rejected and why.

- **Upstream-compatible `run_exp` over a cleaner native API.** A native
  rate-first API (`Experiment(excite=..., silence=..., trials=30)`) would have
  been tidier and is what was originally recommended. Rejected in favour of
  matching upstream's signature and parquet schema, because a researcher's
  existing notebook running unchanged is worth more than a nicer signature.
- **No time-resolved output port for a body.** An earlier proposal was to give
  the engine a per-tick readout of named motor populations. Dropped: checking
  how upstream computes rates showed it is `len(spikes in trial) / t_run`, i.e.
  counts only. The existing `spike_counts` already covers the primary use case,
  and the port solved a problem this project does not have. Per-tick output
  returns in phase 3 for plotting, and in phase 4 if a body ever arrives.
- **Events, not a dense raster.** See "Not planned" below.

## Not planned

- **Model-constant sweeps** (`v_th`, `tau`, ...). The constants are fixed at
  compile time so the closed-form coefficients stay bit-identical to Brian2's.
  Making them per-run parameters is possible; nobody has asked.
- **Full-raster output** (every neuron, every 0.1 ms, dense). 152 MB per trial
  for a 0.001 % occupancy. Events cover every use we know of.
- **Portability.** Metal only. That is the point of the repository.
- **Training or plasticity.** Nothing here learns. See the README.

## Done, outside the phases

[docs/mlx-notes.md](docs/mlx-notes.md): what generalises from this project to
any MLX workload, with the measurements behind it. Writing it disproved the
explanation the README had given for the largest optimisation in the project
(memory traffic, not dispatch count), which is corrected in both places.
