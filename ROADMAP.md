# Roadmap

Where this repository is going, in what order, and what is deliberately not on
it. Numbers here are measured unless marked otherwise.

## Where it stands

The engine runs the published Shiu et al. model at 0.29 s per biological second
on an M4 Pro, parity-gated against Brian2. What it cannot do yet is the thing
the published model is *for*: activate a set of neurons, silence another, run 30
trials and read out firing rates. Today the repository supports exactly one
experiment, and it is a constant in the code. Closing that gap is phase 1.

## Phase 0: MaleCNS pack (in progress)

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

Known property of the published materialization, recorded in the manifest:
11,609 of 166,700 neurons have no signed transmitter and therefore no outgoing
edges. 8,024 of those are histaminergic, and histamine is the photoreceptor
transmitter in the fly, so most of the retina's output is absent from the
signed graph. The model constants are
reused unchanged from Shiu et al.; they were fitted to FlyWire, not to this
dataset.

## Phase 1: experiment layer

`run_exp(exp_name, neu_exc, path_res, neu_slnc=..., neu_exc2=..., ...)` with
the same signature and the same parquet output as
`philshiu/Drosophila_brain_model`, so the analysis notebooks published with the
model run unchanged against this engine.

One standard experiment is 30 trials of 1 biological second:

| | per trial | 30 trials |
|---|---|---|
| Brian2 2.10.1, one core | 62.6 s | 31 min |
| this engine, fused lane | 0.29 s | 9 s (end to end not yet measured) |

Two engine additions are required, both parity-gated across all four lanes:
**spike-event recording** (which neuron fired in which tick) and **silencing**.
Recording also lets the Brian2 validation compare spike *times*, not only
counts, which makes the correctness gate stronger than it is today.

Design: [docs/design/2026-09-14-experiment-layer.md](docs/design/2026-09-14-experiment-layer.md)

## Phase 2: names

FlyWire ships no cell-type names; the published model takes a user-supplied
`flyid2name` dict and so does phase 1. MaleCNS ships `type`, `instance`,
`class` and a `flywireType` cross-reference per neuron. Phase 2 stores those as
a sidecar in the MaleCNS pack and lets `run_exp` resolve names from it, so
neurons can be addressed as `"MN9"` rather than by a 19-digit ID.

## Phase 3: seeing it

Ordered by how much each shows per unit of work.

1. **Rate over time** for chosen neurons: with spike events recorded, this is a
   histogram. A plotting script (matplotlib, optional dependency) that shows,
   for example, sugar input at 0 ms and the proboscis motor neuron responding.
   Every point on that plot is a measured spike.
2. **Whole-brain activity film.** Needs 3D neuron coordinates, which neither
   pack carries today; the MaleCNS annotation table has `somaLocation` and that
   is the first thing to check. Deferred until a coordinate source exists, and
   when built it is labelled as visualization, not evidence.

## Phase 4: a body (conditional)

Coupling to flybody, FlyGym or flyBrain's MuJoCo scene. What this repository
would provide: per-tick rates of named motor populations, and a realtime brake
so one biological second takes one wall-clock second. What it would not own:
gait generators, odour decoders, landing gates and the rest of the engineered
layer between motor neurons and joints. flyBrain's README says plainly that
those are "engineering interfaces, not recovered circuits", and the same would
be true here. Starts only after phases 1 to 3, and only with a partner body.

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
