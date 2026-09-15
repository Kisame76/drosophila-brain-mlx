# Attribution

This project reimplements a published model. It contributes an implementation,
not the model, the data, or the science behind either.

## The model and the data

The leaky integrate-and-fire model, all of its parameters, and the compiled
connectivity come from:

> Shiu et al., *A leaky integrate-and-fire computational model based on the
> connectome of the entire adult Drosophila brain reveals insights into
> sensorimotor processing.*
> https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1

Code and data: https://github.com/philshiu/Drosophila_brain_model (MIT).
`tools/fetch_upstream.sh` downloads `2023_03_23_completeness_630_final.csv` and
`2023_03_23_connectivity_630_final.parquet` from that repository, together with
its `model.py`, `utils.py`, `Readme.md` and `example.ipynb` and the five result
files of that notebook in `results/example`, which `python -m lif.validate_notebook`
compares against; none of these files is redistributed here.

The connectome itself is FlyWire v630. Cite FlyWire per its own policy when
publishing results derived from it: https://flywire.ai

FlyWire's data licence carries a non-commercial restriction. The exact variant
is not restated here because it has not been verified against the source for
this repository; read the terms at https://codex.flywire.ai before any
commercial use. This matters mainly in contrast with MaleCNS below, which does
not carry that restriction.

## MaleCNS v1.0

The second supported pack is built from the MaleCNS v1.0 flat-connectome
release, a male specimen covering brain and ventral nerve cord. It is a
different animal from a different laboratory than FlyWire, and results from the
two packs are not comparable.

> MaleCNS project: https://male-cns.janelia.org/
> v1.0 data and release terms: https://male-cns.janelia.org/download/

Licensed **CC BY 4.0** per the release page, i.e. attribution only, with no
non-commercial restriction. Credit FlyEM at HHMI Janelia, the
University of Cambridge, the MRC LMB, Google Research and the MaleCNS
collaboration. `tools/fetch_male_cns.sh` downloads the three published Feather
tables; none of them is redistributed here.

The node selection and transmitter signs follow the materialization
`male-cns-v1.0-superclass-non-null-known-nt`, which is also what `flyBrain`
publishes, so the two are comparable. The model constants are Shiu et al.'s and
were fitted to FlyWire, not to this dataset; `src/lif/compile_pack_malecns.py`
records that in the pack manifest.

## Taste neuron types

The activity film (`src/lif/activity_film.py`) drives the MaleCNS types LB3b and
LB3c because of

> Tastekin et al., *From Sensory Detection to Motor Action: The Comprehensive
> Drosophila Taste-Feeding Connectome.* bioRxiv, 2025.
> https://www.biorxiv.org/content/10.1101/2025.08.25.671814v1

which matches both types to Gr64f-GAL4 neurons, so they are likely sweet-sensing.
The soma positions the film draws are the `somaLocation` column of the MaleCNS
annotation table above.

## Brian2

Validation runs against Brian2 2.10.1 (CeCILL-2.1), and the closed-form update
coefficients in `src/lif/core.py` are transcribed from the code Brian2 generates
for `method='linear'`. https://briansimulator.org

## flyBrain

`mehrantsi/flyBrain` (MIT) is both a performance reference and a source of ideas.
Its MLX lane demonstrated `mx.fast.metal_kernel` for CSR propagation, and its
description of fusing decay/threshold work with propagation is the basis for the
two-dispatch tick in `src/lif/engine_fused.py` — the largest single speedup in
this repository.
`tools/setup_flybrain_reference.sh` clones and builds it locally; no code from it
is vendored or redistributed here. That script disables its MuJoCo-dependent
modules so the neural benchmark builds without MuJoCo — the compute core
(`rust/src/metal_engine.rs`) is left untouched, and the original `lib.rs` is kept
alongside as `lib.rs.orig`.
