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
`2023_03_23_connectivity_630_final.parquet` from that repository; neither file is
redistributed here.

The connectome itself is FlyWire v630. Cite FlyWire per its own policy when
publishing results derived from it: https://flywire.ai

## Brian2

Validation runs against Brian2 2.10.1 (CeCILL-2.1), and the closed-form update
coefficients in `src/lif/core.py` are transcribed from the code Brian2 generates
for `method='linear'`. https://briansimulator.org

## flyBrain

`mehrantsi/flyBrain` (MIT) is used as a performance reference only.
`tools/setup_flybrain_reference.sh` clones and builds it locally; no code from it
is vendored or redistributed here. That script disables its MuJoCo-dependent
modules so the neural benchmark builds without MuJoCo — the compute core
(`rust/src/metal_engine.rs`) is left untouched, and the original `lib.rs` is kept
alongside as `lib.rs.orig`.
