# Graph Report - mlx-lif-engine  (2026-09-14)

## Corpus Check
- Corpus is ~19,760 words - fits in a single context window. You may not need a graph.

## Summary
- 254 nodes · 484 edges · 11 communities (10 shown, 1 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 43 edges (avg confidence: 0.87)
- Token cost: 131,264 input · 0 output

## Community Hubs (Navigation)
- Engine Lanes and Tick Loop
- MLX and Metal Optimisation
- Pack Compilation and Validation
- Datasets, Licensing and Roadmap
- Upstream Compatibility Layer
- Parity and Determinism Tests
- Independent Pack Verification
- Brian2 Subnetwork Validation
- flyBrain Stimulus Reimplementation
- Project Framing and Scope
- Package Root

## God Nodes (most connected - your core abstractions)
1. `CheckLog` - 18 edges
2. `Pack` - 15 edges
3. `engine_fused: Two-Dispatch Fused Lane` - 12 edges
4. `main()` - 11 edges
5. `Stimulus` - 11 edges
6. `main()` - 10 edges
7. `main()` - 10 edges
8. `run()` - 10 edges
9. `RunResult` - 10 edges
10. `write_pack()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `path_comp / path_con SHA-256 Guard` --semantically_similar_to--> `verify_pack.py: Independent Pack Audit`  [INFERRED] [semantically similar]
  docs/design/2026-09-14-experiment-layer.md → README.md
- `Tunable Unroll K for Skewed Rows` --semantically_similar_to--> `edge_split Knob`  [INFERRED] [semantically similar]
  docs/mlx-notes.md → README.md
- `Integer Atomics Are Deterministic, Float Atomics Are Not` --semantically_similar_to--> `int32 Contact-Count Accumulation`  [INFERRED] [semantically similar]
  docs/mlx-notes.md → README.md
- `Dense Dispatch with Early Exits` --semantically_similar_to--> `Finding 1: Dense Is the Problem, Not Synchronisation`  [INFERRED] [semantically similar]
  docs/mlx-notes.md → README.md
- `MLX Writes Every Intermediate to Memory` --semantically_similar_to--> `Finding 3: Intermediate Materialisation`  [INFERRED] [semantically similar]
  docs/mlx-notes.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Five Engine Lanes Held Together by the Parity Gate** — readme_engine_naive, readme_engine_chunked, readme_engine_metal, readme_engine_fused, readme_engine_ref64, readme_parity_gate [EXTRACTED 1.00]
- **Phase 1 Experiment Layer Components** — docs_design_2026_09_14_experiment_layer_run_exp, docs_design_2026_09_14_experiment_layer_experiment_py, docs_design_2026_09_14_experiment_layer_spike_record_py, docs_design_2026_09_14_experiment_layer_spike_event_recording, docs_design_2026_09_14_experiment_layer_silencing, docs_design_2026_09_14_experiment_layer_make_stimulus_for, docs_design_2026_09_14_experiment_layer_parquet_output [EXTRACTED 1.00]
- **Memory-Traffic Fusion Win (1.51 s to 0.29 s)** — readme_finding_intermediate_materialisation, docs_mlx_notes_intermediate_writes, docs_mlx_notes_eval_sync_unit, readme_engine_fused, readme_mx_fast_metal_kernel, docs_mlx_notes_unmeasured_explanation [INFERRED 0.95]

## Communities (11 total, 1 thin omitted)

### Community 0 - "Engine Lanes and Tick Loop"
Cohesion: 0.08
Nodes (43): _chip(), main(), Reproduce the numbers in README.md. python -m lif.benchmark # all MLX lanes,…, constants_f32(), initial_state(), load_pack(), make_stimulus(), Pack (+35 more)

### Community 1 - "MLX and Metal Optimisation"
Cohesion: 0.07
Nodes (48): Pack Stays Immutable and Hash-Verified, params Guard Raising NotImplementedError, RecordOverflow and the Event Cap, RunResult.events Array, Silencing (Per-Run Mask), Outgoing-Only Silencing Semantics, Spike-Event Recording, Per-Chunk Spike-Event Metal Kernel (+40 more)

### Community 2 - "Pack Compilation and Validation"
Cohesion: 0.13
Nodes (30): build_csr(), check_edges(), check_id_mapping(), CheckLog, load_edges(), load_neurons(), main(), load_edges() (+22 more)

### Community 3 - "Datasets, Licensing and Roadmap"
Cohesion: 0.09
Nodes (29): MaleCNS CC BY 4.0 Licensing, mehrantsi/flyBrain Rust/Metal Engine, FlyWire Non-Commercial Data Licence, FlyWire v630 Connectome, male-cns-v1.0-superclass-non-null-known-nt Materialization, MaleCNS v1.0 Flat-Connectome Release, No Data Redistribution Policy, flywire_id Column Holding MaleCNS Body IDs (+21 more)

### Community 4 - "Upstream Compatibility Layer"
Cohesion: 0.15
Nodes (21): Brian2 2.10.1 Reference Simulator, Brian2 method='linear' Closed-Form Coefficients, philshiu/Drosophila_brain_model (Upstream), experiment.default_params, src/lif/experiment.py, core.make_stimulus_for(): Two Target Sets, Two Rates, Upstream-Compatible Parquet Output, rates() Firing-Rate Readout (+13 more)

### Community 5 - "Parity and Determinism Tests"
Cohesion: 0.14
Nodes (16): fixture, parametrize, fresh(), fresh(), fresh(), pack(), End-to-end checks: every lane must agree, bit for bit. Needs a compiled pack…, Every optimisation must reproduce the baseline exactly -- the phase-3 gate. (+8 more)

### Community 6 - "Independent Pack Verification"
Cohesion: 0.19
Nodes (11): main(), mc_sign(), ndarray, Path, Re-derive selection, signs and edges from the published Feather tables. Mapping…, Independent verification of a compiled CSR pack against its raw source files.…, read_malecns(), reader_for() (+3 more)

### Community 7 - "Brian2 Subnetwork Validation"
Cohesion: 0.23
Nodes (11): as_pack(), build(), Carve a small connected subnetwork out of the full pack. Brian2 cannot run 127k…, A subnetwork in its own 0..n-1 index space., Breadth-first from the highest out-degree hubs until n_target neurons.…, Wrap the subnetwork in a Pack so the engines can run it unchanged., SubNet, main() (+3 more)

### Community 8 - "flyBrain Stimulus Reimplementation"
Cohesion: 0.36
Nodes (7): bernoulli(), ndarray, Reimplementation of flyBrain's splitmix64 counter stimulus. Used only to…, Return a bool[steps, n_targets] draw matrix identical to the Rust engine., Model indices of the sugar GRNs, in the engine's lane order., _splitmix64(), targets()

### Community 9 - "Project Framing and Scope"
Cohesion: 0.33
Nodes (7): Shiu et al. LIF Connectome Model, graphify-out Artifacts (graph.json, wiki, GRAPH_REPORT.md), graphify query / path / explain, Graphify Codebase-Question Workflow, mlx-lif-engine, Not Planned: Portability Beyond Metal, Not Planned: Training or Plasticity

## Knowledge Gaps
- **6 isolated node(s):** `mlx-lif-engine`, `setup_flybrain_reference.sh script`, `PATH`, `src/lif/spike_record.py`, `RecordOverflow and the Event Cap` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 69 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Cross-Lane Parity Gate (SHA-256 Spike Counts)` connect `MLX and Metal Optimisation` to `Engine Lanes and Tick Loop`, `Datasets, Licensing and Roadmap`?**
  _High betweenness centrality (0.171) - this node is a cross-community bridge._
- **Why does `Brian2 2.10.1 Reference Simulator` connect `Upstream Compatibility Layer` to `MLX and Metal Optimisation`, `Datasets, Licensing and Roadmap`, `Brian2 Subnetwork Validation`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `CheckLog` (e.g. with `load_edges()` and `load_nodes()`) actually correct?**
  _`CheckLog` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlx-lif-engine`, `setup_flybrain_reference.sh script`, `PATH` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Engine Lanes and Tick Loop` be split into smaller, more focused modules?**
  _Cohesion score 0.07676767676767676 - nodes in this community are weakly interconnected._
- **Should `MLX and Metal Optimisation` be split into smaller, more focused modules?**
  _Cohesion score 0.07092198581560284 - nodes in this community are weakly interconnected._
- **Should `Pack Compilation and Validation` be split into smaller, more focused modules?**
  _Cohesion score 0.13174603174603175 - nodes in this community are weakly interconnected._