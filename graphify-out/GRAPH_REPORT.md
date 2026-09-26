# Graph Report - mlx-lif-engine  (2026-09-25)

## Corpus Check
- 48 files · ~169,011 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 789 nodes · 1456 edges · 34 communities
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 116 edges (avg confidence: 0.89)
- Token cost: 625,510 input · 0 output

## Community Hubs (Navigation)
- Core Pack and Experiments
- Engine Parity Tests
- Pack Compilers
- Benchmark and Control Demo
- Brian2 Validation
- Datasets and Demo
- Activity Film Rendering
- Names and Pack Audit
- Stimulus Survey
- Sparse Metal Lane
- Experiment Tests
- Spike Event Recording
- Brian2 Benchmark
- Notebook Validation
- Stimulus Tests
- Spike Record Tests
- flyBrain Stimulus Tests
- Pack Audit Tests
- Activity Film Tests
- MLX Performance Notes
- Benchmark Tests
- Notebook Validation Tests
- Fused Lane and flyBrain
- Shuffle Pack Tests
- Names Tests
- Engine Argument Tests
- Dense Chunked Lane
- Stimulus Survey Tests
- Brian2 Parity Tests
- Control Demo Tests
- Pack Compiler Tests
- Edge Split Tuning
- metal_kernel Usage Traps
- Deterministic Integer Atomics

## God Nodes (most connected - your core abstractions)
1. `Pack` - 24 edges
2. `CheckLog` - 21 edges
3. `run()` - 18 edges
4. `run_exp()` - 18 edges
5. `run_lane()` - 16 edges
6. `run()` - 15 edges
7. `run()` - 15 edges
8. `main()` - 14 edges
9. `sha256_file()` - 13 edges
10. `drosophila-brain-mlx` - 13 edges

## Surprising Connections (you probably didn't know these)
- `soma_positions()` --shares_data_with--> `MaleCNS Annotation Table (somaLocation, type, instance)`  [INFERRED]
  src/lif/activity_film.py → ATTRIBUTION.md
- `draw_text()` --implements--> `Time Label and Progress Bar (10 ms to 1000 ms)`  [INFERRED]
  src/lif/activity_film.py → docs/figures/activity-film.png
- `MaleCNS Name Resolution (Instance, Else Type)` --references--> `names_only()`  [INFERRED]
  README.md → src/lif/compile_pack_malecns.py
- `Wiring, Not Degrees, Carries the Sugar Signal to MN9` --conceptually_related_to--> `The control demo: does the wiring matter? Drives the 21 right-hemisphere sugar…`  [INFERRED]
  docs/figures/control-demo.svg → src/lif/control_demo.py
- `decay_v and v0_term Constants (1 ulp Off When Printed)` --references--> `constants_f32()`  [INFERRED]
  docs/mlx-notes.md → src/lif/core.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Four Parity-Gated Engine Lanes** — src_lif_engine_naive, src_lif_engine_chunked, src_lif_engine_metal, src_lif_engine_fused, readme_parity_gate [EXTRACTED 1.00]
- **Three Findings from 22 s to 0.29 s per Biological Second** — readme_finding_dense_problem, readme_finding_load_imbalance, readme_finding_intermediate_materialisation [EXTRACTED 1.00]
- **Three Correctness Gates** — readme_parity_gate, readme_brian2_validation, readme_notebook_validation [EXTRACTED 1.00]
- **Activity Film Rendering: Spike Bins and Soma Positions to Animated PNG** — src_lif_activity_film_spike_bins, src_lif_activity_film_soma_positions, src_lif_activity_film_view, src_lif_activity_film_frame, src_lif_activity_film_apng, docs_figures_activity_film [INFERRED 0.95]
- **Controlled Real vs Shuffled Wiring Comparison** — docs_figures_control_demo_trial_protocol, docs_figures_control_demo_degree_preserving_shuffle_control, docs_figures_control_demo_real_wiring, docs_figures_control_demo_shuffled_wiring, docs_figures_control_demo_input_arrival_check [EXTRACTED 1.00]

## Communities (34 total, 0 thin omitted)

### Community 0 - "Core Pack and Experiments"
Cohesion: 0.06
Nodes (54): Hub Drive (100 Highest Out-Degree Neurons at 150 Hz), Second Stimulus Set (targets2= and rate2_hz=), initial_state(), load_pack(), make_stimulus(), make_stimulus_for(), Pack, array (+46 more)

### Community 1 - "Engine Parity Tests"
Cohesion: 0.06
Nodes (57): pack(), plain_naive(), fixture, parametrize, End-to-end checks: every lane must agree, bit for bit. Needs a compiled pack…, Two implementations of one semantics must agree: a masked copy of the edge…, Guards the parity test above against passing because nothing was silenced., A source S whose single target T has no other input, and a drive on S alone.… (+49 more)

### Community 2 - "Pack Compilers"
Cohesion: 0.09
Nodes (44): male-cns-v1.0-superclass-non-null-known-nt Materialization, Fail-Closed Pack Compilers, 24,469,412 MaleCNS Edges Match flyBrain's Count, build_csr(), check_edges(), check_id_mapping(), CheckLog, load_edges() (+36 more)

### Community 3 - "Benchmark and Control Demo"
Cohesion: 0.07
Nodes (44): philshiu/Drosophila_brain_model Upstream Repository (MIT), Upstream example.ipynb and Its Five Result Files, Upstream v630 Completeness CSV and Connectivity Parquet, Upstream model.py (Published Brian2 Model Code), Control Demo Figure (MN9, Real vs Shuffled Wiring), Binned Rate (Spikes per 20 ms Bin, in Hz), Degree-Preserving Shuffle Control, Input Arrival Check (Sugar GRNs 99.4 Hz) (+36 more)

### Community 4 - "Brian2 Validation"
Cohesion: 0.07
Nodes (36): Brian2 2.10.1 Simulator (CeCILL-2.1), Closed-Form Update Coefficients from Brian2 method='linear', Keep a CPU float64 Reference, bench/results*.json Measured Numbers, Brian2 (unless refractory) Write Shielding, Brian2 Spike-Time Gate (800-Neuron Subnetwork), Source-Major CSR Pack Format, float32 Rounding at the Strict Threshold (+28 more)

### Community 5 - "Datasets and Demo"
Cohesion: 0.08
Nodes (40): CC BY 4.0 Licence (MaleCNS), Shiu Constants Fitted to FlyWire, Not MaleCNS, FlyWire Non-Commercial Data Licence, FlyWire v630 Connectome, Implementation-Only Contribution, LB3a Water-Taste Neurons (ppk28), LB3b and LB3c Sweet-Sensing Taste Neurons (Gr64f-GAL4), MaleCNS Annotation Table (somaLocation, type, instance) (+32 more)

### Community 6 - "Activity Film Rendering"
Cohesion: 0.11
Nodes (32): Activity Film: MaleCNS Spikes at Soma Positions (Animated PNG), Palette-Indexed Animated PNG (100 Frames, 80 ms Each, 251 Colours), Orthographic View Turning Once About the CNS Long Axis, Blue Soma-Position Fog (Brain Above, Ventral Nerve Cord Below), Orange-to-White Spike Glow per 10 ms Bin, Time Label and Progress Bar (10 ms to 1000 ms), apng(), _blur() (+24 more)

### Community 7 - "Names and Pack Audit"
Cohesion: 0.09
Nodes (29): Independent Pack Audit Through a Different Code Path, annotation_rows(), content_sha256(), load(), Names, names_table(), ndarray, Path (+21 more)

### Community 8 - "Stimulus Survey"
Cohesion: 0.10
Nodes (22): Spiking Gathers at the Posterior Tip of the Ventral Nerve Cord, Course, drive_targets(), is_abdominal(), main(), run(), ndarray, Stimulus (+14 more)

### Community 9 - "Sparse Metal Lane"
Cohesion: 0.10
Nodes (25): decay_v and v0_term Constants (1 ulp Off When Printed), Keep the Exit to the One Read That Decides It, Model Constants Fixed at Compile Time, Did Not Work: Testing the Silencing Mask in the Early Exit, Silencing Mask (silenced=), constants_f32(), Model constants rounded to the precision the engines actually compute in. MLX's…, ndarray (+17 more)

### Community 10 - "Experiment Tests"
Cohesion: 0.10
Nodes (19): experiment_file(), pack(), fixture, parametrize, run_exp and rates: the published model's experiment interface on this engine.…, What the published notebook does with the file: load_exps, then get_rate with…, Evaluates the default_params literal of upstream's model.py with Brian2's…, The model constants are compiled into the coefficients; a key upstream does not… (+11 more)

### Community 11 - "Spike Event Recording"
Cohesion: 0.11
Nodes (17): range, async_eval In-Flight Work Raises Peak Memory, Recording Overhead Is a Cost per Chunk, Spike Event Recording (record=True), RuntimeError, array, ndarray, Spike-event recording: which neuron fired in which tick. The lanes count spikes… (+9 more)

### Community 12 - "Brian2 Benchmark"
Cohesion: 0.10
Nodes (20): Brian2 Code-Generation Target (cython 2.07 s vs numpy 8.38 s), ~7x Speedup over Brian2 (0.29 s vs 2.07 s per Biological Second), clean_joblib(), fixture, tools/bench_brian2.py's joblib stand-in, which decides whether upstream's…, sys.modules without joblib, restored afterwards, since stub_joblib writes to it…, model.py does `from joblib import Parallel, delayed, parallel_backend`, so a…, Testing sys.modules alone would shadow a joblib that is installed but not… (+12 more)

### Community 13 - "Notebook Validation"
Cohesion: 0.14
Nodes (21): code_cells(), compare(), Comparison, main(), prepare(), problems(), ndarray, Path (+13 more)

### Community 14 - "Stimulus Tests"
Cohesion: 0.13
Nodes (18): parametrize, What a drive must be before a lane runs it, and the two-rate drive run_exp…, Upstream would give it two Poisson inputs; this refuses rather than guess., rate * dt must be from 0 to 1. A negative or NaN rate would otherwise draw no…, The fused lane maps each neuron to one draw column and keeps the last; the…, A draw that is not bool reads differently per lane: 0.5 is half an input where…, run_exp passes upstream's default neu_exc2=[] through; np.asarray([]) is float., Upstream's poi() sets rfc = 0 ms on every neuron of neu_exc2 whatever r_poi2… (+10 more)

### Community 15 - "Spike Record Tests"
Cohesion: 0.14
Nodes (19): _nonzero_events(), ndarray, parametrize, The spike-event kernel on its own, against np.nonzero. Needs no pack., Engine tick k is Brian2's clock time k * dt; validate_brian2 checks it., bool is what the dense and sparse lanes emit per tick, uint8 the fused lane., Three chunks, the last one short, drained with the one-chunk lag the lanes use., The slot a thread gets is a race; the sorted event set must not be. (+11 more)

### Community 16 - "flyBrain Stimulus Tests"
Cohesion: 0.12
Nodes (15): _bernoulli(), ndarray, parametrize, The sugar drive: its targets, and that its draws are flyBrain's draws. No pack…, `uniform < probability` is False for all three, so an unguarded draw matrix is…, stimulus.rs's splitmix64, in plain integers., EventSchedule::bernoulli from the same file, one draw at a time., The vectorised form is the one that can drift; the scalar one is the spec. (+7 more)

### Community 17 - "Pack Audit Tests"
Cohesion: 0.18
Nodes (16): fixture, lif.verify_pack: the auditor that says a pack matches its raw sources. Pack-…, One edge's weight, nothing else: the row reconstruction is what sees it., Row sums survive this; the per-row and per-column checks are what fail., A pack and the two raw files it was supposedly compiled from., Replace one array of a written pack, its manifest hashes with it., repack(), _sources() (+8 more)

### Community 18 - "Activity Film Tests"
Cohesion: 0.17
Nodes (12): _annotations(), _chunks(), _cloud(), lif.activity_film's pure parts: soma positions in pack order, spikes per bin,…, PNG scanlines with filter 0 or 4 (Paeth), one byte per pixel, back to pixels., test_a_frame_lights_the_pixel_of_the_neuron_that_spiked_and_nothing_when_none_did(), test_half_a_turn_mirrors_the_picture_and_swaps_near_and_far(), test_soma_positions_are_in_pack_order_and_nan_where_the_table_has_none() (+4 more)

### Community 19 - "MLX Performance Notes"
Cohesion: 0.20
Nodes (16): MLX Optimisation Checklist, Pass Float Constants to Compiled Functions as Arguments, Count Bytes, Not Dispatches, 16-Op Chain vs Fused Kernel Microbenchmark (5.9x), eval() Is the Synchronisation Unit, Not the Dispatch, FMA Contraction Probe (2^20 float32 Triples), #pragma clang fp contract(off), Fused Elementwise Kernel (0.75 s to 0.29 s, 2.55x) (+8 more)

### Community 20 - "Benchmark Tests"
Cohesion: 0.19
Nodes (14): parametrize, lif.benchmark's results file must describe the run that wrote it. Needs the…, The per-run times are raw seconds; the headline figure is the best of them., --repeat 0 used to reach `ref = r` with the loop never entered and --ticks 0…, Every other entry point anchors its defaults to the repository root; this one…, --quick used to override --ticks without a word., run_benchmark(), test_a_run_length_below_one_is_refused_before_the_pack_is_read() (+6 more)

### Community 21 - "Notebook Validation Tests"
Cohesion: 0.17
Nodes (13): _comparison(), _from_counts(), _notebook(), lif.validate_notebook's parts: which notebook cells run, running them where…, nbformat 4: a string is a code cell, a ("markdown", text) pair a markdown cell., rows: (t, trial, flywire_id), one per spike, in upstream's columns., _spikes(), test_code_cells_leave_out_cells_with_shell_commands_and_keep_the_order() (+5 more)

### Community 22 - "Fused Lane and flyBrain"
Cohesion: 0.21
Nodes (13): mehrantsi/flyBrain Rust/Metal Engine (MIT), Local flyBrain Reference Build Without MuJoCo, mx.fast.metal_kernel API, FlyWire Benchmark Results (M4 Pro, 10,000 Ticks, Sugar Drive), ~22 % Margin over flyBrain Rust/Metal, Credit to flyBrain for metal_kernel, Fusion and Benchmark Setup, flyBrain MLX Lane (CSR metal_kernel, One Thread per Source Neuron), Host-Bound Wall-Clock Under CPU Load (+5 more)

### Community 23 - "Shuffle Pack Tests"
Cohesion: 0.25
Nodes (13): _csr(), lif.shuffle_pack: the pack with its wiring randomised and its degrees kept, the…, A source-major CSR whose destinations crowd onto three hubs, so a plain…, _real_pack(), _shuffled(), test_a_table_that_is_not_source_major_csr_is_refused(), test_every_neuron_keeps_its_in_and_out_degree(), test_every_source_keeps_its_counts_so_its_sign_and_out_weight() (+5 more)

### Community 24 - "Names Tests"
Cohesion: 0.30
Nodes (11): skipif, _annotations(), _pack(), lif.names: the cell-type names sidecar of a pack and how run_exp resolves them.…, test_a_name_selects_its_instance_else_every_neuron_of_its_type(), test_a_pack_loaded_from_a_string_path_finds_its_sidecar(), test_a_sidecar_that_does_not_match_its_pack_is_refused(), test_run_exp_resolves_names_from_the_malecns_sidecar() (+3 more)

### Community 25 - "Engine Argument Tests"
Cohesion: 0.21
Nodes (11): pack(), fixture, parametrize, The lanes' performance knobs must refuse a value they cannot honour. No pack…, 0 -> 1 -> 2, one contact each, as core.load_pack would hand it over., 0 used to dispatch zero threads and return the zero init_value in silence., min(chunk, n - t) is 0 for these, so the tick cursor never advances., stim() (+3 more)

### Community 26 - "Dense Chunked Lane"
Cohesion: 0.25
Nodes (10): Did Not Work: Atomic-Counter Compaction, Find Out What You Are Bound By, Chunked Steps with mx.async_eval, Lazy Graphs Cannot Express Data-Dependent Output Sizes, Dense Dispatch with Early Exits, .at[].add() Cannot Skip Work, Vary the Work, Not the Step Count, Finding 1: Dense Is the Problem, Not Synchronisation (+2 more)

### Community 27 - "Stimulus Survey Tests"
Cohesion: 0.22
Nodes (5): lif.stimulus_survey's parts: regions from the annotation table, a stimulus…, make_stimulus_for draws one column per target in the order given, so sorting…, _stimulus(), test_drive_targets_keep_run_exps_order_so_the_inputs_are_those_of_its_trials(), test_switch_off_ends_the_input_after_a_tick_and_keeps_the_rest()

### Community 28 - "Brian2 Parity Tests"
Cohesion: 0.36
Nodes (7): _compare(), parametrize, Spike times against Brian2: the correctness gate that can fail when all lanes…, Every (neuron, time) Brian2's SpikeMonitor records, exactly, with engine tick k…, test_fused_spike_times_are_within_float32_rounding(), test_fused_spike_times_equal_brian2_until_float32_moves_one(), test_ref64_spike_times_equal_brian2()

### Community 29 - "Control Demo Tests"
Cohesion: 0.39
Nodes (6): _figure(), _heights(), lif.control_demo's pure parts: a rate over time from spike times, and the SVG…, _row(), test_a_silent_row_still_has_an_axis(), test_the_figure_has_one_bar_per_bin_and_a_y_axis_shared_within_each_row()

### Community 30 - "Pack Compiler Tests"
Cohesion: 0.39
Nodes (7): lif.compile_pack.write_pack: what it is allowed to overwrite. Pack-free, on a…, The mistyped --out: a directory full of packs is not itself a pack., test_it_refuses_a_directory_that_is_not_a_pack(), test_it_refuses_a_directory_whose_manifest_is_not_a_pack_manifest(), test_it_replaces_a_pack_in_place(), test_it_writes_a_pack_that_loads_back(), _write()

### Community 31 - "Edge Split Tuning"
Cohesion: 0.47
Nodes (6): Propagation Cost per K (2, 21 and 107 Active Rows), Tunable Unroll K for Skewed CSR Rows, edge_split Knob (Threads per Neuron Edge List), edge_split End-to-End Sweep (K=1 to 16), Finding 2: Out-Degree Load Imbalance, No Host Readback of Spike Counts

### Community 32 - "metal_kernel Usage Traps"
Cohesion: 0.40
Nodes (5): atomic_outputs=True Makes Every Output Atomic, csr_propagate Kernel Example, Cache Kernel Objects and Template Their Source, mx.fast.metal_kernel Traps, _kernel_for()

### Community 33 - "Deterministic Integer Atomics"
Cohesion: 0.40
Nodes (4): Integer Atomics Are Deterministic, Float Atomics Are Not, Sort an Atomically Compacted Set on the Host, int32 Contact-Count Accumulation, Cross-Lane Parity Gate (Spike-Count SHA-256)

## Knowledge Gaps
- **14 isolated node(s):** `mlx-lif-engine`, `Stimulus`, `setup_flybrain_reference.sh script`, `PATH`, `ORN_DM1_R Olfactory Neurons (Smell Stimulus)` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 285 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Cross-Lane Parity Gate (Spike-Count SHA-256)` connect `Deterministic Integer Atomics` to `Engine Parity Tests`, `Datasets and Demo`, `Names and Pack Audit`, `Sparse Metal Lane`, `Spike Event Recording`, `MLX Performance Notes`, `Fused Lane and flyBrain`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `~7x Speedup over Brian2 (0.29 s vs 2.07 s per Biological Second)` connect `Brian2 Benchmark` to `Spike Event Recording`, `Brian2 Validation`, `Datasets and Demo`, `Fused Lane and flyBrain`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Why does `Orthographic View Turning Once About the CNS Long Axis` connect `Activity Film Rendering` to `Activity Film Tests`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `CheckLog` (e.g. with `Fail-Closed Pack Compilers` and `load_edges()`) actually correct?**
  _`CheckLog` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlx-lif-engine`, `Stimulus`, `setup_flybrain_reference.sh script` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Core Pack and Experiments` be split into smaller, more focused modules?**
  _Cohesion score 0.05536723163841808 - nodes in this community are weakly interconnected._
- **Should `Engine Parity Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.05593220338983051 - nodes in this community are weakly interconnected._