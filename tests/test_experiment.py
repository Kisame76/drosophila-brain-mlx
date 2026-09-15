"""run_exp and rates: the published model's experiment interface on this engine.

Needs the FlyWire pack. The comparisons with upstream also need data/ref and
data/raw (tools/fetch_upstream.sh), pandas and Brian2, and skip without them.
Trials are 100 ms, so the file runs in seconds.
"""

from __future__ import annotations

import ast
import importlib.util
import textwrap

import numpy as np
import pyarrow.parquet as pq
import pytest

from lif import core, experiment, spike_record, stimulus_flybrain

pytestmark = pytest.mark.skipif(
    not (core.PACK_DIR / "manifest.json").exists(),
    reason="no compiled pack; run tools/fetch_upstream.sh then python -m lif.compile_pack",
)

REPO = core.PACK_DIR.parents[2]
SUGAR = stimulus_flybrain.RIGHT_SUGAR_GRN_IDS
SHORT = {"t_run": 0.1, "n_run": 3}   # 1,000 ticks
SEED = 11


@pytest.fixture(scope="module")
def pack():
    return core.load_pack()


@pytest.fixture(scope="module")
def sets(pack):
    """neu_exc the sugar GRNs; neu_exc2 the 5 highest out-degree neurons that are not
    sugar GRNs; neu_slnc the first sugar GRN, so it is driven and silenced at once."""
    sugar = stimulus_flybrain.targets(pack.neuron_ids)
    others = np.setdiff1d(np.arange(pack.n_neurons), sugar)
    exc2 = others[np.lexsort((others, -pack.out_degree[others]))][:5]
    return list(SUGAR), [int(i) for i in pack.neuron_ids[exc2]], [SUGAR[0]]


@pytest.fixture(scope="module")
def experiment_file(pack, sets, tmp_path_factory):
    exc, exc2, slnc = sets
    params = {**experiment.default_params, **SHORT, "r_poi2": 50.0}
    return experiment.run_exp("sugar", exc, tmp_path_factory.mktemp("results"),
                              params=params, neu_slnc=slnc, neu_exc2=exc2, pack=pack,
                              seed=SEED)


def test_each_trial_is_the_engine_run_with_seed_plus_trial(pack, sets, experiment_file):
    """Trial n is the fused lane on make_stimulus_for(..., seed + n) with the
    silencing mask, and the file holds exactly its spikes in upstream's row order."""
    from lif import engine_fused

    exc, exc2, slnc = sets
    index = {int(n): i for i, n in enumerate(pack.neuron_ids)}
    mask = np.zeros(pack.n_neurons, dtype=bool)
    mask[[index[i] for i in slnc]] = True
    table = pq.read_table(experiment_file)
    trial = table["trial"].to_numpy()

    for n in range(SHORT["n_run"]):
        stim = core.make_stimulus_for(pack, [index[i] for i in exc], 150.0, 1_000, SEED + n,
                                      targets2=[index[i] for i in exc2], rate2_hz=50.0)
        r = engine_fused.run(pack, stim, silenced=mask, warmup=0, edge_split=1, record=True)
        ev = r.events[np.lexsort((r.events[:, 0], r.events[:, 1]))]   # neuron, then tick
        rows = trial == n
        assert rows.sum() == len(ev) > 0
        assert np.array_equal(table["flywire_id"].to_numpy()[rows], pack.neuron_ids[ev[:, 1]])
        assert np.array_equal(table["t"].to_numpy()[rows], spike_record.tick_to_seconds(ev[:, 0]))
        if n == 0:
            assert (r.spike_counts[[index[i] for i in exc2]] > 0).any(), "neu_exc2 must fire"
            plain = engine_fused.run(pack, stim, warmup=0, edge_split=1)
            assert plain.counts_sha256() != r.counts_sha256(), "neu_slnc must change the run"


def test_the_file_has_upstreams_columns_and_records_the_experiment(sets, experiment_file):
    table = pq.read_table(experiment_file)
    assert table.schema.names == ["t", "trial", "flywire_id", "exp_name"]
    assert [str(f.type) for f in table.schema] == ["double", "int64", "int64", "string"]
    assert set(table["exp_name"].to_pylist()) == {"sugar"}
    order = np.lexsort((table["t"].to_numpy(), table["flywire_id"].to_numpy(),
                        table["trial"].to_numpy()))
    assert np.array_equal(order, np.arange(table.num_rows)), "sorted by trial, neuron, time"
    column = pq.ParquetFile(experiment_file).metadata.row_group(0).column(0)
    assert column.compression == "BROTLI"

    exc, exc2, slnc = sets
    meta = experiment.metadata(experiment_file)
    assert meta["dataset"] == "flywire-v630" and meta["engine"] == "fused"
    assert (meta["t_run_s"], meta["n_run"], meta["seed"], meta["dt_ms"]) == (0.1, 3, SEED, core.DT)
    assert (meta["r_poi_hz"], meta["r_poi2_hz"]) == (150.0, 50.0)
    assert (meta["neu_exc"], meta["neu_exc2"], meta["neu_slnc"]) == (exc, exc2, slnc)


def _upstream(name):
    path = REPO / "data" / "ref" / f"{name}.py"
    if not path.exists():
        pytest.skip(f"no {path}; run tools/fetch_upstream.sh")
    return path


def test_rates_equal_upstreams_get_rate(experiment_file):
    """What the published notebook does with the file: load_exps, then get_rate with
    t_run and n_run from params, which here are plain seconds and a count."""
    pytest.importorskip("pandas")
    spec = importlib.util.spec_from_file_location("upstream_utils", _upstream("utils"))
    utl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utl)
    params = {**experiment.default_params, **SHORT}
    flyid2name = {f: f"sugar_{i + 1}" for i, f in enumerate(SUGAR)}

    df = utl.load_exps([str(experiment_file)])
    df_rate, df_std = utl.get_rate(df, t_run=params["t_run"], n_run=params["n_run"],
                                   flyid2name=flyid2name)
    got = experiment.rates(experiment_file, names=flyid2name)

    assert got["flywire_id"].to_pylist() == df_rate.index.tolist()
    assert got["name"].to_pylist() == df_rate["name"].tolist()
    assert np.array_equal(got["rate_hz"].to_numpy(), df_rate["sugar"].to_numpy())
    assert np.array_equal(got["std_hz"].to_numpy(), df_std["sugar"].to_numpy())


def test_default_params_are_upstreams_in_si_units():
    """Evaluates the default_params literal of upstream's model.py with Brian2's
    units. Plain numbers here are what float() of each quantity gives."""
    brian2 = pytest.importorskip("brian2")
    path = _upstream("model")
    tree = ast.parse(path.read_text())
    literal = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", None) == "default_params")
    upstream = eval(compile(ast.Expression(literal), str(path), "eval"),
                    {"ms": brian2.ms, "mV": brian2.mV, "Hz": brian2.Hz,
                     "dedent": textwrap.dedent})

    assert experiment.default_params.keys() == upstream.keys()
    for key, value in upstream.items():
        ours = experiment.default_params[key]
        if isinstance(value, str):
            assert ours.split() == value.split(), key
        else:
            assert ours == pytest.approx(float(value), rel=1e-12), key


def test_upstreams_config_runs_and_its_files_are_checked(pack, tmp_path):
    comp = REPO / "data" / "raw" / "completeness_630.csv"
    con = REPO / "data" / "raw" / "connectivity_630.parquet"
    if not (comp.exists() and con.exists()):
        pytest.skip("no data/raw model files; run tools/fetch_upstream.sh")
    config = {"path_res": tmp_path / "res", "path_comp": comp, "path_con": con, "n_proc": -1}

    path = experiment.run_exp(exp_name="sugarR", neu_exc=list(SUGAR), params=dict(SHORT),
                              **config)
    assert path == tmp_path / "res" / "sugarR.parquet" and path.exists()

    with pytest.raises(ValueError, match="sha256"):
        experiment.run_exp(exp_name="swapped", neu_exc=list(SUGAR), params=dict(SHORT),
                           pack=pack, **{**config, "path_comp": con})
    assert not (tmp_path / "res" / "swapped.parquet").exists()


def test_an_existing_file_is_skipped_unless_force_overwrite(pack, tmp_path, capsys):
    kw = {"params": {**SHORT, "n_run": 1}, "pack": pack}
    path = experiment.run_exp("x", list(SUGAR), tmp_path, **kw)
    before = path.read_bytes()
    capsys.readouterr()

    assert experiment.run_exp("x", list(SUGAR), tmp_path, seed=1, **kw) == path
    assert capsys.readouterr().out == (
        f">>> Skipping experiment x because {path} exists and force_overwrite = False\n")
    assert path.read_bytes() == before

    experiment.run_exp("x", list(SUGAR), tmp_path, seed=1, force_overwrite=True, **kw)
    assert path.read_bytes() != before


def test_names_resolve_through_flyid2name(pack, tmp_path):
    flyid2name = {f: f"sugar_{i + 1}" for i, f in enumerate(SUGAR)}
    kw = {"params": {**SHORT, "n_run": 1}, "pack": pack}
    by_id = pq.read_table(experiment.run_exp("ids", list(SUGAR), tmp_path, **kw))
    by_name = pq.read_table(experiment.run_exp(
        "names", [flyid2name[f] for f in SUGAR], tmp_path, names=flyid2name, **kw))
    assert by_id["t"].equals(by_name["t"])
    assert by_id["flywire_id"].equals(by_name["flywire_id"])


def test_unknown_neurons_are_named_all_at_once(pack, tmp_path):
    with pytest.raises(ValueError, match=r"\[1, 2\]"):
        experiment.run_exp("x", [SUGAR[0], 1, 2], tmp_path, pack=pack)
    assert not list(tmp_path.iterdir())


A, B = sorted(SUGAR[:2])


@pytest.mark.parametrize("kw, match", [
    ({"neu_exc": ["sugar_1"]}, "sugar_1"),
    ({"neu_exc": ["x"], "names": {A: "x", B: "x"}}, rf"\[{A}, {B}\]"),
    ({"neu_exc": [A, B, A]}, rf"\[{A}\]"),
    ({"neu_exc": [A, B], "neu_exc2": [B]}, rf"\[{B}\]"),
    ({"neu_exc": [A], "neu_slnc": [3]}, r"\[3\]"),
], ids=["name without names", "name of two neurons", "neuron twice", "neuron in both sets",
        "unknown silenced neuron"])
def test_neurons_that_cannot_be_resolved_once_raise(pack, tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        experiment.run_exp("x", path_res=tmp_path, pack=pack, **kw)


@pytest.mark.parametrize("change, error", [
    ({"v_th": -0.040}, NotImplementedError),
    ({"tau": 0.006}, NotImplementedError),
    ({"eq_rst": "v = v_rst"}, NotImplementedError),
    ({"r_poi3": 10.0}, ValueError),
    ({"t_run": 0.00015}, ValueError),
    ({"n_run": 0}, ValueError),
    ({"n_run": 2.5}, ValueError),
])
def test_params_this_engine_cannot_run_raise(pack, tmp_path, change, error):
    """The model constants are compiled into the coefficients; a key upstream does
    not know is more likely a typo than an intent; t_run must be whole ticks."""
    (key,) = change
    with pytest.raises(error, match=key):
        experiment.run_exp("x", list(SUGAR), tmp_path, params={**SHORT, **change}, pack=pack)
    assert not list(tmp_path.iterdir())


def test_params_take_brian2_quantities(pack, tmp_path):
    """The published notebook sets params['r_poi'] = 100 * Hz before run_exp."""
    brian2 = pytest.importorskip("brian2")
    params = dict(experiment.default_params)
    params.update(t_run=100 * brian2.ms, n_run=1, r_poi=100 * brian2.Hz, v_th=-45 * brian2.mV)
    meta = experiment.metadata(experiment.run_exp("q", list(SUGAR), tmp_path, params=params,
                                                  pack=pack))
    assert meta["t_run_s"] == pytest.approx(0.1) and meta["r_poi_hz"] == 100.0

    with pytest.raises(ValueError, match="t_run"):
        experiment.run_exp("wrong", list(SUGAR), tmp_path, pack=pack,
                           params={**SHORT, "t_run": 100 * brian2.Hz})
