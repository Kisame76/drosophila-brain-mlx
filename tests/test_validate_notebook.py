"""lif.validate_notebook's parts: which notebook cells run, running them where
`model` is this engine, spike counts per trial, and the comparison and its gate.
The validation itself needs upstream's files and the FlyWire pack and is
python -m lif.validate_notebook."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lif import experiment, validate_notebook

REPO = Path(__file__).resolve().parents[1]


def _notebook(*cells):
    """nbformat 4: a string is a code cell, a ("markdown", text) pair a markdown cell."""
    out = []
    for cell in cells:
        kind, text = cell if isinstance(cell, tuple) else ("code", cell)
        out.append({"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)})
    return {"cells": out, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def test_code_cells_leave_out_cells_with_shell_commands_and_keep_the_order():
    nb = _notebook("x = 1\n", ("markdown", "# Title\n"),
                   "# only on Colab\n!pip install brian2\n%cd somewhere\n",
                   "if x != 2:\n    y = x % 2\n")
    nb["cells"][0]["source"] = "x = 1\n"   # nbformat allows one string as well as a list
    cells, skipped = validate_notebook.code_cells(nb)
    assert cells == [(0, "x = 1\n"), (3, "if x != 2:\n    y = x % 2\n")]
    assert skipped == [2]


def _upstream_files(tmp_path):
    ref, raw = tmp_path / "ref", tmp_path / "raw"
    ref.mkdir()
    raw.mkdir()
    (ref / "utils.py").write_text("WHERE = 'upstream utils'\n")
    for name in validate_notebook.DATA_FILES.values():
        (raw / name).write_text(name)
    return ref, raw


def test_the_notebook_runs_where_model_is_this_engine_and_leaves_nothing_behind(
        tmp_path, monkeypatch):
    ref, raw = _upstream_files(tmp_path)
    work = tmp_path / "work"
    validate_notebook.prepare(work, ref, raw)
    for upstream_name, name in validate_notebook.DATA_FILES.items():
        assert (work / upstream_name).resolve() == (raw / name).resolve()
    stale = types.ModuleType("model")
    monkeypatch.setitem(sys.modules, "model", stale)
    utils_before = sys.modules.get("utils")
    cwd, path = Path.cwd(), list(sys.path)

    ns = validate_notebook.run_cells([
        (1, ("from model import run_exp\nfrom model import default_params as params\n"
             "import utils as utl\n")),
        (2, ("with open('./2023_03_23_completeness_630_final.csv') as f:\n    first = f.read()\n"
             "open('written.txt', 'w').write('here')\n")),
    ], work)

    assert ns["run_exp"] is experiment.run_exp and ns["params"] is experiment.default_params
    assert ns["utl"].WHERE == "upstream utils"
    assert ns["first"] == "completeness_630.csv"
    assert (work / "written.txt").read_text() == "here"
    assert Path.cwd() == cwd and sys.path == path
    assert sys.modules["model"] is stale and sys.modules.get("utils") is utils_before


def test_a_failing_cell_is_named_and_the_directory_is_restored(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    cwd = Path.cwd()
    with pytest.raises(ZeroDivisionError) as info:
        validate_notebook.run_cells([(4, "a = 1\n"), (7, "b = a / 0\n")], work)
    assert info.value.__notes__ == ["in code cell 7 of the notebook"]
    assert Path.cwd() == cwd


def _spikes(path, rows):
    """rows: (t, trial, flywire_id), one per spike, in upstream's columns."""
    t, trial, neuron = (list(c) for c in zip(*rows, strict=True))
    pq.write_table(pa.table({
        "t": pa.array(t, pa.float64()), "trial": pa.array(trial, pa.int64()),
        "flywire_id": pa.array(neuron, pa.int64()), "exp_name": pa.array(["e"] * len(t)),
    }), path)
    return path


def _from_counts(path, counts):
    return _spikes(path, [(0.1, trial, neuron) for neuron, per_trial in counts.items()
                          for trial, n in enumerate(per_trial) for _ in range(n)])


def test_trial_counts_count_each_neurons_spikes_in_each_trial(tmp_path):
    path = _spikes(tmp_path / "a.parquet", [(0.1, 0, 7), (0.2, 0, 7), (0.1, 2, 7), (0.5, 1, 3)])
    ids, counts = validate_notebook.trial_counts(path, n_run=3)
    assert ids.tolist() == [3, 7]
    assert counts.tolist() == [[0, 1, 0], [2, 0, 1]]
    with pytest.raises(ValueError, match="trials outside 0 to 1"):
        validate_notebook.trial_counts(path, n_run=2)


def test_compare_gives_rates_and_the_z_of_their_difference_over_trials(tmp_path):
    a = _from_counts(tmp_path / "a.parquet", {1: [1, 2, 3], 3: [2, 2, 2], 4: [1, 1, 1]})
    b = _from_counts(tmp_path / "b.parquet",
                     {1: [4, 5, 6], 2: [0, 0, 1], 3: [2, 2, 2], 4: [2, 2, 2]})
    c = validate_notebook.compare(a, b, n_run=3, t_run=0.5)

    assert c.ids.tolist() == [1, 2, 3, 4]
    np.testing.assert_allclose(c.rate_a, [4.0, 0.0, 4.0, 2.0])
    np.testing.assert_allclose(c.rate_b, [10.0, 2 / 3, 4.0, 4.0])
    # neuron 1: 3 spikes more per trial, variances 1 and 1; neuron 2 only in b
    np.testing.assert_allclose(c.z[:3], [3 / np.sqrt(2 / 3), 1.0, 0.0])
    assert c.z[3] == np.inf, "constant in both trials and different"
    assert (c.spikes_a, c.spikes_b) == pytest.approx((5.0, 28 / 3))
    assert c.z_spikes == pytest.approx((28 / 3 - 5) / np.sqrt((1 + 7 / 3) / 3))
    assert c.of(1) == pytest.approx((4.0, 10.0, 3 / np.sqrt(2 / 3)))
    assert c.of(99) == (0.0, 0.0, 0.0)

    same = validate_notebook.compare(a, a, n_run=3, t_run=0.5)
    assert not same.z.any() and same.z_spikes == 0


def _comparison(z, z_spikes=0.0):
    z = np.asarray(z, dtype=np.float64)
    return validate_notebook.Comparison(
        ids=np.arange(z.size, dtype=np.int64), rate_a=np.full(z.size, 5.0),
        rate_b=np.full(z.size, 6.0), z=z, spikes_a=100.0, spikes_b=101.0, z_spikes=z_spikes)


def test_problems_flag_the_spikes_per_trial_the_readout_neuron_and_too_many_outliers():
    z = np.zeros(200)
    z[1:3] = [5.0, -4.5]   # 2 of 200 neurons beyond |z| 4, which is 1 %
    assert validate_notebook.problems(_comparison(z), neuron=0) == []

    z[0], z[3] = np.nan, 4.1
    assert validate_notebook.problems(_comparison(z, z_spikes=-4.0), neuron=0) == [
        "spikes per trial 100.0 and 101.0, z -4.00",
        "neuron 0 at 5.00 and 6.00 Hz, z +nan",
        "3 of 200 neurons differ by |z| > 4",
    ]


def test_the_published_experiments_are_the_files_fetch_upstream_downloads():
    script = (REPO / "tools" / "fetch_upstream.sh").read_text()
    assert set(re.findall(r"results/example/([\w-]+)\.parquet", script)) == set(
        validate_notebook.PUBLISHED)
    for stem, (_, silenced) in validate_notebook.PUBLISHED.items():
        assert stem.startswith("sugarR")
        if silenced:
            assert stem == f"sugarR-{silenced[0]}"
