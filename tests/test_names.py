"""lif.names: the cell-type names sidecar of a pack and how run_exp resolves them.

The table and sidecar tests build a tiny pack and need no data. The run_exp test
needs the MaleCNS pack with its sidecar (python -m lif.compile_pack_malecns
--names-only) and skips without it.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lif import compile_pack, core, experiment, names

IDS = np.array([10, 20, 30, 40], dtype=np.int64)
MALECNS = core.PACK_DIR.parent / "male_cns_v1"


def _annotations(**change):
    columns = {
        "bodyId": pa.array([40, 30, 99, 20, 10], pa.int64()),
        "superclass": ["motor", "motor", "optic", "descending", "sensory"],
        "type": ["MN9", "MN9", "other", "DNa01", None],
        "instance": ["MN9_R", "MN9_L", "other_L", "DNa01", None],
        "class": [None, None, "x", "descending", None],
        "flywireType": [None, None, None, "DNa01", None],
    }
    columns.update(change)
    return pa.table(columns)


def _pack(path, ids=IDS):
    arrays = {"neuron_ids": ids, "row_ptr": np.zeros(ids.size + 1, np.int32),
              "destinations": np.zeros(0, np.int32), "signed_counts": np.zeros(0, np.int32)}
    compile_pack.write_pack(path, arrays, {}, {}, compile_pack.CheckLog(), dataset="tiny")
    return path


def test_the_table_lists_the_pack_neurons_in_pack_order():
    table = names.names_table(_annotations(), IDS)
    assert table.column_names == ["bodyId", "type", "instance", "class", "flywireType"]
    assert table["bodyId"].to_pylist() == [10, 20, 30, 40]
    assert table["instance"].to_pylist() == [None, "DNa01", "MN9_L", "MN9_R"]
    assert table["flywireType"].to_pylist() == [None, "DNa01", None, None]


def test_the_table_refuses_bodies_it_cannot_place_once():
    with pytest.raises(ValueError, match=r"\[50\]"):
        names.names_table(_annotations(), np.array([10, 50], dtype=np.int64))
    with pytest.raises(ValueError, match=r"more than once: \[40\]"):
        names.names_table(_annotations(bodyId=pa.array([40, 30, 99, 40, 10], pa.int64())), IDS)
    with pytest.raises(ValueError, match="flywireType"):
        names.names_table(_annotations().drop_columns(["flywireType"]), IDS)


def test_a_name_selects_its_instance_else_every_neuron_of_its_type():
    lookup = names.Names(names.names_table(_annotations(), IDS))
    assert lookup.select("MN9_R") == [3]
    assert lookup.select("MN9") == [2, 3]
    assert lookup.select("DNa01") == [1]
    assert lookup.select("no such cell") == []
    assert lookup.labels() == {20: "DNa01", 30: "MN9_L", 40: "MN9_R"}


def test_the_sidecar_is_recorded_in_the_manifest_and_loads_with_the_pack(tmp_path):
    path = _pack(tmp_path / "tiny")
    entry = names.write_names(path, names.names_table(_annotations(), IDS), source="annotations")
    pack = core.load_pack(path)
    assert pack.path == path
    assert pack.manifest["sidecars"]["names"] == entry
    assert entry["file"] == "names.parquet" and entry["rows"] == 4 and entry["source"] == "annotations"
    assert names.load(pack).select("MN9") == [2, 3]
    assert names.load(core.load_pack(_pack(tmp_path / "bare"))) is None


def test_a_sidecar_that_does_not_match_its_pack_is_refused(tmp_path):
    path = _pack(tmp_path / "tiny")
    table = names.names_table(_annotations(), IDS)
    with pytest.raises(ValueError, match="pack order"):
        names.write_names(path, table.take([1, 0, 2, 3]), source="annotations")
    names.write_names(path, table, source="annotations")
    changed = table.set_column(2, "instance", pa.array([None, "DNa01", "MN9_L", "MN9_X"]))
    pq.write_table(changed, path / "names.parquet")
    with pytest.raises(ValueError, match="sha256"):
        names.load(core.load_pack(path))


@pytest.mark.skipif(not (MALECNS / "names.parquet").exists(),
                    reason="no MaleCNS pack with names; python -m lif.compile_pack_malecns --names-only")
def test_run_exp_resolves_names_from_the_malecns_sidecar(tmp_path):
    pack = core.load_pack(MALECNS)
    kw = {"params": {"t_run": 0.001, "n_run": 1}, "pack": pack}
    one = experiment.run_exp("mn9_r", ["MN9_R"], tmp_path, **kw)
    both = experiment.run_exp("mn9", ["MN9"], tmp_path, neu_slnc=["DNa01"], **kw)
    assert experiment.metadata(one)["neu_exc"] == [16949]
    assert experiment.metadata(both)["neu_exc"] == [10331, 16949]
    assert len(experiment.metadata(both)["neu_slnc"]) == len(names.load(pack).select("DNa01")) > 0
    assert names.load(pack).labels()[16949] == "MN9_R"
    with pytest.raises(ValueError, match="no such cell"):
        experiment.run_exp("x", ["no such cell"], tmp_path, **kw)
