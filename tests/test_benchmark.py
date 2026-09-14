"""lif.benchmark's results file must describe the run that wrote it.

Needs the compiled FlyWire pack, like test_engines.py, and skips cleanly without
it. Ten ticks per run: only the metadata is checked here, not the timings.
"""

from __future__ import annotations

import json
import sys

import pytest

from lif import benchmark, core, engine_metal

pytestmark = pytest.mark.skipif(
    not (core.PACK_DIR / "manifest.json").exists(),
    reason="no compiled pack; run tools/fetch_upstream.sh then python -m lif.compile_pack",
)


def run_benchmark(tmp_path, monkeypatch, *args):
    out = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", ["lif.benchmark", *args, "--out", str(out)])
    assert benchmark.main() == 0
    return json.loads(out.read_text())


@pytest.mark.parametrize("stimulus, label", [
    ("sugar", "right_sugar_grns"),
    ("hubs", "top_out_degree_hubs"),
])
def test_results_file_names_the_stimulus_that_ran(stimulus, label, tmp_path, monkeypatch):
    results = run_benchmark(tmp_path, monkeypatch, "--ticks", "10", "--stimulus", stimulus)
    assert results["stimulus"] == label


def test_results_file_names_the_dataset(tmp_path, monkeypatch):
    manifest = json.loads((core.PACK_DIR / "manifest.json").read_text())
    results = run_benchmark(tmp_path, monkeypatch, "--ticks", "10")
    assert results["dataset"] == manifest["dataset"]


@pytest.mark.parametrize("args, metal, fused", [
    (["--stimulus", "sugar"], engine_metal.EDGE_SPLIT_SPARSE, 1),
    (["--stimulus", "hubs"], engine_metal.EDGE_SPLIT, engine_metal.EDGE_SPLIT),
    (["--stimulus", "sugar", "--edge-split", "4"], 4, 4),
], ids=["sugar", "hubs", "override"])
def test_results_file_records_the_edge_split_of_each_kernel_lane(args, metal, fused, tmp_path,
                                                                 monkeypatch):
    lanes = run_benchmark(tmp_path, monkeypatch, "--ticks", "10", *args)["lanes"]
    assert {name: lane["edge_split"] for name, lane in lanes.items()} == {
        "naive dense (eval/tick)": None,
        "chunked dense": None,
        "sparse metal kernel": metal,
        "fused, 2 dispatches": fused,
    }


def test_quick_and_ticks_cannot_be_combined(tmp_path, monkeypatch):
    """--quick used to override --ticks without a word."""
    with pytest.raises(SystemExit) as excinfo:
        run_benchmark(tmp_path, monkeypatch, "--quick", "--ticks", "10")
    assert excinfo.value.code == 2
