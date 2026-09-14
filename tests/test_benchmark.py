"""lif.benchmark's results file must describe the run that wrote it.

Needs the compiled FlyWire pack, like test_engines.py, and skips cleanly without
it. Ten ticks per run: only the metadata is checked here, not the timings.
"""

from __future__ import annotations

import json
import sys

import pytest

from lif import benchmark, core

pytestmark = pytest.mark.skipif(
    not (core.PACK_DIR / "manifest.json").exists(),
    reason="no compiled pack; run tools/fetch_upstream.sh then python -m lif.compile_pack",
)


@pytest.mark.parametrize("stimulus, label", [
    ("sugar", "right_sugar_grns"),
    ("hubs", "top_out_degree_hubs"),
])
def test_results_file_names_the_stimulus_that_ran(stimulus, label, tmp_path, monkeypatch):
    out = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", ["lif.benchmark", "--ticks", "10",
                                      "--stimulus", stimulus, "--out", str(out)])
    assert benchmark.main() == 0
    assert json.loads(out.read_text())["stimulus"] == label
