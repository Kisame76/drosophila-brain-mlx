"""tools/bench_brian2.py's joblib stand-in, which decides whether upstream's
model.py can be imported at all. The measurement itself needs the FlyWire tables
and Brian2 and is python tools/bench_brian2.py --ticks 10000.

The script lives in tools/ rather than the package, so it is imported by path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import bench_brian2


@pytest.fixture
def clean_joblib():
    """sys.modules without joblib, restored afterwards, since stub_joblib writes
    to it and a leftover stand-in would follow the other tests around."""
    saved = sys.modules.pop("joblib", None)
    yield
    sys.modules.pop("joblib", None)
    if saved is not None:
        sys.modules["joblib"] = saved


def test_the_stand_in_supplies_the_three_names_upstream_imports(clean_joblib, monkeypatch):
    """model.py does `from joblib import Parallel, delayed, parallel_backend`,
    so a stand-in missing any of the three fails the import it exists to allow."""
    monkeypatch.setattr(bench_brian2.importlib.util, "find_spec", lambda name: None)
    assert bench_brian2.stub_joblib() is True
    from joblib import Parallel, delayed, parallel_backend

    assert (Parallel, delayed, parallel_backend) == (None, None, None)


def test_an_installed_joblib_wins_over_the_stand_in(clean_joblib, monkeypatch):
    """Testing sys.modules alone would shadow a joblib that is installed but not
    imported yet, which is the whole reason find_spec is consulted."""
    monkeypatch.setattr(bench_brian2.importlib.util, "find_spec", lambda name: object())
    assert bench_brian2.stub_joblib() is False
    assert "joblib" not in sys.modules


def test_an_already_imported_joblib_is_left_alone(clean_joblib):
    sentinel = object()
    sys.modules["joblib"] = sentinel
    assert bench_brian2.stub_joblib() is False
    assert sys.modules["joblib"] is sentinel


def test_peak_mb_reports_this_process_in_megabytes():
    mb = bench_brian2.peak_mb()
    assert 1 < mb < 1e6, f"{mb} MB is not a plausible resident size"
