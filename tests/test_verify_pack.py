"""lif.verify_pack: the auditor that says a pack matches its raw sources.

Pack-free, on an eight-neuron FlyWire-shaped pack with a CSV and a parquet built
to match it. The real packs need the multi-GB upstream tables; what is checked
here is that the verifier fails when it should, which nothing else in the
repository does -- every other pack gate is either the verifier itself or a
compiler check that shares the pipeline it is meant to police.

A verifier is only worth its exit code if its checks can fail. Each test below
corrupts exactly one thing and asserts that `main()` reports it: a check that
silently no-ops would pass every one of them on a correct pack and still return
0 on a broken one.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lif import compile_pack, verify_pack

N = 8
# 0 -> 1 (+3), 0 -> 2 (-2), 1 -> 2 (+5), 1 -> 1 (+4, a self-loop), 4 -> 7 (+1).
# Sources 2, 3, 5, 6, 7 have no outgoing edges, so the empty-row case is sampled.
ROW_PTR = np.array([0, 2, 4, 4, 4, 5, 5, 5, 5], dtype=np.int32)
DEST = np.array([1, 2, 1, 2, 7], dtype=np.int32)
CNT = np.array([3, -2, 4, 5, 1], dtype=np.int32)
IDS = (np.arange(N, dtype=np.int64) + 720575940000000000)


def _sources(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": verify_pack.sha256_file(path)}


@pytest.fixture
def written(tmp_path):
    """A pack and the two raw files it was supposedly compiled from."""
    csv = tmp_path / "completeness.csv"
    csv.write_text("root_id,nt_type\n" + "".join(f"{i},ACH\n" for i in IDS.tolist()))

    src = np.repeat(np.arange(N, dtype=np.int64), np.diff(ROW_PTR.astype(np.int64)))
    parquet = tmp_path / "connectivity.parquet"
    pq.write_table(pa.table({
        verify_pack.COL_PRE: pa.array(src, pa.int64()),
        verify_pack.COL_POST: pa.array(DEST.astype(np.int64), pa.int64()),
        verify_pack.COL_W: pa.array(CNT.astype(np.float64), pa.float64()),
    }), parquet)

    pack = tmp_path / "pack"
    compile_pack.write_pack(
        pack,
        {"neuron_ids": IDS, "row_ptr": ROW_PTR, "destinations": DEST, "signed_counts": CNT},
        {"edges_out": int(DEST.size), "excitatory_edges": int((CNT > 0).sum()),
         "inhibitory_edges": int((CNT < 0).sum()),
         "self_loops": int((src == DEST.astype(np.int64)).sum()),
         "max_out_degree": int(np.diff(ROW_PTR).max()),
         "total_signed_count": int(CNT.sum())},
        {"completeness_csv": _sources(csv), "connectivity_parquet": _sources(parquet)},
        compile_pack.CheckLog(),
    )
    return pack, csv, parquet


def verify(written, monkeypatch, capsys):
    pack, csv, parquet = written
    monkeypatch.setattr(sys, "argv", ["lif.verify_pack", "--pack", str(pack),
                                      "--completeness", str(csv),
                                      "--connectivity", str(parquet)])
    code = verify_pack.main()
    return code, capsys.readouterr().out


def repack(written, name, arr):
    """Replace one array of a written pack, its manifest hashes with it."""
    pack, _, _ = written
    np.save(pack / f"{name}.npy", arr, allow_pickle=False)
    manifest = json.loads((pack / "manifest.json").read_text())
    manifest["arrays"][name]["sha256"] = compile_pack.sha256_array(arr)
    manifest["arrays"][name]["file_sha256"] = verify_pack.sha256_file(pack / f"{name}.npy")
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def test_a_pack_that_matches_its_sources_verifies(written, monkeypatch, capsys):
    code, out = verify(written, monkeypatch, capsys)
    assert code == 0, out
    assert "VERIFIED" in out
    assert "FAIL" not in out
    # Not vacuously: the row reconstruction must actually have compared edges,
    # and the sample must have reached every row. Nine of eight, and seven edges
    # of five, because the busiest source is appended to the sample a second time.
    assert "9 rows reconstructed independently" in out
    assert "7 edges compared; 0 mismatched rows" in out


def test_a_count_that_disagrees_with_the_raw_table_is_caught(written, monkeypatch, capsys):
    """One edge's weight, nothing else: the row reconstruction is what sees it."""
    repack(written, "signed_counts", np.array([3, -2, 4, 6, 1], dtype=np.int32))
    code, out = verify(written, monkeypatch, capsys)
    assert code == 1
    assert "FAIL  total signed count conserved" in out
    assert "mismatched rows" in out and "1 mismatched rows" in out


def test_a_destination_moved_between_two_rows_is_caught(written, monkeypatch, capsys):
    """Row sums survive this; the per-row and per-column checks are what fail."""
    repack(written, "row_ptr", np.array([0, 3, 4, 4, 4, 5, 5, 5, 5], dtype=np.int32))
    code, out = verify(written, monkeypatch, capsys)
    assert code == 1
    assert "FAIL  per-neuron out-weight matches raw" in out


def test_an_array_that_no_longer_matches_its_manifest_hash_is_caught(written, monkeypatch,
                                                                     capsys):
    pack, _, _ = written
    np.save(pack / "destinations.npy", np.array([1, 2, 1, 2, 6], dtype=np.int32),
            allow_pickle=False)
    code, out = verify(written, monkeypatch, capsys)
    assert code == 1
    assert "FAIL  destinations file sha256" in out
    assert "FAIL  destinations buffer sha256" in out


def test_a_raw_source_replaced_since_the_compile_is_caught(written, monkeypatch, capsys):
    pack, csv, _ = written
    csv.write_text(csv.read_text().replace("ACH", "GABA"))
    code, out = verify(written, monkeypatch, capsys)
    assert code == 1
    assert "FAIL  source completeness_csv" in out


def test_a_pack_whose_neurons_are_not_the_csv_neurons_is_caught(written, monkeypatch, capsys):
    ids = IDS.copy()
    ids[3] += 1
    repack(written, "neuron_ids", ids)
    code, out = verify(written, monkeypatch, capsys)
    assert code == 1
    assert "FAIL  ids match CSV text exactly" in out


def test_an_unknown_dataset_is_refused_rather_than_read_as_flywire(written):
    with pytest.raises(SystemExit, match="no verifier for dataset"):
        verify_pack.reader_for("some-other-connectome")
