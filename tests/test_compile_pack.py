"""lif.compile_pack.write_pack: what it is allowed to overwrite.

Pack-free, on a four-neuron CSR. The compilers themselves need the multi-GB
upstream tables and cannot run here; this covers the one step of theirs that
touches a directory the user names on the command line.

write_pack replaces its output directory wholesale, and both compilers hand it
`--out` unexamined. `--out data/pack` instead of data/pack/v630 is one keystroke
away and used to delete every compiled pack under it, print ALL CHECKS PASSED and
leave four .npy files behind. shuffle_pack has guarded against exactly this since
it was written (test_shuffle_pack.py::test_it_never_replaces_a_pack_that_is_not_shuffled);
this is the same guard where all three callers pass through it.
"""

from __future__ import annotations

import numpy as np
import pytest

from lif import compile_pack, core

N = 4


def _write(path, dataset="tiny"):
    row_ptr = np.array([0, 1, 2, 3, 3], dtype=np.int32)
    arrays = {"neuron_ids": np.arange(N, dtype=np.int64) + 720575940000000000,
              "row_ptr": row_ptr,
              "destinations": np.array([1, 2, 3], dtype=np.int32),
              "signed_counts": np.array([7, -7, 7], dtype=np.int32)}
    sources = {"connectivity_parquet": {"path": "connectivity.parquet", "sha256": "0" * 64}}
    return compile_pack.write_pack(path, arrays, {"self_loops": 0}, sources,
                                   compile_pack.CheckLog(), dataset=dataset)


def test_it_writes_a_pack_that_loads_back(tmp_path):
    manifest = _write(tmp_path / "tiny")
    assert core.load_pack(tmp_path / "tiny").manifest == manifest


def test_it_replaces_a_pack_in_place(tmp_path):
    _write(tmp_path / "tiny")
    (tmp_path / "tiny" / "stale.npy").write_bytes(b"left over from an older layout")
    _write(tmp_path / "tiny", dataset="tiny-again")
    assert core.load_pack(tmp_path / "tiny").manifest["dataset"] == "tiny-again"
    assert not (tmp_path / "tiny" / "stale.npy").exists()


def test_it_refuses_a_directory_that_is_not_a_pack(tmp_path):
    """The mistyped --out: a directory full of packs is not itself a pack."""
    _write(tmp_path / "packs" / "v630")
    keep = tmp_path / "packs" / "notes.txt"
    keep.write_text("not a pack")
    with pytest.raises(ValueError, match="not a pack"):
        _write(tmp_path / "packs")
    assert keep.read_text() == "not a pack"
    core.load_pack(tmp_path / "packs" / "v630")


def test_it_refuses_a_directory_whose_manifest_is_not_a_pack_manifest(tmp_path):
    out = tmp_path / "notes"
    out.mkdir()
    (out / "manifest.json").write_text('{"pack_format": "something-else"}')
    with pytest.raises(ValueError, match="not a pack"):
        _write(out)
    (out / "manifest.json").write_text("{ this is not json")
    with pytest.raises(ValueError, match="not a pack"):
        _write(out)
