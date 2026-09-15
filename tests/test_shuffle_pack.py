"""lif.shuffle_pack: the pack with its wiring randomised and its degrees kept, the
control for whether the connectome's wiring matters. Pack-free, on a small CSR."""

import numpy as np
import pytest

from lif import compile_pack, core, shuffle_pack

N = 200


def _csr(seed=1):
    """A source-major CSR whose destinations crowd onto three hubs, so a plain
    permutation of the destinations repeats one within a row and the repair runs."""
    rng = np.random.default_rng(seed)
    out_degree = rng.integers(0, 12, N)
    out_degree[3] = 40
    p = np.full(N, 1.0)
    p[:3] = 20.0
    rows = [np.sort(rng.choice(N, size=d, replace=False, p=p / p.sum())) for d in out_degree]
    row_ptr = np.concatenate([[0], np.cumsum(out_degree)]).astype(np.int32)
    dst = np.concatenate(rows).astype(np.int32)
    sign = np.where(rng.random(N) < 0.3, -1, 1)
    cnt = (np.repeat(sign, out_degree) * rng.integers(1, 50, dst.size)).astype(np.int32)
    return row_ptr, dst, cnt


def _shuffled(seed=0, **kw):
    row_ptr, dst, cnt = _csr()
    return (row_ptr, dst, cnt), shuffle_pack.shuffle_edges(row_ptr, dst, cnt, seed=seed, **kw)


def test_every_neuron_keeps_its_in_and_out_degree():
    (row_ptr, dst, _), (new_dst, new_cnt, _) = _shuffled()
    assert new_dst.dtype == np.int32 and new_cnt.dtype == np.int32
    assert new_dst.size == new_cnt.size == row_ptr[-1]
    np.testing.assert_array_equal(np.bincount(new_dst, minlength=N), np.bincount(dst, minlength=N))


def test_every_source_keeps_its_counts_so_its_sign_and_out_weight():
    (row_ptr, _, cnt), (_, new_cnt, _) = _shuffled()
    for i in range(N):
        a, b = row_ptr[i], row_ptr[i + 1]
        np.testing.assert_array_equal(np.sort(new_cnt[a:b]), np.sort(cnt[a:b]))


def test_rows_ascend_strictly_so_no_parallel_edges_after_repair():
    (row_ptr, _, _), (new_dst, _, stats) = _shuffled()
    assert stats["repair_rounds"] >= 1
    same_row = np.diff(np.repeat(np.arange(N), np.diff(row_ptr))) == 0
    assert (np.diff(new_dst.astype(np.int64))[same_row] > 0).all()


def test_the_wiring_changes_and_the_seed_decides_how():
    (row_ptr, dst, _), (a, _, stats) = _shuffled(seed=0)
    _, (again, _, _) = _shuffled(seed=0)
    _, (other, _, _) = _shuffled(seed=1)
    np.testing.assert_array_equal(a, again)
    assert not np.array_equal(a, other)
    src = np.repeat(np.arange(N), np.diff(row_ptr))
    kept = set(zip(src.tolist(), dst.tolist())) & set(zip(src.tolist(), a.tolist()))
    assert stats["edges_kept"] == len(kept) < dst.size
    assert stats["self_loops"] == int((src == a).sum())


def test_repeats_left_after_the_last_round_are_refused():
    row_ptr, dst, cnt = _csr()
    with pytest.raises(ValueError, match="parallel edges left after 0 repair rounds"):
        shuffle_pack.shuffle_edges(row_ptr, dst, cnt, seed=0, max_rounds=0)


def test_a_table_that_is_not_source_major_csr_is_refused():
    row_ptr, dst, cnt = _csr()
    with pytest.raises(ValueError, match="row_ptr"):
        shuffle_pack.shuffle_edges(row_ptr + 1, dst, cnt, seed=0)
    with pytest.raises(ValueError, match="row_ptr"):
        shuffle_pack.shuffle_edges(row_ptr, dst, cnt[1:], seed=0)
    with pytest.raises(ValueError, match="neuron indices"):
        shuffle_pack.shuffle_edges(row_ptr, dst + N, cnt, seed=0)
    with pytest.raises(ValueError, match="ascend"):
        shuffle_pack.shuffle_edges(row_ptr, dst[::-1].copy(), cnt, seed=0)


def _real_pack(path):
    row_ptr, dst, cnt = _csr()
    arrays = {"neuron_ids": np.arange(N, dtype=np.int64) + 720575940000000000,
              "row_ptr": row_ptr, "destinations": dst, "signed_counts": cnt}
    sources = {"connectivity_parquet": {"path": "connectivity.parquet", "sha256": "0" * 64}}
    return compile_pack.write_pack(path, arrays, {"self_loops": 0}, sources,
                                   compile_pack.CheckLog(), dataset="tiny")


def test_the_written_pack_loads_and_says_it_is_not_the_connectome(tmp_path):
    real = _real_pack(tmp_path / "tiny")
    manifest = shuffle_pack.write_shuffled_pack(tmp_path / "tiny", tmp_path / "shuffled", seed=0)
    pack = core.load_pack(tmp_path / "shuffled")
    assert pack.manifest == manifest
    assert manifest["dataset"] == "tiny-degree-shuffled-seed0"
    assert manifest["sources"] == {}
    assert manifest["shuffle"]["connectome"] is False and manifest["shuffle"]["seed"] == 0
    assert manifest["derived_from"]["dataset"] == "tiny"
    assert manifest["derived_from"]["arrays"] == {k: v["sha256"] for k, v in real["arrays"].items()}
    assert manifest["arrays"]["row_ptr"]["sha256"] == real["arrays"]["row_ptr"]["sha256"]
    assert manifest["arrays"]["destinations"]["sha256"] != real["arrays"]["destinations"]["sha256"]
    assert manifest["stats"]["self_loops"] == manifest["shuffle"]["self_loops"]


def test_it_never_replaces_a_pack_that_is_not_shuffled(tmp_path):
    _real_pack(tmp_path / "tiny")
    with pytest.raises(ValueError, match="not a shuffled pack"):
        shuffle_pack.write_shuffled_pack(tmp_path / "tiny", tmp_path / "tiny", seed=0)
    core.load_pack(tmp_path / "tiny")
    shuffle_pack.write_shuffled_pack(tmp_path / "tiny", tmp_path / "shuffled", seed=0)
    with pytest.raises(ValueError, match="already a shuffled pack"):
        shuffle_pack.write_shuffled_pack(tmp_path / "shuffled", tmp_path / "again", seed=1)
