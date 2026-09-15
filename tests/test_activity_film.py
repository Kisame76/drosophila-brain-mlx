"""lif.activity_film's pure parts: soma positions in pack order, spikes per bin, the
turning view, one frame and the animated PNG. The film itself needs the MaleCNS pack
and its annotation table and is python -m lif.activity_film."""

import struct
import zlib

import numpy as np
import pyarrow as pa
import pytest

from lif import activity_film, spike_record


def _annotations():
    return pa.table({"bodyId": pa.array([30, 10, 20], pa.int64()),
                     "somaLocation": pa.array([[7, 8, 9], None, [1, 2, 3]], pa.list_(pa.int64()))})


def test_soma_positions_are_in_pack_order_and_nan_where_the_table_has_none():
    pos = activity_film.soma_positions(_annotations(), np.array([10, 20, 30]))
    assert pos.dtype == np.float64 and pos.shape == (3, 3)
    assert np.isnan(pos[0]).all()
    np.testing.assert_array_equal(pos[1:], [[1, 2, 3], [7, 8, 9]])


def test_soma_positions_refuse_what_they_cannot_place():
    two = pa.table({"bodyId": pa.array([10], pa.int64()),
                    "somaLocation": pa.array([[1, 2]], pa.list_(pa.int64()))})
    with pytest.raises(ValueError, match="three"):
        activity_film.soma_positions(two, np.array([10]))
    with pytest.raises(ValueError, match=r"\[40\]"):
        activity_film.soma_positions(_annotations(), np.array([10, 40]))
    with pytest.raises(ValueError, match="somaLocation"):
        activity_film.soma_positions(_annotations().drop_columns(["somaLocation"]), np.array([10]))


def test_spikes_are_counted_per_neuron_per_bin_on_whole_ticks():
    tick = np.array([0, 99, 100, 100, 9999, 5000])
    neuron = np.array([4, 4, 4, 7, 7, 4])
    active, counts = activity_film.spike_bins(spike_record.tick_to_seconds(tick), neuron,
                                              t_run=1.0, bin_ms=10.0)
    assert active.tolist() == [4, 7]
    assert counts.shape == (2, 100) and counts.dtype == np.int64
    assert counts[0, 0] == 2 and counts[0, 1] == 1 and counts[0, 50] == 1
    assert counts[1, 1] == 1 and counts[1, 99] == 1
    assert counts.sum() == tick.size


def test_bins_must_be_whole_ticks_that_tile_the_run_and_spikes_must_lie_on_its_ticks():
    t = spike_record.tick_to_seconds(np.array([5]))
    with pytest.raises(ValueError, match="tile"):
        activity_film.spike_bins(t, np.array([0]), t_run=1.0, bin_ms=30.0)
    with pytest.raises(ValueError, match="whole number of"):
        activity_film.spike_bins(t, np.array([0]), t_run=1.0, bin_ms=0.25)
    with pytest.raises(ValueError, match="outside"):
        activity_film.spike_bins(np.array([1.0]), np.array([0]), t_run=1.0, bin_ms=10.0)
    with pytest.raises(ValueError, match="between ticks"):
        activity_film.spike_bins(np.array([0.00005]), np.array([0]), t_run=1.0, bin_ms=10.0)


def _cloud(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * [30000, 20000, 50000] + [50000, 40000, 70000]


def test_the_view_keeps_every_position_in_the_picture_at_every_angle():
    pos = _cloud()
    view = activity_film.View.fit(pos, width=200)
    for theta in np.linspace(0, 2 * np.pi, 13):
        u, v, depth = view.project(pos, theta)
        assert u.min() >= 0 and u.max() <= view.width - 1
        assert v.min() >= 0 and v.max() <= view.scene_height - 1
        assert np.abs(depth).max() <= 1 + 1e-12


def test_half_a_turn_mirrors_the_picture_and_swaps_near_and_far():
    pos = _cloud()
    view = activity_film.View.fit(pos, width=200)
    u0, v0, d0 = view.project(pos, 0.3)
    u1, v1, d1 = view.project(pos, 0.3 + np.pi)
    np.testing.assert_allclose(u0 + u1, view.width - 1)
    np.testing.assert_allclose(v0, v1)
    np.testing.assert_allclose(d0, -d1, atol=1e-12)
    # At angle 0, screen right is x and screen down z, so a larger y is nearer:
    # a right-handed view rather than its mirror image.
    near, far = view.center + np.array([[0, 1e4, 0], [0, -1e4, 0]])
    assert view.project(np.array([near, far]), 0.0)[2].tolist() == pytest.approx([-1e4 / view.radius,
                                                                                   1e4 / view.radius])


def test_a_frame_lights_the_pixel_of_the_neuron_that_spiked_and_nothing_when_none_did():
    pos = _cloud(200)
    view = activity_film.View.fit(pos, width=120)
    spiking = activity_film.frame(view, pos, pos[[5]], np.array([80.0]), 0.0, "10 ms", 0.5)
    silent = activity_film.frame(view, pos, pos[[5]], np.array([0.0]), 0.0, "10 ms", 0.5)
    assert spiking.dtype == np.uint8 and spiking.shape == (view.height, view.width)
    scene = activity_film.ACT_LEVELS * activity_film.FOG_LEVELS
    u, v, _ = view.project(pos[[5]], 0.0)
    level = np.where(spiking < scene, spiking % activity_film.ACT_LEVELS, 0)
    assert level[round(v[0]), round(u[0])] == level.max() > 0
    assert (np.where(silent < scene, silent % activity_film.ACT_LEVELS, 0) == 0).all()
    strip = spiking[view.scene_height:]
    assert (strip == activity_film.TEXT).any() and (strip == activity_film.BAR).any()
    assert len(activity_film.PALETTE) == activity_film.BAR + 1 <= 256


def test_text_comes_from_the_bitmap_font():
    img = np.zeros((20, 40), np.uint8)
    activity_film.draw_text(img, "1 ms", 1, 2, index=9, scale=2)
    assert (img[2:16, 5:7] == 9).all()        # the stem of the 1, glyph column 2
    assert (img[2:16, 1:3] == 0).all()         # glyph column 0, empty in every row
    with pytest.raises(ValueError, match="'x'"):
        activity_film.draw_text(img, "x", 0, 0, index=9)


def _chunks(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    at, out = 8, []
    while at < len(data):
        (length,) = struct.unpack(">I", data[at:at + 4])
        kind, body = data[at + 4:at + 8], data[at + 8:at + 8 + length]
        assert struct.unpack(">I", data[at + 8 + length:at + 12 + length])[0] == zlib.crc32(kind + body)
        out.append((kind, body))
        at += 12 + length
    return out


def _unfilter(data, width, height):
    """PNG scanlines with filter 0 or 4 (Paeth), one byte per pixel, back to pixels."""
    rows, prev = [], [0] * width
    for y in range(height):
        line = data[y * (width + 1):(y + 1) * (width + 1)]
        cur = []
        for x in range(width):
            a, b, c = (cur[x - 1] if x else 0), prev[x], (prev[x - 1] if x else 0)
            if line[0] == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else b if pb <= pc else c
            else:
                assert line[0] == 0
                pred = 0
            cur.append((line[1 + x] + pred) % 256)
        rows.append(cur)
        prev = cur
    return np.array(rows, dtype=np.uint8)


def test_the_animated_png_holds_every_frame_losslessly():
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 6, (7, 9), dtype=np.uint8) for _ in range(3)]
    palette = rng.integers(0, 256, (6, 3), dtype=np.uint8)
    chunks = _chunks(activity_film.apng(frames, palette, delay_ms=80))
    assert [k for k, _ in chunks] == [b"IHDR", b"PLTE", b"acTL", b"fcTL", b"IDAT",
                                      b"fcTL", b"fdAT", b"fcTL", b"fdAT", b"IEND"]
    assert struct.unpack(">IIBBBBB", chunks[0][1]) == (9, 7, 8, 3, 0, 0, 0)
    assert chunks[1][1] == palette.tobytes()
    assert struct.unpack(">II", chunks[2][1]) == (3, 0)
    sequence, decoded = [], []
    for kind, body in chunks:
        if kind == b"fcTL":
            number, *rest = struct.unpack(">IIIIIHHBB", body)
            sequence.append(number)
            assert rest == [9, 7, 0, 0, 80, 1000, 0, 0]
        elif kind == b"IDAT":
            decoded.append(_unfilter(zlib.decompress(body), 9, 7))
        elif kind == b"fdAT":
            sequence.append(struct.unpack(">I", body[:4])[0])
            decoded.append(_unfilter(zlib.decompress(body[4:]), 9, 7))
    assert sequence == list(range(5))
    for got, want in zip(decoded, frames, strict=True):
        np.testing.assert_array_equal(got, want)


def test_the_animated_png_refuses_frames_it_cannot_encode():
    palette = np.zeros((4, 3), np.uint8)
    with pytest.raises(ValueError, match="shape"):
        activity_film.apng([np.zeros((2, 3), np.uint8), np.zeros((3, 2), np.uint8)], palette, 80)
    with pytest.raises(ValueError, match="palette"):
        activity_film.apng([np.full((2, 3), 4, np.uint8)], palette, 80)
