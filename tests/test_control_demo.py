"""lif.control_demo's pure parts: MN9's rate over time from spike times, and the
SVG figure. The run itself needs both packs and is python -m lif.control_demo."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from lif import control_demo

SVG = "{http://www.w3.org/2000/svg}"


def test_rate_over_time_is_spikes_per_trial_per_second_of_bin():
    t = np.array([0.0, 0.005, 0.019, 0.02, 0.999])
    edges, rate = control_demo.rate_over_time(t, n_run=2, t_run=1.0, bin_s=0.02)
    assert edges.size == 51 and rate.size == 50
    assert rate[0] == pytest.approx(3 / (2 * 0.02))
    assert rate[1] == pytest.approx(1 / (2 * 0.02))
    assert rate[-1] == pytest.approx(1 / (2 * 0.02))
    assert rate[2:-1].sum() == 0
    assert rate.mean() == pytest.approx(t.size / (2 * 1.0))


def test_rate_over_time_refuses_bins_that_do_not_tile_the_run_and_spikes_outside_it():
    with pytest.raises(ValueError, match="bin"):
        control_demo.rate_over_time(np.array([0.1]), n_run=1, t_run=1.0, bin_s=0.03)
    with pytest.raises(ValueError, match="outside"):
        control_demo.rate_over_time(np.array([1.0]), n_run=1, t_run=1.0, bin_s=0.02)


def _panels(high=40.0, low=20.0):
    edges = np.linspace(0.0, 1.0, 51)
    return [control_demo.Panel("real wiring", "mean 40.0 Hz", edges, np.full(50, high)),
            control_demo.Panel("shuffled wiring", "mean 20.0 Hz", edges, np.full(50, low))]


def test_the_figure_is_svg_with_one_bar_per_bin_on_a_shared_axis():
    text = control_demo.figure(_panels(), title="MN9 & friends", caption="30 trials")
    bars = ET.fromstring(text).findall(f".//{SVG}rect[@class='bar']")
    assert len(bars) == 100
    heights = [float(b.get("height")) for b in bars]
    assert heights[0] > 0 and heights[0] == pytest.approx(2 * heights[50])
    assert "MN9 &amp; friends" in text and "shuffled wiring" in text
    assert text == control_demo.figure(_panels(), title="MN9 & friends", caption="30 trials")


def test_a_silent_figure_still_has_an_axis():
    text = control_demo.figure(_panels(0.0, 0.0), title="MN9", caption="")
    bars = ET.fromstring(text).findall(f".//{SVG}rect[@class='bar']")
    assert [float(b.get("height")) for b in bars] == [0.0] * 100
