"""lif.control_demo's pure parts: a rate over time from spike times, and the SVG
figure. The run itself needs both packs and is python -m lif.control_demo."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from lif import control_demo

SVG = "{http://www.w3.org/2000/svg}"


def test_rate_over_time_is_spikes_per_neuron_per_trial_per_second_of_bin():
    t = np.array([0.0, 0.005, 0.019, 0.02, 0.999])
    edges, rate = control_demo.rate_over_time(t, n_run=2, t_run=1.0, bin_s=0.02)
    assert edges.size == 51 and rate.size == 50
    assert rate[0] == pytest.approx(3 / (2 * 0.02))
    assert rate[1] == pytest.approx(1 / (2 * 0.02))
    assert rate[-1] == pytest.approx(1 / (2 * 0.02))
    assert rate[2:-1].sum() == 0
    assert rate.mean() == pytest.approx(t.size / (2 * 1.0))
    _, per_neuron = control_demo.rate_over_time(t, n_run=2, t_run=1.0, bin_s=0.02, n_neurons=4)
    np.testing.assert_allclose(per_neuron, rate / 4)


def test_rate_over_time_refuses_bins_that_do_not_tile_the_run_and_spikes_outside_it():
    with pytest.raises(ValueError, match="bin"):
        control_demo.rate_over_time(np.array([0.1]), n_run=1, t_run=1.0, bin_s=0.03)
    with pytest.raises(ValueError, match="outside"):
        control_demo.rate_over_time(np.array([1.0]), n_run=1, t_run=1.0, bin_s=0.02)


def _row(high, low, titles=("real wiring", "shuffled wiring"), note=""):
    edges = np.linspace(0.0, 1.0, 51)
    return [control_demo.Panel(titles[0], f"mean {high} Hz", edges, np.full(50, high)),
            control_demo.Panel(titles[1], f"mean {low} Hz", edges, np.full(50, low), note=note)]


def _figure(rows):
    ylabels = ["MN9 (Hz)", "sugar GRNs (Hz)"][:len(rows)]
    return control_demo.figure(rows, ylabels, title="MN9 & friends", caption="30 trials")


def _heights(text):
    bars = ET.fromstring(text).findall(f".//{SVG}rect[@class='bar']")
    return [float(b.get("height")) for b in bars]


def test_the_figure_has_one_bar_per_bin_and_a_y_axis_shared_within_each_row():
    rows = [_row(40.0, 20.0, note="no MN9 spike in 30 trials"), _row(10.0, 10.0, titles=("", ""))]
    text = _figure(rows)
    heights = _heights(text)
    assert len(heights) == 200
    assert heights[0] > 0 and heights[0] == pytest.approx(2 * heights[50])
    assert heights[100] == heights[150] == pytest.approx(control_demo.PLOT_H)
    assert "MN9 &amp; friends" in text and "shuffled wiring" in text
    assert "no MN9 spike in 30 trials" in text and "sugar GRNs (Hz)" in text
    assert text == _figure(rows)


def test_a_silent_row_still_has_an_axis():
    assert _heights(_figure([_row(0.0, 0.0)])) == [0.0] * 100
