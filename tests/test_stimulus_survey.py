"""lif.stimulus_survey's parts: regions from the annotation table, a stimulus
switched off part way, and spikes per region per window. The survey itself needs
the MaleCNS pack and is python -m lif.stimulus_survey."""

import math

import mlx.core as mx
import numpy as np
import pytest

from lif import activity_film, core
from lif import stimulus_survey as survey


def test_regions_come_from_the_superclass_and_abdominal_from_the_neuromere():
    cases = {"cb_intrinsic": "brain", "ol_sensory": "brain", "visual_projection": "brain",
             "vnc_intrinsic": "VNC", "vnc_motor": "VNC", "vnc_sensory_tbc": "VNC",
             "ascending_neuron": "ascending", "sensory_ascending": "ascending",
             "efferent_ascending": "ascending", "descending_neuron_tbc": "descending",
             "sensory_descending": "descending", "ENS": "other", None: "other"}
    for superclass, region in cases.items():
        assert survey.REGIONS[survey.region_of(superclass)] == region, superclass
    assert [survey.is_abdominal(n) for n in ("A1", "A10", "T1", "GNG", "CG", None)] == [
        True, True, False, False, False, False]


def _stimulus():
    return core.Stimulus(targets=np.array([2, 5], dtype=np.int32),
                         draws=mx.array(np.ones((4, 2), dtype=bool)), n_ticks=4,
                         rate_hz=100.0, seed=3)


def test_switch_off_ends_the_input_after_a_tick_and_keeps_the_rest():
    stim = _stimulus()
    off = survey.switch_off(stim, 1)
    assert np.array(off.draws).tolist() == [[True, True], [False, False], [False, False],
                                            [False, False]]
    assert off.targets.tolist() == [2, 5] and (off.n_ticks, off.seed) == (4, 3)
    assert math.isnan(off.rate_hz), "the draws are no longer at one rate"
    assert np.array(stim.draws).all(), "the stimulus it came from is unchanged"
    assert survey.switch_off(stim, 4).rate_hz == 100.0
    for after in (-1, 5):
        with pytest.raises(ValueError, match="from 0 to 4"):
            survey.switch_off(stim, after)


def test_course_counts_spikes_per_region_and_window_over_trials():
    region = np.array([0, 1, 1, 2, 3, 4])
    course = survey.Course(region, n_ticks=20, window_ticks=10)
    course.add(np.array([[0, 0], [5, 1], [12, 1], [19, 2], [15, 5]], dtype=np.int32))
    course.add(np.array([[3, 1], [11, 3], [18, 4]], dtype=np.int32))

    assert course.trials == 2
    assert course.counts.tolist() == [[1, 0], [2, 2], [0, 1], [0, 1], [0, 1]]
    np.testing.assert_allclose(course.spikes_per_trial(), [1.5, 2.5])
    np.testing.assert_allclose(course.share(survey.REGIONS.index("VNC")), [2 / 3, 2 / 5])
    # the last window is ticks 10 to 19, 1 ms: one spike in two trials is 500 Hz
    np.testing.assert_allclose(course.last_window_hz(), [0, 500, 500, 500, 500, 500])
    # a mean over trials can hide trials that differ: the VNC's last window, trial by trial
    assert course.last_window_per_trial(survey.REGIONS.index("VNC")).tolist() == [2, 0]
    assert course.last_window_per_trial(survey.REGIONS.index("ascending")).tolist() == [0, 1]


def test_course_refuses_windows_that_do_not_tile_the_run_and_spikes_outside_it():
    with pytest.raises(ValueError, match="tile"):
        survey.Course(np.zeros(3, dtype=np.int64), n_ticks=20, window_ticks=7)
    course = survey.Course(np.zeros(3, dtype=np.int64), n_ticks=20, window_ticks=10)
    with pytest.raises(ValueError, match="outside"):
        course.add(np.array([[20, 0]], dtype=np.int32))


def test_drive_targets_keep_run_exps_order_so_the_inputs_are_those_of_its_trials():
    """make_stimulus_for draws one column per target in the order given, so sorting
    the targets would hand each neuron another neuron's input spikes."""
    select = {"LB3b_R": [7, 3], "LB3c_R": [9, 1], "both": [3]}
    assert survey.drive_targets(lambda n: select.get(n, []), ("LB3b_R", "LB3c_R")) == [7, 3, 9, 1]
    with pytest.raises(ValueError, match=r"no neurons named \['x'\]"):
        survey.drive_targets(lambda n: select.get(n, []), ("LB3b_R", "x"))
    with pytest.raises(ValueError, match=r"more than once: \[3\]"):
        survey.drive_targets(lambda n: select.get(n, []), ("LB3b_R", "both"))


def test_the_survey_starts_with_the_films_stimulus_and_names_each_stimulus_once():
    assert survey.STIMULI[0].drive == activity_film.DRIVE
    labels = [s.label for s in survey.STIMULI]
    assert len(set(labels)) == len(labels)
