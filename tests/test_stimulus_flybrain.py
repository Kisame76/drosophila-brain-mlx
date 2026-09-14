"""The sugar drive must drive all of its neurons or refuse to run.

No pack needed: the lookup is checked against synthetic neuron ID arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from lif import stimulus_flybrain

GRNS = stimulus_flybrain.RIGHT_SUGAR_GRN_IDS


def test_targets_follow_the_id_list_not_the_pack_order():
    ids = np.array([5, *reversed(GRNS), 7], dtype=np.int64)
    assert ids[stimulus_flybrain.targets(ids)].tolist() == GRNS


def test_targets_refuses_a_pack_missing_a_sugar_grn():
    ids = np.array([5, *GRNS[:-1], 7], dtype=np.int64)
    with pytest.raises(ValueError, match=str(GRNS[-1])):
        stimulus_flybrain.targets(ids)
