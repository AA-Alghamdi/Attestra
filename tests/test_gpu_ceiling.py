"""Hermetic lock for the GPU-ceiling runner's honesty-critical step: ROW ALIGNMENT (scripts/run_gpu_ceiling.py).

The fine-tune scores the sealed rows in sorted(test_ids) order; the arena scores the frozen champion in
sp['test'] order. McNemar is only valid if the two correctness vectors are paired ROW-FOR-ROW, so the runner
reorders the champion vector by argsort(test). If that remap were wrong, every p-value would silently compare
mismatched rows. This locks the remap against a hand-computed expectation with a fake arena (no torch, no data).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.run_gpu_ceiling as R  # noqa: E402


class _FakeArena:
    """Returns a fixed champion correctness vector in sp['test'] (unsorted) order for one task."""
    def __init__(self, sealed_correct):
        self._c = list(sealed_correct)

    def measure(self, tag):
        return {"task": SimpleNamespace(sealed_correct=self._c)}


def test_champion_correctness_is_remapped_to_sorted_test_order():
    # sp['test'] order and the champion's correctness in THAT order:
    test = np.array([5, 2, 9, 1])
    champ_unsorted = [1, 0, 1, 0]                 # correctness aligned to rows [5, 2, 9, 1]
    order = np.argsort(test)                       # -> indices [3, 1, 0, 2] giving sorted rows [1, 2, 5, 9]

    aligned = R._aligned_champion_correct(_FakeArena(champ_unsorted), "task", order)

    # row 1 -> champ_unsorted[3]=0 ; row 2 -> [1]=0 ; row 5 -> [0]=1 ; row 9 -> [2]=1
    assert aligned == [0, 0, 1, 1]
    # and it is exactly the champion vector sorted by the test row id (the property McNemar needs):
    expected = [champ_unsorted[i] for i in np.argsort(test)]
    assert aligned == expected


def test_already_sorted_test_is_identity():
    test = np.array([0, 1, 2, 3])
    champ = [1, 1, 0, 1]
    aligned = R._aligned_champion_correct(_FakeArena(champ), "task", np.argsort(test))
    assert aligned == champ
