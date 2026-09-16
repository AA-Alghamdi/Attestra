"""Tests for leak-safe splitters: zero group leakage, temporal embargo respected, stratification."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import splits as S


def test_group_split_has_zero_group_leakage():
    # 50 groups, ~4 rows each
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 4)
    rng.shuffle(groups)
    res = S.group_split(groups, seed=1)
    res.assert_no_leakage(groups=groups)        # raises on any straddle
    assert res.all_disjoint()
    # every group lives in exactly one split
    g = np.asarray(groups)
    for u in np.unique(g):
        where = [u in set(g[res.train_idx]), u in set(g[res.val_idx]), u in set(g[res.sealed_idx])]
        assert sum(where) == 1, (u, where)


def test_temporal_split_respects_order_and_embargo():
    times = np.arange(100.0)
    res = S.temporal_split(times, (0.6, 0.2, 0.2), embargo=5)
    res.assert_no_leakage(times=times)
    # forward chaining: every train time < every val time < every sealed time
    assert max(res.train_idx) < min(res.val_idx) < max(res.val_idx) < min(res.sealed_idx)
    # embargo removed rows from coverage
    assert len(res.embargoed_idx) == 10        # 5 at each of the two boundaries
    assert res.all_disjoint()


def test_stratified_split_preserves_class_balance():
    y = np.array([0] * 80 + [1] * 20)
    res = S.stratified_split(y, (0.6, 0.2, 0.2), seed=0)
    res.assert_no_leakage()
    for idx in (res.train_idx, res.val_idx, res.sealed_idx):
        frac1 = np.mean(np.asarray(y)[idx] == 1)
        assert abs(frac1 - 0.2) < 0.1          # roughly preserved minority fraction


def test_make_leaksafe_picks_axis_priority():
    times = np.arange(60.0)
    groups = np.repeat(np.arange(20), 3)
    # time present -> temporal wins
    r = S.make_leaksafe_splits(60, times=times, groups=groups, embargo=0)
    assert r.method == "temporal"
    # only groups -> grouped
    r2 = S.make_leaksafe_splits(60, groups=groups)
    assert r2.method == "grouped"
    r2.assert_no_leakage(groups=groups)


def test_overlapping_split_detected():
    res = S.SplitResult([0, 1, 2], [2, 3], [4, 5], "bad", [])
    with pytest.raises(S.SplitError):
        res.assert_no_leakage()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
