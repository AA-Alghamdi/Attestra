"""Tests for the data pool registry: admission gated by data certificate, replication set, persistence."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import datapool as DP


def _clean(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(int)
    return X, y, list(range(80)), list(range(100, 120))


def test_admission_and_persistence(tmp_path):
    pool = DP.DataPool(str(tmp_path))
    X, y, tr, se = _clean(0)
    pool.add("d1", modality="tabular", task_type="binary", X=X, y=y, train_idx=tr, sealed_idx=se)
    assert "d1" in pool.names()
    # reload from disk -> persisted
    pool2 = DP.DataPool(str(tmp_path))
    assert pool2.get("d1") is not None and pool2.get("d1").n == 120


def test_leaky_dataset_refused(tmp_path):
    pool = DP.DataPool(str(tmp_path))
    X, y, tr, se = _clean(1)
    for j in range(8):                          # inject straddling duplicates
        X[se[j]] = X[tr[j]]
    with pytest.raises(DP.DataPoolError):
        pool.add("leaky", modality="tabular", task_type="binary", X=X, y=y, train_idx=tr, sealed_idx=se)
    assert "leaky" not in pool.names()


def test_replication_set_matches_modality_and_task(tmp_path):
    pool = DP.DataPool(str(tmp_path))
    for i, name in enumerate(["a", "b", "c"]):
        X, y, tr, se = _clean(10 + i)
        pool.add(name, modality="tabular", task_type="binary", X=X, y=y, train_idx=tr, sealed_idx=se)
    # a text dataset should not be in the tabular replication set
    Xt, yt, trt, set_ = _clean(99)
    pool.add("txt", modality="text", task_type="binary", X=Xt, y=yt, train_idx=trt, sealed_idx=set_)
    rep = pool.replication_set("a")
    rep_names = sorted(d.name for d in rep)
    assert rep_names == ["b", "c"]              # same modality+task, anchor excluded, text excluded


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
