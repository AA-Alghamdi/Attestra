"""Tests for the data certificate: near-duplicate straddle fails, clean data passes."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import data_cert as DC


def test_clean_split_passes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)
    train_idx = list(range(120))
    sealed_idx = list(range(160, 200))
    rep = DC.certify_dataset(X, y, train_idx=train_idx, sealed_idx=sealed_idx)
    assert rep.passed is True and rep.near_dup_straddle == 0


def test_near_duplicate_straddle_fails():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)
    # inject: copy 10 train rows into the sealed region (exact duplicates straddling the split)
    sealed_idx = list(range(160, 200))
    train_idx = list(range(120))
    for j in range(10):
        X[sealed_idx[j]] = X[train_idx[j]]
    rep = DC.certify_dataset(X, y, train_idx=train_idx, sealed_idx=sealed_idx)
    assert rep.passed is False
    assert rep.near_dup_straddle >= 10
    assert any("LEAKAGE" in w for w in rep.warnings)


def test_label_noise_flagged_when_high():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 4))
    y = rng.integers(0, 2, size=150)        # labels independent of X -> ~50% kNN disagreement
    rep = DC.certify_dataset(X, y, train_idx=list(range(100)), sealed_idx=list(range(120, 150)),
                             max_label_noise=0.4)
    assert rep.label_noise_est is not None and rep.label_noise_est > 0.3
    assert rep.passed is False


def test_degenerate_feature_detected():
    X = np.column_stack([np.ones(50), np.arange(50.0), np.random.default_rng(3).normal(size=50)])
    y = (X[:, 2] > 0).astype(int)
    rep = DC.certify_dataset(X, y, train_idx=list(range(30)), sealed_idx=list(range(40, 50)),
                             estimate_noise=False)
    assert 0 in rep.degenerate_features and 1 in rep.degenerate_features


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
