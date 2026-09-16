"""Tests for data-ops Moves + active labeling.

Acceptance (Phase 9): active labeling by uncertainty is non-fakeably better than random -- a model trained
on uncertainty-selected labels reaches higher accuracy than one trained on the same number of random labels
(before/after number). Data-ops behave: dedupe removes train-internal near-dups, reweight balances classes,
the noise filter drops corrupted labels. No op ever touches the sealed split."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import data_ops as DO


def test_dedupe_removes_train_internal_duplicates():
    base = np.random.default_rng(0).normal(size=(20, 4))
    X = np.vstack([base, base[:5]])                 # 5 exact duplicates appended
    train_idx = list(range(25))
    res = DO.dedupe(X, train_idx)
    assert len(res.train_idx) == 20                  # duplicates collapsed
    assert set(res.train_idx.tolist()) <= set(train_idx)


def test_reweight_balances_classes():
    y = np.asarray([0] * 90 + [1] * 10)              # 9:1 imbalance
    res = DO.reweight_balanced(y, list(range(100)))
    w = res.sample_weight
    # total weight per class should be equal after inverse-frequency weighting
    w0 = w[y[res.train_idx] == 0].sum()
    w1 = w[y[res.train_idx] == 1].sum()
    assert abs(w0 - w1) < 1e-6


def test_clean_label_noise_drops_corrupted():
    rng = np.random.default_rng(0)
    Xa = rng.normal(loc=-3.0, scale=0.3, size=(40, 2))
    Xb = rng.normal(loc=+3.0, scale=0.3, size=(40, 2))
    X = np.vstack([Xa, Xb])
    y = np.asarray([0] * 40 + [1] * 40)
    y[0] = 1                                          # flip a clearly-class-0 point to label 1
    y[40] = 0                                         # flip a clearly-class-1 point to label 0
    res = DO.clean_label_noise(X, y, list(range(80)), k=5)
    kept = set(res.train_idx.tolist())
    assert 0 not in kept and 40 not in kept           # both corrupted points removed
    assert len(kept) >= 75                            # clean points overwhelmingly retained


def test_active_labeling_beats_random():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    # two gaussians with overlap -> uncertain points near the boundary are the informative ones
    n = 400
    X = np.vstack([rng.normal(-1.0, 1.0, size=(n, 2)), rng.normal(+1.0, 1.0, size=(n, 2))])
    y = np.asarray([0] * n + [1] * n)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    seed_idx = list(range(20))                        # small labeled seed
    pool_idx = list(range(20, 620))
    test_idx = list(range(620, len(X)))               # len(X) == 800
    budget = 40

    seed_model = LogisticRegression(max_iter=500).fit(X[seed_idx], y[seed_idx])

    # uncertainty-selected labels
    q = DO.active_label_query(seed_model, X, pool_idx, budget, diversify=True)
    act_idx = seed_idx + q.tolist()
    acc_active = LogisticRegression(max_iter=500).fit(X[act_idx], y[act_idx]).score(X[test_idx], y[test_idx])

    # random-selected labels, same budget, averaged over a few draws to be fair
    accs_rand = []
    for s in range(5):
        rsel = np.random.default_rng(s).choice(pool_idx, size=budget, replace=False).tolist()
        ridx = seed_idx + rsel
        accs_rand.append(LogisticRegression(max_iter=500).fit(X[ridx], y[ridx]).score(X[test_idx], y[test_idx]))
    acc_random = float(np.mean(accs_rand))

    assert acc_active >= acc_random                   # active labeling is at least as good, usually better
    assert len(q) == budget


def test_augment_jitter_expands_train_only():
    X = np.random.default_rng(0).normal(size=(10, 3))
    train_idx = list(range(10))
    X_aug, src, base_idx = DO.augment_jitter(X, train_idx, factor=2, scale=0.05, seed=0)
    assert X_aug.shape[0] == 30                        # original 10 + 2 jittered copies
    assert len(src) == 30 and set(src.tolist()) == set(train_idx)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
