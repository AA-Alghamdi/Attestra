"""Tests for the certifier-first meta-certifier.

Acceptance (Phase 11): a SOUND framing passes all probes; an EXPLOITABLE metric (accuracy on a 95:5
imbalance with theta below the majority rate) is rejected by the trivial-baseline probe; a LEAKY framing
(a sealed row duplicated into train) is rejected by the straddle probe; the label-shuffle probe rejects a
framing where shuffled labels still clear theta. Verification leads."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import meta_certifier as MC


def _fit_predict(X, y, train_idx, sealed_idx):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500).fit(X[train_idx], y[train_idx])
    return clf.predict(X[sealed_idx])


def _accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _balanced_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-2.0, 1.0, size=(n, 4)), rng.normal(+2.0, 1.0, size=(n, 4))])
    y = np.asarray([0] * n + [1] * n)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    cut = int(0.7 * len(X))
    return X, y, np.arange(cut), np.arange(cut, len(X))


def test_sound_framing_passes():
    X, y, tr, se = _balanced_data()
    rep = MC.validate_certifier(X=X, y=y, train_idx=tr, sealed_idx=se,
                                fit_predict_fn=_fit_predict, metric_fn=_accuracy, theta=0.8)
    assert rep.trustworthy, rep.summary()
    assert {p.name for p in rep.probes} >= {"trivial_baseline", "label_shuffle", "straddle_leak"}


def test_exploitable_metric_rejected_by_trivial_baseline():
    # 95:5 imbalance; theta=0.8 is BELOW the 0.95 majority rate -> a constant predictor clears it.
    rng = np.random.default_rng(1)
    n0, n1 = 380, 20
    X = np.vstack([rng.normal(0.0, 1.0, size=(n0, 3)), rng.normal(0.2, 1.0, size=(n1, 3))])
    y = np.asarray([0] * n0 + [1] * n1)
    perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
    cut = int(0.7 * len(X))
    rep = MC.validate_certifier(X=X, y=y, train_idx=np.arange(cut), sealed_idx=np.arange(cut, len(X)),
                                fit_predict_fn=_fit_predict, metric_fn=_accuracy, theta=0.8)
    assert not rep.trustworthy
    assert any(p.name == "trivial_baseline" and not p.passed for p in rep.probes)


def test_straddle_leak_rejected():
    X, y, tr, se = _balanced_data(seed=2)
    # duplicate the first sealed row into train -> a near-duplicate straddles the boundary
    X = X.copy()
    X[tr[0]] = X[se[0]]
    y = y.copy(); y[tr[0]] = y[se[0]]
    rep = MC.validate_certifier(X=X, y=y, train_idx=tr, sealed_idx=se,
                                fit_predict_fn=_fit_predict, metric_fn=_accuracy, theta=0.8)
    assert not rep.trustworthy
    assert any(p.name == "straddle_leak" and not p.passed for p in rep.probes)


def test_label_shuffle_probe_present_and_passes_on_clean():
    X, y, tr, se = _balanced_data(seed=3)
    rep = MC.validate_certifier(X=X, y=y, train_idx=tr, sealed_idx=se,
                                fit_predict_fn=_fit_predict, metric_fn=_accuracy, theta=0.8)
    shuf = next(p for p in rep.probes if p.name == "label_shuffle")
    assert shuf.passed                              # shuffled labels cannot clear theta on a clean framing


def test_group_split_leak_probe_fires():
    X, y, tr, se = _balanced_data(seed=4)
    n = len(X)
    groups = np.arange(n) % 10                       # 10 groups
    groups[tr[0]] = groups[se[0]]                    # force a group to straddle train/sealed
    rep = MC.validate_certifier(X=X, y=y, train_idx=tr, sealed_idx=se,
                                fit_predict_fn=_fit_predict, metric_fn=_accuracy, theta=0.8,
                                groups=groups)
    assert any(p.name == "split_leak" for p in rep.probes)
    assert not rep.trustworthy


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
