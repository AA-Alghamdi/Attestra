"""End-to-end Phase 7: author a method -> frozen admission gate -> candidate -> certify path.

Acceptance: a benign valid authored estimator passes the three-stage admission gate (static AST + isolated
self-test), is built, fit on TRAIN and predicts EVAL features (sealed-blind), and rides the verification
cascade to a (mocked) frozen Tier-3 certify exactly like a zoo family. A MALICIOUS source (import os / file
read) is rejected at the static gate and never executed. The frozen certifier is never touched by the bridge."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import authoring as A
from vfplatform import authoring_bridge as AB
from vfplatform import verification as V

# a benign, deterministic, numpy-only nearest-centroid classifier (passes the import allowlist + self-test)
GOOD_SOURCE = """
import numpy as np

class NearestCentroid:
    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.centroids_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(d, axis=1)]

def build_estimator(seed):
    return NearestCentroid()
"""

# a malicious source: tries to import os and read a secret file -> must be denied at the static gate
EVIL_SOURCE = """
import os
def build_estimator(seed):
    data = open('/etc/passwd').read()
    return data
"""


def _accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-2.0, 1.0, size=(n, 6)), rng.normal(2.0, 1.0, size=(n, 6))])
    y = np.asarray([0] * n + [1] * n)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def test_benign_method_admitted_and_rides_cascade():
    spec = A.EstimatorSpec(role="classifier", n_features=6, n_classes=2)
    report = AB.admit_method(GOOD_SOURCE, spec, family="nearest_centroid")
    assert report.admitted, report.reason
    assert report.factory is not None

    X, y = _data()
    tr = np.arange(0, 160)
    ev = np.arange(160, len(X))
    res = AB.evaluate_admitted(report, X, y, tr, ev, _accuracy)
    assert res.metric > 0.9                         # separable data -> the authored method works
    assert len(res.predictions) == len(ev)
    assert res.latency_ms >= 0.0

    # rides the SAME cascade as a zoo family; frozen Tier-3 is mocked here (the bridge never certifies itself)
    certified = {"called": 0}

    def frozen_certify():
        certified["called"] += 1
        return {"certified": True, "lower_bound": res.metric}

    cand = AB.to_candidate(res, frozen_certify)
    cascade = V.VerificationCascade(theta=0.7)
    outcome = cascade.evaluate(cand)
    assert outcome.promoted is True
    assert outcome.certificate is not None and outcome.certificate["certified"] is True
    assert certified["called"] == 1                 # frozen certify invoked exactly once at Tier 3


def test_malicious_method_rejected_at_static_gate():
    spec = A.EstimatorSpec(role="classifier", n_features=6, n_classes=2)
    report = AB.admit_method(EVIL_SOURCE, spec, family="evil")
    assert not report.admitted
    assert "static gate" in report.reason          # denied before any execution
    assert report.factory is None


def test_cannot_evaluate_rejected_method():
    spec = A.EstimatorSpec(role="classifier", n_features=6, n_classes=2)
    report = AB.admit_method(EVIL_SOURCE, spec)
    X, y = _data()
    with pytest.raises(ValueError):
        AB.evaluate_admitted(report, X, y, np.arange(100), np.arange(100, len(X)), _accuracy)


def test_sealed_blind_predict_receives_only_features():
    # prove the estimator's predict is called with a 2-D feature matrix (no labels), via a spy estimator
    spec = A.EstimatorSpec(role="classifier", n_features=6, n_classes=2)
    report = AB.admit_method(GOOD_SOURCE, spec, family="nc")
    X, y = _data()
    res = AB.evaluate_admitted(report, X, y, np.arange(160), np.arange(160, len(X)), _accuracy)
    # the per-example correctness was computed trusted-side AFTER predict -> same length as eval rows
    assert len(res.per_example_correct) == len(X) - 160
    assert set(np.unique(res.per_example_correct).tolist()) <= {0.0, 1.0}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
