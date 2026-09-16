"""Hermetic tests for the B2 authoring-benchmark's PURE logic (no LLM, no network, no torch, no CIFAR).

The arena itself needs an Anthropic key + torch + a CIFAR download, so the full run is not a CI test. But the
B2-specific glue that keeps the comparison honest IS pure and must be locked:
  * _fit_score builds an ADMITTED authored method through its own catalog_entry().builder, fits on TRAIN only,
    and scores per-row on EVAL (the same machinery the sealed bound rides);
  * _fit_score NEVER raises -- a method that blows up on real data scores -1 and simply loses selection, so a
    broken authored candidate can never crash the measurement or sneak past selection;
  * _env_context feeds the author ACCURATE installed library versions (trusted instruction-side facts), never
    fabricated ones and never task data.
The select-then-bound + paired-McNemar + BH-FDR logic is inherited verbatim from the B1 helpers, which are
locked by test_vision_transfer_bench.py; the admit -> cascade -> only-frozen-Tier-3 invariant is locked by
test_authoring_bridge.py. The frozen certifier is only ever read here.
"""
import importlib.util
import os

import numpy as np

from vfplatform import authoring as A

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "benchmark_authoring.py")
_spec = importlib.util.spec_from_file_location("benchmark_authoring", _PATH)
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)

# a benign, deterministic, numpy-only nearest-centroid classifier (clears the import allowlist + self-test)
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


def _separable(seed=0, n=160, d=6):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-2.0, 1.0, size=(n, d)), rng.normal(2.0, 1.0, size=(n, d))])
    y = np.asarray([0] * n + [1] * n)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def test_fit_score_scores_admitted_method_on_train_then_eval():
    spec = A.EstimatorSpec(role="classifier", n_features=6, n_classes=2)
    report = A.admit(GOOD_SOURCE, spec, family="nearest_centroid")
    assert report.admitted, report.reason
    X, y = _separable()
    tr, ev = slice(0, 240), slice(240, len(X))
    acc, correct = ba._fit_score(report.factory, X[tr], y[tr], X[ev], y[ev])
    assert acc > 0.9                                   # separable -> an admitted method must work
    assert correct is not None and len(correct) == len(y[ev])
    assert set(np.unique(correct).tolist()) <= {0, 1}  # per-row 0/1 correctness, ready for McNemar pairing


class _ExplodingFactory:
    """An estimator factory whose built model raises on fit -- stands in for an authored method that admits
    on synthetic self-test data but blows up on the real embeddings."""
    def catalog_entry(self):
        class _E:
            @staticmethod
            def builder(params, seed):
                class _M:
                    def fit(self, X, y):
                        raise RuntimeError("kaboom on real data")
                    def predict(self, X):
                        raise RuntimeError("never reached")
                return _M()
        return _E()


def test_fit_score_never_raises_on_broken_method():
    X, y = _separable()
    acc, correct = ba._fit_score(_ExplodingFactory(), X[:240], y[:240], X[240:], y[240:])
    assert acc == -1.0 and correct is None             # honest loss, not a crash -> loses selection silently


def test_env_context_reports_real_installed_versions_not_task_data():
    import sklearn
    ctx = ba._env_context()
    assert ctx["environment"]["sklearn"] == sklearn.__version__
    assert ctx["environment"]["numpy"] == np.__version__
    blob = repr(ctx).lower()
    # trusted instruction-side facts only: no labels / sealed test / accuracies leak into the author's context
    assert "sealed" not in blob and "label" not in blob and "accuracy" not in blob


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
