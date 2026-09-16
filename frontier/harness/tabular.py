"""TabularHarness: the harness for dense numeric tables (classification & regression).

# === WIRING ===
# This is the concrete Harness the router resolves for the "tabular" task-type key. It is
# registered into the shared REGISTRY at import time (see frontier/harness/__init__.py), so the
# integrator just does:
#
#     from frontier.harness import lookup
#     h = lookup("tabular")                  # also reachable via "classification"/"regression"
#     ok, cert = h.self_test()               # GATE before trusting any of its numbers
#     task = h.adapt(X, y, kind="classification", theta=0.85, name="bc")
#     # then feed `task` to ResearchEngine.run as in base.py's wiring block.
#
# It reuses the Phase-0 plumbing wholesale:
#   - adapt() builds a frontier.task.Task, which renders to rows via Task.to_rows() and is split
#     by certify.make_splits (which calls the FROZEN science.make_splits with its leakage dedup).
#     So TabularHarness adds NO new split/leakage logic -- it reuses make_splits by construction.
#   - baseline_suite() returns the same recipe Programs the SeedProposer uses (make_code), so the
#     floor is identical to the spine's and stays a seed/fallback, never a promoter.
#   - metric_for() returns right-axis metrics the frozen certifier supports
#     ("accuracy"/"balanced_accuracy"/"macro_f1" for clf; "r2"/"neg_rmse"/"neg_mae" for reg).
#
# Why TabularHarness still earns its keep when Phase 0 already does tabular: it makes the
# data->Task->metric->split decisions EXPLICIT and SELF-CERTIFYING behind one key, which is what
# the Phase-3 router needs to treat every modality uniformly. The same shape (adapt/baselines/
# split/metric/self_test) is what a vision/text/timeseries harness will implement next.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier.program import Program            # noqa: E402
from frontier.proposers import make_code, recipe_label  # noqa: E402
from frontier.task import Task                  # noqa: E402

from .base import Harness                        # noqa: E402


# Metrics the frozen certifier (vectorforge.science.score_metric) supports per kind. We only
# expose right-axis choices so a harness cannot route a regression task to an accuracy metric.
_VALID_METRICS = {
    "classification": ("accuracy", "balanced_accuracy", "macro_f1"),
    "regression": ("r2", "neg_rmse", "neg_mae"),
}
_DEFAULT_METRIC = {"classification": "accuracy", "regression": "r2"}

# Floor recipes per kind. These mirror the spine's SeedProposer seeds (so the harness floor IS
# the spine floor) and deliberately include a feature-engineering and a target-transform recipe
# so the tabular floor exercises those axes too (the CA-Housing trap fix). All are seeds/fallbacks.
_BASELINE_RECIPES = {
    "classification": [
        {"base": "hist_gbm"},
        {"base": "rf"},
        {"base": "logreg", "scale": True},
    ],
    "regression": [
        {"base": "hist_gbm"},
        {"base": "ridge", "scale": True},
        {"base": "ridge", "scale": True, "poly": 2},   # feature engineering in the floor
    ],
}


class TabularHarness(Harness):
    """Harness for dense numeric feature tables.

    Classification and regression are both handled. adapt() coerces X to a float matrix and
    validates shape/finiteness, then hands off to the frozen Task/split/certify path unchanged.
    """

    key = "tabular"
    kinds = ("classification", "regression")

    # ------------------------------------------------------------------ adapter API
    def metric_for(self, kind: str) -> str:
        """Right-axis default metric for the kind (accuracy for clf, r2 for reg)."""
        if kind not in _DEFAULT_METRIC:
            raise ValueError(f"TabularHarness does not handle kind {kind!r}")
        return _DEFAULT_METRIC[kind]

    def adapt(self, X, y, *, kind: str, theta: float, name: str = "task",
              metric: str = "") -> Task:
        """Validate + coerce a raw table into a Phase-0 Task.

        Checks (raise early rather than emit a silently-corrupt Task -- no swallowed failures):
          - kind is one this harness handles;
          - X is 2D numeric and finite (NaN/Inf in features is a data bug, not a model bug);
          - y length matches X;
          - the metric, if supplied, is on the right axis for the kind.
        theta is the caller's verification standard; the harness does not invent it.
        """
        if kind not in self.kinds:
            raise ValueError(f"TabularHarness handles {self.kinds}, not {kind!r}")
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        if Xa.ndim != 2:
            raise ValueError(f"tabular X must be 2D, got shape {Xa.shape}")
        if not np.all(np.isfinite(Xa)):
            n_bad = int((~np.isfinite(Xa)).sum())
            raise ValueError(f"tabular X has {n_bad} non-finite entries; clean or impute first")
        ya = np.asarray(y)
        if len(ya) != len(Xa):
            raise ValueError(f"X/y length mismatch: {len(Xa)} vs {len(ya)}")

        if metric:
            if metric not in _VALID_METRICS[kind]:
                raise ValueError(
                    f"metric {metric!r} is not a valid {kind} metric; "
                    f"choose one of {_VALID_METRICS[kind]}")
        else:
            metric = self.metric_for(kind)

        # For classification, Task.labels sorts string labels; coerce y to str so the frozen
        # stratified split + Clopper-Pearson path sees stable class identities.
        if kind == "classification":
            ya = ya.astype(str)
        else:
            ya = ya.astype(float)

        return Task(X=Xa, y=ya, kind=kind, theta=float(theta), metric=metric, name=name)

    def baseline_suite(self, kind: str) -> List[Program]:
        """Floor recipes for the kind, as seed Programs (seeds/fallbacks, never promoters)."""
        if kind not in self.kinds:
            raise ValueError(f"TabularHarness handles {self.kinds}, not {kind!r}")
        progs: List[Program] = []
        for recipe in _BASELINE_RECIPES[kind]:
            progs.append(Program(
                code=make_code(recipe, kind),
                source="seed",
                label=recipe_label(recipe),
                provenance={"recipe": dict(recipe), "harness": self.key},
            ))
        return progs

    def split_protocol(self, kind: str) -> Tuple[float, float]:
        """Tabular uses the Phase-0 default 30/20 split (train 50% / val 20% / sealed 30%)."""
        return (0.30, 0.20)

    # ------------------------------------------------------------------ self-test case
    def _self_test_case(self) -> Tuple[np.ndarray, np.ndarray, str, str, float]:
        """Known-good case: sklearn breast-cancer at a conservative theta a baseline clears.

        Breast-cancer (569 samples, 30 features, 2 classes) is linearly separable enough that a
        scaled logistic-regression / gradient-boosting baseline reaches ~0.95+ accuracy; the
        sealed LOWER bound at this n comfortably clears 0.85. We pick 0.85 (not the achievable
        ~0.95) on purpose: the self-test must verify the adapter/metric/split PLUMBING, with
        margin to spare, not benchmark the model. A clf self-test that fails here means the
        harness's plumbing is broken, not that the dataset is hard.
        """
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        return d.data, d.target.astype(str), "classification", "sklearn_breast_cancer", 0.85
