"""PHASE 7 BRIDGE -- an admitted authored method flows to the certify path as JUST ANOTHER CANDIDATE.

STATUS (2026-06): WIRED into the REGENERATIVE researcher. RecipeResearcher._admit() routes any code-bearing
recipe (an LLM-authored featurizer/head/loss carried on Recipe.code_patch) through admit_method BEFORE it
can spend a sealed peek; an unsafe patch (e.g. `import os`) is rejected at the static AST gate. Falsified by
a hermetic lock (test_recipe_research.py::test_authoring_sandbox_rejects_unsafe_code). On the WILDS run
code_prob=0.0 (no authored patches proposed yet), but the seam is live and load-bearing for GPU-day recipes.

NOTE: the B2 measurement (authoring on a FIXED representation vs tuned GBM = 0/5 FDR) is why authoring is a
GATED OPTION on the OPEN recipe, never the lever itself -- the certifier still decides.

WHY THIS EXISTS
---------------
authoring.py already implements the frozen three-stage admission gate (static AST gate -> isolated
execution -> scientific self-test) and yields an AuthoredEstimator whose catalog_entry() is shape-identical
to a hand-written harness catalog family. What was missing is the thin, tested seam that takes an ADMITTED
method, evaluates it on a real split, and packages the result as a verification.Candidate + gate.Measurements
so it rides the EXACT same cascade -> frozen Tier-3 path as a zoo family. This module is that seam.

THE INVARIANTS (D1, D4, D5, security)
-------------------------------------
* The frozen certifier is never imported or touched here; the caller passes in its frozen certify_fn and
  this module only ROUTES the candidate to it (the bridge never promotes).
* SEALED-BLINDNESS: the authored estimator only ever sees TRAIN labels and EVAL/VAL *features*. The metric
  against eval labels is computed on the trusted side, after predict() returns -- labels never cross into
  authored code. (On a novel/untrusted method, real-data fit/predict must additionally run under the OS
  isolation in authored_pod_sandbox.run_authored; admission already runs in a spawned, rlimited child.)
* Admission is SAFETY + BUILDABILITY + CONTRACT only, never a quality bar; quality is decided ONLY by the
  cascade's validation selection and the frozen sealed certify path.

CONTRACT: numpy + the repo's own authoring.py / verification.py / gate.py. No certifier import.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from . import authoring as A
from . import verification as V
from .gate import Measurements

# metric on locked predictions: (y_true, y_pred) -> float in [0,1] (larger better)
Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class AuthoredEvalResult:
    """The trusted-side measurement of an admitted method on one split: per-example correctness (the
    cascade's val signal), the scalar metric, prediction latency, and fit time."""
    name: str
    predictions: np.ndarray
    per_example_correct: np.ndarray
    metric: float
    latency_ms: float
    fit_seconds: float


def admit_method(code: str, spec: A.EstimatorSpec, *, family: Optional[str] = None,
                 params: Optional[dict] = None, param_space: Optional[dict] = None,
                 rationale: str = "") -> A.AdmissionReport:
    """Run the frozen three-stage admission gate. Returns the AdmissionReport (admitted True/False + reason).
    Never raises; a rejection carries the failing stage and the specific violation."""
    return A.admit(code, spec, family=family, params=params, param_space=param_space, rationale=rationale)


def evaluate_admitted(report: A.AdmissionReport, X: np.ndarray, y: np.ndarray,
                      train_idx: Sequence[int], eval_idx: Sequence[int], metric_fn: Metric,
                      *, seed: int = 0) -> AuthoredEvalResult:
    """Build the admitted method, fit on TRAIN, predict on EVAL features, and measure on the trusted side.
    Sealed-blind: only X[eval_idx] (features) are passed to predict; eval labels are used only AFTER, here,
    to score. Requires report.admitted and report.factory."""
    if not report.admitted or report.factory is None:
        raise ValueError(f"cannot evaluate a non-admitted method: {report.reason}")
    builder = report.factory.catalog_entry().builder      # public seam: ({params}, seed) -> estimator
    est = builder({}, seed)
    tr = np.asarray(list(train_idx))
    ev = np.asarray(list(eval_idx))
    t0 = time.perf_counter()
    est.fit(X[tr], y[tr])
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = np.asarray(est.predict(X[ev]))                 # NOTE: only features cross; labels stay here
    latency_ms = (time.perf_counter() - t1) / max(1, len(ev)) * 1000.0
    correct = (pred == y[ev]).astype(float)
    metric = float(metric_fn(y[ev], pred))
    return AuthoredEvalResult(report.family or "authored", pred, correct, metric, latency_ms, fit_s)


def run_authored_predict(code: str, role: str, X: np.ndarray, y: np.ndarray,
                         train_idx: Sequence[int], eval_idx: Sequence[int], *, n_classes: int,
                         family: str = "raw", seed: int = 0):
    """ADMIT + EXECUTE an authored code patch on REAL arena features. Admits `code` through the frozen
    three-stage gate with the arena's true (n_features, n_classes), then -- only if admitted -- builds the
    estimator, fits on TRAIN, and predicts EVAL *features*. Returns (predictions | None, AdmissionReport).
    Sealed-blind: only X[eval_idx] crosses to predict(); the caller scores against labels on the trusted
    side. This is the seam an arena uses to make a code-bearing Recipe actually CHANGE the metric (so the
    frozen certifier can judge authored code), not merely be admitted. Never raises for an admitted method;
    a non-admitted method returns (None, report) so the caller can fall back to the recipe's head."""
    spec = A.EstimatorSpec(role=role, n_features=int(X.shape[1]),
                           n_classes=int(n_classes) if role == "classifier" else 1)
    report = admit_method(code, spec, family=family)
    if not report.admitted or report.factory is None:
        return None, report
    builder = report.factory.catalog_entry().builder
    est = builder({}, seed)
    tr = np.asarray(list(train_idx))
    ev = np.asarray(list(eval_idx))
    est.fit(X[tr], y[tr])
    pred = np.asarray(est.predict(X[ev]))          # NOTE: only features cross; labels stay on the caller side
    return pred, report


def to_candidate(eval_result: AuthoredEvalResult, certify_fn: Optional[Callable[[], dict]],
                 *, surrogate_score: Optional[float] = None) -> V.Candidate:
    """Package the trusted-side measurement as a cascade Candidate. val_outcomes is the per-example
    correctness on the eval/val split (drives Tier-1 racing + Tier-2 bound); certify_fn is the caller's
    FROZEN sealed certify call, invoked at most once if the candidate reaches Tier 3."""
    return V.Candidate(
        name=eval_result.name,
        sanity_ok=True,
        val_outcomes=eval_result.per_example_correct.tolist(),
        surrogate_score=surrogate_score,
        certify_fn=certify_fn,
        meta={"authored": True, "latency_ms": eval_result.latency_ms, "fit_seconds": eval_result.fit_seconds},
    )


def measurements_for(eval_result: AuthoredEvalResult, y_true_sealed: Sequence, y_pred_sealed: Sequence,
                     *, labels: Optional[Sequence] = None, ece: Optional[float] = None,
                     cost_usd: Optional[float] = None) -> Measurements:
    """Build the multi-objective gate's Measurements for an authored winner, carrying the measured latency
    (the authored method's per-row predict time) and the sealed-test pairing (produced by the frozen
    certify, passed in here -- never recomputed against the sealed labels by this module)."""
    return Measurements(latency_ms=eval_result.latency_ms, cost_usd=cost_usd, ece=ece,
                        y_true=list(y_true_sealed), y_pred=list(y_pred_sealed),
                        labels=list(labels) if labels is not None else None)


__all__ = ["AuthoredEvalResult", "admit_method", "evaluate_admitted", "run_authored_predict",
           "to_candidate", "measurements_for", "Metric"]
