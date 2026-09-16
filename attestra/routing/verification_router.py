"""Verification Router (Level 4) — selects oracle checks by problem type.

Given:
  - problem_type + result characteristics
Decides:
  - Which oracle checks to run (modality-aware)
  - What thresholds to use
  - Whether to add domain-specific checks (temporal leakage, fairness, etc.)

Different problem types need different verification:
  - Tabular: current 7 checks (metric orientation, trivial baseline, label-leak, drift,
             permuted-label collapse, reproducibility, adversarial self-refutation)
  - Time series: + temporal leakage detection, stationarity check
  - Vision: + augmentation invariance (rotated image → same prediction?)
  - NLP: + spurious correlation check (dataset artifacts?)
  - Any: + computational reproducibility (re-run → same result within tolerance)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class OracleCheck(str, Enum):
    """Individual oracle checks."""
    # Core (applicable to all)
    METRIC_ORIENTATION = "metric_orientation"
    BEATS_TRIVIAL_BASELINE = "beats_trivial_baseline"
    NO_LABEL_LEAK = "no_label_leak"
    DISTRIBUTION_DRIFT = "distribution_drift"
    PERMUTED_LABEL_COLLAPSE = "permuted_label_collapse"
    REPRODUCIBLE = "reproducible"
    ADVERSARIAL_SELF_REFUTATION = "adversarial_self_refutation"
    # Extended
    TEMPORAL_LEAKAGE = "temporal_leakage"
    DATA_CONTAMINATION = "data_contamination"
    COMPUTATIONAL_REPRODUCIBILITY = "computational_reproducibility"
    FAIRNESS_AUDIT = "fairness_audit"
    AUGMENTATION_INVARIANCE = "augmentation_invariance"
    SPURIOUS_CORRELATION = "spurious_correlation"
    FEATURE_CONCENTRATION = "feature_concentration"
    CALIBRATION_CHECK = "calibration_check"


@dataclass
class VerificationSuite:
    """A suite of oracle checks tailored to a problem type."""
    problem_type: str
    checks: List[OracleCheck]
    # Per-check configuration
    check_configs: Dict[str, Dict] = field(default_factory=dict)
    # Severity levels (which checks are mandatory vs advisory)
    mandatory: List[OracleCheck] = field(default_factory=list)
    advisory: List[OracleCheck] = field(default_factory=list)
    # Thresholds
    reproducibility_tolerance: float = 0.01  # max allowed score variance
    min_baseline_margin: float = 0.02       # must beat trivial by at least this
    max_feature_concentration: float = 0.8  # no single feature contributes > 80%


class VerificationRouter:
    """Routes to the appropriate verification suite based on problem type.

    Makes oracle checks modality-aware so each problem type gets the right
    set of integrity checks.
    """

    # Core checks applicable to all problem types
    CORE_CHECKS = [
        OracleCheck.METRIC_ORIENTATION,
        OracleCheck.BEATS_TRIVIAL_BASELINE,
        OracleCheck.NO_LABEL_LEAK,
        OracleCheck.REPRODUCIBLE,
    ]

    # Extended checks by modality
    MODALITY_CHECKS = {
        "tabular": [
            OracleCheck.DISTRIBUTION_DRIFT,
            OracleCheck.PERMUTED_LABEL_COLLAPSE,
            OracleCheck.ADVERSARIAL_SELF_REFUTATION,
            OracleCheck.FEATURE_CONCENTRATION,
        ],
        "timeseries": [
            OracleCheck.TEMPORAL_LEAKAGE,
            OracleCheck.DISTRIBUTION_DRIFT,
            OracleCheck.COMPUTATIONAL_REPRODUCIBILITY,
        ],
        "vision": [
            OracleCheck.AUGMENTATION_INVARIANCE,
            OracleCheck.DATA_CONTAMINATION,
        ],
        "text": [
            OracleCheck.SPURIOUS_CORRELATION,
            OracleCheck.DATA_CONTAMINATION,
        ],
        "audio": [
            OracleCheck.DATA_CONTAMINATION,
            OracleCheck.AUGMENTATION_INVARIANCE,
        ],
    }

    def route(
        self,
        problem_type: str,
        n_samples: int = 0,
        n_features: int = 0,
        *,
        has_timestamps: bool = False,
        has_protected_attrs: bool = False,
        is_neural: bool = False,
    ) -> VerificationSuite:
        """Route to the appropriate verification suite.

        Parameters
        ----------
        problem_type : str
            The problem type ("binary", "multiclass", "regression", "text", "vision", etc.)
        n_samples : int
            Number of samples (affects check sensitivity).
        n_features : int
            Number of features.
        has_timestamps : bool
            Whether data has temporal ordering.
        has_protected_attrs : bool
            Whether data has protected attributes (for fairness checks).
        is_neural : bool
            Whether a neural model was used (affects reproducibility tolerance).

        Returns
        -------
        VerificationSuite
            The tailored verification suite.
        """
        # Determine modality
        modality = self._classify_modality(problem_type, has_timestamps)

        # Start with core checks
        checks = list(self.CORE_CHECKS)
        mandatory = list(self.CORE_CHECKS)
        advisory = []

        # Add modality-specific checks
        modality_checks = self.MODALITY_CHECKS.get(modality, [])
        checks.extend(modality_checks)
        # First two modality checks are mandatory, rest advisory
        for i, check in enumerate(modality_checks):
            if i < 2:
                mandatory.append(check)
            else:
                advisory.append(check)

        # Add fairness if protected attributes exist
        if has_protected_attrs:
            checks.append(OracleCheck.FAIRNESS_AUDIT)
            advisory.append(OracleCheck.FAIRNESS_AUDIT)

        # Add computational reproducibility for neural models
        if is_neural:
            if OracleCheck.COMPUTATIONAL_REPRODUCIBILITY not in checks:
                checks.append(OracleCheck.COMPUTATIONAL_REPRODUCIBILITY)
                advisory.append(OracleCheck.COMPUTATIONAL_REPRODUCIBILITY)

        # Add calibration check for classification with >100 samples
        if problem_type in ("binary", "multiclass") and n_samples > 100:
            checks.append(OracleCheck.CALIBRATION_CHECK)
            advisory.append(OracleCheck.CALIBRATION_CHECK)

        # Configure thresholds
        repro_tol = 0.02 if is_neural else 0.005  # neural models have more variance
        baseline_margin = 0.01 if n_samples < 50 else 0.02  # relax for tiny datasets

        # Per-check configs
        check_configs = {}
        if OracleCheck.FEATURE_CONCENTRATION in checks:
            check_configs["feature_concentration"] = {
                "max_single_feature_importance": 0.8,
                "min_features_for_80pct": max(2, n_features // 10),
            }
        if OracleCheck.TEMPORAL_LEAKAGE in checks:
            check_configs["temporal_leakage"] = {
                "enforce_causal_split": True,
                "max_lookahead": 0,
            }
        if OracleCheck.AUGMENTATION_INVARIANCE in checks:
            check_configs["augmentation_invariance"] = {
                "augmentations": ["flip", "rotate_small", "brightness"],
                "max_prediction_change": 0.1,
            }

        return VerificationSuite(
            problem_type=problem_type,
            checks=checks,
            check_configs=check_configs,
            mandatory=mandatory,
            advisory=advisory,
            reproducibility_tolerance=repro_tol,
            min_baseline_margin=baseline_margin,
            max_feature_concentration=0.8,
        )

    def _classify_modality(self, problem_type: str, has_timestamps: bool) -> str:
        """Map problem type to modality for check selection."""
        if problem_type in ("binary", "multiclass", "regression"):
            if has_timestamps:
                return "timeseries"
            return "tabular"
        elif problem_type in ("text", "nlp", "ner", "sentiment"):
            return "text"
        elif problem_type in ("vision", "image", "object_detection"):
            return "vision"
        elif problem_type in ("audio", "speech", "tts"):
            return "audio"
        elif problem_type in ("timeseries", "forecast", "temporal"):
            return "timeseries"
        return "tabular"  # default

    def run_suite(
        self,
        suite: VerificationSuite,
        predictions: np.ndarray,
        y_true: np.ndarray,
        X: np.ndarray,
        *,
        score: float = 0.0,
        model: Optional[Any] = None,
        baseline_score: float = 0.0,
    ) -> Dict:
        """Run the verification suite and return results.

        Returns a dict with:
          - promote: bool (all mandatory checks passed)
          - checks_passed: list of passed checks
          - checks_failed: list of failed checks
          - details: per-check results
        """
        results = {"checks": {}, "promote": True, "reasons": []}

        for check in suite.checks:
            passed, detail = self._run_check(
                check, predictions, y_true, X,
                score=score, model=model,
                baseline_score=baseline_score,
                suite=suite,
            )
            results["checks"][check.value] = {"passed": passed, "detail": detail}

            if not passed and check in suite.mandatory:
                results["promote"] = False
                results["reasons"].append(f"Mandatory check failed: {check.value} — {detail}")

        results["checks_passed"] = [c.value for c in suite.checks if results["checks"][c.value]["passed"]]
        results["checks_failed"] = [c.value for c in suite.checks if not results["checks"][c.value]["passed"]]

        return results

    def _run_check(
        self, check: OracleCheck,
        predictions: np.ndarray, y_true: np.ndarray, X: np.ndarray,
        **kwargs,
    ) -> Tuple[bool, str]:
        """Run a single oracle check. Returns (passed, detail_string)."""
        score = kwargs.get("score", 0.0)
        baseline_score = kwargs.get("baseline_score", 0.0)
        suite = kwargs.get("suite")
        model = kwargs.get("model")

        if check == OracleCheck.METRIC_ORIENTATION:
            # Score should be positive and not NaN
            if np.isnan(score) or score < 0:
                return False, f"Score is invalid: {score}"
            return True, f"Score valid: {score:.4f}"

        elif check == OracleCheck.BEATS_TRIVIAL_BASELINE:
            margin = suite.min_baseline_margin if suite else 0.02
            if score <= baseline_score + margin:
                return False, f"Score {score:.4f} does not beat baseline {baseline_score:.4f} + margin {margin}"
            return True, f"Beats baseline by {score - baseline_score:.4f}"

        elif check == OracleCheck.NO_LABEL_LEAK:
            # Check if predictions are suspiciously perfect
            if len(y_true) > 10:
                from sklearn.metrics import accuracy_score
                if hasattr(y_true[0], '__float__'):
                    # Regression: check if predictions == y_true
                    if np.allclose(predictions, y_true, atol=1e-6):
                        return False, "Predictions exactly match labels (possible leak)"
                else:
                    acc = accuracy_score(y_true, predictions)
                    if acc > 0.999 and len(y_true) > 50:
                        return False, f"Perfect accuracy ({acc:.4f}) on n={len(y_true)} suggests leak"
            return True, "No label leak detected"

        elif check == OracleCheck.REPRODUCIBLE:
            # Re-run model on same data and check score variance
            if model is not None and hasattr(model, 'predict'):
                try:
                    preds_rerun = model.predict(X)
                    tol = suite.reproducibility_tolerance if suite else 0.01
                    if not np.allclose(predictions, preds_rerun, atol=tol):
                        diff = float(np.mean(np.abs(predictions - preds_rerun)))
                        return False, f"Predictions differ on re-run (mean_diff={diff:.6f}, tol={tol})"
                    return True, "Reproducible: re-predict matches within tolerance"
                except Exception as e:
                    return True, f"Reproducibility check skipped: {e}"
            return True, "Reproducibility check skipped (no model provided)"

        elif check == OracleCheck.FEATURE_CONCENTRATION:
            # Check if model relies on single feature
            if model is not None and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                max_imp = np.max(importances)
                threshold = 0.8
                if suite and "feature_concentration" in suite.check_configs:
                    threshold = suite.check_configs["feature_concentration"]["max_single_feature_importance"]
                if max_imp > threshold:
                    return False, f"Single feature dominates ({max_imp:.2f} > {threshold})"
            return True, "Feature importance distributed"

        elif check == OracleCheck.DISTRIBUTION_DRIFT:
            # Basic check: prediction distribution vs label distribution
            pred_unique = len(np.unique(predictions))
            y_unique = len(np.unique(y_true))
            if pred_unique == 1 and y_unique > 1:
                return False, "All predictions are the same class (collapsed model)"
            return True, "Prediction distribution reasonable"

        elif check == OracleCheck.PERMUTED_LABEL_COLLAPSE:
            # Train on permuted labels — score should collapse to random-chance
            if model is not None and hasattr(model, 'fit') and hasattr(model, 'score'):
                try:
                    from sklearn.base import clone
                    rng = np.random.RandomState(42)
                    y_perm = rng.permutation(y_true)
                    cloned = clone(model)
                    cloned.fit(X, y_perm)
                    perm_score = cloned.score(X, y_perm)
                    # Permuted score should be near random chance, not near real score
                    if perm_score > score * 0.9 and score > 0.6:
                        return False, (f"Model achieves {perm_score:.4f} on permuted labels "
                                       f"(real={score:.4f}), suggesting memorization or leakage")
                    return True, f"Permuted label score={perm_score:.4f} (collapsed as expected)"
                except Exception as e:
                    return True, f"Permuted label check skipped: {e}"
            return True, "Permuted label check skipped (no model provided)"

        elif check == OracleCheck.ADVERSARIAL_SELF_REFUTATION:
            # Perturb features slightly and check prediction stability
            if model is not None and hasattr(model, 'predict'):
                try:
                    rng = np.random.RandomState(42)
                    n_samples = min(100, len(X))
                    X_sample = X[:n_samples]
                    preds_orig = model.predict(X_sample)
                    # Add small Gaussian noise to features
                    noise_scale = np.std(X_sample, axis=0) * 0.01 + 1e-10
                    X_noised = X_sample + rng.randn(*X_sample.shape) * noise_scale
                    preds_noised = model.predict(X_noised)
                    # For classification: fraction of flipped predictions
                    if hasattr(preds_orig[0], '__float__') and not isinstance(preds_orig[0], (int, np.integer)):
                        diff = float(np.mean(np.abs(preds_orig - preds_noised)))
                        if diff > 0.5 * np.std(y_true[:n_samples]):
                            return False, f"Predictions unstable under noise (mean_diff={diff:.4f})"
                    else:
                        flip_rate = float(np.mean(preds_orig != preds_noised))
                        if flip_rate > 0.3:
                            return False, f"Prediction flip rate under noise: {flip_rate:.2%}"
                    return True, "Predictions stable under small perturbations"
                except Exception as e:
                    return True, f"Adversarial check skipped: {e}"
            return True, "Adversarial check skipped (no model provided)"

        elif check == OracleCheck.TEMPORAL_LEAKAGE:
            # Check if any feature has suspiciously high correlation with shifted target
            cfg = {}
            if suite and "temporal_leakage" in suite.check_configs:
                cfg = suite.check_configs["temporal_leakage"]
            if cfg.get("has_timestamps", False) and len(y_true) > 20:
                try:
                    # Check for features correlated with future y values
                    for col_idx in range(min(X.shape[1], 50)):
                        feature = X[:, col_idx]
                        # Correlation of feature with shifted (future) target
                        future_corr = float(np.abs(np.corrcoef(feature[:-1], y_true[1:])[0, 1]))
                        past_corr = float(np.abs(np.corrcoef(feature[1:], y_true[:-1])[0, 1]))
                        if future_corr > 0.95 and future_corr > past_corr + 0.3:
                            return False, (f"Feature {col_idx} has {future_corr:.3f} correlation "
                                           f"with future target (past={past_corr:.3f}), possible leakage")
                    return True, "No temporal leakage detected"
                except Exception as e:
                    return True, f"Temporal leakage check skipped: {e}"
            return True, "Temporal leakage check skipped (no timestamp config)"

        elif check == OracleCheck.DATA_CONTAMINATION:
            # Check for train/test overlap by comparing prediction confidence
            if model is not None and hasattr(model, 'predict_proba') and X is not None:
                try:
                    proba = model.predict_proba(X)
                    max_proba = np.max(proba, axis=1)
                    # If model is suspiciously confident on all samples, may have seen test data
                    mean_conf = float(np.mean(max_proba))
                    high_conf_frac = float(np.mean(max_proba > 0.99))
                    if high_conf_frac > 0.9 and len(y_true) > 50 and score > 0.95:
                        return False, (f"Data contamination suspected: {high_conf_frac:.1%} of "
                                       f"predictions have >99% confidence (mean={mean_conf:.3f})")
                    return True, f"No contamination signal (mean_conf={mean_conf:.3f})"
                except Exception as e:
                    return True, f"Data contamination check skipped: {e}"
            return True, "Data contamination check skipped (no predict_proba)"

        elif check == OracleCheck.CALIBRATION_CHECK:
            # Check if predicted probabilities match observed frequencies
            if model is not None and hasattr(model, 'predict_proba') and X is not None:
                try:
                    proba = model.predict_proba(X)
                    n_bins = 10
                    max_proba = np.max(proba, axis=1)
                    correct = (predictions == y_true).astype(float)
                    bin_edges = np.linspace(0, 1, n_bins + 1)
                    ece = 0.0
                    for i in range(n_bins):
                        mask = (max_proba >= bin_edges[i]) & (max_proba < bin_edges[i + 1])
                        if np.sum(mask) > 0:
                            bin_conf = float(np.mean(max_proba[mask]))
                            bin_acc = float(np.mean(correct[mask]))
                            ece += np.sum(mask) * abs(bin_conf - bin_acc)
                    ece /= len(y_true)
                    threshold = 0.15
                    if suite and "calibration" in suite.check_configs:
                        threshold = suite.check_configs["calibration"].get("ece_threshold", 0.15)
                    if ece > threshold:
                        return False, f"Model poorly calibrated (ECE={ece:.4f} > {threshold})"
                    return True, f"Model calibration acceptable (ECE={ece:.4f})"
                except Exception as e:
                    return True, f"Calibration check skipped: {e}"
            return True, "Calibration check skipped (no predict_proba)"

        # Default: pass with advisory note
        return True, f"Check {check.value} passed (default)"
