"""Attestra core: frozen statistical verification spine.

The certifier, metrics, leakage auditor, splits, and calibration live here.
Pure numpy + stdlib -- no framework dependencies, reproducible numbers.

MIGRATION NOTE: This is the canonical home for science.py. The old path
(vectorforge.science) re-exports from here for backward compatibility.
"""
from .science import (  # noqa: F401
    # metrics
    accuracy, balanced_accuracy, macro_f1, r2_score, neg_rmse, neg_mae,
    score_metric, score_regression_metric, per_class_recall,
    KNOWN_METRICS, assert_certifiable_metric,
    # certifiers
    certify_accuracy, certify_regression,
    clopper_pearson_lower, bootstrap_lower, bootstrap_metric_lower,
    # calibration
    expected_calibration_error,
    # leakage
    audit,
    # splits
    make_splits,
)
