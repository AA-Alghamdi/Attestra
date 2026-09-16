"""Tests for the Phase-5 data-centric module (frontier/data_ops.py).

Run standalone:
    cd <repo> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_data_ops.py

These assert BEHAVIOR (not just import): the interventions actually transform the train side,
the held-out val/sealed sets are never touched by synthetic rows, the audit detects imbalance /
duplication, and certify_data_change runs both arms through the frozen sealed gate with exactly
one peek per arm and returns an honest go/no-go.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import certify
from frontier.data_ops import (
    Deduplicate,
    Mixup,
    SMOTEOversample,
    GaussianClassSynthesize,
    audit_task,
    audit_rows,
    build_intervened_task,
    certify_data_change,
)
from frontier.engine import EngineConfig
from frontier.task import Task


def _imbalanced_clf(n_major=240, n_minor=30, d=8, seed=0):
    """Two-class synthetic set with a strong imbalance and separable-ish structure."""
    rng = np.random.default_rng(seed)
    Xmaj = rng.normal(0.0, 1.0, size=(n_major, d))
    Xmin = rng.normal(2.2, 1.0, size=(n_minor, d))
    X = np.vstack([Xmaj, Xmin])
    y = np.array(["0"] * n_major + ["1"] * n_minor)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _imbalanced_task(theta=0.55):
    X, y = _imbalanced_clf()
    return Task(X=X, y=y, kind="classification", theta=theta,
                metric="balanced_accuracy", name="imb")


def test_audit_detects_imbalance_and_recommends_oversampling():
    task = _imbalanced_task()
    audit, splits = audit_task(task, seed=0)
    assert audit.kind == "classification"
    assert audit.imbalance_ratio > 1.5, f"audit should see imbalance, got {audit.imbalance_ratio}"
    assert audit.minority_classes, "minority class must be identified"
    recs = audit.recommend_interventions()
    names = {getattr(r, "name", "") for r in recs}
    assert "smote" in names, f"imbalance must recommend SMOTE, got {names}"
    # drift reference came from sealed rows -> drift_scores populated, sealed never modified.
    assert audit.drift_scores, "drift reference (vs sealed) should be computed"
    print(f"[ok] audit: ratio={audit.imbalance_ratio} minority={audit.minority_classes} "
          f"recs={sorted(names)}")


def test_smote_oversamples_minority_and_balances():
    task = _imbalanced_task()
    _, splits = audit_task(task, seed=0)
    rng = np.random.default_rng(0)
    before = audit_rows(splits.train_rows, "classification")
    new_train = SMOTEOversample(k=5, target="balance").apply(
        [dict(r) for r in splits.train_rows], "classification", rng)
    after = audit_rows(new_train, "classification")
    assert len(new_train) > len(splits.train_rows), "SMOTE must add rows"
    assert after.imbalance_ratio <= before.imbalance_ratio, "SMOTE must reduce imbalance"
    assert abs(after.imbalance_ratio - 1.0) < 1e-6, f"balance target should equalize: {after.class_counts}"
    print(f"[ok] smote: {before.class_counts} -> {after.class_counts}")


def test_interventions_never_touch_val_or_sealed():
    """Core integrity property: synthetic rows must not collide with held-out feature vectors."""
    task = _imbalanced_task()
    _, splits = audit_task(task, seed=0)
    held = {tuple(r["_x"]) for r in splits.val_rows} | {tuple(r["_x"]) for r in splits.sealed_rows}
    rng = np.random.default_rng(0)
    for interv in [SMOTEOversample(target="balance"), Mixup(alpha=0.2, n_synth_frac=0.5),
                   GaussianClassSynthesize(target="balance"), Deduplicate()]:
        new_train = interv.apply([dict(r) for r in splits.train_rows], "classification", rng)
        synth_keys = {tuple(r["_x"]) for r in new_train}
        # The intervention's own guard is also exercised via build_intervened_task below.
        assert not (synth_keys & held) or interv.name == "dedup", \
            f"{interv.name} produced a row colliding with val/sealed"
    # build_intervened_task hard-asserts the same invariant and rebuilds a clean variant Task.
    variant, vsplits = build_intervened_task(task, SMOTEOversample(target="balance"), seed=0)
    assert variant.kind == "classification"
    print("[ok] no intervention injected synthetic rows into val/sealed (guard held)")


def test_mixup_and_gaussian_add_rows():
    task = _imbalanced_task()
    _, splits = audit_task(task, seed=0)
    rng = np.random.default_rng(1)
    base_n = len(splits.train_rows)
    mx = Mixup(alpha=0.4, n_synth_frac=0.5).apply([dict(r) for r in splits.train_rows],
                                                  "classification", rng)
    gs = GaussianClassSynthesize(target="balance").apply([dict(r) for r in splits.train_rows],
                                                         "classification", rng)
    assert len(mx) == base_n + int(round(0.5 * base_n)), "mixup adds n_synth_frac*n rows"
    assert len(gs) > base_n, "gaussian synth tops up the minority class"
    # mixup classification labels must remain the original valid class strings (no soft labels)
    assert {r["target"] for r in mx} <= {"0", "1"}, "mixup must keep hard, valid clf labels"
    print(f"[ok] mixup {base_n}->{len(mx)}, gauss {base_n}->{len(gs)}, labels valid")


def test_deduplicate_removes_exact_copies():
    rng = np.random.default_rng(0)
    base = [{"target": "0", "features": {"f0": 1.0}, "_x": [1.0]} for _ in range(3)]
    base += [{"target": "1", "features": {"f0": 2.0}, "_x": [2.0]}]
    out = Deduplicate().apply(base, "classification", rng)
    assert len(out) == 2, f"3 identical + 1 unique -> 2 rows, got {len(out)}"
    a = audit_rows(base, "classification")
    assert a.duplicate_fraction > 0.0, "audit must report duplicate fraction"
    print(f"[ok] dedup 4->{len(out)}, dup_fraction={a.duplicate_fraction}")


def test_certify_data_change_runs_both_arms_through_sealed_gate():
    """End-to-end: both arms certified via the frozen path; each touches sealed exactly once;
    the verdict is an honest go/no-go derived from the two sealed lower bounds."""
    task = _imbalanced_task(theta=0.55)
    cfg = EngineConfig(rounds=1, wall_seconds=45, cpu_seconds=40, seed=0)
    report = certify_data_change(task, SMOTEOversample(k=5, target="balance"), config=cfg, seed=0)

    assert report.invariants_ok, f"train-only invariant must hold: {report.notes}"
    assert report.intervention.n_train > report.base.n_train, "intervention arm has oversampled train"
    # Both arms certified the winner on the SAME sealed split, one peek each.
    for arm in (report.base, report.intervention):
        assert arm.certificate is not None, f"{arm.label} should reach the sealed certifier"
        assert arm.certificate.get("peeks") == 1, f"{arm.label} must use exactly one sealed peek"
        lb = arm.certificate["lower_bound"]
        assert lb <= arm.certificate["observed"] + 1e-9, "lower bound must not exceed observed"
    assert report.decision in ("go", "no-go")
    # caused_lift can only be True if the intervention arm itself certifies (never a relabeled val).
    if report.caused_lift:
        assert report.intervention.certified and report.lower_bound_delta > 0
    print(report.summary())


def test_smote_noop_on_regression_is_honest():
    """SMOTE/Gaussian oversampling is undefined for regression -> honest no-op, not fabrication."""
    rng = np.random.default_rng(0)
    rows = [{"target": float(i), "features": {"f0": float(i)}, "_x": [float(i)]} for i in range(20)]
    out = SMOTEOversample().apply(rows, "regression", rng)
    assert len(out) == len(rows), "SMOTE must be a no-op on regression"
    print("[ok] smote honest no-op on regression")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} data_ops tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
