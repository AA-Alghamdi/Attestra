"""Tests for Phase-5 scientific experiment design (frontier/experiment.py).

Run standalone:
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_experiment.py

These exercise the module on REAL sklearn datasets and assert BEHAVIOR:
  - a beneficial factor (standardizing an RBF-SVM) returns a certified GO;
  - a no-op factor (scaling a scale-invariant tree) returns an honest NO-GO;
  - the sealed test pays multiplicity for BOTH peeks (checks==2);
  - a failing arm yields an honest decline (go=False), not a crash or a fake number;
  - the ablation runner contrasts levels against a shared baseline on one locked test.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import certify
from frontier.experiment import (
    Experiment, Verdict, run_experiment, run_ablation, AblationResult,
    scale_factor, poly_factor, noop_factor, factor_program, design_experiment,
)
from frontier.program import Program
from frontier.task import Task


def _wine_task(theta=0.70):
    """3-class classification where feature scales differ by orders of magnitude, so
    standardizing the RBF-SVM helps a lot (a strong positive control)."""
    from sklearn.datasets import load_wine
    d = load_wine()
    return Task(X=d.data, y=d.target.astype(str), kind="classification",
                theta=theta, name="wine")


def _bc_task(theta=0.80):
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return Task(X=d.data, y=d.target.astype(str), kind="classification",
                theta=theta, name="bc")


# --------------------------------------------------------------------------- beneficial -> GO

def test_beneficial_factor_is_go():
    task = _wine_task(theta=0.60)
    splits = certify.make_splits(task, seed=0)
    exp = Experiment(
        hypothesis="Standardizing features helps the RBF-SVM on wine.",
        factor_varied="standardize_features",
        controls_held_fixed={"base": "svc_rbf", "hyperparams": "fixed",
                             "splits": "shared", "seed": 0},
        decision_rule="go iff treatment sealed lower bound > control + margin",
    )
    treat, ctrl = scale_factor(task.kind, base="svc_rbf")
    v = run_experiment(exp, task, splits, treat, ctrl, wall_seconds=60, cpu_seconds=50,
                       margin=0.0)
    print(v.report())
    assert isinstance(v, Verdict)
    assert v.treatment.ok and v.control.ok, "both arms must run"
    assert v.go, "standardizing the RBF-SVM should be a GO (large, real lift)"
    assert v.lift_lower_bound is not None and v.lift_lower_bound > 0.0
    assert v.checks == 2, f"two sealed peeks must be paid (checks=2), got {v.checks}"
    # the GO is on the conservative LOWER bound, not the point estimate
    assert v.treatment.lower_bound <= v.treatment.observed + 1e-9
    print(f"[ok] beneficial factor GO: lift_lb={v.lift_lower_bound} checks={v.checks}")


# --------------------------------------------------------------------------- no-op -> NO-GO

def test_noop_factor_is_nogo():
    task = _bc_task(theta=0.80)
    splits = certify.make_splits(task, seed=0)
    exp = Experiment(
        hypothesis="Scaling features helps the random forest on breast-cancer.",
        factor_varied="standardize_features",
        controls_held_fixed={"base": "rf", "hyperparams": "fixed",
                             "splits": "shared", "seed": 0},
        decision_rule="go iff treatment sealed lower bound > control + margin",
    )
    # scaling a tree is provably inert -> NO-GO. Use a small positive margin so even a
    # tiny noise lift cannot manufacture a GO.
    treat, ctrl = noop_factor(task.kind, base="rf")
    v = run_experiment(exp, task, splits, treat, ctrl, wall_seconds=60, cpu_seconds=50,
                       margin=0.01)
    print(v.report())
    assert v.treatment.ok and v.control.ok, "both arms must run"
    assert not v.go, "scaling a scale-invariant tree must be an honest NO-GO"
    assert v.checks == 2
    print(f"[ok] no-op factor NO-GO: lift_lb={v.lift_lower_bound} (margin=0.01)")


# --------------------------------------------------------------------------- honest decline

def test_failing_arm_is_honest_decline():
    task = _bc_task(theta=0.80)
    splits = certify.make_splits(task, seed=0)
    exp = Experiment(hypothesis="A broken treatment vs a working control.",
                     factor_varied="broken_step",
                     controls_held_fixed={"base": "logreg"},
                     decision_rule="go iff treatment lower bound > control + margin")
    broken = Program(code="def build_estimator():\n    raise RuntimeError('boom')\n",
                     source="experiment", label="broken:treat")
    ctrl = factor_program(task.kind, {"base": "logreg", "scale": True}, label_suffix="ctrl")
    v = run_experiment(exp, task, splits, broken, ctrl, wall_seconds=60, cpu_seconds=50)
    print(v.report())
    assert not v.go, "a failed arm cannot be a GO"
    assert not v.treatment.ok and v.treatment.error_kind in ("build", "fit", "other")
    assert v.lift_lower_bound is None, "no fabricated lift when an arm failed"
    print(f"[ok] failing arm honest decline: [{v.treatment.error_kind}]")


# --------------------------------------------------------------------------- multiplicity honesty

def test_multiplicity_is_paid():
    """Both arms certified -> the SECOND arm's certificate must carry checks=2 (Bonferroni),
    i.e. the cost of asking two questions of one locked test is paid, not laundered."""
    task = _wine_task(theta=0.50)
    splits = certify.make_splits(task, seed=1)
    treat, ctrl = scale_factor(task.kind, base="svc_rbf")
    exp = design_experiment("classify wine cultivars", "standardize_features", client=None)
    v = run_experiment(exp, task, splits, treat, ctrl, wall_seconds=60, cpu_seconds=50)
    assert v.treatment.certificate["checks"] == 1, "first peek is checks=1"
    assert v.control.certificate["checks"] == 2, "second peek must escalate to checks=2"
    assert v.treatment.certificate["sealed_digest"] == v.control.certificate["sealed_digest"], \
        "both arms must be certified on the SAME locked test"
    print(f"[ok] multiplicity paid: treat checks={v.treatment.certificate['checks']}, "
          f"control checks={v.control.certificate['checks']}, "
          f"shared digest={v.treatment.certificate['sealed_digest'][:12]}")


# --------------------------------------------------------------------------- ablation runner

def test_ablation_runner():
    task = _wine_task(theta=0.50)
    splits = certify.make_splits(task, seed=0)
    exp = Experiment(hypothesis="Preprocessing levels for the RBF-SVM on wine.",
                     factor_varied="preprocessing",
                     controls_held_fixed={"base": "svc_rbf"},
                     decision_rule="level is a go iff its sealed lower bound beats baseline")
    baseline = factor_program(task.kind, {"base": "svc_rbf"}, label_suffix="none")
    levels = {
        "scale": factor_program(task.kind, {"base": "svc_rbf", "scale": True},
                                label_suffix="scale"),
        "scale+poly2": factor_program(task.kind, {"base": "svc_rbf", "scale": True, "poly": 2},
                                      label_suffix="poly"),
    }
    abl = run_ablation(exp, task, splits, baseline, levels, wall_seconds=60, cpu_seconds=50,
                       margin=0.0)
    print(abl.report())
    assert isinstance(abl, AblationResult)
    assert abl.baseline.ok and all(a.ok for a in abl.levels), "all arms must run"
    # baseline + 2 levels = 3 peeks of one locked test
    assert abl.checks == 3, f"three sealed peeks must be paid, got {abl.checks}"
    # scaling clearly beats the unscaled baseline -> at least one go level, and the best is scale-based
    assert "scale" in abl.go_levels, "scaling should beat the unscaled RBF-SVM baseline"
    assert abl.best_level in ("scale", "scale+poly2")
    print(f"[ok] ablation: go_levels={abl.go_levels} best={abl.best_level} "
          f"lift_lb={abl.best_lift_lower} checks={abl.checks}")


# --------------------------------------------------------------------------- design hook degrade

def test_design_experiment_degrades_honestly():
    exp = design_experiment("classify wine", "standardize_features", client=None)
    assert exp.metadata["authored_by"] == "deterministic_template"
    assert "standardize_features" == exp.factor_varied
    # with a fake deterministic "LLM" client, the hypothesis text comes from the client
    exp2 = design_experiment("classify wine", "standardize_features",
                             client=lambda p: "Scaling sharpens the RBF kernel margins.")
    assert exp2.metadata["authored_by"] == "llm"
    assert "Scaling sharpens" in exp2.hypothesis
    print(f"[ok] design hook: template + llm both produce a pre-registerable Experiment")


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
    print(f"\n{len(fns) - failed}/{len(fns)} experiment tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
