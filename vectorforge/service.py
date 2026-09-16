"""Lifecycle API: create a goal, draft a plan, approve, run, predict. The CLI and any UI call these.
"""

import time
import uuid

from . import store, runner, ml, science
from .domain import (Goal, VerificationSpec, ExperimentSurface, Budget,
                     DRAFT, AWAITING_APPROVAL, PASSED)


def create_goal(*, name, kind, labels, raw_rows, objective="", task_desc="",
                metric="accuracy", threshold=0.90, max_latency_ms=50.0, min_heldout_n=200,
                surface=None, budget=None, label_meaning=None) -> Goal:
    science.assert_certifiable_metric(metric)   # F4: reject unknown/typo metric up front, never certify-as-accuracy
    gid = f"goal-{uuid.uuid4().hex[:10]}"
    goal = Goal(id=gid, name=name, kind=kind, labels=sorted(labels), objective=objective or name,
                task_desc=task_desc, label_meaning=label_meaning,
                verification=VerificationSpec(metric=metric, threshold=threshold,
                                              max_latency_ms=max_latency_ms, min_heldout_n=min_heldout_n),
                surface=surface or ExperimentSurface(), budget=budget or Budget(),
                status=DRAFT, created_at=store._now())
    store.save(goal)
    goal.raw_path = store.write_rows(gid, "raw", raw_rows)
    return store.save(goal)


def draft_plan(goal: Goal) -> Goal:
    v = goal.verification
    surfaces = [s for s, on in [("model family", goal.surface.model_families),
                                ("feature engineering", goal.surface.feature_engineering),
                                ("class weighting", goal.surface.class_weighting),
                                ("calibration", goal.surface.calibration),
                                ("synthetic data (gated)", goal.surface.synthetic_data),
                                ("active acquisition", goal.surface.active_acquisition)] if on]
    goal.plan = {
        "objective": goal.objective,
        "verification": {"metric": v.metric, "threshold": v.threshold, "max_latency_ms": v.max_latency_ms,
                         "min_heldout_n": v.min_heldout_n,
                         "certifier": "Clopper-Pearson lower bound > threshold, locked test evaluated once"},
        "stages": ["profile", "leakage audit", "stratified split",
                   "adaptive data engine (clean / measured-generate / request-labels, re-audited before adopt)",
                   "adaptive research loop (diagnose -> act on the highest-value allowed axis -> re-diagnose; select on validation only)",
                   "certify on locked test (evaluated once)", "deploy or honest failure report"],
        "adaptive_loop": {
            "data_engine": "before model search, take at most one data move (CLEAN / GENERATE / REQUEST_LABELS); "
                           "synthetic generation requires surface.synthetic_data AND budget.approve_spend AND a "
                           "measured held-out lift; any adopted train is re-audited against the locked test first",
            "research": "multi-round diagnose->act controller over the allowed experiment surface "
                        "(model family / feature engineering / class weighting / calibration); the winner is "
                        "selected on validation and certified once on the locked test",
        },
        "experiment_surfaces": surfaces,
        "budget": {"max_experiments": goal.budget.max_experiments, "max_rounds": goal.budget.max_rounds,
                   "max_spend_usd": goal.budget.max_spend_usd},
        "assumptions": ["train/val/test disjoint; locked test evaluated once",
                        "labels are the supervision signal only; no leakage features"],
    }
    goal.status = AWAITING_APPROVAL
    return store.save(goal)


def approve(goal: Goal) -> Goal:
    goal.dag.setdefault("approval", {"status": "done", "approved": True})
    return store.save(goal)


def run(goal_id: str) -> Goal:
    return runner.run_goal(store.load(goal_id))


def predict(goal_id: str, example):
    """example: a string (text goal) or a feature dict (tabular goal)."""
    goal = store.load(goal_id)
    if goal.status != PASSED or not goal.deployment or goal.deployment.get("status") != "ready":
        raise RuntimeError(f"goal {goal_id} has no deployed model (status={goal.status})")
    art = store.load_artifact(goal.deployment["artifact_path"])
    pipe, cfg, kind = art["pipe"], art["cfg"], art["kind"]
    row = {"text": example} if kind == "text" else {"features": example}
    X, _ = ml._Xy(kind, [row], cfg)
    return str(pipe.predict([X[0]])[0])
