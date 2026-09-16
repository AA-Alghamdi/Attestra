"""The supervised classifier goal runner: the product's single entry point for "give me a goal and a
verification standard, get back a certified model or an honest non-result."

This module is a thin, defensible FACADE over the verified spine. It does not reinvent any science: it
reuses `service` (create/plan/approve/run/predict), the DAG `runner`, the `science` certifier/auditor,
and the `ml` landscape scan. It adds exactly two things that the product needs and that nothing else
owns:

  1. `run_classifier_goal(...)` -- one call that drives a text-or-tabular goal through the full
     lifecycle (create -> draft_plan -> approve -> run) and returns the finished, durable Goal. Every
     promotion is still gated ONLY by `science.certify_accuracy` (accuracy) or `science.bootstrap_lower`
     (other metrics) on the locked test, evaluated exactly once inside the runner. This wrapper never
     touches the test set, never lowers a threshold, and never fabricates an outcome.

  2. `decide(goal)` -- maps a finished goal's durable state to a single pre-registered ACTION label
     (the decision a competent ML brain would emit). This is what the outcome-aware battery scores. It
     reads only fields the runner already wrote (status, certificate, failure_report, needs_input,
     leaderboard); it performs no new inference and cannot upgrade a non-certified goal to a certified
     action.

  3. `strong_solver_reference(...)` -- runs the full sklearn family scan (`ml.landscape_scan`) and the
     real certifier on a scenario WITHOUT mutating any goal, to answer "is this task actually solvable?"
     for the battery's masked-execution check. It is the strong-solver reference, not a shortcut: it
     selects on validation and certifies once on the locked test, identically to the runner.

Self-test: `python -m vectorforge.classifier` exercises text (core-ml-acceptance) and tabular (rugged)
on the real frozen datasets and prints PASS/FAIL.
"""

from __future__ import annotations

import numpy as np

from . import service, store, science, ml
from .domain import (Goal, PASSED, FAILED, BLOCKED, NEEDS_INPUT,
                     ExperimentSurface, Budget)


# ---- action vocabulary (pre-registered; the battery scores against these) -------------------------
PROMOTE = "PROMOTE"                              # certified a plain winner
REBALANCE = "REBALANCE"                          # certified, winner used class weighting
EXPAND_REPRESENTATION = "EXPAND_REPRESENTATION"  # certified, winner used feature interactions
ACQUIRE_LABELS = "ACQUIRE_LABELS"                # data-limited: too few held-out / train labels
COLLECT_MORE_HELDOUT = "COLLECT_MORE_HELDOUT"    # point estimate clears bar but evidence is underpowered
FIX_LEAKAGE = "FIX_LEAKAGE"                      # leakage / blocked
STOP_HONEST_FAIL = "STOP_HONEST_FAIL"            # genuinely could not certify; stop honestly


# =================================================================================== the goal runner
def run_classifier_goal(*, name, kind, labels, raw_rows, metric="accuracy", threshold=0.90,
                        max_latency_ms=50.0, min_heldout_n=200, objective="", task_desc="",
                        surface: ExperimentSurface | None = None, budget: Budget | None = None,
                        label_meaning=None) -> Goal:
    """Drive a supervised classifier goal end to end and return the finished, durable Goal.

    kind: "text" (rows {id,text,target}) or "tabular" (rows {features:{...},target}). Promotion is gated
    only by the runner's certifier on the locked test; this wrapper adds no path that can certify.
    """
    if kind not in ("text", "tabular"):
        raise ValueError(f"kind must be 'text' or 'tabular', got {kind!r}")
    if len(set(labels)) < 2:
        raise ValueError("a classifier goal needs at least two distinct labels")
    goal = service.create_goal(
        name=name, kind=kind, labels=list(labels), raw_rows=list(raw_rows),
        objective=objective or name, task_desc=task_desc, metric=metric, threshold=threshold,
        max_latency_ms=max_latency_ms, min_heldout_n=min_heldout_n,
        surface=surface, budget=budget, label_meaning=label_meaning)
    goal = service.draft_plan(goal)
    goal = service.approve(goal)
    return service.run(goal.id)


def predict(goal_id, example):
    """Serve from a goal's deployed, reloaded artifact. Raises if the goal is not PASSED/deployed."""
    return service.predict(goal_id, example)


# ============================================================================ decision mapping (read-only)
def _winner_family(goal: Goal) -> str:
    """The certified winner's candidate string from durable state (e.g. 'logistic_balanced|inter').

    The adaptive runner records the winner under the 'research' stage; the older single-scan runner used
    'landscape'. Read whichever is present so decide() recovers the winning family (and thus the
    REBALANCE / EXPAND_REPRESENTATION action) across both runner generations."""
    lp = (goal.dag.get("research") or goal.dag.get("landscape") or {})
    return str(lp.get("winner") or "")


def decide(goal: Goal) -> str:
    """Map a FINISHED goal to one pre-registered action. Pure read of durable fields; never re-infers,
    never upgrades a non-certified goal to a certified action. This is the decision the battery grades.
    """
    st = goal.status

    if st == PASSED:
        fam = _winner_family(goal)
        if "inter" in fam:
            return EXPAND_REPRESENTATION
        if "balanced" in fam:
            return REBALANCE
        return PROMOTE

    if st == NEEDS_INPUT:
        # the runner only raises NEEDS_INPUT for the data-limited (too-few-labels) case
        return ACQUIRE_LABELS

    if st == BLOCKED:
        return FIX_LEAKAGE

    if st == FAILED:
        cert = goal.certificate or {}
        # leakage that survives to the certify stage is reported as a blocked certificate
        if not cert.get("leakage_passed", True):
            return FIX_LEAKAGE
        # underpowered: the POINT estimate clears the bar but the lower confidence bound does not.
        # That is "collect more held-out evidence", not "give up".
        observed = float(cert.get("observed", 0.0))
        lower = float(cert.get("lower_bound", 0.0))
        thr = float(cert.get("threshold", goal.verification.threshold))
        if observed >= thr > lower:
            return COLLECT_MORE_HELDOUT
        return STOP_HONEST_FAIL

    # DRAFT / AWAITING_APPROVAL / RUNNING / CANCELLED: not a terminal classifier outcome
    return STOP_HONEST_FAIL


def certified(goal: Goal) -> bool:
    """True iff the runner promoted (PASSED + a 'certified' certificate). Single source of truth."""
    return goal.status == PASSED and (goal.certificate or {}).get("decision") == "certified"


# ===================================================================== strong-solver reference (no mutation)
def strong_solver_reference(*, kind, train, val, test, labels, metric="accuracy", threshold=0.90,
                            alpha=0.05) -> dict:
    """Run the full sklearn family scan + real certifier on a scenario WITHOUT creating/mutating a goal.

    Answers "is this task actually solvable?" for the battery's masked-execution check. Identical rigor
    to the runner: select the winner on validation only, certify ONCE on the locked test. Returns
    {certified, observed, lower_bound, winner, n}. A non-text/tabular kind raises.
    """
    if not train or not val or not test:
        return {"certified": False, "observed": 0.0, "lower_bound": 0.0,
                "winner": None, "n": len(test), "reason": "empty split"}
    scan = ml.landscape_scan(kind, train, val, test, labels, metric)
    if scan.get("winner_pipe") is None:
        # every candidate failed to fit (e.g. a degenerate single-class split): the reference cannot
        # certify, which is an honest "not solvable by the strong solver here" rather than a crash.
        return {"certified": False, "observed": 0.0, "lower_bound": 0.0,
                "winner": scan.get("winner"), "n": len(test), "reason": "no fittable candidate"}
    pipe, cfg = scan["winner_pipe"], scan["winner_cfg"]
    ev = ml.evaluate(pipe, kind, test, cfg, labels, metric)
    n = len(test)
    if metric == "accuracy":
        c = science.certify_accuracy(ev["accuracy"], n, threshold, checks=1, alpha=alpha)
        return {"certified": bool(c["certified"]), "observed": ev["value"],
                "lower_bound": c["lower_bound"], "winner": scan["winner"], "n": n}
    yt, pr = np.array(ev["y_true"]), np.array(ev["predictions"])
    rng = np.random.default_rng(0)
    boots = [science.score_metric(metric, yt[idx], pr[idx], labels)
             for idx in (rng.integers(0, n, n) for _ in range(800))]
    lower = round(float(np.percentile(boots, 100 * alpha)), 4)
    return {"certified": bool(lower > threshold), "observed": ev["value"],
            "lower_bound": lower, "winner": scan["winner"], "n": n}


# =============================================================================================== self-test
def _selftest() -> bool:
    """Exercise the runner on the REAL frozen datasets (text + tabular) and assert every honest path.

    Datasets:
      text    -> /Users/abdullahalghamdi/core-ml-acceptance/data {id,text,target}, labels class_a/class_b
      tabular -> /Users/abdullahalghamdi/vectorforge-harnesses/rugged/data {features,target}
    """
    import json
    from pathlib import Path

    TEXT = Path("/Users/abdullahalghamdi/core-ml-acceptance/data")
    TAB = Path("/Users/abdullahalghamdi/vectorforge-harnesses/rugged/data")

    def loadj(d, n):
        return [json.loads(l) for l in (d / f"{n}.jsonl").read_text().splitlines() if l.strip()]

    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")

    # ---------- TEXT: an achievable bar must certify + deploy, and serve from a reload ----------
    text_raw = loadj(TEXT, "train") + loadj(TEXT, "validation") + loadj(TEXT, "test")
    gt = run_classifier_goal(name="clf_text_pass", kind="text", labels=["class_a", "class_b"],
                             raw_rows=text_raw, metric="accuracy", threshold=0.70,
                             max_latency_ms=50.0, min_heldout_n=200)
    cert = gt.certificate or {}
    check("text: certified + deployed", certified(gt) and (gt.deployment or {}).get("status") == "ready",
          f"status={gt.status} observed={cert.get('observed')} lb={cert.get('lower_bound')}")
    check("text: certifier lower bound truly clears threshold",
          certified(gt) and cert["lower_bound"] > cert["threshold"],
          f"lb={cert.get('lower_bound')} > thr={cert.get('threshold')}")
    check("text: decide() -> a certified action", decide(gt) in (PROMOTE, REBALANCE, EXPAND_REPRESENTATION),
          decide(gt))
    # reload from disk and serve (persistence + serving)
    reloaded = store.load(gt.id)
    pred = predict(reloaded.id, "a beautifully acted, deeply moving film")
    check("text: predict from reloaded artifact", str(pred) in {"class_a", "class_b"}, f"pred={pred}")

    # ---------- TEXT: an above-ceiling bar must FAIL honestly (never relaxed) ----------
    gf = run_classifier_goal(name="clf_text_hardbar", kind="text", labels=["class_a", "class_b"],
                             raw_rows=text_raw, metric="accuracy", threshold=0.99,
                             max_latency_ms=50.0, min_heldout_n=200)
    check("text: above-ceiling bar -> honest FAILED (not certified)",
          gf.status == FAILED and not certified(gf), f"status={gf.status}")
    check("text: FAILED decide() never PROMOTE",
          decide(gf) in (STOP_HONEST_FAIL, COLLECT_MORE_HELDOUT), decide(gf))

    # ---------- TEXT: too few labels -> honest NEEDS_INPUT ----------
    by = {l: [r for r in loadj(TEXT, "train") if r["target"] == l] for l in ("class_a", "class_b")}
    tiny = [r for l in by for r in by[l][:8]]
    gn = run_classifier_goal(name="clf_text_tiny", kind="text", labels=["class_a", "class_b"],
                             raw_rows=tiny, metric="accuracy", threshold=0.90, min_heldout_n=200)
    check("text: tiny data -> NEEDS_INPUT (ACQUIRE_LABELS)",
          gn.status == NEEDS_INPUT and decide(gn) == ACQUIRE_LABELS, f"status={gn.status}")

    # ---------- TABULAR: rugged must certify at a defensible accuracy bar ----------
    # subsample train for a fast-but-real run; val/test stay realistic. Splitter re-stratifies the union.
    tab_train = loadj(TAB, "train")[:6000]
    tab_val = loadj(TAB, "validation")[:2000]
    tab_test = loadj(TAB, "test")
    tab_raw = tab_train + tab_val + tab_test
    gtab = run_classifier_goal(name="clf_tabular_pass", kind="tabular", labels=["class_a", "class_b"],
                               raw_rows=tab_raw, metric="accuracy", threshold=0.78,
                               max_latency_ms=50.0, min_heldout_n=200)
    ctab = gtab.certificate or {}
    check("tabular: certified + deployed on rugged",
          certified(gtab) and (gtab.deployment or {}).get("status") == "ready",
          f"status={gtab.status} observed={ctab.get('observed')} lb={ctab.get('lower_bound')}")

    # ---------- TABULAR: strong-solver reference agrees the task is solvable ----------
    # reuse the runner's own splits so the reference sees the same locked test
    rtr = store.read_rows(store._dir(gtab.id) / "train.jsonl")
    rva = store.read_rows(store._dir(gtab.id) / "val.jsonl")
    rte = store.read_rows(store._dir(gtab.id) / "test.jsonl")
    ref = strong_solver_reference(kind="tabular", train=rtr, val=rva, test=rte,
                                  labels=["class_a", "class_b"], metric="accuracy", threshold=0.78)
    check("tabular: strong-solver reference confirms solvable",
          ref["certified"], f"ref observed={ref['observed']} lb={ref['lower_bound']} winner={ref['winner']}")

    # ---------- TABULAR: leakage feature -> must be BLOCKED, never certified ----------
    def leaky(rows):
        return [{"features": {"copy_of_target": r["target"]}, "target": r["target"]} for r in rows]
    gleak = run_classifier_goal(name="clf_tabular_leak", kind="tabular", labels=["class_a", "class_b"],
                                raw_rows=leaky(loadj(TAB, "train")[:3000]), metric="accuracy",
                                threshold=0.80, min_heldout_n=1)
    check("tabular: leakage feature -> BLOCKED, not certified",
          gleak.status == BLOCKED and not certified(gleak) and decide(gleak) == FIX_LEAKAGE,
          f"status={gleak.status} decide={decide(gleak)}")

    ok = all(c for _, c, _ in checks)
    print(f"\nclassifier self-test: {sum(c for _, c, _ in checks)}/{len(checks)} checks passed")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
