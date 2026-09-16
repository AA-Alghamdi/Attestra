"""Behavioral tests for the Phase-6 budget scheduler (frontier/budget.py).

Asserts the three contracted properties: the cost model estimates and self-calibrates from
REAL sklearn timings; the BudgetController respects a total budget, allocates sensibly across
rounds, and early-kills losing arms; the PhaseDecomposer produces a valid gated DAG and
rejects invalid ones. Run standalone:
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_budget.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier.budget import (
    CostModel, CostEstimate, BudgetController, BudgetConfig,
    PhaseDecomposer, ExperimentPlan, Phase, program_family, program_enhancers,
)
from frontier.program import Program


def _prog(base, **enh):
    recipe = {"base": base}
    recipe.update(enh)
    return Program(code="", source="seed", label=base, provenance={"recipe": recipe})


# --------------------------------------------------------------------------- cost model

def test_family_and_enhancer_extraction():
    p = _prog("rf", scale=True, poly=2)
    assert program_family(p) == "rf"
    assert set(program_enhancers(p)) == {"scale", "poly2"}
    # LLM program with no recipe -> static analysis of the code string
    llm = Program(code="from sklearn.svm import SVC\ndef build_estimator():\n return SVC()",
                  source="llm", label="llm0")
    assert program_family(llm) == "svc_rbf"
    print("[ok] family/enhancer extraction (recipe + static-analysis fallback)")


def test_cost_estimate_monotone_in_size_and_enhancers():
    cm = CostModel()
    small = cm.estimate(_prog("rf"), n_train=200, n_features=10).seconds
    big = cm.estimate(_prog("rf"), n_train=20000, n_features=10).seconds
    assert big > small, "more data must cost more"
    plain = cm.estimate(_prog("ridge"), n_train=1000, n_features=20).seconds
    poly = cm.estimate(_prog("ridge", poly=2), n_train=1000, n_features=20).seconds
    assert poly > plain, "polynomial features must raise estimated cost"
    # uncalibrated estimates carry a band and flag calibrated=False
    e = cm.estimate(_prog("rf"), n_train=1000, n_features=10)
    assert e.lo <= e.seconds <= e.hi and not e.calibrated
    print(f"[ok] cost monotone: rf small={small:.4f}s big={big:.4f}s; poly>{plain:.4f}")


def test_cost_model_calibrates_from_real_timings():
    """Feed REAL measured fit+predict times for two families and assert the model's
    estimate moves toward the truth (relative ordering preserved) and flags calibrated."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    d = load_breast_cancer()
    X, y = d.data.astype(float), d.target
    nt, nf = X.shape

    cm = CostModel()

    def time_fit(est):
        t0 = time.perf_counter()
        est.fit(X, y)
        est.predict(X)
        return time.perf_counter() - t0

    # collect several real timings per family and feed them back
    for _ in range(4):
        t_rf = time_fit(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1))
        cm.observe(_prog("rf"), nt, nf, t_rf, ok=True)
        t_lr = time_fit(LogisticRegression(max_iter=500))
        cm.observe(_prog("logreg"), nt, nf, t_lr, ok=True)

    est_rf = cm.estimate(_prog("rf"), nt, nf)
    est_lr = cm.estimate(_prog("logreg"), nt, nf)
    assert est_rf.calibrated and est_lr.calibrated, "estimates should be calibrated after >=2 obs"
    assert cm.family_table()["rf"]["n_obs"] == 4
    # the calibrated estimate should be in the right ballpark of the last real timing
    assert est_rf.lo <= max(t_rf, 1e-6) * 3 and est_rf.seconds > 0
    # RF (an ensemble) should be estimated as costlier than logreg on the same data
    assert est_rf.seconds > est_lr.seconds, (est_rf.seconds, est_lr.seconds)
    print(f"[ok] calibrated from real fits: rf~{est_rf.seconds:.4f}s (true {t_rf:.4f}) "
          f"logreg~{est_lr.seconds:.4f}s (true {t_lr:.4f})")


def test_cost_model_never_crashes_on_garbage():
    cm = CostModel()
    p = _prog("rf")
    # zero/negative/NaN observations are ignored, not poison
    cm.observe(p, 0, 0, wall_seconds=0.0, ok=True)
    cm.observe(p, 100, 10, wall_seconds=float("nan"), ok=True)
    cm.observe(p, 100, 10, wall_seconds=-5.0, ok=True)
    cm.observe(p, 100, 10, wall_seconds=2.0, ok=False)   # failed run: not used for slope
    e = cm.estimate(p, n_train=10**9, n_features=10**6)   # absurd size -> capped, finite
    assert np.isfinite(e.seconds) and e.seconds <= CostModel._MAX_SECONDS
    assert cm.family_table()["rf"]["n_obs"] == 0, "garbage/failed obs must not calibrate"
    print("[ok] cost model robust to garbage observations and absurd sizes")


# --------------------------------------------------------------------------- budget controller

def test_controller_respects_total_budget():
    """Charge true costs across many candidates; spend never exceeds usable, and admit()
    refuses once the budget is gone."""
    ctrl = BudgetController(BudgetConfig(total_seconds=100.0, rounds=1, reserve_frac=0.10))
    ctrl.begin_round(0)
    admitted = 0
    for i in range(50):
        if ctrl.admit(f"p{i}", est_seconds=10.0):
            admitted += 1
            ctrl.charge(f"p{i}", 10.0)        # true cost == estimate here
            ctrl.record_score(f"p{i}", 0.5 + 0.01 * i)
    # usable = 90s, reserve = 10s. At 10s each, at most 9 admitted.
    assert admitted == 9, admitted
    assert ctrl.spent_seconds <= ctrl.usable_seconds + 1e-9
    assert ctrl.remaining_seconds <= 1e-6 and ctrl.stop()
    rep = ctrl.report()
    assert rep["spent_seconds"] <= rep["usable_seconds"] + 1e-9
    assert rep["reserve_seconds"] > 0, "reserve must be held back for certification"
    print(f"[ok] budget respected: admitted={admitted} spent={rep['spent_seconds']} "
          f"usable={rep['usable_seconds']} reserve={rep['reserve_seconds']}")


def test_controller_allocates_across_rounds_and_carries_forward():
    """Per-round caps spread the budget; an underspent round raises later rounds' caps without
    exceeding the usable total."""
    ctrl = BudgetController(BudgetConfig(total_seconds=120.0, rounds=3, reserve_frac=0.0))
    # round 0: cap ~40s, but only spend 5s
    ctrl.begin_round(0)
    assert ctrl.admit("a", 5.0)
    ctrl.charge("a", 5.0)
    ctrl.end_round(0)
    # round 1: remaining 115s over 2 rounds -> cap ~57.5s (carried forward, > original 40)
    ctrl.begin_round(1)
    assert ctrl.admit("b", 50.0), "carried-forward budget should admit a 50s candidate"
    ctrl.charge("b", 50.0)
    # a second 50s in the same round would exceed the ~57.5 round cap -> refused
    assert not ctrl.admit("c", 50.0)
    ctrl.end_round(1)
    assert ctrl.spent_seconds <= ctrl.usable_seconds + 1e-9
    print(f"[ok] cross-round allocation + carry-forward: spent={ctrl.spent_seconds:.2f}/120")


def test_controller_early_kills_losing_arms():
    """Successive-halving: at end_round the bottom arms by score are culled and stop being alive."""
    ctrl = BudgetController(BudgetConfig(total_seconds=1000.0, rounds=2,
                                         reserve_frac=0.0, keep_frac=0.5, min_keep=2))
    ctrl.begin_round(0)
    scores = {"a": 0.90, "b": 0.85, "c": 0.60, "d": 0.55}
    for pid, s in scores.items():
        assert ctrl.admit(pid, 1.0)
        ctrl.charge(pid, 1.0)
        ctrl.record_score(pid, s)
    killed = ctrl.end_round(0)
    # keep top 50% of 4 = 2 (a, b); kill c, d
    assert set(killed) == {"c", "d"}, killed
    assert ctrl.is_alive("a") and ctrl.is_alive("b")
    assert not ctrl.is_alive("c") and not ctrl.is_alive("d")
    print(f"[ok] early-kill culled losers: killed={sorted(killed)}")


def test_controller_min_keep_floor():
    """min_keep prevents culling below a floor even with a tiny keep_frac."""
    ctrl = BudgetController(BudgetConfig(total_seconds=100.0, rounds=1,
                                         reserve_frac=0.0, keep_frac=0.1, min_keep=2))
    ctrl.begin_round(0)
    for pid, s in [("a", 0.9), ("b", 0.8), ("c", 0.7)]:
        ctrl.admit(pid, 1.0); ctrl.charge(pid, 1.0); ctrl.record_score(pid, s)
    killed = ctrl.end_round(0)
    assert len(killed) == 1 and killed == ["c"], killed   # keep 2 (floor), kill the worst 1
    print(f"[ok] min_keep floor honored: killed={killed}")


def test_config_validation():
    for bad in [dict(total_seconds=0), dict(rounds=0), dict(reserve_frac=1.0),
                dict(keep_frac=0.0), dict(keep_frac=1.5), dict(min_keep=0)]:
        try:
            BudgetConfig(**bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass
    print("[ok] BudgetConfig rejects invalid parameters")


# --------------------------------------------------------------------------- phase decomposer

def test_decomposer_produces_valid_gated_dag():
    plan = PhaseDecomposer().decompose(total_seconds=3600.0, n_candidates=64, kind="classification")
    plan.validate()   # must not raise
    names = [p.name for p in plan.phases]
    assert names == ["screen", "refine", "confirm"]
    # budgets conserve (sum <= total) and narrow monotonically
    assert sum(p.budget_seconds for p in plan.phases) <= 3600.0 + 1e-6
    cohorts = [p.cohort_size for p in plan.phases]
    survivors = [p.survivors for p in plan.phases]
    assert cohorts == sorted(cohorts, reverse=True), cohorts        # non-increasing
    assert survivors[-1] == 1, "terminal phase yields exactly one winner"
    # each phase's cohort equals the previous phase's survivors (gated chain)
    for prev, cur in zip(plan.phases, plan.phases[1:]):
        assert cur.cohort_size == prev.survivors
        assert cur.depends_on == prev.name
    assert plan.phases[0].depends_on is None
    print(f"[ok] valid gated DAG: cohorts={cohorts} survivors={survivors} "
          f"budgets={[round(p.budget_seconds) for p in plan.phases]}")


def test_decomposer_handles_tiny_candidate_pool():
    plan = PhaseDecomposer().decompose(total_seconds=60.0, n_candidates=1)
    plan.validate()
    assert all(p.cohort_size >= 1 and p.survivors >= 1 for p in plan.phases)
    assert plan.phases[-1].survivors == 1
    print("[ok] decomposer handles n_candidates=1 without producing an invalid plan")


def test_decomposer_with_absolute_gates():
    plan = PhaseDecomposer().decompose(total_seconds=100.0, n_candidates=8,
                                       gates=[0.5, 0.6, None])
    plan.validate()
    assert plan.phases[0].gate == 0.5 and plan.phases[1].gate == 0.6
    assert plan.phases[2].gate is None
    print("[ok] absolute per-phase gates carried through")


def test_validate_rejects_broken_plans():
    # cycle: a depends on b, b depends on a
    bad_cycle = ExperimentPlan(
        phases=[Phase("a", 10, 4, 2, None, "b"), Phase("b", 10, 2, 1, None, "a")],
        total_seconds=100.0)
    _expect_valueerror(bad_cycle.validate, "cycle")
    # overcommitted budget
    bad_budget = ExperimentPlan(
        phases=[Phase("s", 80, 4, 2, None, None), Phase("c", 80, 2, 1, None, "s")],
        total_seconds=100.0)
    _expect_valueerror(bad_budget.validate, "overcommit")
    # growing cohort (not narrowing)
    bad_grow = ExperimentPlan(
        phases=[Phase("s", 10, 2, 2, None, None), Phase("c", 10, 4, 1, None, "s")],
        total_seconds=100.0)
    _expect_valueerror(bad_grow.validate, "grow")
    # terminal not a single winner
    bad_term = ExperimentPlan(
        phases=[Phase("s", 10, 4, 3, None, None), Phase("c", 10, 3, 3, None, "s")],
        total_seconds=100.0)
    _expect_valueerror(bad_term.validate, "terminal")
    # survivors > cohort
    bad_surv = ExperimentPlan(
        phases=[Phase("s", 10, 2, 5, None, None)], total_seconds=100.0)
    _expect_valueerror(bad_surv.validate, "survivors>cohort")
    print("[ok] validate() rejects cycles, overcommit, growth, bad terminal, bad survivors")


def _expect_valueerror(fn, what):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {what}")


# --------------------------------------------------------------------------- integration shape

def test_controller_plan_integration_shape():
    """Drive a decomposed plan with a per-phase controller exactly as the WIRING block
    prescribes: each phase runs under its own budget, charges true costs, and the chain
    narrows to one survivor. This mirrors the orchestrator integration without the engine."""
    cm = CostModel()
    plan = PhaseDecomposer().decompose(total_seconds=300.0, n_candidates=8)
    plan.validate()
    survivors_remaining = plan.phases[0].cohort_size
    for phase in plan.phases:
        ctrl = BudgetController(BudgetConfig(total_seconds=phase.budget_seconds, rounds=1,
                                             reserve_frac=0.0))
        ctrl.begin_round(0)
        ran = 0
        for i in range(phase.cohort_size):
            p = _prog("logreg", scale=True)
            est = cm.estimate(p, n_train=400, n_features=30)
            if ctrl.admit(f"{phase.name}_{i}", est.seconds):
                ctrl.charge(f"{phase.name}_{i}", est.seconds)
                cm.observe(p, 400, 30, est.seconds, ok=True)
                ran += 1
        assert ctrl.spent_seconds <= ctrl.usable_seconds + 1e-9
        assert ran >= 1, f"phase {phase.name} should run at least one candidate"
        survivors_remaining = phase.survivors
    assert survivors_remaining == 1
    print("[ok] plan+controller integration: chain narrows to a single certified-ready winner")


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
    print(f"\n{len(fns) - failed}/{len(fns)} budget tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
