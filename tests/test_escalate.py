"""Tests for vfplatform/escalate.py -- the PURE plateau-detection + strategy-escalation policy.

Proves the policy (1) leaves a still-improving search alone (climbing history -> continue), (2) switches the
KIND of move when the val lower bound has plateaued for K rounds (the gap: today the loop re-sweeps the same
catalog on plateau), walking the cheapest-first ladder model->features->capacity->data_acquisition, (3)
honest-stops when out of budget regardless of plateau, and (4) honest-stops at the top of the ladder.

This module is PURE: the tests import only vfplatform.escalate (no loop / harness / sealed / science). The
policy can never touch a certificate or the sealed peek -- these tests assert only the routing decision.

Run:  PYTHONPATH=. /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_escalate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import escalate
from vfplatform.escalate import (
    escalation_decision, is_plateau, next_class,
    CONTINUE, ESCALATE_TO_FEATURES, ESCALATE_TO_CAPACITY, ESCALATE_TO_DATA, STOP,
)

_PASS = 0


def _ok(cond, label, ctx=None):
    global _PASS
    assert cond, f"FAIL: {label} :: {ctx}"
    _PASS += 1
    print(f"  PASS  {label}")


# ---- plateau detector -----------------------------------------------------------------------------------
def test_is_plateau_basic():
    _ok(not is_plateau([0.60, 0.64, 0.68, 0.72], k=3), "strictly climbing bound is NOT a plateau")
    _ok(is_plateau([0.70, 0.70, 0.70, 0.70], k=3), "4 flat bounds is a 3-round plateau")
    _ok(not is_plateau([0.70, 0.70, 0.70], k=3),
        "only k entries (need k+1) -> not enough evidence -> not a plateau")
    _ok(is_plateau([0.50, 0.70, 0.70, 0.70, 0.70], k=3),
        "improved then flat for 3 -> plateau on the trailing window")
    _ok(not is_plateau([0.70, 0.70, 0.70, 0.71001], k=3, eps=1e-3),
        "a >eps rise on the last round breaks the plateau")


def test_is_plateau_handles_none_and_jitter():
    _ok(is_plateau([0.70, None, 0.70, 0.70], k=3),
        "a None (no computable bound) counts as no-improvement, not progress")
    _ok(is_plateau([0.70, 0.7000000001, 0.70, 0.70], k=3, eps=1e-6),
        "sub-eps float jitter is not improvement -> still a plateau")
    _ok(not is_plateau([], k=3) and not is_plateau([0.7], k=3),
        "empty / single-point history is never a plateau")


def test_next_class_ladder():
    _ok(next_class("model") == "features", "model -> features")
    _ok(next_class("features") == "capacity", "features -> capacity")
    _ok(next_class("capacity") == "data_acquisition", "capacity -> data_acquisition")
    _ok(next_class("data_acquisition") is None, "data_acquisition is the top rung")
    _ok(next_class("bogus") == "features", "unknown class treated as 'model' -> features")


# ---- the policy: the three contract cases from the spec --------------------------------------------------
def test_climbing_history_continues():
    """A still-improving val lower bound -> continue (do NOT churn the move class)."""
    d = escalation_decision([0.60, 0.64, 0.68, 0.72], "model", rounds_since_improve=0, budget_left=50)
    _ok(d.decision == CONTINUE, "climbing history -> continue", d)
    _ok(not d.plateaued and d.to_class is None, "continue carries plateaued=False, no target class", d)


def test_flat_history_escalates_to_a_different_class():
    """A flat history at K rounds, currently on 'model', -> escalate to a DIFFERENT class (features),
    instead of re-sweeping the model grid (the wasted rounds-4-5 behavior today)."""
    d = escalation_decision([0.70, 0.70, 0.70, 0.70], "model", rounds_since_improve=3, budget_left=50)
    _ok(d.decision == ESCALATE_TO_FEATURES, "flat @K on model -> escalate_to_features", d)
    _ok(d.to_class == "features" and d.to_class != "model",
        "escalation switches to a DIFFERENT move class", d)


def test_flat_history_no_budget_stops():
    """Flat history with no budget left -> stop (honest-stop; nothing can be run)."""
    d = escalation_decision([0.70, 0.70, 0.70, 0.70], "model", rounds_since_improve=3, budget_left=0)
    _ok(d.decision == STOP, "flat + no budget -> stop", d)
    # zero budget dominates: stop even if the plateau would otherwise escalate
    d2 = escalation_decision([0.60, 0.64, 0.68, 0.72], "model", rounds_since_improve=0, budget_left=0)
    _ok(d2.decision == STOP, "no budget -> stop even when still climbing (cannot run a move)", d2)


# ---- the full cheapest-first ladder on plateau -----------------------------------------------------------
def test_full_ladder_on_plateau():
    flat = [0.70, 0.70, 0.70, 0.70]
    _ok(escalation_decision(flat, "model", 3, 50).decision == ESCALATE_TO_FEATURES,
        "model plateau -> features")
    _ok(escalation_decision(flat, "features", 3, 50).decision == ESCALATE_TO_CAPACITY,
        "features plateau -> capacity")
    _ok(escalation_decision(flat, "capacity", 3, 50).decision == ESCALATE_TO_DATA,
        "capacity plateau -> data_acquisition")
    _ok(escalation_decision(flat, "data_acquisition", 3, 50).decision == STOP,
        "data_acquisition plateau (top of ladder) -> stop")


def test_data_acquisition_disallowed_ends_at_capacity():
    """With no acquire_fn wired (allow_data_acquisition=False), the ladder ends at capacity: a plateau on
    capacity -> stop, never escalate_to_data_acquisition."""
    flat = [0.70, 0.70, 0.70, 0.70]
    d = escalation_decision(flat, "capacity", 3, 50, allow_data_acquisition=False)
    _ok(d.decision == STOP, "capacity plateau with acquisition disabled -> stop (not data)", d)
    d2 = escalation_decision(flat, "features", 3, 50, allow_data_acquisition=False)
    _ok(d2.decision == ESCALATE_TO_CAPACITY,
        "features plateau still escalates to capacity when acquisition disabled", d2)


def test_history_short_but_loop_counter_trips():
    """Even with a short/empty history, the loop's own no-improve counter (rounds_since_improve >= k) is a
    valid plateau signal -- the policy agrees with the loop's existing stagnation clock."""
    d = escalation_decision([], "model", rounds_since_improve=3, budget_left=50)
    _ok(d.decision == ESCALATE_TO_FEATURES, "rsi>=k with empty history still escalates", d)
    d2 = escalation_decision([0.70, 0.71], "model", rounds_since_improve=0, budget_left=50)
    _ok(d2.decision == CONTINUE, "short history + low rsi -> continue (no premature escalation)", d2)


def test_determinism_and_robust_inputs():
    args = ([0.70, 0.70, 0.70, 0.70], "model", 3, 50)
    _ok(escalation_decision(*args).decision == escalation_decision(*args).decision,
        "deterministic: same inputs -> same decision")
    # robust to junk scalars (never raises)
    _ok(escalation_decision([0.7, 0.7, 0.7, 0.7], "model", None, None).decision == STOP,
        "None budget coerces to 0 -> stop, no exception")
    _ok(escalation_decision([0.7, 0.7, 0.7, 0.7], "model", "x", 50).decision == ESCALATE_TO_FEATURES,
        "junk rounds_since_improve coerces to 0; history plateau still drives escalation")


def main():
    test_is_plateau_basic()
    test_is_plateau_handles_none_and_jitter()
    test_next_class_ladder()
    test_climbing_history_continues()
    test_flat_history_escalates_to_a_different_class()
    test_flat_history_no_budget_stops()
    test_full_ladder_on_plateau()
    test_data_acquisition_disallowed_ends_at_capacity()
    test_history_short_but_loop_counter_trips()
    test_determinism_and_robust_inputs()
    print(f"\nALL ESCALATE TESTS PASSED ({_PASS} assertions)")


if __name__ == "__main__":
    main()
