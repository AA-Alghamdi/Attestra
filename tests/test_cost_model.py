"""Unit tests for vfplatform.cost_model (the explicit cost model: per-hour rates, cost estimate, VoI ratio).

Plain asserts + main(), matching the repo's pytest-free test style. Pure-function tests; no loop, no clock,
no network.

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_cost_model.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vfplatform import cost_model as cm


# ----------------------------------------------------------------- cost_per_hour
def test_cost_per_hour_cpu_is_zero():
    assert cm.cost_per_hour("cpu") == 0.0


def test_cost_per_hour_known_gpu():
    assert cm.cost_per_hour("A100") == 1.99
    assert cm.cost_per_hour("A40") == 0.79


def test_cost_per_hour_gpu_costs_more_than_cpu():
    # the headline property: every GPU flavor is strictly pricier per hour than the (free) CPU path
    for gpu in ("A40", "A100", "A5000", "RTX4090", "L4"):
        assert cm.cost_per_hour(gpu) > cm.cost_per_hour("cpu"), gpu


def test_cost_per_hour_unknown_flavor_conservative_fallback():
    # an unrecognized device must not be under-priced -> conservative (positive) fallback
    assert cm.cost_per_hour("no-such-gpu") > 0.0


# ----------------------------------------------------------------- estimate_cost
def test_estimate_cost_one_hour_equals_rate():
    assert abs(cm.estimate_cost(3600.0, "A100") - 1.99) < 1e-12
    assert abs(cm.estimate_cost(3600.0, "A40") - 0.79) < 1e-12


def test_estimate_cost_cpu_is_free():
    assert cm.estimate_cost(123456.0, "cpu") == 0.0


def test_estimate_cost_scales_linearly():
    # half an hour costs half the hourly rate
    assert abs(cm.estimate_cost(1800.0, "A40") - 0.5 * cm.cost_per_hour("A40")) < 1e-12


def test_estimate_cost_clamps_bad_durations():
    assert cm.estimate_cost(-10.0, "A100") == 0.0
    assert cm.estimate_cost(float("nan"), "A100") == 0.0
    assert cm.estimate_cost(float("inf"), "A100") == 0.0


# ----------------------------------------------------------------- gain_per_cost
def test_gain_per_cost_basic():
    assert abs(cm.gain_per_cost(0.2, 2.0) - 0.1) < 1e-12


def test_gain_per_cost_free_work_finite_large():
    # positive gain at ~0 cost -> large-but-finite (uses the floor), never a div-by-zero
    v = cm.gain_per_cost(0.1, 0.0)
    assert math.isfinite(v) and v == 0.1 / 1e-6


def test_gain_per_cost_nonpositive_gain_is_zero():
    assert cm.gain_per_cost(0.0, 5.0) == 0.0
    assert cm.gain_per_cost(-0.3, 5.0) == 0.0


def test_gain_per_cost_cheaper_is_better():
    # same gain for fewer dollars -> strictly higher gain-per-cost (the VoI ordering scheduling needs)
    assert cm.gain_per_cost(0.1, 1.0) > cm.gain_per_cost(0.1, 10.0)


def test_gain_per_cost_handles_bad_cost():
    # NaN / negative cost falls back to the floor rather than poisoning the ratio
    assert cm.gain_per_cost(0.1, float("nan")) == 0.1 / 1e-6
    assert cm.gain_per_cost(0.1, -5.0) == 0.1 / 1e-6


def test_deterministic_no_clock():
    # pure functions: identical inputs -> identical outputs across calls
    assert cm.estimate_cost(1000.0, "A40") == cm.estimate_cost(1000.0, "A40")
    assert cm.gain_per_cost(0.05, 3.0) == cm.gain_per_cost(0.05, 3.0)


TESTS = [
    test_cost_per_hour_cpu_is_zero,
    test_cost_per_hour_known_gpu,
    test_cost_per_hour_gpu_costs_more_than_cpu,
    test_cost_per_hour_unknown_flavor_conservative_fallback,
    test_estimate_cost_one_hour_equals_rate,
    test_estimate_cost_cpu_is_free,
    test_estimate_cost_scales_linearly,
    test_estimate_cost_clamps_bad_durations,
    test_gain_per_cost_basic,
    test_gain_per_cost_free_work_finite_large,
    test_gain_per_cost_nonpositive_gain_is_zero,
    test_gain_per_cost_cheaper_is_better,
    test_gain_per_cost_handles_bad_cost,
    test_deterministic_no_clock,
]


def run(tests):
    p = f = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}"); f += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {e}"); f += 1
    print(f"  ---- {p} passed, {f} failed ----")
    return p, f


def main():
    _, fails = run(TESTS)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
