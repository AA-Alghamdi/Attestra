"""Problem-type routing tests (NEW module vfplatform/problem_type.py).

Proves the router maps a tabular classification frame to (tabular, multiclass) and DECLINES a clearly
unsupported problem with a concrete reason -- never coercing an unsupported problem into a fake fit. The
LLM path is NON-BINDING with a deterministic fallback, so these tests run with no network/key (use_llm
defaults False; the fallback keeps the deterministic decision). Read-only on every existing module.

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_problem_type.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import problem_type
from vfplatform.sealed import SUPPORTED_METRICS


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


def test_tabular_multiclass_mapping():
    rows = [{"features": {"a": float(i), "b": float(i % 5)}, "target": str(i % 3)} for i in range(60)]
    r = problem_type.classify(rows, "classify each row into one of the categories")
    assert r["supported"] is True, r
    assert r["kind"] == "tabular"
    assert r["task_type"] == "multiclass"
    assert r["metric"] in SUPPORTED_METRICS, f"metric must be certifiable: {r['metric']}"
    assert r["decline_reason"] is None
    assert r["source"] == "deterministic"


def test_tabular_binary_mapping():
    rows = [{"features": {"a": float(i)}, "target": str(i % 2)} for i in range(40)]
    r = problem_type.classify(rows, "predict the class")
    assert r["supported"] and r["kind"] == "tabular" and r["task_type"] == "binary"
    assert r["metric"] == "accuracy"


def test_regression_mapping():
    rows = [{"features": {"a": float(i)}, "target": float(i) * 1.3} for i in range(40)]
    r = problem_type.classify(rows, "predict the continuous outcome")
    assert r["supported"] and r["task_type"] == "regression" and r["metric"] == "r2"


def test_vision_route_for_image_shaped_frame():
    rows = [{"features": {f"p{j}": float((i + j) % 16) for j in range(64)}, "target": str(i % 4)}
            for i in range(60)]
    r = problem_type.classify(rows, "classify these 8x8 pixel images into digit classes")
    assert r["supported"] and r["kind"] == "vision" and r["task_type"] == "multiclass"
    assert r["metric"] in SUPPORTED_METRICS


def test_declines_forecasting_with_reason():
    rows = [{"features": {"a": float(i)}, "target": str(i % 3)} for i in range(60)]
    r = problem_type.classify(rows, "forecast the next day's value of this time series")
    assert r["supported"] is False, "forecasting is out of this module's i.i.d. scope"
    assert r["decline_reason"] and ("time-series" in r["decline_reason"] or "forecast" in r["decline_reason"])


def test_declines_ranking_with_reason():
    rows = [{"features": {"a": float(i)}, "target": str(i % 3)} for i in range(60)]
    r = problem_type.classify(rows, "rank the search results by relevance for each query")
    assert r["supported"] is False
    assert "ranking" in r["decline_reason"]


def test_declines_inadmissible_with_reason():
    rows = [{"features": {"a": float(i)}, "target": "constant"} for i in range(40)]
    r = problem_type.classify(rows, "classify")
    assert r["supported"] is False
    assert "inadmissible" in r["decline_reason"]


def test_never_returns_uncertifiable_metric():
    """Whatever the route, the returned metric is always one the frozen sealed guard accepts (no fake-fit)."""
    for rows, goal in [
        ([{"features": {"a": float(i)}, "target": str(i % 2)} for i in range(40)], "binary"),
        ([{"features": {"a": float(i)}, "target": str(i % 4)} for i in range(60)], "multiclass"),
        ([{"features": {"a": float(i)}, "target": float(i)} for i in range(40)], "regression"),
    ]:
        r = problem_type.classify(rows, goal)
        if r["supported"]:
            assert r["metric"] in SUPPORTED_METRICS, f"uncertifiable metric leaked: {r['metric']}"


def test_supported_combos_are_a_subset_of_built_harnesses():
    """Every (kind, task_type) the router can mark supported must be one the platform actually builds."""
    from vfplatform.harness import is_supported
    for kind, tasks in problem_type.SUPPORTED.items():
        for tt in tasks:
            if kind == "vision":
                continue  # vision is the NEW harness; is_supported() is the existing table (hand-wired separately)
            assert is_supported(kind, tt), f"{kind}/{tt} claimed supported but not built"


TESTS = [test_tabular_multiclass_mapping, test_tabular_binary_mapping, test_regression_mapping,
         test_vision_route_for_image_shaped_frame, test_declines_forecasting_with_reason,
         test_declines_ranking_with_reason, test_declines_inadmissible_with_reason,
         test_never_returns_uncertifiable_metric, test_supported_combos_are_a_subset_of_built_harnesses]


def main():
    print("== test_problem_type (deterministic-first routing) ==")
    p, f = run(TESTS)
    sys.exit(1 if f else 0)


if __name__ == "__main__":
    main()
