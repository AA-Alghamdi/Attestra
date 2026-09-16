"""Tests for the cross-dataset battery + corrected FDR (NEW vfplatform/battery.py, migration Item 3).

The headline guarantee: the battery's FDR is over a candidate-vs-baseline McNemar test, so a winner that does
NOT beat the no-search baseline is NOT a discovery -- the exact bug Codex shipped (binomial-vs-floor null that
promoted tasks at lift 0.0). All offline: synthetic correctness vectors, no network, no model training.

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_battery.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import battery, connectors

_PASS = 0


def _ok(c, label, ctx=None):
    global _PASS
    assert c, f"FAIL: {label} :: {ctx}"
    _PASS += 1
    print(f"  PASS  {label}")


def test_mcnemar_exact():
    _ok(abs(battery.mcnemar_pvalue([1] * 14 + [0] * 6, [0] * 20) - 0.5 ** 14) < 1e-12,
        "McNemar exact: 14 cand-only wins -> 0.5^14")
    _ok(battery.mcnemar_pvalue([1, 1, 1], [1, 1, 1]) == 1.0, "no discordant pairs -> p=1.0")
    _ok(battery.mcnemar_pvalue([], []) == 1.0, "empty vectors -> p=1.0 (no claim)")
    _ok(battery.mcnemar_pvalue([1, 0], None) == 1.0, "missing baseline -> p=1.0")


def test_bh_step_up():
    # p = [0.001, 0.2, 0.04], alpha=0.1, m=3: sorted 0.001(1/3*0.1=0.033 ok), 0.04(2/3*0.1=0.066 ok), 0.2(no)
    rej = battery.benjamini_hochberg([0.001, 0.2, 0.04], alpha=0.1)
    _ok(rej == {0, 2}, "BH rejects the two small p-values, not the large one", rej)
    _ok(battery.benjamini_hochberg([], 0.1) == set(), "empty -> no rejections")


def test_identical_candidate_is_not_a_discovery():
    """THE FIX: a candidate identical to the baseline must get p=1.0 and never be an FDR discovery."""
    def runner(task):
        v = task["_v"]
        return {"certified": True, "decision": "certified", "cand_correct": v[0], "base_correct": v[1],
                "observed": sum(v[0]) / len(v[0]), "lower_bound": None, "n_test": len(v[0]), "source": "synthetic"}
    n = 200
    def acc(a): return [1] * int(a * n) + [0] * (n - int(a * n))
    tasks = [
        {"task_id": "identical", "metric": "accuracy", "threshold": 0.5, "_v": (acc(0.8), acc(0.8))},
        {"task_id": "real-gain", "metric": "accuracy", "threshold": 0.5, "_v": (acc(0.93), acc(0.78))},
    ]
    rep = battery.run_battery(tasks, alpha=0.1, runner=runner)
    by = {r["task_id"]: r for r in rep["tasks"]}
    _ok(by["identical"]["p_value"] == 1.0 and not by["identical"]["beats_baseline_fdr"],
        "identical-to-baseline is NOT a discovery (Codex promoted these at lift 0.0)")
    _ok(by["real-gain"]["beats_baseline_fdr"] and by["real-gain"]["p_value"] < 0.01,
        "a real candidate-vs-baseline gain IS a discovery")
    _ok(rep["null"].startswith("candidate-vs-baseline"), "report names the correct null")


def test_regression_paired_pvalue_directly():
    """The regression paired test mirrors McNemar's one-sided contract on per-row ERROR:
    a winner with genuinely lower per-row error -> small p; equal per-row error -> p=1.0."""
    import numpy as np
    rng = np.random.default_rng(11)
    m = 200
    y = rng.normal(0, 1, m).tolist()
    base_pred = [v + rng.normal(0, 1.0) for v in y]        # large per-row error
    win_pred = [v + rng.normal(0, 0.15) for v in y]        # genuinely lower per-row error
    p_win = battery.regression_paired_pvalue(win_pred, base_pred, y)
    _ok(p_win < 0.01, "winner with lower per-row error -> small one-sided p", p_win)
    # equal error: identical predictions -> no claim -> p=1.0 (mirrors McNemar no-discordant-pairs)
    p_eq = battery.regression_paired_pvalue(list(base_pred), base_pred, y)
    _ok(p_eq == 1.0, "equal-error winner -> p=1.0 (no claim)", p_eq)
    # WORSE winner (higher error) must NOT get a small one-sided p (alternative is 'winner error lower')
    worse_pred = [v + rng.normal(0, 2.0) for v in y]
    p_worse = battery.regression_paired_pvalue(worse_pred, base_pred, y)
    _ok(p_worse > 0.1, "worse winner -> not significant (one-sided, lower-error alternative)", p_worse)
    # guards
    _ok(battery.regression_paired_pvalue([], [], []) == 1.0, "empty regression vectors -> p=1.0")
    _ok(battery.regression_paired_pvalue([1.0], [2.0], [0.0]) == 1.0, "n<2 -> p=1.0 (no claim)")
    # bootstrap fallback path agrees in direction (force it via error='abs', monkeypatch-free: just call it
    # and check the primary already passed; the fallback is exercised when scipy is unavailable).


def test_regression_winner_is_a_discovery_equal_is_not():
    """THE TIER-1 GAP A FIX: run_battery dispatches the regression test on metric in (r2,neg_rmse,neg_mae).
    A regression winner with genuinely lower per-row error IS an FDR discovery; an equal-error winner is NOT."""
    import numpy as np
    rng = np.random.default_rng(3)
    m = 180
    y = rng.normal(0, 1, m).tolist()
    base_pred = [v + rng.normal(0, 1.0) for v in y]
    win_pred = [v + rng.normal(0, 0.2) for v in y]

    def runner(task):
        cp, bp, yt = task["_reg"]
        # NOTE: no is_regression flag -> run_battery must infer regression from metric=neg_rmse.
        return {"certified": False, "decision": "honest_stop", "cand_pred": cp, "base_pred": bp, "y_true": yt,
                "vec_source": "battery_eval_regression", "n_test": len(yt), "source": "synthetic-reg"}

    tasks = [
        {"task_id": "reg-real-gain", "metric": "neg_rmse", "threshold": -1.0, "_reg": (win_pred, base_pred, y)},
        {"task_id": "reg-equal", "metric": "neg_rmse", "threshold": -1.0, "_reg": (list(base_pred), base_pred, y)},
    ]
    rep = battery.run_battery(tasks, alpha=0.1, runner=runner)
    by = {r["task_id"]: r for r in rep["tasks"]}
    _ok(by["reg-real-gain"]["test"] == "regression_paired_error",
        "run_battery picks the regression test from metric (neg_rmse)")
    _ok(by["reg-real-gain"]["p_value"] < 0.01 and by["reg-real-gain"]["beats_baseline_fdr"],
        "a regression winner with genuinely lower error IS a discovery", by["reg-real-gain"])
    _ok(by["reg-equal"]["p_value"] == 1.0 and not by["reg-equal"]["beats_baseline_fdr"],
        "an equal-error regression winner is NOT a discovery", by["reg-equal"])
    _ok((by["reg-real-gain"]["lift_over_baseline"] or 0) > 0,
        "regression lift reports a positive mean-absolute-error reduction for the real gain",
        by["reg-real-gain"]["lift_over_baseline"])


def test_classification_still_uses_mcnemar():
    """Regression dispatch must not regress classification: a classification metric still uses McNemar."""
    n = 200
    def acc(a): return [1] * int(a * n) + [0] * (n - int(a * n))
    def runner(task):
        v = task["_v"]
        return {"certified": True, "decision": "certified", "is_regression": False,
                "cand_correct": v[0], "base_correct": v[1], "n_test": n, "source": "syn"}
    tasks = [{"task_id": "clf", "metric": "accuracy", "threshold": 0.5, "_v": (acc(0.93), acc(0.78))}]
    rep = battery.run_battery(tasks, alpha=0.1, runner=runner)
    by = {r["task_id"]: r for r in rep["tasks"]}
    _ok(by["clf"]["test"] == "mcnemar", "classification metric still routes to McNemar")
    _ok(by["clf"]["beats_baseline_fdr"], "classification gain still a discovery under McNemar")


def test_runner_error_does_not_sink_battery():
    def runner(task):
        raise RuntimeError("offline")
    rep = battery.run_battery([{"task_id": "x", "metric": "accuracy", "threshold": 0.5, "source_uri": "openml://dataset/1"}],
                              alpha=0.1, runner=runner)
    _ok(rep["tasks"][0]["status"] == "error" and rep["n_discoveries"] == 0,
        "a failing shard is recorded as error, not a discovery")


def test_connector_dispatch_rejects_unknown_uri():
    try:
        connectors.materialize("s3://nope")
        _ok(False, "unknown uri should raise")
    except ValueError:
        _ok(True, "materialize() rejects an unknown source uri scheme")


if __name__ == "__main__":
    test_mcnemar_exact()
    test_bh_step_up()
    test_identical_candidate_is_not_a_discovery()
    test_regression_paired_pvalue_directly()
    test_regression_winner_is_a_discovery_equal_is_not()
    test_classification_still_uses_mcnemar()
    test_runner_error_does_not_sink_battery()
    test_connector_dispatch_rejects_unknown_uri()
    print(f"\ntest_battery: {_PASS} passed, 0 failed")
