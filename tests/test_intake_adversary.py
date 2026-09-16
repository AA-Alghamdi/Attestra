"""Tests for the OPERATIONALIZATION ADVERSARY (NEW module vfplatform/intake_adversary.py) and its wiring
into the live frontdoor intake path.

Proves the adversary is a SOFT-EDGE, NON-BINDING, spec-aware screen: it can DECLINE a degenerate spec (theta
at/below the metric's trivial baseline) or WARN about single-feature/forbidden-field gameability, but it never
promotes a model, never takes a sealed peek, and never touches the frozen certifier. Also proves the wiring
does not regress a legitimate goal (non-trivial theta -> the run proceeds normally).

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_intake_adversary.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import intake_adversary
from vfplatform.frontdoor import run as frontdoor_run

_PASS = 0


def _ok(cond, label, ctx=None):
    global _PASS
    assert cond, f"FAIL: {label} :: {ctx}"
    _PASS += 1
    print(f"  PASS  {label}")


def test_trivial_threshold_declines():
    y = ["A"] * 240 + ["B"] * 60                              # 80% majority
    recs = [{"features": {"f": float(np.random.RandomState(i).randn())}, "target": t}
            for i, t in enumerate(y)]
    r = intake_adversary.screen(recs, task_type="binary", metric="accuracy", threshold=0.75)
    _ok(r["verdict"] == "decline", "theta below majority baseline -> decline", r["verdict"])
    _ok(abs(r["trivial_baseline"] - 0.8) < 1e-9, "trivial baseline == majority rate", r["trivial_baseline"])
    r2 = intake_adversary.screen(recs, task_type="binary", metric="accuracy", threshold=0.95)
    _ok(not any(f["check"] == "trivial_threshold" for f in r2["findings"]),
        "theta above baseline -> no trivial-threshold finding")


def test_single_feature_and_forbidden():
    rng = np.random.RandomState(0)
    yb = (["A"] * 150) + (["B"] * 150)
    leak = [{"features": {"leak": (1.0 if t == "B" else 0.0) + 0.01 * rng.randn(),
                          "noise": float(rng.randn())}, "target": t} for t in yb]
    r = intake_adversary.screen(leak, task_type="binary", metric="accuracy", threshold=0.9)
    _ok(any(f["check"] == "single_feature_sufficiency" and f["feature"] == "leak" for f in r["findings"]),
        "single feature that clears theta -> warn")
    _ok(r["single_feature_best"]["feature"] == "leak", "strongest single feature reported", r["single_feature_best"])
    r2 = intake_adversary.screen(leak, task_type="binary", metric="accuracy", threshold=0.9, drop_cols=["leak"])
    _ok(any(f["check"] == "forbidden_field_signal" and f["feature"] == "leak" for f in r2["findings"]),
        "forbidden field carrying signal -> distinct forbidden-field finding")
    _ok(not any(f["check"] == "single_feature_sufficiency" and f["feature"] == "leak" for f in r2["findings"]),
        "a forbidden field is NOT also flagged as a non-forbidden single-feature leak")


def test_regression_baseline():
    rng = np.random.RandomState(1)
    yr = [float(v) for v in rng.randn(200)]
    rr = [{"features": {"x": float(rng.randn())}, "target": t} for t in yr]
    r = intake_adversary.screen(rr, task_type="regression", metric="r2", threshold=0.0)
    _ok(r["verdict"] == "decline", "r2 theta<=0 (mean predictor) -> decline", r["verdict"])
    r2 = intake_adversary.screen(rr, task_type="regression", metric="r2", threshold=0.5)
    _ok(not any(f["check"] == "trivial_threshold" for f in r2["findings"]),
        "r2 theta=0.5 -> no trivial-threshold finding")


def test_wired_into_frontdoor_declines_degenerate_spec():
    """The live frontdoor path must DECLINE a degenerate spec (no loop, no peek) and surface the adversary."""
    y = ["A"] * 240 + ["B"] * 60
    recs = [{"features": {"f": float(np.random.RandomState(i).randn())}, "target": t}
            for i, t in enumerate(y)]
    out = frontdoor_run(recs, "classify, accuracy >= 0.7", threshold=0.7,
                        kind="tabular", task_type="binary", target_key="target", metric="accuracy")
    _ok(out["result"].decision == "declined_spec", "frontdoor declines degenerate spec",
        out["result"].decision)
    _ok(out["result"].certificate is None, "no certificate emitted on a declined spec")
    _ok(out["spec_inferred"]["adversary"]["verdict"] == "decline", "adversary verdict surfaced in spec_inferred")


def test_wired_does_not_regress_legitimate_goal():
    """A non-trivial theta on real signal must still RUN (adversary attached, verdict not 'decline')."""
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    recs = [{"features": {f"x{j}": float(d.data[i, j]) for j in range(0, 10)},
             "target": int(d.target[i])} for i in range(len(d.target))]
    out = frontdoor_run(recs, "classify, accuracy >= 0.92", threshold=0.92,
                        kind="tabular", task_type="binary", target_key="target", metric="accuracy")
    _ok(out["result"].decision != "declined_spec", "legitimate goal is NOT declined by the adversary",
        out["result"].decision)
    _ok("adversary" in out["spec_inferred"], "adversary report attached even on a clean run")


if __name__ == "__main__":
    test_trivial_threshold_declines()
    test_single_feature_and_forbidden()
    test_regression_baseline()
    test_wired_into_frontdoor_declines_degenerate_spec()
    test_wired_does_not_regress_legitimate_goal()
    print(f"\ntest_intake_adversary: {_PASS} passed, 0 failed")
