"""Tests for POWER-AWARE experiment design (NEW module vfplatform/power.py) and its honest-stop wiring.

Proves the power analysis is EXACT for the binomial accuracy certifier (computed from the frozen
certify_accuracy / binom_sf, not re-derived), that it correctly separates UNDERPOWERED stops (too little
data) from MODEL-LIMITED stops (p* <= theta), and that it is honest (power=None) for non-binomial metrics.
Read-only analysis: nothing here promotes a model, takes a sealed peek, or touches the frozen certifier.

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_power.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import power
from vectorforge.science import certify_accuracy

_PASS = 0


def _ok(cond, label, ctx=None):
    global _PASS
    assert cond, f"FAIL: {label} :: {ctx}"
    _PASS += 1
    print(f"  PASS  {label}")


def test_power_monotone_and_bounds():
    _ok(power.power_at_n(50, 0.9, 0.9) < 0.2, "power at p*=theta is small")
    _ok(power.power_at_n(3000, 0.9, 0.95) > 0.9, "power high with ample n and margin")
    _ok(power.power_at_n(2000, 0.9, 0.95) >= power.power_at_n(200, 0.9, 0.95),
        "power trends up with n")
    _ok(power.power_at_n(10, 0.999, 0.999) == 0.0,
        "tiny n where nothing certifies -> power 0")


def test_min_n_consistent_with_frozen_certifier():
    """The returned min_n must be the TRUE smallest n meeting the target, and it must be EXACT vs the frozen
    certifier (a model whose count matches p* at that n actually certifies)."""
    need = power.min_n_for_power(0.9, 0.95, target_power=0.8)
    _ok(need is not None and power.power_at_n(need, 0.9, 0.95) >= 0.8, "min_n meets target", need)
    _ok(power.power_at_n(need - 1, 0.9, 0.95) < 0.8, "n-1 does NOT meet target (true threshold)", need)
    # bigger margin needs fewer samples
    _ok(power.min_n_for_power(0.9, 0.98) < power.min_n_for_power(0.9, 0.93), "more margin -> fewer samples")
    # cross-check exactness: at need, a model achieving ~p* certifies on the frozen path
    k = round(0.95 * need)
    _ok(certify_accuracy(k / need, need, 0.9)["certified"], "frozen certifier agrees at min_n", (k, need))


def test_model_limited_vs_underpowered():
    lim = power.assess(100, 0.9, 0.88)
    _ok(lim["powered"] is False and "model-limited" in lim["note"], "p*<=theta -> model-limited", lim["note"])
    _ok(power.min_n_for_power(0.9, 0.88) is None, "no n certifies a model not above theta")

    under = power.assess(30, 0.9, 0.97)
    _ok(under["powered"] is False and under["min_n_for_target"] and "UNDERPOWERED" in under["note"],
        "good model + tiny n -> underpowered with a target n", under)
    good = power.assess(under["min_n_for_target"], 0.9, 0.97)
    _ok(good["powered"] is True, "same model with enough data is powered")


def test_non_binomial_is_honest_none():
    nb = power.assess(500, 0.9, 0.95, metric="macro_f1")
    _ok(nb["power"] is None and "not computed" in nb["note"], "non-binomial metric -> honest None, no fake number")


def test_wired_into_honest_stop_report():
    """A real certify run forced to honest-stop (theta above the achievable ceiling) must carry the power
    annotation in its failure report, telling the user underpowered vs model-limited."""
    from sklearn.datasets import load_iris
    from vfplatform.frontdoor import run as frontdoor_run
    d = load_iris()
    recs = [{"features": {f"x{j}": float(d.data[i, j]) for j in range(4)}, "target": int(d.target[i])}
            for i in range(len(d.target))]
    out = frontdoor_run(recs, "classify, accuracy >= 0.999", threshold=0.999,
                        kind="tabular", task_type="multiclass", target_key="target", metric="accuracy")
    res = out["result"]
    _ok(res.decision == "honest_stop", "iris @0.999 honest-stops", res.decision)
    rep = res.failure_report or {}
    _ok(isinstance(rep, dict) and "power" in rep, "failure report carries a power annotation", list(rep.keys()))
    _ok(rep["power"]["metric"] == "accuracy" and "note" in rep["power"], "power annotation is well-formed",
        rep.get("power"))


if __name__ == "__main__":
    test_power_monotone_and_bounds()
    test_min_n_consistent_with_frozen_certifier()
    test_model_limited_vs_underpowered()
    test_non_binomial_is_honest_none()
    test_wired_into_honest_stop_report()
    print(f"\ntest_power: {_PASS} passed, 0 failed")
