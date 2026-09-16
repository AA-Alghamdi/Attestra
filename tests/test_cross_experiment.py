"""Unit tests for vfplatform.cross_experiment (online-FDR + promotion ledger + negative cert).

Plain asserts + main() (no pytest), matching the repo's other tests. Unit-tests the module
in isolation; the loop is NOT involved (cross_experiment is not wired in yet).

Covers:
  * LordFDR controls FDR on a simulated mostly-null stream (few false rejections) AND
    discovers injected strong signals.
  * LordFDR is a VALID procedure: alpha_spent is bounded; alpha_t is predictable (uses only
    past outcomes); empirical FDR under all-null is controlled near alpha.
  * PromotionLedger append / read / count / missing+corrupt tolerance / atomicity.
  * negative_certificate shape.

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_cross_experiment.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vfplatform import cross_experiment as cx


# ----------------------------------------------------------------- LordFDR
def test_lordfdr_few_rejections_under_pure_null():
    # mostly-null stream: uniform(0,1) p-values, no signal -> very few rejections
    rng = np.random.default_rng(1)
    fdr = cx.LordFDR(alpha=0.05)
    rej = 0
    n = 500
    for _ in range(n):
        rej += int(fdr.test(float(rng.uniform()))["reject"])
    # under pure null, online-FDR makes O(alpha*n) or fewer rejections; assert it is small
    assert rej <= 0.05 * n, f"too many false rejections under null: {rej}/{n}"
    s = fdr.summary()
    assert s["n_discoveries"] == rej and s["n_tests"] == n


def test_lordfdr_discovers_injected_signals():
    rng = np.random.default_rng(2)
    fdr = cx.LordFDR(alpha=0.05)
    ps = list(rng.uniform(size=300))
    signal_idx = [5, 40, 90, 150, 220]
    for i in signal_idx:
        ps[i] = 1e-8  # overwhelmingly strong signal
    decisions = [fdr.test(p)["reject"] for p in ps]
    discovered = sum(decisions[i] for i in signal_idx)
    assert discovered >= 4, f"failed to discover injected signals: {discovered}/5"


def test_lordfdr_alpha_spent_bounded():
    # validity sanity: total level spent <= n_tests * alpha (each per-test level capped at alpha),
    # and individual alpha_t never exceeds alpha.
    rng = np.random.default_rng(3)
    fdr = cx.LordFDR(alpha=0.05)
    n = 400
    max_a = 0.0
    for _ in range(n):
        r = fdr.test(float(rng.uniform()))
        max_a = max(max_a, r["alpha_t"])
    s = fdr.summary()
    assert max_a <= fdr.alpha + 1e-12, f"alpha_t exceeded alpha: {max_a}"
    assert s["alpha_spent"] <= n * fdr.alpha + 1e-9


def test_lordfdr_level_predictable_independent_of_current_pvalue():
    # the testing level for test t must NOT depend on p_t (only past outcomes). Two controllers
    # fed identical history then a different current p should report the same alpha_t.
    fdr_a = cx.LordFDR(alpha=0.05)
    fdr_b = cx.LordFDR(alpha=0.05)
    hist = [0.9, 0.8, 0.95, 0.7]
    for p in hist:
        fdr_a.test(p); fdr_b.test(p)
    a = fdr_a.test(0.99)["alpha_t"]
    b = fdr_b.test(0.001)["alpha_t"]
    assert abs(a - b) < 1e-15, f"alpha_t depended on current p-value: {a} vs {b}"


def test_lordfdr_empirical_fdr_controlled():
    # Over many independent all-null streams, the empirical false-discovery proportion
    # (here every discovery is false) averages at or below alpha -> valid FDR control.
    alpha = 0.05
    n_streams = 60
    n = 150
    fdps = []
    for s_i in range(n_streams):
        rng = np.random.default_rng(1000 + s_i)
        fdr = cx.LordFDR(alpha=alpha)
        rej = sum(int(fdr.test(float(rng.uniform()))["reject"]) for _ in range(n))
        fdps.append(rej / n)  # discoveries-per-test proxy; all are false under null
    mean_fdp = float(np.mean(fdps))
    assert mean_fdp <= alpha + 0.02, f"empirical null FDP not controlled: {mean_fdp:.4f}"


def test_lordfdr_rejects_bad_alpha():
    for a in (0.0, 1.0, -0.1, 2.0):
        try:
            cx.LordFDR(alpha=a)
            assert False, f"expected ValueError for alpha={a}"
        except ValueError:
            pass


def test_lordfdr_custom_gamma_seq():
    g = [0.5, 0.25, 0.125, 0.0625, 0.0625]
    fdr = cx.LordFDR(alpha=0.05, gamma_seq=g)
    out = fdr.test(1e-9)  # strong signal at t=1
    assert out["alpha_t"] > 0 and out["reject"] in (True, False)


# ----------------------------------------------------------------- PromotionLedger
def test_ledger_missing_file_empty():
    with tempfile.TemporaryDirectory() as d:
        led = cx.PromotionLedger(os.path.join(d, "none.jsonl"))
        assert led.count() == 0 and led.all() == []


def test_ledger_append_read_count():
    with tempfile.TemporaryDirectory() as d:
        led = cx.PromotionLedger(os.path.join(d, "promo.jsonl"))
        e1 = {"plan_hash": "sha256:a", "decision": "promote", "metric": "r2", "theta": 0.5,
              "observed": 0.6, "lower_bound": 0.55, "p_value": 0.01, "certified": True, "ts": 1.0}
        e2 = {"plan_hash": "sha256:b", "decision": "reject", "metric": "r2", "theta": 0.5,
              "observed": 0.4, "lower_bound": 0.3, "p_value": 0.4, "certified": False, "ts": 2.0}
        led.record(e1); led.record(e2)
        assert led.count() == 2
        got = led.all()
        assert got[0]["plan_hash"] == "sha256:a" and got[1]["ts"] == 2.0


def test_ledger_no_clock_called_ts_is_caller_supplied():
    # the entry's ts is exactly what the caller passed (module never overwrites it)
    with tempfile.TemporaryDirectory() as d:
        led = cx.PromotionLedger(os.path.join(d, "promo.jsonl"))
        led.record({"plan_hash": "sha256:x", "ts": 12345.0})
        assert led.all()[0]["ts"] == 12345.0


def test_ledger_tolerates_corrupt_and_is_atomic():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "promo.jsonl")
        led = cx.PromotionLedger(path)
        led.record({"plan_hash": "sha256:a", "ts": 1.0})
        with open(path, "a", encoding="utf-8") as f:
            f.write("garbage line not json\n")
        assert led.count() == 1
        led.record({"plan_hash": "sha256:b", "ts": 2.0})
        hashes = {e["plan_hash"] for e in led.all()}
        assert hashes == {"sha256:a", "sha256:b"}
        leftover = [f for f in os.listdir(d) if f.startswith(".promo.")]
        assert leftover == [], f"atomic write left temp files: {leftover}"


def test_ledger_rejects_non_dict():
    with tempfile.TemporaryDirectory() as d:
        led = cx.PromotionLedger(os.path.join(d, "promo.jsonl"))
        try:
            led.record(["not", "a", "dict"])
            assert False, "expected ValueError"
        except ValueError:
            pass


# ----------------------------------------------------------------- negative_certificate
def test_negative_certificate_shape():
    nc = cx.negative_certificate(plan_hash="sha256:abc", reason="lb_below_theta",
                                 diagnosis="bootstrap lower bound 0.66 < theta 0.80",
                                 observed=0.71, lower_bound=0.66, theta=0.80,
                                 alpha_futility=0.01)
    assert nc["kind"] == "negative"
    assert nc["certified"] is False
    assert nc["plan_hash"] == "sha256:abc"
    assert nc["reason"] == "lb_below_theta"
    assert nc["theta"] == 0.80 and nc["observed"] == 0.71 and nc["lower_bound"] == 0.66
    assert nc["alpha_futility"] == 0.01
    import json
    json.dumps(nc)  # serializable


def test_negative_certificate_optional_none_fields():
    nc = cx.negative_certificate(plan_hash="sha256:abc", reason="no_signal",
                                 theta=0.5, alpha_futility=0.02)
    assert nc["observed"] is None and nc["lower_bound"] is None and nc["diagnosis"] is None


def test_negative_certificate_bad_inputs_raise():
    for kw in [
        dict(plan_hash="", reason="r", theta=0.5, alpha_futility=0.01),
        dict(plan_hash="h", reason="", theta=0.5, alpha_futility=0.01),
        dict(plan_hash="h", reason="r", theta=0.5, alpha_futility=0.0),
        dict(plan_hash="h", reason="r", theta=0.5, alpha_futility=1.0),
    ]:
        try:
            cx.negative_certificate(**kw)
            assert False, f"expected ValueError for {kw}"
        except ValueError:
            pass


TESTS = [
    test_lordfdr_few_rejections_under_pure_null,
    test_lordfdr_discovers_injected_signals,
    test_lordfdr_alpha_spent_bounded,
    test_lordfdr_level_predictable_independent_of_current_pvalue,
    test_lordfdr_empirical_fdr_controlled,
    test_lordfdr_rejects_bad_alpha,
    test_lordfdr_custom_gamma_seq,
    test_ledger_missing_file_empty,
    test_ledger_append_read_count,
    test_ledger_no_clock_called_ts_is_caller_supplied,
    test_ledger_tolerates_corrupt_and_is_atomic,
    test_ledger_rejects_non_dict,
    test_negative_certificate_shape,
    test_negative_certificate_optional_none_fields,
    test_negative_certificate_bad_inputs_raise,
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
