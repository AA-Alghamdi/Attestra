"""Tests for vfplatform/eprocess.py -- the anytime-valid e-process used for repeated-peek racing.

These do NOT exercise the frozen Clopper-Pearson promotion gate (that lives in vectorforge/science.py and
is tested elsewhere). The e-process is an additive racing instrument; here we verify its statistical
contract directly by simulation:
  * VALIDITY (Ville / anytime type-I): under H0 (true p == theta), the probability that sup_n E_n ever
    reaches 1/alpha is <= alpha, even though we peek at every step.
  * POWER: under H1 (true p well above theta), it rejects with high probability after a fraction of the
    stream.
  * LOWER-BOUND COVERAGE: the always-valid lower confidence sequence is <= true p with prob >= 1-alpha.

Run with: /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_eprocess.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.eprocess import EProcess, admissible_lambda_range, grow_lambda  # noqa: E402


# --------------------------------------------------------------------------- structural / unit checks
def test_admissible_range_and_grow_lambda_are_in_bounds():
    for theta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        lo, hi = admissible_lambda_range(theta)
        assert lo < 0 < hi, (theta, lo, hi)
        lam = grow_lambda(theta)
        assert lo <= lam <= hi, (theta, lam, lo, hi)
        # the worst-case factors stay non-negative for any x in [0,1]
        assert 1.0 + lam * (0.0 - theta) >= -1e-12
        assert 1.0 + lam * (1.0 - theta) >= -1e-12


def test_e_value_starts_at_one_and_update_returns_contract():
    ep = EProcess(theta=0.5, alpha=0.05)
    assert abs(ep.e_value - 1.0) < 1e-12
    out = ep.update(0.5)
    assert set(out.keys()) == {"e_value", "reject", "n", "p_value_anytime"}
    assert out["n"] == 1
    assert 0.0 <= out["p_value_anytime"] <= 1.0
    # x == theta is a neutral bet under the grow stake -> E stays ~1
    assert abs(out["e_value"] - 1.0) < 1e-9


def test_rejects_invalid_inputs():
    for bad in [0.0, 1.0, -0.1, 1.2]:
        try:
            EProcess(theta=bad)
            raise AssertionError(f"theta={bad} should be rejected")
        except ValueError:
            pass
    ep = EProcess(theta=0.5)
    for badx in [-0.01, 1.01]:
        try:
            ep.update(badx)
            raise AssertionError(f"x={badx} should be rejected")
        except ValueError:
            pass


def test_deterministic_no_clock():
    # Same inputs -> identical trajectory. No RNG/clock inside.
    xs = [0.2, 0.9, 0.4, 1.0, 0.0, 0.7]
    a = EProcess(theta=0.5, alpha=0.05)
    b = EProcess(theta=0.5, alpha=0.05)
    ra = [a.update(x)["e_value"] for x in xs]
    rb = [b.update(x)["e_value"] for x in xs]
    assert ra == rb
    assert a.lower_bound() == b.lower_bound()


# --------------------------------------------------------------------------- VALIDITY (anytime type-I)
def _run_stream_bernoulli(rng, p, theta, alpha, length, lambdas="grow"):
    """Return ever_rejected: did sup_n E_n reach 1/alpha at any peek over the whole stream."""
    ep = EProcess(theta=theta, alpha=alpha, lambdas=lambdas)
    ever = False
    for _ in range(length):
        x = float(rng.random() < p)
        if ep.update(x)["reject"]:
            ever = True
    return ever


def test_validity_anytime_type_one_under_H0():
    # Under H0 with true p == theta, peeking at EVERY step, empirical sup-rejection rate must be <= alpha
    # (up to Monte-Carlo noise). Ville guarantees <= alpha exactly; we allow a small MC slack.
    rng = np.random.default_rng(20260616)
    theta, alpha = 0.5, 0.05
    n_streams, length = 4000, 200
    rejections = sum(
        int(_run_stream_bernoulli(rng, p=theta, theta=theta, alpha=alpha, length=length))
        for _ in range(n_streams)
    )
    rate = rejections / n_streams
    assert rate <= alpha + 0.015, f"anytime type-I rate {rate:.4f} exceeds {alpha}+slack"


def test_validity_holds_below_H0_boundary():
    # If the truth is BELOW theta (deeper in H0), the rejection rate should be even smaller.
    rng = np.random.default_rng(7)
    theta, alpha = 0.6, 0.05
    n_streams = 2000
    rejections = sum(
        int(_run_stream_bernoulli(rng, p=0.45, theta=theta, alpha=alpha, length=150))
        for _ in range(n_streams)
    )
    rate = rejections / n_streams
    assert rate <= alpha + 0.01, f"sub-boundary rejection rate {rate:.4f} too high"


# --------------------------------------------------------------------------- POWER (under H1)
def test_power_under_H1():
    # True p well above theta: should reject in a large majority of streams, and usually early.
    rng = np.random.default_rng(101)
    theta, alpha = 0.5, 0.05
    n_streams, length = 500, 200
    rejected = 0
    stop_times = []
    for _ in range(n_streams):
        ep = EProcess(theta=theta, alpha=alpha)
        stop = None
        for i in range(length):
            x = float(rng.random() < 0.8)  # p = 0.8 >> theta = 0.5
            if ep.update(x)["reject"]:
                stop = i + 1
                break
        if stop is not None:
            rejected += 1
            stop_times.append(stop)
    power = rejected / n_streams
    assert power >= 0.95, f"power {power:.3f} too low under clear H1"
    med = float(np.median(stop_times))
    assert med <= 0.5 * length, f"median stop {med} not early enough"


def test_p_value_anytime_is_consistent_with_reject():
    # reject iff e_value >= 1/alpha iff p_value_anytime <= alpha.
    rng = np.random.default_rng(3)
    ep = EProcess(theta=0.4, alpha=0.05)
    for _ in range(300):
        out = ep.update(float(rng.random() < 0.75))
        assert out["reject"] == (out["e_value"] >= 1.0 / 0.05)
        assert out["reject"] == (out["p_value_anytime"] <= 0.05 + 1e-12)


# --------------------------------------------------------------------------- LOWER CONFIDENCE SEQUENCE
def test_lower_bound_coverage_le_true_p():
    # The always-valid lower confidence sequence must satisfy L_n <= p with prob >= 1-alpha, for ALL n.
    # Strongest check: does the running L_n ever EXCEED p at any peek?
    rng = np.random.default_rng(555)
    alpha = 0.10
    true_p = 0.7
    n_streams, length = 1500, 120
    violations = 0
    for _ in range(n_streams):
        ep = EProcess(theta=0.5, alpha=alpha)  # theta only sets the e-process H0; LB scans all floors
        violated = False
        for _ in range(length):
            ep.update(float(rng.random() < true_p))
            if ep.lower_bound() > true_p + 1e-9:
                violated = True
                break
        violations += int(violated)
    miss_rate = violations / n_streams
    assert miss_rate <= alpha + 0.02, f"lower-bound miss rate {miss_rate:.4f} exceeds {alpha}+slack"


def test_lower_bound_is_informative_when_p_is_high():
    # With a strongly-above stream, the lower bound should rise above 0 but stay below the true p.
    rng = np.random.default_rng(9)
    ep = EProcess(theta=0.5, alpha=0.05)
    for _ in range(200):
        ep.update(float(rng.random() < 0.9))
    lb = ep.lower_bound()
    assert 0.0 < lb < 0.9 + 1e-9, f"lower bound {lb} not informative / not below true p"


def test_lower_bound_zero_before_any_data():
    ep = EProcess(theta=0.5, alpha=0.05)
    assert ep.lower_bound() == 0.0


# --------------------------------------------------------------------------- runner
TESTS = [
    test_admissible_range_and_grow_lambda_are_in_bounds,
    test_e_value_starts_at_one_and_update_returns_contract,
    test_rejects_invalid_inputs,
    test_deterministic_no_clock,
    test_validity_anytime_type_one_under_H0,
    test_validity_holds_below_H0_boundary,
    test_power_under_H1,
    test_p_value_anytime_is_consistent_with_reject,
    test_lower_bound_coverage_le_true_p,
    test_lower_bound_is_informative_when_p_is_high,
    test_lower_bound_zero_before_any_data,
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
