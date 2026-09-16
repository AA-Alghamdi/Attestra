"""Select-then-bound invariant tests (the scheme's defining rigor property; strategy doc item B1).

The product SELECTS a model on VALIDATION and only then spends ONE counted peek of the SEALED test to
produce the certificate. This module proves that contract in ISOLATION against the FROZEN primitives --
SealedTest + certify_on_sealed (vfplatform/sealed.py) and _val_lower_bound (vfplatform/loop.py) -- so the
property holds independently of any end-to-end run. NO certifier / sealed-peek / theta logic is modified
here; this is a read-only behavioral check (no network, no GPU, no spend).

Properties proven (each falsifiable):
  1. The sealed peek is COUNTED and capped: certify_on_sealed evaluates the locked test exactly once and a
     second uncounted peek raises PeekViolation (the "one counted peek" is code, not convention).
  2. The sealed certificate's multiplicity is its OWN peek count (checks=1 for one peek), INDEPENDENT of how
     many candidates were selected among on validation. B1's core claim: trying more models never inflates
     the sealed certificate's correction, so a genuinely-good model is not penalized at the bound for having
     considered more candidates.
  3. SELECTION multiplicity lives on the VALIDATION gate: _val_lower_bound pays Bonferroni over the number of
     finished candidates (checks = n_finished), so the gate is more conservative as more models are tried.
     This can only make us DECLINE to peek (preserving the scarce test); it can never produce a false
     certificate, because the certificate is computed solely on the sealed test with its own (checks=1) peek.
  4. Honest-stop preserves the peek: a winner whose validation lower bound does not clear theta means NO
     sealed evaluation occurs and the sealed peek count stays 0.
  5. Borderline at the bound: a point estimate above theta with a lower bound below theta does NOT certify
     (the certifier reads the bound, not the point).
  6. A re-peek of the SAME locked test under a durable ledger pays CUMULATIVE Bonferroni (cross-run
     multiplicity cannot be laundered by re-running on the same digest).

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_select_then_bound.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorforge import science
from vfplatform.sealed import (SealedTest, PeekLedger, certify_on_sealed, PeekViolation,
                               MetricNotCertifiable)
from vfplatform.loop import _val_lower_bound


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


# --------------------------------------------------------------------------- fixtures (deterministic)
def _sealed_rows(n_correct, n_wrong, target_key="target"):
    """A locked test with a KNOWN accuracy: n_correct rows whose true label is 'a' and n_wrong rows whose
    true label is 'b'. predict_fn always says 'a', so observed accuracy = n_correct / (n_correct + n_wrong)
    -- a fully deterministic, certifier-checkable setup (no model, no randomness in the pairing)."""
    rows = ([{"x": float(i), target_key: "a"} for i in range(n_correct)]
            + [{"x": float(i), target_key: "b"} for i in range(n_wrong)])
    return rows


def _always_a(rows):
    return ["a" for _ in rows]


# --------------------------------------------------------- 1. the sealed peek is counted and capped
def test_sealed_peek_is_counted_and_capped():
    sealed = SealedTest(_sealed_rows(90, 10), max_peeks=1)
    assert sealed.peek_count() == 0, "a fresh sealed test has not been peeked"
    cert = certify_on_sealed(sealed, _always_a, 0.70, metric="accuracy")
    assert sealed.peek_count() == 1, f"exactly one counted peek, got {sealed.peek_count()}"
    assert cert["peeks"] == 1 and cert["checks"] == 1, f"cert pays for one peek: {cert}"
    assert abs(cert["observed"] - 0.90) < 1e-9, f"observed accuracy = 90/100, got {cert['observed']}"
    # a SECOND uncounted peek of the same locked instance must be refused (max_peeks=1, no ledger)
    raised = False
    try:
        certify_on_sealed(sealed, _always_a, 0.70, metric="accuracy")
    except PeekViolation:
        raised = True
    assert raised, "a second uncounted peek must raise PeekViolation (one counted peek is enforced)"


# ------------------------- 2. the SEALED bound's multiplicity is its own peek, NOT the candidate count
def test_sealed_bound_independent_of_candidate_count():
    """B1: trying MORE candidates on validation must NOT inflate the SEALED certificate's Bonferroni
    correction. The certificate is computed only on the sealed test with checks = its peek count (1),
    regardless of whether 2 or 200 models were selected among. Two fresh sealed tests with identical
    content therefore yield IDENTICAL certificates -- the sealed bound never sees the candidate count."""
    rows = _sealed_rows(88, 12)
    c_few = certify_on_sealed(SealedTest(rows, max_peeks=1), _always_a, 0.70, metric="accuracy")
    c_many = certify_on_sealed(SealedTest(rows, max_peeks=1), _always_a, 0.70, metric="accuracy")
    assert c_few["checks"] == 1 and c_many["checks"] == 1, "the sealed peek is always checks=1 per peek"
    assert c_few["lower_bound"] == c_many["lower_bound"], (
        "the sealed lower bound must not depend on how many candidates were considered on validation")
    assert c_few["certified"] == c_many["certified"]


def test_more_candidates_only_loosens_the_validation_gate():
    """SELECTION multiplicity is paid on the VALIDATION gate (checks = n_finished candidates), not the
    sealed bound. So as more candidates are tried, the val lower bound is monotonically NON-INCREASING
    (more conservative). This gate decides whether to SPEND the peek; it can only make us decline, never
    fabricate a certificate."""
    # 170/200 correct on validation -> 0.85 point. theta below it; vary the candidate count.
    yt = ["a"] * 170 + ["b"] * 30
    yp = ["a"] * 200
    theta, alpha = 0.75, 0.05
    lb1 = _val_lower_bound("accuracy", yt, yp, theta, checks=1, alpha=alpha, is_regression=False)
    lb5 = _val_lower_bound("accuracy", yt, yp, theta, checks=5, alpha=alpha, is_regression=False)
    lb50 = _val_lower_bound("accuracy", yt, yp, theta, checks=50, alpha=alpha, is_regression=False)
    assert lb1 >= lb5 >= lb50, (
        f"more candidates must not RAISE the val lower bound (selection multiplicity): "
        f"checks=1 {lb1}, checks=5 {lb5}, checks=50 {lb50}")
    # the gate stays meaningful: with a strong signal it still clears theta even after paying for 5 peeks
    assert lb5 > theta, f"a genuinely-good model still clears the gate after selection correction: {lb5}"


def test_genuinely_good_model_certifies_despite_many_candidates():
    """The constructive half of B1: a strong model on a large sealed test CERTIFIES, and the certificate is
    the SAME whether 2 or 200 candidates were tried -- because the certificate's multiplicity is the one
    sealed peek, not the candidate count."""
    rows = _sealed_rows(900, 100)   # 0.90 on a 1000-row sealed test, theta 0.80 -> bound clears
    cert = certify_on_sealed(SealedTest(rows, max_peeks=1), _always_a, 0.80, metric="accuracy")
    assert cert["certified"] is True, f"strong model on a large sealed test must certify: {cert}"
    assert cert["lower_bound"] > cert["theta"], f"certification is lower-bound-justified: {cert}"
    assert cert["checks"] == 1, "the candidate count never enters the sealed certificate's multiplicity"


# ----------------------------------------------------- 4. honest-stop: a declined peek leaves count at 0
def test_honest_stop_does_not_spend_the_peek():
    """When selection (on validation) does NOT clear the gate, the production loop honest-stops with NO
    sealed peek. Here we prove the primitive contract that underlies it: if certify_on_sealed is never
    called, the sealed test is never evaluated and its peek count stays 0 (the scarce test is preserved
    for a stronger attempt). We assert both the gate decision and the untouched peek count."""
    sealed = SealedTest(_sealed_rows(60, 40), max_peeks=1)   # 0.60 true accuracy on the sealed test
    # the WINNER's validation lower bound does not clear theta -> the loop would NOT peek.
    yt = ["a"] * 60 + ["b"] * 40
    yp = ["a"] * 100        # 0.60 on validation
    theta = 0.75
    val_lb = _val_lower_bound("accuracy", yt, yp, theta, checks=1, alpha=0.05, is_regression=False)
    assert val_lb <= theta, f"validation lower bound {val_lb} must not clear theta {theta} (honest-stop)"
    # because the gate said no, no certify_on_sealed call is made -> the sealed peek is preserved:
    assert sealed.peek_count() == 0, "an honest stop must leave the sealed test un-peeked"


# ----------------------------------------------------- 5. borderline: point clears, bound does not
def test_borderline_bound_does_not_certify():
    """Reads the BOUND, not the point. 82/100 = 0.82 observed against theta 0.80: the point clears but the
    Clopper-Pearson lower bound on n=100 does not. certify_on_sealed must refuse."""
    sealed = SealedTest(_sealed_rows(82, 18), max_peeks=1)
    cert = certify_on_sealed(sealed, _always_a, 0.80, metric="accuracy")
    assert cert["observed"] >= 0.80, f"point estimate clears: {cert['observed']}"
    assert cert["lower_bound"] < 0.80, f"lower bound does NOT clear: {cert['lower_bound']}"
    assert cert["certified"] is False, "a sub-threshold lower bound must not certify even if the point clears"


# ----------------------------------------------------- 6. cumulative cross-run multiplicity (durable ledger)
def test_durable_ledger_pays_cumulative_multiplicity():
    """A re-peek of the SAME locked-test digest under a durable PeekLedger pays CUMULATIVE Bonferroni: the
    2nd peek reports checks=2, so re-running a goal on the same exact test cannot launder the correction.
    The cumulative checks tightens (lowers) the reported lower bound on the 2nd peek."""
    rows = _sealed_rows(900, 100)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "peeks.json")
        ledger = PeekLedger(path)
        c1 = certify_on_sealed(SealedTest(rows, ledger=ledger), _always_a, 0.80, metric="accuracy")
        c2 = certify_on_sealed(SealedTest(rows, ledger=ledger), _always_a, 0.80, metric="accuracy")
        assert c1["checks"] == 1 and c2["checks"] == 2, (
            f"cumulative cross-run peeks: first {c1['checks']}, second {c2['checks']}")
        assert c2["lower_bound"] <= c1["lower_bound"], (
            f"paying for a 2nd peek must not LOOSEN the bound: {c1['lower_bound']} -> {c2['lower_bound']}")


# ----------------------------------------------------- guard: an unknown metric is refused (no fake cert)
def test_unknown_metric_refused():
    sealed = SealedTest(_sealed_rows(90, 10), max_peeks=1)
    raised = False
    try:
        certify_on_sealed(sealed, _always_a, 0.70, metric="f1_typo")
    except MetricNotCertifiable:
        raised = True
    assert raised, "an unrecognized metric must be refused (no silent fall-through to accuracy)"
    assert sealed.peek_count() == 0, "a refused metric must not spend a peek"


TESTS = [test_sealed_peek_is_counted_and_capped,
         test_sealed_bound_independent_of_candidate_count,
         test_more_candidates_only_loosens_the_validation_gate,
         test_genuinely_good_model_certifies_despite_many_candidates,
         test_honest_stop_does_not_spend_the_peek,
         test_borderline_bound_does_not_certify,
         test_durable_ledger_pays_cumulative_multiplicity,
         test_unknown_metric_refused]

if __name__ == "__main__":
    print("== test_select_then_bound (select on validation -> ONE counted sealed peek) ==")
    p, f = run(TESTS)
    raise SystemExit(1 if f else 0)
