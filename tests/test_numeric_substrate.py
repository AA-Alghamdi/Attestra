"""Hermetic locks for the NUMERICAL SUBSTRATE (vfplatform/numeric_substrate.py) -- the LLM<->numbers firewall.

What these lock (all offline, deterministic):

  * SINGLE SOURCE OF TRUTH -- every substrate producer is byte-identical to calling the FROZEN primitive
    directly (Clopper-Pearson, McNemar, BH-FDR, power, calibration, metrics). The substrate can never become
    a second, weaker source of numerical truth.
  * NO LLM NUMBER DECIDES -- a number an LLM emits is INADMISSIBLE until the substrate recomputes it. A
    CONTRADICTED hint (the LLM lies about a lift) trips NumberLeak in decide(); a VERIFIED hint returns the
    SUBSTRATE'S recomputed value, never the hint; an UNVERIFIABLE hint is refused.
  * AUDITABILITY -- audit() flags every untrusted number and is `clean` iff none could decide by default.
  * RECIPE-NUMBER GUARD -- a proposed recipe is clamped to the audited numeric-gene space; it can never carry
    a decision threshold.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorforge import science as S  # noqa: E402
from vfplatform import battery as B  # noqa: E402
from vfplatform import power as P  # noqa: E402
from vfplatform.numeric_substrate import (CONTRADICTED, UNVERIFIABLE, VERIFIED, NumberLeak,  # noqa: E402
                                          NumericClaim, NumericSubstrate, guard_recipe_numbers)
from vfplatform.recipe import Recipe  # noqa: E402

CAND = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0]
BASE = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]


# ===================================================================== single source of truth (lock to frozen)
def test_accuracy_lower_bound_is_byte_identical_to_frozen_clopper_pearson():
    ns = NumericSubstrate()
    k, n = sum(CAND), len(CAND)
    assert ns.accuracy_lower_bound(CAND, 0.05).value == S.clopper_pearson_lower(k, n, 0.05)


def test_paired_pvalue_is_byte_identical_to_frozen_mcnemar():
    ns = NumericSubstrate()
    assert ns.paired_pvalue(CAND, BASE).value == B.mcnemar_pvalue(CAND, BASE)


def test_fdr_survivors_match_frozen_benjamini_hochberg():
    ns = NumericSubstrate()
    pvals = [0.001, 0.2, 0.04, 0.9, 0.01]
    assert set(ns.fdr_survivors(pvals, alpha=0.1)) == set(B.benjamini_hochberg(pvals, alpha=0.1))


def test_required_n_and_power_match_frozen_power():
    ns = NumericSubstrate()
    assert ns.required_n(0.5, 0.9).value == float(P.min_n_for_power(0.5, 0.9))
    assert ns.power_at_n(40, 0.5, 0.9).value == P.power_at_n(40, 0.5, 0.9)


def test_required_n_is_minus_one_when_uncertifiable_at_any_n():
    # a model not truly above theta cannot be certified above it at ANY n -> honest -1 signal.
    ns = NumericSubstrate()
    assert ns.required_n(0.9, 0.5).value == -1.0


def test_calibration_error_matches_frozen_ece():
    ns = NumericSubstrate()
    conf = [0.9, 0.8, 0.6, 0.55, 0.95, 0.7]
    corr = [1, 1, 0, 1, 1, 0]
    assert ns.calibration_error(conf, corr).value == float(S.expected_calibration_error(conf, corr))


def test_metric_matches_frozen_scorer():
    ns = NumericSubstrate()
    yt = [0, 1, 2, 1, 0, 2]
    yp = [0, 1, 1, 1, 0, 2]
    labels = [0, 1, 2]
    assert ns.metric("accuracy", yt, yp, labels).value == S.score_metric("accuracy", yt, yp, labels)
    assert ns.metric("macro_f1", yt, yp, labels).value == S.score_metric("macro_f1", yt, yp, labels)


def test_paired_lift_is_mean_paired_delta():
    ns = NumericSubstrate()
    expected = sum(a - b for a, b in zip(CAND, BASE)) / len(CAND)
    assert ns.paired_lift(CAND, BASE).value == pytest.approx(expected)


# ===================================================================== no LLM number decides
def test_contradicted_llm_lift_can_never_decide():
    """The decisive falsifiable lock: an LLM claims a big lift on a TIED model. The substrate recomputes ~0,
    marks it CONTRADICTED, and decide() refuses it. The LLM number cannot reach a decision."""
    ns = NumericSubstrate()
    tied_c = [1, 0, 1, 0, 1, 0, 1, 0]
    tied_b = [1, 0, 1, 0, 1, 0, 1, 0]
    hint = ns.claim_llm("lift", 0.15, tol=1e-6)
    ns.recompute(hint, lambda: ns.paired_lift(tied_c, tied_b).value)
    assert hint.verdict == CONTRADICTED
    assert hint.recomputed == 0.0
    with pytest.raises(NumberLeak):
        ns.decide(hint)


def test_verified_llm_hint_returns_substrate_value_not_the_hint():
    """An LLM hint that AGREES is admitted -- but decide() returns the SUBSTRATE'S recomputed number, never
    the hint's (the hint is never trusted in place of the computed value)."""
    ns = NumericSubstrate()
    true_lift = sum(a - b for a, b in zip(CAND, BASE)) / len(CAND)
    hint = ns.claim_llm("lift", true_lift + 1e-9, tol=1e-3)  # within tolerance
    ns.recompute(hint, lambda: ns.paired_lift(CAND, BASE).value)
    assert hint.verdict == VERIFIED
    assert ns.decide(hint) == hint.recomputed == pytest.approx(true_lift)


def test_unverifiable_llm_number_is_refused():
    ns = NumericSubstrate()
    hint = ns.claim_llm("mystery", 0.42)
    ns.recompute(hint, None)                      # no way to recompute -> unverifiable
    assert hint.verdict == UNVERIFIABLE
    with pytest.raises(NumberLeak):
        ns.decide(hint)


def test_recompute_that_raises_is_unverifiable_not_trusted():
    ns = NumericSubstrate()
    hint = ns.claim_llm("boom", 1.0)

    def _raises():
        raise RuntimeError("cannot compute")

    ns.recompute(hint, _raises)
    assert hint.verdict == UNVERIFIABLE
    assert not hint.trusted


def test_trusted_computed_claim_decides_directly():
    ns = NumericSubstrate()
    lb = ns.accuracy_lower_bound(CAND, 0.05)
    assert lb.trusted and ns.decide(lb) == lb.value


def test_unknown_source_is_rejected_at_construction():
    with pytest.raises(ValueError):
        NumericClaim("x", 1.0, source="oracle")


# ===================================================================== auditability
def test_audit_flags_untrusted_and_is_clean_only_when_all_verified():
    ns = NumericSubstrate()
    ns.accuracy_lower_bound(CAND, 0.05)                       # trusted producer
    bad = ns.claim_llm("bad", 0.99, tol=1e-6)
    ns.recompute(bad, lambda: ns.paired_lift([1, 0], [1, 0]).value)   # -> 0.0, contradicted
    a = ns.audit()
    assert a["n_untrusted"] == 1 and a["n_contradicted"] == 1 and a["clean"] is False
    # once the only untrusted claim is verified, the ledger is clean.
    ns2 = NumericSubstrate()
    good = ns2.claim_llm("good", 0.0, tol=1e-6)
    ns2.recompute(good, lambda: ns2.paired_lift([1, 0], [1, 0]).value)  # -> 0.0, verified
    assert ns2.audit()["clean"] is True


def test_every_touched_number_is_in_the_ledger():
    ns = NumericSubstrate()
    ns.paired_lift(CAND, BASE)
    ns.paired_pvalue(CAND, BASE)
    ns.claim_llm("h", 1.0)
    assert ns.audit()["n_claims"] == 3


# ===================================================================== recipe-number guard
def test_guard_clamps_out_of_range_genes_and_is_a_noop_in_range():
    out, info = guard_recipe_numbers(Recipe(backbone="raw", adaptation="linear_probe", head="linear",
                                            lr=10.0, epochs=999))
    assert out.lr == 0.1 and out.epochs == 40
    assert info["clamped"]["lr"] == (10.0, 0.1) and info["clamped"]["epochs"] == (999, 40)

    r = Recipe(backbone="raw", adaptation="linear_probe", head="linear")
    out2, info2 = guard_recipe_numbers(r)
    assert out2 is r and info2["clamped"] == {}


def test_guard_does_not_mutate_the_frozen_input_recipe():
    r = Recipe(backbone="raw", adaptation="linear_probe", head="linear", lr=10.0)
    guard_recipe_numbers(r)
    assert r.lr == 10.0          # frozen input untouched; clamping returns a NEW recipe
