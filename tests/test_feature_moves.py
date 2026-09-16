"""Tests for vfplatform/feature_moves.py (NEW module; read-only on the trust core).

What these prove:
  1. SHAPE/CONTRACT: a wrapped (transform, base_family) candidate is a single object with the EXACT
     sklearn .fit/.predict contract (a Pipeline) that LocalWorker.fit_score and the certifier rely on.
  2. CLAMP PARITY: a wrapped entry's params clamp through the SAME harness clamp_params path (namespaced
     t__/b__), so out-of-range / unknown keys can never reach a transformer or estimator.
  3. FROZEN-PATH INTEGRATION: the wrapped families resolve through harness.resolve_family /
     move_from_proposal and are surfaced by llm_moves.propose_moves' deterministic fallback -- with NO
     change to any frozen module.
  4. SELECTION-VALID-FOR-DATA: k / n_components are always sized to the real feature width at fit time
     (no sklearn out-of-range error on narrow or wide matrices).
  5. THE HEADLINE: on a MADELON-style high-dim problem (5 informative among 200 noise, non-linear target),
     a feature-SELECTION + NON-LINEAR candidate clearly BEATS the raw logistic baseline -- through the same
     builder path the cycle uses.
  6. FROZEN FINGERPRINTS unchanged (science.py / sealed.py).

Run:  PYTHONPATH=. /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_feature_moves.py
Also collectable by pytest (test_* functions, plain asserts).
"""
import hashlib
import os
import sys

import numpy as np
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.harness import (CLASSIFICATION_CATALOG, REGRESSION_CATALOG, resolve_family,
                                move_from_proposal)
from vfplatform import feature_moves as fm
from vfplatform import llm_moves


# ----------------------------------------------------------------- 1. shape / fit-predict contract
def test_wrapped_entry_is_pipeline_with_fit_predict():
    e = fm.make_wrapped_entry("selectk", "hist_gbm", CLASSIFICATION_CATALOG)
    assert e is not None and e.family == "selectk+hist_gbm"
    est = e.build({"t__keep_frac": 0.1, "b__it": 200}, seed=0)
    assert isinstance(est, Pipeline), "wrapped candidate must be a sklearn Pipeline"
    assert hasattr(est, "fit") and hasattr(est, "predict"), "must satisfy the fit/predict contract"
    # exercise the real contract on a tiny matrix
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 40)); y = (X[:, 0] + X[:, 1] > 0).astype(int)
    est.fit(X, y)
    pred = est.predict(X)
    assert pred.shape == (60,) and set(np.unique(pred)).issubset({0, 1})


def test_predict_proba_present_iff_base_has_it():
    # logistic has predict_proba -> the wrapped Pipeline exposes it
    e = fm.make_wrapped_entry("pca", "logistic", CLASSIFICATION_CATALOG)
    est = e.build(e.clamp_params({"t__keep_frac": 0.5, "b__C": 1.0}), seed=0)
    assert hasattr(est, "predict_proba")
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 20)); y = (X[:, 0] > 0).astype(int)
    est.fit(X, y)
    p = est.predict_proba(X)
    assert p.shape == (50, 2)


def test_unknown_names_return_none():
    assert fm.make_wrapped_entry("nope", "hist_gbm", CLASSIFICATION_CATALOG) is None
    assert fm.make_wrapped_entry("selectk", "no_such_family", CLASSIFICATION_CATALOG) is None
    assert fm.make_two_stage_entry("selectk", "poly", "no_such", CLASSIFICATION_CATALOG) is None


# ----------------------------------------------------------------- 2. clamp parity (frozen clamp path)
def test_params_clamp_through_harness_path():
    e = fm.make_wrapped_entry("selectk", "logistic", CLASSIFICATION_CATALOG)
    # out-of-range on BOTH halves + an unknown key -> all clamped/dropped, every spec'd key filled
    clamped = e.clamp_params({"t__keep_frac": 99.0, "b__C": 1e9, "bogus": 123})
    assert "bogus" not in clamped
    assert clamped["t__keep_frac"] <= 1.0, "keep_frac clamped to its (0.01,1.0) range"
    assert clamped["b__C"] <= 1e3, "base C clamped to logistic's (1e-3,1e3) range"
    # absent keys filled with safe defaults -> fully buildable
    clamped2 = e.clamp_params({})
    assert set(clamped2) == set(e.params)
    est = e.build(clamped2, 0)
    assert isinstance(est, Pipeline)


def test_namespacing_prevents_collision():
    # both selectk and pca own a 'keep_frac'; in a two-base scenario the t__/b__ split keeps them distinct.
    e = fm.make_wrapped_entry("pca", "knn", CLASSIFICATION_CATALOG)  # knn has 'k', pca has 'keep_frac'
    assert "t__keep_frac" in e.params and "b__k" in e.params
    assert "keep_frac" not in e.params and "k" not in e.params


# ----------------------------------------------------------------- 3. frozen-path integration
def test_resolve_family_handles_wrapped():
    cat = dict(CLASSIFICATION_CATALOG)
    cat.update(fm.enumerate_feature_candidates(cat, "binary"))
    r = resolve_family(cat, "selectk+hist_gbm", {"t__keep_frac": 0.05, "b__it": 300})
    assert r is not None
    fam, ctor, clamped = r
    assert fam == "selectk+hist_gbm"
    assert isinstance(ctor(0), Pipeline)


def test_move_from_proposal_builds_named_move():
    cat = dict(CLASSIFICATION_CATALOG)
    cat.update(fm.enumerate_feature_candidates(cat, "binary"))
    m = move_from_proposal(cat, "selectk+poly+logistic", {"t1__keep_frac": 0.025, "b__C": 1.0})
    assert m is not None and m.name.startswith("selectk+poly+logistic")
    assert len(m.families) == 1
    name, ctor, params = m.families[0]
    assert params.get("family") == "selectk+poly+logistic"
    assert isinstance(ctor(0), Pipeline)


def test_proposer_fallback_surfaces_feature_families():
    cat = dict(CLASSIFICATION_CATALOG)
    cat.update(fm.enumerate_feature_candidates(cat, "binary"))
    moves, source = llm_moves.propose_moves(
        {"n_features": 200}, "", {"headroom": 0.2}, "binary", "accuracy", set(),
        use_llm=False, catalog=cat, limit=24)
    assert source == "fallback"
    fams = {mv.name.split("|")[0] for mv in moves}
    feat_fams = {f for f in fams if "+" in f and any(
        t in f for t in ("selectk", "variance", "pca", "robust", "poly"))}
    assert feat_fams, f"deterministic proposer must surface feature-eng families; got {sorted(fams)}"


def test_enumerate_only_wraps_present_bases():
    # a catalog with a single base -> only that base gets wrapped (respects provider-filtered catalog)
    tiny = {"logistic": CLASSIFICATION_CATALOG["logistic"]}
    feat = fm.enumerate_feature_candidates(tiny, "binary")
    for name in feat:
        base = name.split("+")[-1]
        assert base == "logistic", f"{name} wraps a base not in the tiny catalog"
    assert "selectk+logistic" in feat
    # regression catalog routes to regression bases
    rfeat = fm.enumerate_feature_candidates(dict(REGRESSION_CATALOG), "regression")
    assert any(n.endswith("ridge") for n in rfeat)
    assert all("logistic" not in n for n in rfeat)


# ----------------------------------------------------------------- 4. selection valid for data width
def test_k_and_components_valid_on_narrow_and_wide():
    rng = np.random.default_rng(2)
    for p in (3, 200):                              # narrow (k floor) and wide
        X = rng.standard_normal((40, p)); y = (X[:, 0] > 0).astype(int)
        for tname in ("selectk", "pca", "variance", "robust"):
            e = fm.make_wrapped_entry(tname, "logistic", CLASSIFICATION_CATALOG)
            est = e.build(e.clamp_params({"t__keep_frac": 0.05}), 0)
            est.fit(X, y)                            # must not raise (k/n_components sized to width)
            assert est.predict(X[:5]).shape == (5,)


# ----------------------------------------------------------------- 5. the headline (MADELON-style win)
def test_feature_selection_beats_raw_logistic_on_madelon():
    X, y = fm._madelon_like(seed=0)
    n = len(y); cut = int(n * 0.7)
    Xtr, Xva, ytr, yva = X[:cut], X[cut:], y[:cut], y[cut:]

    def acc(entry, params):
        est = entry.build(entry.clamp_params(params), 0)
        est.fit(Xtr, ytr)
        return float((est.predict(Xva) == yva).mean())

    cat = CLASSIFICATION_CATALOG
    feat = fm.enumerate_feature_candidates(cat, "binary")
    raw_log = acc(cat["logistic"], {"C": 1.0})
    sel_nonlin = acc(feat["selectk+hist_gbm"], {"t__keep_frac": 0.05, "b__it": 300})
    assert sel_nonlin > raw_log + 0.05, (
        f"selectk+hist_gbm ({sel_nonlin:.4f}) must clearly beat raw logistic ({raw_log:.4f})")


# ----------------------------------------------------------------- 6. frozen fingerprints unchanged
def test_frozen_fingerprints_unchanged():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def sha(rel):
        with open(os.path.join(root, rel), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    assert sha("vectorforge/science.py").startswith("b564fba2"), "science.py changed!"
    assert sha("vfplatform/sealed.py").startswith("30ad6245"), "sealed.py changed!"


_TESTS = [
    test_wrapped_entry_is_pipeline_with_fit_predict,
    test_predict_proba_present_iff_base_has_it,
    test_unknown_names_return_none,
    test_params_clamp_through_harness_path,
    test_namespacing_prevents_collision,
    test_resolve_family_handles_wrapped,
    test_move_from_proposal_builds_named_move,
    test_proposer_fallback_surfaces_feature_families,
    test_enumerate_only_wraps_present_bases,
    test_k_and_components_valid_on_narrow_and_wide,
    test_feature_selection_beats_raw_logistic_on_madelon,
    test_frozen_fingerprints_unchanged,
]


def main():
    p = f = 0
    for t in _TESTS:
        try:
            t(); print(f"  PASS  {t.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}"); f += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  ERROR {t.__name__}: {e}"); traceback.print_exc(); f += 1
    print(f"  ---- {p} passed, {f} failed ----")
    return f


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
