"""Tests for vfplatform/knowledge.py -- the PROPOSE-side, NON-BINDING method-knowledge surface.

ALL tests run OFFLINE: the deterministic rule table needs no network/LLM, and the LLM rung is
exercised by MONKEYPATCHING vectorforge.llm_shell.ops._call_claude (so no key/network is touched).
The contract under test: advice is advisory only, family hints are intersected with the catalog,
feature moves stay in the closed registry, and the LLM/web rungs can only ADD in-vocab hints --
never remove the deterministic floor and never widen the vocabulary.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import knowledge as K
from vfplatform.harness import catalog_for
from vectorforge.llm_shell import ops


# --------------------------------------------------------------------------- fixtures: real catalogs
CLF = catalog_for("tabular", "binary")
REG = catalog_for("tabular", "regression")
TXT = catalog_for("text", "multiclass")


def _prof(**kw):
    base = {"kind": "tabular", "task_type": "binary", "metric": "accuracy",
            "n_train": 2000, "n_val": 500, "n_features": 40, "n_classes": 2, "threshold": 0.8}
    base.update(kw)
    return base


# =========================================================================== deterministic rule table
def test_offline_no_llm_no_network_default():
    """Default call is fully offline (use_llm defaults False); source is 'rules'."""
    adv = K.suggest_methods(_prof(), {"headroom": 0.1, "below_bar": True}, CLF)
    assert adv.source == "rules"
    assert adv.family_hints, "should produce at least one family hint"


def test_family_hints_intersected_with_catalog():
    """Every family hint name MUST be a key of the supplied catalog (the bounding step)."""
    adv = K.suggest_methods(_prof(), {"headroom": 0.12, "below_bar": True,
                                      "min_class_recall": 0.4, "ece": 0.2}, CLF)
    for s in adv.family_hints:
        assert s.name in CLF, f"out-of-catalog family hint leaked: {s.name}"


def test_feature_moves_in_closed_registry():
    """Every feature move name MUST be in the closed FEATURE_MOVES registry."""
    adv = K.suggest_methods(_prof(n_features=400, n_train=300),
                            {"headroom": 0.1, "below_bar": True}, CLF)
    assert adv.feature_moves
    for s in adv.feature_moves:
        assert s.name in K.known_feature_moves(), f"unknown feature move: {s.name}"


def test_high_dim_nonlinear_suggests_mutual_info_and_boosting():
    """The headline example: high-dim + nonlinear headroom -> mutual-info selection + gradient boosting."""
    adv = K.suggest_methods(_prof(n_features=300, n_train=400),
                            {"headroom": 0.12, "below_bar": True, "min_class_recall": 0.7, "ece": 0.04},
                            CLF)
    move_names = {s.name for s in adv.feature_moves}
    fam_names = {s.name for s in adv.family_hints}
    assert "mutual_info_select" in move_names
    assert "hist_gbm" in fam_names              # gradient boosting is suggested
    # boosting should outrank a linear family in the priority ordering
    order = adv.ordered_families(CLF)
    assert order.index("hist_gbm") < order.index("logistic")


def test_imbalance_suggests_class_weight_balance():
    """A low minority-class recall -> class_weight_balance is suggested, high weight."""
    adv = K.suggest_methods(_prof(), {"headroom": 0.01, "below_bar": False,
                                      "min_class_recall": 0.4, "ece": 0.05}, CLF)
    cw = [s for s in adv.feature_moves if s.name == "class_weight_balance"]
    assert cw and cw[0].weight > 0.5


def test_miscalibration_prefers_calibratable_family():
    """High ECE -> a calibratable family (logistic) is preferred over trees in the ordering."""
    adv = K.suggest_methods(_prof(n_train=5000, n_features=20),
                            {"headroom": 0.0, "below_bar": False,
                             "min_class_recall": 0.9, "ece": 0.19}, CLF)
    fam_names = {s.name for s in adv.family_hints}
    assert "logistic" in fam_names
    order = adv.ordered_families(CLF)
    assert order.index("logistic") < order.index("hist_gbm")


def test_text_catalog_suggests_text_families_only():
    """On a text catalog, hints stay inside the text zoo (no tabular family names leak in)."""
    adv = K.suggest_methods({"kind": "text", "task_type": "multiclass", "n_train": 3000,
                             "n_features": 5000, "n_classes": 4},
                            {"headroom": 0.08, "below_bar": True, "min_class_recall": 0.6, "ece": 0.07},
                            TXT)
    for s in adv.family_hints:
        assert s.name in TXT
    assert any(s.name == "tfidf+complement_nb" for s in adv.family_hints)


def test_regression_catalog_suggests_regression_families():
    adv = K.suggest_methods(_prof(task_type="regression", metric="r2"),
                            {"headroom": 0.2, "below_bar": True}, REG)
    for s in adv.family_hints:
        assert s.name in REG
    assert any(s.name == "hist_gbm_reg" for s in adv.family_hints)


def test_tiny_data_deprioritizes_neural():
    """Very small training set -> the neural family gets a NEGATIVE advisory weight."""
    adv = K.suggest_methods(_prof(n_train=120, n_features=10),
                            {"headroom": 0.1, "below_bar": True}, CLF)
    mlp = [s for s in adv.family_hints if s.name == "mlp"]
    assert mlp and mlp[0].weight < 0, "neural family should be de-prioritized on tiny data"


def test_never_empty_clean_profile():
    """A clean above-bar profile with no lever still returns a non-empty, valid ordering."""
    adv = K.suggest_methods(_prof(), {"headroom": -0.05, "below_bar": False,
                                      "min_class_recall": 0.95, "ece": 0.02}, CLF)
    assert adv.family_hints
    assert set(adv.ordered_families(CLF)) == set(CLF)        # complete, no extras


def test_ordered_families_complete_and_bounded():
    """ordered_families must be a permutation of the catalog keys (no missing, no extra)."""
    adv = K.suggest_methods(_prof(n_features=300, n_train=400),
                            {"headroom": 0.1, "below_bar": True}, CLF)
    order = adv.ordered_families(CLF)
    assert sorted(order) == sorted(CLF)


def test_none_diagnosis_round_zero():
    """diagnosis=None (round 0) must not crash and must still advise."""
    adv = K.suggest_methods(_prof(), None, CLF)
    assert adv.family_hints
    assert adv.source == "rules"


def test_as_dict_is_json_shaped():
    adv = K.suggest_methods(_prof(), {"headroom": 0.1, "below_bar": True}, CLF)
    d = adv.as_dict()
    assert set(d) == {"family_hints", "feature_moves", "rationale", "notes", "source"}
    assert all(set(h) == {"kind", "name", "weight", "reason"} for h in d["family_hints"])


# ================================================================================= LLM rung (mocked)
def _fake_claude_factory(payload):
    """Return a stand-in for ops._call_claude that returns (payload, usage) without any network."""
    def _fake(req, api_key, timeout):
        return payload, {"input_tokens": 1, "output_tokens": 1}
    return _fake


def test_llm_rung_adds_in_vocab_hint(monkeypatch, tmp_path):
    """With the LLM mocked to return a VALID in-vocab hint, source becomes rules+llm and the hint is
    unioned in. A fresh cache_path avoids replaying any prior record."""
    payload = {"family_hints": [{"name": "svc_rbf", "weight": 0.95, "reason": "kernel capacity"}],
               "feature_moves": [{"name": "standardize", "weight": 0.7, "reason": "scale for svc"}],
               "rationale": "kernel methods fit this regime"}
    monkeypatch.setattr(ops, "_call_claude", _fake_claude_factory(payload))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    adv = K.suggest_methods(_prof(), {"headroom": 0.1, "below_bar": True}, CLF,
                            use_llm=True, cache_path=str(tmp_path / "c.jsonl"))
    assert adv.source == "rules+llm"
    fam = {s.name for s in adv.family_hints}
    assert "svc_rbf" in fam
    # the deterministic floor is preserved (boosting from R1 is still there)
    assert "hist_gbm" in fam
    assert "LLM:" in adv.rationale


def test_llm_out_of_vocab_is_dropped(monkeypatch, tmp_path):
    """The LLM proposing a family NOT in the catalog (and a move not in the registry) is DROPPED; the
    verify rejects an all-out-of-vocab payload and the surface falls back to the deterministic floor."""
    payload = {"family_hints": [{"name": "xgboost_external", "weight": 1.0, "reason": "not in catalog"}],
               "feature_moves": [{"name": "magic_transform", "weight": 1.0, "reason": "not a move"}],
               "rationale": "should be dropped"}
    monkeypatch.setattr(ops, "_call_claude", _fake_claude_factory(payload))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    adv = K.suggest_methods(_prof(), {"headroom": 0.1, "below_bar": True}, CLF,
                            use_llm=True, cache_path=str(tmp_path / "c.jsonl"))
    fam = {s.name for s in adv.family_hints}
    assert "xgboost_external" not in fam
    for s in adv.feature_moves:
        assert s.name in K.known_feature_moves()
    # verify rejected the all-out-of-vocab payload -> ops used the deterministic fallback
    assert adv.source == "rules"
    assert "hist_gbm" in fam            # deterministic floor intact


def test_llm_failure_degrades_to_rules(monkeypatch, tmp_path):
    """If the LLM call raises, ops returns the deterministic fallback; advice == the offline floor."""
    def _boom(req, api_key, timeout):
        raise RuntimeError("network down")
    monkeypatch.setattr(ops, "_call_claude", _boom)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    offline = K.suggest_methods(_prof(), {"headroom": 0.1, "below_bar": True}, CLF, use_llm=False)
    online = K.suggest_methods(_prof(), {"headroom": 0.1, "below_bar": True}, CLF,
                               use_llm=True, cache_path=str(tmp_path / "c.jsonl"))
    assert online.source == "rules"
    assert {s.name for s in online.family_hints} == {s.name for s in offline.family_hints}


def test_no_key_degrades_to_rules(monkeypatch, tmp_path):
    """use_llm=True but NO key resolvable -> ops short-circuits to fallback; source stays 'rules'."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # neutralize any on-disk key files the resolver might find, by pointing the resolver at nothing:
    monkeypatch.setattr(ops, "resolve_api_key", lambda explicit=None: None)
    adv = K.suggest_methods(_prof(), {"headroom": 0.1, "below_bar": True}, CLF,
                            use_llm=True, cache_path=str(tmp_path / "c.jsonl"))
    assert adv.source == "rules"


# ================================================================================= web rung (injected)
def test_web_search_adds_note_only():
    """An injected web_search adds a prose note and bumps source, but never changes the vocabulary."""
    calls = {}

    def fake_search(q):
        calls["q"] = q
        return ["gradient boosting is a strong tabular baseline", "use mutual information selection"]

    base = K.suggest_methods(_prof(n_features=300, n_train=400),
                             {"headroom": 0.1, "below_bar": True}, CLF)
    adv = K.suggest_methods(_prof(n_features=300, n_train=400),
                            {"headroom": 0.1, "below_bar": True}, CLF, web_search=fake_search)
    assert "q" in calls
    assert adv.notes and adv.notes[0].startswith("web[")
    assert adv.source.endswith("+web")
    # vocabulary is unchanged vs the no-web call
    assert {s.name for s in adv.family_hints} == {s.name for s in base.family_hints}
    assert {s.name for s in adv.feature_moves} == {s.name for s in base.feature_moves}


def test_web_search_failure_is_swallowed():
    """A throwing web_search must not break the advice (it just yields no note)."""
    def boom(q):
        raise RuntimeError("search 500")
    adv = K.suggest_methods(_prof(), {"headroom": 0.1, "below_bar": True}, CLF, web_search=boom)
    assert adv.source == "rules"
    assert adv.notes == []


# ====================================================================================== frozen guard
def test_frozen_fingerprints_unchanged():
    """The trust core must be byte-identical (this module is new-files-only)."""
    import hashlib

    def sha(path):
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert sha(os.path.join(root, "vectorforge", "science.py")).startswith("b564fba2")
    assert sha(os.path.join(root, "vfplatform", "sealed.py")).startswith("30ad6245")
