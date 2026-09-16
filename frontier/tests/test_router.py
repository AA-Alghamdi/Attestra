"""Behavioral tests for the Phase-3 problem router (frontier/harness/router.py).

Run standalone:
    cd <repo root> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_router.py
or with pytest:
    /Users/abdullahalghamdi/jax-env-311/bin/python -m pytest frontier/tests/test_router.py -q

These assert the two acceptance properties from the spec plus the deterministic/LLM contract:
  - deterministic routing picks TabularHarness for a numeric matrix;
  - the adversarial check flags an obviously infeasible goal (block severity);
and the safety/degradation invariants (honest fallback when no LLM, LLM verdict overrides typing
but never drops a deterministic block, regression vs classification typing from data alone).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier.harness import router
from frontier.harness.router import route, type_problem, ProblemSpec, Risk


def _clf_data():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return d.data, d.target.astype(str)


def _reg_data():
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    return d.data, d.target.astype(float)


def test_deterministic_routes_numeric_matrix_to_tabular():
    """ACCEPTANCE 1: a numeric (n,d) matrix routes to TabularHarness with no LLM."""
    X, y = _clf_data()
    h = route("predict whether a tumor is malignant or benign", X, y, llm_client=None)
    assert type(h).__name__ == "TabularHarness", f"expected TabularHarness, got {type(h).__name__}"
    assert h.spec.modality == "tabular", f"modality should be tabular, got {h.spec.modality}"
    assert h.spec.kind == "classification"
    assert h.spec.typed_by == "deterministic", "no client => deterministic typing (honest)"
    print(f"[ok] numeric matrix -> {type(h).__name__} ({h.spec.kind}/{h.spec.modality})")


def test_adversarial_flags_infeasible_goal():
    """ACCEPTANCE 2: an obviously infeasible goal is flagged at block severity."""
    X, y = _clf_data()
    spec = type_problem("guarantee 100% accuracy with zero error on every single prediction", X, y)
    assert spec.blocked, "infeasible perfect-accuracy demand must block"
    codes = {r.code for r in spec.risks if r.severity == "block"}
    assert "infeasible_goal" in codes, f"expected infeasible_goal block, got {codes}"
    print(f"[ok] infeasible goal blocked: {[str(r) for r in spec.risks if r.severity=='block']}")


def test_regression_typed_from_continuous_targets():
    """Continuous targets with high cardinality type as regression, metric r2."""
    X, y = _reg_data()
    spec = type_problem("estimate disease progression one year out", X, y)
    assert spec.kind == "regression", f"expected regression, got {spec.kind}"
    assert spec.metric == "r2"
    assert spec.modality == "tabular"
    print(f"[ok] continuous targets -> {spec.kind} metric={spec.metric}")


def test_imbalanced_classification_suggests_macro_f1():
    """A skewed binary problem should suggest macro_f1 over raw accuracy."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    y = np.array(["pos"] * 30 + ["neg"] * 370)            # ~8% positive => imbalanced
    spec = type_problem("flag the rare fraudulent transactions", X, y)
    assert spec.kind == "classification"
    assert spec.metric == "macro_f1", f"imbalanced should suggest macro_f1, got {spec.metric}"
    print(f"[ok] imbalanced classification -> metric={spec.metric}")


def test_text_modality_from_string_array():
    """A 1-D array of strings types as the text modality (deterministic data signal)."""
    docs = np.array([f"this is review number {i} and it was good" for i in range(50)], dtype=object)
    y = np.array((["pos", "neg"] * 25))
    spec = type_problem("classify the sentiment of each review", docs, y)
    assert spec.modality == "text", f"string array should be text, got {spec.modality}"
    print(f"[ok] string array -> modality={spec.modality}")


def test_degenerate_single_class_blocks():
    """Classification with a single class has no decision to certify -> block."""
    X = np.random.RandomState(0).normal(size=(60, 5))
    y = np.array(["only"] * 60)
    spec = type_problem("classify these", X, y)
    assert spec.blocked, "single-class classification must block"
    assert any(r.code == "degenerate_labels" for r in spec.risks)
    print("[ok] single-class classification blocked")


def test_no_llm_is_honest_fallback():
    """Without a client, typed_by is 'deterministic' and confidence is data-driven (not faked)."""
    X, y = _clf_data()
    spec = type_problem("anything", X, y, llm_client=None)
    assert spec.typed_by == "deterministic"
    assert 0.0 <= spec.confidence <= 1.0
    print(f"[ok] honest fallback: typed_by={spec.typed_by} conf={spec.confidence}")


def test_llm_verdict_overrides_typing():
    """A wired LLM client's valid JSON verdict overrides the deterministic typing."""
    def fake_client(prompt: str) -> str:
        # The deterministic floor would call this regression (continuous), but the LLM (which
        # in a real run might see e.g. that the target is binned ordinal classes) overrides it.
        return ('{"kind": "classification", "modality": "tabular", "metric": "macro_f1", '
                '"theta_hint": 0.7, "confidence": 0.9, "rationale": "ordinal bins", "risks": []}')
    X, y = _reg_data()
    spec = type_problem("classify into severity buckets", X, y, llm_client=fake_client)
    assert spec.typed_by == "llm"
    assert spec.kind == "classification", "LLM verdict should override the deterministic kind"
    assert spec.metric == "macro_f1"
    assert abs(spec.theta_hint - 0.7) < 1e-9
    print(f"[ok] LLM override: kind={spec.kind} metric={spec.metric} theta_hint={spec.theta_hint}")


def test_llm_cannot_drop_deterministic_block():
    """Safety ratchet: even if the LLM says 'all clear', a deterministic block stays."""
    def all_clear_client(prompt: str) -> str:
        return ('{"kind": "classification", "modality": "tabular", "metric": "accuracy", '
                '"theta_hint": 0.6, "confidence": 0.9, "rationale": "fine", "risks": []}')
    X, y = _clf_data()
    spec = type_problem("achieve perfect accuracy with zero error always", X, y,
                        llm_client=all_clear_client)
    assert spec.blocked, "deterministic infeasibility block must survive an LLM all-clear"
    print("[ok] LLM all-clear cannot drop the deterministic infeasibility block")


def test_llm_failure_degrades_to_deterministic():
    """A client that raises or returns junk must NOT crash routing; we fall back honestly."""
    def broken_client(prompt: str) -> str:
        raise RuntimeError("model unavailable")
    def junk_client(prompt: str) -> str:
        return "I think this is probably classification, hard to say."
    X, y = _clf_data()
    s1 = type_problem("classify tumors", X, y, llm_client=broken_client)
    s2 = type_problem("classify tumors", X, y, llm_client=junk_client)
    assert s1.typed_by == "deterministic" and s1.kind == "classification"
    assert s2.typed_by == "deterministic" and s2.kind == "classification"
    print("[ok] LLM failure/junk degrades to deterministic typing")


def test_reward_hacking_leak_request_blocks():
    """An explicit request to peek at / memorize the sealed answers is a block-level hack risk."""
    X, y = _clf_data()
    spec = type_problem("memorize the test labels and peek at the sealed answers to win", X, y)
    assert spec.blocked, "explicit leak/memorize request must block"
    assert any(r.code == "reward_hacking" and r.severity == "block" for r in spec.risks)
    print("[ok] leak/memorize request blocked as reward-hacking")


def test_route_returns_object_with_spec_and_adapt():
    """The routed harness carries .spec and is adaptable (real harness) end-to-end through engine."""
    X, y = _clf_data()
    h = route("classify tumors", X, y)
    assert isinstance(h.spec, ProblemSpec)
    # If the real registry harness was selected, it can adapt into a runnable Task. (If a
    # fallback was returned because the registry was unavailable, adapt() raises a clear error;
    # the routing-by-name assertion above is the load-bearing acceptance check either way.)
    if type(h).__name__ == "TabularHarness":
        task = h.adapt(X, y, kind=h.spec.kind, theta=0.85, metric=h.spec.metric)
        assert task.kind == "classification"
        assert task.n_features == X.shape[1]
        print(f"[ok] routed TabularHarness.adapt -> Task(kind={task.kind}, n_features={task.n_features})")
    else:
        print(f"[ok] routed fallback {type(h).__name__} (registry unavailable); spec carried")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} router tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
