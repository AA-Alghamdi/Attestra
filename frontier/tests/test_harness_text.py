"""Integrity tests for the Phase-3 text harness (frontier/harness/text.py).

These assert BEHAVIOR, not just import:
  - the synthetic corpus is separable and well-formed (no network);
  - adapt() turns raw documents into a valid dense float Task via bounded-vocab tfidf;
  - the baseline suite is text-appropriate (linear + Naive Bayes) and they are seeds;
  - the harness SELF-CERTIFIES through the real Phase-0 frozen sealed gate (the trust gate);
  - a small synthetic corpus certifies end to end through the Phase-0 ResearchEngine via adapt();
  - the harness is registered for the router under the text task-type keys;
  - adapt() rejects degenerate input (single class, length mismatch, empty corpus).

Run standalone:
    cd <repo> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_harness_text.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier.engine import EngineConfig, ResearchEngine
from frontier.harness import lookup
from frontier.harness import text as text_harness
from frontier.harness.base import Harness
from frontier.program import Program
from frontier.task import Task


def test_synth_corpus_is_wellformed_and_separable():
    docs, labels = text_harness.synth_text_corpus(seed=0)
    assert len(docs) == len(labels) == 360, "3 classes x 120 docs"
    assert set(labels) == {"c0", "c1", "c2"}, f"unexpected labels {set(labels)}"
    assert all(isinstance(d, str) and d for d in docs), "documents must be non-empty strings"
    # Separability sanity: a plain linear classifier on tfidf should fit the train split well.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    Xs = TfidfVectorizer().fit_transform(docs)
    clf = LogisticRegression(max_iter=2000).fit(Xs, labels)
    acc = float((clf.predict(Xs) == np.asarray(labels)).mean())
    assert acc > 0.95, f"synthetic corpus should be highly separable, got train acc {acc:.3f}"
    print(f"[ok] synth corpus well-formed, 360 docs/3 classes, train acc {acc:.3f}")


def test_adapt_builds_valid_dense_float_task():
    docs, labels = text_harness.synth_text_corpus(seed=1)
    h = text_harness.TextHarness(max_features=200, ngram_max=1, min_df=2)
    task = h.adapt(docs, labels, kind="classification", theta=0.80, metric="macro_f1", name="t")
    assert isinstance(task, Task)
    assert task.kind == "classification" and task.metric == "macro_f1"
    assert task.X.dtype == float and task.X.ndim == 2
    assert task.X.shape[0] == len(docs)
    assert 0 < task.X.shape[1] <= 200, f"bounded vocab violated: {task.X.shape[1]}"
    assert np.all(task.X >= 0.0), "tfidf features must be non-negative (NB precondition)"
    assert sorted(task.labels) == ["c0", "c1", "c2"]
    print(f"[ok] adapt -> dense float Task X{task.X.shape}, metric={task.metric}")


def test_baseline_suite_is_linear_plus_nb_and_seeds():
    h = text_harness.TextHarness()
    progs = h.baseline_suite("classification")
    labels = {p.label for p in progs}
    assert "text_logreg" in labels and "text_linsvc" in labels, "expected linear baselines"
    assert "text_multinomial_nb" in labels and "text_complement_nb" in labels, "expected NB baselines"
    assert all(p.source == "seed" for p in progs), "baselines must be seeds/fallbacks, not promoters"
    assert all("build_estimator" in p.code for p in progs), "each baseline must define build_estimator"
    print(f"[ok] baseline suite (seeds): {sorted(labels)}")


def test_harness_self_certifies_through_frozen_gate():
    """The Phase-3 trust gate: the harness must certify itself on its known-good corpus."""
    h = text_harness.TextHarness()
    assert not h.trusted, "harness must start untrusted"
    ok, cert = h.self_test(rounds=1, seed=0, wall_seconds=60, cpu_seconds=55)
    assert ok, f"text harness failed self-test: {cert.detail} (sealed={cert.sealed_cert})"
    assert h.trusted, "trusted flag must flip to True after a passing self-test"
    assert cert.task_kind == "classification" and cert.metric == "macro_f1"
    assert cert.sealed_cert is not None and cert.sealed_cert.get("peeks") == 1, \
        "exactly one counted sealed peek for the winner"
    lb = cert.sealed_cert["lower_bound"]
    assert lb > cert.self_theta, f"sealed lower bound {lb} must clear self_theta {cert.self_theta}"
    print(f"[ok] self-cert: winner={cert.winner_label} sealed_lb={lb} "
          f"theta={cert.self_theta} dataset={cert.dataset}")


def test_small_corpus_certifies_end_to_end_via_engine():
    """A user-built small corpus certifies through the SAME Phase-0 engine via adapt()."""
    docs, labels = text_harness.synth_text_corpus(n_per_class=80, n_classes=2, seed=3)
    h = text_harness.TextHarness(max_features=300, ngram_max=1, min_df=2)
    task = h.adapt(docs, labels, kind="classification", theta=0.75, metric="macro_f1",
                   name="small_text")
    # Seed the engine with the harness's text baselines (router would pass these in).
    from frontier.harness.base import _StaticSeedProposer
    from frontier.proposers import SeedProposer, MutationProposer
    proposers = [_StaticSeedProposer(h.baseline_suite("classification")),
                 SeedProposer(), MutationProposer()]
    res = ResearchEngine(EngineConfig(rounds=1, wall_seconds=60, cpu_seconds=55, llm_client=None),
                         proposers=proposers).run(task)
    assert res.winner is not None, "should find a runnable winner"
    assert res.certificate is not None and res.certificate["peeks"] == 1
    assert res.certificate["lower_bound"] <= res.certificate["observed"] + 1e-9
    assert res.certified, (f"separable 2-class text at theta=0.75 should certify; "
                           f"got lb={res.certificate['lower_bound']} decline={res.decline_reason}")
    print(f"[ok] e2e certify: winner={res.winner.label} "
          f"sealed_lb={res.certificate['lower_bound']} certified={res.certified}")


def test_registered_for_router():
    h = lookup("text")
    assert isinstance(h, Harness) and h.key == "text"
    assert lookup("text_classification") is h, "alias must resolve to the same instance"
    print("[ok] text harness registered under 'text' and 'text_classification'")


def test_adapt_rejects_degenerate_input():
    h = text_harness.TextHarness()
    # single class
    raised = 0
    for X, y, why in [
        (["a b c", "d e f"], ["c0", "c0"], "single class"),
        (["a b c"], ["c0", "c1"], "length mismatch"),
        ([], [], "empty corpus"),
    ]:
        try:
            h.adapt(X, y, kind="classification", theta=0.5)
        except ValueError:
            raised += 1
        else:
            raise AssertionError(f"adapt should reject {why}")
    # wrong kind
    try:
        h.adapt(["a b", "c d"], ["c0", "c1"], kind="regression", theta=0.5)
    except ValueError:
        raised += 1
    else:
        raise AssertionError("adapt should reject kind='regression'")
    assert raised == 4
    print("[ok] adapt rejects single-class / length-mismatch / empty / wrong-kind")


def test_baselines_dont_carry_a_score():
    """Firewall sanity at the harness layer: baseline Programs are code, not scores."""
    for p in text_harness.text_baseline_programs():
        assert isinstance(p, Program)
        assert not hasattr(p, "score") and not hasattr(p, "metric")
        assert "predict" not in p.code.lower() or "build_estimator" in p.code
    print("[ok] baselines are code-only Programs, no embedded scores")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} text-harness tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
