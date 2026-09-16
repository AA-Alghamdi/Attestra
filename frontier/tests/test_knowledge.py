"""Behavioral tests for Phase 8 compounding knowledge (frontier/knowledge.py).

Run standalone:
    cd <repo root>
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_knowledge.py

Asserts behavior, not just import:
  - LinUCB reorders a proposal list TOWARD the higher-reward arm after observing outcomes.
  - Warm-start retrieval returns relevant priors for a repeat fingerprint, and [] when cold.
  - The KB is durable: records survive a reload from disk (append-only JSONL).
  - load_from_kb replays history so a fresh bandit inherits the learned ordering.
  - End-to-end on a REAL sklearn dataset (breast cancer): cold cohort builds the KB, then a
    second identical task is warm-started with proven recipes.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier.knowledge import (
    KnowledgeBase, RetrievalProposer, LinUCBRanker, KnowledgeProposer,
    task_descriptor, task_fingerprint, recipe_descriptor, context_vector,
    CONTEXT_DIM, FINGERPRINT_DIM,
)
from frontier.program import Program
from frontier.proposers import make_code, recipe_label
from frontier.task import Task


def _clf_task():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.85, name="bc")


def _prog(recipe, kind="classification", source="seed"):
    return Program(code=make_code(recipe, kind), source=source,
                   label=recipe_label(recipe), provenance={"recipe": dict(recipe)})


def test_descriptor_shapes_are_fixed():
    task = _clf_task()
    fp = task_fingerprint(task)
    assert fp.shape == (FINGERPRINT_DIM,), fp.shape
    td = task_descriptor(task)
    assert td["kind"] == "classification" and td["n_classes"] == 2
    phi = context_vector(td, {"base": "logreg", "scale": True})
    assert phi.shape == (CONTEXT_DIM,), phi.shape
    # recipe one-hot: logreg coordinate set, plus scale flag on, bias on.
    rv = recipe_descriptor({"base": "logreg", "scale": True})
    assert rv[-1] == 1.0 and rv.sum() >= 2.0
    print(f"[ok] shapes fixed: fp={FINGERPRINT_DIM} context={CONTEXT_DIM} td={td}")


def test_linucb_reorders_toward_high_reward_arm():
    """After observing that a 'good' recipe earns high reward and a 'bad' one earns low,
    the bandit must rank the good recipe ahead of the bad one for the same task context."""
    task = _clf_task()
    td = task_descriptor(task)
    ctx = {"task_kind": "classification", "task_descriptor": td,
           "task_fingerprint": task_fingerprint(task)}

    good = _prog({"base": "hist_gbm"})
    bad = _prog({"base": "logreg"})

    ranker = LinUCBRanker(alpha=0.5)
    # Before any data the order is stable (input order preserved on the symmetric prior).
    pre = ranker.rank([bad, good], ctx)
    assert [p.label for p in pre] == [bad.label, good.label], "cold order must be stable/identity"

    # Observe outcomes repeatedly: good=0.97, bad=0.62.
    for _ in range(8):
        ranker.update(ctx, good, reward=0.97)
        ranker.update(ctx, bad, reward=0.62)

    post = ranker.rank([bad, good], ctx)
    assert post[0].label == good.label, (
        f"bandit must rank the high-reward arm first, got {[p.label for p in post]}")
    # And the learned mean reward for 'good' must exceed 'bad'.
    sg, sb = ranker.score(ctx, good), ranker.score(ctx, bad)
    assert sg > sb, f"UCB(good)={sg:.3f} must exceed UCB(bad)={sb:.3f}"
    print(f"[ok] LinUCB reordered toward high-reward arm: UCB good={sg:.3f} > bad={sb:.3f}")


def test_warm_start_retrieval_returns_priors_for_repeat_fingerprint():
    """Record a winning recipe for a fingerprint, then retrieve it for the SAME fingerprint."""
    kb = KnowledgeBase(None)
    task = _clf_task()
    fp = task_fingerprint(task)
    td = task_descriptor(task)

    # cold KB -> no warm start
    rp = RetrievalProposer(kb)
    assert rp.propose({"task_kind": "classification", "task_fingerprint": fp,
                       "tried_labels": set()}) == [], "cold KB must not warm-start"

    # record a strong recipe and a weak (no-gain) recipe
    kb.record(problem_type="classification", fingerprint=fp, task_descriptor=td,
              recipe={"base": "hist_gbm"}, val_gain=0.12, val_score=0.96,
              cost_seconds=2.0, program_id="seed:hist_gbm:abc")
    kb.record(problem_type="classification", fingerprint=fp, task_descriptor=td,
              recipe={"base": "logreg"}, val_gain=0.0, val_score=0.60,
              cost_seconds=1.0, program_id="seed:logreg:def")

    seeds = rp.propose({"task_kind": "classification", "task_fingerprint": fp,
                        "tried_labels": set()})
    labels = [p.label for p in seeds]
    assert "hist_gbm" in labels, f"strong recipe must be retrieved, got {labels}"
    assert "logreg" not in labels, "zero-gain recipe must be filtered out of warm start"
    assert all(p.source == "retrieval" for p in seeds), "warm-start programs are source=retrieval"
    # retrieved programs carry their prior provenance
    assert seeds[0].provenance.get("prior_val_score") == 0.96
    print(f"[ok] warm-start retrieved relevant priors: {labels}")


def test_retrieval_respects_problem_type_and_similarity():
    """A regression recipe must not leak into a classification warm start, and dissimilar
    fingerprints below the threshold are not retrieved."""
    kb = KnowledgeBase(None)
    clf_fp = task_fingerprint(_clf_task())
    # regression record with a regression recipe + a regression-shaped fingerprint
    reg_fp = np.array([0.0, 0.5, 0.3, 0.2, 0.0])
    kb.record(problem_type="regression", fingerprint=reg_fp,
              task_descriptor={"kind": "regression"}, recipe={"base": "ridge", "scale": True},
              val_gain=0.2, val_score=0.8, cost_seconds=1.0, program_id="r")
    rp = RetrievalProposer(kb)
    got = rp.propose({"task_kind": "classification", "task_fingerprint": clf_fp,
                      "tried_labels": set()})
    assert got == [], "regression recipe must not be retrieved for a classification task"
    print("[ok] retrieval respects problem_type + similarity gate")


def test_kb_is_durable_jsonl_roundtrip():
    """Records persist to JSONL and reload identically (append-only durability)."""
    task = _clf_task()
    fp = task_fingerprint(task); td = task_descriptor(task)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "kb.jsonl")
        kb = KnowledgeBase(path)
        kb.record(problem_type="classification", fingerprint=fp, task_descriptor=td,
                  recipe={"base": "hist_gbm"}, val_gain=0.1, val_score=0.95,
                  cost_seconds=2.0, program_id="p1")
        kb.record(problem_type="classification", fingerprint=fp, task_descriptor=td,
                  recipe={"base": "rf"}, val_gain=0.05, val_score=0.90,
                  cost_seconds=3.0, program_id="p2")
        # file has exactly 2 lines, each valid JSON
        with open(path) as fh:
            lines = [l for l in fh if l.strip()]
        assert len(lines) == 2, f"expected 2 atomic lines, got {len(lines)}"
        # reload into a fresh KB
        kb2 = KnowledgeBase(path)
        assert len(kb2) == 2
        recs = kb2.all_records()
        assert {r.recipe_label for r in recs} == {"hist_gbm", "rf"}
        assert recs[0].val_score == 0.95
    print("[ok] KB durable: JSONL append-only roundtrip preserved records")


def test_load_from_kb_warm_starts_bandit():
    """A fresh bandit that replays the KB inherits the learned ordering (cross-run compounding)."""
    kb = KnowledgeBase(None)
    task = _clf_task()
    fp = task_fingerprint(task); td = task_descriptor(task)
    for _ in range(10):
        kb.record(problem_type="classification", fingerprint=fp, task_descriptor=td,
                  recipe={"base": "hist_gbm"}, val_gain=0.1, val_score=0.97,
                  cost_seconds=2.0, program_id="g")
        kb.record(problem_type="classification", fingerprint=fp, task_descriptor=td,
                  recipe={"base": "logreg"}, val_gain=0.0, val_score=0.62,
                  cost_seconds=1.0, program_id="b")
    ranker = LinUCBRanker(alpha=0.5)
    n = ranker.load_from_kb(kb)
    assert n == 20, n
    ctx = {"task_kind": "classification", "task_descriptor": td, "task_fingerprint": fp}
    good = _prog({"base": "hist_gbm"}); bad = _prog({"base": "logreg"})
    order = ranker.rank([bad, good], ctx)
    assert order[0].label == "hist_gbm", "replayed bandit must prefer the historically-good arm"
    print(f"[ok] load_from_kb warm-started bandit from {n} records; prefers proven arm")


def test_knowledge_proposer_end_to_end_on_real_dataset():
    """End-to-end on breast cancer: a cohort of cold runs builds the KB, then a repeat task
    is warm-started and the bandit prefers the recipe that actually scored best."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    task = _clf_task()
    X, y = task.X, task.y
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    with tempfile.TemporaryDirectory() as tmp:
        kp = KnowledgeProposer(KnowledgeBase(os.path.join(tmp, "kb.jsonl")), alpha=0.6)
        ctx = {"task_kind": "classification",
               "task_descriptor": task_descriptor(task),
               "task_fingerprint": task_fingerprint(task),
               "tried_labels": set()}

        # --- cohort: run two recipes for real, record their honest val scores ---
        recipes = [{"base": "hist_gbm"}, {"base": "logreg", "scale": True}]
        scores = {}
        incumbent = None
        for rec in recipes:
            p = _prog(rec)
            if rec["base"] == "hist_gbm":
                est = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
            else:
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
            acc = float((est.predict(Xva) == yva).mean())
            scores[recipe_label(rec)] = acc
            kp.record_outcome(ctx, p, val_score=acc, incumbent_before=incumbent,
                              cost_seconds=1.0, ok=True)
            incumbent = acc if incumbent is None else max(incumbent, acc)

        best_label = max(scores, key=scores.get)
        worst_label = min(scores, key=scores.get)
        assert scores[best_label] > scores[worst_label], scores

        # --- repeat task: warm start should surface proven recipes, bandit prefers the best ---
        warm = kp.propose(ctx)
        warm_labels = [p.label for p in warm]
        assert best_label in warm_labels, (
            f"warm start must surface the proven winner {best_label}; got {warm_labels}")

        ranked = kp.rank([_prog(r) for r in recipes], ctx)
        assert ranked[0].label == best_label, (
            f"bandit must rank the best-scoring recipe first: {best_label} vs "
            f"{[p.label for p in ranked]} (scores={scores})")
        print(f"[ok] e2e real dataset: scores={ {k: round(v,4) for k,v in scores.items()} }; "
              f"warm-start={warm_labels}; bandit-first={ranked[0].label}")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} knowledge tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
