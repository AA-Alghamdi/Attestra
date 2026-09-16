"""Anti-orphan test for cross-run knowledge consumption (item: warmstart).

The claim under test is NOT "the KB loads N prior records and logs it". The claim is that
prior outcomes, persisted on a shared KB path, CHANGE what the CoreOrchestrator proposes and
in what order on a SECOND run over the same dataset fingerprint -- i.e. knowledge is CONSUMED,
not printed.

We prove three things, all on a real sklearn dataset:

  1. OUTCOMES ARE RECORDED. A cold run (empty KB on a temp path) appends one durable JSONL
     record per evaluated candidate. After the run, len(KB) > 0 and the file exists on disk.

  2. KB RETRIEVAL CHANGES ROUND-0 PROPOSALS. Reconstructing the orchestrator's exact round-0
     proposal pool (RetrievalProposer + SeedProposer + MutationProposer, deduped, then LinUCB
     ranked) on the warmed KB yields a DIFFERENT proposal set and ordering than the cold pool.
     Concretely, the warm pool contains a recipe-shaped retrieval program (source=="retrieval")
     that is ABSENT from the cold round-0 floor -- a recipe the previous run discovered and the
     cold floor never proposes in round 0. That extra, re-renderable, KB-sourced proposal is the
     observable behavior change. A run that only logged "loaded N records" would leave the round-0
     set identical and FAIL this assertion.

  3. THE BANDIT WARM-STARTS FROM HISTORY. A KnowledgeProposer built on the warmed KB has a
     LinUCBRanker with n_updates > 0 (replayed from the persisted records), whereas one built on
     an empty KB has n_updates == 0. The two rankers therefore order the SAME pool differently.

Integrity: retrieval/ranking are PROPOSAL-side only (never promote); all recorded numbers are the
honest VAL scores the orchestrator already computed. Nothing here touches the sealed test.

Run:
    cd <repo root> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_warmstart_consumed.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

# Repo root on sys.path (mirrors frontier/tests/test_spine.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import certify, knowledge
from frontier.proposers import SeedProposer, MutationProposer
from frontier.task import Task
from frontier.core.orchestrator import CoreOrchestrator, CoreConfig


def _clf_task() -> Task:
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.85, name="bc")


def _round0_pool(kb: knowledge.KnowledgeBase, task: Task, splits) -> list:
    """Reconstruct the orchestrator's exact round-0 proposal pipeline for this KB.

    Mirrors CoreOrchestrator: build the round-0 context (with the knowledge keys), gather
    RetrievalProposer + Seed + Mutation, dedup by program id (engine.py:127-130 rule), then
    LinUCB-rank. Returns the ordered list of (source, label) tuples actually fed to execution.
    """
    know = knowledge.KnowledgeProposer(kb)
    ctx = {
        "task_kind": task.kind,
        "n_features": task.n_features,
        "n_train": len(splits.train_rows),
        "round": 0,
        "tried_labels": set(),
        "best_label": None,
        "best_score": None,
        "best_id": None,
        "best_recipe": None,
        "recent_errors": [],
        "task_descriptor": knowledge.task_descriptor(task),
        "task_fingerprint": knowledge.task_fingerprint(task),
    }
    sources = [know, SeedProposer(), MutationProposer()]
    pool, seen = [], set()
    for src in sources:
        for p in src.propose(ctx) or []:
            if p.id in seen:
                continue
            seen.add(p.id)
            pool.append(p)
    ordered = know.rank(pool, ctx)
    return [(p.source, p.label) for p in ordered], know


def test_outcomes_recorded_and_kb_grows():
    """A cold orchestrator run must persist durable VAL outcomes to the KB path."""
    task = _clf_task()
    tmp = tempfile.mkdtemp()
    kb_path = os.path.join(tmp, "kb.jsonl")
    assert not os.path.exists(kb_path), "KB must start empty (cold)"

    orch = CoreOrchestrator(CoreConfig(rounds=2, wall_seconds=40, cpu_seconds=35,
                                       kb_path=kb_path, enable_neural=False))
    res = orch.run("classify tumors", task.X, task.y, theta=0.85)
    assert res.winner is not None, "cold run should find a runnable winner"

    assert os.path.exists(kb_path), "KB JSONL must exist after a run (outcomes persisted)"
    kb = knowledge.KnowledgeBase(kb_path)
    assert len(kb) > 0, "KB must contain recorded outcomes after a run (not read-only)"
    # every record carries the honest val score the parent computed (never a sealed number)
    for rec in kb.all_records():
        assert 0.0 <= rec.val_score <= 1.0 or rec.val_score == 0.0
    print(f"[ok] outcomes recorded: KB has {len(kb)} durable records at {kb_path}")


def test_warm_run_changes_round0_proposals():
    """COLD vs WARM round-0 proposal set/ordering must differ -- knowledge is CONSUMED."""
    task = _clf_task()
    splits = certify.make_splits(task, seed=0)

    # COLD: empty KB -> retrieval returns [], bandit untrained -> pure Phase-0 floor.
    cold_kb = knowledge.KnowledgeBase(None)
    cold_pool, cold_know = _round0_pool(cold_kb, task, splits)
    assert cold_know.ranker.n_updates == 0, "cold bandit must be untrained"
    assert all(src != "retrieval" for src, _ in cold_pool), \
        "cold round-0 must have no retrieval proposals (empty KB)"

    # Warm the KB by running a full cold orchestrator pass that RECORDS outcomes to disk.
    tmp = tempfile.mkdtemp()
    kb_path = os.path.join(tmp, "kb.jsonl")
    CoreOrchestrator(CoreConfig(rounds=2, wall_seconds=40, cpu_seconds=35,
                                kb_path=kb_path, enable_neural=False)
                     ).run("classify tumors", task.X, task.y, theta=0.85)

    # WARM: a FRESH process-like load of the persisted KB.
    warm_kb = knowledge.KnowledgeBase(kb_path)
    assert len(warm_kb) > 0, "warm KB must have persisted records"
    warm_pool, warm_know = _round0_pool(warm_kb, task, splits)

    # (3) the bandit warm-started from the persisted history.
    assert warm_know.ranker.n_updates > 0, \
        "warm bandit must have replayed prior records (n_updates > 0)"
    assert warm_know.ranker.n_updates >= len(warm_kb), \
        "every persisted record should fold into the bandit's sufficient statistics"

    # (2a) the WARM pool injects a recipe-shaped retrieval program absent from the cold floor.
    retrieval = [(s, l) for s, l in warm_pool if s == "retrieval"]
    assert retrieval, "warm round-0 must inject at least one KB-sourced retrieval proposal"
    cold_labels = {l for _, l in cold_pool}
    new_from_kb = [l for s, l in retrieval if l not in cold_labels]
    assert new_from_kb, (
        "warm-start must surface a proven recipe the cold round-0 floor never proposes; "
        f"cold={sorted(cold_labels)} retrieval={retrieval}")

    # (2b) the retrieval program is re-renderable (recipe-shaped) -> it actually runs.
    rec = warm_kb.retrieve(task.kind, knowledge.task_fingerprint(task), k=4)
    assert rec, "retrieve() must return historical recipes on the identical fingerprint"
    assert any(knowledge._is_recipe(r.recipe) for r in rec), \
        "at least one retrieved record must be recipe-shaped (re-renderable to code)"

    # (2c) the observable change: warm round-0 set/ordering differs from cold.
    assert warm_pool != cold_pool, \
        f"warm round-0 must differ from cold.\n cold={cold_pool}\n warm={warm_pool}"
    warm_labels = {l for _, l in warm_pool}
    assert warm_labels - cold_labels, \
        "warm round-0 proposal SET must be strictly larger (new KB-sourced recipes)"

    print(f"[ok] knowledge CONSUMED: cold round-0 n={len(cold_pool)} -> "
          f"warm n={len(warm_pool)}; new-from-KB={new_from_kb}; "
          f"bandit n_updates {cold_know.ranker.n_updates}->{warm_know.ranker.n_updates}")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} warmstart-consumption tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
