"""Integrity tests for the Phase-0 spine.

These assert the properties the audit found broken elsewhere actually hold here. Run with:
    /Users/abdullahalghamdi/jax-env-311/bin/python -m pytest frontier/tests/test_spine.py -q
or standalone:
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_spine.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import certify, sandbox
from frontier.engine import ResearchEngine, EngineConfig
from frontier.program import Program
from frontier.proposers import SeedProposer, MutationProposer, make_code
from frontier.task import Task
from vfplatform.sealed import PeekViolation


def _clf_task():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.85, name="bc")


def test_three_way_split_is_disjoint():
    task = _clf_task()
    s = certify.make_splits(task, seed=0)
    def keyset(rows):
        return {tuple(r["_x"]) for r in rows}
    tr, va, se = keyset(s.train_rows), keyset(s.val_rows), keyset(s.sealed_rows)
    assert tr and va and se, "all three splits must be non-empty"
    assert not (tr & se), "train and sealed must be disjoint"
    assert not (va & se), "val and sealed must be disjoint (sealed is held out from selection)"
    print(f"[ok] split disjoint: {s.meta['counts']}")


def test_sealed_one_peek_enforced():
    """A second uncounted peek on the sealed test must raise PeekViolation."""
    task = _clf_task()
    s = certify.make_splits(task, seed=0)
    st = s.sealed_test
    st._count_peek("first")          # the one allowed peek
    raised = False
    try:
        st._count_peek("second")
    except PeekViolation:
        raised = True
    assert raised, "sealed test must reject a second uncounted peek"
    print("[ok] sealed one-peek enforced (PeekViolation on 2nd)")


def test_sandbox_returns_predictions_not_scores():
    """Numeric firewall: the sandbox hands back predictions; the parent scores."""
    task = _clf_task()
    s = certify.make_splits(task, seed=0)
    code = make_code({"base": "logreg", "scale": True}, "classification")
    prog = Program(code=code, source="seed", label="scale+logreg")
    Xtr = Task.rows_to_X(s.train_rows); ytr = Task.rows_to_y(s.train_rows, task.kind)
    Xva = Task.rows_to_X(s.val_rows)
    res = sandbox.run_program(prog, Xtr, ytr, Xva, kind="classification", wall_seconds=40)
    assert res.ok, f"seed program should run: {res.error_kind} {res.error}"
    assert res.preds is not None and len(res.preds) == len(s.val_rows)
    assert not hasattr(res, "score"), "RunResult must not carry a score (firewall)"
    print(f"[ok] sandbox returned {len(res.preds)} predictions, no score")


def test_sandbox_isolates_crash():
    """A broken candidate is captured as a typed error, not a parent-process crash."""
    task = _clf_task()
    s = certify.make_splits(task, seed=0)
    bad = Program(code="def build_estimator():\n    raise RuntimeError('boom')\n",
                  source="seed", label="broken")
    Xtr = Task.rows_to_X(s.train_rows); ytr = Task.rows_to_y(s.train_rows, task.kind)
    Xva = Task.rows_to_X(s.val_rows)
    res = sandbox.run_program(bad, Xtr, ytr, Xva, kind="classification", wall_seconds=40)
    assert not res.ok and res.error_kind in ("build", "fit", "other")
    print(f"[ok] crash isolated as [{res.error_kind}] {res.error[:60]}")


def test_feature_engineering_is_proposable():
    """The CA-Housing trap fix: feature-eng / target-transform recipes are in the space
    regardless of n_features (here only 10 features)."""
    seeds = SeedProposer().propose({"task_kind": "regression", "tried_labels": set()})
    labels = {p.label for p in seeds}
    assert any("poly2" in l for l in labels), f"expected a polynomial-feature seed, got {labels}"
    assert any("tlog" in l for l in labels), f"expected a target-transform seed, got {labels}"
    print(f"[ok] feature-eng proposable offline: {sorted(labels)}")


def test_mutation_is_generative_offline():
    """Mutation regenerates new, untried recipes from the champion with no LLM."""
    ctx = {"task_kind": "regression", "tried_labels": {"scale+ridge"},
           "best_recipe": {"base": "ridge", "scale": True}, "best_id": "x"}
    muts = MutationProposer().propose(ctx)
    assert muts, "mutation must produce variants of the champion"
    assert all(m.label != "scale+ridge" for m in muts), "must not repeat the champion"
    assert any(m.provenance["recipe"].get("poly") or m.provenance["recipe"].get("base") != "ridge"
               for m in muts), "must explore feature-eng or a base swap"
    print(f"[ok] offline generative mutation: {[m.label for m in muts][:6]}")


def test_end_to_end_certifies_winner_on_sealed():
    """Full loop: select on val, certify the winner on the sealed test, honest outcome."""
    task = _clf_task()
    res = ResearchEngine(EngineConfig(rounds=2, wall_seconds=45, cpu_seconds=40)).run(task)
    assert res.winner is not None, "should find a runnable winner"
    assert res.certificate is not None, "winner must be certified on the sealed test"
    c = res.certificate
    assert "lower_bound" in c and "peeks" in c, "certificate must carry the sealed lower bound + peeks"
    assert c["peeks"] == 1, "exactly one sealed peek for the winner"
    # the certified bound is the LOWER bound, which must not exceed the point estimate
    assert c["lower_bound"] <= c["observed"] + 1e-9
    print(f"[ok] e2e: winner={res.winner.label} val={res.winner_val_score} "
          f"sealed_lb={c['lower_bound']} theta={c['theta']} certified={c['certified']}")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} integrity tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
