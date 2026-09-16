"""Tests for the Phase-4 agentic coding loop (frontier/agentic.py).

These assert BEHAVIOR, not just import: a deliberately broken Program (fixable bug) is
repaired by the DETERMINISTIC path and then runs + scores through the real sandbox; the LLM
path is exercised with a stub client; AgenticProposer drops into the spine and honestly
declines when a bug is unrepairable.

Run standalone:
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_agentic.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import certify, sandbox
from frontier.agentic import (AgenticProposer, CodeRepairer, RepairResult, repair_loop,
                              attach_probe_to_context)
from frontier.engine import ResearchEngine, EngineConfig
from frontier.program import Program
from frontier.proposers import MutationProposer, make_code
from frontier.task import Task


def _clf_task():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.85, name="bc")


def _arrays(task):
    s = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(s.train_rows); ytr = Task.rows_to_y(s.train_rows, task.kind)
    Xev = Task.rows_to_X(s.val_rows)
    return s, Xtr, ytr, Xev


# --------------------------------------------------------------------- deterministic unit fixes

def test_deterministic_fix_name_error():
    """A pipeline that references RandomForestClassifier without importing it -> NameError;
    the deterministic repairer adds the missing import."""
    rep = CodeRepairer(llm_client=None)
    code = ("def build_estimator():\n"
            "    return RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=1)\n")
    fix = rep.fix(code, "fit", "name 'RandomForestClassifier' is not defined", [], {})
    assert fix is not None, "should produce a fix for a NameError"
    new_code, src, note = fix
    assert src == "deterministic"
    assert "from sklearn.ensemble import RandomForestClassifier" in new_code
    print(f"[ok] NameError fixed via {note}")


def test_deterministic_fix_unsupported_kwarg():
    """An estimator built with a bogus kwarg -> 'unexpected keyword argument'; the repairer
    strips exactly that kwarg and leaves the rest intact."""
    rep = CodeRepairer(llm_client=None)
    code = ("from sklearn.linear_model import LogisticRegression\n"
            "def build_estimator():\n"
            "    return LogisticRegression(max_iter=2000, frobnicate=7)\n")
    fix = rep.fix(code, "fit", "__init__() got an unexpected keyword argument 'frobnicate'", [], {})
    assert fix is not None, "should strip the unsupported kwarg"
    new_code, _, _ = fix
    assert "frobnicate" not in new_code, "the bad kwarg must be removed"
    assert "max_iter=2000" in new_code, "the good kwarg must survive"
    print("[ok] unsupported kwarg stripped, good kwargs preserved")


def test_deterministic_fix_missing_build_estimator():
    """No build_estimator() defined -> the repairer synthesizes one."""
    rep = CodeRepairer(llm_client=None)
    code = "x = 1\n"
    fix = rep.fix(code, "build", "no callable build_estimator()", [], {})
    assert fix is not None
    new_code, _, _ = fix
    assert "def build_estimator" in new_code
    print("[ok] missing build_estimator synthesized")


def test_repairer_gives_up_when_no_rule_applies():
    """Honest: when no deterministic rule matches, fix() returns None (loop stops, no fabrication)."""
    rep = CodeRepairer(llm_client=None)
    fix = rep.fix("def build_estimator():\n    return object()\n",
                  "fit", "some totally novel runtime error with no known remedy", [], {})
    assert fix is None
    print("[ok] repairer declines unknown error (returns None)")


# --------------------------------------------------------------------- full loop through sandbox

def test_repair_loop_fixes_and_runs_broken_program():
    """End-to-end deterministic repair: a Program with a NameError bug is repaired by the loop
    and then RUNS + produces well-shaped predictions through the REAL sandbox, which the parent
    can then score (the firewall: loop returns predictions, parent scores)."""
    task = _clf_task()
    s, Xtr, ytr, Xev = _arrays(task)
    # fixable bug: RandomForestClassifier used but not imported -> sandbox 'fit' NameError.
    broken = Program(
        code="def build_estimator():\n    return RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=1)\n",
        source="seed", label="broken_rf")
    # sanity: confirm it really fails first
    pre = sandbox.run_program(broken, Xtr, ytr, Xev, kind="classification", wall_seconds=40)
    assert not pre.ok and pre.error_kind in ("fit", "build"), f"expected failure, got {pre}"

    rr = repair_loop(broken, Xtr, ytr, Xev, kind="classification", llm_client=None, k=3,
                     wall_seconds=45, cpu_seconds=40, initial=pre)
    assert isinstance(rr, RepairResult)
    assert rr.repaired and rr.ok, f"loop should repair the program; steps={rr.steps}"
    assert rr.program.id != broken.id, "repaired program must differ from the original"
    assert rr.program.provenance.get("repair"), "repair provenance must be recorded"
    # firewall: the loop hands back predictions; the PARENT scores them.
    assert rr.run.preds is not None and len(rr.run.preds) == len(Xev)
    score = certify.score_val(task, s.val_rows, rr.run.preds)
    assert 0.0 <= score <= 1.0
    print(f"[ok] broken program repaired+ran via {[st.note for st in rr.steps]}; val={score:.4f}")


def test_repair_loop_passthrough_when_already_ok():
    """A working Program returns repaired=False with the original code (no needless rewrite)."""
    task = _clf_task()
    s, Xtr, ytr, Xev = _arrays(task)
    good = Program(code=make_code({"base": "logreg", "scale": True}, "classification"),
                   source="seed", label="ok_logreg")
    rr = repair_loop(good, Xtr, ytr, Xev, kind="classification", llm_client=None, k=2,
                     wall_seconds=40, cpu_seconds=35)
    assert rr.ok and not rr.repaired
    assert rr.program.id == good.id
    print("[ok] already-working program passes through unchanged")


def test_repair_loop_honest_failure_when_unrepairable():
    """A bug no rule can fix stays failing; the loop returns ok=False (honest, not fabricated)."""
    task = _clf_task()
    s, Xtr, ytr, Xev = _arrays(task)
    bad = Program(code="def build_estimator():\n    raise ValueError('intentional unrepairable')\n",
                  source="seed", label="hopeless")
    rr = repair_loop(bad, Xtr, ytr, Xev, kind="classification", llm_client=None, k=2,
                     wall_seconds=40, cpu_seconds=35)
    assert not rr.ok and not rr.repaired
    assert rr.run.preds is None
    print(f"[ok] unrepairable program -> honest failure [{rr.run.error_kind}]")


# --------------------------------------------------------------------- LLM path (stub client)

def test_repair_loop_uses_llm_client_when_present():
    """With a client wired, the repairer asks it for a fix. A stub that returns valid code
    repairs the program through the LLM path (source tagged 'llm')."""
    task = _clf_task()
    s, Xtr, ytr, Xev = _arrays(task)
    fixed_code = make_code({"base": "logreg", "scale": True}, "classification")

    calls = {"n": 0}
    def stub_client(prompt: str) -> str:
        calls["n"] += 1
        assert "build_estimator" in prompt and "error" in prompt.lower()
        return "```python\n" + fixed_code + "\n```"   # also tests fence stripping

    broken = Program(code="def build_estimator():\n    return Glorp()\n",
                     source="llm", label="broken_llm")
    rr = repair_loop(broken, Xtr, ytr, Xev, kind="classification", llm_client=stub_client, k=2,
                     wall_seconds=45, cpu_seconds=40)
    assert rr.repaired and rr.ok, f"LLM stub should repair; steps={rr.steps}"
    assert calls["n"] >= 1, "the llm client must have been called"
    assert any(st.source == "llm" for st in rr.steps), "the successful step should be tagged llm"
    print(f"[ok] LLM repair path used ({calls['n']} call(s))")


def test_llm_falls_back_to_deterministic_on_garbage():
    """If the client returns junk (no build_estimator), fix() degrades to the deterministic
    rule rather than fabricating -- honest degradation."""
    rep = CodeRepairer(llm_client=lambda p: "I cannot help with that.")
    code = ("def build_estimator():\n    return RandomForestClassifier()\n")
    fix = rep.fix(code, "fit", "name 'RandomForestClassifier' is not defined", [], {})
    assert fix is not None and fix[1] == "deterministic"
    print("[ok] garbage LLM output degrades to deterministic fix")


# --------------------------------------------------------------------- proposer integration

def test_agentic_proposer_returns_running_program():
    """AgenticProposer.propose authors (deterministic fallback, no LLM) and returns a program
    that already runs -- on a synthetic probe with NO real arrays attached."""
    ap = AgenticProposer(llm_client=None, n_author=1, k=2, wall_seconds=45, cpu_seconds=40)
    ctx = {"task_kind": "classification", "n_features": 10, "n_train": 200,
           "round": 0, "tried_labels": set(), "recent_errors": []}
    progs = ap.propose(ctx)
    assert progs, "agentic proposer must return at least one program"
    assert all(p.source == "agentic" for p in progs)
    print(f"[ok] AgenticProposer authored: {[p.label for p in progs]}")


def test_agentic_proposer_wraps_base_and_repairs():
    """As a wrapper around MutationProposer, AgenticProposer repairs whatever the base emits.
    We feed a champion recipe so mutation produces variants, then confirm they come back."""
    ap = AgenticProposer(base=MutationProposer(), llm_client=None, k=1,
                         wall_seconds=40, cpu_seconds=35)
    ctx = {"task_kind": "regression", "n_features": 8, "n_train": 150, "round": 1,
           "tried_labels": {"scale+ridge"}, "best_recipe": {"base": "ridge", "scale": True},
           "best_id": "x", "recent_errors": []}
    progs = ap.propose(ctx)
    assert progs, "wrapping a base proposer must yield its (repaired) variants"
    print(f"[ok] AgenticProposer wrapped MutationProposer -> {len(progs)} variants")


def test_attach_probe_does_not_mutate_context():
    """attach_probe_to_context returns a new dict with real arrays for the loop; original
    context is untouched (additive integration, no engine edit needed)."""
    ctx = {"task_kind": "classification", "n_features": 4}
    X = np.zeros((10, 4)); y = np.zeros(10).astype(str); Xp = np.zeros((3, 4))
    new = attach_probe_to_context(ctx, X, y, Xp, "classification")
    assert "X_train" in new and "X_train" not in ctx, "must not mutate caller context"
    assert new["X_probe"].shape == (3, 4)
    print("[ok] attach_probe_to_context is non-mutating")


def test_end_to_end_engine_with_agentic_proposer():
    """The integration the WIRING block documents: AgenticProposer plugged into ResearchEngine
    alongside the seeds runs the full loop and certifies a winner on the sealed test."""
    task = _clf_task()
    cfg = EngineConfig(rounds=1, wall_seconds=45, cpu_seconds=40, llm_client=None)
    engine = ResearchEngine(cfg, proposers=[
        AgenticProposer(llm_client=None, n_author=1, k=2, wall_seconds=45, cpu_seconds=40),
    ])
    res = engine.run(task)
    assert res.winner is not None, "agentic-only engine should find a runnable winner"
    assert res.certificate is not None and res.certificate["peeks"] == 1
    assert res.certificate["lower_bound"] <= res.certificate["observed"] + 1e-9
    print(f"[ok] e2e agentic engine: winner={res.winner.label} "
          f"certified={res.certificate['certified']} lb={res.certificate['lower_bound']}")


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
    print(f"\n{len(fns) - failed}/{len(fns)} agentic tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
