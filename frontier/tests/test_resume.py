"""End-to-end test for frontier.longrun -- WORKING long-horizon resume.

The behavior change this proves: a resumed CoreOrchestrator run SKIPS rounds that a
prior (crashed) attempt already completed, instead of recomputing them from zero
(the audit bug: the orchestrator SAVED a checkpoint but always re-entered the loop
at round 0). Evidence is a DURABLE side-effect counter -- the orchestrator's
per-round `_build_context` (called once per round whose body actually runs, bypassed
by the resume gate's `continue`) -- PLUS the on-disk checkpoint's `completed_rounds`,
`round_lineage`, and `resume_count`. After a crash after round j, the resume runs
ONLY rounds j+1.., yet still certifies the winner EXACTLY ONCE.

Runs on a real sklearn dataset (breast cancer) through the live CoreOrchestrator.
Each candidate executes out-of-process (real sandbox), so to keep wall-time bounded
we cap the admitted batch to ONE arm per round via a per-instance override of
`_admit_with_floor` (this does not touch the resume machinery under test -- it only
reduces how many programs each round executes).

Standalone:
    cd <repo> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_resume.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.datasets import load_breast_cancer

from frontier.core.orchestrator import CoreOrchestrator, CoreConfig
from frontier.longrun import ResumableRun
from frontier.checkpoint import CheckpointStore


class _CrashAfter(Exception):
    """Raised to simulate a process crash AFTER round j has been checkpointed."""


def _data():
    d = load_breast_cancer()
    return d.data, d.target.astype(str)


def _orch(rounds):
    """A CoreOrchestrator whose rounds each execute exactly ONE arm (bounded wall-time)."""
    o = CoreOrchestrator(CoreConfig(rounds=rounds, enable_neural=False,
                                    enable_knowledge=False, wall_seconds=8.0, cpu_seconds=6))
    _orig = o._admit_with_floor
    o._admit_with_floor = lambda ordered, *a, **k: _orig(ordered, *a, **k)[:1]
    return o


def _count_round_bodies(orch, sink):
    """Record which rounds run their BODY (bypassed by the resume gate's `continue`).

    `_build_context` is the SOLE per-round enrichment site; the orchestrator calls it
    once per executed round (and once AFTER the loop with r==cfg.rounds for the KB
    record, which we exclude). A skipped round never reaches it.
    """
    orig = orch._build_context

    def wrapped(task, splits, r, *a, **k):
        if r < orch.cfg.rounds:
            sink.append(r)
        return orig(task, splits, r, *a, **k)

    orch._build_context = wrapped


def test_crash_then_resume_does_not_recompute_finished_rounds():
    """THE core property: crash after round 1; resume must NOT re-run rounds 0,1."""
    X, y = _data()
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "run.json")

        # ---- attempt 1: crash right after round 1's checkpoint is durably written ----
        orch1 = _orch(rounds=3)
        body1: list[int] = []
        _count_round_bodies(orch1, body1)
        run1 = ResumableRun(orch1, ckpt, run_id="attempt-1")

        # Wrap the on-complete hook so the crash fires AFTER round 1 is saved.
        orig_make = run1._make_on_complete

        def make_crashing():
            real = orig_make()

            def crashing(r, best_prog, best_score):
                real(r, best_prog, best_score)        # durable checkpoint FIRST
                if r == 1:
                    raise _CrashAfter("crash after round 1 checkpoint")
            return crashing

        run1._make_on_complete = make_crashing

        crashed = False
        try:
            run1.run(goal="classify tumors", X=X, y=y, theta=0.0)
        except _CrashAfter:
            crashed = True
        assert crashed, "attempt 1 must crash after round 1"
        assert sorted(set(body1)) == [0, 1], f"only rounds 0,1 ran before crash: {body1}"
        assert orch1.sealed_peeks == 0, f"no sealed peek before crash: {orch1.sealed_peeks}"

        # durable checkpoint: rounds 0,1 done, FULL champion persisted for rehydrate.
        st = CheckpointStore(ckpt).load()
        assert st is not None and st.completed_rounds == [0, 1], \
            f"checkpoint must record 0,1 done: {getattr(st, 'completed_rounds', None)}"
        assert st.champion_id is not None
        assert (st.payload or {}).get("champion", {}).get("code"), \
            "full champion Program (code) must be persisted for cross-process rehydrate"

        # ---- attempt 2: resume; rounds 0,1 SKIPPED, only round 2 runs a body ----
        orch2 = _orch(rounds=3)
        body2: list[int] = []
        _count_round_bodies(orch2, body2)
        run2 = ResumableRun(orch2, ckpt, run_id="attempt-2")
        result = run2.run(goal="classify tumors", X=X, y=y, theta=0.0)

        assert run2.resumed is True, "attempt 2 must report a resume"
        assert run2.skipped_rounds == [0, 1], \
            f"finished rounds must be SKIPPED on resume: {run2.skipped_rounds}"
        assert sorted(set(body2)) == [2], \
            f"resume runs ONLY the unfinished round 2 (no recompute of 0,1): {body2}"
        assert run2.executed_rounds == [2], f"only round 2 executed on resume: {run2.executed_rounds}"

        # completed correctly + certified EXACTLY ONCE at the end (sealed-peek discipline).
        assert orch2.sealed_peeks == 1, f"resume certifies exactly once: {orch2.sealed_peeks}"
        if result.certificate is not None:
            assert result.certificate.get("peeks") == 1, "certificate reflects a single peek"
        assert result.winner is not None, "resume produced a winner"

        # final checkpoint: all rounds done + run/seed lineage across the two attempts.
        final = CheckpointStore(ckpt).load()
        assert final.completed_rounds == [0, 1, 2], \
            f"all rounds complete after resume: {final.completed_rounds}"
        assert final.resume_count >= 1, "resume_count bumped"
        assert "attempt-1" in final.lineage, f"run lineage records prior attempt: {final.lineage}"
        rl = (final.payload or {}).get("round_lineage", [])
        by_round = {e["round"]: e["run_id"] for e in rl}
        assert by_round.get(0) == "attempt-1" and by_round.get(1) == "attempt-1", \
            f"rounds 0,1 produced by attempt-1: {by_round}"
        assert by_round.get(2) == "attempt-2", f"round 2 produced by attempt-2: {by_round}"
        # round timestamps are monotone evidence (round 2 was recorded after round 1).
        ts = {e["round"]: e["ts"] for e in rl}
        assert ts[2] >= ts[1] >= ts[0], f"round-completion timestamps monotone: {ts}"
        print(f"[ok] crash@1 then resume: skipped={run2.skipped_rounds} "
              f"executed={run2.executed_rounds} peeks={orch2.sealed_peeks} "
              f"lineage={final.lineage} round_lineage={by_round}")


def test_fully_completed_run_resumes_with_zero_recompute():
    """Resuming an ALREADY-finished run recomputes ZERO rounds and certifies once.

    Reuses the checkpoint produced by the crash+resume test path: we run a 2-round
    job to completion, then resume it -- no round body must execute, yet the carried
    (rehydrated) champion is still certified exactly once.
    """
    X, y = _data()
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "done.json")
        ResumableRun(_orch(rounds=2), ckpt, run_id="first").run(
            goal="classify tumors", X=X, y=y, theta=0.0)
        st = CheckpointStore(ckpt).load()
        assert st.completed_rounds == [0, 1], f"first run completes both rounds: {st}"

        orch2 = _orch(rounds=2)
        body: list[int] = []
        _count_round_bodies(orch2, body)
        run2 = ResumableRun(orch2, ckpt, run_id="second")
        result = run2.run(goal="classify tumors", X=X, y=y, theta=0.0)

        assert run2.resumed is True
        assert run2.executed_rounds == [], f"no round recomputed on full resume: {run2.executed_rounds}"
        assert sorted(set(body)) == [], f"no round body ran on full resume: {body}"
        assert run2.skipped_rounds == [0, 1], f"both rounds skipped: {run2.skipped_rounds}"
        assert orch2.sealed_peeks == 1, "still certifies the carried winner exactly once"
        assert result.winner is not None, "rehydrated champion is the winner"
        print(f"[ok] full resume: zero recompute, winner={result.winner.id}, "
              f"peeks={orch2.sealed_peeks}")


def test_default_orchestrator_unaffected_when_not_wrapped():
    """Sanity: an un-wrapped orchestrator keeps the default seams None (no behavior change)."""
    orch = CoreOrchestrator(CoreConfig(rounds=2))
    assert orch._round_gate is None and orch._on_round_complete is None, \
        "resume seams must default to None so the default path is byte-identical"
    print("[ok] default orchestrator: resume seams are None (no behavior change)")


def _run_all():
    fns = [
        test_default_orchestrator_unaffected_when_not_wrapped,  # fast, no subprocess
        test_crash_then_resume_does_not_recompute_finished_rounds,
        test_fully_completed_run_resumes_with_zero_recompute,
    ]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} resume tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
