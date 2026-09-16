"""Tests for P6 checkpoint-resume (frontier/checkpoint.py).

The headline property -- the bug the PR19 resume got wrong -- is asserted with a
per-round counter side-effect: after a simulated crash at round k, a resume must
NOT re-run rounds 0..k, must finish the remaining rounds, and must produce the
same final state as an uninterrupted run.

Also covered: atomic-durable save survives a torn-write simulation (a half-written
temp does not clobber the good file), lineage records parent ids across resumes,
and a real sklearn round body threads a champion through checkpoints.

Run standalone:
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_checkpoint.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier.checkpoint import (
    CheckpointStore,
    CheckpointedRun,
    RoundOutcome,
    RunState,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _CrashAt(Exception):
    """Raised by a round_fn to simulate a process crash mid-experiment."""


def _make_round_fn(call_counter, crash_at=None):
    """Build a round_fn that records every round index it actually executes.

    `call_counter` is a dict {round_index: times_called}. If `crash_at` is set, the
    function raises _CrashAt the FIRST time it reaches that round, simulating a
    crash *before* that round completes.
    """
    crashed = {"done": False}

    def round_fn(r, state):
        if crash_at is not None and r == crash_at and not crashed["done"]:
            crashed["done"] = True
            raise _CrashAt(f"simulated crash entering round {r}")
        call_counter[r] = call_counter.get(r, 0) + 1
        # Champion improves monotonically with round so we can assert carry-over.
        score = float(r)
        payload = dict(state.payload)
        payload["last_round"] = r
        return RoundOutcome(
            champion_id=f"prog-r{r}",
            champion_score=score,
            records=[{"round": r, "score": score}],
            payload=payload,
        )

    return round_fn


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_uninterrupted_run_runs_each_round_once():
    with tempfile.TemporaryDirectory() as d:
        store = CheckpointStore(os.path.join(d, "run.json"))
        counter = {}
        runner = CheckpointedRun(store, n_rounds=5, round_fn=_make_round_fn(counter),
                                 run_id="run-A")
        state = runner.run()
        assert state.all_rounds_done, "all rounds should complete"
        assert state.completed_rounds == [0, 1, 2, 3, 4]
        assert all(counter[r] == 1 for r in range(5)), f"each round exactly once: {counter}"
        assert state.champion_id == "prog-r4" and state.champion_score == 4.0
        print(f"[ok] uninterrupted: each round ran once, champion={state.champion_id}")


def test_crash_then_resume_does_not_recompute_finished_rounds():
    """THE core property: rounds 0..k are not recomputed after a crash at k+1."""
    with tempfile.TemporaryDirectory() as d:
        store = CheckpointStore(os.path.join(d, "run.json"))
        counter = {}

        # --- attempt 1: crash entering round 3 (so 0,1,2 complete & checkpoint) ---
        run1 = CheckpointedRun(store, n_rounds=6,
                               round_fn=_make_round_fn(counter, crash_at=3),
                               run_id="run-1")
        raised = False
        try:
            run1.run()
        except _CrashAt:
            raised = True
        assert raised, "attempt 1 must crash at round 3"
        assert counter == {0: 1, 1: 1, 2: 1}, f"only 0,1,2 ran before crash: {counter}"

        # the checkpoint on disk must show 0,1,2 done and champion at round 2
        persisted = store.load()
        assert persisted is not None and persisted.completed_rounds == [0, 1, 2]
        assert persisted.champion_id == "prog-r2"

        # --- attempt 2: resume; rounds 0,1,2 must be SKIPPED (counter unchanged) ---
        run2 = CheckpointedRun(store, n_rounds=6,
                               round_fn=_make_round_fn(counter),  # no crash now
                               run_id="run-2")
        state = run2.run()
        assert run2.resumed, "attempt 2 must report it resumed prior state"
        assert state.all_rounds_done, "resume must finish all rounds"
        # rounds 0,1,2 still ran exactly ONCE (from attempt 1); 3,4,5 ran once now.
        assert counter == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, \
            f"finished rounds must NOT recompute: {counter}"
        assert state.champion_id == "prog-r5" and state.champion_score == 5.0
        print(f"[ok] crash@3 then resume: counter={counter} (0,1,2 not recomputed)")


def test_resume_matches_uninterrupted_final_state():
    """A crash+resume run must reach the same final champion as a clean run."""
    with tempfile.TemporaryDirectory() as d:
        # clean reference run
        ref_store = CheckpointStore(os.path.join(d, "ref.json"))
        ref_state = CheckpointedRun(ref_store, 4, _make_round_fn({}), "ref").run()

        # interrupted run, then resumed
        store = CheckpointStore(os.path.join(d, "run.json"))
        try:
            CheckpointedRun(store, 4, _make_round_fn({}, crash_at=2), "r1").run()
        except _CrashAt:
            pass
        resumed = CheckpointedRun(store, 4, _make_round_fn({}), "r2").run()

        assert resumed.champion_id == ref_state.champion_id
        assert resumed.champion_score == ref_state.champion_score
        assert resumed.completed_rounds == ref_state.completed_rounds
        # the resumed run's payload should reflect the LAST round, like the clean run
        assert resumed.payload.get("last_round") == ref_state.payload.get("last_round") == 3
        print("[ok] resume reaches identical final state to a clean run")


def test_lineage_records_parent_ids_across_resumes():
    with tempfile.TemporaryDirectory() as d:
        store = CheckpointStore(os.path.join(d, "run.json"))
        # attempt 1 crashes at 1
        try:
            CheckpointedRun(store, 4, _make_round_fn({}, crash_at=1), "gen0").run()
        except _CrashAt:
            pass
        # attempt 2 crashes at 2
        try:
            CheckpointedRun(store, 4, _make_round_fn({}, crash_at=2), "gen1").run()
        except _CrashAt:
            pass
        # attempt 3 finishes
        final = CheckpointedRun(store, 4, _make_round_fn({}), "gen2").run()

        assert final.run_id == "gen2"
        assert final.parent_run_id == "gen1", "immediate parent recorded"
        # full ancestry oldest-first
        assert final.lineage == ["gen0", "gen1"], f"lineage chain: {final.lineage}"
        assert final.resume_count == 2, "two resumes happened"
        print(f"[ok] lineage: run={final.run_id} parent={final.parent_run_id} "
              f"lineage={final.lineage} resumes={final.resume_count}")


def test_atomic_save_is_durable_and_loadable():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.json")
        store = CheckpointStore(path)
        st = RunState(run_id="x", n_rounds=3, completed_rounds=[0, 1],
                      champion_id="c", champion_score=0.9, payload={"k": "v"})
        store.save(st)
        assert os.path.isfile(path), "checkpoint file must exist after save"
        # no orphan temp files left behind
        leftovers = [f for f in os.listdir(d) if f.startswith(".ckpt-")]
        assert not leftovers, f"no temp files should linger: {leftovers}"
        back = store.load()
        assert back.run_id == "x" and back.completed_rounds == [0, 1]
        assert back.champion_id == "c" and back.payload == {"k": "v"}
        print("[ok] atomic save: file present, no temp leftovers, round-trips")


def test_torn_write_does_not_clobber_good_checkpoint():
    """Simulate a crash between fsync and rename: a stray .tmp must not destroy the
    committed checkpoint, and load() must return the good prior state."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.json")
        store = CheckpointStore(path)
        good = RunState(run_id="good", n_rounds=2, completed_rounds=[0])
        store.save(good)

        # Plant a half-written temp file (what a crash-before-rename would leave).
        torn = os.path.join(d, ".ckpt-torn.tmp")
        with open(torn, "w", encoding="utf-8") as f:
            f.write('{"run_id": "garbage", "completed_rounds": [0,1')  # truncated JSON

        # The committed file is untouched; load() returns the good state.
        back = store.load()
        assert back is not None and back.run_id == "good" and back.completed_rounds == [0]

        # A subsequent successful save still commits atomically and the good data wins.
        good.completed_rounds = [0, 1]
        store.save(good)
        assert store.load().completed_rounds == [0, 1]
        print("[ok] torn temp file does not clobber the committed checkpoint")


def test_corrupt_checkpoint_loads_as_none_not_crash():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("not json at all {{{")
        store = CheckpointStore(path)
        assert store.load() is None, "corrupt checkpoint must load as None, not raise"
        # a fresh run treats it as no checkpoint and starts clean (resumed stays False)
        runner = CheckpointedRun(store, 2, _make_round_fn({}), "fresh")
        state = runner.run()
        assert state.all_rounds_done, "fresh run completes all rounds"
        assert runner.resumed is False, "corrupt prior must NOT count as a resume"
        print("[ok] corrupt checkpoint -> None -> clean start")


def test_real_sklearn_round_body_with_checkpointing():
    """End-to-end with a real sklearn workload: each round fits a different model on
    breast-cancer, scores on a holdout, carries the best as champion across
    atomic checkpoints. Then a crash+resume reaches the same champion without
    refitting completed rounds."""
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score

    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3,
                                          random_state=0, stratify=data.target)

    def model_for(r):
        return [
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            DecisionTreeClassifier(max_depth=3, random_state=0),
            RandomForestClassifier(n_estimators=50, random_state=0),
        ][r]

    fit_counter = {}

    def round_fn(r, state):
        fit_counter[r] = fit_counter.get(r, 0) + 1
        est = model_for(r).fit(Xtr, ytr)
        acc = float(accuracy_score(yte, est.predict(Xte)))
        # carry the BEST champion across rounds (caller owns the "best" decision)
        prev = state.champion_score if state.champion_score is not None else -1.0
        if acc > prev:
            champ_id, champ_score = f"model-{r}", acc
        else:
            champ_id, champ_score = state.champion_id, state.champion_score
        return RoundOutcome(champion_id=champ_id, champion_score=champ_score,
                            records=[{"round": r, "model": r, "acc": round(acc, 4)}],
                            payload={"scores": (state.payload.get("scores", []) + [round(acc, 4)])})

    with tempfile.TemporaryDirectory() as d:
        store = CheckpointStore(os.path.join(d, "sk.json"))

        # clean run for a reference champion
        ref = CheckpointedRun(store, 3, round_fn, "ref").run()
        assert ref.all_rounds_done
        assert fit_counter == {0: 1, 1: 1, 2: 1}, f"clean run fits each once: {fit_counter}"
        assert ref.champion_id is not None and ref.champion_score > 0.9, \
            f"a real model should clear 0.9 on breast-cancer: {ref.champion_score}"

        # wipe to a fresh store and replay with a crash at round 2, then resume
        store2 = CheckpointStore(os.path.join(d, "sk2.json"))
        fit_counter.clear()

        def crashing(r, state):
            if r == 2 and 2 not in fit_counter and "crashed" not in fit_counter:
                fit_counter["crashed"] = True
                raise _CrashAt("crash at round 2")
            return round_fn(r, state)

        try:
            CheckpointedRun(store2, 3, crashing, "c1").run()
        except _CrashAt:
            pass
        assert fit_counter.get(0) == 1 and fit_counter.get(1) == 1 and 2 not in fit_counter

        resume_runner = CheckpointedRun(store2, 3, round_fn, "c2")
        resumed = resume_runner.run()
        assert resume_runner.resumed
        # rounds 0,1 were NOT refit on resume; only round 2 fit (once).
        assert fit_counter.get(0) == 1 and fit_counter.get(1) == 1 and fit_counter.get(2) == 1, \
            f"resume must not refit finished rounds: {fit_counter}"
        assert resumed.champion_id == ref.champion_id
        assert abs(resumed.champion_score - ref.champion_score) < 1e-12
        print(f"[ok] sklearn e2e: champion={resumed.champion_id} "
              f"acc={resumed.champion_score:.4f}, no refit of finished rounds")


def test_should_certify_gate_one_shot():
    state = RunState(run_id="x", n_rounds=1, champion_id="c", payload={})
    assert CheckpointedRun.should_certify(state), "should certify when not yet certified"
    state.payload["certified"] = True
    assert not CheckpointedRun.should_certify(state), "should NOT re-certify after a peek"
    state2 = RunState(run_id="y", n_rounds=1, champion_id=None, payload={})
    assert not CheckpointedRun.should_certify(state2), "no champion -> nothing to certify"
    print("[ok] certification gate is one-shot (no re-peek on resume)")


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
    print(f"\n{len(fns) - failed}/{len(fns)} checkpoint tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
