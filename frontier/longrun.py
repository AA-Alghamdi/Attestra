"""frontier.longrun -- WORKING long-horizon resume for the CoreOrchestrator.

WHY THIS EXISTS (the audit finding it fixes)
--------------------------------------------
`frontier/checkpoint.py` ships a sound, well-tested resumable round runner
(`CheckpointedRun`), but the live end-to-end loop -- `CoreOrchestrator.run`
(frontier/core/orchestrator.py) -- did NOT use it for resume. Its loop is

    for r in range(cfg.rounds):
        ...                       # propose -> sandbox -> score -> select
        self._checkpoint_round(...)   # SAVES state

The orchestrator *saved* a checkpoint after each round, but on a fresh `run()`
it always re-entered the loop at `r = 0` and recomputed every round from scratch.
Saving state is not resuming. A crash after round j followed by a restart re-ran
rounds 0..j (all of their proposal/sandbox/portfolio work), wasting the entire
prior compute budget. THIS module makes a resumed run actually SKIP rounds 0..j.

HOW IT WORKS (additive, no Phase-0 edit, certify-once preserved)
---------------------------------------------------------------
The orchestrator exposes two optional, default-None seams (added in this build):

  * `orch._round_gate(r) -> (skip, champion, champion_score)` -- consulted at the
    TOP of every round. When `skip` is True the round body never executes and the
    durably-recorded champion is restored into the loop's `best_prog`/`best_score`.
  * `orch._on_round_complete(r, best_prog, best_score)` -- fired AFTER a round's
    body finishes, so we can checkpoint the FULL champion + lineage atomically.

`ResumableRun` installs those seams, backed by `CheckpointStore` (the same atomic
tmp+fsync+os.replace store + run/seed lineage from checkpoint.py). On resume it:

  1. loads the prior `RunState` (its `completed_rounds` and serialized champion),
  2. rehydrates the champion `Program` from the persisted code/source/label,
  3. gates the loop so rounds in `completed_rounds` are SKIPPED (no recompute),
     restoring the champion so end-of-loop certification sees the right winner,
  4. checkpoints atomically after every freshly-run round, recording round/seed
     lineage and the full champion so the NEXT crash also loses at most one round.

THE SEALED PEEK IS UNTOUCHED. The single `certify_on_sealed` call lives OUTSIDE
the round loop (orchestrator `_certify_winner`). This module never moves it,
never gates it, never checkpoints it mid-flight. A resume re-runs the loop with
zero new rounds (all skipped), arrives at the champion, and certifies EXACTLY
ONCE at the end -- `result.sealed_peeks == 1`, same as a clean run. We do not
persist a certificate across processes (a sealed peek is one-shot and atomic;
there is no partial peek to resume), so resume always certifies the final winner
once. This is the honest, conservative choice: it never *skips* certification and
never *double*-counts a peek within a single process.

WIRING (this is the live path -- not a print/log)
-------------------------------------------------
    from frontier.core.orchestrator import CoreOrchestrator, CoreConfig
    from frontier.longrun import ResumableRun

    orch = CoreOrchestrator(CoreConfig(rounds=8))
    run  = ResumableRun(orch, checkpoint_path="runs/tumors/state.json", run_id="attempt-1")
    result = run.run(goal="classify tumors", X=X, y=y, theta=0.90)
    # ^ if state.json already records rounds 0..j done, those rounds are SKIPPED;
    #   only j+1.. run, then the winner is certified once. result is a CoreResult.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier.checkpoint import CheckpointStore, RunState  # noqa: E402
from frontier.program import Program                       # noqa: E402
from frontier.core.orchestrator import CoreOrchestrator, CoreResult  # noqa: E402


def _serialize_champion(prog: Optional[Program]) -> Optional[dict]:
    """Persist a champion Program in full so a different process can rehydrate it.

    We store everything the Program dataclass needs (code is required by the spine;
    source is required by the Program contract). Storing only the id would be
    useless across processes -- the proposers are stochastic and may never re-mint
    the same code -- so we keep the code verbatim. This is selection-side state; it
    never carries a metric the parent did not compute (the score is stored
    separately and was computed by certify.score_val).
    """
    if prog is None:
        return None
    return {
        "code": prog.code,
        "source": prog.source,
        "label": prog.label,
        "parent_id": prog.parent_id,
        "provenance": dict(prog.provenance or {}),
        "id": prog.id,
    }


def _rehydrate_champion(d: Optional[dict]) -> Optional[Program]:
    """Rebuild a Program from `_serialize_champion` output (or None)."""
    if not d or not d.get("code"):
        return None
    return Program(
        code=d["code"],
        source=d.get("source", "resume"),
        label=d.get("label", ""),
        parent_id=d.get("parent_id"),
        provenance=dict(d.get("provenance") or {}),
    )


class ResumableRun:
    """Drive a CoreOrchestrator with durable, crash-safe, skip-completed-work resume.

    Parameters
    ----------
    orchestrator : CoreOrchestrator
        The live end-to-end loop. We install the resume seams on it for the duration
        of `run()` and restore them afterward (so the same orchestrator instance is
        reusable and the default path stays untouched).
    checkpoint_path : str
        Where the atomic durable RunState is persisted (one JSON file; the store
        uses tmp+fsync+os.replace + a directory fsync).
    run_id : str | None
        Identity of THIS attempt (for round/seed lineage). Auto-minted if None.
    """

    def __init__(self, orchestrator: CoreOrchestrator, checkpoint_path: str,
                 run_id: Optional[str] = None):
        self.orch = orchestrator
        self.store = CheckpointStore(checkpoint_path)
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        # Live state for this attempt (set in run()).
        self.state: Optional[RunState] = None
        self.resumed: bool = False
        # Observability: which rounds THIS attempt actually executed (vs skipped).
        # A durable side-effect that proves finished rounds are not recomputed.
        self.executed_rounds: list[int] = []
        self.skipped_rounds: list[int] = []

    # ------------------------------------------------------------------ resume load
    def _load_or_init(self, n_rounds: int, seed: int) -> RunState:
        """Load a prior checkpoint (resume) or mint a fresh RunState with lineage."""
        prior = self.store.load()
        now = time.time()
        if prior is None:
            return RunState(
                run_id=self.run_id, n_rounds=n_rounds,
                lineage=[], created_at=now, updated_at=now,
                payload={"seed": seed, "round_lineage": []},
            )
        # genuine resume: keep completed_rounds intact, extend run/seed lineage.
        self.resumed = True
        new_lineage = list(prior.lineage)
        if prior.run_id and prior.run_id not in new_lineage:
            new_lineage.append(prior.run_id)
        prior.parent_run_id = prior.run_id
        prior.run_id = self.run_id
        prior.lineage = new_lineage
        prior.resume_count = int(prior.resume_count) + 1
        prior.n_rounds = max(prior.n_rounds, n_rounds)
        if not isinstance(prior.payload, dict):
            prior.payload = {}
        prior.payload.setdefault("seed", seed)
        prior.payload.setdefault("round_lineage", [])
        return prior

    # ------------------------------------------------------------------ the seams
    def _make_round_gate(self):
        """Gate consulted at the top of each round. Skip => round body not executed.

        Returns (skip, champion_program, champion_score). When the prior state marks
        round r complete, we skip and hand back the rehydrated champion so the loop
        carries the correct winner into the single end-of-loop certification.
        """
        carried = _rehydrate_champion((self.state.payload or {}).get("champion"))
        carried_score = self.state.champion_score

        def gate(r: int) -> Tuple[bool, Optional[Program], Optional[float]]:
            if self.state.is_round_done(r):
                self.skipped_rounds.append(r)
                return True, carried, carried_score
            return False, None, None

        return gate

    def _make_on_complete(self):
        """Checkpoint AFTER a freshly-run round: full champion + round/seed lineage."""
        def on_complete(r: int, best_prog: Optional[Program],
                        best_score: Optional[float]) -> None:
            self.executed_rounds.append(r)
            st = self.state
            if r not in st.completed_rounds:
                st.completed_rounds.append(r)
                st.completed_rounds.sort()
            st.champion_id = best_prog.id if best_prog is not None else None
            st.champion_score = best_score
            if not isinstance(st.payload, dict):
                st.payload = {}
            st.payload["champion"] = _serialize_champion(best_prog)
            # round/seed lineage: append (round, run_id, ts) so the audit trail shows
            # which attempt produced each completed round.
            st.payload.setdefault("round_lineage", [])
            st.payload["round_lineage"].append(
                {"round": r, "run_id": st.run_id, "ts": time.time()})
            self.store.save(st)  # atomic + durable: a crash now keeps round r done

        return on_complete

    # ------------------------------------------------------------------ public run
    def run(self, goal: str, X, y, *, theta: float = 0.0, metric: str = "",
            name: str = "task") -> CoreResult:
        """Run the orchestrator with resume installed. Skips completed rounds.

        Behavior change vs a bare orchestrator: if `checkpoint_path` already records
        completed rounds, those rounds' proposal/sandbox/portfolio work is NOT redone.
        The single sealed peek still happens exactly once at the end.
        """
        cfg = self.orch.cfg
        self.executed_rounds = []
        self.skipped_rounds = []
        self.resumed = False
        self.state = self._load_or_init(cfg.rounds, cfg.seed)
        # If the loaded state has MORE rounds than this config asks for, honor the
        # larger so we never silently drop work the prior attempt scheduled.
        if self.state.n_rounds > cfg.rounds:
            cfg.rounds = self.state.n_rounds
        # Persist resume bookkeeping immediately so lineage is durable even if we
        # crash before completing the first fresh round.
        self.store.save(self.state)

        gate = self._make_round_gate()
        on_complete = self._make_on_complete()
        prev_gate = getattr(self.orch, "_round_gate", None)
        prev_complete = getattr(self.orch, "_on_round_complete", None)
        self.orch._round_gate = gate
        self.orch._on_round_complete = on_complete
        try:
            result = self.orch.run(goal, X, y, theta=theta, metric=metric, name=name)
        finally:
            # restore the orchestrator's default (unwrapped) behavior.
            self.orch._round_gate = prev_gate
            self.orch._on_round_complete = prev_complete
        return result


__all__ = ["ResumableRun", "RunState", "CheckpointStore"]
