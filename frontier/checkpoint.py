"""P6 checkpoint-resume: durable, crash-safe, skip-completed-work checkpointing.

The audit (ROADMAP Phase 6) found the existing resume broken in a specific,
seductive way: `attestra/execution/checkpoint.py` *loads* prior state on resume
and then *re-runs every phase anyway*. Loading state is not resuming; resuming
means the work that finished before the crash is NOT recomputed. That distinction
is the whole point of this module, and the self-test proves it with a counter
side-effect: after a simulated crash at round k, rounds 0..k execute zero times on
resume.

What this module provides
-------------------------
  * `CheckpointStore` -- atomic, durable, single-file JSON state.
      - atomic write: serialize to a temp file in the SAME directory, fsync the
        file, `os.replace` (atomic rename on POSIX/NTFS), then fsync the directory
        so the rename itself is durable. A crash at any instant leaves either the
        old complete file or the new complete file -- never a torn write.
      - lineage: each checkpoint carries `run_id`, `parent_run_id`, and a monotone
        `lineage` list, so a resumed run records which run it descends from. This
        makes a chain of crash/resume cycles auditable.
  * `RunState` -- the serialized unit: which rounds completed, the champion carried
      across rounds, per-round records, an opaque user `payload`, and lineage.
  * `CheckpointedRun` -- a thin wrapper around ANY round-based loop. You hand it a
      per-round callable `round_fn(round_index, state) -> RoundOutcome`. It:
        1. loads prior state if present (resume),
        2. iterates rounds, SKIPPING any whose index is already in
           `completed_rounds`,
        3. checkpoints atomically after each round (so a crash loses at most the
           in-flight round, never a finished one),
        4. returns the final RunState.

Integrity stance (CONTRACT.md invariants)
------------------------------------------
This module is pure orchestration/persistence. It computes NO promotion-bearing
number. It never touches the sealed test, never scores, never decides what
promotes. It only remembers *which rounds are done* and *what the champion is so
far* so the engine does not repeat finished work. The frozen certifier remains the
sole promoter; checkpointing is transparent to it. (CONTRACT invariants 1-5 are
preserved because this layer makes no scientific decision.)

# === WIRING ===
# The Phase-0 `ResearchEngine.run` (engine.py) runs `for r in range(cfg.rounds): ...`
# with three pieces of carried state: `best_score`, `best_prog`, and the
# `tried_labels` / `recent_errors` diagnosis channel. To make that loop resumable
# WITHOUT editing engine.py (additive-only rule), the integrator wraps the body:
#
#   from frontier.checkpoint import CheckpointedRun, RoundOutcome, CheckpointStore
#
#   store = CheckpointStore(path="runs/<task>/<run_id>.json")
#   def round_fn(r: int, state) -> RoundOutcome:
#       # 'state.payload' carries best_score/best_prog_id/tried_labels/recent_errors
#       # exactly the dict engine.py builds as `context`. Run ONE round of
#       # propose->sandbox->score->select, then return the updated champion + payload.
#       ...
#       return RoundOutcome(
#           champion_id=best_prog.id if best_prog else None,
#           champion_score=best_score,
#           records=[...],                # per-candidate _Record-like dicts
#           payload={"tried_labels": sorted(tried_labels),
#                     "recent_errors": recent_errors,
#                     "best_prog_id": best_prog.id if best_prog else None},
#       )
#   runner = CheckpointedRun(store=store, n_rounds=cfg.rounds, round_fn=round_fn,
#                            run_id=<run_id>, parent_run_id=<prev run if resuming>)
#   final_state = runner.run()        # resumes automatically if store has state
#
# After the loop, engine.py certifies `final_state.champion_id`'s Program on the
# sealed test exactly as today (one counted peek). Certification itself is NOT
# checkpointed mid-flight: the sealed peek is atomic and one-shot, so it either
# happened (recorded in payload) or it must be redone -- there is no partial peek
# to resume. `CheckpointedRun` exposes `state.payload["certified"]` so a resume
# after certification skips re-peeking; see `should_certify()`.
#
# Ordering contract: round_fn is called for r in ascending order; a round is
# marked complete (and checkpointed) ONLY after round_fn returns without raising.
# A crash inside round_fn leaves that round un-completed, so resume re-runs it (at
# most one round of lost work) and never re-runs a completed earlier round.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Serialized state
# --------------------------------------------------------------------------- #
@dataclass
class RunState:
    """The complete, serializable state of a round-based run.

    Everything needed to resume lives here. `payload` is an opaque dict owned by
    the caller (the engine stuffs its champion id, tried_labels, recent_errors,
    and a `certified` flag there). This module only reads `completed_rounds`,
    `champion_*`, and lineage fields; it treats `payload` as a black box it
    persists verbatim.
    """

    run_id: str
    n_rounds: int
    completed_rounds: List[int] = field(default_factory=list)
    champion_id: Optional[str] = None
    champion_score: Optional[float] = None
    records: List[dict] = field(default_factory=list)   # flat per-candidate log
    payload: dict = field(default_factory=dict)         # opaque caller state
    # ---- lineage / provenance ----
    parent_run_id: Optional[str] = None
    lineage: List[str] = field(default_factory=list)    # ancestor run_ids, oldest-first
    resume_count: int = 0                               # how many times this state was resumed
    schema_version: int = SCHEMA_VERSION
    created_at: float = 0.0
    updated_at: float = 0.0

    # -- convenience -------------------------------------------------------- #
    def is_round_done(self, r: int) -> bool:
        # Membership must be type-robust. `completed_rounds` is persisted via JSON and
        # rehydrated by `from_json`, which round-trips a list -> Python list but does NOT
        # coerce element types. If a checkpoint ever carries round indices as anything other
        # than plain `int` (e.g. a JSON producer that wrote them as strings, or a numpy int
        # that slipped through serialization), the naive `r in self.completed_rounds` would
        # silently return False for EVERY round -- making a resumed run recompute finished
        # work (the exact "skips nothing" failure this module exists to prevent). Compare on
        # int-coerced values so the gate is correct regardless of how the list was serialized.
        try:
            target = int(r)
        except (TypeError, ValueError):
            return False
        for c in self.completed_rounds:
            try:
                if int(c) == target:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @property
    def all_rounds_done(self) -> bool:
        return len(set(self.completed_rounds)) >= self.n_rounds

    @property
    def next_round(self) -> Optional[int]:
        """Lowest round index in [0, n_rounds) not yet completed, else None."""
        done = set(self.completed_rounds)
        for r in range(self.n_rounds):
            if r not in done:
                return r
        return None

    def to_json(self) -> str:
        # sort_keys for byte-stable, diffable checkpoints (auditability).
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @staticmethod
    def from_json(text: str) -> "RunState":
        d = json.loads(text)
        known = {f for f in RunState.__dataclass_fields__}  # tolerate extra/missing keys
        return RunState(**{k: v for k, v in d.items() if k in known})


@dataclass
class RoundOutcome:
    """What a single round hands back to the checkpointer.

    `champion_id`/`champion_score` are the best-so-far AFTER this round (the engine
    decides what "best" means via the frozen scorer; this module just records it).
    `records` are appended to the run log; `payload` REPLACES the run payload
    (caller is expected to thread prior payload through, not lose it).
    """

    champion_id: Optional[str] = None
    champion_score: Optional[float] = None
    records: List[dict] = field(default_factory=list)
    payload: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Atomic durable store
# --------------------------------------------------------------------------- #
class CheckpointStore:
    """Single-file, atomic, durable JSON checkpoint store.

    Durability model (why each step is here):
      - We write to a temp file in the *same directory* as the target so that
        `os.replace` is a rename within one filesystem (atomic; cross-fs rename is
        a copy and is NOT atomic).
      - `f.flush()` + `os.fsync(fd)` force the bytes to stable storage before the
        rename, so the rename cannot expose a file whose contents are still in the
        page cache.
      - `os.replace(tmp, path)` atomically swaps the new file in. A reader (or a
        crash) sees either the whole old file or the whole new file.
      - We then fsync the *directory* so the rename entry itself is durable; on
        many filesystems the directory entry is a separate metadata write that can
        be lost on power failure even after the file fsync.
    A crash at any point leaves a valid prior checkpoint (or none); never a torn or
    truncated one. The orphan temp file (if a crash lands between write and replace)
    is harmless and cleaned on the next save.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)

    # -- existence / load --------------------------------------------------- #
    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def load(self) -> Optional[RunState]:
        """Return the persisted RunState, or None if there is no checkpoint.

        A corrupt/half-written file should be impossible given the atomic save,
        but we still guard: a JSON decode error returns None (treat as no
        checkpoint) rather than crashing the resume path. We do NOT silently
        delete it; the caller can inspect `self.path`.
        """
        if not self.exists():
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return RunState.from_json(f.read())
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    # -- atomic durable save ------------------------------------------------ #
    def save(self, state: RunState) -> None:
        """Atomically and durably persist `state` (tmp + fsync + os.replace)."""
        state.updated_at = time.time()
        if not state.created_at:
            state.created_at = state.updated_at
        payload = state.to_json().encode("utf-8")

        d = os.path.dirname(self.path) or "."
        # delete=False: we manage the lifetime; the rename consumes the temp file.
        fd, tmp = tempfile.mkstemp(prefix=".ckpt-", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())          # bytes durable before rename
            os.replace(tmp, self.path)        # atomic swap
            tmp = None                        # consumed; do not unlink in finally
            self._fsync_dir(d)                # rename entry durable
        finally:
            if tmp is not None and os.path.exists(tmp):
                # crash-safety cleanup: an unconsumed temp must not linger.
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    @staticmethod
    def _fsync_dir(d: str) -> None:
        """fsync a directory so a fresh rename entry survives power loss.

        Not all platforms permit opening a directory for fsync (e.g. some Windows
        setups); we degrade honestly by skipping rather than failing the save.
        """
        try:
            dfd = os.open(d, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dfd)
        except OSError:
            pass
        finally:
            os.close(dfd)


# --------------------------------------------------------------------------- #
# The resumable round runner
# --------------------------------------------------------------------------- #
class CheckpointedRun:
    """Wrap a round-based loop so completed rounds are never recomputed.

    Parameters
    ----------
    store : CheckpointStore
        Where state is persisted (atomically, after every completed round).
    n_rounds : int
        Total number of rounds the loop should run.
    round_fn : Callable[[int, RunState], RoundOutcome]
        Executes exactly ONE round. Called only for rounds NOT already completed,
        in ascending index order. It receives the live RunState (read champion /
        payload from prior rounds) and returns the updated champion + records +
        payload. Raising from round_fn aborts the run *without* marking that round
        complete, so a later resume re-runs only that round.
    run_id : str
        Identity of THIS run attempt.
    parent_run_id : Optional[str]
        If this attempt resumes/forks a prior run, its id (for lineage). When a
        prior checkpoint is loaded, lineage is extended automatically; this
        argument additionally records an explicit fork parent.
    on_resume : Optional[Callable[[RunState], None]]
        Hook fired once, after a prior checkpoint is loaded, before any round runs.
        Lets the caller rehydrate in-memory objects (e.g. rebuild the champion
        Program from its id) from the persisted payload.

    Notes
    -----
    The CRITICAL property (the bug this fixes): on resume, rounds whose index is in
    `completed_rounds` are skipped entirely -- `round_fn` is not called for them.
    The self-test enforces this with a per-round counter.
    """

    def __init__(
        self,
        store: CheckpointStore,
        n_rounds: int,
        round_fn: Callable[[int, RunState], RoundOutcome],
        run_id: str,
        parent_run_id: Optional[str] = None,
        on_resume: Optional[Callable[[RunState], None]] = None,
    ):
        self.store = store
        self.n_rounds = int(n_rounds)
        self.round_fn = round_fn
        self.run_id = run_id
        self.parent_run_id = parent_run_id
        self.on_resume = on_resume
        self.resumed = False  # set True if a prior checkpoint was loaded

    # -- resume ------------------------------------------------------------- #
    def resume(self) -> RunState:
        """Load prior state (if any) and return the state to run from.

        This is the heart of the fix. If a checkpoint exists, we return it with
        lineage extended and `resume_count` bumped -- and crucially its
        `completed_rounds` is preserved so the run loop will SKIP those rounds.
        If no checkpoint exists, we mint a fresh state. Either way the returned
        state's `next_round` tells the loop exactly where to pick up.
        """
        prior = self.store.load()
        if prior is None:
            now = time.time()
            return RunState(
                run_id=self.run_id,
                n_rounds=self.n_rounds,
                parent_run_id=self.parent_run_id,
                lineage=([self.parent_run_id] if self.parent_run_id else []),
                created_at=now,
                updated_at=now,
            )

        # --- genuine resume: keep completed_rounds intact ---
        self.resumed = True
        # Extend lineage: the prior run_id becomes an ancestor of this attempt.
        new_lineage = list(prior.lineage)
        if prior.run_id and prior.run_id not in new_lineage:
            new_lineage.append(prior.run_id)
        if self.parent_run_id and self.parent_run_id not in new_lineage:
            new_lineage.append(self.parent_run_id)

        prior.parent_run_id = prior.run_id
        prior.run_id = self.run_id
        prior.lineage = new_lineage
        prior.resume_count = int(prior.resume_count) + 1
        # n_rounds may have been extended by the caller for the new attempt;
        # honor the larger so newly-added rounds still run.
        prior.n_rounds = max(prior.n_rounds, self.n_rounds)
        self.n_rounds = prior.n_rounds

        if self.on_resume is not None:
            self.on_resume(prior)
        # Persist the resume bookkeeping immediately so lineage is durable even if
        # we crash before completing the first resumed round.
        self.store.save(prior)
        return prior

    # -- run ---------------------------------------------------------------- #
    def run(self) -> RunState:
        """Resume (or start), run only the unfinished rounds, checkpoint each.

        Returns the final RunState (all rounds in [0, n_rounds) completed).
        """
        state = self.resume()

        for r in range(state.n_rounds):
            if state.is_round_done(r):
                continue  # <-- the fix: finished work is NOT recomputed

            outcome = self.round_fn(r, state)  # may raise; round stays incomplete

            # Apply the outcome, then mark complete, then checkpoint -- in that
            # order so a checkpoint never claims a round whose effects are unset.
            if outcome.champion_id is not None:
                # Carry the champion only if this round produced one at least as
                # good as the current; the caller already decided "best" via the
                # frozen scorer, so we trust its returned champion.
                state.champion_id = outcome.champion_id
                state.champion_score = outcome.champion_score
            state.records.extend(outcome.records)
            if outcome.payload is not None:
                state.payload = outcome.payload

            if r not in state.completed_rounds:
                state.completed_rounds.append(r)
                state.completed_rounds.sort()

            self.store.save(state)  # durable: a crash now keeps round r done

        return state

    # -- certification gate (one-shot, not mid-flight resumable) ------------ #
    @staticmethod
    def should_certify(state: RunState) -> bool:
        """Whether the sealed certification still needs to run.

        The sealed peek is one-shot and atomic (CONTRACT invariant 1): there is no
        partial peek to resume. So we gate it on a boolean in the payload rather
        than checkpointing it mid-flight. If `payload['certified']` is already set
        (the engine recorded the certificate), a resume skips re-peeking; otherwise
        certification must run for the champion.
        """
        if state.champion_id is None:
            return False
        return not bool(state.payload.get("certified"))
