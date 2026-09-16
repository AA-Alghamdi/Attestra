"""Context/compaction robustness for LONG autonomous runs (the recursive loop's memory surface).

WHERE THIS SITS IN THE CYCLE
----------------------------
run_goal_loop does, each round: baseline -> DIAGNOSE (loop._diagnose: val_lb, headroom,
min_class_recall, ece) -> PROPOSE (llm_moves.propose_moves) -> VoI-rank -> execute -> MEASURE
-> repeat. Over a LONG run that history grows without bound: every round's diagnosis, the moves
that were tried, the leaderboard. Two failure modes today:

  (1) The full raw history is fed to the proposer/knowledge surface verbatim. On a long run that
      blows the context window of whatever consumes it (the LLM proposer, a serialized run report).
  (2) On a plateau the loop re-sweeps the fixed catalog and FORGETS every prior round, so it cannot
      tell "I already tried stronger_model twice and it did not move val_lb" from "fresh start".

This module is the bounded, summarized, OVERFLOW-AWARE memory that fixes both, WITHOUT touching the
trust core. It is pure proposal-policy / read-only state:

  * It NEVER produces or alters a certificate, the sealed peek, or select-then-bound. It only records
    what the loop already measured and re-emits a compact view of it.
  * It ACCEPTS a diagnosis dict (the exact shape loop._diagnose returns) -- it does NOT import loop,
    so there is no circular dependency and no risk of editing a frozen/hot file.

DESIGN CONTRACTS
----------------
(a) BOUNDED, SUMMARIZED HISTORY. Each round is compacted to a single small RoundSummary
    (diagnosis signals + best val_lb so far + the move names tried). Every persisted entry is held
    under a per-entry character ceiling (`per_entry_char_ceiling`); a round that does not fit is NOT
    silently truncated -- see (b). The number of FULL-detail rounds kept is bounded
    (`max_detailed_rounds`); older rounds fold into a running aggregate so total size stays O(1) in
    the number of rounds, not O(rounds).

(b) OVERFLOW RAISES, IT DOES NOT SILENTLY DROP. This is the learn-from-the-incident pattern: when a
    single round's summary exceeds the per-entry ceiling, OR the assembled run-state would exceed the
    total budget even after folding the oldest detailed rounds, we raise CompactionError with the
    measured sizes and the offending field. The loop is expected to treat this as a first-class
    incident (log it, stop, narrow the spec) rather than quietly forgetting rounds and re-sweeping --
    silent truncation is exactly what produces the "forgets every run" gap.

(c) COMPACT RUN STATE for the proposer/knowledge surface. `run_state()` returns a small dict the
    proposer can consume INSTEAD of the raw history: current best val_lb, latest diagnosis, the set
    of moves already tried (so PROPOSE never re-proposes them), a plateau signal (rounds since
    val_lb last improved), and a bounded tail of recent round summaries. This is the read-only memory
    that lets PROPOSE escalate ("tried stronger_model 2x, no movement -> change axis") instead of
    re-sweeping.

This module is CPU-only and has no third-party dependencies (no torch, no jax).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# Sizing. We measure "tokens" with a deterministic, dependency-free proxy: a
# token is ~4 characters (the conventional English-text ratio). The proxy is
# monotonic in true token count, which is all the budget logic needs -- and it
# is reproducible, so an overflow is a fact about the run, not about which
# tokenizer happened to be installed. Callers may pass budgets in tokens
# (default) or characters.
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4


def _char_len(obj) -> int:
    """Canonical serialized size of `obj` in characters. Deterministic
    (sorted keys, no whitespace surprises) so size measurements are stable."""
    if isinstance(obj, str):
        return len(obj)
    return len(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str))


def chars_to_tokens(n_chars: int) -> int:
    """Proxy token count for a character length. Ceil so a non-empty string is
    never reported as 0 tokens."""
    return int(math.ceil(n_chars / _CHARS_PER_TOKEN))


class CompactionError(RuntimeError):
    """Raised when history/run-state would overflow its budget.

    This is the learn-from-the-incident signal: the loop should surface it
    (an honest "context budget exceeded" incident) rather than silently
    dropping rounds. Carries the measured sizes so the incident is auditable.
    """

    def __init__(self, message, *, field=None, size=None, budget=None, unit="tokens"):
        super().__init__(message)
        self.field = field
        self.size = size
        self.budget = budget
        self.unit = unit

    def as_dict(self) -> dict:
        return {
            "error": "CompactionError",
            "message": str(self),
            "field": self.field,
            "size": self.size,
            "budget": self.budget,
            "unit": self.unit,
        }


# Diagnosis keys we PRESERVE verbatim (the key signal the proposer steers on).
# These mirror loop._diagnose's output exactly. Anything else in the diagnosis
# dict is dropped from the per-round summary to keep entries small -- the full
# diagnosis is never the load-bearing thing the proposer needs round-over-round.
_DIAGNOSIS_KEYS = ("val", "val_lb", "headroom", "ece", "below_bar", "min_class_recall")


def _compact_diagnosis(diagnosis) -> dict:
    """Keep only the measured signals the proposer uses. Round floats so the
    serialized form is small and stable."""
    d = diagnosis or {}
    out = {}
    for k in _DIAGNOSIS_KEYS:
        if k not in d:
            continue
        v = d[k]
        if isinstance(v, float):
            v = round(v, 4)
        out[k] = v
    return out


@dataclass
class RoundSummary:
    """One round, compacted. This is the bounded unit of history."""

    round_index: int
    diagnosis: dict                 # compacted diagnosis (the measured signals only)
    best_val_lb: float | None       # best validation lower bound AS OF this round
    moves_tried: list               # move NAMES executed this round (sorted, deduped)
    note: str = ""                  # optional one-line human note (e.g. a fallback reason)

    def char_size(self) -> int:
        return _char_len(asdict(self))

    def token_size(self) -> int:
        return chars_to_tokens(self.char_size())


@dataclass
class _Aggregate:
    """Running O(1) fold of rounds that have aged out of the detailed tail.

    Preserves the KEY signal without per-round storage: how many rounds, the
    union of every move ever tried, the best val_lb ever seen and when, and the
    earliest/latest diagnosis below_bar status. This is what lets the proposer
    answer 'have I tried X before?' over an unbounded run with bounded memory.
    """

    rounds_folded: int = 0
    moves_ever_tried: set = field(default_factory=set)
    best_val_lb: float | None = None
    best_val_lb_round: int | None = None
    first_round_index: int | None = None
    last_round_index: int | None = None

    def fold(self, rs: RoundSummary) -> None:
        self.rounds_folded += 1
        for m in rs.moves_tried:
            self.moves_ever_tried.add(m)
        if rs.best_val_lb is not None and (
            self.best_val_lb is None or rs.best_val_lb > self.best_val_lb
        ):
            self.best_val_lb = rs.best_val_lb
            self.best_val_lb_round = rs.round_index
        if self.first_round_index is None:
            self.first_round_index = rs.round_index
        self.last_round_index = rs.round_index

    def as_dict(self) -> dict:
        return {
            "rounds_folded": self.rounds_folded,
            "moves_ever_tried": sorted(self.moves_ever_tried),
            "best_val_lb": self.best_val_lb,
            "best_val_lb_round": self.best_val_lb_round,
            "round_span": (
                None
                if self.first_round_index is None
                else [self.first_round_index, self.last_round_index]
            ),
        }


class RunContext:
    """Bounded, overflow-aware memory for a long autonomous run.

    Parameters
    ----------
    total_budget : int
        Maximum size of the assembled run-state (the thing handed to the
        proposer / knowledge surface). In `unit` (default "tokens").
    per_entry_ceiling : int
        Maximum size of a SINGLE round summary. A round exceeding this raises
        CompactionError (it is never truncated to fit).
    max_detailed_rounds : int
        How many recent rounds are kept at full RoundSummary detail. Older
        rounds fold into the O(1) aggregate. Bounds history regardless of run
        length.
    unit : "tokens" | "chars"
        Unit for the two budgets above.
    improve_eps : float
        Minimum val_lb increase counted as an improvement (plateau detection).
    """

    def __init__(
        self,
        total_budget: int = 4000,
        per_entry_ceiling: int = 600,
        max_detailed_rounds: int = 8,
        unit: str = "tokens",
        improve_eps: float = 1e-9,
    ):
        if unit not in ("tokens", "chars"):
            raise ValueError(f"unit must be 'tokens' or 'chars', got {unit!r}")
        if total_budget <= 0 or per_entry_ceiling <= 0:
            raise ValueError("budgets must be positive")
        if max_detailed_rounds <= 0:
            raise ValueError("max_detailed_rounds must be positive")
        self.total_budget = total_budget
        self.per_entry_ceiling = per_entry_ceiling
        self.max_detailed_rounds = max_detailed_rounds
        self.unit = unit
        self.improve_eps = improve_eps

        self._detailed: list[RoundSummary] = []
        self._aggregate = _Aggregate()
        self._n_rounds = 0
        # Plateau bookkeeping: best val_lb seen and how many rounds since it last improved.
        self._best_val_lb: float | None = None
        self._best_val_lb_round: int | None = None
        self._rounds_since_improvement = 0
        self._latest_diagnosis: dict = {}

    # -- sizing helpers ----------------------------------------------------

    def _size(self, obj) -> int:
        c = _char_len(obj)
        return c if self.unit == "chars" else chars_to_tokens(c)

    def _entry_size(self, rs: RoundSummary) -> int:
        return rs.char_size() if self.unit == "chars" else rs.token_size()

    # -- recording ---------------------------------------------------------

    def record_round(self, diagnosis, moves_tried, *, val_lb=None, note="") -> RoundSummary:
        """Compact one round and add it to history.

        diagnosis    the round's measured DIAGNOSE signals (loop._diagnose output shape).
        moves_tried  iterable of move NAMES executed this round.
        val_lb       this round's best validation lower bound (falls back to diagnosis["val_lb"]).
        note         optional one-line note (e.g. proposer fallback reason).

        Raises CompactionError if the compacted round exceeds per_entry_ceiling.
        Returns the RoundSummary that was stored.
        """
        comp_diag = _compact_diagnosis(diagnosis)
        if val_lb is None:
            val_lb = comp_diag.get("val_lb")
        moves = sorted({str(m) for m in (moves_tried or [])})
        note = str(note or "")

        rs = RoundSummary(
            round_index=self._n_rounds,
            diagnosis=comp_diag,
            best_val_lb=val_lb,
            moves_tried=moves,
            note=note,
        )

        # (b) per-entry overflow: a single round too big to summarize is an
        # incident, NOT something to truncate. Identify the offending field so
        # the incident is actionable.
        size = self._entry_size(rs)
        if size > self.per_entry_ceiling:
            offending = self._largest_field(rs)
            raise CompactionError(
                f"round {rs.round_index} summary is {size} {self.unit} "
                f"(ceiling {self.per_entry_ceiling}); largest field={offending}. "
                f"Refusing to silently truncate -- treat as a context-budget incident.",
                field=offending,
                size=size,
                budget=self.per_entry_ceiling,
                unit=self.unit,
            )

        # plateau / best-bound bookkeeping (drives the proposer's escalation).
        improved = False
        if val_lb is not None and (
            self._best_val_lb is None or val_lb > self._best_val_lb + self.improve_eps
        ):
            self._best_val_lb = val_lb
            self._best_val_lb_round = rs.round_index
            improved = True
        if improved:
            self._rounds_since_improvement = 0
        elif self._n_rounds > 0:
            self._rounds_since_improvement += 1

        self._latest_diagnosis = comp_diag
        self._n_rounds += 1

        # (a) bound the detailed tail: fold the oldest into the O(1) aggregate.
        self._detailed.append(rs)
        while len(self._detailed) > self.max_detailed_rounds:
            self._aggregate.fold(self._detailed.pop(0))

        return rs

    @staticmethod
    def _largest_field(rs: RoundSummary) -> str:
        sizes = {k: _char_len(v) for k, v in asdict(rs).items()}
        return max(sizes, key=sizes.get)

    # -- introspection -----------------------------------------------------

    @property
    def n_rounds(self) -> int:
        return self._n_rounds

    @property
    def best_val_lb(self):
        return self._best_val_lb

    @property
    def rounds_since_improvement(self) -> int:
        return self._rounds_since_improvement

    def moves_ever_tried(self) -> list:
        """Union of every move name tried across the WHOLE run (detailed tail +
        folded aggregate). This is what PROPOSE consumes so it never re-proposes
        a config already tried, even one that aged out of the detailed tail."""
        s = set(self._aggregate.moves_ever_tried)
        for rs in self._detailed:
            s.update(rs.moves_tried)
        return sorted(s)

    # -- the compact run state (c) ----------------------------------------

    def run_state(self, *, recent_rounds=None) -> dict:
        """The compact 'run state' the proposer / knowledge surface consumes
        INSTEAD of the raw history.

        recent_rounds : optional cap on how many detailed round summaries to
                        include in the tail (defaults to all detailed rounds).

        Always under `total_budget`. If the assembled state would overflow, we
        first shrink the recent tail (oldest first) toward the aggregate -- the
        aggregate still preserves the key signal (moves ever tried, best
        val_lb). If even the minimal state (no tail) overflows, we raise
        CompactionError rather than silently dropping the signal.
        """
        tail_all = self._detailed
        if recent_rounds is not None:
            tail_all = tail_all[-max(0, int(recent_rounds)):]

        def assemble(tail, extra_folded):
            agg = self._aggregate.as_dict()
            agg["rounds_folded"] = agg["rounds_folded"] + extra_folded
            return {
                "n_rounds": self._n_rounds,
                "best_val_lb": self._best_val_lb,
                "best_val_lb_round": self._best_val_lb_round,
                "rounds_since_improvement": self._rounds_since_improvement,
                "plateaued": self._rounds_since_improvement >= max(2, self.max_detailed_rounds // 2),
                "latest_diagnosis": self._latest_diagnosis,
                "moves_ever_tried": self.moves_ever_tried(),
                "aggregate": agg,
                "recent_rounds": [asdict(rs) for rs in tail],
                "budget": {"unit": self.unit, "total": self.total_budget},
            }

        tail = list(tail_all)
        extra_folded = 0
        state = assemble(tail, extra_folded)
        # Shrink the tail oldest-first while overflowing. moves_ever_tried and
        # the aggregate keep the key signal, so dropping a round from the TAIL
        # does not lose "what was tried" -- only its per-round detail.
        while self._size(state) > self.total_budget and tail:
            tail.pop(0)
            extra_folded += 1
            state = assemble(tail, extra_folded)

        size = self._size(state)
        if size > self.total_budget:
            # Even the minimal state (no recent detail) overflows. This is the
            # incident: the proposer cannot be fed a within-budget view. Raise
            # rather than silently dropping moves_ever_tried / the diagnosis.
            offending = max(
                ("moves_ever_tried", "latest_diagnosis", "aggregate"),
                key=lambda k: _char_len(state.get(k)),
            )
            raise CompactionError(
                f"run_state is {size} {self.unit} (budget {self.total_budget}) "
                f"even with no recent-round detail; largest field={offending}. "
                f"Refusing to silently drop the run signal -- narrow the run or raise the budget.",
                field=offending,
                size=size,
                budget=self.total_budget,
                unit=self.unit,
            )
        return state

    # -- convenience for the knowledge surface -----------------------------

    def proposer_view(self, *, recent_rounds=3) -> dict:
        """A tighter view tailored to llm_moves.propose_moves: the fields it
        steers on plus an explicit `tried` set in the form the proposer expects
        (a set of move names). Built on top of run_state so it inherits the same
        overflow guarantees."""
        st = self.run_state(recent_rounds=recent_rounds)
        return {
            "diagnosis": st["latest_diagnosis"],
            "tried": st["moves_ever_tried"],
            "best_val_lb": st["best_val_lb"],
            "plateaued": st["plateaued"],
            "rounds_since_improvement": st["rounds_since_improvement"],
            "recent_rounds": st["recent_rounds"],
        }


def _selftest():
    print("[run_context] selftest")
    rc = RunContext(total_budget=2000, per_entry_ceiling=400, max_detailed_rounds=4, unit="tokens")
    for i in range(12):
        diag = {
            "val": 0.70 + i * 0.005,
            "val_lb": 0.60 + (0.0 if i >= 5 else i * 0.01),  # plateaus after round 5
            "headroom": 0.10,
            "ece": 0.05,
            "below_bar": True,
            "min_class_recall": 0.55,
        }
        rc.record_round(diag, [f"stronger_model_{i%3}", "regularize"], note="fallback")
    st = rc.run_state()
    print("  n_rounds:", st["n_rounds"], "best_val_lb:", st["best_val_lb"])
    print("  plateaued:", st["plateaued"], "since_improve:", st["rounds_since_improvement"])
    print("  moves_ever_tried:", st["moves_ever_tried"])
    print("  detailed tail kept:", len(st["recent_rounds"]), "folded:", st["aggregate"]["rounds_folded"])
    print("  run_state size (tokens):", rc._size(st), "/", rc.total_budget)
    # overflow demonstration
    try:
        rc2 = RunContext(per_entry_ceiling=5, unit="tokens")
        rc2.record_round({"val_lb": 0.5}, ["a_very_long_move_name_that_will_overflow"])
    except CompactionError as e:
        print("  per-entry overflow raised as expected:", e.field, e.size, ">", e.budget)
    print("[run_context] OK")


if __name__ == "__main__":
    _selftest()
