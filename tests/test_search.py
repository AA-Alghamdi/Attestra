"""Tests for recursive best-first search + budget economy + escalation.

Acceptance (Phase 5): on a planted-signal problem where the win is only reachable after ESCALATING the
move class (model -> features), the search (a) finds the signal and certifies it, (b) escalates the move
class to get there, and (c) spends FAR fewer expensive Tier-3 peeks than a 'certify every candidate'
baseline (trial-rungs drop)."""
import random
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import search as SR
from vfplatform.verification import Candidate

THETA = 0.80


def _stable_seed(label: str) -> int:
    return zlib.crc32(label.encode()) & 0xFFFFFFFF


def _stream(p: float, n: int, label: str) -> List[float]:
    rng = random.Random(_stable_seed(label))
    return [1.0 if rng.random() < p else 0.0 for _ in range(n)]


@dataclass(frozen=True)
class St:
    label: str
    true_p: float


class PlantedProblem(SR.SearchProblem):
    """The certifiable signal lives only on the 'features' move class. The 'model' class is saturated
    (every variant is well below theta), so the searcher MUST plateau-escalate to find it."""

    def root(self) -> St:
        return St("root", 0.50)

    def expand(self, state: St, move_class: str) -> List[St]:
        if move_class == "model":
            return [St(f"model_{i}", 0.55 + 0.01 * i) for i in range(4)]      # all << theta
        if move_class == "features":
            return [St("feat_SIGNAL", 0.95), St("feat_noise1", 0.60), St("feat_noise2", 0.58)]
        if move_class == "capacity":
            return [St(f"cap_{i}", 0.66) for i in range(3)]
        return []

    def make_candidate(self, state: St) -> Candidate:
        return Candidate(
            name=state.label,
            val_outcomes=_stream(state.true_p, 400, state.label),
            certify_fn=(lambda p=state.true_p: {"certified": p > THETA,
                                                 "lower_bound": round(p - 0.05, 4)}))


def test_search_finds_planted_signal_via_escalation():
    problem = PlantedProblem()
    searcher = SR.BudgetedSearch(problem, THETA, peek_budget=3, max_depth=4,
                                 max_expansions=200, plateau_k=2)
    res = searcher.run()
    assert res.certified is True
    assert res.winner_state is not None and res.winner_state.label == "feat_SIGNAL"
    # it had to ESCALATE the move class to find the signal
    assert "features" in res.move_class_path
    # budget economy: it certified the winner with very few expensive peeks...
    assert res.peeks_used <= 3
    # ...far fewer than a 'certify every screened candidate' baseline would have spent
    assert res.peeks_used < res.nodes_screened
    print(f"peeks_used={res.peeks_used} nodes_screened={res.nodes_screened} "
          f"eligible={res.eligible_count} move_path={res.move_class_path}")


def test_no_signal_returns_honest_no_certify():
    class NoSignal(PlantedProblem):
        def expand(self, state, move_class):
            return [St(f"{move_class}_{i}", 0.60) for i in range(3)]   # nothing ever clears theta
    res = SR.BudgetedSearch(NoSignal(), THETA, peek_budget=2, max_depth=3, plateau_k=1).run()
    assert res.certified is False and res.winner_state is None
    assert res.certificate is None


def test_peek_budget_is_respected():
    # a problem where everything is eligible but nothing certifies -> peeks must stop at the budget
    class AllEligibleNonePass(SR.SearchProblem):
        def root(self):
            return St("r", 0.90)
        def expand(self, state, move_class):
            return [St(f"{move_class}_{state.label}_{i}", 0.90) for i in range(3)]
        def make_candidate(self, state):
            return Candidate(name=state.label, val_outcomes=_stream(0.90, 400, state.label),
                             certify_fn=lambda: {"certified": False, "lower_bound": 0.85})
    res = SR.BudgetedSearch(AllEligibleNonePass(), THETA, peek_budget=2, max_depth=5,
                            max_expansions=500).run()
    assert res.certified is False
    assert res.peeks_used <= 2          # never exceeds the verification-budget


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
