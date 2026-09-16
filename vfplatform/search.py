"""RECURSIVE BEST-FIRST SEARCH with a VERIFICATION-BUDGET ECONOMY.

========================================================================================================
STATUS (2026-06): WIRED into the REGENERATIVE researcher (vfplatform/recipe_research.py) as the alternate
  driver (RecipeResearcher(..., driver="search") -> RecipeSearchProblem + BudgetedSearch). Reached a
  certified NON-SEED champion in the hermetic locks (test_recipe_research.py::test_search_driver_reaches_
  non_seed_champion), same Tier-3 sole-promoter discipline as the climb driver.

  The earlier measurement still stands and is WHY this now has a real job: on a FIXED representation,
  best-first search + budget economy added ~zero over a tuned GBM (B1: CYCLE vs EMB-STRONG 0/5 FDR;
  CYCLE vs RANDOM 0/5). So search is not pointed at authoring-on-a-fixed-rep; it is pointed at the OPEN
  recipe space -- where each expand() can CHANGE the backbone/recipe (the proven lever). Best-first
  recursion + plateau escalation over recipes, with the frozen certifier as the only promoter.
========================================================================================================

WHY THIS EXISTS
---------------
An average researcher searches a little, abandons a branch and never returns, and has one biased verifier.
This searcher does the opposite: it explores a tree of proposals best-first (so it BACKTRACKS to promising
abandoned branches), escalates the KIND of move when a level plateaus (escalate.py: model -> features ->
capacity -> data), and -- crucially -- treats expensive verification as a SCARCE CURRENCY.

THE VERIFICATION-BUDGET ECONOMY
-------------------------------
The only thing that mints a certificate is the frozen Tier-3 sealed certifier, and sealed peeks are finite
(the FDR/peek wealth lives in the frozen layer). So:
  * cheap tiers (0..2 of verification.VerificationCascade) are spent FREELY to prune and to RANK proposals;
  * a node becomes Tier-3-ELIGIBLE only by surviving the cheap screen (its validation lower bound clears
    theta -- the select-then-bound gate);
  * among eligible nodes the searcher spends ONE unit of Tier-3 wealth on the single best (select-then-bound
    discipline), and only while `peek_budget` remains.
The objective is therefore "maximize certified discoveries per unit of verification wealth", which is one
clean resource-allocation problem unifying recursion + verification + allocation.

INVARIANTS
----------
  * D4: only Tier-3 promotes. The searcher ranks/prunes with cheap signals and the surrogate, but a
    SearchResult.certificate can only come from cascade.certify_tier3 (the frozen sealed path).
  * The searcher never relaxes theta or widens anything; it only decides WHERE to spend search + peeks.
  * Pure policy. It imports the cascade, escalate, and (optionally) a surrogate; never the frozen certifier
    directly. Deterministic given a seed.

CONTRACT: the problem is supplied as a SearchProblem[S] (S = an opaque domain state). The searcher knows
nothing about models/features; it only expands, screens, ranks, and (rarely) certifies.
"""
from __future__ import annotations

import heapq
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, List, Optional, Tuple, TypeVar

from . import escalate as ESC
from .verification import Candidate, VerificationCascade

S = TypeVar("S")

# Priorities are validation means in [0,1]; an escalated branch is pushed above all of them so a strategic
# move-class change is explored before the saturated class's leftover children.
_ESCALATED_PRIORITY = 2.0


class SearchProblem(Generic[S], ABC):
    """The domain adapter. Implement these for tabular/vision/text/etc.; the searcher stays generic."""

    @abstractmethod
    def root(self) -> S:
        """The starting state (e.g. the base model / empty feature set)."""

    @abstractmethod
    def expand(self, state: S, move_class: str) -> List[S]:
        """Proposals (children) from `state` under the given move class. The move class lets escalation
        change the KIND of proposal (model -> features -> capacity -> data_acquisition)."""

    @abstractmethod
    def make_candidate(self, state: S) -> Candidate:
        """Map a state to a verification.Candidate (val stream, surrogate score, frozen certify_fn)."""

    def priority(self, state: S) -> float:
        """Cheap scalar used to order the frontier (higher = explore sooner). Default: the candidate's
        observed validation mean (a cheap, already-computed signal)."""
        cand = self.make_candidate(state)
        if cand.val_outcomes:
            return sum(float(x) for x in cand.val_outcomes) / len(cand.val_outcomes)
        return 0.0

    def describe(self, state: S) -> str:
        return str(state)


@dataclass
class Node(Generic[S]):
    state: S
    move_class: str
    depth: int
    priority: float
    parent_id: Optional[int]
    id: int


@dataclass
class SearchResult(Generic[S]):
    certified: bool
    winner_state: Optional[S]
    certificate: Optional[dict]
    peeks_used: int
    cheap_cost: int
    expansions: int
    nodes_screened: int
    eligible_count: int
    move_class_path: List[str]
    log: List[str] = field(default_factory=list)

    @property
    def winner_description(self) -> Optional[str]:
        return None if self.winner_state is None else str(self.winner_state)


class BudgetedSearch(Generic[S]):
    """Best-first recursive search with plateau escalation and a Tier-3 peek budget."""

    def __init__(self, problem: SearchProblem[S], theta: float, *, peek_budget: int = 3,
                 max_depth: int = 4, max_expansions: int = 200, beam: int = 8,
                 cascade: Optional[VerificationCascade] = None, plateau_k: int = 2,
                 allow_data_acquisition: bool = True, seed: int = 0):
        if peek_budget < 1:
            raise ValueError("peek_budget must be >= 1 (need at least one Tier-3 peek to ever promote)")
        self.problem = problem
        self.theta = float(theta)
        self.peek_budget = int(peek_budget)
        self.max_depth = int(max_depth)
        self.max_expansions = int(max_expansions)
        self.beam = int(beam)
        self.cascade = cascade if cascade is not None else VerificationCascade(theta)
        self.plateau_k = int(plateau_k)
        self.allow_data_acquisition = bool(allow_data_acquisition)
        self.seed = int(seed)

    def run(self) -> SearchResult:
        counter = itertools.count()
        # max-heap via negated priority; tie-break by insertion order (deterministic)
        frontier: List[Tuple[float, int, Node[S]]] = []

        def push(node: Node[S]) -> None:
            heapq.heappush(frontier, (-node.priority, node.id, node))

        root_state = self.problem.root()
        move_class = "model"
        root = Node(root_state, move_class, 0, self.problem.priority(root_state), None, next(counter))
        push(root)

        peeks_used = 0
        cheap_cost = 0
        expansions = 0
        nodes_screened = 0
        eligible_count = 0
        best_lb_history: List[Optional[float]] = []
        move_class_path: List[str] = [move_class]
        log: List[str] = []
        best_cert: Optional[dict] = None
        winner_state: Optional[S] = None

        rounds_since_improve = 0
        best_lb_so_far = -1.0

        while frontier and expansions < self.max_expansions and peeks_used < self.peek_budget:
            _, _, node = heapq.heappop(frontier)
            if node.depth >= self.max_depth:
                continue
            children = self.problem.expand(node.state, node.move_class)
            expansions += 1
            if not children:
                continue

            # cheaply screen + rank all children; collect Tier-3-eligible ones (select-then-bound)
            scored: List[Tuple[float, Node[S]]] = []
            eligible: List[Tuple[float, Node[S], Candidate]] = []
            round_best_lb: Optional[float] = None
            for ch_state in children:
                cand = self.problem.make_candidate(ch_state)
                screen = self.cascade.cheap_screen(cand)
                cheap_cost += screen.cost
                nodes_screened += 1
                prio = self.problem.priority(ch_state)
                child = Node(ch_state, node.move_class, node.depth + 1, prio, node.id, next(counter))
                if screen.survived:
                    eligible_count += 1
                    eligible.append((prio, child, cand))
                    round_best_lb = prio if round_best_lb is None else max(round_best_lb, prio)
                else:
                    # not promotable, but may still be worth exploring deeper for its descendants
                    scored.append((prio, child))

            # SPEND a Tier-3 peek on the single best eligible child (select-then-bound)
            if eligible and peeks_used < self.peek_budget:
                eligible.sort(key=lambda t: (-t[0], t[1].id))
                _, best_child, best_cand = eligible[0]
                res = self.cascade.certify_tier3(best_cand)
                peeks_used += 1
                log.append(f"peek#{peeks_used} on {self.problem.describe(best_child.state)} -> "
                           f"{'CERTIFIED' if res.promoted else 'not certified'}")
                if res.promoted:
                    return SearchResult(True, best_child.state, res.certificate, peeks_used, cheap_cost,
                                        expansions, nodes_screened, eligible_count, move_class_path, log)
                # certified-eligible but sealed bound missed: keep exploring its branch too
                scored.append((eligible[0][0], best_child))
                scored.extend((p, c) for p, c, _ in eligible[1:])

            # push the most promising children (beam) back onto the frontier for backtracking
            scored.sort(key=lambda t: (-t[0], t[1].id))
            for _, child in scored[: self.beam]:
                push(child)

            # plateau tracking on the best cheap lower bound seen this round -> escalate the move class
            best_lb_history.append(round_best_lb)
            if round_best_lb is not None and round_best_lb > best_lb_so_far + 1e-9:
                best_lb_so_far = round_best_lb
                rounds_since_improve = 0
            else:
                rounds_since_improve += 1

            decision = ESC.escalation_decision(
                best_lb_history, move_class, rounds_since_improve,
                budget_left=self.peek_budget - peeks_used, k=self.plateau_k,
                allow_data_acquisition=self.allow_data_acquisition)
            nxt = _decision_to_class(decision.decision)
            if nxt is not None and nxt != move_class:
                move_class = nxt
                move_class_path.append(move_class)
                rounds_since_improve = 0
                log.append(f"plateau -> escalate move class to '{move_class}'")
                # re-seed the frontier root under the new move class with TOP priority: escalation is a
                # deliberate strategic decision (the current class is saturated), so the new proposal kind
                # should be pursued before the stale, lower-bound children of the saturated class.
                push(Node(root_state, move_class, 0, _ESCALATED_PRIORITY, None, next(counter)))

        return SearchResult(False, winner_state, best_cert, peeks_used, cheap_cost, expansions,
                            nodes_screened, eligible_count, move_class_path, log)


def _decision_to_class(decision: str) -> Optional[str]:
    """Map an escalate decision string to the move class it escalates INTO (or None for continue/stop)."""
    return {
        ESC.ESCALATE_TO_FEATURES: "features",
        ESC.ESCALATE_TO_CAPACITY: "capacity",
        ESC.ESCALATE_TO_DATA: "data_acquisition",
    }.get(decision)


__all__ = ["SearchProblem", "BudgetedSearch", "Node", "SearchResult"]
