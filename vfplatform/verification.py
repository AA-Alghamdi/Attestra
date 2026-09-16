"""The COST-STRATIFIED VERIFICATION CASCADE -- cheap tiers prune, only the frozen tier promotes.

STATUS (2026-06): WIRED into the REGENERATIVE researcher as the validity cascade. RecipeResearcher uses
VerificationCascade/Candidate so a regenerated recipe must clear the cheap tiers (sanity -> competence
screen -> val bound) before it may spend a sealed peek, and ONLY Tier 3 (the frozen sealed certifier)
promotes. Falsified by the hermetic locks in test_recipe_research.py (open discovery + sole-promoter on
sealed). Tier 0.5 surrogate stays OPTIONAL/dark by design (the cheap competence screen already prunes
losers; the learned verifier is future fuel, not a promoter) -- see vfplatform/surrogate.py.

WHY THIS EXISTS
---------------
Recursion is only affordable if verification is cheap until it must be expensive. A single fixed-n sealed
certificate per candidate would make a thousand-trial search unthinkable. The cascade spends cheap
verification freely to PRUNE and reserves the one expensive, promoting check for the rare survivor:

    Tier 0   sanity        ~free   -- valid, non-degenerate predictions? (kill obvious failures)
    Tier 0.5 surrogate     ~free   -- learned verifier predicts the verdict; prunes likely losers
    Tier 1   e-process     cheap   -- anytime-valid racing on a validation stream; EARLY-KILL hopeless
                                       candidates and early-flag clearly-strong ones (Ville's inequality,
                                       valid under repeated peeking -- vfplatform/eprocess.py)
    Tier 2   val bound      med    -- frozen Clopper-Pearson lower bound on VALIDATION clears theta?
                                       (the select-then-bound gate that earns the right to a sealed peek)
    Tier 3   sealed cert    high   -- the FROZEN sealed certifier. THE SOLE PROMOTER.

THE INVARIANT (D4)
------------------
Cheap tiers and the learned surrogate NEVER promote. They only ALLOCATE search: a kill at Tier 0/0.5/1/2
means "don't spend more compute here", not "this is certified false". Only Tier 3 -- the frozen sealed
certifier reached through vfplatform/sealed.py -- can set promoted=True. `assert_only_tier3_promotes`
makes this a runtime guard. The cheap tiers are statistically honest where they make claims (Tier 1 is
anytime-valid; Tier 2 is the exact frozen bound) but their honest job is pruning, not certification.

CONTRACT: imports the frozen `science` (read certifier helpers) and the standalone `eprocess`. Produces
no certificate of its own; Tier 3 delegates entirely to the caller-supplied frozen sealed certify_fn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .eprocess import EProcess

TIER0 = "tier0_sanity"
TIER05 = "tier0.5_surrogate"
TIER1 = "tier1_eprocess"
TIER2 = "tier2_val_bound"
TIER3 = "tier3_sealed"

# Relative "rung" cost of reaching each tier. The cascade's whole point is to minimize total cost by
# killing most candidates at the cheap rungs so few ever reach the expensive ones (esp. Tier 3).
TIER_COST = {TIER0: 1, TIER05: 1, TIER1: 5, TIER2: 20, TIER3: 100}

ADVANCE = "advance"   # survived this tier; proceed to the next
KILL = "kill"         # pruned here (NOT a certificate of falsehood -- just stop spending)
PROMOTE = "promote"   # ONLY emitted by Tier 3 on a real frozen certificate


@dataclass
class Candidate:
    """A single trial as seen by the cascade. Only the fields a given tier needs must be populated.

    val_outcomes: per-example graded correctness in [0,1] on a VALIDATION stream (drives Tier 1 racing AND
        Tier 2's count-based lower bound -- one consistent signal).
    surrogate_score: Tier 0.5 predicted pass-probability in [0,1] (filled by surrogate.py); None = skip.
    certify_fn: the FROZEN sealed certify call, returning a certificate dict with a boolean 'certified'.
        Called at most once, only if the candidate reaches Tier 3. None means "not certifiable here".
    """
    name: str
    sanity_ok: bool = True
    val_outcomes: Optional[Sequence[float]] = None
    surrogate_score: Optional[float] = None
    certify_fn: Optional[Callable[[], dict]] = None
    meta: dict = field(default_factory=dict)


@dataclass
class TierOutcome:
    tier: str
    status: str
    detail: str = ""
    n_used: int = 0       # for Tier 1: how many stream items were consumed before deciding (early-kill win)


@dataclass
class CascadeResult:
    candidate: str
    outcomes: List[TierOutcome]
    promoted: bool
    certificate: Optional[dict]
    rungs_cost: int
    reached_tier: str

    @property
    def killed_at(self) -> Optional[str]:
        for o in self.outcomes:
            if o.status == KILL:
                return o.tier
        return None


def race_eprocess(outcomes: Sequence[float], theta: float, alpha: float = 0.05):
    """Anytime-valid race on a graded validation stream for H: true mean vs theta. Maintains TWO
    e-processes (Ville-valid under repeated peeking):
      * ep_good against H0: p <= theta      -> rejecting it = evidence p > theta (clearly STRONG)
      * ep_bad  against H0: (1-p) <= (1-theta) on flipped outcomes -> rejecting = evidence p < theta (HOPELESS)
    Returns (verdict, n_used) with verdict in {'promising','kill','inconclusive'}; n_used is the number of
    stream items consumed before a decision (the early-stop saving). Pure; no clock, no RNG."""
    if not outcomes:
        return "inconclusive", 0
    t = min(max(float(theta), 1e-6), 1.0 - 1e-6)
    ep_good = EProcess(theta=t, alpha=alpha)
    ep_bad = EProcess(theta=1.0 - t, alpha=alpha)
    for i, x in enumerate(outcomes, start=1):
        xv = min(max(float(x), 0.0), 1.0)
        g = ep_good.update(xv)
        b = ep_bad.update(1.0 - xv)
        if b["reject"]:
            return "kill", i           # anytime-significantly BELOW theta
        if g["reject"]:
            return "promising", i      # anytime-significantly ABOVE theta
    return "inconclusive", len(outcomes)


class VerificationCascade:
    """Run candidates through cheap->expensive tiers; only Tier 3 promotes. Tracks total rung cost so the
    'fewer expensive evaluations' benefit is a measured number, not a claim."""

    def __init__(self, theta: float, *, alpha: float = 0.05, surrogate_prune: float = 0.15,
                 min_val_n: int = 20, eprocess_alpha: Optional[float] = None):
        if not (0.0 < theta < 1.0):
            raise ValueError(f"theta must be in (0,1); got {theta}")
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.surrogate_prune = float(surrogate_prune)
        self.min_val_n = int(min_val_n)
        self.eprocess_alpha = float(eprocess_alpha if eprocess_alpha is not None else alpha)

    # -- the cheap tiers (allocation only; never promote) ------------------------------------------
    def _tier0(self, cand: Candidate) -> TierOutcome:
        if not cand.sanity_ok:
            return TierOutcome(TIER0, KILL, "sanity check failed (degenerate/invalid predictions)")
        if cand.val_outcomes is not None and len(cand.val_outcomes) == 0:
            return TierOutcome(TIER0, KILL, "empty validation stream")
        return TierOutcome(TIER0, ADVANCE, "sane")

    def _tier05(self, cand: Candidate) -> TierOutcome:
        if cand.surrogate_score is None:
            return TierOutcome(TIER05, ADVANCE, "no surrogate score (skipped)")
        if cand.surrogate_score < self.surrogate_prune:
            return TierOutcome(TIER05, KILL,
                               f"surrogate pass-prob {cand.surrogate_score:.3f} < prune {self.surrogate_prune}")
        return TierOutcome(TIER05, ADVANCE, f"surrogate pass-prob {cand.surrogate_score:.3f}")

    def _tier1(self, cand: Candidate) -> TierOutcome:
        if cand.val_outcomes is None:
            return TierOutcome(TIER1, ADVANCE, "no validation stream (skipped)")
        verdict, n_used = race_eprocess(cand.val_outcomes, self.theta, self.eprocess_alpha)
        if verdict == "kill":
            return TierOutcome(TIER1, KILL, "e-process anytime-significantly below theta", n_used)
        return TierOutcome(TIER1, ADVANCE, f"e-process verdict={verdict}", n_used)

    def _tier2(self, cand: Candidate) -> TierOutcome:
        if cand.val_outcomes is None:
            return TierOutcome(TIER2, ADVANCE, "no validation stream (skipped)")
        from vectorforge import science
        n = len(cand.val_outcomes)
        if n < self.min_val_n:
            return TierOutcome(TIER2, KILL, f"validation n={n} < min_val_n {self.min_val_n} (underpowered)")
        observed = sum(float(x) for x in cand.val_outcomes) / n
        cert = science.certify_accuracy(observed, n, self.theta, checks=1, alpha=self.alpha)
        if cert["certified"]:
            return TierOutcome(TIER2, ADVANCE,
                               f"val lower bound {cert['lower_bound']} clears theta {self.theta}")
        return TierOutcome(TIER2, KILL,
                           f"val lower bound {cert['lower_bound']} does not clear theta {self.theta}")

    def _tier3(self, cand: Candidate):
        if cand.certify_fn is None:
            return TierOutcome(TIER3, KILL, "no frozen certify_fn supplied"), None
        cert = cand.certify_fn()
        promoted = bool(cert.get("certified", False))
        return TierOutcome(TIER3, PROMOTE if promoted else KILL,
                           "frozen sealed certificate: "
                           + ("CERTIFIED" if promoted else "did not clear theta")), cert

    def cheap_screen(self, cand: Candidate) -> "CheapResult":
        """Run ONLY the cheap tiers (0, 0.5, 1, 2) -- never the expensive frozen Tier 3. Returns whether the
        candidate survived to become Tier-3-ELIGIBLE. This lets a search loop spend its scarce Tier-3
        verification wealth deliberately (select-then-bound), instead of certifying every leaf."""
        outcomes: List[TierOutcome] = []
        cost = 0
        for tier_fn, tier_name in ((self._tier0, TIER0), (self._tier05, TIER05),
                                   (self._tier1, TIER1), (self._tier2, TIER2)):
            cost += TIER_COST[tier_name]
            o = tier_fn(cand)
            outcomes.append(o)
            if o.status == KILL:
                return CheapResult(False, outcomes, cost, tier_name)
        return CheapResult(True, outcomes, cost, None)

    def certify_tier3(self, cand: Candidate) -> CascadeResult:
        """Spend the expensive frozen Tier-3 sealed certificate on a Tier-3-eligible candidate. THE ONLY
        promotion path. Callers must have screened with cheap_screen first (select-then-bound discipline)."""
        o3, cert = self._tier3(cand)
        res = CascadeResult(cand.name, [o3], o3.status == PROMOTE, cert, TIER_COST[TIER3], TIER3)
        assert_only_tier3_promotes(res)
        return res

    def evaluate(self, cand: Candidate) -> CascadeResult:
        outcomes: List[TierOutcome] = []
        cost = 0
        cert = None
        promoted = False
        for tier_fn, tier_name in ((self._tier0, TIER0), (self._tier05, TIER05),
                                   (self._tier1, TIER1), (self._tier2, TIER2)):
            cost += TIER_COST[tier_name]
            o = tier_fn(cand)
            outcomes.append(o)
            if o.status == KILL:
                return CascadeResult(cand.name, outcomes, False, None, cost, tier_name)
        # reached the expensive frozen tier
        cost += TIER_COST[TIER3]
        o3, cert = self._tier3(cand)
        outcomes.append(o3)
        promoted = (o3.status == PROMOTE)
        res = CascadeResult(cand.name, outcomes, promoted, cert, cost, TIER3)
        assert_only_tier3_promotes(res)
        return res

    def evaluate_batch(self, cands: Sequence[Candidate]) -> "BatchResult":
        results = [self.evaluate(c) for c in cands]
        reached_t3 = sum(1 for r in results if r.reached_tier == TIER3)
        promoted = [r for r in results if r.promoted]
        total_cost = sum(r.rungs_cost for r in results)
        # baseline cost: with NO cascade, every candidate pays Tier 2 + Tier 3 (full val bound + sealed cert)
        baseline_cost = len(cands) * (TIER_COST[TIER2] + TIER_COST[TIER3])
        return BatchResult(results=results, reached_tier3=reached_t3, n_promoted=len(promoted),
                           total_cost=total_cost, baseline_cost=baseline_cost,
                           promoted_names=[r.candidate for r in promoted])


@dataclass
class CheapResult:
    survived: bool
    outcomes: List[TierOutcome]
    cost: int
    killed_at: Optional[str]


@dataclass
class BatchResult:
    results: List[CascadeResult]
    reached_tier3: int
    n_promoted: int
    total_cost: int
    baseline_cost: int
    promoted_names: List[str]

    @property
    def early_kill_rate(self) -> float:
        n = len(self.results)
        if n == 0:
            return 0.0
        killed_cheap = sum(1 for r in self.results if r.reached_tier != TIER3)
        return killed_cheap / n

    @property
    def cost_saved_fraction(self) -> float:
        if self.baseline_cost == 0:
            return 0.0
        return max(0.0, 1.0 - self.total_cost / self.baseline_cost)


def assert_only_tier3_promotes(result: CascadeResult) -> None:
    """Invariant guard (D4): a promotion may ONLY come from Tier 3 (the frozen sealed certifier)."""
    if result.promoted:
        promoting = [o.tier for o in result.outcomes if o.status == PROMOTE]
        if promoting != [TIER3]:
            raise AssertionError(
                f"INVARIANT VIOLATION: promotion came from {promoting}, not exactly the frozen Tier 3. "
                "Cheap tiers and the surrogate may only allocate search, never promote.")


__all__ = ["Candidate", "TierOutcome", "CascadeResult", "BatchResult", "CheapResult", "VerificationCascade",
           "race_eprocess", "assert_only_tier3_promotes",
           "TIER0", "TIER05", "TIER1", "TIER2", "TIER3", "TIER_COST",
           "ADVANCE", "KILL", "PROMOTE"]
