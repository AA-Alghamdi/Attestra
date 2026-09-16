"""Hermetic locks for the #6 LLM strategist (vfplatform/llm_strategist.py) -- the SAFETY contract, no network.

A strategist can only reorder/prune the legal proposals; it can never weaken a certificate. These tests use
FAKE strategists (no API) over the SAME synthetic FakeArena + REAL frozen certifier as test_repr_researcher, and
lock four properties:

  (1) IDENTITY: the deterministic strategist reproduces the bare ReprResearcher's champion + peek count exactly.
  (2) PRUNING SAVES PEEKS WITHOUT CHANGING THE VERDICT: a strategist that proposes ONLY the genuine winner
      reaches the same champion in strictly fewer sealed peeks (it skips the validation lure's peek).
  (3) HALLUCINATIONS ARE NEUTRALISED: a strategist that proposes illegal/garbage ids is intersected down to the
      legal set, so the champion + promotions are identical to deterministic -- it cannot inject a candidate the
      policy would not allow.
  (4) EMPTY PRUNE NEVER BLOCKS: a strategist that prunes everything falls back to the full legal set (the climb
      still finds the champion).
"""
import hashlib
import os

from vfplatform.llm_strategist import StrategistResearcher, Strategist
from vfplatform.repr_researcher import ReprResearcher

from test_repr_researcher import FakeArena, REGISTRY, FROZEN_EXPECTED  # reuse the synthetic arena + registry

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FixedStrategist(Strategist):
    """Returns a fixed list of (tag, partner) proposals regardless of state (filtered to legal by the researcher)."""
    def __init__(self, proposals):
        self._p = list(proposals)

    def rank(self, move_class, champion, tried, registry, legal):
        return list(self._p)


class _DropStrategist(Strategist):
    """Keeps the legal proposals in order but PRUNES any whose tag/partner is in `drop` -- the realistic
    strategist move (skip an encoder it predicts will lose, to save its sealed peek)."""
    def __init__(self, drop):
        self._drop = set(drop)

    def rank(self, move_class, champion, tried, registry, legal):
        return [(t, p) for (t, p) in legal if t not in self._drop and p not in self._drop]


def _det(arena, registry=REGISTRY):
    return ReprResearcher(registry, arena, start_tag="weak", alpha=0.1, theta_floor=0.5,
                          peek_budget=12, competence_ceiling=0.90)


def _strat(arena, strategist, registry=REGISTRY):
    return StrategistResearcher(registry, arena, start_tag="weak", alpha=0.1, theta_floor=0.5,
                                peek_budget=12, competence_ceiling=0.90, strategist=strategist)


def test_frozen_certifier_unchanged():
    got = {rel: hashlib.sha256(open(os.path.join(_ROOT, rel), "rb").read()).hexdigest()[:8]
           for rel in FROZEN_EXPECTED}
    assert got == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED: {got} != {FROZEN_EXPECTED}"


def test_deterministic_strategist_is_identity():
    base = _det(FakeArena()).run()
    strat = _strat(FakeArena(), Strategist()).run()
    assert strat.champion == base.champion == "strong"
    assert strat.peeks_used == base.peeks_used
    assert [p.to_tag for p in strat.promotions] == [p.to_tag for p in base.promotions]


def test_pruning_reaches_same_champion_in_fewer_peeks():
    """Pruning the validation lure (predicted loser) skips its sealed peek -> same champion, fewer peeks."""
    base = _det(FakeArena()).run()
    strat = _strat(FakeArena(), _DropStrategist({"lure"})).run()
    assert strat.champion == base.champion == "strong"
    assert strat.peeks_used < base.peeks_used      # the lure's wasted peek (and its fusion) is saved


def test_hallucinated_illegal_proposals_are_neutralised():
    """Garbage/illegal ids are intersected away; the climb is identical to deterministic (cannot inject them)."""
    bogus = _FixedStrategist([("does_not_exist", None), ("weak", "ghost"), ("strong", "phantom")])
    strat = _strat(FakeArena(), bogus).run()
    base = _det(FakeArena()).run()
    assert strat.champion == base.champion == "strong"
    assert [p.to_tag for p in strat.promotions] == [p.to_tag for p in base.promotions]


def test_deliberate_empty_prune_skips_the_rung():
    """A DELIBERATE empty prune skips the rung (the strategist's 'none worth a peek' move). Skipping the
    family rung forfeits the only path to the winner -> champion stays 'weak'. This is an honest negative the
    head-to-head catches; it can never mint a false certificate (the frozen gate still rules)."""
    base = _det(FakeArena()).run()
    strat = _strat(FakeArena(), _FixedStrategist([])).run()
    assert base.champion == "strong"
    assert strat.champion == "weak"                # rungs skipped -> no climb (but no false promotion either)
    assert strat.peeks_used == 0                   # and zero sealed peeks were spent
