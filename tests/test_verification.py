"""Tests for the cost-stratified verification cascade.

Acceptance: the cascade early-kills hopeless candidates at cheap tiers, total rung-cost drops vs a
no-cascade baseline, only Tier 3 promotes, and the certified outcome is UNCHANGED (the strong candidate
still promotes; the weak ones still do not)."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import verification as V


def _stream(p, n, seed):
    rng = random.Random(seed)
    return [1.0 if rng.random() < p else 0.0 for _ in range(n)]


def test_race_kills_hopeless_and_flags_strong():
    # clearly below theta -> kill, and it stops EARLY (well before consuming the whole stream)
    verdict, n_used = V.race_eprocess(_stream(0.40, 400, 1), theta=0.80, alpha=0.05)
    assert verdict == "kill" and n_used < 400
    # clearly above theta -> promising
    verdict2, _ = V.race_eprocess(_stream(0.97, 400, 2), theta=0.80, alpha=0.05)
    assert verdict2 == "promising"


def test_tier0_kills_degenerate():
    c = V.VerificationCascade(theta=0.8)
    res = c.evaluate(V.Candidate(name="bad", sanity_ok=False))
    assert res.promoted is False and res.killed_at == V.TIER0


def test_surrogate_prunes_but_never_promotes():
    c = V.VerificationCascade(theta=0.8, surrogate_prune=0.2)
    res = c.evaluate(V.Candidate(name="lowscore", surrogate_score=0.05,
                                 val_outcomes=_stream(0.99, 200, 3),
                                 certify_fn=lambda: {"certified": True}))
    assert res.killed_at == V.TIER05 and res.promoted is False


def test_only_tier3_promotes():
    c = V.VerificationCascade(theta=0.8)
    # strong candidate with a certify_fn that certifies
    res = c.evaluate(V.Candidate(name="strong", val_outcomes=_stream(0.95, 300, 4),
                                 certify_fn=lambda: {"certified": True, "lower_bound": 0.9}))
    assert res.promoted is True and res.reached_tier == V.TIER3
    # the promotion came only from tier3
    V.assert_only_tier3_promotes(res)


def test_cascade_reduces_cost_and_preserves_outcome():
    theta = 0.80
    c = V.VerificationCascade(theta=theta)
    cands = []
    # 1 strong candidate (true p=0.95) that should certify; 19 hopeless (true p=0.45) that should not.
    strong = V.Candidate(name="strong", val_outcomes=_stream(0.95, 400, 100),
                         certify_fn=lambda: {"certified": True, "lower_bound": 0.9})
    cands.append(strong)
    for i in range(19):
        cands.append(V.Candidate(name=f"weak{i}", val_outcomes=_stream(0.45, 400, 200 + i),
                                 certify_fn=lambda: {"certified": False, "lower_bound": 0.4}))
    batch = c.evaluate_batch(cands)
    # only the strong one reaches tier3 and promotes; the weak ones are killed cheaply
    assert batch.n_promoted == 1 and batch.promoted_names == ["strong"]
    assert batch.reached_tier3 <= 2          # weak ones pruned before the expensive tier
    assert batch.early_kill_rate >= 0.80
    assert batch.cost_saved_fraction > 0.5   # large measured cost reduction vs no-cascade baseline
    # CERTIFIED OUTCOME UNCHANGED: certify everyone directly (the no-cascade reference) -> same winner set
    direct_promoted = [c2.name for c2 in cands if c2.certify_fn()["certified"]]
    assert direct_promoted == batch.promoted_names


def test_tier2_kills_underpowered_and_below_bar():
    c = V.VerificationCascade(theta=0.8, min_val_n=50)
    # too few examples -> killed at tier2 as underpowered
    res = c.evaluate(V.Candidate(name="tiny", val_outcomes=_stream(0.99, 10, 5),
                                 certify_fn=lambda: {"certified": True}))
    assert res.killed_at == V.TIER2
    # enough examples but genuinely below theta -> killed at tier2 (val bound doesn't clear)
    res2 = c.evaluate(V.Candidate(name="midbar", val_outcomes=_stream(0.70, 400, 6),
                                  certify_fn=lambda: {"certified": True}))
    assert res2.killed_at in (V.TIER1, V.TIER2) and res2.promoted is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
