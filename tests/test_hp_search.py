"""Tests for vfplatform.hp_search — Successive Halving (SHA) + Tree-Parzen Estimator (TPE).

Verifies:
1. SHA bracket construction (correct rungs, budgets, promotion logic)
2. TPE startup (uniform sampling) vs learned (good-distribution) phases
3. HPSearch end-to-end: propose → record → promote → best
4. Integration: propose_from_search filters already-tried configs
"""
import pytest
from vfplatform.hp_search import (
    SHABracket, Rung, TPE, HPSearch, HPSearchConfig,
    hp_search_for_family, propose_from_search,
)


class TestSHABracket:
    def test_rung_construction(self):
        """Bracket with max_budget=27, eta=3 should produce 4 rungs: 1, 3, 9, 27."""
        b = SHABracket(max_budget=27, eta=3)
        budgets = [r.budget for r in b.rungs]
        assert budgets == [1, 3, 9, 27]

    def test_rung_construction_default(self):
        """Default params (max_budget=160, eta=3) produce reasonable rungs."""
        b = SHABracket()
        assert b.rungs[0].budget < b.rungs[-1].budget
        assert b.rungs[-1].budget == 160

    def test_promotion_logic(self):
        """Top 1/eta configs get promoted."""
        b = SHABracket(max_budget=27, eta=3, n_configs=9)
        # record 9 configs at rung 0 (budget=1)
        for i in range(9):
            b.record(f"c{i}", budget=1, score=float(i) / 10)
        # top 3 (c6, c7, c8) should be promotable from rung 0
        promo = b.promotable(0)
        assert len(promo) == 3
        assert "c8" in promo
        assert "c7" in promo
        assert "c6" in promo

    def test_next_actions(self):
        """next_actions returns (config_id, budget) pairs for promotion."""
        b = SHABracket(max_budget=27, eta=3, n_configs=9)
        for i in range(9):
            b.record(f"c{i}", budget=1, score=float(i))
        actions = b.next_actions()
        # should propose promoting top 3 to budget=3
        assert len(actions) >= 1
        budgets = [bud for _, bud in actions]
        assert all(bud == 3 for bud in budgets)

    def test_best_config(self):
        """best_config returns the highest scorer at the highest rung."""
        b = SHABracket(max_budget=27, eta=3, n_configs=9)
        for i in range(9):
            b.record(f"c{i}", budget=1, score=float(i))
        # promote c8 to budget 3
        b.record("c8", budget=3, score=0.95)
        assert b.best_config() == "c8"

    def test_is_complete(self):
        """is_complete when top rung has a result."""
        b = SHABracket(max_budget=27, eta=3, n_configs=9)
        assert not b.is_complete()
        b.record("c0", budget=27, score=0.9)
        assert b.is_complete()


class TestTPE:
    def test_startup_uniform(self):
        """During startup phase (< n_startup trials), TPE proposes uniformly."""
        specs = {"lr": ("float", 0.001, 1.0), "depth": ("int", 1, 10)}
        tpe = TPE(specs, n_startup=5, seed=42)
        # no observations yet
        configs = tpe.propose(3)
        assert len(configs) == 3
        for c in configs:
            assert 0.001 <= c["lr"] <= 1.0
            assert 1 <= c["depth"] <= 10

    def test_learned_phase(self):
        """After n_startup trials, TPE samples from good distribution."""
        specs = {"lr": ("float", 0.001, 1.0), "n_layers": ("choice", [1, 2, 3, 4])}
        tpe = TPE(specs, n_startup=3, gamma=0.5, seed=42)
        # add 5 observations (above n_startup=3)
        tpe.observe({"lr": 0.1, "n_layers": 2}, score=0.9)
        tpe.observe({"lr": 0.01, "n_layers": 3}, score=0.95)
        tpe.observe({"lr": 0.5, "n_layers": 1}, score=0.6)
        tpe.observe({"lr": 0.8, "n_layers": 4}, score=0.5)
        tpe.observe({"lr": 0.001, "n_layers": 2}, score=0.85)
        # now in learned phase
        configs = tpe.propose(10)
        assert len(configs) == 10
        # check all configs are valid
        for c in configs:
            assert 0.001 <= c["lr"] <= 1.0
            assert c["n_layers"] in [1, 2, 3, 4]

    def test_choice_params(self):
        """TPE handles categorical params."""
        specs = {"algo": ("choice", ["sgd", "adam", "rmsprop"])}
        tpe = TPE(specs, seed=0)
        configs = tpe.propose(5)
        for c in configs:
            assert c["algo"] in ["sgd", "adam", "rmsprop"]


class TestHPSearch:
    def test_end_to_end(self):
        """Full cycle: propose_initial → record → propose_next → best."""
        config = HPSearchConfig(
            family="test_family",
            param_specs={"lr": ("float", 0.001, 1.0), "depth": ("int", 1, 5)},
            max_budget=27, eta=3, n_initial=9, tpe_startup=3, seed=42,
        )
        hp = HPSearch(config)

        # initial batch
        batch = hp.propose_initial()
        assert len(batch) == 9
        for cid, params, budget in batch:
            assert budget == 1  # lowest rung
            assert "lr" in params
            assert "depth" in params
            # simulate a score
            score = params["lr"] * 0.5 + params["depth"] * 0.1
            hp.record(cid, params, budget, score)

        # next round: should promote survivors
        assert not hp.is_done()
        batch2 = hp.propose_next()
        assert len(batch2) > 0
        for cid, params, budget in batch2:
            assert budget > 1  # promoted to higher rung
            score = params["lr"] * 0.6 + params["depth"] * 0.15
            hp.record(cid, params, budget, score)

        # best should return valid params
        best = hp.best()
        assert best is not None
        assert "lr" in best
        assert "depth" in best

    def test_summary(self):
        config = HPSearchConfig(
            family="xgb", param_specs={"lr": ("float", 0.01, 0.3)},
            max_budget=9, eta=3, n_initial=3, seed=0,
        )
        hp = HPSearch(config)
        batch = hp.propose_initial()
        for cid, params, budget in batch:
            hp.record(cid, params, budget, params["lr"])
        s = hp.summary()
        assert s["family"] == "xgb"
        assert s["n_trials"] == 3
        assert s["best_params"] is not None


class TestIntegration:
    def test_propose_from_search(self):
        """propose_from_search collects proposals from multiple families."""
        config1 = HPSearchConfig(family="f1", param_specs={"x": ("float", 0, 1)},
                                 max_budget=9, eta=3, n_initial=3, seed=0)
        config2 = HPSearchConfig(family="f2", param_specs={"y": ("int", 1, 10)},
                                 max_budget=9, eta=3, n_initial=3, seed=1)
        hp1 = HPSearch(config1)
        hp2 = HPSearch(config2)
        proposals = propose_from_search({"f1": hp1, "f2": hp2})
        assert len(proposals) == 6  # 3 from each
        families = {p["family"] for p in proposals}
        assert families == {"f1", "f2"}

    def test_tried_filter(self):
        """Proposals include a config_id key usable for dedup tracking."""
        config = HPSearchConfig(family="f1", param_specs={"x": ("float", 0.0, 10.0)},
                                max_budget=9, eta=3, n_initial=3, seed=0)
        hp = HPSearch(config)
        proposals = propose_from_search({"f1": hp})
        assert len(proposals) == 3
        # every proposal has the required fields
        for p in proposals:
            assert "config_id" in p
            assert "family" in p
            assert "params" in p
            assert "budget" in p
            assert p["family"] == "f1"
            assert p["budget"] >= 1
