"""End-to-end tests for the full Attestra pipeline (no LLM, deterministic only).

Tests the complete flow: profile -> design -> cycle -> certify -> registry -> strategy.
Runs on multiple datasets to verify generalization.
"""
import os
import tempfile

import numpy as np
import pytest
from sklearn.datasets import (
    load_iris, load_digits, load_wine, load_breast_cancer, load_diabetes,
)


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestEndToEndAttestra:
    """Full pipeline tests on real sklearn datasets."""

    def _run(self, X, y, goal, tmpdir, **kwargs):
        from attestra.orchestration.orchestrator import OrchestrateConfig, orchestrate
        config = OrchestrateConfig(
            goal=goal,
            X=X,
            y=y,
            max_rounds=kwargs.get("max_rounds", 3),
            time_budget_s=kwargs.get("time_budget_s", 60),
            verbose=False,
            registry_path=os.path.join(tmpdir, "registry.jsonl"),
            strategy_path=os.path.join(tmpdir, "strategy.jsonl"),
        )
        return orchestrate(config)

    def test_iris(self, tmpdir):
        data = load_iris()
        result = self._run(data.data, data.target, "classify iris flowers", tmpdir)
        assert result.decision in ("certified", "do_not_certify")
        assert result.best_score > 0.7
        assert result.profile.task_type in ("multiclass", "binary")
        assert result.plan is not None
        assert result.registry_updated
        assert result.strategy_updated

    def test_digits(self, tmpdir):
        data = load_digits()
        result = self._run(data.data, data.target, "classify handwritten digits", tmpdir)
        assert result.best_score > 0.7
        assert result.n_proposals > 0

    def test_wine(self, tmpdir):
        data = load_wine()
        result = self._run(data.data, data.target, "classify wine cultivars", tmpdir)
        assert result.best_score > 0.7

    def test_breast_cancer(self, tmpdir):
        data = load_breast_cancer()
        result = self._run(data.data, data.target, "classify breast cancer", tmpdir)
        assert result.best_score > 0.8

    def test_diabetes_regression(self, tmpdir):
        data = load_diabetes()
        result = self._run(data.data, data.target, "predict diabetes progression", tmpdir)
        assert result.n_proposals > 0
        # Regression is harder, just verify it runs

    def test_registry_compounds_across_runs(self, tmpdir):
        """Two runs on different datasets, registry tracks both."""
        from attestra.ledger.registry import ExperimentRegistry
        iris = load_iris()
        wine = load_wine()

        self._run(iris.data, iris.target, "classify iris", tmpdir)
        self._run(wine.data, wine.target, "classify wine", tmpdir)

        registry = ExperimentRegistry(os.path.join(tmpdir, "registry.jsonl"))
        assert registry.count() == 2
        summary = registry.summary()
        assert summary["total"] == 2
        assert summary["unique_datasets"] >= 1

    def test_strategy_learner_compounds(self, tmpdir):
        """Strategy learner tracks outcomes across runs."""
        from attestra.improvement.strategy_learner import StrategyLearner
        iris = load_iris()

        self._run(iris.data, iris.target, "classify iris", tmpdir, max_rounds=2)
        self._run(iris.data, iris.target, "classify iris again", tmpdir, max_rounds=2)

        learner = StrategyLearner(os.path.join(tmpdir, "strategy.jsonl"))
        summary = learner.summary()
        assert summary["total_outcomes"] > 0

    def test_small_dataset(self, tmpdir):
        """50 samples, verify system handles small data gracefully."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        y = (X[:, 0] > 0).astype(int)
        result = self._run(X, y, "binary on tiny data", tmpdir, max_rounds=2)
        assert result.n_proposals > 0
        assert result.plan is not None

    def test_high_dimensional(self, tmpdir):
        """100 samples, 200 features. System should handle gracefully."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 200))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        result = self._run(X, y, "binary on high-dim", tmpdir, max_rounds=2)
        assert result.n_proposals > 0

    def test_noisy_data(self, tmpdir):
        """Pure noise -- system should give honest failure."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 10))
        y = rng.integers(0, 2, 200)
        result = self._run(X, y, "classify pure noise", tmpdir, max_rounds=3)
        # Should still run, but may not certify
        assert result.n_proposals > 0
