"""Tests for the new attestra/ package -- core, intake, design, cycle, orchestration."""
import json
import os
import tempfile

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes


# ============================================================================== core/science migration

class TestCoreScienceMigration:
    """Verify science.py is accessible from the new attestra.core path."""

    def test_import_certify_accuracy(self):
        from attestra.core.science import certify_accuracy
        cert = certify_accuracy(0.95, 100, 0.50, checks=1)
        assert cert["certified"] is True

    def test_import_score_metric(self):
        from attestra.core.science import score_metric
        score = score_metric("accuracy", [1, 1, 0, 0], [1, 1, 0, 0], [0, 1])
        assert score == 1.0

    def test_import_via_init(self):
        from attestra.core import certify_accuracy, score_metric
        assert callable(certify_accuracy)
        assert callable(score_metric)

    def test_certify_regression(self):
        from attestra.core.science import certify_regression
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1] * 20
        cert = certify_regression(y_true, y_pred, "r2", 0.5)
        assert "lower_bound" in cert


# ============================================================================== intake/profiler

class TestDataProfiler:
    def test_profile_iris(self):
        from attestra.intake.profiler import profile_data
        data = load_iris()
        profile = profile_data(data.data, data.target)
        assert profile.n_samples == 150
        assert profile.n_features == 4
        assert profile.task_type in ("multiclass", "binary")
        assert profile.n_classes == 3
        assert profile.quality_score > 0
        assert len(profile.features) == 4

    def test_profile_regression(self):
        from attestra.intake.profiler import profile_data
        data = load_diabetes()
        profile = profile_data(data.data, data.target)
        assert profile.task_type == "regression"
        assert profile.n_classes == 0
        assert "mean" in profile.target_stats

    def test_profile_to_llm_context(self):
        from attestra.intake.profiler import profile_data
        data = load_iris()
        profile = profile_data(data.data, data.target)
        ctx = profile.to_llm_context()
        assert "150 samples" in ctx
        assert "4 features" in ctx

    def test_profile_detects_constant_features(self):
        from attestra.intake.profiler import profile_data
        X = np.column_stack([np.ones(100), np.random.randn(100, 3)])
        y = np.random.randint(0, 2, 100)
        profile = profile_data(X, y)
        assert profile.n_constant_features >= 1
        assert any("constant" in issue.lower() for issue in profile.issues)

    def test_profile_detects_missing(self):
        from attestra.intake.profiler import profile_data
        X = np.random.randn(100, 4)
        X[::3, 0] = np.nan
        y = np.random.randint(0, 2, 100)
        profile = profile_data(X, y)
        assert profile.missing_rate > 0


# ============================================================================== design/experiment_plan

class TestExperimentDesign:
    def test_design_classification(self):
        from attestra.intake.profiler import profile_data
        from attestra.design.experiment_plan import design_experiment
        data = load_iris()
        profile = profile_data(data.data, data.target)
        plan = design_experiment("classify iris flowers", profile)
        assert plan.goal == "classify iris flowers"
        assert plan.verification.metric in ("accuracy", "balanced_accuracy")
        assert plan.verification.threshold > 0
        assert len(plan.surface.model_families) > 0
        assert plan.budget.max_rounds > 0
        assert len(plan.plan_hash) == 16
        assert plan.power.n_test > 0

    def test_design_regression(self):
        from attestra.intake.profiler import profile_data
        from attestra.design.experiment_plan import design_experiment
        data = load_diabetes()
        profile = profile_data(data.data, data.target)
        plan = design_experiment("predict diabetes progression", profile)
        assert plan.verification.metric == "r2"
        assert plan.verification.threshold > 0

    def test_plan_content_hash_deterministic(self):
        from attestra.intake.profiler import profile_data
        from attestra.design.experiment_plan import design_experiment
        data = load_iris()
        profile = profile_data(data.data, data.target)
        plan1 = design_experiment("test", profile)
        plan2 = design_experiment("test", profile)
        assert plan1.plan_hash == plan2.plan_hash

    def test_plan_to_llm_context(self):
        from attestra.intake.profiler import profile_data
        from attestra.design.experiment_plan import design_experiment
        data = load_iris()
        profile = profile_data(data.data, data.target)
        plan = design_experiment("classify iris", profile)
        ctx = plan.to_llm_context()
        assert "EXPERIMENT PLAN" in ctx
        assert "VERIFICATION" in ctx
        assert "BUDGET" in ctx


# ============================================================================== orchestration/error_taxonomy

class TestErrorTaxonomy:
    def test_classify_oom(self):
        from attestra.orchestration.error_taxonomy import classify_error, ErrorCategory
        err = MemoryError("Out of memory")
        classified = classify_error(err)
        assert classified.category == ErrorCategory.INFRASTRUCTURE
        assert not classified.is_fatal
        assert "reduce_batch_size" in classified.recovery_actions

    def test_classify_syntax_error(self):
        from attestra.orchestration.error_taxonomy import classify_error, ErrorCategory
        err = SyntaxError("invalid syntax")
        classified = classify_error(err)
        assert classified.category == ErrorCategory.CODE
        assert "regenerate_code" in classified.recovery_actions

    def test_classify_nan(self):
        from attestra.orchestration.error_taxonomy import classify_error_string, ErrorCategory
        classified = classify_error_string("Input contains NaN")
        assert classified.category == ErrorCategory.DATA

    def test_classify_convergence(self):
        from attestra.orchestration.error_taxonomy import classify_error_string, ErrorCategory
        classified = classify_error_string("ConvergenceWarning: Solver did not converge")
        assert classified.category == ErrorCategory.NUMERICAL

    def test_error_tracker(self):
        from attestra.orchestration.error_taxonomy import (
            ErrorTracker, classify_error, ErrorCategory
        )
        tracker = ErrorTracker()
        tracker.record(classify_error(MemoryError("OOM")))
        tracker.record(classify_error(SyntaxError("bad code")))
        tracker.record(classify_error(MemoryError("OOM again")))
        assert tracker.total == 3
        assert tracker.most_common_category() == ErrorCategory.INFRASTRUCTURE
        assert len(tracker.recovery_suggestions()) > 0


# ============================================================================== orchestration/health

class TestHealthMonitor:
    def test_healthy_run(self):
        from attestra.orchestration.health import HealthMonitor, HealthStatus
        monitor = HealthMonitor(time_budget_s=100)
        monitor.observe_round(0, 0.8, 5.0)
        monitor.observe_round(1, 0.85, 5.0)
        monitor.observe_round(2, 0.88, 5.0)
        report = monitor.check()
        assert report.overall == HealthStatus.HEALTHY
        assert report.should_continue

    def test_stalled_detection(self):
        from attestra.orchestration.health import HealthMonitor, HealthStatus
        monitor = HealthMonitor(time_budget_s=1000, stall_patience=3)
        for i in range(6):
            monitor.observe_round(i, 0.8, 5.0)  # same score every round
        report = monitor.check()
        assert report.overall in (HealthStatus.STALLED, HealthStatus.WARNING)

    def test_high_error_rate(self):
        from attestra.orchestration.health import HealthMonitor, HealthStatus
        monitor = HealthMonitor(time_budget_s=1000, max_error_rate=0.5)
        for i in range(5):
            monitor.observe_round(i, None, 5.0, error=True)
        report = monitor.check()
        assert report.overall == HealthStatus.CRITICAL


# ============================================================================== ledger/registry

class TestExperimentRegistry:
    def test_record_and_read(self):
        from attestra.ledger.registry import ExperimentRegistry, ExperimentRecord
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ExperimentRegistry(os.path.join(tmpdir, "test.jsonl"))
            record = ExperimentRecord(
                plan_hash="abc123",
                goal="test",
                dataset_fingerprint="fp1",
                metric="accuracy",
                threshold=0.5,
                decision="certified",
                best_score=0.95,
                best_technique="hist_gbm",
            )
            reg.record(record)
            records = reg.all_records()
            assert len(records) == 1
            assert records[0].is_positive
            assert records[0].best_score == 0.95

    def test_positive_and_negative(self):
        from attestra.ledger.registry import ExperimentRegistry, ExperimentRecord
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ExperimentRegistry(os.path.join(tmpdir, "test.jsonl"))
            reg.record(ExperimentRecord(
                plan_hash="a", goal="g", dataset_fingerprint="fp",
                metric="accuracy", threshold=0.5,
                decision="certified", best_score=0.9, best_technique="rf",
            ))
            reg.record(ExperimentRecord(
                plan_hash="b", goal="g", dataset_fingerprint="fp",
                metric="accuracy", threshold=0.5,
                decision="do_not_certify", best_score=0.4, best_technique="lr",
            ))
            assert len(reg.positive_certificates()) == 1
            assert len(reg.negative_certificates()) == 1

    def test_fingerprint(self):
        from attestra.ledger.registry import fingerprint_dataset
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        fp1 = fingerprint_dataset(X, y)
        fp2 = fingerprint_dataset(X, y)
        assert fp1 == fp2
        assert len(fp1) == 16


# ============================================================================== improvement/strategy_learner

class TestStrategyLearner:
    def test_record_and_rank(self):
        from attestra.improvement.strategy_learner import StrategyLearner, StrategyOutcome
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = StrategyLearner(os.path.join(tmpdir, "strat.jsonl"))
            learner.record(StrategyOutcome(
                strategy_name="hist_gbm", source="catalog",
                task_type="binary", n_samples=100, n_features=10,
                success=True, score=0.95,
            ))
            learner.record(StrategyOutcome(
                strategy_name="lr", source="catalog",
                task_type="binary", n_samples=100, n_features=10,
                success=True, score=0.85,
            ))
            ranked = learner.rank_strategies("binary")
            assert len(ranked) == 2
            assert ranked[0][0] == "hist_gbm"

    def test_success_rate(self):
        from attestra.improvement.strategy_learner import StrategyLearner, StrategyOutcome
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = StrategyLearner(os.path.join(tmpdir, "strat.jsonl"))
            for i in range(10):
                learner.record(StrategyOutcome(
                    strategy_name="hist_gbm", source="catalog",
                    task_type="binary", n_samples=100, n_features=10,
                    success=(i < 8), score=0.9 if i < 8 else None,
                ))
            rate = learner.success_rate("hist_gbm")
            assert abs(rate - 0.8) < 0.01


# ============================================================================== cycle/proposals/catalog

class TestCatalogProposals:
    def test_classification_proposals(self):
        from attestra.intake.profiler import profile_data
        from attestra.cycle.proposals.catalog import propose_from_catalog
        data = load_iris()
        profile = profile_data(data.data, data.target)
        proposals = propose_from_catalog(profile)
        assert len(proposals) > 0
        names = {p.name for p in proposals}
        assert "hist_gbm_tuned" in names

    def test_regression_proposals(self):
        from attestra.intake.profiler import profile_data
        from attestra.cycle.proposals.catalog import propose_from_catalog
        data = load_diabetes()
        profile = profile_data(data.data, data.target)
        proposals = propose_from_catalog(profile)
        assert len(proposals) > 0
        names = {p.name for p in proposals}
        assert "hist_gbm_reg_tuned" in names

    def test_proposals_build_and_fit(self):
        from attestra.intake.profiler import profile_data
        from attestra.cycle.proposals.catalog import propose_from_catalog
        data = load_iris()
        profile = profile_data(data.data, data.target)
        proposals = propose_from_catalog(profile, max_proposals=2)
        for p in proposals:
            model = p.build_fn(42)
            model.fit(data.data[:100], data.target[:100])
            preds = model.predict(data.data[100:])
            assert len(preds) == 50


# ============================================================================== cycle/engine

class TestResearchEngine:
    def test_iris_no_llm(self):
        from attestra.cycle.engine import ResearchEngine
        data = load_iris()
        engine = ResearchEngine(
            data.data, data.target,
            goal="classify iris flowers",
            max_rounds=3,
            time_budget_s=60,
            verbose=False,
        )
        result = engine.run()
        assert result.decision in ("certified", "do_not_certify", "honest_stop")
        assert result.best_score > 0.5
        assert result.n_proposals > 0
        assert result.plan is not None
        assert result.profile is not None

    def test_diabetes_no_llm(self):
        from attestra.cycle.engine import ResearchEngine
        data = load_diabetes()
        engine = ResearchEngine(
            data.data, data.target,
            goal="predict diabetes",
            max_rounds=3,
            time_budget_s=60,
            verbose=False,
        )
        result = engine.run()
        assert result.decision in ("certified", "do_not_certify", "honest_stop")
        assert result.n_proposals > 0

    def test_error_summary_populated(self):
        from attestra.cycle.engine import ResearchEngine
        data = load_iris()
        engine = ResearchEngine(
            data.data, data.target,
            goal="classify iris",
            max_rounds=2,
            time_budget_s=30,
            verbose=False,
        )
        result = engine.run()
        assert result.error_summary is not None
        assert "total_errors" in result.error_summary


# ============================================================================== orchestration/orchestrator

class TestOrchestrator:
    def test_full_orchestrate(self):
        from attestra.orchestration.orchestrator import OrchestrateConfig, orchestrate
        data = load_iris()
        config = OrchestrateConfig(
            goal="classify iris",
            X=data.data,
            y=data.target,
            max_rounds=3,
            time_budget_s=60,
            verbose=False,
        )
        result = orchestrate(config)
        assert result.decision in ("certified", "do_not_certify", "honest_stop", "error")
        assert result.best_score > 0
        assert result.registry_updated
        assert result.strategy_updated

    def test_orchestrate_from_text(self):
        from attestra.orchestration.orchestrator import orchestrate_from_text
        data = load_iris()
        result = orchestrate_from_text(
            "classify iris",
            (data.data, data.target),
            time_budget_s=30,
            max_rounds=2,
            verbose=False,
        )
        assert result.decision in ("certified", "do_not_certify", "honest_stop", "error")


# ============================================================================== adapters

class TestAdapters:
    def test_tabular_adapter(self):
        from attestra.adapters.base import TabularAdapter
        adapter = TabularAdapter()
        assert adapter.modality() == "tabular"
        assert adapter.default_metric() == "accuracy"
        assert len(adapter.recommended_families()) > 0

    def test_text_adapter(self):
        from attestra.adapters.base import TextClassificationAdapter
        adapter = TextClassificationAdapter()
        assert adapter.modality() == "text"
        assert adapter.default_metric() == "macro_f1"

    def test_adapter_registry(self):
        from attestra.adapters.base import get_adapter, available_modalities
        assert "tabular" in available_modalities()
        adapter = get_adapter("tabular")
        assert adapter.modality() == "tabular"

    def test_text_featurize(self):
        from attestra.adapters.base import TextClassificationAdapter
        adapter = TextClassificationAdapter()
        texts = ["hello world", "good morning", "bad day", "great news"] * 5
        labels = [1, 1, 0, 1] * 5
        X, y = adapter.featurize((texts, labels))
        assert X.shape[0] == 20
        assert len(y) == 20


# ============================================================================== CLI

class TestCLI:
    def test_cli_status(self):
        from attestra.cli.main import main
        result = main(["status"])
        assert result == 0

    def test_cli_run_iris(self):
        from attestra.cli.main import main
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            outpath = f.name
        try:
            result = main([
                "run", "--goal", "classify iris",
                "--data", "iris",
                "--rounds", "2", "--time", "30",
                "--no-llm", "--quiet",
                "--output", outpath,
            ])
            assert result in (0, 1)
            with open(outpath) as f:
                data = json.loads(f.read())
            assert "decision" in data
            assert "best_score" in data
        finally:
            os.unlink(outpath)
