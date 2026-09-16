"""Tests for all 10 gap implementations.

Covers: problem typing, portfolio, torch harness, checkpoint/resume,
FDR control, data augmentation, tradeoff engine, architecture builder,
literature search, experiment management, meta-learner.
"""
import json
import os
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge


# ============================================================================== Gap 1: Problem Typing

class TestProblemTyping:
    def test_heuristic_classification(self):
        from attestra.intake.problem_typing import type_problem, ProblemDomain
        spec = type_problem("Classify images into 10 categories", data_profile={
            "task_type": "multiclass", "n_classes": 10, "n_samples": 1000, "n_features": 784,
        })
        assert spec.domain in (ProblemDomain.TABULAR_CLASSIFICATION, ProblemDomain.IMAGE_CLASSIFICATION)
        assert spec.confidence > 0
        assert spec.spec_hash != ""

    def test_heuristic_regression(self):
        from attestra.intake.problem_typing import type_problem, ProblemDomain
        spec = type_problem("Predict housing prices", data_profile={
            "task_type": "regression", "n_samples": 20000, "n_features": 8,
        })
        assert spec.domain == ProblemDomain.TABULAR_REGRESSION
        assert "r2" in spec.suggested_metric or "neg_rmse" in spec.suggested_metric

    def test_heuristic_tts(self):
        from attestra.intake.problem_typing import type_problem, ProblemDomain
        spec = type_problem("Build an expressive TTS model for Arabic")
        assert spec.domain == ProblemDomain.SPEECH_SYNTHESIS

    def test_adversarial_check_small_data(self):
        from attestra.intake.problem_typing import adversarial_check
        check = adversarial_check("Achieve 99% accuracy", data_profile={
            "task_type": "binary", "n_samples": 20, "n_features": 100,
        })
        assert check.has_issues
        assert any("few" in s.lower() or "sample" in s.lower() for s in check.infeasibilities)

    def test_adversarial_check_p_gt_n(self):
        from attestra.intake.problem_typing import adversarial_check, RiskLevel
        check = adversarial_check("Classify accurately", data_profile={
            "task_type": "binary", "n_samples": 50, "n_features": 200,
        })
        assert check.overall_risk in (RiskLevel.MEDIUM, RiskLevel.HIGH)
        assert any("overfit" in r.lower() for r in check.reward_hacking_risks)

    def test_content_addressing(self):
        from attestra.intake.problem_typing import type_problem
        spec1 = type_problem("Classify iris", data_profile={"task_type": "multiclass", "n_classes": 3, "n_samples": 150, "n_features": 4})
        spec2 = type_problem("Classify iris", data_profile={"task_type": "multiclass", "n_classes": 3, "n_samples": 150, "n_features": 4})
        assert spec1.spec_hash == spec2.spec_hash


# ============================================================================== Gap 2: Portfolio

class TestPortfolio:
    def test_basic_execution(self):
        from attestra.orchestration.portfolio import Portfolio, ArmResult
        p = Portfolio(max_workers=2, time_budget_s=10)
        p.add_arm("fast", lambda: ArmResult(name="fast", score=0.8, success=True))
        p.add_arm("slow", lambda: ArmResult(name="slow", score=0.9, success=True))
        result = p.execute(max_rounds=2)
        assert result.best_score >= 0.8
        assert result.arms_succeeded >= 1

    def test_handles_failure(self):
        from attestra.orchestration.portfolio import Portfolio, ArmResult

        def fail_arm():
            raise ValueError("intentional")

        p = Portfolio(max_workers=2, time_budget_s=5)
        p.add_arm("good", lambda: ArmResult(name="good", score=0.7, success=True))
        p.add_arm("bad", fail_arm)
        result = p.execute(max_rounds=1)
        assert result.best_score >= 0.7
        assert result.arms_succeeded >= 1

    def test_thompson_sampling(self):
        from attestra.orchestration.portfolio import Portfolio, ArmResult
        p = Portfolio(max_workers=2)
        # One arm always succeeds, one always fails
        p.add_arm("winner", lambda: ArmResult(name="winner", score=0.95, success=True))
        p.add_arm("loser", lambda: ArmResult(name="loser", score=0.1, success=True))
        result = p.execute(max_rounds=3, top_k=2)
        assert result.best_arm == "winner"


# ============================================================================== Gap 3: Torch Harness

class TestTorchHarness:
    @pytest.fixture
    def has_torch(self):
        try:
            import torch
            return True
        except ImportError:
            pytest.skip("torch not installed")

    def test_detect_device(self, has_torch):
        from attestra.execution.torch_harness import detect_device
        device = detect_device()
        assert device in ("cpu", "cuda", "mps")

    def test_train_mlp(self, has_torch):
        from attestra.execution.torch_harness import TorchHarness, TrainConfig
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X = X.astype(np.float32)
        config = TrainConfig(epochs=20, batch_size=32, patience=5, verbose=False, device="cpu")
        harness = TorchHarness(config)
        harness.build_model(input_dim=10, output_dim=2, task="classification")
        result = harness.train(X[:150], y[:150], X[150:], y[150:])
        assert result.total_epochs > 0
        assert result.val_metrics["accuracy"] > 0.4

    def test_predict(self, has_torch):
        from attestra.execution.torch_harness import TorchHarness, TrainConfig
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X = X.astype(np.float32)
        config = TrainConfig(epochs=10, batch_size=32, patience=5, verbose=False, device="cpu")
        harness = TorchHarness(config)
        harness.build_model(input_dim=5, output_dim=2, task="classification")
        harness.train(X[:80], y[:80], X[80:], y[80:])
        preds = harness.predict(X[80:])
        assert len(preds) == 20
        assert all(p in (0, 1) for p in preds)


# ============================================================================== Gap 4: Checkpoint/Resume

class TestCheckpoint:
    def test_save_and_load(self):
        from attestra.execution.checkpoint import CheckpointManager, ExperimentState
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager("test_exp", base_dir=tmp)
            state = ExperimentState(
                experiment_id="test_exp", goal="test",
                round_num=5, best_score=0.85, best_technique="hist_gbm",
                tried_families=["rf", "gbm"],
            )
            ckpt_id = mgr.save(state, model={"weights": [1, 2, 3]})
            assert ckpt_id.startswith("ckpt_")

            loaded_state, loaded_model = mgr.load(ckpt_id)
            assert loaded_state.round_num == 5
            assert loaded_state.best_score == 0.85
            assert loaded_model == {"weights": [1, 2, 3]}

    def test_load_latest(self):
        from attestra.execution.checkpoint import CheckpointManager, ExperimentState
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager("exp2", base_dir=tmp)
            for i in range(3):
                state = ExperimentState(experiment_id="exp2", goal="test",
                                        round_num=i, best_score=i * 0.1)
                mgr.save(state)
            state, _ = mgr.load_latest()
            assert state.round_num == 2

    def test_cleanup(self):
        from attestra.execution.checkpoint import CheckpointManager, ExperimentState
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager("exp3", base_dir=tmp)
            for i in range(10):
                state = ExperimentState(experiment_id="exp3", goal="test", round_num=i)
                mgr.save(state)
            removed = mgr.cleanup(keep_last=3)
            assert removed > 0
            assert len(mgr.list_checkpoints()) == 10  # meta still has all


# ============================================================================== Gap 5: FDR Control

class TestFDR:
    def test_bh_procedure(self):
        from attestra.certification.fdr import FDRController, ExperimentCertificate
        fdr = FDRController(alpha=0.05, method="benjamini_hochberg")

        # Submit 10 experiments with various p-values
        p_values = [0.001, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.5, 0.8]
        for i, p in enumerate(p_values):
            cert = ExperimentCertificate(
                experiment_id=f"exp_{i}", metric="accuracy",
                threshold=0.8, observed_lower_bound=0.85,
                p_value=p, certified_locally=p < 0.05,
            )
            fdr.submit(cert)

        summary = fdr.summary()
        # BH should accept some but not all
        assert summary["n_accepted"] > 0
        assert summary["n_accepted"] < 10
        assert summary["estimated_fdr"] <= 0.05

    def test_holm_bonferroni(self):
        from attestra.certification.fdr import FDRController, ExperimentCertificate
        fdr = FDRController(alpha=0.05, method="holm_bonferroni")

        for i in range(5):
            cert = ExperimentCertificate(
                experiment_id=f"exp_{i}", metric="accuracy",
                threshold=0.8, observed_lower_bound=0.9,
                p_value=0.001 * (i + 1), certified_locally=True,
            )
            fdr.submit(cert)

        # Holm is stricter than BH
        summary = fdr.summary()
        assert summary["n_accepted"] > 0

    def test_persistence(self):
        from attestra.certification.fdr import FDRController, ExperimentCertificate
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            fdr = FDRController(alpha=0.05, persist_path=path)
            cert = ExperimentCertificate(
                experiment_id="exp_persist", metric="accuracy",
                threshold=0.8, observed_lower_bound=0.9,
                p_value=0.01, certified_locally=True,
            )
            fdr.submit(cert)

            # Reload
            fdr2 = FDRController(alpha=0.05, persist_path=path)
            assert len(fdr2.certificates) == 1
        finally:
            os.unlink(path)


# ============================================================================== Gap 6: Data Augmentation

class TestDataAugmentation:
    def test_smote(self):
        from attestra.augmentation.augmentation import DataAugmenter
        # Imbalanced dataset
        X = np.random.randn(100, 5)
        y = np.array([0] * 90 + [1] * 10)
        aug = DataAugmenter(X, y, task="classification")
        result = aug.apply("smote")
        assert result.n_generated > 0
        assert len(result.X_aug) > len(X)
        # Check minority class is boosted
        unique, counts = np.unique(result.y_aug, return_counts=True)
        assert counts[1] > 10

    def test_noise(self):
        from attestra.augmentation.augmentation import DataAugmenter
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        aug = DataAugmenter(X, y, task="regression")
        result = aug.apply("noise", scale=0.05)
        assert result.n_generated == 50

    def test_mixup(self):
        from attestra.augmentation.augmentation import DataAugmenter
        X = np.random.randn(80, 4)
        y = np.random.randint(0, 3, 80)
        aug = DataAugmenter(X, y, task="classification")
        result = aug.apply("mixup", n_samples=40)
        assert result.n_generated == 40

    def test_bootstrap(self):
        from attestra.augmentation.augmentation import DataAugmenter
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        aug = DataAugmenter(X, y, task="regression")
        result = aug.apply("bootstrap", n_samples=20)
        assert len(result.X_aug) == 50  # 30 + 20

    def test_recommend_strategies(self):
        from attestra.augmentation.augmentation import DataAugmenter
        # Imbalanced small dataset
        X = np.random.randn(60, 5)
        y = np.array([0] * 55 + [1] * 5)
        aug = DataAugmenter(X, y, task="classification")
        recommended = aug.recommend_strategies()
        assert "smote" in recommended

    def test_copula(self):
        from attestra.augmentation.augmentation import DataAugmenter
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        aug = DataAugmenter(X, y, task="classification")
        result = aug.apply("copula", n_samples=50)
        assert result.n_generated == 50
        assert result.X_aug.shape[1] == 10


# ============================================================================== Gap 7: Tradeoff Engine

class TestTradeoff:
    def test_profile_and_recommend(self):
        from attestra.design.tradeoff import TradeoffEngine, DeploymentConstraints
        X_val = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 200)
        X_train = np.random.randn(200, 10)

        clf1 = HistGradientBoostingClassifier(max_iter=10, random_state=42).fit(X_train, y_train)
        clf2 = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_train, y_train)

        engine = TradeoffEngine()
        engine.profile_model("HistGBM", clf1, X_val, accuracy=0.85, training_time_s=2.0)
        engine.profile_model("RF", clf2, X_val, accuracy=0.82, training_time_s=1.0)

        result = engine.recommend(DeploymentConstraints(priority="accuracy"))
        assert result.recommended.name == "HistGBM"
        assert result.n_pareto >= 1

    def test_latency_constraint(self):
        from attestra.design.tradeoff import TradeoffEngine, DeploymentConstraints, ModelProfile
        engine = TradeoffEngine()
        engine.add_profile(ModelProfile(name="fast", accuracy=0.80, inference_latency_ms=1.0))
        engine.add_profile(ModelProfile(name="slow", accuracy=0.90, inference_latency_ms=100.0))

        result = engine.recommend(DeploymentConstraints(max_latency_ms=5.0, priority="accuracy"))
        assert result.recommended.name == "fast"  # slow violates constraint


# ============================================================================== Gap 8: Architecture Builder

class TestArchitectureBuilder:
    @pytest.fixture
    def has_torch(self):
        try:
            import torch
            return True
        except ImportError:
            pytest.skip("torch not installed")

    def test_from_library_mlp(self, has_torch):
        from attestra.execution.architecture_builder import ArchitectureBuilder
        builder = ArchitectureBuilder()
        arch = builder.from_library("mlp_medium", input_dim=64, output_dim=10)
        assert arch.validated
        assert arch.n_parameters > 0

    def test_from_library_resnet(self, has_torch):
        from attestra.execution.architecture_builder import ArchitectureBuilder
        builder = ArchitectureBuilder()
        arch = builder.from_library("resnet_tabular", input_dim=32, output_dim=5)
        assert arch.validated
        assert arch.n_parameters > 0

    def test_recommend(self):
        from attestra.execution.architecture_builder import ArchitectureBuilder
        builder = ArchitectureBuilder()
        name = builder.recommend_architecture(10000, 100, 5, "classification")
        assert name in ("ft_transformer", "resnet_tabular", "mlp_medium")


# ============================================================================== Gap 9: Experiment Manager

class TestExperimentManager:
    def test_plan_simple(self):
        from attestra.orchestration.experiment_manager import ExperimentManager
        mgr = ExperimentManager()
        plan = mgr.plan_experiment("classify iris", complexity="simple", time_budget_s=60)
        assert len(plan.phases) == 1
        assert plan.phases[0].phase_type == "baseline"

    def test_plan_complex(self):
        from attestra.orchestration.experiment_manager import ExperimentManager
        mgr = ExperimentManager()
        plan = mgr.plan_experiment("build vision model", complexity="complex", time_budget_s=3600)
        assert len(plan.phases) >= 4
        assert plan.phases[0].phase_type == "baseline"

    def test_phase_execution(self):
        from attestra.orchestration.experiment_manager import ExperimentManager, PhaseStatus
        mgr = ExperimentManager()
        plan = mgr.plan_experiment("test", complexity="medium", time_budget_s=300)
        
        # Get next phases
        ready = mgr.next_phases(plan)
        assert len(ready) == 1  # baseline is first
        
        # Complete baseline
        phase = ready[0]
        mgr.start_phase(plan, phase)
        assert phase.status == PhaseStatus.RUNNING
        
        cont = mgr.complete_phase(plan, phase, {"best_score": 0.8, "best_technique": "rf", "elapsed_s": 10})
        assert cont  # should continue
        assert plan.overall_best_score == 0.8

    def test_early_stop(self):
        from attestra.orchestration.experiment_manager import ExperimentManager, PhaseStatus
        mgr = ExperimentManager()
        plan = mgr.plan_experiment("test", complexity="medium", time_budget_s=300,
                                    early_stop_threshold=0.9)
        ready = mgr.next_phases(plan)
        phase = ready[0]
        mgr.start_phase(plan, phase)
        cont = mgr.complete_phase(plan, phase, {"best_score": 0.95, "best_technique": "gbm", "elapsed_s": 5})
        assert not cont  # should stop early
        # Remaining phases should be skipped
        assert any(p.status == PhaseStatus.SKIPPED for p in plan.phases)

    def test_classify_complexity(self):
        from attestra.orchestration.experiment_manager import ExperimentManager, ExperimentComplexity
        mgr = ExperimentManager()
        assert mgr.classify_complexity(100, 10, "binary", "classify") == ExperimentComplexity.SIMPLE
        assert mgr.classify_complexity(50000, 200, "multiclass", "build model") == ExperimentComplexity.COMPLEX
        assert mgr.classify_complexity(1000, 20, "binary", "novel state of the art TTS") == ExperimentComplexity.FRONTIER


# ============================================================================== Gap 10: Meta-Learner

class TestMetaLearner:
    def test_observe_and_recommend(self):
        from attestra.improvement.meta_learner import MetaLearner, Experience
        learner = MetaLearner(alpha=0.1)  # Low exploration to test exploitation

        # Observe some experiences
        for _ in range(20):
            learner.observe(Experience(
                n_samples=1000, n_features=20, task_type="binary",
                strategy="hist_gbm", score=0.85, improvement=0.05, success=True,
            ))
        for _ in range(20):
            learner.observe(Experience(
                n_samples=1000, n_features=20, task_type="binary",
                strategy="logistic", score=0.7, improvement=0.01, success=True,
            ))
        # Also observe rf so it's not pure exploration
        for _ in range(5):
            learner.observe(Experience(
                n_samples=1000, n_features=20, task_type="binary",
                strategy="rf", score=0.75, improvement=0.02, success=True,
            ))

        # Recommend — hist_gbm should have highest expected reward
        state = Experience(n_samples=1000, n_features=20, task_type="binary")
        recs = learner.recommend(state, ["hist_gbm", "logistic", "rf"])
        assert len(recs) > 0
        # hist_gbm should have highest expected reward (most improvement)
        expected_rewards = {r.strategy: r.expected_reward for r in recs}
        assert expected_rewards["hist_gbm"] > expected_rewards["logistic"]

    def test_capability_boundary(self):
        from attestra.improvement.meta_learner import MetaLearner, Experience
        learner = MetaLearner()
        # hist_gbm always succeeds for binary
        for _ in range(5):
            learner.observe(Experience(
                n_samples=500, n_features=10, task_type="binary",
                strategy="hist_gbm", success=True, improvement=0.1,
            ))
        # svm always fails for binary
        for _ in range(5):
            learner.observe(Experience(
                n_samples=500, n_features=10, task_type="binary",
                strategy="svm", success=False, improvement=0.0,
            ))

        boundary = learner.capability_boundary("binary")
        assert "hist_gbm" in boundary
        assert boundary["hist_gbm"] > 0.5
        assert boundary["svm"] < 0.5

    def test_avoid_list(self):
        from attestra.improvement.meta_learner import MetaLearner, Experience
        learner = MetaLearner()
        for _ in range(10):
            learner.observe(Experience(
                n_samples=100, n_features=5, task_type="regression",
                strategy="bad_model", success=False, improvement=0.0,
            ))
        avoid = learner.avoid_list("regression", min_attempts=5)
        assert "bad_model" in avoid

    def test_persistence(self):
        from attestra.improvement.meta_learner import MetaLearner, Experience
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            learner = MetaLearner(persist_path=path)
            learner.observe(Experience(
                n_samples=100, n_features=5, task_type="binary",
                strategy="rf", success=True, improvement=0.05,
            ))
            # Reload
            learner2 = MetaLearner(persist_path=path)
            assert len(learner2._arms) == 1
        finally:
            os.unlink(path)


# ============================================================================== Integration test

class TestIntegration:
    def test_full_pipeline_with_new_modules(self):
        """Test the full pipeline using all new modules together."""
        from attestra.intake.problem_typing import type_problem
        from attestra.intake.profiler import profile_data
        from attestra.augmentation.augmentation import DataAugmenter
        from attestra.design.tradeoff import TradeoffEngine
        from attestra.certification.fdr import FDRController, ExperimentCertificate
        from attestra.orchestration.experiment_manager import ExperimentManager
        from attestra.improvement.meta_learner import MetaLearner, Experience

        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target

        # 1. Problem typing
        profile = profile_data(X, y)
        spec = type_problem("Classify iris flowers", data_profile=profile.to_dict() if hasattr(profile, 'to_dict') else None)
        assert spec.domain.value in ("tabular_classification", "unknown")

        # 2. Data augmentation
        aug = DataAugmenter(X, y, task="classification")
        aug_result = aug.apply("noise", scale=0.1)
        assert aug_result.n_generated > 0

        # 3. Train two models
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        clf1 = HistGradientBoostingClassifier(max_iter=20, random_state=42).fit(X_train, y_train)
        clf2 = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
        acc1 = float(np.mean(clf1.predict(X_val) == y_val))
        acc2 = float(np.mean(clf2.predict(X_val) == y_val))

        # 4. Tradeoff analysis
        engine = TradeoffEngine()
        engine.profile_model("HistGBM", clf1, X_val, accuracy=acc1, training_time_s=0.5)
        engine.profile_model("RF", clf2, X_val, accuracy=acc2, training_time_s=0.3)
        result = engine.recommend()
        assert result.recommended.accuracy > 0

        # 5. FDR control
        fdr = FDRController(alpha=0.05)
        cert = ExperimentCertificate(
            experiment_id="iris_test", metric="accuracy",
            threshold=0.8, observed_lower_bound=acc1,
            p_value=0.01, certified_locally=True,
        )
        decision = fdr.submit(cert)
        assert decision.accepted

        # 6. Meta-learner
        learner = MetaLearner()
        learner.observe(Experience(
            n_samples=len(X), n_features=X.shape[1], task_type="multiclass",
            strategy="hist_gbm", success=True, improvement=acc1 - 0.5,
        ))
        recs = learner.recommend(
            Experience(n_samples=150, n_features=4, task_type="multiclass"),
            ["hist_gbm", "rf", "svm"]
        )
        assert len(recs) > 0
