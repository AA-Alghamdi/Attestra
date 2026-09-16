"""Experiment plan: the structured contract between intake and execution.

Before ANY experiment runs, the system produces an ExperimentPlan that specifies:
  1. The experiment plan (what will be tried)
  2. The assumptions and verification criteria (what must be true to certify)
  3. The experiment surfaces (what the optimizer is allowed to change)
  4. The budget/time estimate (what resources are allocated)

This is the pre-registration spec. Content-addressed (SHA256) so it cannot
be retroactively changed to match results.

The plan is informed by the DataProfile from intake and the power analysis.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..intake.profiler import DataProfile


@dataclass
class VerificationCriteria:
    """What must be true to certify the result."""
    metric: str                          # "accuracy", "r2", etc.
    threshold: float                     # lower bound to clear
    alpha: float = 0.05                  # significance level
    max_peeks: int = 1                   # how many sealed test evaluations
    required_slices: List[Dict] = field(default_factory=list)  # per-subgroup gates
    max_ece: Optional[float] = None      # calibration requirement
    max_latency_ms: Optional[float] = None
    max_cost_usd: Optional[float] = None


@dataclass
class ExperimentSurface:
    """What the optimizer is allowed to change."""
    model_families: List[str]            # which model families to try
    feature_engineering: bool = True      # allow feature transforms
    hyperparameter_search: bool = True    # allow HP optimization
    ensemble: bool = True                 # allow model combination
    data_augmentation: bool = False       # allow synthetic data
    architecture_search: bool = False     # allow custom architectures (LLM)
    max_model_complexity: str = "medium"  # "simple" | "medium" | "complex" | "frontier"


@dataclass
class BudgetEstimate:
    """Resource allocation for the experiment."""
    time_budget_s: float                 # wall-clock budget
    max_rounds: int                      # max research iterations
    max_proposals: int                   # max proposals to evaluate
    estimated_cost_usd: float = 0.0      # estimated compute cost
    gpu_required: bool = False
    parallelism: int = 1                 # how many concurrent experiments


@dataclass
class PowerAnalysis:
    """Statistical power analysis: can we even certify with this data?"""
    n_test: int                          # available test samples
    estimated_effect_size: float         # expected improvement over baseline
    required_n: int                      # samples needed for power >= 0.8
    has_power: bool                      # True if n_test >= required_n
    power_at_n: float                    # estimated power at current n
    recommendation: str                  # what to do if underpowered


@dataclass
class ExperimentPlan:
    """The complete experiment contract. Content-addressed."""
    goal: str
    data_profile: DataProfile
    verification: VerificationCriteria
    surface: ExperimentSurface
    budget: BudgetEstimate
    power: PowerAnalysis
    # metadata
    created_at: float = field(default_factory=time.time)
    plan_hash: str = ""
    assumptions: List[str] = field(default_factory=list)
    strategy_notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.plan_hash:
            self.plan_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Content-address the plan (SHA256). Cannot be retroactively changed."""
        content = json.dumps({
            "goal": self.goal,
            "n_samples": self.data_profile.n_samples,
            "n_features": self.data_profile.n_features,
            "task_type": self.data_profile.task_type,
            "metric": self.verification.metric,
            "threshold": self.verification.threshold,
            "alpha": self.verification.alpha,
            "model_families": sorted(self.surface.model_families),
            "time_budget_s": self.budget.time_budget_s,
            "max_rounds": self.budget.max_rounds,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_llm_context(self) -> str:
        """Format plan for LLM consumption."""
        lines = [
            f"EXPERIMENT PLAN (hash: {self.plan_hash})",
            f"Goal: {self.goal}",
            "",
            "DATA:",
            self.data_profile.to_llm_context(),
            "",
            "VERIFICATION:",
            f"  Metric: {self.verification.metric}",
            f"  Threshold: {self.verification.threshold}",
            f"  Alpha: {self.verification.alpha}",
            "",
            "ALLOWED SURFACES:",
            f"  Model families: {', '.join(self.surface.model_families)}",
            f"  Feature engineering: {self.surface.feature_engineering}",
            f"  HP search: {self.surface.hyperparameter_search}",
            f"  Ensemble: {self.surface.ensemble}",
            f"  Architecture search: {self.surface.architecture_search}",
            "",
            "BUDGET:",
            f"  Time: {self.budget.time_budget_s:.0f}s",
            f"  Max rounds: {self.budget.max_rounds}",
            f"  Max proposals: {self.budget.max_proposals}",
            f"  GPU: {'required' if self.budget.gpu_required else 'not required'}",
            "",
            "POWER ANALYSIS:",
            f"  Test samples: {self.power.n_test}",
            f"  Has power: {self.power.has_power}",
            f"  Estimated power: {self.power.power_at_n:.2f}",
            f"  {self.power.recommendation}",
        ]
        if self.assumptions:
            lines.append("")
            lines.append("ASSUMPTIONS:")
            for a in self.assumptions:
                lines.append(f"  - {a}")
        if self.strategy_notes:
            lines.append("")
            lines.append("STRATEGY:")
            for s in self.strategy_notes:
                lines.append(f"  - {s}")
        return "\n".join(lines)


def design_experiment(
    goal: str,
    profile: DataProfile,
    *,
    metric: Optional[str] = None,
    threshold: Optional[float] = None,
    time_budget_s: float = 300.0,
    max_rounds: int = 15,
    gpu: bool = False,
    use_llm: bool = True,
) -> ExperimentPlan:
    """Design an experiment plan from a goal + data profile.

    This is the experiment design stage. It:
      1. Infers metric and threshold from task type if not provided
      2. Determines which model families are appropriate
      3. Estimates power (can we certify?)
      4. Sets the budget
      5. Produces the content-addressed plan
    """
    # Infer metric
    if metric is None:
        if profile.task_type == "regression":
            metric = "r2"
        elif profile.n_classes > 2:
            metric = "balanced_accuracy"
        else:
            metric = "accuracy"

    # Infer threshold
    if threshold is None:
        threshold = _infer_threshold(profile, metric)

    # Determine model families
    families = _select_model_families(profile, gpu)

    # Power analysis
    power = _power_analysis(profile, metric, threshold)

    # Surface
    surface = ExperimentSurface(
        model_families=families,
        feature_engineering=True,
        hyperparameter_search=True,
        ensemble=True,
        data_augmentation=(profile.n_samples < 500),
        architecture_search=use_llm,
        max_model_complexity="frontier" if gpu else ("complex" if profile.n_samples > 1000 else "medium"),
    )

    # Budget
    proposals_per_round = 3 if use_llm else 2
    budget = BudgetEstimate(
        time_budget_s=time_budget_s,
        max_rounds=max_rounds,
        max_proposals=max_rounds * proposals_per_round,
        estimated_cost_usd=_estimate_cost(profile, time_budget_s, gpu),
        gpu_required=gpu,
        parallelism=1,
    )

    # Verification
    verification = VerificationCriteria(
        metric=metric,
        threshold=threshold,
        alpha=0.05,
        max_peeks=1,
    )

    # Assumptions
    assumptions = _generate_assumptions(profile, metric, threshold)

    # Strategy notes
    strategy_notes = _generate_strategy(profile, families, use_llm)

    return ExperimentPlan(
        goal=goal,
        data_profile=profile,
        verification=verification,
        surface=surface,
        budget=budget,
        power=power,
        assumptions=assumptions,
        strategy_notes=strategy_notes,
    )


def _infer_threshold(profile: DataProfile, metric: str) -> float:
    """Infer a reasonable threshold from the data profile."""
    if metric == "r2":
        if profile.n_samples < 200:
            return 0.3
        return 0.5
    if metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        n_classes = max(profile.n_classes, 2)
        chance = 1.0 / n_classes
        # Threshold = midpoint between chance and perfect
        return round(chance + (1.0 - chance) * 0.5, 2)
    if metric in ("neg_rmse", "neg_mae"):
        return 0.0  # "better than predicting the mean"
    return 0.5


def _select_model_families(profile: DataProfile, gpu: bool) -> List[str]:
    """Select appropriate model families based on data profile."""
    families = []

    # Always start with baselines
    if profile.task_type == "regression":
        families.extend(["ridge", "lasso", "elastic_net"])
    else:
        families.extend(["logistic_regression", "linear_svc"])

    # Tree-based (almost always appropriate for tabular)
    families.extend(["random_forest", "hist_gradient_boosting", "extra_trees"])

    # If enough data, add more complex models
    if profile.n_samples > 500:
        families.extend(["gradient_boosting", "stacking"])
    if profile.n_samples > 1000:
        families.append("voting_ensemble")

    # If few features, SVMs can work well
    if profile.n_features < 50 and profile.n_samples < 10000:
        families.append("svm")

    # KNN for small datasets
    if profile.n_samples < 5000:
        families.append("knn")

    # Neural nets if GPU or enough data
    if gpu or profile.n_samples > 5000:
        families.append("mlp")
    if gpu:
        families.extend(["deep_model", "transfer_learning"])

    return families


def _power_analysis(profile: DataProfile, metric: str, threshold: float) -> PowerAnalysis:
    """Estimate statistical power for the certification test."""
    n = profile.n_samples
    # Reserve ~20% for test
    n_test = max(30, int(n * 0.2))

    if metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        n_classes = max(profile.n_classes, 2)
        # Approximate: we need the effect to be detectable at this n
        # Using normal approximation to binomial
        effect = max(0.05, threshold - 1.0 / n_classes)
        # Required n for 80% power (approximate)
        if effect > 0:
            required_n = int(math.ceil((1.96 + 0.84) ** 2 * threshold * (1 - threshold) / (effect ** 2)))
        else:
            required_n = 10000
        power_at_n = min(0.99, 1.0 - math.exp(-n_test * effect ** 2 / (2 * threshold * (1 - threshold) + 1e-9)))
    else:
        # Regression: rougher estimate
        effect = 0.1
        required_n = max(100, int(20 / max(effect, 0.01) ** 2))
        power_at_n = min(0.99, n_test / max(required_n, 1))

    has_power = n_test >= required_n
    if has_power:
        recommendation = f"Sufficient power (n_test={n_test} >= required {required_n})"
    else:
        recommendation = (f"UNDERPOWERED: n_test={n_test} < required {required_n}. "
                          f"Consider: (a) lower threshold, (b) more data, (c) simpler model with less variance")

    return PowerAnalysis(
        n_test=n_test,
        estimated_effect_size=effect,
        required_n=required_n,
        has_power=has_power,
        power_at_n=round(power_at_n, 3),
        recommendation=recommendation,
    )


def _estimate_cost(profile: DataProfile, time_budget_s: float, gpu: bool) -> float:
    """Rough cost estimate in USD."""
    if gpu:
        # ~$1/hr for a decent GPU
        return round(time_budget_s / 3600 * 1.0, 2)
    return 0.0


def _generate_assumptions(profile: DataProfile, metric: str, threshold: float) -> List[str]:
    """Generate explicit assumptions for the experiment."""
    assumptions = [
        "Data is iid (rows are independent samples from the same distribution)",
        f"The {metric} metric is appropriate for this task",
        f"A threshold of {threshold} represents a meaningful improvement over baseline",
        "The sealed test set is representative of the deployment distribution",
    ]
    if profile.missing_rate > 0:
        assumptions.append("Missing values are MAR (missing at random), not MNAR")
    if profile.class_balance:
        minority = min(profile.class_balance.values())
        if minority < 0.1:
            assumptions.append("Class imbalance reflects the true distribution (not sampling bias)")
    return assumptions


def _generate_strategy(profile: DataProfile, families: List[str],
                       use_llm: bool) -> List[str]:
    """Generate strategy notes for the experiment."""
    notes = []
    if profile.n_samples < 200:
        notes.append("Small dataset -> prioritize simple models, regularization, cross-validation")
    elif profile.n_samples > 10000:
        notes.append("Large dataset -> can afford complex models, less risk of overfitting")

    if profile.n_constant_features > 0:
        notes.append(f"Remove {profile.n_constant_features} constant features in preprocessing")

    if profile.n_id_like_features > 0:
        notes.append(f"Drop {profile.n_id_like_features} ID-like features (overfitting risk)")

    if profile.missing_rate > 0.1:
        notes.append("High missing rate -> use HistGBM (native NaN handling) or imputation pipeline")

    if use_llm:
        notes.append("LLM proposals enabled -> will generate novel architectures beyond catalog")
        notes.append("Deterministic baselines run in parallel as safety net")

    if profile.top_correlations:
        top = profile.top_correlations[0]
        if top[2] > 0.95:
            notes.append(f"Highly correlated features detected ({top[0]} <-> {top[1]}: {top[2]:.2f}) -> consider PCA")

    return notes
