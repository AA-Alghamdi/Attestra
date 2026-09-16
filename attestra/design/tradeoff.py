"""Latency/cost/accuracy tradeoff engine.

Multi-objective optimization over the three axes of deployment:
  1. ACCURACY: How well the model performs (primary metric)
  2. LATENCY: How fast inference is (ms per sample)
  3. COST: How expensive training + inference is (compute $)

The tradeoff engine:
  - Profiles model inference latency
  - Estimates training cost from compute requirements
  - Builds a Pareto frontier over (accuracy, latency, cost)
  - Recommends the best model given user constraints
  - Supports deployment-aware model selection

This is used AFTER evaluation: given a set of candidate models with known
accuracy, the engine picks the one that best satisfies deployment constraints.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ModelProfile:
    """Complete profile of a model candidate."""
    name: str
    accuracy: float                       # primary metric score
    # Latency
    inference_latency_ms: float = 0.0     # avg ms per sample
    batch_latency_ms: float = 0.0         # avg ms per batch (100 samples)
    # Cost
    training_time_s: float = 0.0
    training_cost_usd: float = 0.0        # estimated $ for training
    inference_cost_usd: float = 0.0       # $ per 1000 inferences
    # Size
    model_size_mb: float = 0.0
    n_parameters: int = 0
    # Meta
    estimator: Any = None
    technique: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def total_cost_per_1k(self) -> float:
        """Total cost per 1000 inferences (amortized training + inference)."""
        # Amortize training over 1M inferences
        amortized_training = self.training_cost_usd / 1000.0
        return amortized_training + self.inference_cost_usd

    def dominates(self, other: "ModelProfile") -> bool:
        """True if this model dominates other on all three axes."""
        return (self.accuracy >= other.accuracy and
                self.inference_latency_ms <= other.inference_latency_ms and
                self.total_cost_per_1k <= other.total_cost_per_1k and
                (self.accuracy > other.accuracy or
                 self.inference_latency_ms < other.inference_latency_ms or
                 self.total_cost_per_1k < other.total_cost_per_1k))


@dataclass
class DeploymentConstraints:
    """User-specified deployment constraints."""
    max_latency_ms: float = float("inf")        # hard latency ceiling
    max_cost_per_1k_usd: float = float("inf")   # hard cost ceiling
    min_accuracy: float = 0.0                    # hard accuracy floor
    max_model_size_mb: float = float("inf")     # size constraint
    priority: str = "accuracy"                   # "accuracy" | "latency" | "cost" | "balanced"


@dataclass
class TradeoffResult:
    """Result of tradeoff analysis."""
    recommended: ModelProfile                     # single best model given constraints
    pareto_frontier: List[ModelProfile]            # non-dominated set
    all_profiles: List[ModelProfile]
    # Analysis
    n_feasible: int = 0                           # satisfy all constraints
    n_pareto: int = 0
    analysis: str = ""


class TradeoffEngine:
    """Multi-objective model selection engine.

    Usage:
        engine = TradeoffEngine()
        
        # Profile models
        engine.profile_model("HistGBM", estimator, X_val)
        engine.profile_model("RF", estimator2, X_val)
        engine.profile_model("MLP", estimator3, X_val)
        
        # Set deployment constraints
        constraints = DeploymentConstraints(max_latency_ms=10, priority="accuracy")
        
        # Get recommendation
        result = engine.recommend(constraints)
    """

    def __init__(self):
        self.profiles: List[ModelProfile] = []

    def profile_model(self, name: str, estimator: Any, X_sample: np.ndarray,
                      accuracy: float, training_time_s: float = 0.0,
                      technique: str = "") -> ModelProfile:
        """Profile a model for latency, cost, and size."""
        # Inference latency (average over multiple runs)
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = estimator.predict(X_sample[:1])
            latencies.append((time.perf_counter() - t0) * 1000)

        inference_ms = float(np.median(latencies))

        # Batch latency
        batch_size = min(100, len(X_sample))
        t0 = time.perf_counter()
        _ = estimator.predict(X_sample[:batch_size])
        batch_ms = (time.perf_counter() - t0) * 1000

        # Model size (approximate from pickle)
        import pickle
        try:
            model_bytes = len(pickle.dumps(estimator))
            model_size_mb = model_bytes / (1024 * 1024)
        except Exception:
            model_size_mb = 0.0

        # Parameter count (heuristic)
        n_params = self._estimate_params(estimator)

        # Cost estimation (simple heuristic: $0.001/GPU-second for training)
        training_cost = training_time_s * 0.001
        # Inference cost: proportional to latency
        inference_cost = inference_ms * 0.000001  # $0.001 per 1000 ms

        profile = ModelProfile(
            name=name,
            accuracy=accuracy,
            inference_latency_ms=inference_ms,
            batch_latency_ms=batch_ms,
            training_time_s=training_time_s,
            training_cost_usd=training_cost,
            inference_cost_usd=inference_cost * 1000,  # per 1k inferences
            model_size_mb=model_size_mb,
            n_parameters=n_params,
            estimator=estimator,
            technique=technique,
        )
        self.profiles.append(profile)
        return profile

    def add_profile(self, profile: ModelProfile) -> None:
        """Add a pre-computed profile."""
        self.profiles.append(profile)

    def recommend(self, constraints: Optional[DeploymentConstraints] = None
                  ) -> TradeoffResult:
        """Recommend the best model given constraints.

        Returns:
            TradeoffResult with recommendation, Pareto frontier, and analysis
        """
        if constraints is None:
            constraints = DeploymentConstraints()

        # Filter feasible models
        feasible = [p for p in self.profiles if self._is_feasible(p, constraints)]
        n_feasible = len(feasible)

        # If nothing feasible, relax constraints
        if not feasible:
            feasible = self.profiles.copy()

        # Compute Pareto frontier
        pareto = self._pareto_frontier(feasible)

        # Rank by priority
        if constraints.priority == "accuracy":
            ranked = sorted(feasible, key=lambda p: -p.accuracy)
        elif constraints.priority == "latency":
            ranked = sorted(feasible, key=lambda p: p.inference_latency_ms)
        elif constraints.priority == "cost":
            ranked = sorted(feasible, key=lambda p: p.total_cost_per_1k)
        else:  # balanced
            ranked = sorted(feasible, key=lambda p: self._balanced_score(p))

        recommended = ranked[0] if ranked else (self.profiles[0] if self.profiles else
                                                ModelProfile(name="none", accuracy=0.0))

        analysis = self._generate_analysis(recommended, pareto, constraints, n_feasible)

        return TradeoffResult(
            recommended=recommended,
            pareto_frontier=pareto,
            all_profiles=self.profiles,
            n_feasible=n_feasible,
            n_pareto=len(pareto),
            analysis=analysis,
        )

    def _is_feasible(self, p: ModelProfile, c: DeploymentConstraints) -> bool:
        return (p.inference_latency_ms <= c.max_latency_ms and
                p.total_cost_per_1k <= c.max_cost_per_1k_usd and
                p.accuracy >= c.min_accuracy and
                p.model_size_mb <= c.max_model_size_mb)

    def _pareto_frontier(self, profiles: List[ModelProfile]) -> List[ModelProfile]:
        """Compute non-dominated set."""
        pareto = []
        for p in profiles:
            dominated = False
            for q in profiles:
                if q is not p and q.dominates(p):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)
        return pareto

    def _balanced_score(self, p: ModelProfile) -> float:
        """Composite score balancing accuracy, latency, cost."""
        # Normalize each axis to [0, 1] range relative to all profiles
        if not self.profiles:
            return 0.0
        accs = [pr.accuracy for pr in self.profiles]
        lats = [pr.inference_latency_ms for pr in self.profiles]
        costs = [pr.total_cost_per_1k for pr in self.profiles]

        acc_range = max(accs) - min(accs) if max(accs) > min(accs) else 1.0
        lat_range = max(lats) - min(lats) if max(lats) > min(lats) else 1.0
        cost_range = max(costs) - min(costs) if max(costs) > min(costs) else 1.0

        # Higher accuracy = better (negate for minimization)
        norm_acc = (p.accuracy - min(accs)) / acc_range
        # Lower latency = better
        norm_lat = 1.0 - (p.inference_latency_ms - min(lats)) / lat_range
        # Lower cost = better
        norm_cost = 1.0 - (p.total_cost_per_1k - min(costs)) / cost_range

        # Weighted sum (accuracy weighted 2x)
        return -(2 * norm_acc + norm_lat + norm_cost)

    def _estimate_params(self, estimator: Any) -> int:
        """Estimate parameter count heuristically."""
        try:
            # sklearn models
            if hasattr(estimator, "n_features_in_") and hasattr(estimator, "n_estimators"):
                return estimator.n_features_in_ * estimator.n_estimators * 10
            if hasattr(estimator, "coef_"):
                return int(np.prod(np.array(estimator.coef_).shape))
        except Exception:
            pass
        return 0

    def _generate_analysis(self, recommended: ModelProfile,
                           pareto: List[ModelProfile],
                           constraints: DeploymentConstraints,
                           n_feasible: int) -> str:
        """Generate human-readable analysis."""
        lines = [
            f"Tradeoff Analysis ({len(self.profiles)} models profiled)",
            f"  Feasible: {n_feasible}/{len(self.profiles)}",
            f"  Pareto frontier: {len(pareto)} models",
            f"  Recommended: {recommended.name}",
            f"    Accuracy: {recommended.accuracy:.4f}",
            f"    Latency: {recommended.inference_latency_ms:.2f}ms/sample",
            f"    Cost: ${recommended.total_cost_per_1k:.4f}/1k inferences",
            f"    Size: {recommended.model_size_mb:.1f}MB",
        ]
        if constraints.max_latency_ms < float("inf"):
            lines.append(f"  Constraint: latency <= {constraints.max_latency_ms}ms")
        if constraints.min_accuracy > 0:
            lines.append(f"  Constraint: accuracy >= {constraints.min_accuracy}")
        return "\n".join(lines)
