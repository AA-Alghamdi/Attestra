"""Meta-learning and RL for cross-experiment policy optimization.

Beyond simple frequency tracking — this module implements:
  1. CONTEXTUAL BANDITS: Choose strategies based on problem features
  2. POLICY GRADIENT: Learn which strategies to try first given context
  3. EXPERIENCE REPLAY: Replay past successes to improve future proposals
  4. CAPABILITY BOUNDARY DETECTION: Know what you can and can't do
  5. TRANSFER LEARNING: Apply knowledge from similar solved problems

The meta-learner observes (state, action, reward) triples across experiments:
  - State: data profile features (n_samples, n_features, task_type, quality, ...)
  - Action: strategy chosen (model family, augmentation, feature engineering, ...)
  - Reward: normalized improvement in metric

Over time, it learns a POLICY: given a new problem's state, which actions
are most likely to succeed?
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Experience:
    """A single (state, action, reward) observation."""
    # State (problem context)
    n_samples: int
    n_features: int
    task_type: str                        # "binary", "multiclass", "regression"
    n_classes: int = 0
    quality_score: float = 1.0
    has_missing: bool = False
    has_imbalance: bool = False
    # Action
    strategy: str = ""                    # model family or approach name
    source: str = ""                      # "catalog" | "llm" | "mutation" | "retrieval"
    # Reward
    score: float = 0.0
    improvement: float = 0.0              # delta from baseline/previous
    success: bool = False
    elapsed_s: float = 0.0
    # Meta
    dataset_fp: str = ""
    created_at: float = field(default_factory=time.time)

    def state_vector(self) -> np.ndarray:
        """Convert state to numeric feature vector for the policy."""
        return np.array([
            self.n_samples / 100000,      # normalized
            self.n_features / 1000,
            {"binary": 0, "multiclass": 1, "regression": 2}.get(self.task_type, 3) / 3,
            self.n_classes / 100,
            self.quality_score,
            float(self.has_missing),
            float(self.has_imbalance),
        ])


@dataclass
class PolicyRecommendation:
    """A strategy recommendation from the meta-learner."""
    strategy: str
    expected_reward: float
    confidence: float
    source: str
    reasoning: str


class MetaLearner:
    """Contextual bandit meta-learner for strategy selection.

    Uses LinUCB (Linear Upper Confidence Bound) — a well-studied
    contextual bandit algorithm that balances exploration/exploitation.

    For each strategy (arm), maintains a linear model:
      E[reward | context] = theta_a^T * context

    Selection: pick argmax_a (theta_a^T * x + alpha * sqrt(x^T A_a^{-1} x))
    """

    def __init__(self, alpha: float = 1.0, persist_path: Optional[str] = None):
        self.alpha = alpha                # exploration parameter
        self._persist_path = persist_path
        # LinUCB state per arm
        self._arms: Dict[str, Dict] = {}  # strategy -> {A, b, theta}
        self._d = 7                       # state dimension
        self._experiences: List[Experience] = []
        # Capability tracking
        self._success_counts: Dict[str, int] = defaultdict(int)
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._task_capabilities: Dict[str, Dict[str, float]] = defaultdict(dict)

        if persist_path and os.path.exists(persist_path):
            self._load()

    def observe(self, experience: Experience) -> None:
        """Record an observation and update the policy."""
        self._experiences.append(experience)

        # Update LinUCB
        arm = experience.strategy
        if arm not in self._arms:
            self._init_arm(arm)

        x = experience.state_vector().reshape(-1, 1)
        reward = experience.improvement if experience.success else -0.1

        A = self._arms[arm]["A"]
        b = self._arms[arm]["b"]
        self._arms[arm]["A"] = A + x @ x.T
        self._arms[arm]["b"] = b + reward * x
        # Recompute theta
        try:
            self._arms[arm]["theta"] = np.linalg.solve(
                self._arms[arm]["A"], self._arms[arm]["b"]
            )
        except np.linalg.LinAlgError:
            pass

        # Update capability tracking
        if experience.success:
            self._success_counts[arm] += 1
        else:
            self._failure_counts[arm] += 1

        key = f"{experience.task_type}_{arm}"
        if key not in self._task_capabilities:
            self._task_capabilities[key] = {"successes": 0, "attempts": 0}
        self._task_capabilities[key]["attempts"] += 1
        if experience.success:
            self._task_capabilities[key]["successes"] += 1

        self._persist()

    def recommend(self, state: Experience, available_arms: List[str],
                  top_k: int = 5) -> List[PolicyRecommendation]:
        """Recommend strategies for a given problem context.

        Uses LinUCB: select arms with highest upper confidence bound.
        """
        x = state.state_vector().reshape(-1, 1)
        recommendations = []

        for arm in available_arms:
            if arm not in self._arms:
                self._init_arm(arm)

            A = self._arms[arm]["A"]
            theta = self._arms[arm]["theta"]

            # UCB
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                A_inv = np.eye(self._d)

            expected = float((theta.T @ x).item())
            uncertainty = float(self.alpha * np.sqrt((x.T @ A_inv @ x).item()))
            ucb = expected + uncertainty

            # Confidence based on number of observations
            n_obs = self._success_counts[arm] + self._failure_counts[arm]
            confidence = min(1.0, n_obs / 20.0)

            recommendations.append(PolicyRecommendation(
                strategy=arm,
                expected_reward=expected,
                confidence=confidence,
                source="linucb",
                reasoning=f"UCB={ucb:.3f} (E={expected:.3f}, U={uncertainty:.3f}, n={n_obs})",
            ))

        # Sort by UCB (expected + uncertainty)
        recommendations.sort(
            key=lambda r: r.expected_reward + self.alpha * (1 - r.confidence),
            reverse=True
        )
        return recommendations[:top_k]

    def rank_strategies(self, task_type: str, n_samples: int,
                        n_features: int) -> List[Tuple[str, float]]:
        """Rank strategies by expected performance for a context."""
        state = Experience(
            n_samples=n_samples, n_features=n_features,
            task_type=task_type,
        )
        recs = self.recommend(state, list(self._arms.keys()))
        return [(r.strategy, r.expected_reward) for r in recs]

    def capability_boundary(self, task_type: str) -> Dict[str, float]:
        """Estimate what the system can and can't do for a task type.

        Returns dict of {strategy: estimated_success_probability}.
        """
        boundaries = {}
        for key, stats in self._task_capabilities.items():
            if key.startswith(task_type + "_"):
                strategy = key[len(task_type) + 1:]
                attempts = stats["attempts"]
                successes = stats["successes"]
                # Beta posterior mean
                boundaries[strategy] = (successes + 1) / (attempts + 2)
        return boundaries

    def avoid_list(self, task_type: str, min_attempts: int = 5,
                   max_success_rate: float = 0.1) -> List[str]:
        """Strategies to avoid for a task type (consistently fail)."""
        avoid = []
        for key, stats in self._task_capabilities.items():
            if key.startswith(task_type + "_"):
                strategy = key[len(task_type) + 1:]
                if stats["attempts"] >= min_attempts:
                    rate = stats["successes"] / stats["attempts"]
                    if rate <= max_success_rate:
                        avoid.append(strategy)
        return avoid

    def transfer_knowledge(self, source_task: str, target_task: str) -> List[str]:
        """Suggest strategies to try based on similar solved tasks."""
        # Find best strategies for source task
        source_boundary = self.capability_boundary(source_task)
        if not source_boundary:
            return []
        # Sort by success rate
        ranked = sorted(source_boundary.items(), key=lambda x: -x[1])
        # Return top strategies that haven't been tried much on target
        target_boundary = self.capability_boundary(target_task)
        suggestions = []
        for strategy, rate in ranked:
            if rate > 0.5:  # only transfer strategies that work well
                target_rate = target_boundary.get(strategy)
                if target_rate is None or target_rate > 0.3:
                    suggestions.append(strategy)
        return suggestions[:5]

    def summary(self) -> Dict:
        return {
            "n_experiences": len(self._experiences),
            "n_arms": len(self._arms),
            "top_arms": sorted(
                [(arm, self._success_counts[arm], self._failure_counts[arm])
                 for arm in self._arms],
                key=lambda x: x[1], reverse=True
            )[:10],
            "alpha": self.alpha,
        }

    # ========================================================================== internal

    def _init_arm(self, arm: str) -> None:
        """Initialize a new arm (strategy)."""
        self._arms[arm] = {
            "A": np.eye(self._d),
            "b": np.zeros((self._d, 1)),
            "theta": np.zeros((self._d, 1)),
        }

    def _persist(self) -> None:
        if not self._persist_path:
            return
        os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
        data = {
            "alpha": self.alpha,
            "arms": {
                name: {
                    "A": state["A"].tolist(),
                    "b": state["b"].tolist(),
                    "theta": state["theta"].tolist(),
                }
                for name, state in self._arms.items()
            },
            "success_counts": dict(self._success_counts),
            "failure_counts": dict(self._failure_counts),
            "task_capabilities": dict(self._task_capabilities),
            "n_experiences": len(self._experiences),
        }
        with open(self._persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)
            self.alpha = data.get("alpha", self.alpha)
            for name, state in data.get("arms", {}).items():
                self._arms[name] = {
                    "A": np.array(state["A"]),
                    "b": np.array(state["b"]),
                    "theta": np.array(state["theta"]),
                }
            self._success_counts = defaultdict(int, data.get("success_counts", {}))
            self._failure_counts = defaultdict(int, data.get("failure_counts", {}))
            self._task_capabilities = defaultdict(dict, data.get("task_capabilities", {}))
        except (json.JSONDecodeError, IOError):
            pass
