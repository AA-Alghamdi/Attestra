"""Strategy learner: learn which strategies work for which problem types.

Over time, the system builds a model of:
  - Which model families work best for which data profiles
  - Which proposal sources are most effective
  - Which error patterns predict failure
  - How to allocate budget across approaches

This is the "recursive self-improvement" component: the system gets better
at CHOOSING what to try, not just at trying things.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StrategyOutcome:
    """Record of a strategy application and its outcome."""
    strategy_name: str           # e.g., "hist_gradient_boosting", "stacking", "llm_proposal"
    source: str                  # "catalog" | "llm" | "retrieval" | "mutation"
    task_type: str               # "binary" | "multiclass" | "regression"
    n_samples: int
    n_features: int
    # outcome
    success: bool
    score: Optional[float] = None
    error_category: Optional[str] = None
    elapsed_s: float = 0.0
    # dataset fingerprint for cross-dataset learning
    dataset_fp: str = ""
    created_at: float = field(default_factory=time.time)


class StrategyLearner:
    """Learns which strategies work for which problems.

    Maintains a durable JSONL store of strategy outcomes and computes:
      - Strategy success rates by problem type
      - Strategy effectiveness (average score when successful)
      - Error pattern frequencies
      - Recommended ordering for new problems
    """

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.expanduser("~/.attestra/strategy_learner.jsonl")
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._cache: Optional[List[StrategyOutcome]] = None

    def record(self, outcome: StrategyOutcome) -> None:
        """Record a strategy outcome."""
        d = {
            "strategy_name": outcome.strategy_name,
            "source": outcome.source,
            "task_type": outcome.task_type,
            "n_samples": outcome.n_samples,
            "n_features": outcome.n_features,
            "success": outcome.success,
            "score": outcome.score,
            "error_category": outcome.error_category,
            "elapsed_s": outcome.elapsed_s,
            "dataset_fp": outcome.dataset_fp,
            "created_at": outcome.created_at,
        }
        line = json.dumps(d, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self._cache = None

    def _load(self) -> List[StrategyOutcome]:
        if self._cache is not None:
            return self._cache
        outcomes = []
        if not os.path.exists(self.path):
            self._cache = outcomes
            return outcomes
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    outcomes.append(StrategyOutcome(**{
                        k: v for k, v in d.items()
                        if k in StrategyOutcome.__dataclass_fields__
                    }))
                except (json.JSONDecodeError, TypeError):
                    continue
        self._cache = outcomes
        return outcomes

    def success_rate(self, strategy_name: str, task_type: str = "") -> float:
        """Success rate of a strategy (optionally filtered by task type)."""
        outcomes = [o for o in self._load() if o.strategy_name == strategy_name]
        if task_type:
            outcomes = [o for o in outcomes if o.task_type == task_type]
        if not outcomes:
            return 0.5  # prior: 50% for unknown strategies
        return sum(1 for o in outcomes if o.success) / len(outcomes)

    def avg_score(self, strategy_name: str, task_type: str = "") -> Optional[float]:
        """Average score of a strategy when successful."""
        outcomes = [o for o in self._load()
                    if o.strategy_name == strategy_name and o.success and o.score is not None]
        if task_type:
            outcomes = [o for o in outcomes if o.task_type == task_type]
        if not outcomes:
            return None
        return np.mean([o.score for o in outcomes])

    def rank_strategies(self, task_type: str, *, n_samples: int = 0,
                        n_features: int = 0) -> List[Tuple[str, float]]:
        """Rank strategies by expected value for a problem type.

        Returns list of (strategy_name, expected_value) sorted desc.
        """
        outcomes = self._load()
        if not outcomes:
            return []

        # Filter by task type
        relevant = [o for o in outcomes if o.task_type == task_type]
        if not relevant:
            relevant = outcomes

        # Group by strategy
        by_strategy: Dict[str, List[StrategyOutcome]] = defaultdict(list)
        for o in relevant:
            by_strategy[o.strategy_name].append(o)

        # Expected value = success_rate * avg_score_when_successful
        ranked = []
        for name, outs in by_strategy.items():
            successes = [o for o in outs if o.success and o.score is not None]
            rate = len(successes) / max(len(outs), 1)
            avg_sc = np.mean([o.score for o in successes]) if successes else 0
            ev = rate * avg_sc
            ranked.append((name, float(ev)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def error_patterns(self, task_type: str = "") -> Dict[str, int]:
        """Most common error categories for a task type."""
        outcomes = [o for o in self._load()
                    if not o.success and o.error_category]
        if task_type:
            outcomes = [o for o in outcomes if o.task_type == task_type]
        counts: Dict[str, int] = defaultdict(int)
        for o in outcomes:
            counts[o.error_category] += 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def avoid_list(self, task_type: str, *, min_attempts: int = 3,
                   max_success_rate: float = 0.2) -> List[str]:
        """Strategies to avoid for a task type (low success rate with enough data)."""
        outcomes = [o for o in self._load() if o.task_type == task_type]
        by_strategy: Dict[str, List[bool]] = defaultdict(list)
        for o in outcomes:
            by_strategy[o.strategy_name].append(o.success)

        avoid = []
        for name, results in by_strategy.items():
            if len(results) >= min_attempts:
                rate = sum(results) / len(results)
                if rate <= max_success_rate:
                    avoid.append(name)
        return avoid

    def summary(self) -> Dict[str, Any]:
        outcomes = self._load()
        return {
            "total_outcomes": len(outcomes),
            "unique_strategies": len(set(o.strategy_name for o in outcomes)),
            "unique_datasets": len(set(o.dataset_fp for o in outcomes if o.dataset_fp)),
            "overall_success_rate": (sum(1 for o in outcomes if o.success) /
                                     max(len(outcomes), 1)),
        }
