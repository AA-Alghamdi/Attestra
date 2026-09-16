"""Portfolio fan-out: run multiple approaches concurrently.

Instead of sequential proposal evaluation, the portfolio system:
  1. Fans out N approaches in parallel (thread pool or process pool)
  2. Each approach runs independently with its own error tracking
  3. Results are collected, ranked, and the best is promoted
  4. Budget is allocated across approaches using Thompson Sampling

This is the "portfolio of researchers" model — multiple independent research
tracks compete for budget, and the orchestrator allocates more resources to
tracks that are producing results.
"""
from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PortfolioArm:
    """A single research arm (approach) in the portfolio."""
    name: str
    execute_fn: Callable                  # () -> ArmResult
    priority: float = 1.0                 # initial priority
    # Thompson sampling state
    successes: int = 1                    # Beta prior: alpha
    failures: int = 1                     # Beta prior: beta
    total_budget_s: float = 0.0
    total_score: float = 0.0
    n_runs: int = 0
    best_score: float = float("-inf")

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.successes + self.failures, 1)

    @property
    def thompson_sample(self) -> float:
        """Draw from Beta posterior for Thompson Sampling."""
        return np.random.beta(self.successes, self.failures)


@dataclass
class ArmResult:
    """Result from a single arm execution."""
    name: str
    score: float
    success: bool
    estimator: Any = None
    technique: str = ""
    elapsed_s: float = 0.0
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class PortfolioResult:
    """Result from a full portfolio fan-out."""
    best_arm: str
    best_score: float
    best_estimator: Any = None
    best_technique: str = ""
    arms_executed: int = 0
    arms_succeeded: int = 0
    total_elapsed_s: float = 0.0
    arm_results: List[ArmResult] = field(default_factory=list)
    budget_allocation: Dict[str, float] = field(default_factory=dict)


class Portfolio:
    """Portfolio manager: Thompson Sampling over multiple research arms.

    Usage:
        portfolio = Portfolio(max_workers=4)
        portfolio.add_arm("hist_gbm", lambda: train_and_eval_hist_gbm())
        portfolio.add_arm("random_forest", lambda: train_and_eval_rf())
        portfolio.add_arm("llm_proposal", lambda: run_llm_proposal())
        result = portfolio.execute(time_budget_s=60, max_rounds=5)
    """

    def __init__(self, max_workers: int = 4, time_budget_s: float = 300.0):
        self.max_workers = max_workers
        self.time_budget_s = time_budget_s
        self.arms: List[PortfolioArm] = []
        self._history: List[ArmResult] = []

    def add_arm(self, name: str, execute_fn: Callable,
                priority: float = 1.0) -> None:
        """Add a research arm to the portfolio."""
        self.arms.append(PortfolioArm(
            name=name, execute_fn=execute_fn, priority=priority,
        ))

    def execute(self, *, time_budget_s: Optional[float] = None,
                max_rounds: int = 5, top_k: int = 3) -> PortfolioResult:
        """Execute the portfolio with Thompson Sampling allocation.

        Each round:
          1. Sample from each arm's Beta posterior
          2. Select top_k arms to run this round
          3. Execute selected arms in parallel
          4. Update posteriors based on results
          5. Repeat until budget exhausted or max_rounds reached

        Args:
            time_budget_s: Total time budget (defaults to self.time_budget_s)
            max_rounds: Maximum portfolio rounds
            top_k: How many arms to run per round

        Returns:
            PortfolioResult with best arm and full history
        """
        if time_budget_s is None:
            time_budget_s = self.time_budget_s

        t0 = time.time()
        best_score = float("-inf")
        best_arm_name = ""
        best_estimator = None
        best_technique = ""
        all_results: List[ArmResult] = []

        for round_num in range(max_rounds):
            elapsed = time.time() - t0
            if elapsed >= time_budget_s:
                break

            # Thompson Sampling: select top_k arms
            selected = self._select_arms(top_k)
            if not selected:
                break

            # Execute selected arms in parallel
            round_budget = (time_budget_s - elapsed) / max(max_rounds - round_num, 1)
            results = self._execute_parallel(selected, round_budget)

            # Update posteriors and track results
            for result in results:
                all_results.append(result)
                arm = self._get_arm(result.name)
                if arm is None:
                    continue
                arm.n_runs += 1
                arm.total_budget_s += result.elapsed_s
                if result.success:
                    arm.successes += 1
                    arm.total_score += result.score
                    arm.best_score = max(arm.best_score, result.score)
                    if result.score > best_score:
                        best_score = result.score
                        best_arm_name = result.name
                        best_estimator = result.estimator
                        best_technique = result.technique
                else:
                    arm.failures += 1

            self._history.extend(results)

        return PortfolioResult(
            best_arm=best_arm_name,
            best_score=best_score,
            best_estimator=best_estimator,
            best_technique=best_technique,
            arms_executed=len(all_results),
            arms_succeeded=sum(1 for r in all_results if r.success),
            total_elapsed_s=time.time() - t0,
            arm_results=all_results,
            budget_allocation={
                arm.name: arm.total_budget_s for arm in self.arms
            },
        )

    def _select_arms(self, k: int) -> List[PortfolioArm]:
        """Select top-k arms using Thompson Sampling."""
        if not self.arms:
            return []
        # Draw from each arm's posterior
        samples = [(arm, arm.thompson_sample * arm.priority) for arm in self.arms]
        # Sort by sampled value
        samples.sort(key=lambda x: x[1], reverse=True)
        return [arm for arm, _ in samples[:k]]

    def _execute_parallel(self, arms: List[PortfolioArm],
                          budget_s: float) -> List[ArmResult]:
        """Execute arms in parallel with thread pool."""
        results: List[ArmResult] = []

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(arms))) as executor:
            futures: Dict[Future, PortfolioArm] = {}
            for arm in arms:
                future = executor.submit(self._run_arm_safe, arm, budget_s)
                futures[future] = arm

            for future in as_completed(futures, timeout=budget_s + 30):
                try:
                    result = future.result(timeout=5)
                    results.append(result)
                except Exception as e:
                    arm = futures[future]
                    results.append(ArmResult(
                        name=arm.name, score=float("-inf"),
                        success=False, error=str(e),
                    ))

        return results

    def _run_arm_safe(self, arm: PortfolioArm, budget_s: float) -> ArmResult:
        """Execute a single arm with error handling."""
        t0 = time.time()
        try:
            result = arm.execute_fn()
            if isinstance(result, ArmResult):
                result.elapsed_s = time.time() - t0
                return result
            # If execute_fn returns a score directly
            return ArmResult(
                name=arm.name,
                score=float(result) if result is not None else float("-inf"),
                success=result is not None,
                elapsed_s=time.time() - t0,
            )
        except Exception as e:
            return ArmResult(
                name=arm.name,
                score=float("-inf"),
                success=False,
                error=f"{type(e).__name__}: {e}",
                elapsed_s=time.time() - t0,
            )

    def _get_arm(self, name: str) -> Optional[PortfolioArm]:
        for arm in self.arms:
            if arm.name == name:
                return arm
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "n_arms": len(self.arms),
            "total_executions": sum(a.n_runs for a in self.arms),
            "arms": [{
                "name": a.name,
                "success_rate": a.success_rate,
                "best_score": a.best_score,
                "n_runs": a.n_runs,
                "budget_used_s": a.total_budget_s,
            } for a in self.arms],
        }
