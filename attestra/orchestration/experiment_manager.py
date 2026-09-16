"""Complex experiment management: meta-experiments, phase splitting, DAGs.

Handles experiments that are too complex for a single research cycle:
  1. PHASE SPLITTING: Break long experiments into phases (baseline → search → refine)
  2. META-EXPERIMENTS: Experiments-of-experiments (e.g., "which augmentation works best?")
  3. DAG EXECUTION: Dependencies between sub-experiments
  4. BUDGET ALLOCATION: Time/compute budget across phases
  5. PROGRESSIVE COMPLEXITY: Start simple, increase complexity only if needed

The key insight: complex ML projects aren't single experiments.
They're DAGs of sub-experiments with dependencies and budget constraints.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExperimentComplexity(str, Enum):
    """Complexity tier determines experiment strategy."""
    SIMPLE = "simple"           # Single model, single dataset, < 5 min
    MEDIUM = "medium"           # Multiple models, hyperparameter search, < 1 hour
    COMPLEX = "complex"         # Architecture search, augmentation, < 1 day
    RESEARCH = "research"       # Novel approaches, literature review, < 1 week
    FRONTIER = "frontier"       # Multi-week, multi-GPU, novel research


@dataclass
class ExperimentPhase:
    """A single phase in a multi-phase experiment."""
    name: str
    description: str
    phase_type: str                       # "baseline" | "search" | "refine" | "validate" | "ablation"
    # Budget
    time_budget_s: float = 60.0
    max_rounds: int = 5
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # phase names
    # State
    status: PhaseStatus = PhaseStatus.PENDING
    result: Optional[Dict] = None
    best_score: float = float("-inf")
    best_technique: str = ""
    elapsed_s: float = 0.0
    # Configuration
    config: Dict = field(default_factory=dict)

    @property
    def is_done(self) -> bool:
        return self.status in (PhaseStatus.COMPLETED, PhaseStatus.FAILED, PhaseStatus.SKIPPED)


@dataclass
class ExperimentPlan:
    """A plan for a complex experiment with multiple phases."""
    experiment_id: str
    goal: str
    complexity: ExperimentComplexity
    phases: List[ExperimentPhase] = field(default_factory=list)
    # Budget
    total_time_budget_s: float = 3600.0
    # State
    current_phase_idx: int = 0
    overall_best_score: float = float("-inf")
    overall_best_technique: str = ""
    started_at: float = 0.0
    # Progress gates
    early_stop_threshold: float = 0.0     # stop if reached early
    min_improvement_per_phase: float = 0.01

    @property
    def current_phase(self) -> Optional[ExperimentPhase]:
        if 0 <= self.current_phase_idx < len(self.phases):
            return self.phases[self.current_phase_idx]
        return None

    @property
    def progress(self) -> float:
        if not self.phases:
            return 0.0
        done = sum(1 for p in self.phases if p.is_done)
        return done / len(self.phases)

    @property
    def elapsed_s(self) -> float:
        return sum(p.elapsed_s for p in self.phases)


class ExperimentManager:
    """Manages complex multi-phase experiments.

    Usage:
        mgr = ExperimentManager()
        
        # Plan a complex experiment
        plan = mgr.plan_experiment(goal, complexity="complex", time_budget_s=3600)
        
        # Execute phases in order
        for phase in mgr.next_phases(plan):
            result = run_phase(phase)
            mgr.complete_phase(plan, phase, result)
        
        # Get overall result
        result = mgr.finalize(plan)
    """

    # Standard phase templates by complexity
    PHASE_TEMPLATES = {
        ExperimentComplexity.SIMPLE: [
            {"name": "baseline", "type": "baseline", "budget_frac": 1.0, "rounds": 5},
        ],
        ExperimentComplexity.MEDIUM: [
            {"name": "baseline", "type": "baseline", "budget_frac": 0.2, "rounds": 3},
            {"name": "search", "type": "search", "budget_frac": 0.6, "rounds": 10},
            {"name": "refine", "type": "refine", "budget_frac": 0.2, "rounds": 5},
        ],
        ExperimentComplexity.COMPLEX: [
            {"name": "baseline", "type": "baseline", "budget_frac": 0.1, "rounds": 3},
            {"name": "augmentation", "type": "search", "budget_frac": 0.15, "rounds": 5},
            {"name": "architecture_search", "type": "search", "budget_frac": 0.3, "rounds": 15},
            {"name": "hyperparameter_tuning", "type": "refine", "budget_frac": 0.25, "rounds": 10},
            {"name": "ensemble", "type": "refine", "budget_frac": 0.1, "rounds": 5},
            {"name": "validation", "type": "validate", "budget_frac": 0.1, "rounds": 1},
        ],
        ExperimentComplexity.RESEARCH: [
            {"name": "literature_review", "type": "baseline", "budget_frac": 0.05, "rounds": 1},
            {"name": "baseline", "type": "baseline", "budget_frac": 0.1, "rounds": 5},
            {"name": "novel_approach_1", "type": "search", "budget_frac": 0.2, "rounds": 10},
            {"name": "novel_approach_2", "type": "search", "budget_frac": 0.2, "rounds": 10},
            {"name": "best_of_breed", "type": "refine", "budget_frac": 0.15, "rounds": 10},
            {"name": "ablation", "type": "ablation", "budget_frac": 0.1, "rounds": 5},
            {"name": "scaling", "type": "search", "budget_frac": 0.1, "rounds": 5},
            {"name": "final_validation", "type": "validate", "budget_frac": 0.1, "rounds": 1},
        ],
        ExperimentComplexity.FRONTIER: [
            {"name": "literature_review", "type": "baseline", "budget_frac": 0.03, "rounds": 1},
            {"name": "landscape_mapping", "type": "baseline", "budget_frac": 0.05, "rounds": 3},
            {"name": "baseline_suite", "type": "baseline", "budget_frac": 0.07, "rounds": 5},
            {"name": "approach_a", "type": "search", "budget_frac": 0.15, "rounds": 15},
            {"name": "approach_b", "type": "search", "budget_frac": 0.15, "rounds": 15},
            {"name": "approach_c", "type": "search", "budget_frac": 0.15, "rounds": 15},
            {"name": "fusion", "type": "refine", "budget_frac": 0.1, "rounds": 10},
            {"name": "ablation_study", "type": "ablation", "budget_frac": 0.08, "rounds": 5},
            {"name": "scaling_study", "type": "search", "budget_frac": 0.07, "rounds": 5},
            {"name": "robustness_check", "type": "validate", "budget_frac": 0.05, "rounds": 3},
            {"name": "final_certification", "type": "validate", "budget_frac": 0.1, "rounds": 1},
        ],
    }

    def __init__(self):
        self._experiments: Dict[str, ExperimentPlan] = {}

    def plan_experiment(self, goal: str, complexity: str = "medium",
                        time_budget_s: float = 3600.0,
                        early_stop_threshold: float = 0.0) -> ExperimentPlan:
        """Create a multi-phase experiment plan.

        Args:
            goal: Research goal
            complexity: "simple"|"medium"|"complex"|"research"|"frontier"
            time_budget_s: Total time budget
            early_stop_threshold: Stop if this score is reached
        """
        comp = ExperimentComplexity(complexity)
        templates = self.PHASE_TEMPLATES[comp]

        phases = []
        prev_name = None
        for tmpl in templates:
            phase = ExperimentPhase(
                name=tmpl["name"],
                description=f"{tmpl['type']} phase for {goal[:50]}",
                phase_type=tmpl["type"],
                time_budget_s=time_budget_s * tmpl["budget_frac"],
                max_rounds=tmpl["rounds"],
                depends_on=[prev_name] if prev_name else [],
            )
            phases.append(phase)
            prev_name = tmpl["name"]

        exp_id = hashlib.sha256(f"{goal}{time.time()}".encode()).hexdigest()[:12]
        plan = ExperimentPlan(
            experiment_id=exp_id,
            goal=goal,
            complexity=comp,
            phases=phases,
            total_time_budget_s=time_budget_s,
            early_stop_threshold=early_stop_threshold,
            started_at=time.time(),
        )
        self._experiments[exp_id] = plan
        return plan

    def next_phases(self, plan: ExperimentPlan) -> List[ExperimentPhase]:
        """Get the next phases ready to run (dependencies satisfied)."""
        ready = []
        completed_names = {p.name for p in plan.phases if p.is_done}
        for phase in plan.phases:
            if phase.is_done or phase.status == PhaseStatus.RUNNING:
                continue
            deps_met = all(d in completed_names for d in phase.depends_on)
            if deps_met:
                ready.append(phase)
        return ready

    def start_phase(self, plan: ExperimentPlan, phase: ExperimentPhase) -> None:
        """Mark a phase as running."""
        phase.status = PhaseStatus.RUNNING

    def complete_phase(self, plan: ExperimentPlan, phase: ExperimentPhase,
                       result: Dict) -> bool:
        """Complete a phase and update plan state.

        Returns:
            True if experiment should continue, False if early-stop triggered
        """
        phase.status = PhaseStatus.COMPLETED
        phase.result = result
        phase.best_score = result.get("best_score", float("-inf"))
        phase.best_technique = result.get("best_technique", "")
        phase.elapsed_s = result.get("elapsed_s", 0.0)

        # Update plan
        if phase.best_score > plan.overall_best_score:
            plan.overall_best_score = phase.best_score
            plan.overall_best_technique = phase.best_technique

        # Check early stop
        if (plan.early_stop_threshold > 0 and
                plan.overall_best_score >= plan.early_stop_threshold):
            # Skip remaining phases
            for p in plan.phases:
                if p.status == PhaseStatus.PENDING:
                    p.status = PhaseStatus.SKIPPED
            return False

        # Advance phase index
        plan.current_phase_idx = next(
            (i for i, p in enumerate(plan.phases) if not p.is_done),
            len(plan.phases)
        )
        return True

    def fail_phase(self, plan: ExperimentPlan, phase: ExperimentPhase,
                   error: str) -> None:
        """Mark a phase as failed."""
        phase.status = PhaseStatus.FAILED
        phase.result = {"error": error}

    def classify_complexity(self, n_samples: int, n_features: int,
                            task_type: str, goal: str) -> ExperimentComplexity:
        """Auto-classify experiment complexity from data + goal."""
        # Heuristics
        goal_lower = goal.lower()

        # Frontier indicators
        if any(w in goal_lower for w in ["novel", "state of the art", "sota",
                                          "frontier", "week", "month"]):
            return ExperimentComplexity.FRONTIER

        # Research indicators
        if any(w in goal_lower for w in ["research", "architecture", "new model",
                                          "tts", "generation", "transformer"]):
            return ExperimentComplexity.RESEARCH

        # Size-based classification
        if n_samples > 50000 or n_features > 500:
            return ExperimentComplexity.COMPLEX
        elif n_samples > 5000 or n_features > 100:
            return ExperimentComplexity.COMPLEX
        elif n_samples > 1000 or n_features > 20:
            return ExperimentComplexity.MEDIUM
        else:
            return ExperimentComplexity.SIMPLE

    def summary(self, plan: ExperimentPlan) -> Dict:
        return {
            "experiment_id": plan.experiment_id,
            "goal": plan.goal,
            "complexity": plan.complexity.value,
            "progress": plan.progress,
            "current_phase": plan.current_phase.name if plan.current_phase else "done",
            "overall_best_score": plan.overall_best_score,
            "overall_best_technique": plan.overall_best_technique,
            "elapsed_s": plan.elapsed_s,
            "time_budget_s": plan.total_time_budget_s,
            "phases": [{
                "name": p.name,
                "type": p.phase_type,
                "status": p.status.value,
                "best_score": p.best_score,
                "elapsed_s": p.elapsed_s,
            } for p in plan.phases],
        }


# ============================================================================== meta-experiments

@dataclass
class MetaExperimentResult:
    """Result of a meta-experiment (experiment-of-experiments)."""
    question: str                          # what we're testing
    conditions: List[Dict]                 # each condition's config + result
    winner: str                            # name of best condition
    effect_size: float                     # how much better the winner is
    confidence: float                      # statistical confidence
    recommendation: str                    # action recommendation


class MetaExperimentRunner:
    """Run experiments-of-experiments.

    Tests hypotheses like:
      - "Does SMOTE help on this dataset?"
      - "Is HistGBM better than RF for this problem?"
      - "Does feature selection improve accuracy?"

    Each condition is a full sub-experiment. The meta-experiment compares them.
    """

    def __init__(self, run_fn: Callable):
        """
        Args:
            run_fn: Function that takes config dict and returns result dict
                    with at least {"score": float, "elapsed_s": float}
        """
        self.run_fn = run_fn
        self._results: List[MetaExperimentResult] = []

    def test_hypothesis(self, question: str,
                        conditions: Dict[str, Dict],
                        n_repeats: int = 3) -> MetaExperimentResult:
        """Test a hypothesis by running multiple conditions.

        Args:
            question: What we're testing (e.g., "Does SMOTE help?")
            conditions: {"control": {...config...}, "treatment": {...config...}}
            n_repeats: How many times to repeat each condition

        Returns:
            MetaExperimentResult with winner and confidence
        """
        from scipy import stats

        condition_scores: Dict[str, List[float]] = {}

        for name, config in conditions.items():
            scores = []
            for i in range(n_repeats):
                config_with_seed = {**config, "seed": 42 + i}
                try:
                    result = self.run_fn(config_with_seed)
                    scores.append(result.get("score", 0.0))
                except Exception:
                    scores.append(float("-inf"))
            condition_scores[name] = scores

        # Statistical comparison
        names = list(condition_scores.keys())
        all_scores = [condition_scores[n] for n in names]

        # Find best condition
        means = {n: float(np.mean(s)) for n, s in condition_scores.items()
                 if s and all(sc != float("-inf") for sc in s)}
        if not means:
            return MetaExperimentResult(
                question=question, conditions=[], winner="none",
                effect_size=0, confidence=0, recommendation="All conditions failed",
            )

        import numpy as np
        winner = max(means, key=lambda n: means[n])

        # Effect size (Cohen's d if 2 conditions)
        effect_size = 0.0
        confidence = 0.0
        if len(names) == 2 and all(len(condition_scores[n]) > 1 for n in names):
            s1 = np.array(condition_scores[names[0]])
            s2 = np.array(condition_scores[names[1]])
            pooled_std = np.sqrt((np.var(s1) + np.var(s2)) / 2)
            if pooled_std > 0:
                effect_size = float((np.mean(s1) - np.mean(s2)) / pooled_std)
            # t-test
            t_stat, p_val = stats.ttest_ind(s1, s2)
            confidence = float(1 - p_val)
        elif len(names) > 2:
            # ANOVA
            valid = [np.array(condition_scores[n]) for n in names
                     if len(condition_scores[n]) > 1]
            if len(valid) >= 2:
                f_stat, p_val = stats.f_oneway(*valid)
                confidence = float(1 - p_val)

        # Build condition summaries
        cond_summaries = []
        for name in names:
            scores = condition_scores[name]
            cond_summaries.append({
                "name": name,
                "scores": scores,
                "mean": float(np.mean(scores)) if scores else 0,
                "std": float(np.std(scores)) if scores else 0,
            })

        recommendation = f"Use '{winner}'"
        if confidence < 0.8:
            recommendation += " (low confidence — run more repeats)"

        result = MetaExperimentResult(
            question=question,
            conditions=cond_summaries,
            winner=winner,
            effect_size=abs(effect_size),
            confidence=confidence,
            recommendation=recommendation,
        )
        self._results.append(result)
        return result
