"""Strategy Loop (Loop 2) — the outer retry with escalation.

When the inner research cycle fails to certify, this loop:
1. Diagnoses WHY the strategy failed (oracle veto, plateau, execution failure, etc.)
2. Escalates to a different approach via the escalation ladder
3. Retries with adjusted config, carrying knowledge forward
4. Manages budget allocation across attempts

This is the "loop forever until result" capability:
    while budget_remaining and not (certified or honest_decline):
        result = run_inner_cycle(strategy, budget_slice)
        if certified: break
        diagnosis = diagnose_strategy_failure(result)
        strategy = escalate(diagnosis)

The loop terminates on:
  - Certification (success)
  - Budget exhaustion (honest decline with full explanation)
  - Escalation ladder exhausted (all strategies tried, honest decline)
  - Hard failure (infeasible problem detected)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Strategy representation
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """A research strategy — the approach for one attempt."""
    name: str
    engine_tier: str = "frontier"  # "frontier" | "generative" | "catalog"
    # Inner loop config overrides
    max_rounds: int = 6
    time_budget_s: float = 120.0
    # Strategy-specific params
    model_families: Optional[List[str]] = None  # restrict proposal space
    feature_strategy: str = "auto"  # "auto" | "aggressive" | "minimal" | "pca" | "none"
    exploration_rate: float = 0.3   # 0.0=pure exploitation, 1.0=pure exploration
    ensemble: bool = False          # ensemble top-K from prior attempts
    prior_winners: List[Dict] = field(default_factory=list)  # carry forward
    # Constraints
    avoid_families: List[str] = field(default_factory=list)
    force_multi_feature: bool = False  # oracle veto on single-feature -> force
    seed_pinning: bool = False         # oracle veto on reproducibility -> pin
    # Metadata
    rung: int = 0                     # position on escalation ladder
    attempt: int = 0                  # which attempt this is


@dataclass
class StrategyDiagnosis:
    """Why a strategy failed — structured, actionable."""
    failure_mode: str  # "oracle_veto" | "plateau" | "execution_failure" | "budget_exhausted"
                       # | "no_proposals" | "threshold_gap" | "statistical_miss"
    oracle_veto_reason: Optional[str] = None  # which oracle check vetoed
    plateau_rounds: int = 0           # how many rounds with no improvement
    execution_failure_rate: float = 0.0  # fraction of proposals that crashed
    dominant_error: Optional[str] = None  # most common error type
    best_score_achieved: float = 0.0
    threshold_gap: float = 0.0        # how far below threshold
    sealed_lower_bound: Optional[float] = None
    # What was tried
    families_tried: List[str] = field(default_factory=list)
    n_proposals: int = 0
    n_successful: int = 0
    # Actionable recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class StrategyLoopResult:
    """Full result from the strategy loop."""
    # Final outcome
    decision: str = "error"  # "certified" | "do_not_certify" | "honest_decline" | "error"
    best_score: float = 0.0
    best_technique: str = ""
    certificate: Optional[Dict] = None
    # Loop metadata
    n_attempts: int = 0
    strategies_tried: List[str] = field(default_factory=list)
    diagnoses: List[StrategyDiagnosis] = field(default_factory=list)
    total_elapsed_s: float = 0.0
    total_budget_used_s: float = 0.0
    budget_remaining_s: float = 0.0
    # Per-attempt results (for analysis)
    attempt_results: List[Dict] = field(default_factory=list)
    # The final inner result (for downstream consumers)
    inner_result: Optional[Any] = None
    # Knowledge accumulated across attempts
    accumulated_knowledge: Dict = field(default_factory=dict)
    # Escalation history
    escalation_history: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy Loop Config
# ---------------------------------------------------------------------------

@dataclass
class StrategyLoopConfig:
    """Configuration for the strategy loop."""
    # Total budget for all attempts
    total_budget_s: float = 1800.0  # 30 minutes default
    # Max attempts before giving up
    max_attempts: int = 8
    # Budget allocation: fraction of remaining budget per attempt
    budget_fraction: float = 0.4  # each attempt gets 40% of remaining
    min_attempt_budget_s: float = 30.0  # minimum budget per attempt
    # Exploration/exploitation balance
    initial_exploration: float = 0.3
    exploration_decay: float = 0.8  # multiply exploration by this each attempt
    # Early stopping
    patience: int = 3  # stop after N attempts with no score improvement
    min_improvement: float = 0.005  # minimum score improvement to count as progress


# ---------------------------------------------------------------------------
# Escalation Ladder
# ---------------------------------------------------------------------------

ESCALATION_LADDER = [
    # (rung, name, description, cost_multiplier)
    (0, "baseline", "Standard frontier with default config", 1.0),
    (1, "more_rounds", "Same strategy, more rounds + time", 1.5),
    (2, "different_hyperparams", "Different hyperparameter ranges", 1.2),
    (3, "different_families", "Switch model families", 1.5),
    (4, "aggressive_features", "Aggressive feature engineering", 2.0),
    (5, "ensemble_top_k", "Ensemble top-K from prior attempts", 1.8),
    (6, "neural_architectures", "Switch to neural/deep models", 3.0),
    (7, "data_representation", "Different data representation entirely", 2.5),
]


# ---------------------------------------------------------------------------
# Diagnosis engine
# ---------------------------------------------------------------------------

def diagnose_strategy_failure(result: Any, strategy: Strategy, threshold: float) -> StrategyDiagnosis:
    """Diagnose WHY a strategy failed to certify.

    Reads the inner result and produces a structured StrategyDiagnosis
    that maps to escalation actions.
    """
    diag = StrategyDiagnosis(failure_mode="unknown")

    # Extract fields from OrchestrateResult or similar
    decision = getattr(result, 'decision', '') or ''
    cert = getattr(result, 'certificate', None) or {}
    adversarial = getattr(result, 'adversarial', None) or {}
    n_proposals = getattr(result, 'n_proposals', 0)
    n_successful = getattr(result, 'n_successful', 0)
    n_failed = getattr(result, 'n_failed', 0)
    best_score = getattr(result, 'best_score', 0.0)
    history = getattr(result, 'history', []) or []
    failure_report = getattr(result, 'failure_report', None) or {}

    diag.best_score_achieved = best_score
    diag.n_proposals = n_proposals
    diag.n_successful = n_successful
    diag.threshold_gap = max(0, threshold - best_score)
    diag.sealed_lower_bound = cert.get("lower_bound") if cert else None

    # Collect families tried
    families = set()
    for h in history:
        tech = h.get("technique", "") if isinstance(h, dict) else getattr(h, 'label', '')
        if tech:
            families.add(tech.split("_")[0] if "_" in tech else tech)
    diag.families_tried = sorted(families)

    # Determine failure mode
    oracle_verdict = adversarial.get("oracle_verdict") if adversarial else None

    if oracle_verdict and not oracle_verdict.get("promote", True):
        # Oracle vetoed
        diag.failure_mode = "oracle_veto"
        reasons = oracle_verdict.get("reasons", [])
        diag.oracle_veto_reason = reasons[0] if reasons else "unknown oracle check"
        diag.recommendations = _oracle_veto_recommendations(oracle_verdict)

    elif n_proposals > 0 and n_failed > 0 and n_failed / max(n_proposals, 1) > 0.7:
        # Most proposals crashed
        diag.failure_mode = "execution_failure"
        diag.execution_failure_rate = n_failed / max(n_proposals, 1)
        # Find dominant error
        errors = [h.get("error", "") for h in history
                  if isinstance(h, dict) and h.get("status") == "failed"]
        if errors:
            from collections import Counter
            diag.dominant_error = Counter(errors).most_common(1)[0][0]
        diag.recommendations = [
            "switch_execution_substrate",
            "simplify_proposals",
            "increase_timeout",
        ]

    elif n_proposals == 0 or n_successful == 0:
        # No proposals generated or none succeeded
        diag.failure_mode = "no_proposals"
        diag.recommendations = [
            "expand_search_space",
            "lower_constraints",
            "switch_engine_tier",
        ]

    elif best_score > 0 and diag.threshold_gap > 0.1:
        # Score is far from threshold
        diag.failure_mode = "threshold_gap"
        diag.recommendations = [
            "aggressive_feature_engineering",
            "try_neural",
            "ensemble_top_k",
        ]

    elif best_score > 0 and diag.threshold_gap > 0:
        # Close to threshold but sealed test didn't clear
        diag.failure_mode = "statistical_miss"
        diag.recommendations = [
            "more_rounds",
            "more_data",
            "ensemble_for_stability",
        ]

    else:
        # General plateau
        diag.failure_mode = "plateau"
        diag.recommendations = [
            "different_families",
            "feature_engineering",
            "exploration_boost",
        ]

    return diag


def _oracle_veto_recommendations(verdict: Dict) -> List[str]:
    """Map oracle veto reasons to escalation recommendations."""
    reasons = verdict.get("reasons", [])
    recs = []
    for reason in reasons:
        r = reason.lower()
        if "single" in r or "feature" in r or "column" in r:
            recs.append("force_multi_feature")
        elif "reproduc" in r:
            recs.append("seed_pinning")
        elif "trivial" in r or "baseline" in r:
            recs.append("beat_trivial_baseline")
        elif "drift" in r or "distribution" in r:
            recs.append("check_data_quality")
        elif "permut" in r or "label" in r:
            recs.append("check_label_integrity")
        elif "refut" in r or "adversarial" in r:
            recs.append("force_multi_feature")
        else:
            recs.append("different_approach")
    return recs or ["different_approach"]


# ---------------------------------------------------------------------------
# Escalation engine
# ---------------------------------------------------------------------------

def escalate(
    diagnosis: StrategyDiagnosis,
    current_strategy: Strategy,
    attempt: int,
    prior_results: List[Dict],
) -> Strategy:
    """Given a diagnosis, produce the next strategy (escalate on the ladder).

    Carries forward knowledge from prior attempts and targets the specific
    weakness identified by the diagnosis.
    """
    next_rung = current_strategy.rung + 1

    # Collect winners from prior attempts for ensembling
    prior_winners = []
    for pr in prior_results:
        if pr.get("best_score", 0) > 0:
            prior_winners.append({
                "technique": pr.get("best_technique", ""),
                "score": pr.get("best_score", 0),
                "attempt": pr.get("attempt", 0),
            })

    # Build next strategy based on diagnosis
    next_strategy = Strategy(
        name=f"escalation_rung_{next_rung}",
        rung=next_rung,
        attempt=attempt + 1,
        prior_winners=prior_winners,
    )

    mode = diagnosis.failure_mode

    if mode == "oracle_veto":
        # Target the specific oracle weakness
        next_strategy.name = f"oracle_fix_{diagnosis.oracle_veto_reason or 'unknown'}"
        if "force_multi_feature" in diagnosis.recommendations:
            next_strategy.force_multi_feature = True
        if "seed_pinning" in diagnosis.recommendations:
            next_strategy.seed_pinning = True
        # Don't move up the ladder for oracle fixes — same rung, different constraints
        next_strategy.rung = current_strategy.rung
        next_strategy.max_rounds = current_strategy.max_rounds + 2
        next_strategy.time_budget_s = current_strategy.time_budget_s * 1.3

    elif mode == "execution_failure":
        # Simplify: fewer features, more time, avoid crashing families
        next_strategy.name = "simplified_execution"
        next_strategy.feature_strategy = "minimal"
        next_strategy.time_budget_s = current_strategy.time_budget_s * 1.5
        next_strategy.avoid_families = list(diagnosis.families_tried)[:3]

    elif mode == "no_proposals":
        # Expand search: more exploration, different engine
        next_strategy.name = "expanded_search"
        next_strategy.exploration_rate = min(0.8, current_strategy.exploration_rate + 0.3)
        if current_strategy.engine_tier == "frontier":
            next_strategy.engine_tier = "generative"  # try a different engine
        next_strategy.max_rounds = current_strategy.max_rounds + 3

    elif mode == "threshold_gap":
        # Far from goal: aggressive escalation
        if next_rung <= 4:
            next_strategy.name = "aggressive_features"
            next_strategy.feature_strategy = "aggressive"
            next_strategy.max_rounds = current_strategy.max_rounds + 4
            next_strategy.time_budget_s = current_strategy.time_budget_s * 2.0
        elif next_rung == 5:
            next_strategy.name = "ensemble_top_k"
            next_strategy.ensemble = True
            next_strategy.max_rounds = current_strategy.max_rounds
        else:
            next_strategy.name = "neural_escalation"
            next_strategy.model_families = ["neural", "deep_tabular", "transformer"]
            next_strategy.time_budget_s = current_strategy.time_budget_s * 3.0

    elif mode == "statistical_miss":
        # Close to goal: more samples, more rounds, ensemble for stability
        next_strategy.name = "stability_push"
        next_strategy.max_rounds = current_strategy.max_rounds + 3
        next_strategy.ensemble = len(prior_winners) >= 2
        next_strategy.time_budget_s = current_strategy.time_budget_s * 1.5

    elif mode == "plateau":
        # Stuck: switch families, boost exploration
        if next_rung <= 3:
            next_strategy.name = "family_switch"
            next_strategy.avoid_families = list(diagnosis.families_tried)
            next_strategy.exploration_rate = min(0.7, current_strategy.exploration_rate + 0.2)
        elif next_rung <= 5:
            next_strategy.name = "ensemble_diverse"
            next_strategy.ensemble = True
            next_strategy.feature_strategy = "aggressive"
        else:
            next_strategy.name = "deep_exploration"
            next_strategy.model_families = ["neural", "deep_tabular"]
            next_strategy.exploration_rate = 0.8
            next_strategy.time_budget_s = current_strategy.time_budget_s * 2.5

    else:
        # Unknown: generic escalation
        next_strategy.name = f"generic_rung_{next_rung}"
        next_strategy.max_rounds = current_strategy.max_rounds + 2
        next_strategy.exploration_rate = min(0.7, current_strategy.exploration_rate + 0.1)
        next_strategy.time_budget_s = current_strategy.time_budget_s * 1.3

    return next_strategy


# ---------------------------------------------------------------------------
# The Strategy Loop
# ---------------------------------------------------------------------------

def run_strategy_loop(
    config: Any,  # OrchestrateConfig
    loop_config: Optional[StrategyLoopConfig] = None,
    orchestrate_fn: Optional[Callable] = None,
) -> StrategyLoopResult:
    """Run the full strategy loop: retry with escalation until certified or budget exhausted.

    Parameters
    ----------
    config : OrchestrateConfig
        The base configuration for the research run.
    loop_config : StrategyLoopConfig, optional
        Configuration for the loop itself (budget, max attempts, etc.).
    orchestrate_fn : callable, optional
        The inner orchestrate function to call. Defaults to the module-level orchestrate().

    Returns
    -------
    StrategyLoopResult
        Complete result including all attempts, diagnoses, and escalation history.
    """
    if loop_config is None:
        loop_config = StrategyLoopConfig(total_budget_s=config.time_budget_s * 3)

    if orchestrate_fn is None:
        from .orchestrator import orchestrate
        orchestrate_fn = orchestrate

    t0 = time.time()
    loop_result = StrategyLoopResult()
    budget_remaining = loop_config.total_budget_s
    best_score_seen = 0.0
    patience_counter = 0

    # Consult meta-learner for initial strategy bias
    meta_avoid: List[str] = []
    meta_preferred: List[str] = []
    try:
        from ..improvement.meta_learner import MetaLearner
        _ml = MetaLearner(persist_path=os.path.join(os.path.expanduser("~/.attestra"), "meta_learner.json"))
        _profile = getattr(config, 'X', None)
        _n_samples = len(_profile) if _profile is not None else 0
        _n_features = _profile.shape[1] if hasattr(_profile, 'shape') and len(getattr(_profile, 'shape', ())) > 1 else 0
        _task_type = ""
        try:
            from ..intake.profiler import profile_data
            import numpy as _np
            _p = profile_data(_np.asarray(config.X), _np.asarray(config.y))
            _task_type = _p.task_type
        except Exception:
            pass
        meta_recs = _ml.rank_strategies(_task_type, _n_samples, _n_features)
        meta_avoid = _ml.avoid_list(_task_type)
        meta_preferred = [name for name, _ in meta_recs[:5]]
    except Exception:
        pass

    # Initial strategy (seeded with meta-learner guidance)
    current_strategy = Strategy(
        name="baseline",
        rung=0,
        attempt=0,
        max_rounds=min(config.max_rounds, 6),
        time_budget_s=min(config.time_budget_s, budget_remaining * loop_config.budget_fraction),
        exploration_rate=loop_config.initial_exploration,
        avoid_families=meta_avoid,
        model_families=meta_preferred if meta_preferred else None,
    )

    prior_results: List[Dict] = []
    verbose = getattr(config, 'verbose', True)
    if verbose and (meta_avoid or meta_preferred):
        print(f"[strategy_loop] Meta-learner seeded: prefer={meta_preferred[:3]}, avoid={meta_avoid}")

    for attempt in range(loop_config.max_attempts):
        if budget_remaining < loop_config.min_attempt_budget_s:
            if verbose:
                print(f"[strategy_loop] Budget exhausted ({budget_remaining:.1f}s remaining). "
                      f"Stopping after {attempt} attempts.")
            break

        # Allocate budget for this attempt
        attempt_budget = max(
            loop_config.min_attempt_budget_s,
            min(budget_remaining * loop_config.budget_fraction, budget_remaining - 10.0),
        )

        # Apply strategy to config
        attempt_config = _apply_strategy_to_config(config, current_strategy, attempt_budget)

        if verbose:
            print(f"\n[strategy_loop] ═══ Attempt {attempt + 1}/{loop_config.max_attempts} "
                  f"═══ strategy='{current_strategy.name}' "
                  f"rung={current_strategy.rung} "
                  f"budget={attempt_budget:.0f}s "
                  f"remaining={budget_remaining:.0f}s")

        # Run the inner cycle
        attempt_t0 = time.time()
        try:
            inner_result = orchestrate_fn(attempt_config)
        except Exception as e:
            if verbose:
                print(f"[strategy_loop]   Inner cycle crashed: {e}")
            inner_result = type('MockResult', (), {
                'decision': 'error', 'best_score': 0.0, 'best_technique': '',
                'certificate': None, 'n_proposals': 0, 'n_successful': 0,
                'n_failed': 0, 'history': [], 'adversarial': None,
                'failure_report': {'error': str(e)},
            })()

        attempt_elapsed = time.time() - attempt_t0
        budget_remaining -= attempt_elapsed
        loop_result.n_attempts += 1
        loop_result.strategies_tried.append(current_strategy.name)

        # Record attempt result
        attempt_record = {
            "attempt": attempt,
            "strategy": current_strategy.name,
            "rung": current_strategy.rung,
            "decision": getattr(inner_result, 'decision', 'error'),
            "best_score": getattr(inner_result, 'best_score', 0.0),
            "best_technique": getattr(inner_result, 'best_technique', ''),
            "elapsed_s": attempt_elapsed,
            "budget_used_s": attempt_elapsed,
        }
        loop_result.attempt_results.append(attempt_record)
        prior_results.append(attempt_record)

        # Check if certified
        decision = getattr(inner_result, 'decision', '')
        if decision == "certified":
            if verbose:
                print(f"[strategy_loop]   ✓ CERTIFIED on attempt {attempt + 1} "
                      f"(strategy='{current_strategy.name}')")
            loop_result.decision = "certified"
            loop_result.best_score = getattr(inner_result, 'best_score', 0.0)
            loop_result.best_technique = getattr(inner_result, 'best_technique', '')
            loop_result.certificate = getattr(inner_result, 'certificate', None)
            loop_result.inner_result = inner_result
            break

        # Track progress for patience
        current_best = getattr(inner_result, 'best_score', 0.0) or 0.0
        if current_best > best_score_seen + loop_config.min_improvement:
            best_score_seen = current_best
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= loop_config.patience:
            if verbose:
                print(f"[strategy_loop]   No improvement for {patience_counter} attempts. "
                      f"Escalating aggressively or stopping.")
            # One last aggressive attempt before giving up (max 1 reset to prevent infinite loop)
            if current_strategy.rung < len(ESCALATION_LADDER) - 1 and not getattr(loop_result, '_patience_reset_used', False):
                patience_counter = 0
                loop_result._patience_reset_used = True
            else:
                break

        # Diagnose failure
        threshold = getattr(config, 'threshold', 0.0) or 0.0
        diagnosis = diagnose_strategy_failure(inner_result, current_strategy, threshold)
        loop_result.diagnoses.append(diagnosis)

        if verbose:
            print(f"[strategy_loop]   Diagnosis: {diagnosis.failure_mode} | "
                  f"score={diagnosis.best_score_achieved:.4f} | "
                  f"gap={diagnosis.threshold_gap:.4f}")
            if diagnosis.recommendations:
                print(f"[strategy_loop]   Recommendations: {diagnosis.recommendations[:3]}")

        # Record escalation
        loop_result.escalation_history.append({
            "from_strategy": current_strategy.name,
            "from_rung": current_strategy.rung,
            "diagnosis": diagnosis.failure_mode,
            "oracle_veto": diagnosis.oracle_veto_reason,
        })

        # Check if ladder is exhausted
        if current_strategy.rung >= len(ESCALATION_LADDER) - 1:
            if verbose:
                print("[strategy_loop]   Escalation ladder exhausted. Honest decline.")
            break

        # Escalate
        current_strategy = escalate(diagnosis, current_strategy, attempt, prior_results)

        # Decay exploration over attempts
        current_strategy.exploration_rate *= loop_config.exploration_decay

    # Finalize result
    loop_result.total_elapsed_s = time.time() - t0
    loop_result.total_budget_used_s = loop_config.total_budget_s - budget_remaining
    loop_result.budget_remaining_s = max(0, budget_remaining)

    if loop_result.decision != "certified":
        # Find the best result across all attempts
        best_attempt = max(prior_results, key=lambda x: x.get("best_score", 0)) if prior_results else {}
        loop_result.best_score = best_attempt.get("best_score", 0.0)
        loop_result.best_technique = best_attempt.get("best_technique", "")
        loop_result.decision = "honest_decline"

    # Accumulate knowledge for future loops
    loop_result.accumulated_knowledge = {
        "strategies_tried": loop_result.strategies_tried,
        "best_families": [d.families_tried for d in loop_result.diagnoses],
        "failure_modes": [d.failure_mode for d in loop_result.diagnoses],
        "best_score": loop_result.best_score,
        "n_attempts": loop_result.n_attempts,
    }

    if verbose:
        print(f"\n[strategy_loop] ═══ DONE ═══ "
              f"decision={loop_result.decision} | "
              f"best={loop_result.best_score:.4f} | "
              f"{loop_result.n_attempts} attempts | "
              f"{loop_result.total_elapsed_s:.1f}s total")

    return loop_result


def _apply_strategy_to_config(base_config: Any, strategy: Strategy, budget_s: float) -> Any:
    """Create a modified config that applies the strategy's parameters.

    Returns a new config object (shallow copy with overrides) so the
    original config is not mutated.
    """
    from .orchestrator import OrchestrateConfig

    # Build a new config with strategy overrides
    new_config = OrchestrateConfig(
        goal=base_config.goal,
        X=base_config.X,
        y=base_config.y,
        metric=base_config.metric,
        threshold=base_config.threshold,
        max_rounds=strategy.max_rounds,
        time_budget_s=budget_s,
        llm_call=base_config.llm_call,
        api_key=base_config.api_key,
        use_retrieval=base_config.use_retrieval,
        gpu=base_config.gpu,
        feature_names=base_config.feature_names,
        seed=base_config.seed + strategy.attempt,  # different seed each attempt
        verbose=base_config.verbose,
        on_event=base_config.on_event,
        registry_path=base_config.registry_path,
        strategy_path=base_config.strategy_path,
        deployment=base_config.deployment,
        checkpoint_dir=base_config.checkpoint_dir,
        use_augmentation=base_config.use_augmentation,
        complexity=base_config.complexity,
        use_generative=base_config.use_generative,
        use_literature=base_config.use_literature,
    )

    # Apply engine tier override
    if strategy.engine_tier == "catalog":
        new_config.use_generative = False
    elif strategy.engine_tier == "generative":
        # Force generative but not frontier (handled internally)
        new_config.use_generative = True

    # Inject meta-learner guidance into goal context
    if strategy.avoid_families or strategy.model_families:
        guidance_parts = []
        if strategy.avoid_families:
            guidance_parts.append(
                f"AVOID these approaches (historically underperform): {', '.join(strategy.avoid_families)}"
            )
        if strategy.model_families:
            guidance_parts.append(
                f"PREFER these approaches (historically strong): {', '.join(strategy.model_families)}"
            )
        if strategy.force_multi_feature:
            guidance_parts.append("REQUIREMENT: use multiple features (single-feature solutions will be vetoed)")
        new_config.goal = base_config.goal + "\n\n[STRATEGY GUIDANCE: " + ". ".join(guidance_parts) + "]"

    return new_config


# ---------------------------------------------------------------------------
# Strategy generation from accumulated evidence (Loop 3 integration)
# ---------------------------------------------------------------------------

def _generate_strategies_from_evidence(config, result) -> None:
    """Generate new strategies using StrategyGenerator from accumulated patterns.

    Called after each experiment to allow the system to discover
    and codify new strategy rules from accumulated evidence.
    """
    from ..improvement.self_improvement import StrategyGenerator
    from ..improvement.strategy_learner import StrategyLearner

    learner = StrategyLearner(config.strategy_path)
    generator = StrategyGenerator()

    # Convert learner outcomes into the format StrategyGenerator expects
    outcomes = learner._load()
    meta_data = [
        {
            "task_type": o.task_type,
            "strategy": o.strategy_name,
            "success": o.success,
            "score": o.score if o.score is not None else 0.0,
            "n_samples": o.n_samples,
            "n_features": o.n_features,
        }
        for o in outcomes
    ]

    if len(meta_data) >= 5:
        new_strategies = generator.analyze_and_generate(meta_data, min_evidence=5)
        if new_strategies and config.verbose:
            print(f"[attestra]   Strategy generator: {len(new_strategies)} new rules discovered")
