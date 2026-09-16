"""The unified orchestrator: the FULL pipeline wiring all modules.

Single entry point for end-to-end autonomous ML research.

Full pipeline:
  1. INTAKE: profile + problem typing + adversarial check
  2. PRE-REGISTRATION: content-addressed experiment plan
  3. EXPERIMENT DESIGN: complexity classification + phase planning
  4. ORCHESTRATION: run phases, each phase runs the research engine
  5. CERTIFICATION: frozen certifier + cross-experiment FDR
  6. LEDGER: positive AND negative certificates
  7. SELF-IMPROVEMENT: meta-learner + strategy learner
  8. DEPLOYMENT: tradeoff analysis for winning model
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..cycle.engine import CycleResult, ResearchEngine
from ..cycle.generative import GenerativeEngine, GenerativeResult
from ..intake.profiler import DataProfile, profile_data
from ..intake.problem_typing import (
    ProblemSpec, adversarial_check, type_problem,
)
from ..design.experiment_plan import ExperimentPlan
from ..design.tradeoff import DeploymentConstraints, TradeoffEngine
from ..certification.fdr import ExperimentCertificate, FDRController
from ..execution.checkpoint import CheckpointManager, ExperimentState
from ..augmentation.augmentation import DataAugmenter
from ..orchestration.experiment_manager import ExperimentManager
from ..improvement.meta_learner import Experience, MetaLearner
from ..ledger.registry import ExperimentRecord, ExperimentRegistry, fingerprint_dataset
from ..improvement.strategy_learner import StrategyLearner, StrategyOutcome
from ..data.versioning import DataFingerprint, DataVersion
from ..data.experiments import DataExperimentManager
from ..observability.events import EventEmitter, EventType
from ..observability.provenance import ProvenanceTracker
from ..routing.strategy_router import StrategyRouter
from ..routing.resource_router import ResourceRouter
from ..routing.verification_router import VerificationRouter
from ..improvement.self_improvement import HarnessLibrary, OracleEvolver, PromptEvolver
from ..execution.parallel import ProposalPool, ProposalResult
from ..execution.torch_harness import TorchHarness, TrainConfig, TrainResult, detect_device
from ..execution.architecture_builder import ArchitectureBuilder, ArchitectureSpec
from ..execution.gpu_backend import (
    GPUStatus, build_llm_client, build_frontier_llm_client,
    check_gpu_status, dispatch_gpu_training, find_best_gpu_offer,
)


# ============================================================================== paths

_HOME = os.path.expanduser("~/.attestra")


def _path(name: str) -> str:
    os.makedirs(_HOME, exist_ok=True)
    return os.path.join(_HOME, name)


def _diag_subsample(X, y, max_rows: int, task_type: Optional[str], seed: int):
    """Stratified subsample for cheap pre-loop diagnostics.

    Returns ``(X, y)`` unchanged when ``max_rows`` is falsy or the data already
    fits. Otherwise draws a class-stratified (classification) or uniform
    (regression) sample of ``max_rows`` rows so the augmentation / data-
    experiment stages estimate lift on a representative slice instead of the
    full matrix. This never touches the data used for the actual search or
    certification.
    """
    n = int(X.shape[0])
    if not max_rows or n <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    if task_type in ("binary", "multiclass") or (
        task_type is None and y.dtype.kind in ("i", "u", "b")
    ):
        classes, inv = np.unique(y, return_inverse=True)
        idx_parts = []
        for ci in range(len(classes)):
            cls_idx = np.where(inv == ci)[0]
            take = max(1, int(round(len(cls_idx) * max_rows / n)))
            take = min(take, len(cls_idx))
            idx_parts.append(rng.choice(cls_idx, size=take, replace=False))
        idx = np.concatenate(idx_parts)
    else:
        idx = rng.choice(n, size=max_rows, replace=False)
    rng.shuffle(idx)
    return X[idx], y[idx]


# ============================================================================== config / result

@dataclass
class OrchestrateConfig:
    """Configuration for a single orchestrated research run."""
    goal: str
    X: Any                               # feature matrix
    y: Any                               # target vector
    metric: Optional[str] = None
    threshold: Optional[float] = None
    max_rounds: int = 15
    time_budget_s: float = 300.0
    # LLM
    llm_call: Optional[Callable] = None
    api_key: Optional[str] = None
    # Opt-in for live LLM calls. When False (default), the orchestrator runs
    # offline even if PRIME_INTELLECT_API_KEY is present in the environment.
    # This keeps the test suite hermetic (no live network calls / hangs) and
    # makes LLM usage an explicit decision of the caller (the CLI sets this).
    use_llm: bool = False
    use_retrieval: bool = True
    # GPU
    gpu: bool = False
    # Metadata
    feature_names: Optional[List[str]] = None
    seed: int = 42
    verbose: bool = True
    on_event: Optional[Callable] = None
    # Persistence paths
    registry_path: Optional[str] = None
    strategy_path: Optional[str] = None
    # New: deployment constraints
    deployment: Optional[DeploymentConstraints] = None
    # New: checkpoint dir
    checkpoint_dir: Optional[str] = None
    # New: enable augmentation proposals
    use_augmentation: bool = True
    # Diagnostic-stage row cap. The pre-loop exploratory stages (augmentation
    # recommendation, data-as-variable experiments) only need a representative
    # sample to estimate lift — running them on the full matrix makes large
    # datasets (100K+ rows) spend minutes fitting throwaway GBMs. We cap them
    # at a stratified subsample; the actual search/certification still sees the
    # full data. Set to 0 to disable the cap.
    diag_max_rows: int = 10000
    # New: complexity override
    complexity: Optional[str] = None  # "simple"|"medium"|"complex"|"research"|"frontier"
    # New: use generative engine (LLM writes complete solutions, not catalog)
    use_generative: bool = True
    # New: use literature search (arXiv/PapersWithCode)
    use_literature: bool = True
    # New: run data-as-variable experiments (augment/clean/resplit hypothesis testing)
    use_data_experiments: bool = True
    # New: parallel proposal execution (uses ProposalPool for batch evaluation)
    use_parallel: bool = True
    parallel_workers: int = 4
    # Engine selection: "auto" (default: frontier > generative > catalog),
    # "frontier", "generative", "catalog"
    engine: str = "auto"


@dataclass
class OrchestrateResult:
    """Full result from an orchestrated research run."""
    # Core result
    decision: str                         # "certified" | "do_not_certify" | "honest_stop" | "error"
    best_score: float = 0.0
    best_technique: str = ""
    certificate: Optional[Dict] = None
    # The plan that was executed
    plan: Optional[ExperimentPlan] = None
    profile: Optional[DataProfile] = None
    # Intake
    problem_spec: Optional[ProblemSpec] = None
    adversarial: Optional[Dict] = None
    # Cycle details
    n_proposals: int = 0
    n_successful: int = 0
    n_failed: int = 0
    elapsed_s: float = 0.0
    # Reports
    failure_report: Optional[Dict] = None
    error_summary: Optional[Dict] = None
    health_summary: Optional[Dict] = None
    history: List[Dict] = field(default_factory=list)
    # Solution tree (generative engine)
    tree_summary: Optional[Dict] = None
    engine_type: str = ""  # "generative" or "catalog"
    # FDR
    fdr_decision: Optional[Dict] = None
    # Tradeoff
    tradeoff_analysis: Optional[str] = None
    recommended_model: Optional[str] = None
    # Phase tracking
    phases_completed: int = 0
    phases_total: int = 0
    # Registry
    registry_updated: bool = False
    strategy_updated: bool = False
    meta_learner_updated: bool = False
    checkpoint_saved: bool = False
    # Data experiments
    data_experiments_run: int = 0
    data_experiments_accepted: int = 0
    data_best_transform: Optional[str] = None


def orchestrate(config: OrchestrateConfig) -> OrchestrateResult:
    """Execute a full orchestrated research run.

    This is THE entry point for Attestra. Give it a goal and data, it does the rest.

    Full pipeline:
      INTAKE -> PRE-REGISTRATION -> EXPERIMENT DESIGN -> ORCHESTRATION
      -> RECURSIVE CYCLE -> CERTIFICATION -> FDR -> LEDGER -> SELF-IMPROVEMENT
    """
    t0 = time.time()
    result = OrchestrateResult(decision="error")

    # Setup LLM — use unified GPU backend for smart model selection.
    # Only build a live client when the caller explicitly opted in (use_llm)
    # or provided an llm_call/api_key directly. We never auto-enable from the
    # ambient env var alone, so test runs stay offline and deterministic.
    llm_call = config.llm_call
    if llm_call is None and config.use_llm:
        key = config.api_key or os.environ.get("PRIME_INTELLECT_API_KEY")
        if key:
            llm_call = build_llm_client(api_key=key)

    try:
        X = np.asarray(config.X)
        y = np.asarray(config.y)

        # ================================================================ OBSERVABILITY SETUP
        emitter = EventEmitter(
            experiment_id=f"exp_{int(t0)}",
            file_path=os.path.join(_HOME, "events.jsonl") if config.verbose else None,
            callback=config.on_event,
        )
        provenance = ProvenanceTracker(f"exp_{int(t0)}")
        provenance.record_environment()
        provenance.record_data(X, y, seed=config.seed)
        provenance.record_start()

        # ================================================================ DATA VERSIONING
        data_version = DataVersion.create(
            X, y, description=f"Input data for: {config.goal[:80]}"
        )

        emitter.experiment_start(config.goal, {
            "max_rounds": config.max_rounds,
            "time_budget_s": config.time_budget_s,
            "data_version": data_version.version_id,
            "data_fingerprint": data_version.fingerprint.hash_hex[:16],
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]) if X.ndim > 1 else 1,
        })
        emitter.update_budget(config.time_budget_s)

        # ================================================================ STAGE 1: INTAKE
        if config.verbose:
            print("[attestra] Stage 1: INTAKE — profiling + problem typing")

        profile = profile_data(X, y, feature_names=config.feature_names)
        result.profile = profile

        # Problem typing (LLM or heuristic)
        data_profile_dict = {
            "task_type": profile.task_type,
            "n_samples": profile.n_samples,
            "n_features": profile.n_features,
            "n_classes": profile.n_classes,
            "quality_score": profile.quality_score,
            "issues": profile.issues,
        }
        problem_spec = type_problem(
            config.goal, data_profile=data_profile_dict, llm_call=llm_call,
        )
        result.problem_spec = problem_spec

        if config.verbose:
            print(f"[attestra]   Domain: {problem_spec.domain.value} "
                  f"(confidence={problem_spec.confidence:.2f})")
            print(f"[attestra]   Metric: {problem_spec.suggested_metric}, "
                  f"feasibility: {problem_spec.feasibility.value}")

        # Adversarial check
        adv = adversarial_check(config.goal, data_profile=data_profile_dict, llm_call=llm_call)
        result.adversarial = {
            "has_issues": adv.has_issues,
            "overall_risk": adv.overall_risk.value,
            "contradictions": adv.contradictions,
            "infeasibilities": adv.infeasibilities,
            "reward_hacking_risks": adv.reward_hacking_risks,
            "recommendations": adv.recommendations,
        }
        if config.verbose and adv.has_issues:
            print(f"[attestra]   Adversarial: risk={adv.overall_risk.value}, "
                  f"{len(adv.infeasibilities)} infeasibilities, "
                  f"{len(adv.reward_hacking_risks)} reward-hacking risks")

        # Use problem spec to override metric/threshold if not explicitly set
        if config.metric is None and problem_spec.suggested_metric:
            config.metric = problem_spec.suggested_metric
        if config.threshold is None and problem_spec.suggested_threshold > 0:
            config.threshold = problem_spec.suggested_threshold

        # ================================================================ STAGE 2: EXPERIMENT DESIGN
        if config.verbose:
            print("[attestra] Stage 2: EXPERIMENT DESIGN — complexity + phases")

        exp_mgr = ExperimentManager()
        if config.complexity:
            complexity = config.complexity
        else:
            complexity = exp_mgr.classify_complexity(
                profile.n_samples, profile.n_features,
                profile.task_type, config.goal,
            ).value

        exp_plan = exp_mgr.plan_experiment(
            config.goal, complexity=complexity,
            time_budget_s=config.time_budget_s,
        )
        result.phases_total = len(exp_plan.phases)
        result.plan = exp_plan  # available for all engine paths

        if config.verbose:
            print(f"[attestra]   Complexity: {complexity}, "
                  f"{len(exp_plan.phases)} phases")

        # Representative slice for the cheap pre-loop diagnostic stages so large
        # datasets don't spend minutes fitting throwaway models on full data.
        X_diag, y_diag = _diag_subsample(
            X, y, config.diag_max_rows, profile.task_type, config.seed
        )
        if config.verbose and X_diag.shape[0] < X.shape[0]:
            print(f"[attestra]   Diagnostics on stratified subsample: "
                  f"{X.shape[0]} -> {X_diag.shape[0]} rows "
                  f"(full data used for search/certification)")

        # ================================================================ STAGE 3: DATA AUGMENTATION
        aug_data = None
        if config.use_augmentation and profile.task_type in ("binary", "multiclass"):
            augmenter = DataAugmenter(X_diag, y_diag, task="classification", seed=config.seed)
            recommended = augmenter.recommend_strategies()
            if recommended:
                try:
                    aug_result = augmenter.apply(recommended[0])
                    aug_data = (aug_result.X_aug, aug_result.y_aug)
                    if config.verbose:
                        print(f"[attestra]   Augmentation: {recommended[0]} "
                              f"({aug_result.n_generated} synthetic samples)")
                except Exception:
                    pass

        # ================================================================ STAGE 3b: DATA EXPERIMENTS
        data_transforms_to_apply = []
        if config.use_data_experiments:
            data_exp_mgr = DataExperimentManager()
            data_proposals = data_exp_mgr.propose_experiments(
                X_diag, y_diag, task_type=profile.task_type, llm_call=llm_call,
            )
            if data_proposals and config.verbose:
                print(f"[attestra] Stage 3b: DATA EXPERIMENTS — "
                      f"{len(data_proposals)} hypotheses to test")

            if data_proposals:
                from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
                from sklearn.metrics import accuracy_score, r2_score

                def _model_fn(X_tr, y_tr):
                    if profile.task_type == "regression":
                        m = GradientBoostingRegressor(n_estimators=50, random_state=config.seed)
                    else:
                        m = GradientBoostingClassifier(n_estimators=50, random_state=config.seed)
                    m.fit(X_tr, y_tr)
                    return m

                def _eval_fn(model, X_te, y_te):
                    preds = model.predict(X_te)
                    if profile.task_type == "regression":
                        return r2_score(y_te, preds)
                    return accuracy_score(y_te, preds)

                completed = data_exp_mgr.run_experiments(
                    data_proposals, X_diag, y_diag, _model_fn, _eval_fn, seed=config.seed,
                )
                result.data_experiments_run = len(completed)
                accepted = [e for e in completed if e.decision == "accept"]
                result.data_experiments_accepted = len(accepted)

                # Use best accepted transform to augment training data
                best_transforms = data_exp_mgr.best_transforms()
                if best_transforms:
                    data_transforms_to_apply = best_transforms[:1]
                    result.data_best_transform = best_transforms[0].name
                    if config.verbose:
                        print(f"[attestra]   Best data transform: {best_transforms[0].name} "
                              f"(lift={accepted[0].lift*100:+.1f}%)")

                if config.verbose:
                    print(f"[attestra]   Data experiments: {len(accepted)}/{len(completed)} accepted")

        # Apply winning data transforms to training data
        if data_transforms_to_apply:
            for transform in data_transforms_to_apply:
                try:
                    X, y = transform.apply(X, y)
                except Exception:
                    pass  # non-fatal, continue with original data

        # ================================================================ STAGE 4: CHECKPOINT SETUP
        ckpt_mgr = None
        resume_round = 0
        resume_state = None
        if config.checkpoint_dir:
            ckpt_mgr = CheckpointManager(
                exp_plan.experiment_id, base_dir=config.checkpoint_dir,
            )
            # Resume from checkpoint if available — skip completed rounds
            if ckpt_mgr.has_checkpoint():
                resume_state, _ = ckpt_mgr.load_latest()
                if resume_state is not None:
                    resume_round = resume_state.round_num + 1  # start AFTER last completed
                    if config.verbose:
                        print(f"[attestra]   Resuming from round {resume_round} "
                              f"(completed {resume_state.round_num}), "
                              f"best={resume_state.best_score:.4f}")

        # ================================================================ STAGE 5: META-LEARNER + ROUTING + REGISTRY READ-BACK
        meta_learner = MetaLearner(
            persist_path=_path("meta_learner.json"),
        )

        # Get strategy recommendations from meta-learner
        meta_recs = meta_learner.rank_strategies(
            profile.task_type, profile.n_samples, profile.n_features,
        )
        avoid = meta_learner.avoid_list(profile.task_type)
        if config.verbose and meta_recs:
            top3 = [(n, f"{s:.3f}") for n, s in meta_recs[:3]]
            print(f"[attestra]   Meta-learner top: {top3}")
            print(f"[attestra]   (meta-learner guidance computed; consumed by strategy router)")
        if config.verbose and avoid:
            print(f"[attestra]   Avoid: {avoid}")

        # ─── Strategy Router: route to optimal strategy class ───
        strategy_router = StrategyRouter(meta_learner=meta_learner)
        strategy_recs = strategy_router.route(
            task_type=profile.task_type,
            n_samples=profile.n_samples,
            n_features=profile.n_features,
            n_classes=profile.n_classes,
            budget_remaining_s=config.time_budget_s,
            quality_score=profile.quality_score,
        )
        if config.verbose and strategy_recs:
            print(f"[attestra]   Strategy router: {strategy_recs[0].strategy_class.value} "
                  f"(confidence={strategy_recs[0].confidence:.2f})")

        # ─── GPU Status: check all backends (local + remote + inference) ───
        # Only query the remote provider (a network call) when GPU was requested;
        # otherwise inspect local backends only so test runs stay offline.
        _gpu_key = (config.api_key or os.environ.get("PRIME_INTELLECT_API_KEY")) if config.gpu else None
        gpu_status = check_gpu_status(api_key=_gpu_key)
        _effective_gpu = config.gpu or gpu_status.any_gpu
        if config.verbose and gpu_status.api_key_set:
            print(f"[attestra]   GPU status: {gpu_status.summary()}")

        # ─── Resource Router: decide execution substrate ───
        resource_router = ResourceRouter(
            gpu_available=_effective_gpu,
            serverless_enabled=gpu_status.remote_available,
        )
        _strategy_cls = strategy_recs[0].strategy_class.value if strategy_recs else "gradient_boosting"
        resource_rec = resource_router.route(
            strategy_class=_strategy_cls,
            n_samples=profile.n_samples,
            n_features=profile.n_features,
            budget_s=config.time_budget_s,
        )
        if config.verbose:
            print(f"[attestra]   Resource router: {resource_rec.substrate.value} "
                  f"(workers={resource_rec.n_workers})")

        # ─── Registry Read-Back: cross-experiment knowledge transfer ───
        registry = ExperimentRegistry(config.registry_path)
        fp = fingerprint_dataset(X, y)
        prior_successes = registry.what_works(fp, task_type=profile.task_type)
        prior_failures = registry.what_fails(fp, task_type=profile.task_type)
        if config.verbose and prior_successes:
            print(f"[attestra]   Registry read-back: {len(prior_successes)} successful "
                  f"techniques on similar data")
        if config.verbose and prior_failures:
            print(f"[attestra]   Registry failures to avoid: {prior_failures[:5]}")

        # ─── Self-Improvement: load prompt evolver for proposal generation ───
        prompt_evolver = PromptEvolver(persist_path=_path("prompt_evolution.jsonl"))
        # Register a default template so record_outcome has a valid target
        _default_tid = prompt_evolver.register_template(
            template="Generate a machine learning solution for the given problem.",
            purpose="proposal_generation",
            source="default",
        )
        oracle_evolver = OracleEvolver(persist_path=_path("oracle_evolution.jsonl"))

        # ─── Harness Library: permanent storage of authored harnesses ───
        harness_library = HarnessLibrary(persist_path=_path("harness_library.jsonl"))
        authored_harness = harness_library.get_harness(profile.task_type)
        if authored_harness and config.verbose:
            print(f"[attestra]   Harness library: found '{authored_harness.harness_id}' "
                  f"for {profile.task_type} (uses={authored_harness.uses}, "
                  f"successes={authored_harness.successes})")

        # ================================================================ STAGE 6: RESEARCH CYCLE
        #
        # Three paths (in priority order):
        #   1. FRONTIER (primary): CoreOrchestrator with oracles, diagnosis feed-forward,
        #      agentic repair, knowledge/LinUCB, features, portfolio/budget, report.
        #      Works with AND without LLM (honest degradation).
        #   2. GENERATIVE (fallback): LLM writes complete solutions, diagnose->improve loop.
        #      Requires LLM. Falls back to catalog if LLM unavailable.
        #   3. CATALOG (no LLM): classic catalog-based proposal engine with phase splitting.
        #
        frontier_ok = False
        try:
            from frontier.core.orchestrator import CoreOrchestrator, CoreConfig as FrontierConfig
            frontier_ok = True
        except ImportError:
            pass

        # Engine selection: explicit choice or auto-routing
        _engine = config.engine.lower() if config.engine else "auto"
        if _engine == "frontier":
            use_frontier = frontier_ok
            use_gen = False
            if not frontier_ok and config.verbose:
                print("[attestra] WARNING: --engine frontier requested but frontier "
                      "package not installed. Falling back to catalog.")
        elif _engine == "generative":
            use_frontier = False
            use_gen = config.use_generative and llm_call is not None
            if not use_gen and config.verbose:
                print("[attestra] WARNING: --engine generative requested but LLM "
                      "unavailable. Falling back to catalog.")
        elif _engine == "catalog":
            use_frontier = False
            use_gen = False
        else:  # "auto": frontier > generative > catalog
            use_frontier = frontier_ok
            use_gen = config.use_generative and llm_call is not None

        if config.verbose and _engine != "auto":
            print(f"[attestra]   Engine override: {_engine}")

        if use_frontier:
            # ─────────────── FRONTIER PATH (primary) ───────────────
            if config.verbose:
                print("[attestra] Stage 6: FRONTIER RESEARCH CYCLE")
                print("[attestra]   (oracles + diagnosis + agentic repair + LinUCB + portfolio)")

            result.engine_type = "frontier"

            # Build LLM client adapter: frontier expects (prompt -> code), not (sys, user -> text).
            # Gated on the same opt-in as above so test runs never touch the network.
            frontier_llm = None
            if config.use_llm or config.llm_call is not None:
                frontier_llm = build_frontier_llm_client(
                    api_key=config.api_key or os.environ.get("PRIME_INTELLECT_API_KEY"),
                )
            if frontier_llm is None and llm_call is not None:
                # Fallback: wrap the (sys, user) -> text interface
                def _frontier_llm_adapter(prompt: str) -> str:
                    text, _ = llm_call(
                        "You are an expert ML engineer. Write clean Python code.",
                        prompt,
                    )
                    return text
                frontier_llm = _frontier_llm_adapter

            # Compute effective rounds (skip already-completed via checkpoint)
            effective_rounds = max(1, min(config.max_rounds, 6) - resume_round)

            frontier_cfg = FrontierConfig(
                rounds=effective_rounds,
                seed=config.seed + resume_round,  # advance seed past completed rounds
                wall_seconds=min(config.time_budget_s / max(config.max_rounds, 1), 120.0),
                cpu_seconds=int(min(config.time_budget_s / max(config.max_rounds, 1), 110.0)),
                llm_client=frontier_llm,
                total_seconds=config.time_budget_s,
                kb_path=_path("frontier_kb.jsonl"),
                checkpoint_path=_path("frontier_ckpt.json") if config.checkpoint_dir else None,
                report_dir=config.checkpoint_dir,
            )

            # Wire cross-experiment knowledge into frontier context:
            # meta-learner avoid list, registry prior successes/failures, strategy router
            meta_augmented_goal = config.goal
            _guidance_parts = []
            if avoid:
                _guidance_parts.append(
                    f"Avoid: {', '.join(avoid)}. "
                    f"Prefer: {', '.join(n for n, _ in meta_recs[:3]) if meta_recs else 'explore broadly'}"
                )
            if prior_successes:
                _top_prior = [(t, f"{s:.3f}") for t, s in list(prior_successes.items())[:3]]
                _guidance_parts.append(f"Prior successes on similar data: {_top_prior}")
            if prior_failures:
                _guidance_parts.append(f"Prior failures to skip: {prior_failures[:3]}")
            if strategy_recs:
                _guidance_parts.append(
                    f"Strategy router recommends: {strategy_recs[0].strategy_class.value} "
                    f"(confidence={strategy_recs[0].confidence:.2f})"
                )
            if _guidance_parts:
                meta_augmented_goal += "\n\n[CROSS-EXPERIMENT GUIDANCE: " + "; ".join(_guidance_parts) + "]"

            core_orch = CoreOrchestrator(frontier_cfg)
            core_result = core_orch.run(
                goal=meta_augmented_goal, X=X, y=y,
                theta=config.threshold or 0.0,
                metric=config.metric or "",
            )

            # Map CoreResult into OrchestrateResult
            best_overall_score = core_result.winner_val_score or 0.0
            best_overall_technique = core_result.winner.label if core_result.winner else ""
            best_overall_estimator = None  # frontier returns Programs, not estimators
            best_overall_cert = core_result.certificate
            total_proposals = len(core_result.history)
            total_successful = sum(1 for r in core_result.history if r.ok)
            total_failed = sum(1 for r in core_result.history if not r.ok)
            all_history = [
                {
                    "technique": r.label, "source": r.source,
                    "score": r.val_score, "status": "success" if r.ok else "failed",
                    "error": r.error,
                }
                for r in core_result.history
            ]
            result.phases_completed = 1

            # Carry frontier provenance
            if core_result.oracle_verdict:
                result.adversarial = {
                    "has_issues": not core_result.oracle_verdict.get("promote", True),
                    "overall_risk": "oracle_veto" if not core_result.oracle_verdict.get("promote", True) else "none",
                    "oracle_verdict": core_result.oracle_verdict,
                }

            if config.verbose:
                print(core_result.summary())

            result.meta_learner_updated = True

            # Emit observability events for frontier path
            emitter.update_score(best_overall_score)
            for r in core_result.history:
                emitter.proposal_result(
                    label=r.label, score=r.val_score,
                    success=r.ok, error=r.error or "",
                )

            # Self-improvement: record outcomes for prompt evolution
            for r in core_result.history:
                prompt_evolver.record_outcome(
                    _default_tid, r.val_score or 0.0, r.ok,
                )
            # Oracle evolution: record failure patterns
            if core_result.oracle_verdict and not core_result.oracle_verdict.get("promote", True):
                oracle_evolver.record_failure_pattern(
                    pattern="oracle_veto",
                    details=core_result.oracle_verdict,
                    oracle_caught=True,
                )

        elif use_gen and not use_frontier:
            # ─────────────── GENERATIVE PATH (fallback with LLM) ───────────────
            if config.verbose:
                print("[attestra] Stage 6: GENERATIVE RESEARCH CYCLE")

            result.engine_type = "generative"

            gen_engine = GenerativeEngine(
                X, y,
                goal=config.goal,
                metric=config.metric,
                threshold=config.threshold,
                max_rounds=config.max_rounds,
                time_budget_s=config.time_budget_s,
                llm_call=llm_call,
                use_literature=config.use_literature,
                feature_names=config.feature_names,
                seed=config.seed,
                verbose=config.verbose,
            )

            gen_result = gen_engine.run()

            # Map generative result into OrchestrateResult
            best_overall_score = gen_result.best_score
            best_overall_technique = gen_result.best_technique
            best_overall_estimator = gen_result.best_estimator
            best_overall_cert = gen_result.certificate
            total_proposals = gen_result.n_solutions
            total_successful = gen_result.n_successful
            total_failed = gen_result.n_failed
            all_history = gen_result.history
            result.phases_completed = 1
            result.tree_summary = gen_result.tree_summary

            # Meta-learner: observe from solution tree
            for sol_info in gen_result.history:
                if sol_info.get("score") is not None:
                    meta_learner.observe(Experience(
                        n_samples=profile.n_samples,
                        n_features=profile.n_features,
                        task_type=profile.task_type,
                        n_classes=profile.n_classes,
                        quality_score=profile.quality_score,
                        strategy=sol_info.get("name", ""),
                        source=sol_info.get("source", "generative"),
                        score=sol_info.get("score", 0.0),
                        improvement=0.0,
                        success=sol_info.get("score", 0) > 0,
                        elapsed_s=0.0,
                    ))
            result.meta_learner_updated = True

            # Checkpoint — record absolute round (resume_round + rounds completed)
            if ckpt_mgr:
                completed_round = resume_round + config.max_rounds
                state = ExperimentState(
                    experiment_id=exp_plan.experiment_id,
                    goal=config.goal,
                    round_num=completed_round,
                    best_score=best_overall_score,
                    best_technique=best_overall_technique,
                    tried_families=list({
                        h.get("technique", "").split("_")[0]
                        for h in all_history if h.get("technique")
                    }),
                    proposal_history=all_history,
                    total_elapsed_s=time.time() - t0,
                    time_budget_s=config.time_budget_s,
                )
                ckpt_mgr.save(state, model=best_overall_estimator)
                result.checkpoint_saved = True

        else:
            # ─────────────── CATALOG PATH (no LLM) ───────────────
            if config.verbose:
                print("[attestra] Stage 6: RECURSIVE CYCLE — running research phases")
                if config.use_parallel:
                    print(f"[attestra]   Parallel execution enabled (workers={config.parallel_workers})")

            result.engine_type = "catalog"

            best_overall_score = 0.0
            best_overall_technique = ""
            best_overall_estimator = None
            best_overall_cert = None
            total_proposals = 0
            total_successful = 0
            total_failed = 0
            all_history = []
            cycle_result = None  # initialized before loop in case no phases run

            # Initialize parallel pool for batch evaluation
            _pool = None
            if config.use_parallel:
                _pool = ProposalPool(
                    max_workers=config.parallel_workers,
                    cpu_seconds_per_proposal=min(int(config.time_budget_s / 4), 90),
                    mem_mb_per_proposal=2048,
                )

            # Inject meta-learner + registry knowledge into catalog goal context
            catalog_goal = config.goal
            _guidance_parts = []
            if avoid:
                _guidance_parts.append(
                    f"AVOID these approaches (historically underperform): {', '.join(avoid)}"
                )
            if meta_recs:
                _guidance_parts.append(
                    f"PREFER: {', '.join(n for n, _ in meta_recs[:3])}"
                )
            if prior_successes:
                _top_prior = sorted(prior_successes.items(), key=lambda x: -x[1])[:3]
                _guidance_parts.append(
                    f"Techniques that worked on similar data: {', '.join(f'{t} ({s:.3f})' for t, s in _top_prior)}"
                )
            if prior_failures:
                _guidance_parts.append(
                    f"Techniques that failed on similar data: {', '.join(prior_failures[:5])}"
                )
            if _guidance_parts:
                catalog_goal += "\n\n[STRATEGY GUIDANCE: " + ". ".join(_guidance_parts) + "]"

            # Checkpoint resume for catalog: skip completed phases
            _catalog_start_phase = 0
            if resume_state is not None and resume_round > 0:
                _catalog_start_phase = min(resume_round, len(exp_plan.phases) - 1)
                best_overall_score = resume_state.best_score
                best_overall_technique = resume_state.best_technique
                if config.verbose:
                    print(f"[attestra]   Catalog resume: skipping {_catalog_start_phase} completed phases")

            for phase_idx, phase in enumerate(exp_plan.phases):
                if phase_idx < _catalog_start_phase:
                    result.phases_completed += 1
                    continue
                # Global wall-clock guard: phases run sequentially, so stop
                # starting new phases once the total budget is exhausted.
                _remaining = config.time_budget_s - (time.time() - t0)
                if _remaining <= 0:
                    if config.verbose:
                        print(f"[attestra]   Time budget exhausted, skipping remaining phases")
                    break
                exp_mgr.start_phase(exp_plan, phase)
                # Clamp this phase's budget to what's globally left.
                phase_budget = min(phase.time_budget_s, _remaining)
                phase_rounds = phase.max_rounds

                if config.verbose:
                    print(f"[attestra]   Phase: {phase.name} ({phase.phase_type}, "
                          f"{phase_budget:.0f}s, {phase_rounds} rounds)")

                # Choose data: augmented for search phases, original for validate
                if aug_data and phase.phase_type == "search":
                    X_phase, y_phase = aug_data
                else:
                    X_phase, y_phase = X, y

                engine = ResearchEngine(
                    X_phase, y_phase,
                    goal=catalog_goal,
                    metric=config.metric,
                    threshold=config.threshold,
                    max_rounds=phase_rounds,
                    time_budget_s=phase_budget,
                    llm_call=llm_call,
                    use_retrieval=config.use_retrieval,
                    gpu=config.gpu,
                    feature_names=config.feature_names,
                    seed=config.seed,
                    verbose=config.verbose,
                    on_event=config.on_event,
                )

                cycle_result = engine.run()

                phase_result = {
                    "best_score": cycle_result.best_score,
                    "best_technique": cycle_result.best_technique,
                    "elapsed_s": cycle_result.elapsed_s,
                }
                cont = exp_mgr.complete_phase(exp_plan, phase, phase_result)

                total_proposals += cycle_result.n_proposals
                total_successful += cycle_result.n_successful
                total_failed += cycle_result.n_failed
                all_history.extend(cycle_result.history)
                result.phases_completed += 1

                if cycle_result.best_score > best_overall_score:
                    best_overall_score = cycle_result.best_score
                    best_overall_technique = cycle_result.best_technique
                    best_overall_estimator = cycle_result.best_estimator
                    best_overall_cert = cycle_result.certificate

                if ckpt_mgr:
                    state = ExperimentState(
                        experiment_id=exp_plan.experiment_id,
                        goal=config.goal,
                        round_num=result.phases_completed,
                        best_score=best_overall_score,
                        best_technique=best_overall_technique,
                        tried_families=list(engine.state.tried_families),
                        total_elapsed_s=time.time() - t0,
                        time_budget_s=config.time_budget_s,
                    )
                    ckpt_mgr.save(state, model=best_overall_estimator)
                    result.checkpoint_saved = True

                for ep in engine.evaluated.values():
                    meta_learner.observe(Experience(
                        n_samples=profile.n_samples,
                        n_features=profile.n_features,
                        task_type=profile.task_type,
                        n_classes=profile.n_classes,
                        quality_score=profile.quality_score,
                        strategy=ep.proposal.technique,
                        source=ep.proposal.source,
                        score=ep.val_score or 0.0,
                        improvement=(ep.val_score or 0.0) - best_overall_score
                        if ep.val_score and ep.val_score > best_overall_score else 0.0,
                        success=(ep.status == "success"),
                        elapsed_s=ep.elapsed_s,
                    ))
                result.meta_learner_updated = True

                # Parallel batch evaluation: supplementary proposals via pool
                if _pool and cycle_result.n_proposals > 0:
                    _batch_proposals = _build_batch_proposals(
                        engine, cycle_result, all_history,
                        prior_successes=prior_successes,
                        avoid=avoid,
                    )
                    if _batch_proposals:
                        _parallel_results = _pool.execute_batch(
                            _batch_proposals,
                            engine.X_train, engine.y_train,
                            engine.X_val, engine.y_val,
                            metric=engine.plan.verification.metric,
                            wall_timeout_s=max(30, phase_budget * 0.3),
                        )
                        for pr in _parallel_results:
                            if pr.success and pr.score is not None:
                                total_proposals += 1
                                total_successful += 1
                                all_history.append({
                                    "technique": pr.proposal_id,
                                    "source": "parallel_pool",
                                    "score": pr.score,
                                    "status": "success",
                                    "error": "",
                                })
                                if pr.score > best_overall_score:
                                    best_overall_score = pr.score
                                    best_overall_technique = pr.proposal_id
                                    if config.verbose:
                                        print(f"[attestra]   Parallel pool beat: {pr.score:.4f} "
                                              f"({pr.proposal_id})")
                            elif not pr.success:
                                total_proposals += 1
                                total_failed += 1
                        if config.verbose:
                            n_pool_ok = sum(1 for r in _parallel_results if r.success)
                            print(f"[attestra]   Parallel pool: {n_pool_ok}/{len(_parallel_results)} ok")

                if not cont:
                    if config.verbose:
                        print(f"[attestra]   Early stop: threshold reached")
                    break

        # ================================================================ STAGE 6b: GPU ESCALATION
        # Try GPU if: resource router recommended it, or explicitly requested, or
        # remote GPU is available and we haven't hit threshold.
        _gpu_substrates = {"local_gpu", "serverless_gpu", "parallel_gpu"}
        _should_try_gpu = (
            best_overall_score < (config.threshold or 1.0)
            and (
                (resource_rec.substrate.value in _gpu_substrates and _effective_gpu)
                or gpu_status.remote_available
            )
        )
        if _should_try_gpu:
            try:
                _torch_device = detect_device()
                _has_local_gpu = _torch_device != "cpu"
                _has_remote_gpu = gpu_status.remote_available

                if _has_local_gpu or _has_remote_gpu or config.gpu:
                    _gpu_source = "local" if _has_local_gpu else ("remote" if _has_remote_gpu else "cpu_fallback")
                    if config.verbose:
                        print(f"[attestra] Stage 6b: GPU ESCALATION — "
                              f"source={_gpu_source}, device={_torch_device}")

                    n_features = profile.n_features
                    if profile.task_type == "regression":
                        n_outputs = 1
                        _torch_task = "regression"
                    else:
                        n_outputs = profile.n_classes
                        _torch_task = "classification"

                    if _has_local_gpu:
                        # ─── LOCAL GPU PATH ───
                        arch_builder = ArchitectureBuilder()
                        arch_spec = ArchitectureSpec(
                            name=f"neural_{profile.task_type}",
                            task=_torch_task,
                            input_shape=(n_features,),
                            output_shape=(n_outputs,),
                            backbone="mlp",
                            hidden_dims=[256, 128, 64],
                            dropout=0.2,
                        )

                        if llm_call is not None:
                            built_arch = arch_builder.from_llm(arch_spec, llm_call)
                        else:
                            built_arch = arch_builder.from_spec(arch_spec)

                        if built_arch.validated and built_arch.module is not None:
                            train_cfg = TrainConfig(
                                architecture="custom",
                                epochs=50,
                                batch_size=min(64, len(X) // 4),
                                learning_rate=1e-3,
                                patience=8,
                                device=_torch_device,
                                mixed_precision=(_torch_device == "cuda"),
                                seed=config.seed,
                                verbose=config.verbose,
                            )
                            harness = TorchHarness(train_cfg)
                            harness.build_model(
                                n_features, n_outputs, task=_torch_task,
                                custom_model=built_arch.module,
                            )

                            from sklearn.model_selection import train_test_split
                            _X_tr, _X_val, _y_tr, _y_val = train_test_split(
                                X, y, test_size=0.2, random_state=config.seed,
                            )
                            torch_result = harness.train(_X_tr, _y_tr, _X_val, _y_val)

                            if config.metric and config.metric in torch_result.val_metrics:
                                _torch_score = torch_result.val_metrics[config.metric]
                            else:
                                _torch_score = 1.0 - torch_result.best_val_loss

                            if _torch_score > best_overall_score:
                                best_overall_score = _torch_score
                                best_overall_technique = f"neural_{built_arch.name}"
                            all_history.append({
                                "technique": f"neural_{built_arch.name}",
                                "source": "gpu_local",
                                "score": _torch_score,
                                "status": "success",
                                "error": "",
                            })
                            total_proposals += 1
                            total_successful += 1
                            if config.verbose:
                                print(f"[attestra]   Local GPU score: {_torch_score:.4f}")
                        elif config.verbose:
                            print(f"[attestra]   Local GPU: architecture validation failed")

                    elif _has_remote_gpu:
                        # ─── REMOTE GPU PATH (Prime Intellect) ───
                        if config.verbose:
                            print(f"[attestra]   Dispatching to remote GPU pod...")

                        # Generate neural training code via LLM or template
                        _neural_code = _build_neural_training_code(
                            profile.task_type, n_features, n_outputs, llm_call,
                        )
                        if _neural_code:
                            from sklearn.model_selection import train_test_split
                            _X_tr, _X_val, _y_tr, _y_val = train_test_split(
                                X, y, test_size=0.2, random_state=config.seed,
                            )
                            gpu_result = dispatch_gpu_training(
                                _neural_code, _X_tr, _y_tr, _X_val, _y_val,
                                api_key=config.api_key or os.environ.get("PRIME_INTELLECT_API_KEY"),
                                prefer_local=False,
                                max_remote_price_usd=5.0,
                                timeout_s=int(config.time_budget_s * 0.5),
                            )
                            if gpu_result.ok:
                                if gpu_result.score > best_overall_score:
                                    best_overall_score = gpu_result.score
                                    best_overall_technique = f"neural_remote_{gpu_result.gpu_type}"
                                all_history.append({
                                    "technique": f"neural_remote_{gpu_result.gpu_type}",
                                    "source": "gpu_remote",
                                    "score": gpu_result.score,
                                    "status": "success",
                                    "error": "",
                                })
                                total_proposals += 1
                                total_successful += 1
                                if config.verbose:
                                    print(f"[attestra]   Remote GPU score: {gpu_result.score:.4f} "
                                          f"({gpu_result.gpu_type}, {gpu_result.wall_seconds:.0f}s)")
                            else:
                                all_history.append({
                                    "technique": "neural_remote",
                                    "source": "gpu_remote",
                                    "score": None,
                                    "status": "failed",
                                    "error": gpu_result.error[:200],
                                })
                                total_proposals += 1
                                total_failed += 1
                                if config.verbose:
                                    print(f"[attestra]   Remote GPU failed: {gpu_result.error[:100]}")
                    else:
                        if config.verbose:
                            print(f"[attestra]   GPU escalation: no GPU available (local or remote)")
            except Exception as _gpu_err:
                if config.verbose:
                    print(f"[attestra]   GPU escalation failed: {_gpu_err}")
                all_history.append({
                    "technique": "neural_escalation",
                    "source": "gpu_escalation",
                    "score": None,
                    "status": "failed",
                    "error": str(_gpu_err)[:200],
                })
                total_proposals += 1
                total_failed += 1

        # ================================================================ STAGE 7: CERTIFICATION + FDR
        result.best_score = best_overall_score
        result.best_technique = best_overall_technique
        if result.engine_type == "catalog":
            result.plan = cycle_result.plan if cycle_result else None
        result.n_proposals = total_proposals
        result.n_successful = total_successful
        result.n_failed = total_failed
        result.elapsed_s = time.time() - t0
        result.history = all_history
        if result.engine_type == "catalog":
            result.error_summary = cycle_result.error_summary if cycle_result else None
            result.health_summary = cycle_result.health_summary if cycle_result else None

        # For frontier path, use the oracle-gated decision directly
        if result.engine_type == "frontier":
            is_certified = (best_overall_cert is not None
                            and best_overall_cert.get("certified")
                            and core_result.certified)
        else:
            is_certified = best_overall_cert and best_overall_cert.get("certified")

        if is_certified:
            result.decision = "certified"
            result.certificate = best_overall_cert

            # Feed to FDR controller
            fdr = FDRController(alpha=0.05, persist_path=_path("fdr_state.json"))
            p_value = best_overall_cert.get("p_value", 0.01)
            cert = ExperimentCertificate(
                experiment_id=exp_plan.experiment_id,
                metric=config.metric or "accuracy",
                threshold=config.threshold or 0.0,
                observed_lower_bound=best_overall_cert.get("lower_bound", best_overall_score),
                p_value=p_value if isinstance(p_value, (int, float)) else 0.01,
                certified_locally=True,
                technique=best_overall_technique,
                n_test=int(best_overall_cert.get("n", 0)),
            )
            fdr_decision = fdr.submit(cert)
            result.fdr_decision = {
                "accepted": fdr_decision.accepted,
                "adjusted_p_value": fdr_decision.adjusted_p_value,
                "method": fdr_decision.method,
                "stream_size": fdr.summary()["n_experiments"],
                "acceptance_rate": fdr.acceptance_rate(),
            }
            if config.verbose:
                print(f"[attestra]   FDR: accepted={fdr_decision.accepted}, "
                      f"adjusted_p={fdr_decision.adjusted_p_value:.4f}")
        elif best_overall_cert:
            result.decision = "do_not_certify"
            result.certificate = best_overall_cert
            decline = ""
            if result.engine_type == "frontier" and hasattr(core_result, 'decline_reason'):
                decline = core_result.decline_reason
            result.failure_report = {
                "dominant_source": decline or "sealed lower bound below threshold",
                "sealed_lower_bound": best_overall_cert.get("lower_bound"),
                "threshold": config.threshold,
            }
        else:
            result.decision = "honest_stop" if best_overall_score > 0 else "error"
            result.failure_report = {
                "dominant_source": "no certified result",
                "best_val_score": best_overall_score,
                "n_proposals": total_proposals,
                "n_failed": total_failed,
            }

        # ================================================================ STAGE 8: TRADEOFF ANALYSIS
        if best_overall_estimator is not None and config.deployment:
            try:
                tradeoff = TradeoffEngine()
                X_sample = X[:min(100, len(X))]
                tradeoff.profile_model(
                    best_overall_technique, best_overall_estimator,
                    X_sample, accuracy=best_overall_score,
                )
                tr_result = tradeoff.recommend(config.deployment)
                result.tradeoff_analysis = tr_result.analysis
                result.recommended_model = tr_result.recommended.name
            except Exception:
                pass

        # ================================================================ STAGE 9: LEDGER
        _record_to_registry(config, result)
        result.registry_updated = True

        # ================================================================ STAGE 10: STRATEGY LEARNER + GENERATION
        _update_strategy_learner(config, all_history, result)
        result.strategy_updated = True

        # Generate new strategies from accumulated patterns
        try:
            from .strategy_loop import _generate_strategies_from_evidence
            _generate_strategies_from_evidence(config, result)
        except Exception:
            pass

        # ================================================================ STAGE 11: SELF-IMPROVEMENT
        # Prompt evolution: try to evolve prompts if enough data
        if llm_call and prompt_evolver._templates:
            try:
                prompt_evolver.evolve("proposal_generation", llm_call=llm_call)
            except Exception:
                pass

        # Harness library: record usage and author new harnesses if needed
        if authored_harness:
            harness_library.record_use(
                authored_harness.harness_id,
                success=(result.decision == "certified"),
            )
        elif llm_call and result.decision != "error":
            # No existing harness — author one for future use
            try:
                new_harness = harness_library.author_harness(
                    profile.task_type, llm_call,
                    validation_data=(X[:min(100, len(X))], y[:min(100, len(y))]),
                )
                if new_harness and config.verbose:
                    print(f"[attestra]   Authored harness '{new_harness.harness_id}' "
                          f"for {profile.task_type}")
            except Exception:
                pass

        # Verification router: select appropriate oracle suite for this task
        verification_router = VerificationRouter()
        verification_suite = verification_router.route(
            problem_type=profile.task_type,
            n_samples=profile.n_samples,
            n_features=profile.n_features,
        )
        if config.verbose and verification_suite:
            print(f"[attestra]   Verification suite: {len(verification_suite.checks)} checks computed (advisory, post-certification)")

        # ================================================================ STAGE 12: PROVENANCE + OBSERVABILITY
        provenance.record_result(
            technique=result.best_technique,
            score=result.best_score,
            code="",  # frontier returns programs, not raw code in this path
        )
        provenance.finalize()

        emitter.experiment_end(
            decision=result.decision,
            score=result.best_score,
        )

        if config.verbose:
            print(f"[attestra] Done: {result.decision} | best={result.best_score:.4f} | "
                  f"{result.n_proposals} proposals ({result.n_successful} ok, "
                  f"{result.n_failed} failed) | "
                  f"{result.phases_completed}/{result.phases_total} phases | "
                  f"{result.elapsed_s:.1f}s")
            print(f"[attestra]   Data version: {data_version.version_id} | "
                  f"Provenance: {provenance._record.record_id}")
            if strategy_recs:
                print(f"[attestra]   Strategy: {strategy_recs[0].strategy_class.value} | "
                      f"Resource: {resource_rec.substrate}")

    except Exception as e:
        result.decision = "error"
        result.elapsed_s = time.time() - t0
        result.failure_report = {"error": str(e), "type": type(e).__name__}
        if config.verbose:
            import traceback
            print(f"[attestra] Orchestration error: {e}")
            traceback.print_exc()

    return result


def orchestrate_from_text(
    goal: str,
    data: Any,
    target: str = "target",
    *,
    api_key: Optional[str] = None,
    time_budget_s: float = 300.0,
    max_rounds: int = 15,
    verbose: bool = True,
) -> OrchestrateResult:
    """Convenience wrapper: orchestrate from a goal string + data."""
    X, y, feature_names = _extract_xy(data, target)
    # Only enable live LLM calls when the caller explicitly passes an api_key.
    # Falling back to the ambient env var alone does NOT auto-enable the LLM,
    # which keeps wrapper-based test runs hermetic.
    config = OrchestrateConfig(
        goal=goal, X=X, y=y,
        api_key=api_key or os.environ.get("PRIME_INTELLECT_API_KEY"),
        use_llm=api_key is not None,
        time_budget_s=time_budget_s,
        max_rounds=max_rounds,
        feature_names=feature_names,
        verbose=verbose,
    )
    return orchestrate(config)


# ============================================================================== helpers

def _build_llm_call(api_key: str):
    """Build an LLM call function from an API key.

    Uses the unified GPU backend for smart model selection.
    Falls back to direct OpenAI client if gpu_backend is unavailable.
    """
    result = build_llm_client(api_key=api_key)
    if result is not None:
        return result
    # Fallback: direct OpenAI client
    try:
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.pinference.ai/api/v1",
            timeout=120,
        )

        def llm_call(system: str, user: str):
            resp = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            content = resp.choices[0].message.content
            usage = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                }
            return content, usage

        return llm_call
    except ImportError:
        pass
    return None


def _build_neural_training_code(
    task_type: str,
    n_features: int,
    n_outputs: int,
    llm_call: Optional[Callable] = None,
) -> Optional[str]:
    """Generate neural network training code for remote GPU execution."""
    if llm_call is not None:
        try:
            prompt = (
                f"Write a PyTorch neural network training function for a {task_type} task.\n"
                f"Input features: {n_features}, Output classes/values: {n_outputs}.\n"
                f"The function must be named 'solve' with signature:\n"
                f"  def solve(X_train, y_train, X_test) -> predictions\n"
                f"X_train, X_test are numpy arrays. y_train is numpy array.\n"
                f"Use torch, train for 50-100 epochs with early stopping.\n"
                f"Return numpy predictions. Use CUDA if available.\n"
                f"Return ONLY the code, no markdown fences."
            )
            code, _ = llm_call(
                "You are an expert ML engineer. Write production-quality PyTorch code.",
                prompt,
            )
            if code and "def solve" in code:
                return code
        except Exception:
            pass

    # Template fallback
    if task_type == "regression":
        return _neural_regression_template(n_features)
    return _neural_classification_template(n_features, n_outputs)


def _neural_classification_template(n_features: int, n_classes: int) -> str:
    return f'''import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

def solve(X_train, y_train, X_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train.astype(int), dtype=torch.long, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    model = nn.Sequential(
        nn.Linear({n_features}, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, {n_classes}),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience, patience_count = 15, 0
    batch_size = min(256, len(X_tr_t))

    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        epoch_loss = 0
        for i in range(0, len(X_tr_t), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / max(1, len(X_tr_t) // batch_size)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).argmax(dim=1).cpu().numpy()
    return preds
'''


def _neural_regression_template(n_features: int) -> str:
    return f'''import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

def solve(X_train, y_train, X_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
    y_tr_norm = (y_train - y_mean) / y_std

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr_norm, dtype=torch.float32, device=device).unsqueeze(1)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    model = nn.Sequential(
        nn.Linear({n_features}, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience, patience_count = 15, 0
    batch_size = min(256, len(X_tr_t))

    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        epoch_loss = 0
        for i in range(0, len(X_tr_t), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / max(1, len(X_tr_t) // batch_size)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).squeeze(1).cpu().numpy()
    return preds * y_std + y_mean
'''


def _extract_xy(data, target: str):
    """Extract X, y from various data formats."""
    feature_names = None
    if isinstance(data, tuple) and len(data) == 2:
        return np.asarray(data[0]), np.asarray(data[1]), feature_names
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            feature_names = [c for c in data.columns if c != target]
            return data[feature_names].values, data[target].values, feature_names
    except ImportError:
        pass
    if isinstance(data, dict) and target in data:
        y = np.asarray(data[target])
        feature_names = [k for k in data if k != target]
        X = np.column_stack([np.asarray(data[k]) for k in feature_names])
        return X, y, feature_names
    raise ValueError(f"Cannot extract X, y from {type(data)}. Pass (X, y) tuple or DataFrame.")


def _record_to_registry(config: OrchestrateConfig, result: OrchestrateResult) -> None:
    """Record experiment outcome to the registry."""
    try:
        registry = ExperimentRegistry(config.registry_path)
        fp = fingerprint_dataset(config.X, config.y)
        plan = result.plan
        plan_hash = getattr(plan, "plan_hash", "") or (
            getattr(plan, "experiment_id", "") if plan else ""
        )
        plan_metric = ""
        plan_threshold = 0
        if plan is not None:
            verification = getattr(plan, "verification", None)
            if verification is not None:
                plan_metric = getattr(verification, "metric", "")
                plan_threshold = getattr(verification, "threshold", 0)
        record = ExperimentRecord(
            plan_hash=plan_hash,
            goal=config.goal,
            dataset_fingerprint=fp,
            metric=plan_metric or config.metric or "",
            threshold=plan_threshold or config.threshold or 0,
            decision=result.decision,
            best_score=result.best_score,
            best_technique=result.best_technique,
            certificate=result.certificate,
            n_samples=len(config.X),
            n_features=config.X.shape[1] if hasattr(config.X, 'shape') else 0,
            task_type=result.profile.task_type if result.profile else "",
            n_proposals=result.n_proposals,
            n_successful=result.n_successful,
            n_failed=result.n_failed,
            elapsed_s=result.elapsed_s,
            error_summary=result.error_summary,
            failure_report=result.failure_report,
        )
        registry.record(record)
    except Exception:
        pass


def _update_strategy_learner(config: OrchestrateConfig, history: List[Dict],
                             result: OrchestrateResult) -> None:
    """Update strategy learner from this run's history."""
    try:
        learner = StrategyLearner(config.strategy_path)
        fp = fingerprint_dataset(config.X, config.y)
        task_type = result.profile.task_type if result.profile else ""

        for entry in history:
            if entry.get("technique"):
                learner.record(StrategyOutcome(
                    strategy_name=entry["technique"],
                    source=entry.get("source", ""),
                    task_type=task_type,
                    n_samples=len(config.X),
                    n_features=config.X.shape[1] if hasattr(config.X, 'shape') else 0,
                    success=(entry.get("status") == "success"),
                    score=entry.get("score"),
                    elapsed_s=0.0,
                    dataset_fp=fp,
                ))
    except Exception:
        pass


# ============================================================================== Strategy Loop API

def orchestrate_with_strategy_loop(config: OrchestrateConfig, **loop_kwargs):
    """Top-level entry point with full strategy loop (Loop 2).

    This wraps orchestrate() in the strategy loop, which retries with
    escalation until certified or budget exhausted.

    Parameters
    ----------
    config : OrchestrateConfig
        Base research configuration.
    **loop_kwargs
        Passed to StrategyLoopConfig (total_budget_s, max_attempts, patience, etc.)

    Returns
    -------
    StrategyLoopResult
        Complete result with all attempts, diagnoses, escalation history.
    """
    from .strategy_loop import run_strategy_loop, StrategyLoopConfig
    loop_config = StrategyLoopConfig(**loop_kwargs) if loop_kwargs else None
    return run_strategy_loop(config, loop_config=loop_config, orchestrate_fn=orchestrate)


def _build_batch_proposals(
    engine: "ResearchEngine",
    cycle_result: "CycleResult",
    history: List[Dict],
    prior_successes: Optional[Dict] = None,
    avoid: Optional[List[str]] = None,
) -> List[Dict]:
    """Build a batch of code-string proposals for parallel pool evaluation.

    Extracts the top-performing techniques from the engine's history and
    creates variant proposals (different hyperparameters, feature subsets)
    that the parallel pool can evaluate as raw code.
    """
    from .proposals_codegen import generate_variant_proposals
    return generate_variant_proposals(
        engine, cycle_result, history,
        prior_successes=prior_successes or {},
        avoid=avoid or [],
        max_proposals=6,
    )
