"""Unified research cycle engine.

ONE recursive loop that:
  1. PROFILES the data (DataProfile)
  2. DESIGNS the experiment (ExperimentPlan)
  3. Runs the recursive cycle:
     diagnose -> propose (from ALL sources) -> rank -> execute -> measure -> update
  4. CERTIFIES the winner through the frozen certifier (one sealed peek)
  5. Records to the LEDGER (positive AND negative certificates)

Proposal sources are PLUGGABLE: catalog (deterministic), LLM (generative),
retrieval (HF/arXiv-informed), mutation (combine successful solutions).

The frozen certifier is the SOLE promoter. The engine proposes; it never promotes.
"""
from __future__ import annotations

import ast
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..intake.profiler import DataProfile, profile_data
from ..design.experiment_plan import ExperimentPlan, design_experiment
from ..orchestration.error_taxonomy import (
    ClassifiedError, ErrorCategory, ErrorTracker, Severity, classify_error,
)
from ..orchestration.health import HealthMonitor, HealthStatus
from .proposals.catalog import CatalogProposal, propose_from_catalog


# ============================================================================== data structures

@dataclass
class Proposal:
    """A candidate solution from any source."""
    id: str
    source: str                  # "catalog" | "llm" | "retrieval" | "mutation" | "combination"
    name: str
    build_fn: Optional[Callable] = None   # for catalog proposals
    code: Optional[str] = None            # for LLM proposals
    parent_id: Optional[str] = None
    priority: float = 1.0
    diagnosis: str = ""
    technique: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class EvaluatedProposal:
    """A proposal after execution and evaluation."""
    proposal: Proposal
    val_score: Optional[float] = None
    estimator: Any = None
    predictions: Optional[np.ndarray] = None
    status: str = "pending"       # "success" | "failed" | "timeout" | "skipped"
    error: Optional[ClassifiedError] = None
    elapsed_s: float = 0.0
    generation: int = 0


@dataclass
class CycleState:
    """Current state of the research cycle, shared across all proposal sources."""
    round_num: int = 0
    best_score: float = 0.0
    best_technique: str = ""
    best_proposal_id: str = ""
    history: List[Dict] = field(default_factory=list)
    tried_families: set = field(default_factory=set)
    error_tracker: ErrorTracker = field(default_factory=ErrorTracker)
    # For LLM context
    recent_errors: List[str] = field(default_factory=list)
    recent_successes: List[Dict] = field(default_factory=list)


@dataclass
class CycleResult:
    """Result of the unified research cycle."""
    decision: str                        # "certified" | "do_not_certify" | "honest_stop" | "error"
    best_score: float = 0.0
    best_technique: str = ""
    best_estimator: Any = None
    certificate: Optional[Dict] = None
    plan: Optional[ExperimentPlan] = None
    profile: Optional[DataProfile] = None
    n_proposals: int = 0
    n_successful: int = 0
    n_failed: int = 0
    elapsed_s: float = 0.0
    history: List[Dict] = field(default_factory=list)
    failure_report: Optional[Dict] = None
    error_summary: Optional[Dict] = None
    health_summary: Optional[Dict] = None


# ============================================================================== the engine

class ResearchEngine:
    """The unified research cycle engine.

    Usage:
        engine = ResearchEngine(X, y, goal="Classify digits with >0.95 accuracy")
        result = engine.run()
        # result.decision == "certified" | "do_not_certify" | ...
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        goal: str = "",
        metric: Optional[str] = None,
        threshold: Optional[float] = None,
        max_rounds: int = 15,
        time_budget_s: float = 300.0,
        llm_call: Optional[Callable] = None,
        use_retrieval: bool = True,
        gpu: bool = False,
        feature_names: Optional[List[str]] = None,
        seed: int = 42,
        verbose: bool = True,
        on_event: Optional[Callable] = None,
    ):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.goal = goal
        self.llm_call = llm_call
        self.use_retrieval = use_retrieval
        self.gpu = gpu
        self.seed = seed
        self.verbose = verbose
        self.on_event = on_event
        self._proposal_counter = 0

        # Phase 1: PROFILE the data
        self.profile = profile_data(self.X, self.y, feature_names=feature_names)

        # Phase 2: DESIGN the experiment
        self.plan = design_experiment(
            goal, self.profile,
            metric=metric, threshold=threshold,
            time_budget_s=time_budget_s, max_rounds=max_rounds,
            gpu=gpu, use_llm=(llm_call is not None),
        )

        # Split data: train (60%) / val (20%) / sealed test (20%)
        self._split_data()

        # Labels for metrics
        if self.profile.task_type != "regression":
            self.labels = sorted(set(self.y.tolist()))
        else:
            self.labels = None

        # Cycle state
        self.state = CycleState()
        self.evaluated: Dict[str, EvaluatedProposal] = {}

        # Health monitor
        self.health = HealthMonitor(
            time_budget_s=time_budget_s,
            stall_patience=max(3, max_rounds // 4),
        )

        # Retriever (lazy init)
        self._retriever = None

    def run(self) -> CycleResult:
        """Execute the full research cycle."""
        t0 = time.time()
        deadline = t0 + float(self.plan.budget.time_budget_s)
        result = CycleResult(decision="do_not_certify", plan=self.plan, profile=self.profile)

        if self.verbose:
            print(f"[attestra] Goal: {self.goal[:80]}")
            print(f"[attestra] Data: {self.profile.n_samples}x{self.profile.n_features}, "
                  f"task={self.profile.task_type}, metric={self.plan.verification.metric}")
            print(f"[attestra] Plan: {self.plan.budget.max_rounds} rounds, "
                  f"{self.plan.budget.time_budget_s:.0f}s budget")
            if self.profile.issues:
                print(f"[attestra] Issues: {', '.join(self.profile.issues[:3])}")
            if not self.plan.power.has_power:
                print(f"[attestra] WARNING: {self.plan.power.recommendation}")

        # Phase 3: RECURSIVE CYCLE
        for round_num in range(self.plan.budget.max_rounds):
            self.state.round_num = round_num

            # Wall-clock budget guard (between rounds)
            if time.time() >= deadline:
                if self.verbose:
                    print(f"[attestra] Time budget exhausted before round {round_num+1}, stopping")
                break

            # Health check
            health_report = self.health.check()
            if health_report.should_abort:
                if self.verbose:
                    print(f"[attestra] Aborting: {health_report.signals[0].message}")
                break

            # Diagnose current state
            diagnosis = self._diagnose()

            # Propose from ALL sources
            proposals = self._propose_all(diagnosis, health_report)
            if not proposals:
                if self.verbose:
                    print(f"[attestra] No proposals available, stopping")
                break

            # Execute and evaluate each proposal
            for proposal in proposals:
                # Wall-clock budget guard (within a round): a single round can
                # hold many multi-second sandbox proposals, so enforce the
                # deadline here too rather than only between rounds.
                if time.time() >= deadline:
                    if self.verbose:
                        print(f"[attestra] Time budget exhausted mid-round {round_num+1}, stopping")
                    break
                ep = self._execute_and_evaluate(proposal)
                self.evaluated[proposal.id] = ep
                result.n_proposals += 1

                if ep.status == "success":
                    result.n_successful += 1
                    self.health.observe_round(
                        round_num, ep.val_score,
                        elapsed_s=ep.elapsed_s, error=False,
                    )
                else:
                    result.n_failed += 1
                    self.health.observe_round(
                        round_num, None,
                        elapsed_s=ep.elapsed_s, error=True,
                    )

            # Update state
            self._update_state()

            if self.verbose and self.state.best_score > 0:
                print(f"[attestra] Round {round_num+1}: best={self.state.best_score:.4f} "
                      f"({self.state.best_technique})")

            # Early stop if we've clearly exceeded the threshold on validation
            threshold = self.plan.verification.threshold
            if self.state.best_score > threshold + 0.05:
                if self.verbose:
                    print(f"[attestra] Val score {self.state.best_score:.4f} >> threshold {threshold:.4f}, "
                          f"proceeding to certification")
                break

        # Phase 4: CERTIFICATION
        result.best_score = self.state.best_score
        result.best_technique = self.state.best_technique
        result.elapsed_s = time.time() - t0
        result.history = self.state.history
        result.error_summary = self.state.error_tracker.summary()
        result.health_summary = self.health.summary()

        # Find the best estimator
        best_ep = self._best_evaluated()
        if best_ep and best_ep.estimator is not None:
            result.best_estimator = best_ep.estimator
            result.decision = "do_not_certify"  # default until certified

            # Attempt certification through the frozen certifier
            cert = self._certify(best_ep)
            if cert and cert.get("certified"):
                result.decision = "certified"
                result.certificate = cert
            elif cert:
                result.certificate = cert
                result.failure_report = {
                    "dominant_source": "validation winner did not clear sealed test threshold",
                    "sealed_lower_bound": cert.get("lower_bound"),
                    "threshold": self.plan.verification.threshold,
                    "cheapest_unblock": "more data or stronger model",
                }
        else:
            result.decision = "honest_stop"
            result.failure_report = {
                "dominant_source": "no successful proposal found",
                "n_proposals": result.n_proposals,
                "n_failed": result.n_failed,
                "error_summary": result.error_summary,
                "cheapest_unblock": self.state.error_tracker.recovery_suggestions()[:3],
            }

        if self.verbose:
            print(f"[attestra] Done: {result.decision} | best={result.best_score:.4f} | "
                  f"{result.n_proposals} proposals ({result.n_successful} ok, {result.n_failed} failed) | "
                  f"{result.elapsed_s:.1f}s")

        return result

    # ========================================================================== internal

    def _split_data(self) -> None:
        """3-way split: train (60%), val (20%), sealed test (20%).

        The sealed test is held out and NEVER used for model selection.
        Only the certification step peeks at it (checks=1).
        """
        from sklearn.model_selection import train_test_split
        stratify = self.y if self.profile.task_type != "regression" else None
        try:
            X_trainval, self.X_test, y_trainval, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=self.seed, stratify=stratify,
            )
            stratify2 = y_trainval if self.profile.task_type != "regression" else None
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_trainval, y_trainval, test_size=0.25, random_state=self.seed, stratify=stratify2,
            )
        except ValueError:
            X_trainval, self.X_test, y_trainval, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=self.seed,
            )
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_trainval, y_trainval, test_size=0.25, random_state=self.seed,
            )

    def _diagnose(self) -> Dict[str, Any]:
        """Diagnose the current state to guide proposals."""
        diagnosis = {
            "round": self.state.round_num,
            "best_score": self.state.best_score,
            "threshold": self.plan.verification.threshold,
            "gap": self.plan.verification.threshold - self.state.best_score,
            "n_tried": len(self.state.tried_families),
            "error_rate": (self.state.error_tracker.total /
                           max(len(self.evaluated), 1)),
        }

        # Analyze what's working and what's not
        if self.state.recent_successes:
            top_families = [s["family"] for s in sorted(
                self.state.recent_successes, key=lambda x: x.get("score", 0), reverse=True
            )[:3]]
            diagnosis["top_families"] = top_families

        if self.state.error_tracker.total > 0:
            diagnosis["common_errors"] = self.state.error_tracker.count_by_category()
            diagnosis["recovery_suggestions"] = self.state.error_tracker.recovery_suggestions()[:3]

        return diagnosis

    def _propose_all(self, diagnosis: Dict, health_report) -> List[Proposal]:
        """Gather proposals from all sources."""
        proposals = []

        # Source 1: Catalog (always)
        catalog_proposals = propose_from_catalog(
            self.profile, tried_families=self.state.tried_families, max_proposals=2,
        )
        for cp in catalog_proposals:
            proposals.append(Proposal(
                id=self._next_id(),
                source="catalog",
                name=cp.name,
                build_fn=cp.build_fn,
                priority=cp.priority,
                technique=cp.family,
                tags=cp.tags,
            ))

        # Source 2: LLM (if available)
        if self.llm_call is not None and self.state.round_num < self.plan.budget.max_rounds:
            llm_proposals = self._propose_from_llm(diagnosis)
            proposals.extend(llm_proposals)

        # Source 3: Mutation/combination of successful solutions
        if self.state.round_num > 1:
            mutation_proposals = self._propose_mutations()
            proposals.extend(mutation_proposals)

        # Source 4: Retrieval-informed (every other round)
        if (self.use_retrieval and self.llm_call is not None
                and self.state.round_num % 2 == 0):
            retrieval_proposals = self._propose_from_retrieval()
            proposals.extend(retrieval_proposals)

        # If stalling, pivot: try strategies we haven't tried
        if health_report.overall == HealthStatus.STALLED:
            pivot_proposals = self._propose_pivot(diagnosis)
            proposals.extend(pivot_proposals)

        # Sort by priority, limit to budget
        proposals.sort(key=lambda p: p.priority, reverse=True)
        max_per_round = 4
        return proposals[:max_per_round]

    def _propose_from_llm(self, diagnosis: Dict) -> List[Proposal]:
        """Generate proposals using the LLM."""
        proposals = []
        if self.llm_call is None:
            return proposals

        # Build context for the LLM
        context = self.plan.to_llm_context()
        history_str = self._format_history_for_llm()
        error_context = ""
        if self.state.recent_errors:
            error_context = "\nRecent FAILURES (avoid these):\n" + "\n".join(
                f"  - {e}" for e in self.state.recent_errors[-3:]
            )

        system = self._build_llm_system_prompt()
        user = (
            f"{context}\n\n"
            f"Current round: {self.state.round_num + 1}\n"
            f"Best so far: {self.state.best_score:.4f} ({self.state.best_technique})\n"
            f"Gap to threshold: {diagnosis['gap']:.4f}\n"
            f"History:\n{history_str}\n"
            f"{error_context}\n\n"
            f"Propose a novel solution. Respond with JSON only:\n"
            f'{{"diagnosis": "...", "technique": "...", "rationale": "...", '
            f'"code": "complete Python code defining build_estimator(seed)"}}'
        )

        try:
            raw, _ = self.llm_call(system, user)
            parsed = self._parse_llm_response(raw)
            if parsed and "code" in parsed:
                proposals.append(Proposal(
                    id=self._next_id(),
                    source="llm",
                    name=parsed.get("technique", "llm_proposal"),
                    code=parsed["code"],
                    priority=8.0,
                    diagnosis=parsed.get("diagnosis", ""),
                    technique=parsed.get("technique", "llm_generated"),
                ))
        except Exception:
            pass

        return proposals

    def _propose_mutations(self) -> List[Proposal]:
        """Mutate successful solutions to create new candidates."""
        successful = [ep for ep in self.evaluated.values()
                      if ep.status == "success" and ep.val_score is not None]
        if len(successful) < 2:
            return []

        # Sort by score, take top 2
        top = sorted(successful, key=lambda e: e.val_score or 0, reverse=True)[:2]

        # If we have LLM, ask it to combine top solutions
        if self.llm_call is not None:
            return self._propose_llm_combination(top)

        return []

    def _propose_llm_combination(self, top_solutions: List[EvaluatedProposal]) -> List[Proposal]:
        """Ask LLM to combine top solutions."""
        if self.llm_call is None:
            return []

        solutions_desc = []
        for ep in top_solutions:
            if ep.proposal.code:
                solutions_desc.append(
                    f"Solution ({ep.proposal.technique}, score={ep.val_score:.4f}):\n"
                    f"```python\n{ep.proposal.code}\n```"
                )
            else:
                solutions_desc.append(
                    f"Solution ({ep.proposal.technique}, score={ep.val_score:.4f}): "
                    f"[catalog model: {ep.proposal.name}]"
                )

        system = (
            "You are an expert ML researcher. Combine the strengths of the given solutions "
            "into a SINGLE superior solution.\n\n"
            "RULES:\n"
            "- MUST define `build_estimator(seed)` returning a sklearn-compatible estimator\n"
            "- Combine the best aspects: use VotingClassifier, StackingClassifier, or a Pipeline\n"
            "- Do NOT use GridSearchCV or train_test_split\n"
            "- Must be FAST (under 30s on 10k samples)\n"
            "- May import: numpy, scipy, sklearn (any submodule)\n\n"
            'Respond with JSON: {"diagnosis": "...", "technique": "...", "code": "..."}\n'
        )
        user = (
            f"Problem: {self.goal}\n"
            f"Shape: {self.profile.n_samples}x{self.profile.n_features}, "
            f"task={self.profile.task_type}\n\n"
            f"Solutions to combine:\n{''.join(solutions_desc)}\n\n"
            f"Create a combined solution. JSON only."
        )

        try:
            raw, _ = self.llm_call(system, user)
            parsed = self._parse_llm_response(raw)
            if parsed and "code" in parsed:
                return [Proposal(
                    id=self._next_id(),
                    source="combination",
                    name=parsed.get("technique", "combined"),
                    code=parsed["code"],
                    priority=9.0,
                    diagnosis="combining top solutions",
                    technique=parsed.get("technique", "combined_ensemble"),
                )]
        except Exception:
            pass
        return []

    def _propose_from_retrieval(self) -> List[Proposal]:
        """Generate proposals informed by HF/arXiv retrieval."""
        if self.llm_call is None:
            return []

        if self._retriever is None:
            try:
                from ..retrieval.unified import UnifiedRetriever
                self._retriever = UnifiedRetriever(timeout=8.0)
            except Exception:
                return []

        try:
            results = self._retriever.search(
                f"{self.goal} {self.profile.task_type}",
                task=self._hf_task_tag(), limit=3,
            )
            if not results:
                return []

            formatted = self._retriever.format_for_llm(results)
            system = self._build_llm_system_prompt()
            user = (
                f"Problem: {self.goal}\n"
                f"Shape: {self.profile.n_samples}x{self.profile.n_features}, "
                f"task={self.profile.task_type}\n\n"
                f"Retrieved knowledge from HuggingFace/arXiv:\n{formatted}\n\n"
                f"Use this knowledge to propose a novel solution. JSON only:\n"
                f'{{"diagnosis": "...", "technique": "...", "code": "..."}}'
            )

            raw, _ = self.llm_call(system, user)
            parsed = self._parse_llm_response(raw)
            if parsed and "code" in parsed:
                return [Proposal(
                    id=self._next_id(),
                    source="retrieval",
                    name=parsed.get("technique", "retrieval_informed"),
                    code=parsed["code"],
                    priority=7.0,
                    diagnosis=parsed.get("diagnosis", ""),
                    technique=parsed.get("technique", "retrieval_generated"),
                )]
        except Exception:
            pass
        return []

    def _propose_pivot(self, diagnosis: Dict) -> List[Proposal]:
        """When stalling, propose strategies we haven't tried."""
        proposals = []
        untried = propose_from_catalog(
            self.profile, tried_families=self.state.tried_families, max_proposals=2,
        )
        for cp in untried:
            proposals.append(Proposal(
                id=self._next_id(),
                source="catalog",
                name=f"pivot_{cp.name}",
                build_fn=cp.build_fn,
                priority=cp.priority + 2.0,  # boost priority for pivots
                technique=f"pivot_{cp.family}",
                tags=cp.tags + ["pivot"],
            ))
        return proposals

    def _execute_and_evaluate(self, proposal: Proposal) -> EvaluatedProposal:
        """Execute a proposal and evaluate with the frozen metric."""
        ep = EvaluatedProposal(proposal=proposal, generation=self.state.round_num)
        t0 = time.time()

        try:
            # Build the estimator
            if proposal.build_fn is not None:
                estimator = proposal.build_fn(self.seed)
            elif proposal.code is not None:
                estimator = self._execute_code(proposal.code)
            else:
                ep.status = "failed"
                ep.error = ClassifiedError(
                    category=ErrorCategory.CODE,
                    severity=Severity.RECOVERABLE,
                    message="No build function or code provided",
                )
                return ep

            if estimator is None:
                ep.status = "failed"
                ep.error = ClassifiedError(
                    category=ErrorCategory.CODE,
                    severity=Severity.RECOVERABLE,
                    message="Estimator build returned None",
                )
                return ep

            # Fit
            estimator.fit(self.X_train, self.y_train)

            # Predict
            predictions = estimator.predict(self.X_val)

            # Score with frozen metric
            from ..core.science import score_metric
            score = float(score_metric(
                self.plan.verification.metric,
                self.y_val.tolist(), predictions.tolist(),
                self.labels,
            ))

            ep.val_score = score
            ep.estimator = estimator
            ep.predictions = predictions
            ep.status = "success"
            ep.elapsed_s = time.time() - t0

        except Exception as e:
            ep.status = "failed"
            ep.error = classify_error(e, context={"proposal": proposal.name, "source": proposal.source})
            ep.elapsed_s = time.time() - t0
            self.state.error_tracker.record(ep.error)

        return ep

    def _execute_code(self, code: str) -> Any:
        """Execute LLM-generated code and return the estimator."""
        # Unescape JSON-encoded newlines
        if "\\n" in code:
            real_nl = code.count("\n")
            escaped_nl = code.count("\\n")
            if escaped_nl > real_nl:
                code = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')

        # Strip GridSearchCV if present
        if "GridSearchCV" in code or "RandomizedSearchCV" in code:
            import re
            lines = [l for l in code.split("\n")
                     if "GridSearchCV" not in l and "RandomizedSearchCV" not in l]
            code = "\n".join(lines)

        # AST validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise RuntimeError(f"Syntax error at line {e.lineno}: {e.msg}") from e

        # Import preamble
        preamble = (
            "import numpy as np\nimport scipy\nimport sklearn\n"
            "from sklearn.pipeline import Pipeline, make_pipeline\n"
            "from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, PowerTransformer\n"
            "from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,\n"
            "    ExtraTreesClassifier, BaggingClassifier, VotingClassifier, StackingClassifier,\n"
            "    HistGradientBoostingClassifier, AdaBoostClassifier,\n"
            "    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,\n"
            "    HistGradientBoostingRegressor, BaggingRegressor, VotingRegressor, StackingRegressor)\n"
            "from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, SGDClassifier\n"
            "from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR\n"
            "from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor\n"
            "from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor\n"
            "from sklearn.neural_network import MLPClassifier, MLPRegressor\n"
            "from sklearn.decomposition import PCA, TruncatedSVD\n"
            "from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif\n"
            "from sklearn.kernel_approximation import RBFSampler, Nystroem\n"
            "from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin\n"
            "from sklearn.metrics import accuracy_score, r2_score\n"
        )

        # Fix hallucinated names
        fixes = {
            "StackClassifier": "StackingClassifier",
            "StackRegressor": "StackingRegressor",
            "GBMClassifier": "GradientBoostingClassifier",
            "GBMRegressor": "GradientBoostingRegressor",
            "XGBClassifier": "HistGradientBoostingClassifier",
            "XGBRegressor": "HistGradientBoostingRegressor",
            "LGBMClassifier": "HistGradientBoostingClassifier",
            "LGBMRegressor": "HistGradientBoostingRegressor",
            "base_estimator=": "estimator=",
        }
        for wrong, right in fixes.items():
            code = code.replace(wrong, right)

        full_code = preamble + "\n" + code
        namespace: Dict[str, Any] = {"__builtins__": __builtins__, "np": np, "numpy": np}

        try:
            exec(full_code, namespace)  # noqa: S102
        except ImportError:
            import re
            cleaned = re.sub(r'^(from\s+\S+\s+import\s+.+|import\s+.+)$', '', code, flags=re.MULTILINE)
            full_code = preamble + "\n" + cleaned
            exec(full_code, namespace)  # noqa: S102

        build_fn = namespace.get("build_estimator") or namespace.get("build")
        if build_fn is None:
            raise RuntimeError("Code does not define build_estimator(seed)")

        return build_fn(self.seed)

    def _certify(self, best_ep: EvaluatedProposal) -> Optional[Dict]:
        """Certify the validation winner on the SEALED test (ONE peek).

        The sealed test (self.X_test, self.y_test) is separate from the validation set
        used for model selection. This is the ONLY place the sealed test is peeked at.
        """
        if best_ep.estimator is None:
            return None

        try:
            from ..core.science import (
                certify_accuracy, certify_regression, score_metric,
            )

            # Re-train on full train+val for the sealed test
            X_full = np.vstack([self.X_train, self.X_val])
            y_full = np.concatenate([self.y_train, self.y_val])

            if best_ep.proposal.build_fn is not None:
                final_estimator = best_ep.proposal.build_fn(self.seed)
            elif best_ep.proposal.code is not None:
                final_estimator = self._execute_code(best_ep.proposal.code)
            else:
                final_estimator = best_ep.estimator

            final_estimator.fit(X_full, y_full)

            # The SEALED test: predict on held-out test set (never seen during selection)
            predictions = final_estimator.predict(self.X_test)
            metric = self.plan.verification.metric
            threshold = self.plan.verification.threshold

            score = float(score_metric(metric, self.y_test.tolist(), predictions.tolist(), self.labels))

            if metric in ("r2", "neg_rmse", "neg_mae"):
                cert = certify_regression(
                    self.y_test.tolist(), predictions.tolist(),
                    metric, threshold, checks=1, alpha=0.05,
                )
            elif metric == "accuracy":
                cert = certify_accuracy(
                    score, len(self.y_test), threshold, checks=1, alpha=0.05,
                )
            else:
                # balanced_accuracy, macro_f1: use bootstrap
                from ..core.science import _bootstrap_classification_metric_lower
                lb, point = _bootstrap_classification_metric_lower(
                    metric, self.y_test.tolist(), predictions.tolist(), self.labels,
                )
                cert = {
                    "observed": round(point, 4),
                    "lower_bound": lb,
                    "threshold": threshold,
                    "certified": bool(lb > threshold),
                    "metric": metric,
                    "n": len(self.y_test),
                    "reason": ("lower bound clears threshold" if lb > threshold
                               else "lower bound does not clear threshold"),
                }

            cert["technique"] = best_ep.proposal.technique
            cert["plan_hash"] = self.plan.plan_hash
            return cert

        except Exception as e:
            if self.verbose:
                print(f"[attestra] Certification failed: {e}")
            return None

    def _update_state(self) -> None:
        """Update cycle state from evaluated proposals."""
        for ep in self.evaluated.values():
            if ep.status == "success" and ep.val_score is not None:
                if ep.val_score > self.state.best_score:
                    self.state.best_score = ep.val_score
                    self.state.best_technique = ep.proposal.technique
                    self.state.best_proposal_id = ep.proposal.id
                self.state.recent_successes.append({
                    "technique": ep.proposal.technique,
                    "family": ep.proposal.source,
                    "score": ep.val_score,
                })
            if ep.status == "failed" and ep.error:
                self.state.recent_errors.append(ep.error.message[:100])
            # Track tried families
            self.state.tried_families.add(ep.proposal.name)

        self.state.history = [
            {
                "id": ep.proposal.id,
                "source": ep.proposal.source,
                "technique": ep.proposal.technique,
                "score": ep.val_score,
                "status": ep.status,
                "error": ep.error.message[:100] if ep.error else None,
            }
            for ep in self.evaluated.values()
        ]

    def _best_evaluated(self) -> Optional[EvaluatedProposal]:
        """Return the best successful proposal."""
        best = None
        for ep in self.evaluated.values():
            if ep.status == "success" and ep.val_score is not None:
                if best is None or ep.val_score > (best.val_score or 0):
                    best = ep
        return best

    def _next_id(self) -> str:
        self._proposal_counter += 1
        return f"p-{self._proposal_counter:04d}"

    def _hf_task_tag(self) -> str:
        mapping = {"binary": "tabular-classification", "multiclass": "tabular-classification",
                   "regression": "tabular-regression"}
        return mapping.get(self.profile.task_type, "")

    def _format_history_for_llm(self) -> str:
        lines = []
        for ep in sorted(self.evaluated.values(),
                         key=lambda e: e.val_score or -1, reverse=True)[:8]:
            if ep.status == "success":
                lines.append(f"  OK: {ep.proposal.technique} = {ep.val_score:.4f}")
            else:
                lines.append(f"  FAIL: {ep.proposal.technique}: {ep.error.message[:60] if ep.error else '?'}")
        return "\n".join(lines) if lines else "(no history yet)"

    def _build_llm_system_prompt(self) -> str:
        metric = self.plan.verification.metric
        return (
            f"You are an expert ML researcher. Given the data profile and experiment plan,\n"
            f"propose a novel {self.profile.task_type} solution targeting {metric}.\n\n"
            f"Your response MUST be valid JSON with:\n"
            f'{{"diagnosis": "one sentence on the bottleneck", '
            f'"technique": "name of the approach", '
            f'"rationale": "why this addresses the bottleneck", '
            f'"code": "complete Python code defining build_estimator(seed)"}}\n\n'
            f"RULES for the code:\n"
            f"- MUST define `build_estimator(seed)` returning an sklearn-compatible estimator\n"
            f"- FAST: fit+predict under 30s on 10k samples\n"
            f"- Do NOT use GridSearchCV, RandomizedSearchCV, cross_val_score, or train_test_split\n"
            f"- May import: numpy, scipy, sklearn (any submodule)\n"
            f"- Use the seed parameter for reproducibility\n"
            f"- Target: {self.profile.n_features} features, {self.profile.n_classes} classes, "
            f"{self.profile.task_type}\n"
            f"- Be NOVEL: not just a default RandomForest. Think like a Kaggle grandmaster.\n"
        )

    def _parse_llm_response(self, raw: str) -> Optional[Dict]:
        """Parse LLM response into a dict with 'code' field."""
        import json
        import re
        text = raw.strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Try direct JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "code" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        # Extract code field manually (handles literal newlines in JSON)
        code_marker = re.search(r'"code"\s*:\s*"', text)
        if code_marker:
            code_start = code_marker.end()
            depth = 0
            i = code_start
            while i < len(text):
                c = text[i]
                if c == '\\' and i + 1 < len(text):
                    i += 2
                    continue
                if c == '"' and depth == 0:
                    code_text = text[code_start:i]
                    code_text = (code_text.replace('\\n', '\n')
                                 .replace('\\t', '\t')
                                 .replace('\\"', '"'))
                    # Extract other fields
                    diag = re.search(r'"diagnosis"\s*:\s*"([^"]*)"', text)
                    tech = re.search(r'"technique"\s*:\s*"([^"]*)"', text)
                    return {
                        "code": code_text,
                        "diagnosis": diag.group(1) if diag else "",
                        "technique": tech.group(1) if tech else "llm_generated",
                    }
                i += 1

        # Last resort: extract Python code blocks
        code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            code = code_blocks[0]
            if "def build_estimator" in code or "def build(" in code:
                return {"code": code, "diagnosis": "", "technique": "llm_generated"}

        return None
