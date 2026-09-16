"""Generative research engine — LLM writes complete solutions, not menu selections.

Architecture:
  - Each "solution" is a complete Python function: (X_train, y_train, X_test) -> predictions
  - The LLM writes everything: preprocessing, feature engineering, model, training
  - A diagnose→improve loop feeds error analysis back to the LLM
  - Solutions form a tree: parent → children (improvements/variants)
  - Literature search (arXiv/PapersWithCode) grounds proposals in recent work
  - The frozen certifier is the SOLE promoter — it never changes

The catalog is gone. The LLM is the sole proposal source. Catalog-quality baselines
are generated as seed solutions to bootstrap the tree.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from ..intake.profiler import DataProfile, profile_data
from ..design.experiment_plan import ExperimentPlan, design_experiment
from ..execution.sandbox import run_sandboxed, static_check
from ..orchestration.error_taxonomy import (
    ClassifiedError, ErrorCategory, ErrorTracker, Severity, classify_error,
)
from ..orchestration.health import HealthMonitor, HealthStatus


# ─────────────────────────────────────────────── data structures

@dataclass
class Solution:
    """A complete solution node in the tree."""
    id: str
    code: str                        # complete Python code
    source: str                      # "seed" | "llm" | "improve" | "branch" | "literature" | "merge"
    name: str
    parent_id: Optional[str] = None
    generation: int = 0
    # Evaluation
    val_score: Optional[float] = None
    test_score: Optional[float] = None
    status: str = "pending"          # "success" | "failed" | "timeout"
    error_msg: str = ""
    # Metadata
    diagnosis: str = ""
    rationale: str = ""
    literature_context: str = ""
    elapsed_s: float = 0.0

    @property
    def is_success(self) -> bool:
        return self.status == "success" and self.val_score is not None


@dataclass
class SolutionTree:
    """A tree of solutions with branching and pruning."""
    nodes: Dict[str, Solution] = field(default_factory=dict)
    root_ids: List[str] = field(default_factory=list)

    def add(self, solution: Solution) -> None:
        self.nodes[solution.id] = solution
        if solution.parent_id is None:
            self.root_ids.append(solution.id)

    def children(self, node_id: str) -> List[Solution]:
        return [s for s in self.nodes.values() if s.parent_id == node_id]

    def best(self, top_k: int = 1) -> List[Solution]:
        successful = [s for s in self.nodes.values() if s.is_success]
        successful.sort(key=lambda s: s.val_score or 0, reverse=True)
        return successful[:top_k]

    def best_score(self) -> float:
        b = self.best(1)
        return b[0].val_score if b else 0.0

    def all_successful(self) -> List[Solution]:
        return [s for s in self.nodes.values() if s.is_success]

    def summary(self) -> Dict:
        total = len(self.nodes)
        ok = sum(1 for s in self.nodes.values() if s.is_success)
        failed = sum(1 for s in self.nodes.values() if s.status == "failed")
        return {
            "total": total, "successful": ok, "failed": failed,
            "best_score": self.best_score(),
            "best_name": self.best(1)[0].name if self.best(1) else "",
            "generations": max((s.generation for s in self.nodes.values()), default=0),
        }


@dataclass
class GenerativeResult:
    """Result of the generative research cycle."""
    decision: str                        # "certified" | "do_not_certify" | "honest_stop" | "error"
    best_score: float = 0.0
    best_technique: str = ""
    best_estimator: Any = None
    best_solution: Optional[Solution] = None
    certificate: Optional[Dict] = None
    plan: Optional[ExperimentPlan] = None
    profile: Optional[DataProfile] = None
    tree_summary: Optional[Dict] = None
    n_solutions: int = 0
    n_successful: int = 0
    n_failed: int = 0
    elapsed_s: float = 0.0
    history: List[Dict] = field(default_factory=list)
    failure_report: Optional[Dict] = None
    literature_queries: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────── the engine

# Import preamble injected into every solution
_PREAMBLE = textwrap.dedent("""\
    import numpy as np
    import scipy
    import scipy.stats
    import sklearn
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.preprocessing import (
        StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
        PolynomialFeatures, PowerTransformer, QuantileTransformer,
        FunctionTransformer, SplineTransformer, OneHotEncoder, OrdinalEncoder,
        KBinsDiscretizer, Binarizer, LabelEncoder,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import (
        SelectKBest, f_classif, f_regression, mutual_info_classif,
        mutual_info_regression, VarianceThreshold,
    )
    from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
    from sklearn.kernel_approximation import RBFSampler, Nystroem
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier, BaggingClassifier, VotingClassifier, StackingClassifier,
        HistGradientBoostingClassifier, AdaBoostClassifier, IsolationForest,
        RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
        HistGradientBoostingRegressor, BaggingRegressor, VotingRegressor, StackingRegressor,
        AdaBoostRegressor,
    )
    from sklearn.linear_model import (
        LogisticRegression, Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor,
        BayesianRidge, HuberRegressor, PassiveAggressiveClassifier,
    )
    from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
    from sklearn.model_selection import cross_val_score
""")

# Hallucinated name fixes
_NAME_FIXES = {
    "StackClassifier": "StackingClassifier",
    "StackRegressor": "StackingRegressor",
    "GBMClassifier": "GradientBoostingClassifier",
    "GBMRegressor": "GradientBoostingRegressor",
    "XGBClassifier": "HistGradientBoostingClassifier",
    "XGBRegressor": "HistGradientBoostingRegressor",
    "LGBMClassifier": "HistGradientBoostingClassifier",
    "LGBMRegressor": "HistGradientBoostingRegressor",
    "CatBoostClassifier": "HistGradientBoostingClassifier",
    "CatBoostRegressor": "HistGradientBoostingRegressor",
    "base_estimator=": "estimator=",
}


class GenerativeEngine:
    """Generative research engine — LLM writes complete solutions.

    Usage:
        engine = GenerativeEngine(X, y, goal="...", llm_call=my_llm)
        result = engine.run()
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
        use_literature: bool = True,
        feature_names: Optional[List[str]] = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.goal = goal
        self.llm_call = llm_call
        self.use_literature = use_literature
        self.seed = seed
        self.verbose = verbose
        self.max_rounds = max_rounds
        self._sol_counter = 0

        # Profile the data
        self.profile = profile_data(self.X, self.y, feature_names=feature_names)

        # Design the experiment
        self.plan = design_experiment(
            goal, self.profile,
            metric=metric, threshold=threshold,
            time_budget_s=time_budget_s, max_rounds=max_rounds,
            gpu=False, use_llm=(llm_call is not None),
        )

        # Split: 60% train, 20% val, 20% sealed test
        self._split_data()

        # Labels for classification metrics
        self.labels = sorted(set(self.y.tolist())) if self.profile.task_type != "regression" else None

        # Solution tree
        self.tree = SolutionTree()

        # Health monitor
        self.health = HealthMonitor(
            time_budget_s=time_budget_s,
            stall_patience=max(3, max_rounds // 3),
        )

        # Error tracking
        self.error_tracker = ErrorTracker()

        # Literature search (lazy init)
        self._literature = None

        # Residual analysis cache
        self._residual_analysis = ""

    def run(self) -> GenerativeResult:
        """Execute the generative research cycle."""
        t0 = time.time()
        result = GenerativeResult(
            decision="do_not_certify", plan=self.plan, profile=self.profile,
        )

        if self.verbose:
            self._log(f"Goal: {self.goal[:100]}")
            self._log(f"Data: {self.profile.n_samples}x{self.profile.n_features}, "
                       f"task={self.profile.task_type}, metric={self.plan.verification.metric}")
            self._log(f"Budget: {self.max_rounds} rounds, {self.plan.budget.time_budget_s:.0f}s")
            if self.profile.issues:
                self._log(f"Issues: {', '.join(self.profile.issues[:3])}")

        # Phase 1: SEED — generate initial baselines
        self._generate_seeds()

        # Phase 2: RECURSIVE IMPROVEMENT — diagnose → improve → branch → merge
        for round_num in range(self.max_rounds):
            # Health check
            health = self.health.check()
            if health.should_abort:
                if self.verbose:
                    self._log(f"Aborting: {health.signals[0].message}")
                break

            # Get current best solutions
            top_solutions = self.tree.best(top_k=3)
            if not top_solutions:
                if self.verbose:
                    self._log("No successful solutions, generating more seeds")
                self._generate_seeds(n=3)
                continue

            best = top_solutions[0]
            gap = self.plan.verification.threshold - best.val_score
            if self.verbose:
                self._log(f"Round {round_num+1}: best={best.val_score:.4f} ({best.name}), "
                           f"gap={gap:.4f}")

            # Early stop if clearly above threshold
            if best.val_score > self.plan.verification.threshold + 0.05:
                if self.verbose:
                    self._log(f"Val {best.val_score:.4f} >> threshold "
                               f"{self.plan.verification.threshold:.4f}, proceeding to certification")
                break

            # Diagnose bottleneck on the best solution
            diagnosis = self._diagnose(best)

            # Strategy selection based on round and progress
            if round_num == 0 or (round_num % 4 == 0 and self.use_literature):
                # Literature-grounded proposal every 4 rounds
                self._propose_from_literature(best, diagnosis)

            # Improve the best solution
            self._improve_solution(best, diagnosis)

            # Branch: try a different approach from 2nd best
            if len(top_solutions) >= 2 and round_num % 3 == 0:
                self._improve_solution(top_solutions[1], diagnosis)

            # Merge: combine top-2 solutions
            if len(top_solutions) >= 2 and round_num % 5 == 2:
                self._merge_solutions(top_solutions[0], top_solutions[1])

            # Update health
            score = self.tree.best_score()
            self.health.observe_round(round_num, score, elapsed_s=time.time() - t0, error=False)

        # Phase 3: CERTIFICATION
        best_solutions = self.tree.best(top_k=1)
        result.tree_summary = self.tree.summary()
        result.n_solutions = len(self.tree.nodes)
        result.n_successful = sum(1 for s in self.tree.nodes.values() if s.is_success)
        result.n_failed = sum(1 for s in self.tree.nodes.values() if s.status == "failed")
        result.elapsed_s = time.time() - t0
        result.history = [
            {"id": s.id, "name": s.name, "score": s.val_score, "source": s.source,
             "generation": s.generation, "parent": s.parent_id,
             "status": s.status, "error": s.error_msg}
            for s in self.tree.nodes.values()
        ]

        if best_solutions:
            best = best_solutions[0]
            result.best_score = best.val_score
            result.best_technique = best.name

            # Certify on sealed test
            cert, estimator = self._certify(best)
            if cert and cert.get("certified"):
                result.decision = "certified"
                result.certificate = cert
                result.best_estimator = estimator
                result.best_solution = best
            elif cert:
                result.decision = "do_not_certify"
                result.certificate = cert
                result.best_estimator = estimator
                result.best_solution = best
                result.failure_report = {
                    "reason": "validation winner did not clear sealed test",
                    "sealed_lower_bound": cert.get("lower_bound"),
                    "threshold": self.plan.verification.threshold,
                }
            else:
                result.decision = "do_not_certify"
                result.best_solution = best
        else:
            result.decision = "honest_stop"
            result.failure_report = {
                "reason": "no successful solution found",
                "n_solutions": result.n_solutions,
                "n_failed": result.n_failed,
            }

        if self.verbose:
            self._log(f"Done: {result.decision} | best={result.best_score:.4f} | "
                       f"{result.n_solutions} solutions ({result.n_successful} ok, "
                       f"{result.n_failed} failed) | {result.elapsed_s:.1f}s")

        return result

    # ─────────────────────────────────────────── seed generation

    def _generate_seeds(self, n: int = 2) -> None:
        """Generate seed solutions to bootstrap the tree."""
        task = self.profile.task_type
        metric = self.plan.verification.metric
        n_features = self.profile.n_features
        n_samples = self.profile.n_samples

        # Seed 1: Strong deterministic baseline
        if task == "regression":
            seed_code = self._make_seed_regression()
        else:
            seed_code = self._make_seed_classification()

        sol = self._evaluate_solution(seed_code, "seed", "baseline_ensemble", generation=0)
        self.tree.add(sol)

        # Seed 2: LLM-generated if available, otherwise second deterministic
        if self.llm_call is not None and n >= 2:
            llm_seed = self._generate_llm_seed()
            if llm_seed:
                sol2 = self._evaluate_solution(
                    llm_seed, "seed", "llm_initial", generation=0,
                )
                self.tree.add(sol2)
        elif n >= 2:
            if task == "regression":
                seed2_code = self._make_seed_regression_alt()
            else:
                seed2_code = self._make_seed_classification_alt()
            sol2 = self._evaluate_solution(seed2_code, "seed", "alt_baseline", generation=0)
            self.tree.add(sol2)

    def _make_seed_regression(self) -> str:
        return textwrap.dedent("""\
            def solve(X_train, y_train, X_test):
                from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
                from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
                from sklearn.linear_model import Ridge
                from sklearn.compose import TransformedTargetRegressor
                from sklearn.preprocessing import StandardScaler, PolynomialFeatures
                from sklearn.pipeline import make_pipeline
                import numpy as np

                # Feature engineering: add pairwise interactions and ratios
                def add_features(X):
                    feats = [X]
                    n_cols = X.shape[1]
                    # Pairwise products of first 4 most-important features
                    top_k = min(4, n_cols)
                    pairs = []
                    for i in range(top_k):
                        for j in range(i+1, top_k):
                            pairs.append((X[:, i] * X[:, j]).reshape(-1, 1))
                    if pairs:
                        feats.append(np.hstack(pairs))
                    # Ratio features
                    for i in range(min(3, n_cols)):
                        for j in range(i+1, min(4, n_cols)):
                            feats.append((X[:, i] / (X[:, j] + 1e-8)).reshape(-1, 1))
                    # Log of positive features
                    for i in range(min(n_cols, 8)):
                        col = X[:, i]
                        if np.all(col >= 0):
                            feats.append(np.log1p(col).reshape(-1, 1))
                    return np.hstack(feats)

                X_train_fe = add_features(X_train)
                X_test_fe = add_features(X_test)

                # Stacking with target transform (log) for skewed targets
                model = TransformedTargetRegressor(
                    regressor=StackingRegressor(
                        estimators=[
                            ('hgb', HistGradientBoostingRegressor(
                                max_iter=500, learning_rate=0.05, max_depth=6,
                                min_samples_leaf=10, random_state=42)),
                            ('et', ExtraTreesRegressor(
                                n_estimators=200, max_depth=15, min_samples_leaf=3, random_state=42)),
                            ('rf', RandomForestRegressor(
                                n_estimators=200, max_depth=15, min_samples_leaf=3, random_state=42)),
                        ],
                        final_estimator=Ridge(alpha=1.0),
                        cv=3,
                    ),
                    func=np.log1p, inverse_func=np.expm1,
                )
                model.fit(X_train_fe, y_train)
                return model.predict(X_test_fe)
        """)

    def _make_seed_regression_alt(self) -> str:
        return textwrap.dedent("""\
            def solve(X_train, y_train, X_test):
                from sklearn.ensemble import HistGradientBoostingRegressor
                from sklearn.compose import TransformedTargetRegressor
                import numpy as np

                model = TransformedTargetRegressor(
                    regressor=HistGradientBoostingRegressor(
                        max_iter=800, learning_rate=0.03, max_depth=8,
                        min_samples_leaf=5, l2_regularization=0.1, random_state=42),
                    func=np.log1p, inverse_func=np.expm1,
                )
                model.fit(X_train, y_train)
                return model.predict(X_test)
        """)

    def _make_seed_classification(self) -> str:
        return textwrap.dedent("""\
            def solve(X_train, y_train, X_test):
                from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
                from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import make_pipeline
                import numpy as np

                stack = StackingClassifier(
                    estimators=[
                        ('hgb', HistGradientBoostingClassifier(
                            max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)),
                        ('et', ExtraTreesClassifier(
                            n_estimators=200, max_depth=15, random_state=42)),
                        ('rf', RandomForestClassifier(
                            n_estimators=200, max_depth=15, random_state=42)),
                    ],
                    final_estimator=LogisticRegression(max_iter=1000),
                    cv=3,
                )
                stack.fit(X_train, y_train)
                return stack.predict(X_test)
        """)

    def _make_seed_classification_alt(self) -> str:
        return textwrap.dedent("""\
            def solve(X_train, y_train, X_test):
                from sklearn.ensemble import HistGradientBoostingClassifier
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import make_pipeline
                import numpy as np

                model = make_pipeline(
                    StandardScaler(),
                    HistGradientBoostingClassifier(
                        max_iter=500, learning_rate=0.03, max_depth=8,
                        min_samples_leaf=5, random_state=42),
                )
                model.fit(X_train, y_train)
                return model.predict(X_test)
        """)

    def _generate_llm_seed(self) -> Optional[str]:
        """Ask LLM to write an initial solution from scratch."""
        if self.llm_call is None:
            return None

        data_desc = self._data_description()
        system = self._system_prompt()
        user = (
            f"Write a COMPLETE solution for this ML problem.\n\n"
            f"{data_desc}\n\n"
            f"This is the FIRST attempt — write the best solution you can think of.\n"
            f"Consider: feature engineering, preprocessing, model selection, ensembling.\n\n"
            f"Respond with JSON:\n"
            f'{{"name": "descriptive_name", "rationale": "why this approach", '
            f'"code": "def solve(X_train, y_train, X_test):\\n    ...\\n    return predictions"}}'
        )

        try:
            raw, _ = self.llm_call(system, user)
            parsed = self._parse_json(raw)
            if parsed and "code" in parsed:
                return parsed["code"]
        except Exception:
            pass
        return None

    # ─────────────────────────────────────────── diagnosis

    def _diagnose(self, solution: Solution) -> str:
        """Analyze why the current best solution isn't better."""
        if self.llm_call is None:
            return f"Score {solution.val_score:.4f}, gap {self.plan.verification.threshold - solution.val_score:.4f}"

        # Compute residual analysis
        residual_info = self._analyze_residuals(solution)
        error_patterns = self._error_patterns()
        data_desc = self._data_description()

        system = (
            "You are an expert ML researcher diagnosing why a solution isn't performing better.\n"
            "Analyze the residuals, error patterns, and data characteristics to identify the\n"
            "SPECIFIC bottleneck. Be concrete — don't say 'try more features', say WHICH features\n"
            "and WHY they'd help.\n\n"
            "Respond with a diagnosis (2-3 sentences, max 200 words)."
        )
        user = (
            f"Problem: {self.goal}\n"
            f"{data_desc}\n\n"
            f"Current solution: {solution.name} (score={solution.val_score:.4f})\n"
            f"Code:\n```python\n{solution.code}\n```\n\n"
            f"Residual analysis:\n{residual_info}\n\n"
            f"Error patterns from failed solutions:\n{error_patterns}\n\n"
            f"What is the SPECIFIC bottleneck? What concrete change would improve the score?"
        )

        try:
            raw, _ = self.llm_call(system, user)
            self._residual_analysis = residual_info
            return raw.strip()
        except Exception:
            return f"Score {solution.val_score:.4f}, gap {self.plan.verification.threshold - solution.val_score:.4f}"

    def _analyze_residuals(self, solution: Solution) -> str:
        """Compute residual analysis for the best solution."""
        try:
            estimator = self._build_and_fit(solution.code)
            if estimator is None:
                return "Could not rebuild estimator for residual analysis"

            preds = estimator.predict(self.X_val)

            if self.profile.task_type == "regression":
                residuals = self.y_val - preds
                return (
                    f"Residual stats: mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f}, "
                    f"max_abs={np.max(np.abs(residuals)):.4f}\n"
                    f"Residual percentiles: p10={np.percentile(residuals, 10):.4f}, "
                    f"p50={np.percentile(residuals, 50):.4f}, p90={np.percentile(residuals, 90):.4f}\n"
                    f"Predictions range: [{np.min(preds):.4f}, {np.max(preds):.4f}]\n"
                    f"Target range: [{np.min(self.y_val):.4f}, {np.max(self.y_val):.4f}]\n"
                    f"High-error samples: {np.sum(np.abs(residuals) > 2 * np.std(residuals))} "
                    f"out of {len(residuals)}"
                )
            else:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(self.y_val, preds)
                n_classes = len(np.unique(self.y_val))
                per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
                worst_class = int(np.argmin(per_class_acc))
                return (
                    f"Per-class accuracy: {dict(enumerate(np.round(per_class_acc, 3).tolist()))}\n"
                    f"Worst class: {worst_class} (acc={per_class_acc[worst_class]:.3f})\n"
                    f"Total classes: {n_classes}, samples: {len(self.y_val)}"
                )
        except Exception as e:
            return f"Residual analysis failed: {e}"

    def _error_patterns(self) -> str:
        """Summarize error patterns from failed solutions."""
        failed = [s for s in self.tree.nodes.values() if s.status == "failed"]
        if not failed:
            return "No failed solutions yet."
        patterns = {}
        for s in failed[-5:]:
            key = s.error_msg.split(":")[0] if s.error_msg else "unknown"
            patterns[key] = patterns.get(key, 0) + 1
        lines = [f"  {k}: {v}x" for k, v in sorted(patterns.items(), key=lambda x: -x[1])]
        return "\n".join(lines)

    # ─────────────────────────────────────────── improvement

    def _improve_solution(self, parent: Solution, diagnosis: str) -> None:
        """Ask LLM to improve a solution based on diagnosis."""
        if self.llm_call is None:
            return

        data_desc = self._data_description()
        tree_context = self._tree_context()

        system = self._system_prompt()
        user = (
            f"IMPROVE this solution based on the diagnosis.\n\n"
            f"{data_desc}\n\n"
            f"Current solution ({parent.name}, score={parent.val_score:.4f}):\n"
            f"```python\n{parent.code}\n```\n\n"
            f"Diagnosis of bottleneck:\n{diagnosis}\n\n"
            f"{tree_context}\n\n"
            f"IMPORTANT IMPROVEMENT RULES:\n"
            f"1. KEEP what works — the current solution scores {parent.val_score:.4f}. Don't break it.\n"
            f"2. Make ONE or TWO specific improvements based on the diagnosis.\n"
            f"3. Your improved solution MUST score HIGHER than {parent.val_score:.4f}.\n"
            f"4. DO NOT drop features from the input — add engineered features instead.\n"
            f"5. If the current solution doesn't do feature engineering, ADD some.\n"
            f"6. If the current solution doesn't do target transform, ADD one (log, sqrt).\n"
            f"7. Start from the current solution's structure and make incremental improvements.\n\n"
            f"Respond with JSON:\n"
            f'{{"name": "descriptive_name", "rationale": "what you changed and why", '
            f'"code": "def solve(X_train, y_train, X_test):\\n    ...\\n    return predictions"}}'
        )

        try:
            raw, _ = self.llm_call(system, user)
            parsed = self._parse_json(raw)
            if parsed and "code" in parsed:
                sol = self._evaluate_solution(
                    parsed["code"], "improve", parsed.get("name", "improved"),
                    parent_id=parent.id,
                    generation=parent.generation + 1,
                    rationale=parsed.get("rationale", ""),
                )
                self.tree.add(sol)
        except Exception:
            pass

    def _merge_solutions(self, sol_a: Solution, sol_b: Solution) -> None:
        """Ask LLM to merge two solutions into one superior solution."""
        if self.llm_call is None:
            return

        system = self._system_prompt()
        user = (
            f"MERGE these two solutions into one that combines the best of both.\n\n"
            f"Solution A ({sol_a.name}, score={sol_a.val_score:.4f}):\n"
            f"```python\n{sol_a.code}\n```\n\n"
            f"Solution B ({sol_b.name}, score={sol_b.val_score:.4f}):\n"
            f"```python\n{sol_b.code}\n```\n\n"
            f"Create a merged solution that uses the best preprocessing from one\n"
            f"and the best model from the other, or creates an ensemble of both.\n\n"
            f"Respond with JSON:\n"
            f'{{"name": "merged_name", "rationale": "how you combined them", '
            f'"code": "def solve(X_train, y_train, X_test):\\n    ...\\n    return predictions"}}'
        )

        try:
            raw, _ = self.llm_call(system, user)
            parsed = self._parse_json(raw)
            if parsed and "code" in parsed:
                sol = self._evaluate_solution(
                    parsed["code"], "merge", parsed.get("name", "merged"),
                    parent_id=sol_a.id,
                    generation=max(sol_a.generation, sol_b.generation) + 1,
                    rationale=parsed.get("rationale", ""),
                )
                self.tree.add(sol)
        except Exception:
            pass

    # ─────────────────────────────────────────── literature

    def _propose_from_literature(self, current_best: Solution, diagnosis: str) -> None:
        """Search literature and generate a proposal grounded in recent papers."""
        if self.llm_call is None:
            return

        # Lazy init literature search
        if self._literature is None:
            try:
                from ..research.literature import LiteratureSearch
                self._literature = LiteratureSearch(llm_call=self.llm_call)
            except Exception:
                return

        # Build query from goal and task type
        query = f"{self.profile.task_type} {self.goal}"
        if self.profile.task_type == "regression":
            query += " feature engineering tabular regression"
        else:
            query += " classification ensemble"

        try:
            context = self._literature.research(query, task_type=self.profile.task_type)

            if not context.techniques and not context.papers:
                return

            # Format techniques for LLM
            techniques_text = ""
            for t in context.techniques[:5]:
                techniques_text += f"\n- {t.name}: {t.description}"
                if t.implementation_hint:
                    techniques_text += f"\n  Implementation hint: {t.implementation_hint}"

            papers_text = ""
            for p in context.papers[:3]:
                papers_text += f"\n- [{p.title}] {p.abstract[:150]}..."

            data_desc = self._data_description()
            system = self._system_prompt()
            user = (
                f"Write a solution GROUNDED IN RECENT LITERATURE.\n\n"
                f"{data_desc}\n\n"
                f"Current best: {current_best.name} (score={current_best.val_score:.4f})\n"
                f"Diagnosis: {diagnosis}\n\n"
                f"Relevant papers:{papers_text}\n\n"
                f"Extracted techniques:{techniques_text}\n\n"
                f"Landscape: {context.landscape_summary}\n\n"
                f"Use insights from these papers to write a BETTER solution.\n\n"
                f"Respond with JSON:\n"
                f'{{"name": "descriptive_name", "rationale": "which paper technique you adapted", '
                f'"code": "def solve(X_train, y_train, X_test):\\n    ...\\n    return predictions"}}'
            )

            raw, _ = self.llm_call(system, user)
            parsed = self._parse_json(raw)
            if parsed and "code" in parsed:
                sol = self._evaluate_solution(
                    parsed["code"], "literature", parsed.get("name", "lit_informed"),
                    parent_id=current_best.id,
                    generation=current_best.generation + 1,
                    rationale=parsed.get("rationale", ""),
                    literature_context=context.landscape_summary,
                )
                self.tree.add(sol)

        except Exception:
            pass

    # ─────────────────────────────────────────── execution

    def _evaluate_solution(
        self, code: str, source: str, name: str, *,
        parent_id: Optional[str] = None, generation: int = 0,
        rationale: str = "", literature_context: str = "",
    ) -> Solution:
        """Execute a solution and evaluate it."""
        self._sol_counter += 1
        sol_id = f"sol_{self._sol_counter:04d}"

        sol = Solution(
            id=sol_id, code=code, source=source, name=name,
            parent_id=parent_id, generation=generation,
            rationale=rationale, literature_context=literature_context,
        )

        t0 = time.time()
        try:
            predictions = self._execute_code(code)
            if predictions is None:
                sol.status = "failed"
                sol.error_msg = "solve() returned None"
                return sol

            # Score with frozen metric
            from ..core.science import score_metric
            score = float(score_metric(
                self.plan.verification.metric,
                self.y_val.tolist(), predictions.tolist(),
                self.labels,
            ))

            sol.val_score = score
            sol.status = "success"
            sol.elapsed_s = time.time() - t0

            if self.verbose:
                marker = " *NEW BEST*" if score > self.tree.best_score() else ""
                self._log(f"  [{source}] {name}: {score:.4f}{marker}")

        except Exception as e:
            sol.status = "failed"
            sol.error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            sol.elapsed_s = time.time() - t0
            if self.verbose:
                self._log(f"  [{source}] {name}: FAILED ({sol.error_msg[:80]})")

        return sol

    def _clean_code(self, code: str) -> str:
        """Unescape, fix hallucinated names, strip forbidden patterns."""
        if "\\n" in code:
            real_nl = code.count("\n")
            escaped_nl = code.count("\\n")
            if escaped_nl > real_nl:
                code = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')

        for wrong, right in _NAME_FIXES.items():
            code = code.replace(wrong, right)

        if "GridSearchCV" in code or "RandomizedSearchCV" in code:
            lines = [l for l in code.split("\n")
                     if "GridSearchCV" not in l and "RandomizedSearchCV" not in l]
            code = "\n".join(lines)

        return code

    def _execute_code(self, code: str) -> Optional[np.ndarray]:
        """Execute solution code in process-isolated sandbox, return predictions on X_val."""
        code = self._clean_code(code)

        # Static check on user code BEFORE prepending preamble (preamble is trusted)
        from ..execution.sandbox import static_check
        report = static_check(code)
        if not report.ok:
            raise RuntimeError(f"static check failed: {'; '.join(report.violations[:5])}")

        full_code = _PREAMBLE + "\n" + code

        result = run_sandboxed(
            full_code, self.X_train, self.y_train, self.X_val,
            cpu_s=90, mem_mb=4096, wall_s=120.0,
            skip_static_check=True,  # already checked user code above; full code has preamble
        )
        if result.ok:
            return result.predictions
        else:
            raise RuntimeError(result.error)

    def _build_and_fit(self, code: str) -> Any:
        """Build and fit a solution via sandbox, returning a wrapper for residual analysis."""
        try:
            code = self._clean_code(code)

            # Static check on user code BEFORE prepending preamble
            from ..execution.sandbox import static_check
            report = static_check(code)
            if not report.ok:
                return None

            full_code = _PREAMBLE + "\n" + code

            # Create a wrapper that re-executes in sandbox for each predict call
            X_train, y_train = self.X_train.copy(), self.y_train.copy()
            class _SandboxWrapper:
                def __init__(self, fc):
                    self._full_code = fc
                    self._X_train = X_train
                    self._y_train = y_train
                def predict(self, X):
                    res = run_sandboxed(
                        self._full_code, self._X_train, self._y_train, X,
                        cpu_s=90, mem_mb=4096, wall_s=120.0,
                        skip_static_check=True,  # already checked user code
                    )
                    if res.ok:
                        return res.predictions
                    return None

            return _SandboxWrapper(full_code)
        except Exception:
            return None

    def _certify(self, best: Solution) -> Tuple[Optional[Dict], Any]:
        """Certify the best solution on the sealed test set via sandbox execution."""
        try:
            from ..core.science import certify_accuracy, certify_regression, score_metric

            # Execute solution: train on train+val, predict on sealed test
            X_full = np.vstack([self.X_train, self.X_val])
            y_full = np.concatenate([self.y_train, self.y_val])

            code = self._clean_code(best.code)
            full_code = _PREAMBLE + "\n" + code

            # Static check on user code before certification
            from ..execution.sandbox import static_check as _sc
            _report = _sc(code)
            if not _report.ok:
                if self.verbose:
                    self._log(f"Certification static check failed: {_report.violations[:3]}")
                return None, None

            # Run in sandbox: train on full train+val, predict on sealed test
            result = run_sandboxed(
                full_code, X_full, y_full, self.X_test,
                cpu_s=120, mem_mb=4096, wall_s=180.0,
                skip_static_check=True,  # already checked user code
            )
            if not result.ok:
                if self.verbose:
                    self._log(f"Certification sandbox failed: {result.error}")
                return None, None

            predictions = result.predictions
            metric = self.plan.verification.metric
            threshold = self.plan.verification.threshold

            score = float(score_metric(
                metric, self.y_test.tolist(), predictions.tolist(), self.labels,
            ))

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

            cert["technique"] = best.name
            cert["plan_hash"] = self.plan.plan_hash

            return cert, None  # no estimator object available from sandbox

        except Exception as e:
            if self.verbose:
                self._log(f"Certification failed: {e}")
            return None, None

    # ─────────────────────────────────────────── helpers

    def _split_data(self) -> None:
        """Split into train (60%), val (20%), sealed test (20%)."""
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

    def _next_id(self) -> str:
        self._sol_counter += 1
        return f"sol_{self._sol_counter:04d}"

    def _data_description(self) -> str:
        """Detailed data description for LLM context, including per-feature stats."""
        p = self.profile
        feature_names = p.feature_names if hasattr(p, 'feature_names') and p.feature_names else None

        # Per-feature statistics
        feature_info = ""
        if feature_names:
            feature_lines = []
            for i, name in enumerate(feature_names[:20]):
                col = self.X_train[:, i] if i < self.X_train.shape[1] else None
                if col is not None:
                    feature_lines.append(
                        f"  col[{i}] {name}: mean={np.mean(col):.3f}, "
                        f"std={np.std(col):.3f}, "
                        f"range=[{np.min(col):.3f}, {np.max(col):.3f}]"
                    )
            feature_info = "\nFeature columns (accessed as X[:, i]):\n" + "\n".join(feature_lines)
        else:
            feature_info = f"\n{p.n_features} unnamed features"

        stats = ""
        if p.task_type == "regression":
            stats = (f"\nTarget y stats: mean={np.mean(self.y):.3f}, std={np.std(self.y):.3f}, "
                     f"range=[{np.min(self.y):.3f}, {np.max(self.y):.3f}], "
                     f"median={np.median(self.y):.3f}")
            # Skewness
            from scipy.stats import skew
            y_skew = skew(self.y)
            if abs(y_skew) > 0.5:
                stats += f"\nTarget is skewed (skew={y_skew:.2f}) — consider log transform"
        else:
            unique, counts = np.unique(self.y, return_counts=True)
            n_classes = len(unique)
            balance = f"min={min(counts)}, max={max(counts)}, ratio={max(counts)/max(min(counts),1):.1f}:1"
            stats = f"\nClasses: {n_classes}, balance: {balance}"

        # Correlation with target (for regression)
        corr_info = ""
        if p.task_type == "regression" and feature_names:
            corrs = []
            for i, name in enumerate(feature_names[:20]):
                if i < self.X_train.shape[1]:
                    c = np.corrcoef(self.X_train[:, i], self.y_train)[0, 1]
                    corrs.append((name, c))
            corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            top_corrs = [f"{n}={c:.3f}" for n, c in corrs[:5]]
            corr_info = f"\nTop correlations with target: {', '.join(top_corrs)}"

        return (
            f"Goal: {self.goal}\n"
            f"Task: {p.task_type}, Metric: {self.plan.verification.metric}, "
            f"Threshold: {self.plan.verification.threshold}\n"
            f"Shape: {p.n_samples} samples, {p.n_features} features\n"
            f"Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)} (sealed)"
            f"{feature_info}{stats}{corr_info}"
            f"\nIssues: {', '.join(p.issues[:3]) if p.issues else 'none'}"
        )

    def _tree_context(self) -> str:
        """Summary of the solution tree for LLM context."""
        successful = self.tree.all_successful()
        if not successful:
            return "No successful solutions yet."

        top = sorted(successful, key=lambda s: s.val_score or 0, reverse=True)[:5]
        lines = ["Previous solutions (best to worst):"]
        for s in top:
            lines.append(f"  - {s.name}: {s.val_score:.4f} (source={s.source}, gen={s.generation})")
        return "\n".join(lines)

    def _system_prompt(self) -> str:
        """System prompt for all LLM calls."""
        return (
            "You are an expert ML engineer writing complete Python solutions.\n\n"
            "CRITICAL RULES:\n"
            "1. Define `def solve(X_train, y_train, X_test):` that returns a numpy array of predictions\n"
            "2. X_train, y_train, X_test are numpy arrays. X has feature columns in fixed order.\n"
            "3. You MAY import: numpy, scipy, sklearn (any submodule)\n"
            "4. You MUST NOT import: pandas, xgboost, lightgbm, torch, tensorflow, catboost\n"
            "5. You MUST NOT use: GridSearchCV, RandomizedSearchCV, train_test_split\n"
            "6. Be FAST — solution must run in <60s on 20k samples\n"
            "7. NEVER drop features without strong justification. ALL features may be informative.\n"
            "8. For feature engineering, ADD new columns via np.column_stack, don't drop originals.\n"
            "9. For regression, consider TransformedTargetRegressor with np.log1p/np.expm1.\n"
            "10. The strongest sklearn models are usually: HistGradientBoostingRegressor/Classifier,\n"
            "    StackingRegressor/Classifier with diverse base learners.\n"
            "11. Always set random_state=42 for reproducibility.\n\n"
            "GOOD FEATURE ENGINEERING PATTERNS:\n"
            "- Ratio features: X[:, i] / (X[:, j] + 1e-8)\n"
            "- Interaction features: X[:, i] * X[:, j]\n"
            "- Log transforms: np.log1p(np.abs(X[:, i]))\n"
            "- Polynomial: sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=True)\n"
            "- Target transform: TransformedTargetRegressor(regressor=..., func=np.log1p, inverse_func=np.expm1)\n\n"
            "Respond ONLY with valid JSON (escape newlines in code as \\n):\n"
            '{"name": "descriptive_name", "rationale": "reasoning", '
            '"code": "def solve(X_train, y_train, X_test):\\n    ...\\n    return predictions"}'
        )

    def _parse_json(self, raw: str) -> Optional[Dict]:
        """Parse JSON from LLM response, handling markdown fences and multiline code."""
        raw = raw.strip()
        # Remove markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)

        # Strategy 1: Direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract code block separately, then parse JSON envelope
        # LLMs often put real newlines inside the "code" field which breaks JSON
        code_match = re.search(r'"code"\s*:\s*"(.*?)"\s*\}', raw, re.DOTALL)
        if not code_match:
            # Try with code block in triple-quoted style
            code_match = re.search(r'"code"\s*:\s*"(.*?)(?:"\s*\})', raw, re.DOTALL)

        if code_match:
            code_raw = code_match.group(1)
            # Escape real newlines so JSON is valid
            code_escaped = code_raw.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace('"', '\\"')
            # Rebuild the JSON with escaped code
            try:
                # Find everything before "code"
                before_code = raw[:raw.index('"code"')]
                rebuilt = before_code + '"code": "' + code_escaped + '"}'
                return json.loads(rebuilt)
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: Extract name/rationale/code with regex
        name_match = re.search(r'"name"\s*:\s*"([^"]*)"', raw)
        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', raw)

        # Find code between "code": " and the closing of the JSON
        code_block = None
        code_start = raw.find('"code"')
        if code_start >= 0:
            # Find the opening quote after "code":
            colon_pos = raw.index(":", code_start)
            quote_start = raw.index('"', colon_pos + 1)
            # Find the code content — it might have real newlines
            # Look for the pattern: "\n}\n" or ""\n}" at the end
            remaining = raw[quote_start + 1:]
            # Find the end: either "\n}" or just "}" with possible whitespace
            end_patterns = ['"\n}', '"\r\n}', '"}']
            end_pos = -1
            for pat in end_patterns:
                pos = remaining.rfind(pat)
                if pos > 0:
                    end_pos = pos
                    break
            if end_pos > 0:
                code_block = remaining[:end_pos]

        if code_block and name_match:
            return {
                "name": name_match.group(1),
                "rationale": rationale_match.group(1) if rationale_match else "",
                "code": code_block,
            }

        # Strategy 4: Just extract any Python function definition
        func_match = re.search(r'(def solve\(.*?\n(?:(?:    |\t).*\n)*)', raw)
        if func_match:
            return {
                "name": "llm_generated",
                "rationale": "",
                "code": func_match.group(1),
            }

        return None

    def _log(self, msg: str) -> None:
        print(f"[attestra:gen] {msg}")
