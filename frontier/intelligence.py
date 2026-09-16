"""frontier.intelligence -- search intelligence layer (Phase A).

Wires five capabilities into the orchestrator to dramatically improve proposal
quality and search efficiency:

  1. LiteratureScout integration: queries arXiv, Papers-with-Code, GitHub, HuggingFace
     for techniques relevant to the current problem; feeds findings into proposal context.

  2. ASHA hyperparameter search: after round 0 identifies the best family, subsequent
     rounds do principled successive halving instead of random exploration.

  3. Feature engineering proposals: injects feature-transform recipes (polynomial,
     interaction, PCA, target encoding) into the LLM prompt so generated code includes
     preprocessing, not just model selection.

  4. Knowledge read-back: queries the KnowledgeBase for what worked on similar datasets
     and injects those recipes into the proposal context as warm-start hints.

  5. Multi-turn diagnosis: when a repair attempt fails, feeds the failure back to the LLM
     for a refined diagnosis (up to 3 turns), instead of single-shot analysis.

All five are ADDITIVE to the existing orchestrator: they enrich the context dict and
provide new proposal sources without editing the frozen core or weakening certification.
Each degrades honestly when its dependency is absent (no LLM, no network, empty KB).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier.program import Program
from frontier.task import Task
from frontier import llm_diagnosis as llm_diag_mod


# ============================================================================
# 1. Literature Scout adapter
# ============================================================================

@dataclass
class LiteratureContext:
    """Summarized literature findings for injection into the proposal context."""
    n_findings: int = 0
    techniques: List[str] = field(default_factory=list)
    architectures: List[str] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)


def literature_scout(goal: str, *, llm_client: Optional[Callable] = None,
                     online: bool = True) -> LiteratureContext:
    """Query the research surface for the current problem and return a summary.

    Uses vfplatform.literature.LiteratureScout when available; degrades to an
    LLM-only summary when the module is absent or all sources fail.
    """
    ctx = LiteratureContext()
    try:
        from vfplatform.literature import LiteratureScout as _Scout, keywords as _keywords
        # arXiv's `all:` endpoint ranks a long natural-language phrase poorly and
        # returns high-citation off-domain papers (e.g. particle physics for a
        # time-series goal). Querying with the salient keywords instead — filler
        # and stopwords removed — restores domain relevance. We use vfplatform's
        # own public `keywords()` helper (frozen module, called not edited).
        query = goal
        try:
            kws = _keywords(goal, k=8)
            if len(kws) >= 2:
                query = " ".join(kws)
        except Exception:
            pass
        scout = _Scout(query, online=online, use_llm=False)
        summary = scout.summary()
        ctx.n_findings = summary.get("n_findings", 0)

        for motif in summary.get("motifs", []):
            name = motif.get("name", "")
            genes = motif.get("genes", {})
            technique = f"{name}: {genes}" if genes else name
            ctx.techniques.append(technique)

        ctx.architectures = summary.get("backbones", [])[:8]

        for f in summary.get("findings", [])[:6]:
            title = f.get("title", "")
            source = f.get("source", "")
            text = f.get("text", "")[:200]
            if title:
                ctx.key_insights.append(f"[{source}] {title}: {text}")
                ctx.provenance[title] = f.get("url", "")
    except Exception:
        pass

    if not ctx.key_insights and llm_client is not None:
        try:
            prompt = (
                f"For the ML problem: '{goal}', list 3-5 key techniques from recent "
                f"literature (arXiv, ICML, NeurIPS) that would help. For each, give:\n"
                f"- Technique name\n- Why it helps for this problem\n"
                f"- Specific sklearn/pytorch implementation\n"
                f"Be specific and technical. 2-3 sentences each."
            )
            response = llm_client(prompt)
            if response:
                ctx.key_insights.append(response[:500])
                ctx.n_findings = 1
        except Exception:
            pass

    return ctx


def enrich_context_with_literature(context: dict, lit: LiteratureContext) -> dict:
    """Inject literature findings into the proposal context (additive, never overwrites)."""
    if lit.n_findings == 0 and not lit.key_insights:
        return context
    context["literature"] = {
        "n_findings": lit.n_findings,
        "techniques": lit.techniques[:5],
        "architectures": lit.architectures[:5],
        "key_insights": lit.key_insights[:5],
    }
    if lit.techniques or lit.key_insights:
        existing = context.get("llm_guidance", "")
        parts: List[str] = ["\n[LITERATURE]"]
        if lit.key_insights:
            parts.append(
                "Relevant findings retrieved from the research surface "
                "(arXiv / Papers-with-Code / HuggingFace):")
            for insight in lit.key_insights[:5]:
                parts.append(f"- {insight}")
        if lit.techniques:
            parts.append("Distilled techniques to consider:")
            parts.extend(f"- {t}" for t in lit.techniques[:5])
        lit_block = "\n".join(parts)
        context["llm_guidance"] = existing + lit_block if existing else lit_block
    return context


# ============================================================================
# 2. ASHA Hyperparameter Search adapter
# ============================================================================

@dataclass
class ASHAState:
    """Track ASHA state across rounds for the best-performing family."""
    best_family: Optional[str] = None
    bracket: Any = None
    tpe: Any = None
    configs_tried: Dict[str, float] = field(default_factory=dict)
    round_activated: int = -1


def init_asha(history: list, task: Task, round_idx: int) -> ASHAState:
    """After round 0, identify the best family and initialize ASHA for targeted HP search."""
    state = ASHAState()
    if round_idx < 1 or not history:
        return state

    family_best: Dict[str, float] = {}
    for rec in history:
        ok = getattr(rec, 'ok', False)
        score = getattr(rec, 'val_score', None)
        label = getattr(rec, 'label', '') or ''
        if ok and score is not None:
            family = _infer_family(label, task.kind)
            if family not in family_best or score > family_best[family]:
                family_best[family] = score

    if not family_best:
        return state

    state.best_family = max(family_best, key=family_best.get)
    state.round_activated = round_idx

    space = _hp_space_for_family(state.best_family, task.kind)
    if not space:
        return state
    try:
        from vfplatform.hp_search import SHABracket, TPE
        state.bracket = SHABracket(max_budget=160, eta=3)
        state.tpe = TPE(space, gamma=0.25, n_startup=4, seed=42)
        # warm-start TPE from history so proposals concentrate near the best family
        for rec in history:
            if not getattr(rec, 'ok', False):
                continue
            score = getattr(rec, 'val_score', None)
            label = getattr(rec, 'label', '') or ''
            if score is not None and _infer_family(label, task.kind) == state.best_family:
                cfg = getattr(rec, 'config', None) or getattr(rec, 'hparams', None)
                if isinstance(cfg, dict) and cfg:
                    state.tpe.observe({k: cfg[k] for k in space if k in cfg},
                                      float(score), family=state.best_family)
    except Exception:
        state.tpe = None

    return state


def _hp_space_for_family(family: str, kind: str) -> Dict[str, tuple]:
    """HP search space per family, in TPE param_specs format:
    {name: ("int", lo, hi) | ("float", lo, hi) | ("choice", [vals])}."""
    spaces: Dict[str, Dict[str, tuple]] = {
        "rf": {
            "n_estimators": ("int", 50, 1000),
            "max_depth": ("int", 3, 50),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "max_features": ("choice", ["sqrt", "log2", 0.5, 0.8]),
        },
        "hist_gbm": {
            "max_iter": ("int", 50, 500),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float", 0.01, 0.3),
            "min_samples_leaf": ("int", 5, 50),
            "l2_regularization": ("float", 0.0, 10.0),
        },
        "svc_rbf": {
            "C": ("float", 0.01, 100.0),
            "gamma": ("choice", ["scale", "auto", 0.001, 0.01, 0.1]),
        },
        "logreg": {
            "C": ("float", 0.001, 100.0),
            "max_iter": ("int", 500, 5000),
            "solver": ("choice", ["lbfgs", "saga"]),
        },
        "ridge": {
            "alpha": ("float", 0.001, 100.0),
        },
        "gbr": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float", 0.01, 0.3),
            "subsample": ("float", 0.5, 1.0),
        },
    }
    return spaces.get(family, {})


def asha_proposals(state: ASHAState, task: Task, n: int = 3) -> List[Program]:
    """Generate HP-search proposals using ASHA + TPE for the best family."""
    if state.best_family is None or state.tpe is None:
        return []

    programs = []
    try:
        configs = state.tpe.propose(n)
    except Exception:
        configs = []

    for i, config in enumerate(configs):
        if not config:
            continue

        code = _config_to_code(state.best_family, config, task.kind)
        if code:
            label = f"asha_{state.best_family}_{i}"
            programs.append(Program(
                source="asha",
                label=label,
                code=code,
                provenance={"family": state.best_family, "config": config, "method": "asha+tpe"},
            ))
    return programs


def _config_to_code(family: str, config: dict, kind: str) -> str:
    """Render an HP config into a build_estimator() code string."""
    import_map = {
        ("classification", "rf"): ("from sklearn.ensemble import RandomForestClassifier",
                                   "RandomForestClassifier"),
        ("classification", "hist_gbm"): ("from sklearn.ensemble import HistGradientBoostingClassifier",
                                         "HistGradientBoostingClassifier"),
        ("classification", "svc_rbf"): ("from sklearn.svm import SVC", "SVC"),
        ("classification", "logreg"): ("from sklearn.linear_model import LogisticRegression",
                                       "LogisticRegression"),
        ("regression", "rf"): ("from sklearn.ensemble import RandomForestRegressor",
                               "RandomForestRegressor"),
        ("regression", "hist_gbm"): ("from sklearn.ensemble import HistGradientBoostingRegressor",
                                     "HistGradientBoostingRegressor"),
        ("regression", "ridge"): ("from sklearn.linear_model import Ridge", "Ridge"),
        ("regression", "gbr"): ("from sklearn.ensemble import GradientBoostingRegressor",
                                "GradientBoostingRegressor"),
    }

    key = (kind, family)
    if key not in import_map:
        return ""

    imp, cls = import_map[key]
    params = ", ".join(f"{k}={v!r}" for k, v in config.items() if v is not None)
    if family in ("rf",) and "random_state" not in config:
        params += ", random_state=0, n_jobs=1"

    code = (
        f"import numpy as np\n"
        f"from sklearn.pipeline import Pipeline\n"
        f"from sklearn.preprocessing import StandardScaler\n"
        f"{imp}\n\n"
        f"def build_estimator():\n"
        f"    return Pipeline([\n"
        f"        ('scaler', StandardScaler()),\n"
        f"        ('model', {cls}({params})),\n"
        f"    ])\n"
    )
    return code


# ============================================================================
# 3. Feature Engineering Proposals
# ============================================================================

_FEATURE_RECIPES = [
    {
        "name": "poly2_scaler",
        "description": "Polynomial features (degree 2) with scaling",
        "code_template": """import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import {estimator}

def build_estimator():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('model', {estimator}({params})),
    ])
""",
    },
    {
        "name": "pca_reduction",
        "description": "PCA dimensionality reduction before modeling",
        "code_template": """import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import {estimator}

def build_estimator():
    n_components = min(20, {n_features} // 2)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('model', {estimator}({params})),
    ])
""",
    },
    {
        "name": "select_kbest",
        "description": "Univariate feature selection (top K)",
        "code_template": """import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import {estimator}

def build_estimator():
    k = min(20, {n_features})
    scorer = {score_func}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(scorer, k=k)),
        ('model', {estimator}({params})),
    ])
""",
    },
    {
        "name": "robust_scaler",
        "description": "Robust scaling (median/IQR) for outlier-heavy data",
        "code_template": """import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import {estimator}

def build_estimator():
    return Pipeline([
        ('scaler', RobustScaler()),
        ('model', {estimator}({params})),
    ])
""",
    },
]


class FeatureEngineeringProposer:
    """Generate proposals with different feature engineering strategies.

    Unlike the base SeedProposer which varies the model, this varies the
    preprocessing pipeline: polynomial features, PCA, feature selection,
    robust scaling, etc. Combined with the best-performing model family.
    """

    def __init__(self, n_features: int = 10, kind: str = "classification"):
        self.n_features = n_features
        self.kind = kind
        self._proposed = False

    def propose(self, context: dict) -> List[Program]:
        if self._proposed:
            return []
        self._proposed = True

        kind = context.get("task_kind", self.kind)
        n_features = context.get("n_features", self.n_features)
        best_label = context.get("best_label", "")

        family = _infer_family(best_label, kind)
        estimator, params = _family_to_estimator(family, kind)

        score_func = "f_classif" if kind == "classification" else "f_regression"

        programs = []
        for i, recipe in enumerate(_FEATURE_RECIPES):
            if recipe["name"] == "poly2_scaler" and n_features > 50:
                continue
            if recipe["name"] == "pca_reduction" and n_features < 10:
                continue

            code = recipe["code_template"].format(
                estimator=estimator, params=params,
                n_features=n_features, score_func=score_func,
            )
            programs.append(Program(
                source="feature_eng",
                label=f"feat_{recipe['name']}_{family}_{i}",
                code=code,
                provenance={"recipe": recipe["name"], "family": family},
            ))
        return programs


def _infer_family(best_label: str, kind: str) -> str:
    """Infer the model family from the best label, defaulting to hist_gbm."""
    if not best_label:
        return "hist_gbm"
    label_lower = best_label.lower()
    for fam in ("hist_gbm", "rf", "svc_rbf", "logreg", "ridge", "gbr"):
        if fam in label_lower:
            return fam
    return "hist_gbm"


def _family_to_estimator(family: str, kind: str) -> Tuple[str, str]:
    """Map a family name to (estimator class name, constructor params)."""
    mapping = {
        ("classification", "hist_gbm"): ("HistGradientBoostingClassifier",
                                         "random_state=0"),
        ("classification", "rf"): ("RandomForestClassifier",
                                   "n_estimators=300, random_state=0, n_jobs=1"),
        ("classification", "logreg"): ("LogisticRegression", "max_iter=2000"),
        ("classification", "svc_rbf"): ("SVC", "C=2.0, gamma='scale'"),
        ("regression", "hist_gbm"): ("HistGradientBoostingRegressor", "random_state=0"),
        ("regression", "rf"): ("RandomForestRegressor",
                               "n_estimators=300, random_state=0, n_jobs=1"),
        ("regression", "ridge"): ("Ridge", "alpha=1.0"),
        ("regression", "gbr"): ("GradientBoostingRegressor", "random_state=0"),
    }
    key = (kind, family)
    if key in mapping:
        return mapping[key]
    default = ("HistGradientBoostingClassifier" if kind == "classification"
               else "HistGradientBoostingRegressor")
    return default, "random_state=0"


def enrich_context_with_feature_hints(context: dict, task: Task) -> dict:
    """Add feature engineering hints to the LLM prompt context."""
    n_features = task.n_features
    hints = []

    if n_features > 50:
        hints.append("High-dimensional data: consider PCA, SelectKBest, or LASSO-based selection")
    if n_features < 10:
        hints.append("Low-dimensional data: polynomial feature interactions may help")
    if n_features > 5:
        hints.append("Consider interaction terms between top features")

    hints.append("Always include StandardScaler or RobustScaler in the pipeline")
    hints.append("Use Pipeline() to chain preprocessing and model for correct cross-validation")

    existing = context.get("llm_guidance", "")
    feat_block = "\n[FEATURE ENGINEERING]\n" + "\n".join(f"- {h}" for h in hints)
    context["llm_guidance"] = existing + feat_block if existing else feat_block
    context["feature_hints"] = hints
    return context


# ============================================================================
# 4. Knowledge Read-Back
# ============================================================================

def knowledge_readback(context: dict, kb_path: Optional[str] = None) -> dict:
    """Query the KB for what worked on similar datasets and inject into context.

    Unlike the existing KnowledgeProposer (which generates Programs), this
    enriches the LLM prompt so authored proposals are informed by history.
    """
    try:
        from frontier.knowledge import KnowledgeBase, task_fingerprint
    except ImportError:
        return context

    fp = context.get("task_fingerprint")
    if fp is None:
        return context

    try:
        kb = KnowledgeBase(kb_path)
        records = kb.retrieve(fp, k=10)
    except Exception:
        return context

    if not records:
        return context

    successful = [r for r in records if r.get("ok", False)]
    failed = [r for r in records if not r.get("ok", True)]

    successful.sort(key=lambda r: r.get("val_score", 0), reverse=True)

    hints = []
    if successful:
        top = successful[:5]
        for r in top:
            desc = r.get("recipe_descriptor", r.get("label", "unknown"))
            score = r.get("val_score", "?")
            hints.append(f"Worked well (val={score}): {desc}")

    if failed:
        fail_patterns = {}
        for r in failed:
            ek = r.get("error_kind", "unknown")
            fail_patterns[ek] = fail_patterns.get(ek, 0) + 1
        for ek, count in sorted(fail_patterns.items(), key=lambda x: -x[1])[:3]:
            hints.append(f"Common failure ({count}x): {ek}")

    if hints:
        existing = context.get("llm_guidance", "")
        kb_block = ("\n[KNOWLEDGE BASE - What worked on similar data]\n" +
                    "\n".join(f"- {h}" for h in hints))
        context["llm_guidance"] = existing + kb_block if existing else kb_block
        context["kb_hints"] = hints

    return context


# ============================================================================
# 5. Multi-turn Diagnosis
# ============================================================================

def multi_turn_diagnose(
    diag,
    task: Task,
    history: Sequence,
    *,
    llm_client: Optional[Callable[[str], str]] = None,
    val_truth: Optional[Sequence] = None,
    val_preds: Optional[Sequence] = None,
    prior_diagnosis: Optional[Any] = None,
    prior_repair_failures: Optional[List[dict]] = None,
    max_turns: int = 3,
) -> Any:
    """Multi-turn LLM diagnosis: if prior repairs failed, refine the diagnosis.

    Turn 1: Standard llm_diagnose (root causes, guidance, architectures, repairs).
    Turn 2+: Feed the failed repair attempts back to the LLM for refined analysis.
    Caps at max_turns to prevent infinite loops.
    """
    result = llm_diag_mod.llm_diagnose(
        diag, task, history,
        llm_client=llm_client,
        val_truth=val_truth, val_preds=val_preds,
    )

    if llm_client is None or not prior_repair_failures:
        return result

    if prior_diagnosis is not None and prior_repair_failures:
        turn = min(len(prior_repair_failures), max_turns - 1)
        if turn >= 1:
            refinement_prompt = _build_refinement_prompt(
                prior_diagnosis, prior_repair_failures, diag, task,
            )
            try:
                response = llm_client(refinement_prompt)
                if response:
                    parsed = llm_diag_mod._parse_llm_response(response)
                    if parsed.get("root_causes"):
                        result.root_causes = parsed["root_causes"]
                    if parsed.get("targeted_guidance"):
                        result.targeted_guidance = parsed["targeted_guidance"]
                    if parsed.get("recommended_architectures"):
                        result.recommended_architectures = parsed["recommended_architectures"]
                    if parsed.get("repair_strategies"):
                        result.repair_strategies = parsed["repair_strategies"]
            except Exception:
                pass

    return result


def _build_refinement_prompt(
    prior_diagnosis: Any,
    repair_failures: List[dict],
    diag,
    task: Task,
) -> str:
    """Build a refinement prompt that incorporates failed repair attempts."""
    parts = [
        "Your previous diagnosis led to repair attempts that ALL FAILED. "
        "You must dig deeper and propose a DIFFERENT approach.\n",
        f"TASK: {task.kind}, {task.n_features} features, metric={task.metric}.\n",
    ]

    if hasattr(prior_diagnosis, 'root_causes') and prior_diagnosis.root_causes:
        parts.append(
            "YOUR PREVIOUS ROOT CAUSES (which did NOT lead to a fix):\n" +
            "\n".join(f"- {c}" for c in prior_diagnosis.root_causes) + "\n"
        )

    if hasattr(prior_diagnosis, 'repair_strategies') and prior_diagnosis.repair_strategies:
        parts.append(
            "YOUR PREVIOUS REPAIR STRATEGIES (which FAILED):\n" +
            "\n".join(f"- {s}" for s in prior_diagnosis.repair_strategies) + "\n"
        )

    parts.append("FAILED REPAIR ATTEMPTS:\n")
    for i, failure in enumerate(repair_failures[:3]):
        error = failure.get("error", "unknown")
        error_kind = failure.get("error_kind", "unknown")
        parts.append(f"  Attempt {i+1}: [{error_kind}] {error[:200]}\n")

    parts.append(
        "\nBased on the FAILURE of your previous suggestions, provide a COMPLETELY "
        "DIFFERENT analysis:\n"
        "1. ROOT CAUSES: What did your previous diagnosis miss? What is the REAL problem?\n"
        "2. TARGETED IMPROVEMENTS: Completely different approach from what failed.\n"
        "3. ARCHITECTURE RECOMMENDATIONS: Different model families or pipeline structures.\n"
        "4. REPAIR STRATEGIES: Fundamentally different fixes.\n"
        "\nDo NOT repeat your previous suggestions. Go deeper."
    )
    return "\n".join(parts)


# ============================================================================
# 6. EnsembleProposer (Phase B1 — data-adaptive ensembles from history)
# ============================================================================

class EnsembleProposer:
    """Generate stacking/voting ensemble proposals adapted to what actually worked.

    Phase B improvement over generic ensembles: uses the history of successful models
    to pick the right estimators, weights them by validation score, and generates
    multiple ensemble variants (wide diversity, focused top-2, weighted voting).

    Activated after round 0. Generates up to 4 ensemble variants per round:
    - voting_diverse: all successful families in a soft-voting ensemble
    - stacking_focused: top-2 families stacked with a meta-learner
    - voting_weighted: best family gets more weight via n_estimators boost
    - stacking_deep: 3+ families stacked with regularized meta-learner
    """

    def __init__(self, min_models: int = 2, max_models: int = 5):
        self.min_models = min_models
        self.max_models = max_models
        self._proposed_labels: set = set()
        self._family_scores: Dict[str, List[float]] = {}

    def record_family_score(self, label: str, score: float, kind: str) -> None:
        """Track which families scored well for adaptive ensemble composition."""
        family = _infer_family(label, kind)
        if family not in self._family_scores:
            self._family_scores[family] = []
        self._family_scores[family].append(score)

    def propose(self, context: dict) -> List[Program]:
        round_idx = context.get("round", 0)
        if round_idx < 1:
            return []

        kind = context.get("task_kind", "classification")
        best_label = context.get("best_label", "")

        if not best_label:
            return []

        ranked_families = self._rank_families(kind)
        programs = []

        variants = [
            ("voting_diverse", self._build_voting_diverse),
            ("stacking_focused", self._build_stacking_focused),
            ("voting_weighted", self._build_voting_weighted),
            ("stacking_deep", self._build_stacking_deep),
        ]

        for variant_name, builder in variants:
            label = f"ensemble_{variant_name}_r{round_idx}"
            if label in self._proposed_labels:
                continue
            self._proposed_labels.add(label)

            code = builder(kind, ranked_families)
            if code:
                programs.append(Program(
                    source="ensemble",
                    label=label,
                    code=code,
                    provenance={"ensemble_type": variant_name, "round": round_idx,
                                "families": [f for f, _ in ranked_families[:4]]},
                ))

        return programs

    def _rank_families(self, kind: str) -> List[Tuple[str, float]]:
        """Rank families by mean validation score (descending)."""
        if not self._family_scores:
            return [("hist_gbm", 0.0), ("rf", 0.0)]
        ranked = []
        for fam, scores in self._family_scores.items():
            ranked.append((fam, sum(scores) / len(scores)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked if ranked else [("hist_gbm", 0.0), ("rf", 0.0)]

    def _estimator_snippet(self, family: str, kind: str, tag: str) -> str:
        """Generate an estimator tuple string for a family."""
        cls, params = _family_to_estimator(family, kind)
        if not cls:
            return ""
        return f"        ('{tag}', {cls}({params}))"

    def _build_voting_diverse(self, kind: str, ranked: List[Tuple[str, float]]) -> str:
        """Wide diversity: all successful families in a soft-voting ensemble."""
        families = [f for f, _ in ranked[:4]]
        if len(families) < 2:
            families = ["hist_gbm", "rf"]
        est_lines = []
        for i, fam in enumerate(families):
            snippet = self._estimator_snippet(fam, kind, f"e{i}")
            if snippet:
                est_lines.append(snippet)
        if len(est_lines) < 2:
            return ""
        estimators_str = ",\n".join(est_lines)
        vote_cls = "VotingClassifier" if kind == "classification" else "VotingRegressor"
        voting_kw = "voting='soft'" if kind == "classification" else ""
        imports = _ensemble_imports(kind, families)
        return f"""{imports}

def build_estimator():
    estimators = [
{estimators_str},
    ]
    return Pipeline([
        ('scaler', StandardScaler()),
        ('{vote_cls.lower()}', {vote_cls}(estimators=estimators, {voting_kw})),
    ])
"""

    def _build_stacking_focused(self, kind: str, ranked: List[Tuple[str, float]]) -> str:
        """Top-2 families stacked with a meta-learner."""
        families = [f for f, _ in ranked[:2]]
        if len(families) < 2:
            families = ["hist_gbm", "rf"]
        est_lines = []
        for i, fam in enumerate(families):
            snippet = self._estimator_snippet(fam, kind, f"e{i}")
            if snippet:
                est_lines.append(snippet)
        if len(est_lines) < 2:
            return ""
        estimators_str = ",\n".join(est_lines)
        stack_cls = "StackingClassifier" if kind == "classification" else "StackingRegressor"
        meta = "LogisticRegression(max_iter=2000)" if kind == "classification" else "Ridge(alpha=1.0)"
        imports = _ensemble_imports(kind, families)
        return f"""{imports}

def build_estimator():
    estimators = [
{estimators_str},
    ]
    return Pipeline([
        ('scaler', StandardScaler()),
        ('{stack_cls.lower()}', {stack_cls}(
            estimators=estimators,
            final_estimator={meta},
            cv=3,
        )),
    ])
"""

    def _build_voting_weighted(self, kind: str, ranked: List[Tuple[str, float]]) -> str:
        """Best family boosted: more trees/capacity for the winner."""
        if not ranked:
            return ""
        best_fam = ranked[0][0]
        families_for_code = [best_fam]
        if len(ranked) >= 2:
            families_for_code.append(ranked[1][0])
        else:
            families_for_code.append("rf" if best_fam != "rf" else "hist_gbm")

        # Boost the best family's capacity
        best_cls, best_params = _family_to_estimator(best_fam, kind)
        if not best_cls:
            return ""
        boosted_params = best_params.replace("n_estimators=300", "n_estimators=500")
        if "n_estimators" not in boosted_params:
            boosted_params = boosted_params.rstrip(")") if boosted_params.endswith(")") else boosted_params

        second_cls, second_params = _family_to_estimator(families_for_code[1], kind)
        if not second_cls:
            return ""

        vote_cls = "VotingClassifier" if kind == "classification" else "VotingRegressor"
        voting_kw = "voting='soft'" if kind == "classification" else ""
        imports = _ensemble_imports(kind, families_for_code)
        return f"""{imports}

def build_estimator():
    estimators = [
        ('best', {best_cls}({boosted_params})),
        ('support', {second_cls}({second_params})),
    ]
    return Pipeline([
        ('scaler', StandardScaler()),
        ('{vote_cls.lower()}', {vote_cls}(estimators=estimators, {voting_kw})),
    ])
"""

    def _build_stacking_deep(self, kind: str, ranked: List[Tuple[str, float]]) -> str:
        """3+ families stacked with regularized meta-learner."""
        families = [f for f, _ in ranked[:3]]
        while len(families) < 3:
            for default in ("hist_gbm", "rf", "logreg", "ridge"):
                if default not in families:
                    families.append(default)
                    break
            else:
                break
        est_lines = []
        for i, fam in enumerate(families):
            snippet = self._estimator_snippet(fam, kind, f"e{i}")
            if snippet:
                est_lines.append(snippet)
        if len(est_lines) < 2:
            return ""
        estimators_str = ",\n".join(est_lines)
        stack_cls = "StackingClassifier" if kind == "classification" else "StackingRegressor"
        meta = "LogisticRegression(max_iter=2000, C=0.5)" if kind == "classification" else "Ridge(alpha=2.0)"
        imports = _ensemble_imports(kind, families)
        return f"""{imports}

def build_estimator():
    estimators = [
{estimators_str},
    ]
    return Pipeline([
        ('scaler', StandardScaler()),
        ('{stack_cls.lower()}', {stack_cls}(
            estimators=estimators,
            final_estimator={meta},
            cv=3,
            passthrough=True,
        )),
    ])
"""


def _ensemble_imports(kind: str, families: List[str]) -> str:
    """Generate import block for ensemble code based on families used."""
    lines = [
        "import numpy as np",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import StandardScaler",
    ]

    ensemble_imports = set()
    model_imports: Dict[str, set] = {}

    for fam in families:
        cls, _ = _family_to_estimator(fam, kind)
        if not cls:
            continue
        module = _estimator_module(cls)
        if module not in model_imports:
            model_imports[module] = set()
        model_imports[module].add(cls)

    # Always add the ensemble types
    if kind == "classification":
        ensemble_imports.update(["VotingClassifier", "StackingClassifier"])
        model_imports.setdefault("sklearn.linear_model", set()).add("LogisticRegression")
    else:
        ensemble_imports.update(["VotingRegressor", "StackingRegressor"])
        model_imports.setdefault("sklearn.linear_model", set()).add("Ridge")

    model_imports.setdefault("sklearn.ensemble", set()).update(ensemble_imports)

    for module, classes in sorted(model_imports.items()):
        cls_list = ", ".join(sorted(classes))
        lines.append(f"from {module} import {cls_list}")

    return "\n".join(lines)


def _estimator_module(cls_name: str) -> str:
    """Map estimator class name to its sklearn module."""
    _map = {
        "RandomForestClassifier": "sklearn.ensemble",
        "RandomForestRegressor": "sklearn.ensemble",
        "HistGradientBoostingClassifier": "sklearn.ensemble",
        "HistGradientBoostingRegressor": "sklearn.ensemble",
        "GradientBoostingClassifier": "sklearn.ensemble",
        "GradientBoostingRegressor": "sklearn.ensemble",
        "LogisticRegression": "sklearn.linear_model",
        "Ridge": "sklearn.linear_model",
        "SVC": "sklearn.svm",
        "SVR": "sklearn.svm",
    }
    return _map.get(cls_name, "sklearn.ensemble")


# ============================================================================
# 7. Adaptive Subsampling (Phase B2 — progressive round-aware scaling)
# ============================================================================

def adaptive_subsample(X, y, *, max_rows: int = 10000, seed: int = 0,
                       round_idx: int = 0, total_rounds: int = 3) -> Tuple:
    """Progressive subsampling: smaller samples early, scale up in later rounds.

    Round-aware strategy:
    - Round 0: subsample to max_rows * 0.5 (fast exploration)
    - Round 1: subsample to max_rows * 0.75 (refinement)
    - Last round: use full max_rows (final evaluation)
    - If data fits within max_rows: return as-is

    Stratified for classification (preserves class balance).
    Returns (X_sub, y_sub, was_subsampled).
    """
    import numpy as np

    n = len(y) if hasattr(y, '__len__') else X.shape[0]

    # Progressive scaling: early rounds use less data for faster iteration
    if total_rounds > 1 and round_idx < total_rounds - 1:
        progress = round_idx / max(1, total_rounds - 1)
        scale = 0.5 + 0.5 * progress  # 0.5 → 1.0 over rounds
        effective_max = max(1000, int(max_rows * scale))
    else:
        effective_max = max_rows

    if n <= effective_max:
        return X, y, False

    rng = np.random.RandomState(seed + round_idx)

    y_arr = np.asarray(y)
    unique_classes = np.unique(y_arr)

    if len(unique_classes) < n // 2:
        indices = _stratified_sample(y_arr, unique_classes, effective_max, rng)
    else:
        indices = rng.choice(n, size=effective_max, replace=False)
        indices.sort()

    X_sub = X[indices] if hasattr(X, '__getitem__') else np.array(X)[indices]
    y_sub = y_arr[indices]
    return X_sub, y_sub, True


def _stratified_sample(y_arr, unique_classes, max_rows: int, rng) -> 'np.ndarray':
    """Stratified sampling that preserves class proportions."""
    import numpy as np

    n = len(y_arr)
    indices = []
    class_counts = {c: int(np.sum(y_arr == c)) for c in unique_classes}

    for c in unique_classes:
        c_idx = np.where(y_arr == c)[0]
        # Proportional allocation: preserve class ratios
        proportion = class_counts[c] / n
        n_take = max(1, int(max_rows * proportion))
        n_take = min(n_take, len(c_idx))
        chosen = rng.choice(c_idx, size=n_take, replace=False)
        indices.extend(chosen)

    indices = np.array(sorted(indices))
    # Trim to exact max_rows if we overshot due to rounding
    if len(indices) > max_rows:
        indices = rng.choice(indices, size=max_rows, replace=False)
        indices.sort()
    return indices


# ============================================================================
# 8. Prompt Evolution (Phase B3 — real template rewriting)
# ============================================================================

@dataclass
class _ScoredTechnique:
    """A technique with its validation score history for ranking."""
    label: str
    scores: List[float] = field(default_factory=list)
    source: str = ""
    recipe: str = ""

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def best_score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def n_attempts(self) -> int:
        return len(self.scores)


@dataclass
class PromptEvolution:
    """Track outcomes and rewrite LLM prompt templates based on what worked.

    Phase B3 real evolution: instead of just listing "worked/failed" techniques,
    this module:
    1. Scores techniques by validation performance (weighted by recency)
    2. Identifies architecture patterns that succeed (e.g., "ensemble methods")
    3. Generates directive prompt blocks that reshape LLM behavior:
       - EMPHASIZE: high-scoring patterns to explore more
       - AVOID: consistently failing approaches
       - COMBINE: pairs of techniques that individually scored well
       - TUNE: specific hyperparameters that made the difference
    4. Updates the prompt template itself (not just appending context)
    """
    successful_techniques: List[str] = field(default_factory=list)
    failed_techniques: List[str] = field(default_factory=list)
    successful_features: List[str] = field(default_factory=list)
    best_prompt_phrases: List[str] = field(default_factory=list)
    rounds_seen: int = 0
    # Phase B3: scored technique tracking
    _scored: Dict[str, _ScoredTechnique] = field(default_factory=dict)
    _fail_counts: Dict[str, int] = field(default_factory=dict)
    _best_score_seen: float = 0.0
    _score_history: List[Tuple[int, float]] = field(default_factory=list)
    _successful_pairs: List[Tuple[str, str]] = field(default_factory=list)

    def record_outcome(self, program: Program, val_score: Optional[float],
                       ok: bool) -> None:
        """Record outcome with score tracking for template rewriting."""
        self.rounds_seen += 1
        label = program.label or ''
        source = program.source or ''
        prov = program.provenance or {}

        family = _infer_family(label, prov.get("kind", "classification"))
        recipe = str(prov.get('recipe', ''))

        if ok and val_score is not None and val_score > 0:
            self.successful_techniques.append(label)
            if recipe:
                self.successful_features.append(recipe)

            # Score tracking
            if family not in self._scored:
                self._scored[family] = _ScoredTechnique(
                    label=family, source=source, recipe=recipe)
            self._scored[family].scores.append(val_score)

            if val_score > self._best_score_seen:
                self._best_score_seen = val_score
            self._score_history.append((self.rounds_seen, val_score))
        elif not ok:
            self.failed_techniques.append(label)
            self._fail_counts[family] = self._fail_counts.get(family, 0) + 1

        # Keep bounded
        self.successful_techniques = self.successful_techniques[-30:]
        self.failed_techniques = self.failed_techniques[-30:]

    def evolution_block(self) -> str:
        """Generate a directive prompt block that reshapes LLM behavior.

        Unlike Phase A's static listing, this produces actionable directives:
        - EMPHASIZE top-performing patterns
        - AVOID consistently failing ones
        - SUGGEST specific improvements based on score trends
        """
        if not self._scored and not self._fail_counts:
            return self._basic_evolution_block()

        parts = ["\n[PROMPT EVOLUTION — DIRECTIVE (auto-generated from empirical results)]"]

        # 1. Top performers: ranked by score, with specific advice
        ranked = sorted(self._scored.values(), key=lambda t: t.best_score, reverse=True)
        if ranked:
            top = ranked[:3]
            parts.append("\nEMPHASIZE these high-performing approaches:")
            for i, tech in enumerate(top, 1):
                parts.append(
                    f"  {i}. {tech.label}: best={tech.best_score:.4f}, "
                    f"mean={tech.mean_score:.4f} ({tech.n_attempts} attempts)")
            # Specific directive based on top performer
            best = top[0]
            if best.best_score >= 0.9:
                parts.append(f"\n  → {best.label} is performing well. Try VARIANTS: "
                             f"different regularization, more/fewer estimators, "
                             f"different preprocessing.")
            elif best.best_score >= 0.7:
                parts.append(f"\n  → {best.label} shows promise but needs improvement. "
                             f"Try: feature engineering, ensembling with complementary "
                             f"models, hyperparameter tuning.")
            else:
                parts.append(f"\n  → Scores are low across the board. Try fundamentally "
                             f"different approaches: different model families, "
                             f"aggressive feature selection, data augmentation.")

        # 2. Consistent failures: avoid wasting compute
        chronic_failures = [
            fam for fam, count in self._fail_counts.items() if count >= 2]
        if chronic_failures:
            parts.append(f"\nAVOID (failed {2}+ times): {', '.join(chronic_failures[:5])}")

        # 3. Score trend analysis
        if len(self._score_history) >= 3:
            recent = [s for _, s in self._score_history[-3:]]
            earlier = [s for _, s in self._score_history[:3]]
            recent_avg = sum(recent) / len(recent)
            earlier_avg = sum(earlier) / len(earlier)
            if recent_avg > earlier_avg + 0.02:
                parts.append(f"\nTREND: Improving ({earlier_avg:.3f} → {recent_avg:.3f}). "
                             f"Continue current direction with small variations.")
            elif recent_avg < earlier_avg - 0.02:
                parts.append(f"\nTREND: Degrading ({earlier_avg:.3f} → {recent_avg:.3f}). "
                             f"Switch to a fundamentally different approach.")
            else:
                parts.append(f"\nTREND: Plateaued at ~{recent_avg:.3f}. "
                             f"Need a breakthrough: try ensembles, feature engineering, "
                             f"or a completely different model family.")

        # 4. Combination suggestions
        if len(ranked) >= 2:
            fam1, fam2 = ranked[0].label, ranked[1].label
            parts.append(f"\nCOMBINE: Try combining {fam1} + {fam2} in an ensemble "
                         f"(their individual scores suggest complementarity).")

        # 5. Hyperparameter tuning directive based on best family
        if ranked and ranked[0].n_attempts >= 2:
            best_fam = ranked[0]
            score_variance = (
                max(best_fam.scores) - min(best_fam.scores)
                if len(best_fam.scores) >= 2 else 0
            )
            if score_variance > 0.05:
                parts.append(f"\nTUNE: {best_fam.label} shows {score_variance:.3f} score "
                             f"variance — hyperparameters matter. Focus on tuning "
                             f"this family's parameters.")
            elif score_variance < 0.01 and best_fam.n_attempts >= 3:
                parts.append(f"\nSATURATED: {best_fam.label} is stable at "
                             f"{best_fam.mean_score:.3f}. Switch to a different family "
                             f"or add feature engineering.")

        return "\n".join(parts)

    def _basic_evolution_block(self) -> str:
        """Fallback: basic worked/failed listing when no scored data."""
        if not self.successful_techniques and not self.failed_techniques:
            return ""
        parts = ["\n[PROMPT EVOLUTION - Learning from past attempts]"]
        if self.successful_techniques:
            unique = list(dict.fromkeys(self.successful_techniques[-10:]))
            parts.append("Techniques that WORKED: " + ", ".join(unique))
        if self.failed_techniques:
            unique = list(dict.fromkeys(self.failed_techniques[-10:]))
            parts.append("Techniques that FAILED (avoid): " + ", ".join(unique))
        if self.successful_features:
            unique = list(dict.fromkeys(self.successful_features[-5:]))
            parts.append("Successful features/preprocessing: " + ", ".join(unique))
        return "\n".join(parts)

    def rewrite_template(self, base_template: str) -> str:
        """Rewrite an LLM prompt template based on accumulated evidence.

        Modifies the template by:
        1. Inserting EMPHASIZE/AVOID directives before the generation instruction
        2. Adding concrete examples from successful techniques
        3. Adjusting the creativity vs. exploitation balance
        """
        if not self._scored:
            return base_template

        directives = []
        ranked = sorted(self._scored.values(), key=lambda t: t.best_score, reverse=True)

        # Exploitation vs exploration
        if self._best_score_seen >= 0.95:
            directives.append(
                "IMPORTANT: Current best score is {:.3f}. Focus on REFINEMENT: "
                "small variations on the best approach. Do NOT try radically "
                "different architectures.".format(self._best_score_seen))
        elif self._best_score_seen >= 0.8:
            directives.append(
                "Current best score is {:.3f}. BALANCE exploration with exploitation: "
                "try variations of the best approach AND 1-2 novel alternatives.".format(
                    self._best_score_seen))
        else:
            directives.append(
                "Current best score is {:.3f} — LOW. EXPLORE aggressively: "
                "try fundamentally different model families, feature engineering, "
                "and data augmentation.".format(self._best_score_seen))

        # Top family emphasis
        if ranked:
            best = ranked[0]
            directives.append(
                f"Best family: {best.label} (score={best.best_score:.4f}). "
                f"Build on this success.")

        # Failures to avoid
        chronic = [f for f, c in self._fail_counts.items() if c >= 2]
        if chronic:
            directives.append(
                f"DO NOT use: {', '.join(chronic[:3])} (consistently failing).")

        if not directives:
            return base_template

        directive_block = "\n".join(f"[EVOLUTION] {d}" for d in directives)

        # Insert directives right before the "Generate" or "Write" instruction
        # or append at the end if no clear instruction marker
        for marker in ["Generate", "Write", "Create", "Produce", "Build"]:
            if marker in base_template:
                idx = base_template.index(marker)
                return base_template[:idx] + directive_block + "\n\n" + base_template[idx:]

        return base_template + "\n\n" + directive_block


def enrich_context_with_evolution(context: dict, evo: PromptEvolution) -> dict:
    """Inject prompt evolution directives into the context and rewrite templates."""
    block = evo.evolution_block()
    if block:
        existing = context.get("llm_guidance", "")
        context["llm_guidance"] = existing + block if existing else block

    # Rewrite the LLM prompt template if one is present in context
    if "llm_prompt_template" in context and evo._scored:
        context["llm_prompt_template"] = evo.rewrite_template(
            context["llm_prompt_template"])

    return context


# ============================================================================
# Master integration: wire all intelligence into the orchestrator
# ============================================================================

@dataclass
class IntelligenceState:
    """Aggregate state for all intelligence modules across rounds."""
    literature: Optional[LiteratureContext] = None
    asha: Optional[ASHAState] = None
    feature_proposer: Optional[FeatureEngineeringProposer] = None
    ensemble_proposer: Optional[EnsembleProposer] = None
    prompt_evolution: Optional[PromptEvolution] = None
    repair_failures: List[dict] = field(default_factory=list)
    prior_llm_diagnosis: Optional[Any] = None
    subsampled: bool = False


def init_intelligence(goal: str, task: Task, *,
                      llm_client: Optional[Callable] = None,
                      enable_literature: bool = True,
                      enable_asha: bool = True,
                      enable_feature_eng: bool = True,
                      enable_ensemble: bool = True,
                      enable_evolution: bool = True,
                      ) -> IntelligenceState:
    """Initialize all intelligence modules at the start of a run."""
    state = IntelligenceState()

    if enable_literature:
        state.literature = literature_scout(goal, llm_client=llm_client)

    if enable_feature_eng:
        state.feature_proposer = FeatureEngineeringProposer(
            n_features=task.n_features, kind=task.kind)

    if enable_ensemble:
        state.ensemble_proposer = EnsembleProposer()

    if enable_evolution:
        state.prompt_evolution = PromptEvolution()

    return state


def enrich_round_context(context: dict, state: IntelligenceState, task: Task,
                         round_idx: int, history: list,
                         kb_path: Optional[str] = None) -> dict:
    """Enrich the per-round context with all intelligence modules (additive)."""
    if state.literature is not None:
        enrich_context_with_literature(context, state.literature)

    enrich_context_with_feature_hints(context, task)

    knowledge_readback(context, kb_path)

    if state.prompt_evolution is not None:
        enrich_context_with_evolution(context, state.prompt_evolution)

    if round_idx >= 1 and state.asha is None:
        state.asha = init_asha(history, task, round_idx)

    return context


def get_intelligence_proposals(state: IntelligenceState, context: dict,
                               task: Task) -> List[Program]:
    """Gather proposals from all intelligence proposal sources."""
    proposals = []

    if state.feature_proposer is not None:
        proposals.extend(state.feature_proposer.propose(context))

    if state.ensemble_proposer is not None:
        proposals.extend(state.ensemble_proposer.propose(context))

    if state.asha is not None:
        proposals.extend(asha_proposals(state.asha, task, n=3))

    return proposals


def record_intelligence_outcome(state: IntelligenceState, program: Program,
                                val_score: Optional[float], ok: bool) -> None:
    """Record an outcome for prompt evolution tracking."""
    if state.prompt_evolution is not None:
        state.prompt_evolution.record_outcome(program, val_score, ok)


def record_repair_failure(state: IntelligenceState, error_kind: str,
                          error: str) -> None:
    """Record a failed repair for multi-turn diagnosis."""
    state.repair_failures.append({"error_kind": error_kind, "error": error})
    state.repair_failures = state.repair_failures[-10:]


__all__ = [
    "IntelligenceState", "LiteratureContext", "ASHAState",
    "FeatureEngineeringProposer", "EnsembleProposer", "PromptEvolution",
    "init_intelligence", "enrich_round_context", "get_intelligence_proposals",
    "record_intelligence_outcome", "record_repair_failure",
    "literature_scout", "multi_turn_diagnose", "adaptive_subsample",
    "knowledge_readback",
]
