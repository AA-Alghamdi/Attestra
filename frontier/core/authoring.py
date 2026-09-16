"""frontier.core.authoring -- the generative model-authoring engine (design 01).

This is THE CORE of the autoresearcher's *generative* half: instead of selecting a model
from a fixed catalog, an LLM AUTHORS arbitrary ``build_estimator()`` code conditioned on a
typed diagnosis plus (optional) retrieved literature, and those authored Programs flow into
the Phase-0 ``ResearchEngine`` as just another ``ProposalSource``. The deterministic recipe
library (SeedProposer + MutationProposer) is demoted to a FLOOR/FALLBACK -- it guarantees a
non-empty pool and an honest result when no client is wired, but it never out-ranks a
better-validating authored program and is never the promotion-bearing decision.

================================================================================
=== WIRING ===
================================================================================
How the Phase-0 integrator composes this module into the spine (NO Phase-0 edits):

    from frontier.engine import ResearchEngine, EngineConfig
    from frontier.proposers import SeedProposer, MutationProposer
    from frontier.core.authoring import CoreAuthoringProposer, AuthoringConfig

    client = resolve_backend(...)          # vfplatform; a Callable[[str],str]; or None
    proposers = [
        SeedProposer(),                                    # FLOOR: guaranteed non-empty
        MutationProposer(),                                # FALLBACK: offline generative
        CoreAuthoringProposer(client=client,               # CORE: arbitrary authored code
                              config=AuthoringConfig(n=3),
                              retriever=retriever),         # optional knowledge.py plugin
    ]
    engine = ResearchEngine(EngineConfig(rounds=4, llm_client=client), proposers=proposers)
    result = engine.run(task)

Contract points that make this drop-in with ZERO Phase-0 edits:

  1. ``CoreAuthoringProposer`` SUBCLASSES ``frontier.proposers.LLMProposer``. The engine's
     ``llm_active`` detection (engine.py:103) is
        ``any(isinstance(p, LLMProposer) and p.client is not None for p in self.proposers)``
     The subclass + the inherited ``.client`` attribute make that read True iff a client is
     wired -- honest, with no edit to engine.py.

  2. ``propose(context) -> list[Program]`` is the only method the engine calls on a source.
     We return ``Program(source="llm", code=<FULL authored model code>)`` objects. The engine
     then runs each in ``frontier.sandbox.run_program`` (out-of-process, predictions-only) and
     scores them on VAL via ``frontier.certify.score_val`` -- the trusted parent. This module
     NEVER runs candidate code and NEVER reports a metric.

  3. ``client=None`` => ``propose`` returns ``[]`` and the engine runs on the recipe floor;
     ``llm_active`` reads False. Honest degrade, no fabricated proposals.

  4. The three-stage STATIC FIREWALL runs here, BEFORE the sandbox, as defense in depth:
        autocorrect_names -> strip_forbidden -> validate(AST allow-list) -> add_preamble
     A program failing ``validate`` is rejected (logged in ``self.rejections``) and never
     reaches the sandbox. Even a slip-through still runs out-of-process under the sandbox's
     rlimits and returns only ``preds.npy`` -- the second integrity layer.

  5. Optional plugins are DUCK-TYPED and never hard-imported (they are authored in parallel:
     knowledge.py supplies a ``retriever`` with ``.retrieve(context) -> list[Motif]``;
     agentic.py supplies a ``repair`` with ``.repair(program, run_result, context, sandbox_run)``).
     Absent => that capability is skipped, never faked.

Design 01 splits firewall / schema into separate files; per the build assignment they are
INLINED here into a single additive module. The public API (AuthoringEngine,
CoreAuthoringProposer, AuthoringConfig, ProgramSpec, Firewall, Motif) is unchanged.
================================================================================
"""

from __future__ import annotations

import ast
import builtins as _builtins
import difflib
import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# Phase-0 imports (frozen contract). We import LLMProposer to SUBCLASS it so the engine's
# isinstance + .client llm_active check fires correctly with zero engine.py edits.
from ..program import Program
from ..proposers import LLMProposer


# A client is any callable: prompt(str) -> completion(str). It is the ONLY external
# dependence. Wire vfplatform.resolve_backend / a Prime Intellect client behind this type.
# None => honest degrade: propose() returns [] and the engine runs on the recipe floor.
LLMClient = Callable[[str], str]


# =============================================================================
# Motif (literature grounding) -- the contract a retriever (knowledge.py) returns.
# Defined here so this module has no hard dependency on the parallel knowledge module.
# =============================================================================
@dataclass
class Motif:
    """A retrieved literature finding used to GROUND generation (never executed verbatim).

    A retriever (the separate ``knowledge.py`` module, duck-typed) returns a list of these
    from ``retrieve(context)``. ``skeleton`` is an OPTIONAL firewall-passing code fragment
    (innovation 7.3): a vetted, fillable pattern tied to a citable claim. It is validated
    before being offered so a poisoned skeleton cannot widen the safety surface.
    """
    id: str                          # e.g. "arxiv:2106.11959" or "skl:hist_gbm_tabular"
    title: str = ""
    claim: str = ""                  # one-line takeaway
    skeleton: Optional[str] = None   # optional retrieval-grounded code skeleton
    score: float = 0.0               # retrieval relevance


# =============================================================================
# ProgramSpec -- innovation 7.1: typed, spec-conditioned program synthesis.
# The engine first synthesizes a typed spec from the live diagnosis, then conditions code
# generation on it. The spec is a checkable post-condition (spec.check flags violations,
# feeding the revise loop) AND a stable target the knowledge ranker can featurize.
# It is REMOVABLE SCAFFOLDING (a heuristic shaping the prompt), not a search bound.
# =============================================================================
@dataclass
class ProgramSpec:
    """A typed contract derived from diagnosis that the authored code SHOULD satisfy.

    ``must_handle`` / ``encourage`` / ``forbid`` are rendered into prompt directives
    (``to_directives``) and post-checked against authored code (``check``). None of these
    bound the search (the model may ignore them) or promote anything; they are scaffolding.
    """
    task_kind: str
    must_handle: List[str] = field(default_factory=list)   # e.g. ["scale_sensitive_model"]
    encourage: List[str] = field(default_factory=list)     # e.g. ["nonlinear_model"]
    forbid: List[str] = field(default_factory=list)        # e.g. ["high_degree_poly"]
    budget: dict = field(default_factory=dict)             # {"max_features_after_eng": 5000}

    # Human-readable phrasing for each typed tag. Kept as comments-in-data so the prompt
    # text and the AST check share one source of truth.
    _ENCOURAGE_TEXT = {
        "nonlinear_model": "probe nonlinearity (gradient boosting, or polynomial/spline "
                           "features into a linear model)",
        "feature_selection": "add SelectKBest or PCA before the model",
        "feature_engineering": "emphasize feature engineering (poly, spline, interactions)",
        "target_transform": "consider a target transform (log1p / PowerTransformer / quantile)",
        "structurally_different": "use a structurally DIFFERENT model class from the champion",
        "regularize": "add or increase regularization",
        "stacking": "stack 2-3 diverse base estimators with a simple meta-learner",
    }
    _MUST_TEXT = {
        "scale_sensitive_model": "scale features for any distance/gradient-sensitive model",
        "skewed_target": "handle a skewed target (a target transform is appropriate)",
        "high_dim_few_rows": "guard against overfitting in high dimensions with few rows "
                             "(regularize, select features, avoid degree>2 polynomials)",
    }
    _FORBID_TEXT = {
        "high_degree_poly": "do NOT use PolynomialFeatures with degree > 2 (a prior pipeline "
                            "was numerically unstable)",
        "svc_on_large_n": "do NOT use SVC on large n (it was too slow under the CPU limit)",
    }

    def to_directives(self) -> str:
        """Render the typed spec to the prompt's allowed-construct / constraint menu."""
        lines: List[str] = []
        if self.must_handle:
            reqs = "; ".join(self._MUST_TEXT.get(t, t) for t in self.must_handle)
            lines.append(f"REQUIRED: {reqs}.")
        if self.encourage:
            enc = "; ".join(self._ENCOURAGE_TEXT.get(t, t) for t in self.encourage)
            lines.append(f"ENCOURAGED: {enc}.")
        if self.forbid:
            fbd = "; ".join(self._FORBID_TEXT.get(t, t) for t in self.forbid)
            lines.append(f"FORBIDDEN BY DIAGNOSIS: {fbd}.")
        if self.budget:
            lines.append(f"BUDGET: {self.budget}.")
        # The reachable construct menu (section 2.4): tells the model the full space.
        lines.append(
            "YOU MAY USE ANY OF: feature_engineering{PolynomialFeatures, SplineTransformer, "
            "PCA, SelectKBest, interactions}, target_transform{TransformedTargetRegressor with "
            "log1p / PowerTransformer / QuantileTransformer}, stacking{StackingClassifier/"
            "StackingRegressor over 2-3 diverse bases}, calibration{CalibratedClassifierCV}. "
            "Neural nets via sklearn.neural_network.MLP* are allowed; torch is NOT yet available."
        )
        return "\n".join(lines)

    def check(self, program_code: str) -> List[str]:
        """Post-hoc AST check: which typed constraints did the authored code VIOLATE?

        Used as a VAL-only signal feeding the revise loop and the knowledge ranker. It never
        rejects a program by itself (the firewall ``validate`` does the hard gating); it only
        surfaces spec violations for diagnosis. Returns a list of violation tags.
        """
        violations: List[str] = []
        try:
            tree = ast.parse(program_code)
        except SyntaxError:
            return ["unparseable"]
        # Collect every constructor call name + its keywords for cheap structural checks.
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        names = {c.func.id for c in calls if isinstance(c.func, ast.Name)}
        names |= {c.func.attr for c in calls if isinstance(c.func, ast.Attribute)}

        if "high_degree_poly" in self.forbid:
            for c in calls:
                fn = c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", "")
                if fn == "PolynomialFeatures":
                    for kw in c.keywords:
                        if kw.arg == "degree" and isinstance(kw.value, ast.Constant) \
                                and isinstance(kw.value.value, (int, float)) and kw.value.value > 2:
                            violations.append("high_degree_poly")
        if "svc_on_large_n" in self.forbid and "SVC" in names:
            violations.append("svc_on_large_n")
        if "target_transform" in self.encourage and "TransformedTargetRegressor" not in names:
            # encourage-violations are soft; recorded but lower-signal than forbid-violations.
            violations.append("missing_target_transform")
        if "scale_sensitive_model" in self.must_handle:
            scalers = {"StandardScaler", "MinMaxScaler", "RobustScaler", "QuantileTransformer",
                       "PowerTransformer", "Normalizer"}
            sensitive = {"SVC", "SVR", "KNeighborsClassifier", "KNeighborsRegressor",
                         "LogisticRegression", "Ridge", "Lasso", "ElasticNet", "MLPClassifier",
                         "MLPRegressor"}
            if names & sensitive and not (names & scalers):
                violations.append("scale_sensitive_unscaled")
        return violations

    def to_dict(self) -> dict:
        return {"task_kind": self.task_kind, "must_handle": list(self.must_handle),
                "encourage": list(self.encourage), "forbid": list(self.forbid),
                "budget": dict(self.budget)}


# =============================================================================
# Firewall -- static, pre-execution gate (design section 3). INLINED from firewall.py.
# Order (mirrors AuthoringEngine._materialize):
#   autocorrect_names -> strip_forbidden -> validate(AST allow-list) -> add_preamble
# Defense in depth, NOT a replacement for the out-of-process sandbox.
# =============================================================================

# Imports the authored code may use. torch added when that substrate lands (design 02/03).
ALLOWED_IMPORT_ROOTS = {"numpy", "sklearn"}

# AST node allow-list: everything else => reject. Covers function defs, sklearn pipeline
# construction, arithmetic for feature lists, comprehensions, lambdas (FunctionTransformer).
_ALLOWED_NODE_TYPES = {
    ast.Module, ast.FunctionDef, ast.Return, ast.Assign, ast.AnnAssign, ast.Expr,
    ast.Call, ast.Attribute, ast.Name, ast.Load, ast.Store, ast.Del, ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.keyword, ast.arguments, ast.arg,
    ast.Import, ast.ImportFrom, ast.alias, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Subscript, ast.Slice, ast.Lambda, ast.IfExp, ast.Compare, ast.BoolOp,
    ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp, ast.comprehension,
    ast.Starred,
}
# ast.Index exists only on <3.9; guard so the set is valid on 3.11.
if hasattr(ast, "Index"):
    _ALLOWED_NODE_TYPES.add(ast.Index)

# Calls that are dangerous regardless of context => hard reject.
FORBIDDEN_CALL_NAMES = {"eval", "exec", "open", "compile", "__import__", "input",
                        "globals", "locals", "getattr", "setattr", "delattr", "vars",
                        "exit", "quit", "memoryview"}
# Attribute access that signals IO / network / process control => hard reject.
FORBIDDEN_ATTR = {"system", "popen", "fork", "remove", "unlink", "rmtree", "spawn",
                  "urlopen", "request", "Request", "urlretrieve", "Socket", "socket",
                  "connect", "sendall", "recv", "Popen", "call", "check_output", "run"}
# Search wrappers that could peek at the eval split if misused. Conservatively rejected by
# default (allow_internal_cv=False); the eval split is structurally unreachable, so this is
# a provably-safe simplest rule, with a documented relaxation toggle.
LEAKAGE_CALLS = {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV",
                 "HalvingRandomSearchCV", "cross_val_score", "cross_validate",
                 "cross_val_predict", "learning_curve", "validation_curve"}

_PYTHON_BUILTINS = set(dir(_builtins)) | {"np", "build_estimator"}

# The import preamble (design 3.2): guarantees common sklearn HELPER symbols are importable
# so a forgotten import does not cause an avoidable NameError. It imports only allow-listed
# roots, so it cannot widen the safety surface. Estimators are intentionally NOT blanket-
# imported here -- autocorrect_names resolves those, giving a clean provenance trail.
PREAMBLE = (
    "import numpy as np\n"
    "from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion\n"
    "from sklearn.compose import ColumnTransformer, TransformedTargetRegressor\n"
    "from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,\n"
    "    PolynomialFeatures, PowerTransformer, QuantileTransformer, FunctionTransformer,\n"
    "    SplineTransformer, OneHotEncoder, Normalizer)\n"
    "from sklearn.feature_selection import (SelectKBest, f_classif, f_regression,\n"
    "    mutual_info_classif, mutual_info_regression, VarianceThreshold)\n"
    "from sklearn.decomposition import PCA, TruncatedSVD\n"
)


def _build_symbol_registry() -> dict:
    """Map every valid sklearn public estimator name -> its ``from ... import Name`` line.

    Built once by walking ``sklearn.utils.all_estimators()`` (the canonical public registry)
    plus a curated map of helper transformers/composition objects the preamble already covers.
    Used by ``autocorrect_names`` to (a) supply a missing import for a valid symbol and
    (b) rename an obvious typo to the nearest valid symbol. The registry NEVER invents an
    estimator the model did not reference.
    """
    registry: dict = {}
    try:
        from sklearn.utils import all_estimators
        for name, cls in all_estimators():
            mod = cls.__module__
            # Collapse private submodule paths to the public package (sklearn.x).
            parts = mod.split(".")
            public = ".".join(parts[:2]) if len(parts) >= 2 else mod
            registry[name] = f"from {public} import {name}"
    except Exception:
        # If sklearn is somehow unavailable, autocorrect simply does nothing (validate still
        # gates). We never fabricate a registry.
        pass
    # Curated helpers (composition + target transform + metrics-free utilities). These map to
    # stable public paths and round out names all_estimators() may not surface.
    curated = {
        "Pipeline": "from sklearn.pipeline import Pipeline",
        "make_pipeline": "from sklearn.pipeline import make_pipeline",
        "FeatureUnion": "from sklearn.pipeline import FeatureUnion",
        "ColumnTransformer": "from sklearn.compose import ColumnTransformer",
        "TransformedTargetRegressor": "from sklearn.compose import TransformedTargetRegressor",
        "f_classif": "from sklearn.feature_selection import f_classif",
        "f_regression": "from sklearn.feature_selection import f_regression",
        "mutual_info_classif": "from sklearn.feature_selection import mutual_info_classif",
        "mutual_info_regression": "from sklearn.feature_selection import mutual_info_regression",
    }
    for k, v in curated.items():
        registry.setdefault(k, v)
    return registry


_REGISTRY = _build_symbol_registry()


def strip_fences(raw: str) -> str:
    """Remove markdown code fences and any prose preamble/trailer around the code block.

    LLMs habitually wrap code in ```python ... ``` or add commentary. We extract the largest
    fenced block if present; otherwise we return the raw text trimmed. A leading ``# motif:``
    line (the model citing a finding) is preserved by the caller before this runs.
    """
    if raw is None:
        return ""
    text = raw.strip()
    # Prefer an explicitly fenced block.
    fence = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fence:
        # take the longest fenced block (most likely the actual code)
        return max(fence, key=len).strip() + "\n"
    # No fences: drop any stray leading lines before the first import/def, conservatively.
    return text + "\n"


def _collect_imported_names(tree: ast.AST) -> set:
    """Names made available by import statements in the tree (for autocorrect bookkeeping)."""
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.asname or a.name)
    return names


def _rename_symbol(code: str, old: str, new: str) -> str:
    """Whole-word rename of a referenced symbol (typo fix). Word-boundary regex avoids
    clobbering substrings (e.g. renaming ``Ridge`` must not touch ``RidgeCV``)."""
    return re.sub(rf"\b{re.escape(old)}\b", new, code)


# ---- task-type validation helpers --------------------------------------------------------
# Regressor names that must NOT appear as the primary estimator for classification tasks.
_REGRESSOR_ONLY = frozenset({
    "LinearRegression", "Ridge", "RidgeCV", "Lasso", "LassoCV", "ElasticNet", "ElasticNetCV",
    "Lars", "LarsCV", "LassoLars", "LassoLarsCV", "BayesianRidge", "ARDRegression",
    "HuberRegressor", "TheilSenRegressor", "RANSACRegressor",
    "SVR", "NuSVR", "LinearSVR",
    "KNeighborsRegressor", "RadiusNeighborsRegressor",
    "DecisionTreeRegressor", "ExtraTreeRegressor",
    "RandomForestRegressor", "ExtraTreesRegressor",
    "GradientBoostingRegressor", "HistGradientBoostingRegressor",
    "AdaBoostRegressor", "BaggingRegressor",
    "MLPRegressor",
    "GaussianProcessRegressor",
    "PLSRegression", "PLSCanonical",
    "StackingRegressor", "VotingRegressor",
    "TransformedTargetRegressor",
    "KernelRidge",
})

_CLASSIFIER_ONLY = frozenset({
    "LogisticRegression", "LogisticRegressionCV",
    "SVC", "NuSVC", "LinearSVC",
    "KNeighborsClassifier", "RadiusNeighborsClassifier",
    "DecisionTreeClassifier", "ExtraTreeClassifier",
    "RandomForestClassifier", "ExtraTreesClassifier",
    "GradientBoostingClassifier", "HistGradientBoostingClassifier",
    "AdaBoostClassifier", "BaggingClassifier",
    "MLPClassifier",
    "GaussianProcessClassifier",
    "GaussianNB", "MultinomialNB", "ComplementNB", "BernoulliNB", "CategoricalNB",
    "StackingClassifier", "VotingClassifier",
    "CalibratedClassifierCV",
    "SGDClassifier", "Perceptron", "PassiveAggressiveClassifier",
    "RidgeClassifier", "RidgeClassifierCV",
    "LabelPropagation", "LabelSpreading",
    "QuadraticDiscriminantAnalysis", "LinearDiscriminantAnalysis",
})

# Estimators that work for both classification and regression (never a mismatch)
# SGDRegressor, PassiveAggressiveRegressor, etc. are regressor-only and already in _REGRESSOR_ONLY


def _check_task_type_match(names: set, kind: str) -> str:
    """Return a rejection reason if the primary estimator type mismatches the task kind.

    A classification task using regressors (or vice versa) is rejected. Pipeline/
    ColumnTransformer wrappers are ignored — only the terminal estimator matters. Returns
    empty string if no mismatch (i.e., OK).
    """
    if kind == "classification":
        # Reject if code uses regressor-only estimators and NO classifier
        regressors_used = names & _REGRESSOR_ONLY
        classifiers_used = names & _CLASSIFIER_ONLY
        if regressors_used and not classifiers_used:
            sample = sorted(regressors_used)[:3]
            return f"task-type mismatch: classification task but code uses regressor(s): {', '.join(sample)}"
    elif kind == "regression":
        classifiers_used = names & _CLASSIFIER_ONLY
        regressors_used = names & _REGRESSOR_ONLY
        if classifiers_used and not regressors_used:
            sample = sorted(classifiers_used)[:3]
            return f"task-type mismatch: regression task but code uses classifier(s): {', '.join(sample)}"
    return ""


@dataclass
class Firewall:
    """Static, pre-execution allow-list firewall (defense in depth before the sandbox).

    Stateful only for audit: ``last_corrections`` / ``last_strips`` record exactly what the
    firewall changed on the most recent ``autocorrect_names`` / ``strip_forbidden`` call, and
    those land in ``Program.provenance`` so a reviewer can see every modification.
    """
    allow_internal_cv: bool = False           # ship False; True is a flagged spec relaxation
    preamble_includes_estimators: bool = False
    last_corrections: List[str] = field(default_factory=list)
    last_strips: List[str] = field(default_factory=list)

    # ---- stage 1: hallucinated-name auto-correction (design 3.3) -------------------------
    def autocorrect_names(self, code: str) -> str:
        """Resolve missing imports for valid symbols and rename obvious typos.

        (a) A referenced valid sklearn symbol with no import -> prepend the exact import.
        (b) A near-miss spelling (cutoff 0.85) -> rename to the nearest valid symbol + import.
        Anything ambiguous is left to fail loudly and inform the revise loop. All changes are
        recorded in ``self.last_corrections``. Never invents an unrequested estimator.
        """
        self.last_corrections = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code  # validate() will reject; nothing to correct on unparseable code
        referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        imported = _collect_imported_names(tree)
        header: List[str] = []
        for name in sorted(referenced - imported - _PYTHON_BUILTINS):
            if name in _REGISTRY:                       # valid symbol, just missing import
                header.append(_REGISTRY[name])
                self.last_corrections.append(f"import+:{name}")
            else:
                cand = difflib.get_close_matches(name, list(_REGISTRY), n=1, cutoff=0.85)
                if cand:                                # near-miss typo -> rename + import
                    code = _rename_symbol(code, name, cand[0])
                    header.append(_REGISTRY[cand[0]])
                    self.last_corrections.append(f"rename:{name}->{cand[0]}")
                # else: leave it; validate/sandbox surface a clean error for the revise loop
        if header:
            # dedup imports while preserving order
            seen, uniq = set(), []
            for h in header:
                if h not in seen:
                    seen.add(h)
                    uniq.append(h)
            code = "\n".join(uniq) + "\n" + code
        return code

    # ---- stage 2: strip mechanical hazards (design 3.4) ----------------------------------
    def strip_forbidden(self, code: str) -> str:
        """Rewrite the few mechanical hazards (higher yield than rejecting); log each.

        - n_jobs=<anything> -> n_jobs=1   (RLIMIT_CPU correctness; parallel backends fork)
        - verbose=<anything> -> dropped   (stdout pollution corrupts the sandbox status line)
        Anything genuinely dangerous (os import, open(), network attr, leakage wrapper) is NOT
        rewritten here -- ``validate`` is the hard gate so we never silently 'fix' a program
        into something the model did not author.
        """
        self.last_strips = []
        new = re.sub(r"n_jobs\s*=\s*-?\d+", "n_jobs=1", code)
        if new != code:
            self.last_strips.append("n_jobs->1")
            code = new
        # drop verbose=... keyword (handles verbose=True / 1 / 2). Leaves a clean comma list.
        new = re.sub(r",\s*verbose\s*=\s*[^,)\n]+", "", code)
        new = re.sub(r"verbose\s*=\s*[^,)\n]+\s*,\s*", "", new)
        if new != code:
            self.last_strips.append("verbose-dropped")
            code = new
        return code

    # ---- stage 3: AST allow-list validation (design 3.1) ---------------------------------
    def validate(self, code: str, *, kind: str) -> Tuple[bool, str]:
        """Hard gate: parse, require exactly the build_estimator() shape, walk every node.

        Returns (ok, reason). A False here means the program is rejected BEFORE the sandbox.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"syntax: {e}"
        # 1. a build_estimator def must exist
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not any(f.name == "build_estimator" for f in funcs):
            return False, "no build_estimator()"
        # 2. only imports + function defs at module level (no module-level fit/side effects)
        for stmt in tree.body:
            if not isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
                return False, f"module-level statement not allowed: {type(stmt).__name__}"
        # 3. walk every node: type allow-list, import roots, forbidden calls/attrs, leakage
        for node in ast.walk(tree):
            if type(node) not in _ALLOWED_NODE_TYPES:
                return False, f"forbidden node {type(node).__name__}"
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    return False, f"forbidden import {node.module}"
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                        return False, f"forbidden import {a.name}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALL_NAMES:
                    return False, f"forbidden call {node.func.id}"
                if node.func.id in LEAKAGE_CALLS and not self.allow_internal_cv:
                    return False, f"leakage construct {node.func.id}"
            if isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_ATTR:
                    return False, f"forbidden attribute .{node.attr}"
                if node.attr in LEAKAGE_CALLS and not self.allow_internal_cv:
                    return False, f"leakage construct .{node.attr}"
        # 4. task-type match: reject regressors for classification and vice versa
        all_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        all_names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        mismatch = _check_task_type_match(all_names, kind)
        if mismatch:
            return False, mismatch
        return True, "ok"

    # ---- stage 4: symbol-guaranteeing preamble (design 3.2) ------------------------------
    def add_preamble(self, code: str) -> str:
        """Prepend the fixed helper-import header (added ONLY after validate passed).

        Re-import is idempotent in Python, so this never overrides an import the model wrote.
        Imports only allow-listed roots, so it cannot widen the safety surface.
        """
        return PREAMBLE + "\n" + code


# =============================================================================
# AuthoringConfig + AuthoringEngine (design section 1.3) + CoreAuthoringProposer (1.4)
# =============================================================================
@dataclass
class AuthoringConfig:
    n: int = 3                                # candidates authored per round (the "N")
    temperatures: Tuple[float, ...] = (0.2, 0.6, 0.9)   # one per candidate; recycled if n>len
    max_completion_chars: int = 12000
    inject_literature: bool = True
    inject_diagnosis: bool = True
    enable_revise: bool = True
    revise_max_attempts: int = 1
    schema_mode: str = "typed"                # "typed" (ProgramSpec directives) | "freeform"


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# The frozen system contract for build_estimator() (design 2.1). Sent verbatim every call.
SYSTEM_CONTRACT = (
    "You author one scikit-learn pipeline as Python code. Output rules:\n"
    "- Define exactly one function: def build_estimator(): returning a fresh, UNFITTED\n"
    "  sklearn-compatible estimator (any object with .fit(X, y) and .predict(X)).\n"
    "- The estimator is fit on a training split and asked to predict an evaluation split by\n"
    "  the caller. You do NOT fit, score, print, read files, or access the network. You do\n"
    "  NOT see y beyond what fit() receives.\n"
    "- You MAY compose: ColumnTransformer, Pipeline, FunctionTransformer, StandardScaler,\n"
    "  PolynomialFeatures, PCA, SelectKBest, QuantileTransformer, PowerTransformer, target\n"
    "  transforms via TransformedTargetRegressor, and stacking via StackingClassifier/Regressor.\n"
    "- Allowed imports: numpy as np, and sklearn.* only. No other top-level packages.\n"
    "- FORBIDDEN: any GridSearchCV/RandomizedSearchCV/cross_val_* ; any import of os, sys,\n"
    "  subprocess, socket, requests, urllib; open(); pickle of external files; eval/exec.\n"
    "- For regression you MAY wrap the pipeline in TransformedTargetRegressor.\n"
    "- Set random_state=0 wherever an estimator accepts it. Set n_jobs=1.\n"
    "- Return ONLY the code. No markdown fences, no commentary.\n"
)

# Per-candidate steering lines so the N candidates are genuinely different hypotheses
# (design 2.5), not N near-duplicates from sampling noise alone.
_VARIANT_DIRECTIVES = {
    0: "Author the single pipeline you are most confident certifies above the threshold.",
    1: "Author a structurally DIFFERENT model class from candidate 0 and from the champion.",
    2: "Author a pipeline emphasizing feature engineering / target transform over a model swap.",
}


class AuthoringEngine:
    """Stateless-per-call author: round context -> validated, firewalled Program objects.

    It owns the prompt, the diversity strategy, parsing, and the firewall handoff. It does
    NOT run code (the sandbox does, invoked by ResearchEngine) and does NOT score (that is
    ``certify.score_val``). All authored Programs carry ``source="llm"``.
    """

    SYSTEM = SYSTEM_CONTRACT

    def __init__(self,
                 client: Optional[LLMClient] = None,
                 config: Optional[AuthoringConfig] = None,
                 retriever=None,            # duck-typed: .retrieve(context) -> list[Motif]
                 firewall: Optional[Firewall] = None,
                 repair=None):              # duck-typed RepairHandoff (agentic.py); optional
        self.client = client
        self.cfg = config or AuthoringConfig()
        self.retriever = retriever
        self.fw = firewall or Firewall()
        self.repair = repair
        self.rejections: List[dict] = []   # audit trail: every rejected candidate + reason

    # ---- the one method a proposer needs -------------------------------------------------
    def author(self, context: dict) -> List[Program]:
        """Author up to cfg.n firewalled Programs for this round. [] if no client (honest degrade)."""
        if self.client is None:
            return []
        spec = self._build_spec(context)
        motifs = self._retrieve(context) if self.cfg.inject_literature else []
        programs: List[Program] = []
        seen: set = set()
        for i in range(self.cfg.n):
            temp = self.cfg.temperatures[i % len(self.cfg.temperatures)]
            ctx_i = dict(context)
            ctx_i["_temp"] = temp
            prompt = self._build_prompt(ctx_i, spec, motifs, variant=i)
            raw = self._call(prompt, temperature=temp)
            if raw is None:
                continue
            prog = self._materialize(raw, ctx_i, spec, motifs, prompt)
            if prog is None or prog.id in seen:
                continue
            seen.add(prog.id)
            programs.append(prog)
        return programs

    # ---- innovation 7.1: diagnosis -> typed ProgramSpec ----------------------------------
    def _build_spec(self, context: dict) -> ProgramSpec:
        """Synthesize a typed ProgramSpec from the live diagnosis (NOT from any answer key).

        Every directive below is a HEURISTIC that shapes the prompt -- labeled as such. It
        never bounds the search (the model may ignore it) and never promotes anything. These
        are removable scaffolding as base models improve (consistent with 'powered by LLMs').
        """
        kind = context.get("task_kind", "classification")
        spec = ProgramSpec(task_kind=kind)
        if not self.cfg.inject_diagnosis:
            return spec
        recent = context.get("recent_errors", []) or []
        best_label = (context.get("best_label") or "").lower()

        # plateau (Phase-2 diagnosis key; absent -> 0). Stalled linear gains -> structural change.
        plateau = int(context.get("plateau_rounds", 0) or 0)
        if plateau >= 2:
            spec.encourage.append("structurally_different")

        # linear champion on a regression task -> probe nonlinearity / feature engineering.
        if kind == "regression" and any(t in best_label for t in ("ridge", "lasso", "linear")):
            spec.encourage.append("nonlinear_model")
            spec.encourage.append("feature_engineering")

        # error-kind-driven constraints (mirrors the design directive table).
        for entry in recent:
            ek, msg = "", ""
            if isinstance(entry, (tuple, list)) and len(entry) >= 3:
                _, ek, msg = entry[0], entry[1], entry[2]
            elif isinstance(entry, dict):
                ek, msg = entry.get("error_kind", ""), entry.get("error", "")
            ek = (ek or "").lower()
            msg = (msg or "")
            if ek == "fit" and "LinAlg" in msg:
                spec.forbid.append("high_degree_poly")
                spec.encourage.append("regularize")
            if ek in ("timeout", "cpu"):
                spec.forbid.append("svc_on_large_n")
            if ek == "oom":
                spec.encourage.append("feature_selection")

        # high-dim / few-rows guard.
        nf, ntr = context.get("n_features"), context.get("n_train")
        if nf and ntr and nf >= 50 and ntr <= 5 * nf:
            spec.must_handle.append("high_dim_few_rows")
            spec.encourage.append("feature_selection")

        # residual structure hint (Phase-2 diagnosis key); skewed target -> target transform.
        if context.get("residual_note") or context.get("skewed_target"):
            spec.must_handle.append("skewed_target")
            spec.encourage.append("target_transform")

        # any scale-sensitive base in play -> require scaling.
        if any(t in best_label for t in ("svc", "logreg", "knn", "ridge", "lasso", "mlp")):
            spec.must_handle.append("scale_sensitive_model")

        # dedup while preserving order
        def _dedup(xs):
            seen, out = set(), []
            for x in xs:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out
        spec.must_handle = _dedup(spec.must_handle)
        spec.encourage = _dedup(spec.encourage)
        spec.forbid = _dedup(spec.forbid)
        return spec

    def _retrieve(self, context: dict) -> List[Motif]:
        """Pull literature motifs from the (optional, duck-typed) retriever. [] if absent."""
        if self.retriever is None:
            return []
        try:
            motifs = self.retriever.retrieve(context) or []
        except Exception:
            return []                       # a broken retriever degrades to no grounding
        out = []
        for m in motifs:
            # a skeleton, if present, must itself pass validate before being offered (7.3).
            if getattr(m, "skeleton", None):
                ok, _ = self.fw.validate(m.skeleton, kind=context.get("task_kind", "classification"))
                if not ok:
                    m = Motif(id=m.id, title=getattr(m, "title", ""),
                              claim=getattr(m, "claim", ""), skeleton=None,
                              score=getattr(m, "score", 0.0))
            out.append(m)
        return out

    # ---- prompt assembly (design section 2) ----------------------------------------------
    def _render_diagnosis(self, context: dict) -> str:
        """Render the numeric diagnosis block (degrades gracefully on absent keys)."""
        lines = [
            f"TASK: {context.get('task_kind')}, {context.get('n_features')} features, "
            f"{context.get('n_train')} train rows."
        ]
        if context.get("best_score") is not None:
            lines.append(f"CHAMPION: {context.get('best_label')} validating at "
                         f"{context.get('best_score')}.")
        fam = context.get("family_ranking")
        if fam:
            try:
                fam_txt = "   ".join(f"{k}: {v}" for k, v in list(fam.items())[:6])
                lines.append(f"PER-FAMILY (val score): {fam_txt}")
            except Exception:
                pass
        if context.get("plateau_rounds"):
            lines.append(f"PLATEAU: best score has not improved in "
                         f"{context.get('plateau_rounds')} rounds.")
        recent = context.get("recent_errors", []) or []
        if recent:
            ferr = []
            for entry in recent[:5]:
                if isinstance(entry, (tuple, list)) and len(entry) >= 3:
                    ferr.append(f"  - {entry[0]}: [{entry[1]}] {entry[2]}")
                elif isinstance(entry, dict):
                    ferr.append(f"  - {entry.get('label')}: [{entry.get('error_kind')}] "
                                f"{entry.get('error')}")
            if ferr:
                lines.append("RECENT FAILURES (do not repeat these error modes):")
                lines.extend(ferr)
        if context.get("residual_note"):
            lines.append(f"RESIDUAL HINT: {context.get('residual_note')}")
        # LLM-powered diagnosis: deeper analysis from llm_diagnosis.py
        if context.get("llm_guidance"):
            lines.append(f"\nLLM ANALYSIS:\n{context['llm_guidance']}")
        confusion = context.get("confusion")
        if confusion:
            weak = confusion.get("weakest_class")
            pairs = confusion.get("confused_pairs", [])
            if weak is not None:
                lines.append(f"WEAKEST CLASS: '{weak}' (target for improvement)")
            if pairs:
                pair_txt = "; ".join(
                    f"'{p['true']}'->'{p['pred']}' ({p['count']}x)"
                    for p in pairs[:3]
                )
                lines.append(f"MOST CONFUSED: {pair_txt}")
        return "\n".join(lines)

    def _render_motifs(self, motifs: List[Motif]) -> str:
        if not motifs:
            return ""
        lines = ["RELEVANT FINDINGS (use as evidence, adapt; cite the id you used in a "
                 "leading '# motif: <id>' comment):"]
        for m in motifs[:5]:
            lines.append(f"  [{m.id}] {m.claim or m.title}")
            if getattr(m, "skeleton", None):
                lines.append(f"  known-good pattern for [{m.id}] (adapt, do not copy blindly):")
                lines.append("  " + m.skeleton.replace("\n", "\n  "))
        return "\n".join(lines)

    def _build_prompt(self, context: dict, spec: ProgramSpec, motifs: List[Motif],
                      variant: int) -> str:
        """Assemble system contract + diagnosis + typed spec directives + motifs + variant."""
        parts = [self.SYSTEM]
        if self.cfg.inject_diagnosis:
            parts.append(self._render_diagnosis(context))
        if self.cfg.schema_mode == "typed":
            parts.append(spec.to_directives())
        motif_block = self._render_motifs(motifs)
        if motif_block:
            parts.append(motif_block)
        steer = _VARIANT_DIRECTIVES.get(
            variant,
            "Author a pipeline that combines the strengths of strong prior candidates "
            "(e.g. a stacking ensemble).")
        parts.append(f"THIS CANDIDATE: {steer}")
        return "\n\n".join(parts)

    # ---- LLM call (adapts a bare Callable or a ChatLLMClient) -----------------------------
    def _call(self, prompt: str, temperature: float = 0.4) -> Optional[str]:
        """Call the client; return raw text or None on any failure (honest, never fabricated).

        Accepts either a bare ``Callable[[str], str]`` or a chat client exposing
        ``complete(messages)``. Temperature is passed only if the client accepts it; a plain
        callable that takes one positional arg is called with just the prompt.
        """
        if self.client is None:
            return None
        try:
            client = self.client
            if hasattr(client, "complete"):
                # ChatLLMClient: single-turn here; revise() uses multi-turn.
                return client.complete([{"role": "system", "content": self.SYSTEM},
                                        {"role": "user", "content": prompt}])
            # bare callable: try (prompt, temperature) then fall back to (prompt).
            try:
                out = client(prompt, temperature)        # type: ignore[call-arg]
            except TypeError:
                out = client(prompt)
            return out
        except Exception:
            return None

    # ---- the firewall pipeline (design 1.3) ----------------------------------------------
    def _materialize(self, raw: str, context: dict, spec: Optional[ProgramSpec],
                     motifs: List[Motif], prompt: str) -> Optional[Program]:
        """raw completion -> firewalled Program (or None, with a logged rejection).

        Pipeline order is load-bearing:
            strip_fences -> autocorrect_names -> strip_forbidden -> validate -> add_preamble
        """
        # preserve a leading '# motif: <id>' citation before stripping fences/prose.
        motif_cite = None
        m = re.match(r"\s*#\s*motif:\s*(\S+)", raw or "")
        if m:
            motif_cite = m.group(1)
        code = strip_fences(raw)
        code = self.fw.autocorrect_names(code)
        corrections = list(self.fw.last_corrections)
        code = self.fw.strip_forbidden(code)
        strips = list(self.fw.last_strips)
        kind = context.get("task_kind", "classification")
        ok, reason = self.fw.validate(code, kind=kind)
        if not ok:
            self._reject(reason, code)
            return None
        code = self.fw.add_preamble(code)
        spec_violations = spec.check(code) if spec is not None else []
        label = self._label(context, spec)
        return Program(
            code=code,
            source="llm",
            label=label,
            parent_id=None,
            provenance={
                "prompt_sha": sha12(prompt),
                "spec": spec.to_dict() if spec is not None else None,
                "spec_violations": spec_violations,
                "motif_ids": [getattr(mm, "id", "") for mm in motifs],
                "motif_cite": motif_cite,
                "temperature": context.get("_temp"),
                "authored_round": context.get("round"),
                "corrections": corrections,
                "strips": strips,
            },
        )

    def _reject(self, reason: str, code: str) -> None:
        """Record a rejected candidate (never silently dropped) for the audit trail."""
        self.rejections.append({"reason": reason, "code_sha": sha12(code),
                                "code_head": code[:200]})

    def _label(self, context: dict, spec: Optional[ProgramSpec]) -> str:
        r = context.get("round", 0)
        t = context.get("_temp")
        tt = f"t{int(round(float(t) * 10))}" if t is not None else "t"
        return f"llm{r}_{tt}_{len(self.rejections)}"

    # ---- error-feedback revise loop (design section 4) -----------------------------------
    def revise(self, failed: Program, run_result, context: dict) -> Optional[Program]:
        """Re-author a failed candidate given its traceback. None if disabled / no client / no fix.

        This is the one-shot fix. The full agentic loop (write->run->read->fix) lives in the
        separate ``agentic.py``; when ``self.repair`` is set the caller may hand off after this.
        """
        if not self.cfg.enable_revise or self.client is None:
            return None
        ek = getattr(run_result, "error_kind", "") or ""
        err = getattr(run_result, "error", "") or ""
        hints = {
            "import": "Only numpy and sklearn import. Remove the offending import; use an "
                      "sklearn equivalent.",
            "build": "build_estimator() raised at construction. Check constructor arg names "
                     "against sklearn.",
            "fit": "Failed during fit. Likely a shape / dtype / numerical issue. Add scaling "
                   "or imputation; reduce polynomial degree.",
            "timeout": "Too slow under a CPU limit. Use HistGradientBoosting or a linear model, "
                       "cap n_estimators, set n_jobs=1.",
            "cpu": "Too slow under a CPU limit. Use HistGradientBoosting or a linear model, "
                   "cap n_estimators, set n_jobs=1.",
            "oom": "Out of memory. Avoid dense polynomial expansion on many features; use PCA "
                   "or SelectKBest first.",
        }
        hint = hints.get(ek, "Return a corrected build_estimator() that avoids this error.")
        prompt = (
            f"{self.SYSTEM}\n\n"
            f"Your previous build_estimator() failed when run.\n"
            f"error_kind={ek}\nerror={err}\n"
            f"{hint}\n\n"
            f"Previous code:\n{failed.code}\n\n"
            "Return a corrected build_estimator(). Same rules apply. Return only code."
        )
        raw = self._call(prompt, temperature=0.3)        # low temp: we want a fix, not exploration
        if raw is None:
            return None
        spec = None
        spec_d = failed.provenance.get("spec") if failed.provenance else None
        if isinstance(spec_d, dict):
            # reconstruct the typed spec from its stored dict so spec.check still runs
            spec = ProgramSpec(
                task_kind=spec_d.get("task_kind", context.get("task_kind", "classification")),
                must_handle=list(spec_d.get("must_handle", [])),
                encourage=list(spec_d.get("encourage", [])),
                forbid=list(spec_d.get("forbid", [])),
                budget=dict(spec_d.get("budget", {})),
            )
        prog = self._materialize(raw, context, spec, motifs=[], prompt=prompt)
        if prog is not None:
            prog.parent_id = failed.id
            prog.label = f"{failed.label}_rev"
            prog.provenance["revised_from"] = failed.id
            prog.provenance["revise_error_kind"] = ek
        return prog

    # ---- regeneration: champion code-mutation (design 5.1) -------------------------------
    def mutate_code(self, champion: Program, context: dict) -> Optional[Program]:
        """Ask the LLM to improve the champion's LITERAL code (not a recipe abstraction).

        This is what ``best_recipe`` could not express: an arbitrary authored pipeline with no
        recipe dict is still mutated because we mutate its source. ``parent_id`` -> champion.
        """
        if self.client is None:
            return None
        prompt = (
            f"{self.SYSTEM}\n\n"
            f"This pipeline is the current best (val {context.get('best_score')}):\n"
            f"{champion.code}\n\n"
            "Author an improved build_estimator(): change ONE thing that plausibly raises the "
            "validation metric (add feature engineering, swap/regularize the model, add a "
            "target transform). Keep what is working. Return only code."
        )
        raw = self._call(prompt, temperature=0.5)
        if raw is None:
            return None
        prog = self._materialize(raw, context, spec=None, motifs=[], prompt=prompt)
        if prog is not None:
            prog.parent_id = champion.id
            prog.provenance["mutated_from"] = champion.id
        return prog

    # ---- regeneration: ensemble combination of top-N (design 5.2) ------------------------
    def combine(self, top_programs: List[Program], context: dict) -> List[Program]:
        """Author ONE program combining the strengths of the strongest prior pipelines.

        The val scores shown in the prompt are the TRUSTED PARENT's numbers (attached by the
        caller as ``p.provenance['val_score']``), never LLM-reported. The combined program is
        authored fresh and re-validated + re-run; we never trust the LLM's claim it is better.
        """
        if self.client is None or len(top_programs) < 2:
            return []
        bodies = []
        for i, p in enumerate(top_programs):
            val = p.provenance.get("val_score") if p.provenance else None
            bodies.append(f"# candidate {i} (val {val})\n{p.code}")
        prompt = (
            f"{self.SYSTEM}\n\n"
            "Here are the strongest pipelines so far, with their validation scores:\n"
            + "\n\n# ---\n".join(bodies)
            + "\n\nAuthor ONE build_estimator() that combines their strengths, e.g. a "
            "StackingRegressor/StackingClassifier over the diverse base estimators with a "
            "simple meta-learner, or a feature-union of their preprocessing. Return only code."
        )
        raw = self._call(prompt, temperature=0.4)
        if raw is None:
            return []
        prog = self._materialize(raw, context, spec=None, motifs=[], prompt=prompt)
        if prog is None:
            return []
        prog.provenance["combined_from"] = [p.id for p in top_programs]
        return [prog]

    # ---- innovation 7.2: self-consistency over authored programs (VAL-only signal) -------
    @staticmethod
    def self_consistency(scored_programs: List[Tuple[Program, float]]) -> dict:
        """Structural agreement among the high-VAL authored programs (NON-promoting metadata).

        Input: (Program, val_score) pairs, val_score computed by the TRUSTED parent
        (certify.score_val). Output: a confidence report surfaced alongside the result --
        e.g. '4/5 high-scoring authored pipelines independently chose gradient boosting'.

        CRITICAL: this NEVER short-circuits the sealed certificate. It uses only trusted VAL
        numbers and is metadata on top of the frozen gate. It is self-consistency
        (known for reasoning) transplanted to PROGRAM space, grounded in held-out scores
        rather than the model's own confidence.
        """
        if not scored_programs:
            return {"n": 0, "agreement": 0.0, "modal_family": None, "note": "no programs"}
        # keep only those that ran and scored (the certified-or-promising set)
        valid = [(p, s) for p, s in scored_programs if s is not None]
        if not valid:
            return {"n": 0, "agreement": 0.0, "modal_family": None, "note": "none scored"}
        # consider the top half by VAL score as the 'high-scoring' set (min 2)
        valid.sort(key=lambda ps: ps[1], reverse=True)
        k = max(2, len(valid) // 2)
        top = valid[:k]
        families = [AuthoringEngine._family_of(p.code) for p, _ in top]
        from collections import Counter
        counts = Counter(f for f in families if f)
        if not counts:
            return {"n": len(top), "agreement": 0.0, "modal_family": None,
                    "note": "no recognizable family"}
        modal, modal_n = counts.most_common(1)[0]
        agreement = modal_n / len(families)
        has_tt = sum(1 for p, _ in top if "TransformedTargetRegressor" in p.code)
        return {
            "n": len(top),
            "agreement": round(agreement, 3),
            "modal_family": modal,
            "modal_count": modal_n,
            "uses_target_transform": has_tt,
            "note": f"{modal_n}/{len(families)} high-scoring authored pipelines chose {modal}",
        }

    @staticmethod
    def _family_of(code: str) -> Optional[str]:
        """Best-effort: identify the terminal estimator family in authored code (for 7.2)."""
        families = [
            ("HistGradientBoosting", "hist_gbm"),
            ("GradientBoosting", "gbm"),
            ("RandomForest", "random_forest"),
            ("ExtraTrees", "extra_trees"),
            ("LogisticRegression", "logreg"),
            ("Ridge", "ridge"), ("Lasso", "lasso"), ("ElasticNet", "elasticnet"),
            ("LinearRegression", "linreg"),
            ("SVC", "svc"), ("SVR", "svr"),
            ("KNeighbors", "knn"),
            ("MLP", "mlp"),
            ("Stacking", "stacking"),
            ("GaussianProcess", "gp"),
            ("KernelRidge", "kernel_ridge"),
        ]
        for needle, fam in families:
            if needle in code:
                return fam
        return None


class CoreAuthoringProposer(LLMProposer):
    """ProposalSource that authors arbitrary build_estimator() code via an LLM.

    Drop-in for the Phase-0 ``LLMProposer`` (it SUBCLASSES it): same Protocol, richer behavior.
    The subclass route is the design choice (section 1.4) because engine.py:103 detects an
    active LLM by ``isinstance(p, LLMProposer) and p.client is not None`` -- subclassing makes
    ``llm_active`` honest with ZERO Phase-0 edits. When client is None, ``propose()`` returns
    ``[]`` and the engine runs on the recipe floor (honest degrade).
    """

    def __init__(self, client: Optional[LLMClient] = None,
                 config: Optional[AuthoringConfig] = None,
                 retriever=None, firewall=None, repair=None):
        cfg = config or AuthoringConfig()
        # Initialize the Phase-0 base so .client / .n exist exactly as the engine expects.
        super().__init__(client=client, n=cfg.n)
        self.engine = AuthoringEngine(client=client, config=cfg, retriever=retriever,
                                      firewall=firewall, repair=repair)

    def propose(self, context: dict) -> List[Program]:
        """Author firewalled Programs for this round. [] when no client (engine -> floor)."""
        return self.engine.author(context)

    # Surface the engine's rejection log so the integrator can audit firewall activity.
    @property
    def rejections(self) -> List[dict]:
        return self.engine.rejections


__all__ = [
    "LLMClient", "Motif", "ProgramSpec", "Firewall",
    "AuthoringConfig", "AuthoringEngine", "CoreAuthoringProposer",
    "SYSTEM_CONTRACT", "PREAMBLE", "strip_fences", "sha12",
    "ALLOWED_IMPORT_ROOTS", "FORBIDDEN_CALL_NAMES", "FORBIDDEN_ATTR", "LEAKAGE_CALLS",
]
