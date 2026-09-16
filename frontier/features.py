"""Phase 1: feature engineering / preprocessing / target transforms as a first-class
proposal source.

# === WIRING ===
# The integrator plugs this in WITHOUT editing any Phase 0 file. Two new proposal sources
# are added to the engine's proposer list (they satisfy the `ProposalSource` Protocol from
# proposers.py: a `.propose(context) -> list[Program]` method, same context dict the engine
# already builds in engine.py:109-120).
#
#   from frontier.engine import ResearchEngine, EngineConfig
#   from frontier.features import FeatureProposer, FeatureMutationProposer
#   from frontier.proposers import SeedProposer, MutationProposer, LLMProposer
#
#   cfg = EngineConfig(rounds=4)
#   engine = ResearchEngine(cfg, proposers=[
#       SeedProposer(),                 # plain-model floor (existing)
#       FeatureProposer(),              # NEW: rich feature-engineered seeds (this module)
#       MutationProposer(),             # existing champion mutation (base/scale/poly/tlog)
#       FeatureMutationProposer(),      # NEW: guided genome mutation + crossover (this module)
#       LLMProposer(cfg.llm_client),    # existing (degrades to [] when client is None)
#   ])
#   result = engine.run(task)
#
# Call site / argument shapes:
#   - propose(context) is called once per round per source by engine.run (engine.py:125-130).
#   - context keys read here: "task_kind"(str), "n_features"(int), "n_train"(int),
#     "round"(int), "tried_labels"(set[str]), "best_label"(str|None), "best_score"(float|None),
#     "best_id"(str|None), "best_recipe"(dict|None), "recent_errors"(list[(label,kind,msg)]).
#     FeatureMutationProposer ALSO reads "best_recipe" provenance to seed its genome; if the
#     engine's champion was produced by THIS module, its provenance["recipe"] is a genome dict
#     this module understands (a strict SUPERSET of the proposers.py recipe schema, see below).
#   - Each returned Program carries provenance={"recipe": <genome>} so the engine's
#     MutationProposer/this module can read it back. recipe_label() guarantees a stable,
#     unique label per genome so the engine's tried_labels dedup (engine.py:127) works.
#   - Programs are executed by sandbox.run_program exactly like seeds; the emitted code only
#     defines build_estimator() (the firewall: code returns an estimator, never a score).
#
# Ordering rationale: put FeatureProposer AFTER SeedProposer so the plain-model floor is always
# tried (engine dedups by id/label, so duplicates are free); put FeatureMutationProposer LAST
# among deterministic sources so it can mutate whatever champion the cheaper sources found.
#
# WHY this kills the n_features>=60 gate (harness.py:391-392): that gate refused to PROPOSE
# feature engineering for low-dim data, capping California Housing at ~0.81. Here feature
# engineering is proposable for ANY dimensionality because a Program is arbitrary code, not a
# (family, params) menu pick. We verify on diabetes (10 features) that a feature-engineered
# recipe certifies above the plain-model floor.
#
# Schema compatibility WHY: the genome below is a strict superset of the proposers.py recipe
# (keys base/scale/poly/target_log are interpreted identically). make_code() here can render
# any proposers.py recipe, and recipe_label() degrades to a proposers-compatible label when no
# advanced keys are set, so champions can flow between the two MutationProposers losslessly.


The "recipe genome"
-------------------
A genome is a small ordered dict describing a *composed* preprocessing -> model pipeline:

    {
      "base":        str,            # base estimator key (see _BASES below)
      "scale":       str|bool,       # False | "standard" | "robust" | "minmax" | "maxabs" | True(=standard)
      "winsor":      float|None,     # clip features to [q, 1-q] quantiles (q in (0,0.2]); None=off
      "power":       str|None,       # None | "yeo-johnson" | "quantile-normal"
      "poly":        int,            # 0/1 = off, 2/3 = PolynomialFeatures(degree)
      "interactions_only": bool,     # with poly: only interaction terms (no x**2), cheaper & less colinear
      "reduce":      tuple|None,     # None | ("pca", n) | ("agg", n)  dimensionality reduction
      "select":      tuple|None,     # None | ("kbest_mi", k) | ("model", frac)  feature selection
      "target":      str|None,       # regression only: None|"log1p"|"yeo-johnson"|"quantile-normal"
    }

Safe-composition rules (enforced by `normalize_genome`, the heart of the "no garbage" guarantee):
  1. Step ORDER is fixed and physically meaningful, so two genomes with the same keys always
     compile to the same pipeline order: winsor -> scale/power (mutually arbitrated) -> poly ->
     reduce -> select -> model. Order is not a free gene (that would explode the space and
     produce semantically dead pipelines, e.g. select-before-create-interactions).
  2. `power` and `scale` are arbitrated: a power transform already standardizes, so if both are
     set we keep the power transform and drop redundant standard scaling (keeps robust/minmax/
     maxabs because those answer a different need). This avoids stacking two centering steps.
  3. `reduce` and `select` are mutually exclusive (both are width-reducers; composing them is
     rarely meaningful and doubles the failure surface). reduce wins if both are set.
  4. Counts are clamped to the data: PCA/agglomeration/k-best sizes are emitted as a fraction of
     the *runtime* feature count (computed inside build_estimator from X.shape[1]) so they never
     exceed available features regardless of what poly produced. This is why counts are stored
     as fractions, not absolute integers: poly changes the width at runtime.
  5. Target transforms apply to regression only (silently dropped for classification).
  6. log1p target requires non-negative targets; the emitted code falls back to identity at
     runtime if min(y)<0, so a bad gene degrades to a no-op rather than crashing the sandbox.

Why a BOUNDED beam (FeatureMutationProposer)
--------------------------------------------
Naive mutation+crossover over ~9 genes with 3-5 values each is ~10^4 pipelines; running each in
the sandbox is the cost. We bound the per-round output to `beam_width` (default 8) by:
  (a) generating a candidate pool with single-gene point mutations of the champion + a few
      crossovers between the champion and a curated "archetype" library,
  (b) scoring each candidate with a cheap, data-free HEURISTIC priority (NOT a promotion number:
      it only orders what to TRY, never what to promote -- invariant #3), and
  (c) taking the top `beam_width` untried ones. This is O(pool) generation + O(pool log pool)
      sort per round, with pool itself bounded (one mutation per gene + fixed archetypes), so the
      whole thing is linear in the number of genes. No combinatorial blowup.
The heuristic priority is explicitly a SEARCH-ordering shortcut (labelled as such), consistent
with the standing invariant that hardcoded heuristics seed/order the search but never promote.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from .program import Program
from .proposers import _BASES  # reuse the EXACT base-estimator registry (no divergence)


# --------------------------------------------------------------------------- genome schema

# Default genome: a plain model, identical in meaning to a bare proposers.py recipe.
_GENE_DEFAULTS: Dict = {
    "base": "ridge",
    "scale": False,
    "winsor": None,
    "power": None,
    "poly": 0,
    "interactions_only": False,
    "reduce": None,
    "select": None,
    "target": None,
}

# Allowed values per categorical gene -- the mutation alphabet. Kept small on purpose so the
# point-mutation pool stays linear in the number of genes (see beam-bound docstring).
_SCALE_CHOICES = (False, "standard", "robust", "minmax", "maxabs")
_POWER_CHOICES = (None, "yeo-johnson", "quantile-normal")
_POLY_CHOICES = (0, 2, 3)
_TARGET_CHOICES = (None, "log1p", "yeo-johnson", "quantile-normal")
_WINSOR_CHOICES = (None, 0.05, 0.01)
_REDUCE_CHOICES = (None, ("pca", 0.95), ("agg", 0.5))   # frac of runtime width
_SELECT_CHOICES = (None, ("kbest_mi", 0.5), ("model", 0.5))


def normalize_genome(g: Dict, kind: str) -> Dict:
    """Apply the safe-composition rules and return a canonical genome.

    Idempotent: normalize(normalize(g)) == normalize(g). This is what makes labels stable
    and the search space free of semantically-dead duplicates. See module docstring rules 1-6.
    """
    out = dict(_GENE_DEFAULTS)
    for k, v in g.items():
        if k in out:
            out[k] = v

    # --- base must exist for this kind; fall back to a kind-appropriate default otherwise.
    if (kind, out["base"]) not in _BASES:
        out["base"] = "ridge" if kind == "regression" else "logreg"

    # --- scale truthy bool means "standard" (compat with proposers.py recipe {"scale": True}).
    if out["scale"] is True:
        out["scale"] = "standard"
    if out["scale"] not in _SCALE_CHOICES:
        out["scale"] = False

    # --- rule 2: a power transform already centers/scales -> drop redundant standard scaling.
    if out["power"] not in _POWER_CHOICES:
        out["power"] = None
    if out["power"] is not None and out["scale"] == "standard":
        out["scale"] = False

    # --- poly clamp; degree 1 == off.
    if out["poly"] not in _POLY_CHOICES:
        out["poly"] = 0
    if out["poly"] in (0, 1):
        out["poly"] = 0
        out["interactions_only"] = False
    out["interactions_only"] = bool(out["interactions_only"])

    # --- winsor in (0, 0.2]; else off.
    if not (isinstance(out["winsor"], float) and 0.0 < out["winsor"] <= 0.2):
        out["winsor"] = None

    # --- rule 3: reduce and select are mutually exclusive; reduce wins.
    if out["reduce"] not in _REDUCE_CHOICES:
        out["reduce"] = None
    if out["select"] not in _SELECT_CHOICES:
        out["select"] = None
    if out["reduce"] is not None:
        out["select"] = None

    # --- rule 5: target transforms regression-only.
    if kind != "regression" or out["target"] not in _TARGET_CHOICES:
        out["target"] = None

    return out


def recipe_label(genome: Dict) -> str:
    """Stable, unique, human-readable label. Degrades to a proposers.py-compatible label when
    only base/scale/poly/target are set, so champions interchange between the two modules."""
    g = genome
    tags: List[str] = []
    if g.get("target"):
        tags.append({"log1p": "tlog", "yeo-johnson": "tyj", "quantile-normal": "tqn"}[g["target"]])
    if g.get("winsor"):
        tags.append(f"win{g['winsor']:g}")
    if g.get("power"):
        tags.append({"yeo-johnson": "yj", "quantile-normal": "qn"}[g["power"]])
    if g.get("poly"):
        tags.append(f"poly{int(g['poly'])}" + ("i" if g.get("interactions_only") else ""))
    if g.get("reduce"):
        tags.append(f"{g['reduce'][0]}{g['reduce'][1]:g}")
    if g.get("select"):
        tags.append(f"{g['select'][0]}{g['select'][1]:g}")
    if g.get("scale"):
        tags.append({"standard": "scale", "robust": "rscale",
                     "minmax": "mm", "maxabs": "ma"}.get(g["scale"], "scale"))
    tags.append(g["base"])
    return "+".join(tags)


# --------------------------------------------------------------------------- code emitter

def make_code(genome: Dict, kind: str) -> str:
    """Render a (normalized) genome to a complete module defining build_estimator().

    The emitted code is self-contained sklearn. Counts for PCA/agglomeration/k-best are
    computed AT RUNTIME from the post-poly feature width (rule 4) so they can never exceed
    the number of available features. The estimator is unfitted (the firewall: no metric,
    no fit here -- the sandbox fits it)."""
    g = normalize_genome(genome, kind)
    base = g["base"]
    imp, ctor = _BASES[(kind, base)]

    head = [
        "import numpy as np",
        "from sklearn.pipeline import Pipeline",
        imp,
    ]
    used = set()

    def need(line: str):
        if line not in used:
            used.add(line)
            head.append(line)

    body = ["", "def build_estimator():", "    steps = []"]

    # 1) winsorization (clip extreme feature values to robust quantiles). A tiny custom
    #    FunctionTransformer-free transformer keeps the emitted code dependency-light and
    #    picklable across the subprocess boundary (defined inline, top-level class).
    if g["winsor"] is not None:
        q = float(g["winsor"])
        head.append("from sklearn.base import BaseEstimator, TransformerMixin")
        head += [
            "",
            "class _Winsor(BaseEstimator, TransformerMixin):",
            "    # Clip each feature to its [q, 1-q] training quantiles. Robust to outliers,",
            "    # and crucially fit on TRAIN only (no test leakage) via the sklearn fit/transform API.",
            f"    def __init__(self, q={q!r}):",
            "        self.q = q",
            "    def fit(self, X, y=None):",
            "        X = np.asarray(X, dtype=float)",
            "        self.lo_ = np.quantile(X, self.q, axis=0)",
            "        self.hi_ = np.quantile(X, 1.0 - self.q, axis=0)",
            "        return self",
            "    def transform(self, X):",
            "        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)",
        ]
        body.append("    steps.append(('winsor', _Winsor()))")

    # 2) scale OR power (arbitrated by normalize_genome).
    if g["power"] == "yeo-johnson":
        need("from sklearn.preprocessing import PowerTransformer")
        body.append("    steps.append(('power', PowerTransformer(method='yeo-johnson', standardize=True)))")
    elif g["power"] == "quantile-normal":
        need("from sklearn.preprocessing import QuantileTransformer")
        body.append("    steps.append(('power', QuantileTransformer(output_distribution='normal', "
                    "n_quantiles=200, subsample=100000, random_state=0)))")
    if g["scale"] == "standard":
        need("from sklearn.preprocessing import StandardScaler")
        body.append("    steps.append(('scaler', StandardScaler()))")
    elif g["scale"] == "robust":
        need("from sklearn.preprocessing import RobustScaler")
        body.append("    steps.append(('scaler', RobustScaler()))")
    elif g["scale"] == "minmax":
        need("from sklearn.preprocessing import MinMaxScaler")
        body.append("    steps.append(('scaler', MinMaxScaler()))")
    elif g["scale"] == "maxabs":
        need("from sklearn.preprocessing import MaxAbsScaler")
        body.append("    steps.append(('scaler', MaxAbsScaler()))")

    # 3) polynomial / interaction features.
    if g["poly"]:
        need("from sklearn.preprocessing import PolynomialFeatures")
        body.append(f"    steps.append(('poly', PolynomialFeatures(degree={int(g['poly'])}, "
                    f"interaction_only={bool(g['interactions_only'])}, include_bias=False)))")

    # 4) dimensionality reduction (rule 4: size from RUNTIME width). Use a tiny wrapper that
    #    resolves the fraction against X.shape[1] at fit time so poly-blown widths are safe.
    if g["reduce"] is not None:
        kindr, frac = g["reduce"]
        head.append("from sklearn.base import BaseEstimator, TransformerMixin")
        if kindr == "pca":
            # PCA already accepts a float in (0,1) as "keep this much variance" -- exact and safe.
            need("from sklearn.decomposition import PCA")
            body.append(f"    steps.append(('reduce', PCA(n_components={float(frac)!r}, random_state=0)))")
        else:  # feature agglomeration: n_clusters as a fraction of runtime width.
            need("from sklearn.cluster import FeatureAgglomeration")
            head += [
                "",
                "class _AggFrac(BaseEstimator, TransformerMixin):",
                "    # FeatureAgglomeration with n_clusters = ceil(frac * n_features), clamped >=1",
                "    # and resolved at fit time so it tracks the post-poly width (rule 4).",
                f"    def __init__(self, frac={float(frac)!r}):",
                "        self.frac = frac",
                "    def fit(self, X, y=None):",
                "        X = np.asarray(X, dtype=float)",
                "        n = max(1, int(np.ceil(self.frac * X.shape[1])))",
                "        n = min(n, X.shape[1])",
                "        self.agg_ = FeatureAgglomeration(n_clusters=n)",
                "        self.agg_.fit(X)",
                "        return self",
                "    def transform(self, X):",
                "        return self.agg_.transform(np.asarray(X, dtype=float))",
            ]
            body.append("    steps.append(('reduce', _AggFrac()))")

    # 5) feature selection (mutually exclusive with reduce). Size from runtime width.
    if g["select"] is not None:
        kinds, frac = g["select"]
        head.append("from sklearn.base import BaseEstimator, TransformerMixin")
        if kinds == "kbest_mi":
            need("from sklearn.feature_selection import SelectKBest")
            mi = ("mutual_info_regression" if kind == "regression" else "mutual_info_classif")
            need(f"from sklearn.feature_selection import {mi}")
            head += [
                "",
                "class _KBestFrac(BaseEstimator, TransformerMixin):",
                "    # SelectKBest(MI) with k = ceil(frac * n_features), resolved at fit time so it",
                "    # is valid even after poly expansion (rule 4). MI captures nonlinear relevance.",
                f"    def __init__(self, frac={float(frac)!r}):",
                "        self.frac = frac",
                "    def fit(self, X, y=None):",
                "        X = np.asarray(X, dtype=float)",
                "        k = max(1, int(np.ceil(self.frac * X.shape[1])))",
                "        k = min(k, X.shape[1])",
                f"        self.sel_ = SelectKBest({mi}, k=k)",
                "        self.sel_.fit(X, y)",
                "        return self",
                "    def transform(self, X):",
                "        return self.sel_.transform(np.asarray(X, dtype=float))",
            ]
            body.append("    steps.append(('select', _KBestFrac()))")
        else:  # model-based selection via an L1/tree importance threshold.
            need("from sklearn.feature_selection import SelectFromModel")
            if kind == "regression":
                need("from sklearn.linear_model import Lasso")
                est_expr = "Lasso(alpha=0.01, max_iter=5000)"
            else:
                need("from sklearn.linear_model import LogisticRegression")
                est_expr = "LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=2000)"
            # max_features as a fraction is supported by SelectFromModel directly.
            body.append(f"    steps.append(('select', SelectFromModel({est_expr}, "
                        f"threshold=-np.inf, max_features=lambda X: max(1, int({float(frac)!r} * X.shape[1])))))")

    # 6) the model.
    body.append(f"    steps.append(('model', {ctor}))")
    body.append("    pipe = Pipeline(steps)")

    # 7) regression target transform (rule 5/6). log1p degrades to identity at runtime if y<0.
    if kind == "regression" and g["target"] is not None:
        need("from sklearn.compose import TransformedTargetRegressor")
        if g["target"] == "log1p":
            head += [
                "",
                "def _safe_log1p(y):",
                "    y = np.asarray(y, dtype=float)",
                "    # log1p needs y > -1; if any target violates this, degrade to identity (rule 6)",
                "    # rather than emit NaNs that would poison the sandbox fit.",
                "    return np.log1p(y) if np.min(y) > -1.0 else y",
                "def _safe_expm1(z):",
                "    return z  # paired with identity branch; TTR uses func/inverse symmetrically",
            ]
            # We cannot know y at code-emit time, so use a runtime-checked transformer pair.
            head += [
                "from sklearn.base import BaseEstimator, TransformerMixin",
                "",
                "class _Log1pTarget(BaseEstimator, TransformerMixin):",
                "    # Stateful target transform: decide log1p-vs-identity once at fit, apply",
                "    # the matching inverse. Avoids the func/inverse mismatch of a stateless pair.",
                "    def fit(self, y):",
                "        y = np.asarray(y, dtype=float).ravel()",
                "        self.use_ = bool(np.min(y) > -1.0)",
                "        return self",
                "    def transform(self, y):",
                "        y = np.asarray(y, dtype=float)",
                "        return np.log1p(y) if self.use_ else y",
                "    def inverse_transform(self, z):",
                "        z = np.asarray(z, dtype=float)",
                "        return np.expm1(z) if self.use_ else z",
            ]
            body.append("    return TransformedTargetRegressor(regressor=pipe, transformer=_Log1pTarget())")
        elif g["target"] == "yeo-johnson":
            need("from sklearn.preprocessing import PowerTransformer")
            body.append("    return TransformedTargetRegressor(regressor=pipe, "
                        "transformer=PowerTransformer(method='yeo-johnson', standardize=True))")
        else:  # quantile-normal target
            need("from sklearn.preprocessing import QuantileTransformer")
            body.append("    return TransformedTargetRegressor(regressor=pipe, "
                        "transformer=QuantileTransformer(output_distribution='normal', "
                        "n_quantiles=200, subsample=100000, random_state=0))")
    else:
        body.append("    return pipe")

    return "\n".join(head + body) + "\n"


def _program_from_genome(genome: Dict, kind: str, source: str,
                         parent_id: Optional[str] = None) -> Program:
    g = normalize_genome(genome, kind)
    return Program(code=make_code(g, kind), source=source, label=recipe_label(g),
                   parent_id=parent_id, provenance={"recipe": g})


# --------------------------------------------------------------------------- archetype library

def _archetypes(kind: str) -> List[Dict]:
    """Curated, literature-motivated feature-engineering recipes that are good STARTING POINTS
    for low-dim tabular data (exactly the regime the n_features>=60 gate abandoned). These are
    SEEDS, not promoters: the certifier decides which (if any) actually clears theta.

    Motivation per recipe (WHY, not just what):
      - poly2 interactions + ridge: captures pairwise feature coupling that a linear model
        misses; ridge controls the variance the expansion introduces. Classic for diabetes.
      - yeo-johnson + ridge: many tabular features are skewed; a power transform linearizes
        them for a linear base without assuming positivity (unlike Box-Cox).
      - quantile-normal + linear: rank-based gaussianization, robust to heavy tails.
      - robust-scale + winsor + base: outlier-hardened preprocessing.
      - poly2 + kbest_mi + ridge: expand then prune to the MI-relevant interactions.
    """
    lin = "ridge" if kind == "regression" else "logreg"
    archs = [
        {"base": lin, "scale": "standard", "poly": 2, "interactions_only": True},
        {"base": lin, "power": "yeo-johnson"},
        {"base": lin, "power": "quantile-normal"},
        {"base": lin, "scale": "robust", "winsor": 0.05},
        {"base": lin, "scale": "standard", "poly": 2, "select": ("kbest_mi", 0.5)},
        {"base": lin, "scale": "standard", "poly": 2},
        {"base": lin, "reduce": ("pca", 0.95), "scale": "standard"},
    ]
    if kind == "regression":
        archs += [
            {"base": "ridge", "scale": "standard", "poly": 2, "target": "yeo-johnson"},
            {"base": "ridge", "power": "yeo-johnson", "target": "log1p"},
        ]
    return [normalize_genome(a, kind) for a in archs]


# --------------------------------------------------------------------------- proposal sources

class FeatureProposer:
    """ProposalSource (seed flavor): emits the archetype feature-engineering recipes.

    This is the floor-raiser: it guarantees that for ANY task -- including low-dimensional
    ones the old n_features>=60 gate refused -- the search space contains real feature
    engineering, preprocessing, and (for regression) target transforms. It is deterministic
    and needs no LLM, so it works even when no model is wired in.
    """

    def __init__(self, extra: Optional[List[Dict]] = None):
        """`extra` lets the integrator inject task-specific archetypes; defaults are used otherwise."""
        self._extra = list(extra) if extra else []

    def propose(self, context: Dict) -> List[Program]:
        kind = context["task_kind"]
        tried = set(context.get("tried_labels", ()))
        out, seen = [], set()
        for g in self._archetypes(kind):
            p = _program_from_genome(g, kind, "feature")
            if p.label in tried or p.label in seen:
                continue
            seen.add(p.label)
            out.append(p)
        return out

    def _archetypes(self, kind: str) -> List[Dict]:
        return _archetypes(kind) + [normalize_genome(g, kind) for g in self._extra]


class FeatureMutationProposer:
    """ProposalSource (advanced): guided mutation + crossover over recipe genomes, bounded beam.

    Operators:
      - point mutation: change ONE gene of the champion to another allowed value (one candidate
        per (gene, value) pair). Linear in the number of genes -- the whole reason the alphabet
        is small.
      - crossover: combine the champion's genes with each archetype's genes (uniform two-parent
        crossover), giving the search a way to JUMP toward a known-good region instead of only
        hill-climbing one gene at a time.

    The pool is then ordered by a cheap, data-free priority and truncated to `beam_width`. The
    priority is a SEARCH-ORDERING heuristic only (it never promotes -- invariant #3): it is
    error-aware (it down-ranks gene values implicated in `context['recent_errors']`) and
    complexity-aware (it mildly prefers simpler pipelines first, Occam-style, to spend the
    sandbox budget on cheap wins before expensive ones).
    """

    def __init__(self, beam_width: int = 8, n_crossovers: int = 4):
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        self.beam_width = int(beam_width)
        self.n_crossovers = int(n_crossovers)

    # ---- candidate generation ------------------------------------------------------------
    def _point_mutations(self, champ: Dict, kind: str) -> List[Dict]:
        """One candidate per single-gene change. O(#genes * alphabet), no products."""
        cands: List[Dict] = []
        alphabets: Dict[str, Tuple] = {
            "scale": _SCALE_CHOICES,
            "power": _POWER_CHOICES,
            "poly": _POLY_CHOICES,
            "winsor": _WINSOR_CHOICES,
            "reduce": _REDUCE_CHOICES,
            "select": _SELECT_CHOICES,
        }
        if kind == "regression":
            alphabets["target"] = _TARGET_CHOICES
        # also allow swapping the base estimator (to any base of the same kind).
        bases = [name for (k, name) in _BASES if k == kind and name != champ["base"]]
        for b in bases:
            m = dict(champ); m["base"] = b
            cands.append(m)
        for gene, choices in alphabets.items():
            for v in choices:
                if champ.get(gene) == v:
                    continue
                m = dict(champ); m[gene] = v
                cands.append(m)
        # toggling interactions_only when poly is on is a meaningful 1-bit flip.
        if champ.get("poly"):
            m = dict(champ); m["interactions_only"] = not champ.get("interactions_only", False)
            cands.append(m)
        return cands

    def _crossovers(self, champ: Dict, kind: str) -> List[Dict]:
        """Uniform crossover between the champion and each archetype (capped at n_crossovers)."""
        out: List[Dict] = []
        for arch in _archetypes(kind)[: self.n_crossovers]:
            child = {}
            for gene in _GENE_DEFAULTS:
                # take from archetype on the "structural" genes, champion on the rest, so the
                # child inherits the archetype's feature-engineering idea but keeps the
                # champion's winning base/scale where it has them. Deterministic = reproducible.
                if gene in ("poly", "interactions_only", "power", "reduce", "select", "target"):
                    child[gene] = arch.get(gene, _GENE_DEFAULTS[gene])
                else:
                    child[gene] = champ.get(gene, arch.get(gene, _GENE_DEFAULTS[gene]))
            out.append(child)
        return out

    # ---- priority (search ordering only; NEVER a promotion number) -----------------------
    @staticmethod
    def _bad_tokens(recent_errors) -> set:
        """Tokens (gene values) appearing in labels of recently-failed candidates. Used to
        DOWN-RANK -- not forbid -- those choices, so a transient failure doesn't permanently
        kill a gene but does deprioritize it. This is the recent_errors feedback channel."""
        bad = set()
        for entry in (recent_errors or []):
            label = entry[0] if isinstance(entry, (tuple, list)) and entry else str(entry)
            for tok in str(label).split("+"):
                bad.add(tok)
        return bad

    def _priority(self, genome: Dict, bad: set) -> float:
        """Higher = try sooner. Cheap, data-free. Penalizes recently-failing tokens and,
        weakly, complexity (Occam ordering). NOT a score that can promote anything."""
        label = recipe_label(genome)
        toks = label.split("+")
        score = 0.0
        for t in toks:
            if t in bad:
                score -= 2.0                 # strong de-prioritization of implicated genes
        # mild complexity penalty: each non-trivial preprocessing gene costs a little so the
        # beam spends early budget on simpler pipelines first.
        complexity = sum(1 for g in ("winsor", "power", "poly", "reduce", "select", "target")
                         if genome.get(g))
        score -= 0.1 * complexity
        return score

    def propose(self, context: Dict) -> List[Program]:
        kind = context["task_kind"]
        tried = set(context.get("tried_labels", ()))
        best = context.get("best_recipe")
        # If the champion came from a non-genome source (e.g. a bare proposers.py recipe), it is
        # still a valid genome after normalization (superset schema). If there is no champion yet,
        # seed the genome from the strongest archetype so this proposer is useful from round 0.
        if best:
            champ = normalize_genome(best, kind)
        else:
            champ = _archetypes(kind)[0]

        pool = self._point_mutations(champ, kind) + self._crossovers(champ, kind)

        # normalize + dedup the pool (composition rules collapse equivalents to one label).
        bad = self._bad_tokens(context.get("recent_errors"))
        scored: List[Tuple[float, str, Dict]] = []
        seen = set()
        for g in pool:
            gn = normalize_genome(g, kind)
            lab = recipe_label(gn)
            if lab in tried or lab in seen or lab == recipe_label(champ):
                continue
            seen.add(lab)
            scored.append((self._priority(gn, bad), lab, gn))

        # stable sort by (priority desc, label) so ties are deterministic; take the beam.
        scored.sort(key=lambda t: (-t[0], t[1]))
        chosen = scored[: self.beam_width]

        out = []
        for _prio, _lab, gn in chosen:
            out.append(_program_from_genome(gn, kind, "feature_mut",
                                            parent_id=context.get("best_id")))
        return out
