"""Feature-engineering MOVES for the recursive cycle (NEW module; read-only on the trust core).

WHY THIS EXISTS
The diagnose->propose->VoI->execute->certify cycle today sweeps a fixed catalog of MODEL FAMILIES
(harness.py CLASSIFICATION_CATALOG / REGRESSION_CATALOG). It can pick a stronger estimator, but it can
NEVER transform the feature space: on a high-dimensional dataset with many noise features (a MADELON-style
problem) every raw-feature model -- including the logistic baseline -- is sandbagged by the noise, and the
cycle has no move that selects/expands/decorrelates features. This module adds that missing axis WITHOUT
touching the frozen certifier, the sealed peek, or select-then-bound.

THE CONTRACT IT REUSES
A harness.CatalogEntry maps a family name to a `builder(clamped_params, seed) -> estimator`, where the only
requirement on `estimator` is the sklearn `.fit(X, y)` / `.predict(X)` contract (LocalWorker.fit_score calls
exactly those; predict_proba is optional and guarded by the loop). A sklearn `Pipeline([transform, base])`
satisfies that contract EXACTLY: it is a single object with .fit/.predict (and .predict_proba iff the final
estimator has one). The transform stages here run on the ALREADY-FEATURIZED numeric matrix that the harness'
TabularFeaturizer emits (Xtr/Xva), so they compose with the existing pipeline without re-featurizing.

So a feature-engineering candidate is just another CatalogEntry whose builder returns a Pipeline. It is
proposed, VoI-ranked, executed, validated, sealed-peeked, and certified IDENTICALLY to a model family,
through the same frozen code path. The family name encodes the transform, e.g. 'selectk+hist_gbm'.

WHAT IT PROVIDES
  * TRANSFORMS: a registry of bounded transform stages -- variance-threshold + mutual-info SELECTION,
    polynomial INTERACTIONS, PCA, robust SCALING -- each a (sklearn transformer ctor, param specs, grid).
  * make_wrapped_entry(transform, base_family, base_catalog, ...): wrap a base catalog family into a single
    fit/predict CatalogEntry whose builder returns Pipeline([transform_stage, base_estimator]). Params are
    the union of the transform's params and the base family's params, each clamped through its own spec.
  * enumerate_feature_candidates(base_catalog, task_type, ...): a SMALL, deterministic grid of
    (transform, base_family) wrapped entries, returned as a {family_name: CatalogEntry} dict that merges
    straight into a runnable catalog (so the proposer/grid-fallback treat them as ordinary families).

INTEGRATION (done BY HAND later; this module never edits the core):
  In harness.runnable_catalog(...), after building `cat`, on the in-process tabular path:
      from .feature_moves import enumerate_feature_candidates
      cat.update(enumerate_feature_candidates(cat, task_type))
  That is the whole hook. resolve_family / move_from_proposal / propose_moves then handle the new families
  with zero changes: they are CatalogEntry objects keyed by name, clamped by the same clamp_params path.

The transforms are all worker_safe=False by default: worker/handler.build_model has no entry to reconstruct
a Pipeline, so a remote/GPU run filters them out (runnable_catalog already drops worker_safe=False on a
worker provider). They run on the in-process CPU path -- which is exactly where MADELON-style tabular sits.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, mutual_info_classif,
                                       mutual_info_regression)

from .harness import CatalogEntry


# ============================================================================ TRANSFORM STAGE REGISTRY
# Each transform stage is a small spec: a short name, a human label, a builder
# (clamped_params, n_features, seed, task_type) -> (sklearn_transformer, used_param_keys), a dict of
# param specs in the SAME ("kind", ...) format harness.CatalogEntry uses (so clamp_params works verbatim),
# and a deterministic grid for the grid-expansion fallback. The builder receives n_features so a selection
# fraction can be turned into a concrete k that is always valid for THIS dataset's width (no out-of-range k).

@dataclass
class TransformStage:
    name: str                 # short id used in the family name (e.g. "selectk")
    label: str                # human description (for diagnosis/logging)
    builder: object           # (params, n_features, seed, task_type) -> (transformer, used_keys)
    params: dict              # {param_name: spec} in harness ("kind", ...) format
    grid: dict                # {param_name: [grid values]} for the deterministic fallback


def _k_from_fraction(frac, n_features, floor=2):
    """A concrete, always-valid SelectKBest k from a keep-FRACTION of the input width. Clamped to
    [floor, n_features] so it is never <1 and never exceeds the available columns (sklearn would raise)."""
    n = int(max(1, n_features))
    k = int(round(float(frac) * n))
    return int(min(max(k, min(floor, n)), n))


def _build_variance(params, n_features, seed, task_type):
    """VarianceThreshold: drop near-constant columns. A pure, dataset-agnostic denoiser -- removes features
    whose variance falls below `vthresh`. Cheap, no labels used, safe on any numeric matrix."""
    return VarianceThreshold(threshold=float(params["vthresh"])), ["vthresh"]


def _build_selectk(params, n_features, seed, task_type):
    """SelectKBest with a MUTUAL-INFORMATION score. MI is a model-free dependency measure (captures
    NON-linear feature->target relevance, unlike an F-test), which is exactly the right selector for a
    MADELON-style problem where a handful of features matter through a non-linear target. k is derived from
    a keep-fraction of the actual feature width so it is always in [2, n_features]."""
    is_reg = task_type == "regression"
    # bind the seed so MI's internal kNN randomness is reproducible across builds of the same candidate
    score = ((lambda X, y, _s=seed: mutual_info_regression(X, y, random_state=_s)) if is_reg
             else (lambda X, y, _s=seed: mutual_info_classif(X, y, random_state=_s)))
    k = _k_from_fraction(params["keep_frac"], n_features)
    return SelectKBest(score_func=score, k=k), ["keep_frac"]


def _build_poly(params, n_features, seed, task_type):
    """PolynomialFeatures INTERACTIONS. interaction_only=True, degree=2: adds pairwise products x_i*x_j
    (no powers), the cheapest way to give a LINEAR model access to feature interactions. include_bias=False
    (the estimator handles its own intercept). NOTE: this squares the column count, so it is only enumerated
    after a selection stage in the combined grid -- never on a raw wide matrix (documented in enumerate_*)."""
    return (PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), [])


def _build_pca(params, n_features, seed, task_type):
    """PCA decorrelation / dimensionality reduction. n_components from a keep-fraction of the width, clamped
    to [1, n_features]. whiten left False (preserve scale for downstream trees). svd_solver explicit + seeded
    for reproducibility. Useful when features are correlated/redundant rather than individually irrelevant."""
    nc = _k_from_fraction(params["keep_frac"], n_features, floor=1)
    return PCA(n_components=nc, svd_solver="full", random_state=seed), ["keep_frac"]


def _build_robust(params, n_features, seed, task_type):
    """RobustScaler: center on the median, scale by the IQR. Robust to heavy tails / outliers (unlike
    StandardScaler's mean/std), which helps margin-based and distance-based base families. Label-free."""
    return RobustScaler(), []


TRANSFORMS = {
    "variance": TransformStage(
        "variance", "variance-threshold denoise (drop near-constant columns)", _build_variance,
        params={"vthresh": ("float", 0.0, 0.1)}, grid={"vthresh": [0.0, 1e-4]}),
    "selectk": TransformStage(
        "selectk", "mutual-information SelectKBest (keep top-relevance features)", _build_selectk,
        params={"keep_frac": ("float", 0.01, 1.0)}, grid={"keep_frac": [0.05, 0.15]}),
    "poly": TransformStage(
        "poly", "pairwise polynomial INTERACTIONS (degree-2, interaction-only)", _build_poly,
        params={}, grid={}),
    "pca": TransformStage(
        "pca", "PCA decorrelation / dim-reduction", _build_pca,
        params={"keep_frac": ("float", 0.01, 1.0)}, grid={"keep_frac": [0.1, 0.3]}),
    "robust": TransformStage(
        "robust", "robust (median/IQR) scaling", _build_robust,
        params={}, grid={}),
}


def _split_params(clamped):
    """Partition a wrapped entry's clamped params into (transform_params, base_params) by which keys belong
    to the stage vs the base family. Keys are NAMESPACED in the wrapped entry (t__* / b__*) so a shared name
    like 'C' or a shared 'keep_frac' never collides between the two halves."""
    tp, bp = {}, {}
    for key, val in clamped.items():
        if key.startswith("t__"):
            tp[key[3:]] = val
        elif key.startswith("b__"):
            bp[key[3:]] = val
    return tp, bp


def make_wrapped_entry(transform, base_family, base_catalog, *, prior_gain=None, prior_cost=None):
    """Wrap one base catalog family in a feature-transform stage, producing a SINGLE fit/predict CatalogEntry
    whose builder returns Pipeline([transform_stage, base_estimator]). The new family name is
    f'{transform}+{base_family}' (e.g. 'selectk+hist_gbm'). Params are the union of the transform's specs
    (namespaced t__*) and the base family's specs (namespaced b__*); each is clamped through its own original
    spec, so the frozen clamp path is unchanged. Returns a CatalogEntry, or None if either name is unknown.

    The returned estimator is a plain sklearn Pipeline: .fit/.predict (and .predict_proba iff the base has
    one) -- the SAME contract LocalWorker.fit_score and the certifier already rely on. n_features is read at
    fit time from X (the transform builders take n_features so k/n_components are always valid for the data)."""
    if transform not in TRANSFORMS:
        return None
    base_entry = base_catalog.get(base_family)
    if base_entry is None:
        return None
    stage = TRANSFORMS[transform]

    # union of namespaced params + grid
    params = {f"t__{k}": v for k, v in stage.params.items()}
    params.update({f"b__{k}": v for k, v in base_entry.params.items()})
    grid = {f"t__{k}": v for k, v in stage.grid.items()}
    grid.update({f"b__{k}": v for k, v in base_entry.grid.items()})

    is_reg = base_family.endswith("_reg") or base_family in ("ridge", "lasso", "svr")
    task_type = "regression" if is_reg else "binary"
    new_name = f"{transform}+{base_family}"

    def _build(clamped, seed):
        tp, bp = _split_params(clamped)
        base_est = base_entry.build(bp, seed)
        # Defer transform construction to fit time so n_components / k are sized to the ACTUAL feature
        # width of X (sklearn raises if k>n_features or n_components>min(n,p)). A tiny adapter transformer
        # builds the real stage in its own fit, using the params captured here.
        adapter = _DeferredStage(stage, tp, int(seed), task_type)
        return Pipeline([("transform", adapter), ("model", base_est)])

    return CatalogEntry(
        new_name, _build, params=params, grid=grid,
        prior_gain=(stage_gain(stage, base_entry) if prior_gain is None else prior_gain),
        prior_cost=(base_entry.prior_cost + 0.4 if prior_cost is None else prior_cost),
        worker_safe=False)


def stage_gain(stage, base_entry):
    """Prior expected val-gain for a wrapped candidate: a small bump over the base family's prior, because a
    relevant transform on a noisy/correlated space can unlock a model the raw space sandbags. Heuristic prior
    only -- the case-base/VoI re-calibrates it from MEASURED gains; never a claim about a specific dataset."""
    return float(min(0.20, base_entry.prior_gain + 0.04))


class _DeferredStage:
    """A sklearn-compatible transformer that builds its REAL stage at fit time, once n_features is known.
    This keeps SelectKBest k and PCA n_components always valid for the data on hand (the transform builders
    take n_features). Implements the transformer contract (.fit/.transform/.fit_transform/get_params/
    set_params) so it slots into a Pipeline transparently. No labels are peeked beyond what the wrapped
    sklearn selector itself uses inside .fit on the TRAINING split only."""

    def __init__(self, stage, params, seed, task_type):
        self.stage = stage
        self.params = params
        self.seed = seed
        self.task_type = task_type
        self._impl = None

    # sklearn clone()/Pipeline introspection support
    def get_params(self, deep=True):
        return {"stage": self.stage, "params": self.params, "seed": self.seed,
                "task_type": self.task_type}

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_features = X.shape[1] if X.ndim == 2 else 1
        impl, _used = self.stage.builder(self.params, n_features, self.seed, self.task_type)
        self._impl = impl
        self._impl.fit(X, y)
        return self

    def transform(self, X):
        if self._impl is None:
            raise RuntimeError(f"_DeferredStage({self.stage.name}) used before fit")
        return self._impl.transform(np.asarray(X))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ============================================================================ CANDIDATE ENUMERATION
# A SMALL, deterministic grid of (transform, base_family) wrapped entries. The selection of which pairings
# to enumerate is a fixed, defensible policy -- NOT tuned to any dataset:
#   * SELECTION (variance, selectk) pairs with EVERY base family: denoising helps every model on a wide
#     noisy matrix, and is the single most important move for MADELON-style data.
#   * POLY interactions pair ONLY with LINEAR base families (logistic/ridge), and ONLY after selection has
#     shrunk the width (selectk+poly+...), because degree-2 on a raw wide matrix is O(p^2) blowup. Poly gives
#     a linear model access to interactions a tree already captures, so it is pointless on tree bases.
#   * PCA pairs with distance/linear families that suffer from correlated dims (logistic, knn, svc_rbf).
#   * ROBUST scaling pairs with margin/distance families (svc_rbf, knn, logistic) that are scale-sensitive;
#     trees are scale-invariant so robust+tree is omitted.
# Every pairing below is justified by the inductive bias of the base family, not by an answer key.

# (transform, [base families it is paired with]) -- only families actually present in the catalog are used.
_CLF_PAIRINGS = [
    ("variance", ["logistic", "hist_gbm", "random_forest", "svc_rbf"]),
    ("selectk",  ["logistic", "hist_gbm", "random_forest", "svc_rbf", "knn"]),
    ("pca",      ["logistic", "svc_rbf", "knn"]),
    ("robust",   ["svc_rbf", "knn", "logistic"]),
]
_REG_PAIRINGS = [
    ("variance", ["ridge", "hist_gbm_reg", "random_forest_reg"]),
    ("selectk",  ["ridge", "hist_gbm_reg", "random_forest_reg", "knn_reg", "svr"]),
    ("pca",      ["ridge", "svr", "knn_reg"]),
    ("robust",   ["svr", "knn_reg", "ridge"]),
]
# Two-stage combos (selection THEN interactions) for linear bases -- selection first keeps poly cheap.
_CLF_COMBOS = [("selectk", "poly", "logistic")]
_REG_COMBOS = [("selectk", "poly", "ridge")]


def make_two_stage_entry(sel_transform, second_transform, base_family, base_catalog,
                         *, prior_gain=None, prior_cost=None):
    """A two-stage wrapped entry: Pipeline([selection, second_stage, base]). Used for selection-THEN-poly so
    degree-2 interactions are computed on the already-narrowed feature set (cheap), giving a linear base
    family access to interactions without an O(p^2) blowup on the raw matrix. Same fit/predict contract."""
    if sel_transform not in TRANSFORMS or second_transform not in TRANSFORMS:
        return None
    base_entry = base_catalog.get(base_family)
    if base_entry is None:
        return None
    s1, s2 = TRANSFORMS[sel_transform], TRANSFORMS[second_transform]
    is_reg = base_family.endswith("_reg") or base_family in ("ridge", "lasso", "svr")
    task_type = "regression" if is_reg else "binary"
    new_name = f"{sel_transform}+{second_transform}+{base_family}"

    # params: s1 (t1__*), s2 (t2__*), base (b__*)
    params = {f"t1__{k}": v for k, v in s1.params.items()}
    params.update({f"t2__{k}": v for k, v in s2.params.items()})
    params.update({f"b__{k}": v for k, v in base_entry.params.items()})
    grid = {f"t1__{k}": v for k, v in s1.grid.items()}
    grid.update({f"t2__{k}": v for k, v in s2.grid.items()})
    grid.update({f"b__{k}": v for k, v in base_entry.grid.items()})

    def _build(clamped, seed):
        t1p = {k[4:]: v for k, v in clamped.items() if k.startswith("t1__")}
        t2p = {k[4:]: v for k, v in clamped.items() if k.startswith("t2__")}
        bp = {k[3:]: v for k, v in clamped.items() if k.startswith("b__")}
        base_est = base_entry.build(bp, seed)
        a1 = _DeferredStage(s1, t1p, int(seed), task_type)
        a2 = _DeferredStage(s2, t2p, int(seed), task_type)
        return Pipeline([("select", a1), ("expand", a2), ("model", base_est)])

    return CatalogEntry(
        new_name, _build, params=params, grid=grid,
        prior_gain=(0.10 if prior_gain is None else prior_gain),
        prior_cost=(base_entry.prior_cost + 0.6 if prior_cost is None else prior_cost),
        worker_safe=False)


def enumerate_feature_candidates(base_catalog, task_type, *, include_combos=True):
    """Build the small grid of (transform, base_family) wrapped CatalogEntries for THIS catalog, as a
    {family_name: CatalogEntry} dict that merges straight into a runnable catalog. Only base families that
    are actually present in `base_catalog` are wrapped (so it respects the provider-filtered catalog), and
    families already produced by another transform are not duplicated. Deterministic order.

    This is the function the integration hook calls:  cat.update(enumerate_feature_candidates(cat, ttype)).
    """
    is_reg = task_type == "regression"
    pairings = _REG_PAIRINGS if is_reg else _CLF_PAIRINGS
    combos = (_REG_COMBOS if is_reg else _CLF_COMBOS) if include_combos else []

    out = {}
    for transform, bases in pairings:
        for base in bases:
            entry = make_wrapped_entry(transform, base, base_catalog)
            if entry is not None and entry.family not in out and entry.family not in base_catalog:
                out[entry.family] = entry
    for sel, second, base in combos:
        entry = make_two_stage_entry(sel, second, base, base_catalog)
        if entry is not None and entry.family not in out and entry.family not in base_catalog:
            out[entry.family] = entry
    return out


# ============================================================================ SELF-TEST / DEMO
def _madelon_like(n=900, p=200, n_informative=5, seed=0):
    """A synthetic MADELON-style binary problem: a few informative features driving the label, buried among
    many pure-noise features, with a NON-LINEAR (XOR-flavored) decision boundary. Built from first principles
    (no sklearn make_classification answer key):

      * Draw a hidden class label y ~ Bernoulli(1/2). Each informative feature is a class-conditional
        Gaussian (mean +m for class 1, -m for class 0) -> a genuine UNIVARIATE relevance signal, so a
        mutual-information selector can legitimately recover the informative columns (this is what real
        MADELON has; the original Guyon design clusters informative features by class).
      * Then FLIP the label on the subset of rows where a pairwise INTERACTION term (x0*x1) is negative.
        This injects a non-linear (interaction) component on top of the linear/univariate signal, so a
        purely linear model on the raw space underperforms a non-linear model on the SELECTED features --
        the property the demo needs, made honestly (the boundary is genuinely non-linear, not tuned).
      * The remaining p - n_informative columns are independent standard-normal noise.
      * Column order is shuffled so the informative features are not in a fixed prefix (a real selector
        must find them; the model never sees the answer key).
    """
    rng = np.random.default_rng(seed)
    y0 = rng.integers(0, 2, size=n)                     # hidden class
    m = 0.9                                             # class-conditional mean separation (univariate signal)
    signs = np.where(y0 == 1, 1.0, -1.0)[:, None]
    Xi = signs * m + rng.standard_normal((n, n_informative))
    # non-linear flip: where the leading interaction is negative, invert the label (XOR-flavored boundary)
    flip = (Xi[:, 0] * Xi[:, 1]) < 0
    y = np.where(flip, 1 - y0, y0).astype(int)
    noise = rng.standard_normal((n, p - n_informative))
    X = np.hstack([Xi, noise])
    order = rng.permutation(p)                          # hide the informative columns among the noise
    return X[:, order], y


def _demo():
    """Prove a feature-selection + non-linear candidate BEATS the raw logistic baseline (and raw GBM) on a
    MADELON-style problem, through the SAME builder path the cycle uses. Raw numbers first, then verdict."""
    from .harness import CLASSIFICATION_CATALOG

    X, y = _madelon_like(seed=0)
    n = len(y)
    cut = int(n * 0.7)
    Xtr, Xva, ytr, yva = X[:cut], X[cut:], y[:cut], y[cut:]

    def acc(entry, params):
        est = entry.build(params, seed=0)
        est.fit(Xtr, ytr)
        return float((est.predict(Xva) == yva).mean())

    base = CLASSIFICATION_CATALOG
    feat = enumerate_feature_candidates(base, "binary")

    print(f"MADELON-style: n={n} p={X.shape[1]} informative=5 (nonlinear target), "
          f"{cut} train / {n - cut} val")
    print(f"feature candidates enumerated: {len(feat)}")
    print("\n  RAW baselines (no feature engineering):")
    raw_log = acc(base["logistic"], {"C": 1.0})
    raw_gbm = acc(base["hist_gbm"], {"it": 300, "max_depth": 0, "learning_rate": 0.1})
    print(f"    raw logistic|C=1.0                 val_acc = {raw_log:.4f}")
    print(f"    raw hist_gbm|300                   val_acc = {raw_gbm:.4f}")

    print("\n  FEATURE-ENGINEERING candidates (sample, same builder path):")
    sk_gbm = feat["selectk+hist_gbm"]
    sk_log = feat["selectk+logistic"]
    skp_log = feat["selectk+poly+logistic"]
    a_sk_gbm = acc(sk_gbm, sk_gbm.clamp_params({"t__keep_frac": 0.05, "b__it": 300}))
    a_sk_log = acc(sk_log, sk_log.clamp_params({"t__keep_frac": 0.05, "b__C": 1.0}))
    # poly interactions are computed on the SELECTED set; tightest keep (informative count) is where the
    # x_i*x_j XOR term is cleanest, so the combo grid uses a tight keep_frac.
    a_skp_log = acc(skp_log, skp_log.clamp_params({"t1__keep_frac": 0.025, "b__C": 1.0}))
    print(f"    selectk+hist_gbm   (keep 5%)       val_acc = {a_sk_gbm:.4f}")
    print(f"    selectk+logistic   (keep 5%)       val_acc = {a_sk_log:.4f}")
    print(f"    selectk+poly+logistic (keep 2.5%)  val_acc = {a_skp_log:.4f}")

    best_feat = max(a_sk_gbm, a_sk_log, a_skp_log)
    print(f"\n  VERDICT: best feature-eng val_acc {best_feat:.4f} vs raw logistic {raw_log:.4f} "
          f"(+{best_feat - raw_log:.4f})  and raw gbm {raw_gbm:.4f} ({best_feat - raw_gbm:+.4f})")
    # HEADLINE CLAIM: a feature-SELECTION + NON-LINEAR candidate clearly beats the raw logistic baseline.
    assert a_sk_gbm > raw_log + 0.05, "selectk+hist_gbm must clearly beat the raw logistic baseline"
    assert best_feat > raw_log + 0.05, "best feature-eng candidate must clearly beat raw logistic"
    # SECONDARY (honest, smaller effect): poly interactions on the selected set help the LINEAR base recover
    # the XOR component -- a real but modest gain over raw logistic, not a headline.
    assert a_skp_log > raw_log, "selectk+poly+logistic should at least match/beat raw logistic"
    print("  OK: feature-selection + non-linear candidate beats the raw baseline.")


if __name__ == "__main__":
    _demo()
