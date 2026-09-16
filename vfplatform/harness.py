"""Per-ML-subfield Harness ABC + the live harnesses (build-order step 1-2, 11-seam).

A Harness encapsulates everything subfield-specific so the loop dispatches uniformly: the featurizer, the
target encoding, the validation scorer, and the MOVE MENU (the closed set of proposable expansions the
diagnose->propose->VoI loop chooses among). The frozen certifier + the sealed-test guard are shared and
never live here.
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              RandomForestRegressor, HistGradientBoostingRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.calibration import CalibratedClassifierCV

from .models import TabularFeaturizer, TextFeaturizer


def _calibrated_gbm(s):
    return CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=150, random_state=s),
                                  method="isotonic", cv=3)


# ============================================================================================== CATALOG
# The CATALOG is the single source of truth for "family name -> real estimator ctor + safe hyperparameter
# ranges". It lives HERE (not in llm_moves.py) so the family->sklearn-ctor build stays in ONE place and in
# parity with the worker build path (worker/handler.py build_model). The LLM-guided proposer (llm_moves.py)
# is bounded to choose ONLY from these families and only WITHIN these ranges; the frozen resolve_family()
# below CLAMPS every proposed param to its safe range and drops unknown families. The LLM never invents a
# family or an out-of-range hyperparameter that reaches an estimator.
#
# A CatalogEntry maps a family name to:
#   * build(params, seed) -> a fitted-ready sklearn estimator (CLAMPED params),
#   * params: {param_name: spec}, where spec is one of
#         ("float", lo, hi)  ("int", lo, hi)  ("choice", [allowed,...])
#     used both to validate/clamp LLM proposals AND to enumerate the deterministic grid fallback,
#   * grid: {param_name: [values...]} -- the deterministic grid-expansion values (a Cartesian product over
#     these is the no-LLM proposal set; each cell is still clamped through `params`),
#   * worker_safe: True iff worker/handler.py build_model can reconstruct this family (so a GPU/worker run
#     only proposes families the remote handler can build; the local CPU path may use the full zoo).
#
# Default params (when a key is absent from a proposal) come from clamping the MIDPOINT/first-choice, so a
# partial proposal is always buildable.


def _clamp(spec, value, default):
    """Clamp one proposed value to its catalog spec. spec = ("float"|"int", lo, hi) | ("choice", [vals])."""
    kind = spec[0]
    if kind == "choice":
        allowed = spec[1]
        return value if value in allowed else default
    lo, hi = spec[1], spec[2]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    v = min(max(v, lo), hi)
    return int(round(v)) if kind == "int" else v


def _default_for(spec):
    """A safe default value for a spec (first choice / range midpoint), used for absent or rejected keys."""
    kind = spec[0]
    if kind == "choice":
        return spec[1][0]
    lo, hi = spec[1], spec[2]
    mid = (lo + hi) / 2.0
    return int(round(mid)) if kind == "int" else mid


@dataclass
class CatalogEntry:
    family: str
    builder: object                  # (clamped_params: dict, seed: int) -> estimator
    params: dict                     # {name: spec}
    grid: dict                       # {name: [grid values]}
    prior_gain: float = 0.05
    prior_cost: float = 1.0
    worker_safe: bool = True         # worker/handler.build_model can reconstruct this family

    def clamp_params(self, raw):
        """Frozen verify/clamp: keep only known keys, clamp each to its safe range, fill absent keys with a
        safe default. Returns a fully-specified, buildable params dict (never trusts the proposal blindly)."""
        raw = raw or {}
        out = {}
        for name, spec in self.params.items():
            default = _default_for(spec)
            out[name] = _clamp(spec, raw.get(name, default), default) if name in raw else default
        return out

    def build(self, params, seed):
        return self.builder(self.clamp_params(params), int(seed))


@dataclass
class Move:
    """A proposable expansion: a named candidate set + VoI priors (expected val gain, relative cost)."""
    name: str
    families: list                  # [(family_name, ctor(seed)->estimator, params_dict)]
    prior_gain: float = 0.05        # expected validation improvement if chosen (calibrated by the case-base)
    prior_cost: float = 1.0         # relative cost (fan-out size x family weight)


def _torch_mlp_ctor(hidden, epochs, lr, dropout):
    """In-process estimator builder for a torch_mlp candidate. Only invoked on the IN-PROCESS path
    (LocalWorkerProvider); on a remote worker the loop serializes the family name+params and the worker's
    build_model reconstructs the model -- this ctor is never called there. Imports torch lazily so a
    machine without torch can still import this module (the move is gated out for such providers)."""
    def ctor(seed):
        import os
        import sys
        wp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker")
        if wp not in sys.path:
            sys.path.insert(0, wp)
        from torch_models import TorchMLPClassifier
        h = tuple(int(x) for x in str(hidden).split("x"))
        return TorchMLPClassifier(hidden=h, dropout=dropout, lr=lr, epochs=epochs, seed=seed)
    return ctor


def _torch_runnable(provider):
    """True iff a torch_mlp candidate can actually be executed by this provider. A remote GPU worker
    (RunPodProvider, device=gpu) has torch+cuda in its image. An in-process worker can run torch only if
    torch is importable locally. A non-worker (LocalCpuProvider, sklearn ctors) cannot, so the move is
    gated out -- never proposed where it would crash."""
    if provider is None:
        return False
    caps = {}
    try:
        caps = provider.capabilities() or {}
    except Exception:  # noqa: BLE001
        caps = {}
    if caps.get("device") == "gpu":                 # remote GPU worker: torch ships in the image
        return True
    if getattr(provider, "execution_mode", None) == "worker":
        import importlib.util
        return importlib.util.find_spec("torch") is not None
    return False


def _torch_catalog_entry(task_type):
    """A torch-MLP candidate family for the RUNNABLE CATALOG, so the recursive PROPOSE step actually proposes
    it (not just the static moves() menu, which the cycle never executes). Reconstructed on the worker via
    build_model (family starts with 'torch_mlp'; params carry hidden/epochs/lr/dropout) -> runs on cuda when
    the worker has a GPU. Added ONLY when _torch_runnable(provider); a non-torch provider never sees it. The
    frozen certifier is unchanged -- this is just another candidate family scored by the same sealed path."""
    is_reg = task_type == "regression"

    def _build(p, s):
        import os
        import sys
        wp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker")
        if wp not in sys.path:
            sys.path.insert(0, wp)
        from torch_models import TorchMLPClassifier, TorchMLPRegressor
        hidden = tuple(int(h) for h in str(p.get("hidden", "128x64")).split("x"))
        kw = dict(hidden=hidden, dropout=float(p.get("dropout", 0.1)), lr=float(p.get("lr", 1e-3)),
                  epochs=int(p.get("epochs", 80)), seed=int(s))
        return TorchMLPRegressor(**kw) if is_reg else TorchMLPClassifier(**kw)

    return CatalogEntry(
        "torch_mlp", _build,
        params={"hidden": ("choice", ["64", "128x64", "256x128"]), "epochs": ("int", 40, 200),
                "lr": ("float", 1e-4, 1e-2), "dropout": ("float", 0.0, 0.5)},
        grid={"hidden": ["128x64", "256x128"], "epochs": [80, 150], "lr": [1e-3], "dropout": [0.1]},
        prior_gain=0.08, prior_cost=2.5, worker_safe=True)


def _torch_cnn_catalog_entry():
    """A torch CNN classifier family (the GPU SPATIAL arm). Self-gating in worker/torch_models._ConvNet: a
    perfect-square feature width -> 2D conv over the image, else 1D conv -- so it never errors on shape and can
    sit in the catalog alongside torch_mlp. Reconstructed on the worker via build_model (family 'torch_cnn');
    runs on cuda when the worker has a GPU. Classification only. Frozen certifier scores it like any candidate."""
    def _build(p, s):
        import os
        import sys
        wp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker")
        if wp not in sys.path:
            sys.path.insert(0, wp)
        from torch_models import TorchCNNClassifier
        return TorchCNNClassifier(ch=int(p.get("ch", 16)), epochs=int(p.get("epochs", 40)),
                                  lr=float(p.get("lr", 1e-3)), seed=int(s))

    return CatalogEntry(
        "torch_cnn", _build,
        params={"ch": ("choice", [16, 32]), "epochs": ("int", 30, 120), "lr": ("float", 1e-4, 1e-2)},
        grid={"ch": [16, 32], "epochs": [40, 80], "lr": [1e-3]},
        prior_gain=0.08, prior_cost=2.8, worker_safe=True)


# ---------------------------------------------------------------------------- the per-task CATALOGS
# Family names that DO have a worker/handler.build_model entry are worker_safe=True (a GPU/worker run may
# propose them). Families with no worker entry (knn/mlp/svc_linear/lasso/*_reg analogues) are worker_safe=
# False -- proposed only on the local CPU path; on a worker provider they are filtered with a logged note so
# a remote run never dispatches a family the handler cannot reconstruct.

CLASSIFICATION_CATALOG = {
    "logistic": CatalogEntry(
        "logistic", lambda p, s: LogisticRegression(max_iter=2000, C=p["C"]),
        params={"C": ("float", 1e-3, 1e3)}, grid={"C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]},
        prior_gain=0.03, prior_cost=0.4, worker_safe=True),
    "svc_rbf": CatalogEntry(
        "svc_rbf", lambda p, s: SVC(C=p["C"], gamma=p["gamma"], kernel="rbf", probability=True,
                                    random_state=s),
        params={"C": ("float", 1e-2, 1e3), "gamma": ("choice", ["scale", "auto"])},
        grid={"C": [0.3, 1.0, 3.0, 10.0, 30.0], "gamma": ["scale", "auto"]},
        prior_gain=0.10, prior_cost=1.8, worker_safe=True),
    "svc_linear": CatalogEntry(
        "svc_linear", lambda p, s: SVC(C=p["C"], kernel="linear", probability=True, random_state=s),
        params={"C": ("float", 1e-2, 1e2)}, grid={"C": [0.1, 1.0, 10.0]},
        prior_gain=0.05, prior_cost=1.2, worker_safe=False),
    "random_forest": CatalogEntry(
        "random_forest", lambda p, s: RandomForestClassifier(
            n_estimators=p["n"], max_depth=(None if p["max_depth"] <= 0 else p["max_depth"]),
            random_state=s, n_jobs=1),
        params={"n": ("int", 50, 800), "max_depth": ("int", 0, 40)},
        grid={"n": [200, 400, 600], "max_depth": [0, 8, 16]},
        prior_gain=0.10, prior_cost=1.6, worker_safe=True),
    "extra_trees": CatalogEntry(
        "extra_trees", lambda p, s: ExtraTreesClassifier(n_estimators=p["n"], random_state=s, n_jobs=1),
        params={"n": ("int", 100, 800)}, grid={"n": [200, 400, 600]},
        prior_gain=0.09, prior_cost=1.4, worker_safe=True),
    "hist_gbm": CatalogEntry(
        "hist_gbm", lambda p, s: HistGradientBoostingClassifier(
            max_iter=p["it"], max_depth=(None if p["max_depth"] <= 0 else p["max_depth"]),
            learning_rate=p["learning_rate"], random_state=s),
        params={"it": ("int", 50, 600), "max_depth": ("int", 0, 16),
                "learning_rate": ("float", 1e-2, 0.5)},
        grid={"it": [150, 300, 500], "max_depth": [0, 6], "learning_rate": [0.05, 0.1, 0.2]},
        prior_gain=0.11, prior_cost=2.0, worker_safe=True),
    "knn": CatalogEntry(
        "knn", lambda p, s: KNeighborsClassifier(n_neighbors=p["k"]),
        params={"k": ("int", 1, 64)}, grid={"k": [3, 5, 11, 21, 41]},
        prior_gain=0.05, prior_cost=0.7, worker_safe=False),
    "mlp": CatalogEntry(
        "mlp", lambda p, s: MLPClassifier(
            hidden_layer_sizes={"64": (64,), "128": (128,), "128x64": (128, 64),
                                "256x128": (256, 128)}[p["hidden"]],
            alpha=p["alpha"], max_iter=300, random_state=s),
        params={"hidden": ("choice", ["64", "128", "128x64", "256x128"]),
                "alpha": ("float", 1e-5, 1e-1)},
        grid={"hidden": ["128", "128x64"], "alpha": [1e-4, 1e-2]},
        prior_gain=0.08, prior_cost=2.2, worker_safe=False),
}

REGRESSION_CATALOG = {
    "ridge": CatalogEntry(
        "ridge", lambda p, s: Ridge(alpha=p["alpha"]),
        params={"alpha": ("float", 1e-3, 1e3)}, grid={"alpha": [0.1, 1.0, 10.0, 100.0]},
        prior_gain=0.03, prior_cost=0.4, worker_safe=True),
    "lasso": CatalogEntry(
        "lasso", lambda p, s: Lasso(alpha=p["alpha"], max_iter=5000),
        params={"alpha": ("float", 1e-4, 1e1)}, grid={"alpha": [0.001, 0.01, 0.1, 1.0]},
        prior_gain=0.03, prior_cost=0.4, worker_safe=False),
    "random_forest_reg": CatalogEntry(
        "random_forest_reg", lambda p, s: RandomForestRegressor(
            n_estimators=p["n"], max_depth=(None if p["max_depth"] <= 0 else p["max_depth"]),
            random_state=s, n_jobs=1),
        params={"n": ("int", 50, 800), "max_depth": ("int", 0, 40)},
        grid={"n": [200, 400, 600], "max_depth": [0, 12]},
        prior_gain=0.12, prior_cost=1.6, worker_safe=True),
    "extra_trees_reg": CatalogEntry(
        "extra_trees_reg", lambda p, s: ExtraTreesRegressor(n_estimators=p["n"], random_state=s, n_jobs=1),
        params={"n": ("int", 100, 800)}, grid={"n": [200, 400, 600]},
        prior_gain=0.10, prior_cost=1.4, worker_safe=False),
    "hist_gbm_reg": CatalogEntry(
        "hist_gbm_reg", lambda p, s: HistGradientBoostingRegressor(
            max_iter=p["it"], max_depth=(None if p["max_depth"] <= 0 else p["max_depth"]),
            learning_rate=p["learning_rate"], random_state=s),
        params={"it": ("int", 50, 600), "max_depth": ("int", 0, 16),
                "learning_rate": ("float", 1e-2, 0.5)},
        grid={"it": [150, 300, 500], "max_depth": [0, 6], "learning_rate": [0.05, 0.1]},
        prior_gain=0.13, prior_cost=2.0, worker_safe=True),
    "knn_reg": CatalogEntry(
        "knn_reg", lambda p, s: KNeighborsRegressor(n_neighbors=p["k"]),
        params={"k": ("int", 1, 64)}, grid={"k": [3, 5, 11, 21]},
        prior_gain=0.05, prior_cost=0.7, worker_safe=False),
    "svr": CatalogEntry(
        "svr", lambda p, s: SVR(C=p["C"], gamma=p["gamma"], kernel="rbf"),
        params={"C": ("float", 1e-2, 1e3), "gamma": ("choice", ["scale", "auto"])},
        grid={"C": [1.0, 10.0, 100.0], "gamma": ["scale", "auto"]},
        prior_gain=0.09, prior_cost=1.8, worker_safe=False),
    "mlp_reg": CatalogEntry(
        "mlp_reg", lambda p, s: MLPRegressor(
            hidden_layer_sizes={"64": (64,), "128": (128,), "128x64": (128, 64)}[p["hidden"]],
            alpha=p["alpha"], max_iter=400, random_state=s),
        params={"hidden": ("choice", ["64", "128", "128x64"]), "alpha": ("float", 1e-5, 1e-1)},
        grid={"hidden": ["128", "128x64"], "alpha": [1e-4, 1e-2]},
        prior_gain=0.07, prior_cost=2.2, worker_safe=False),
}

# Text keeps tfidf + sparse-friendly families. The TextFeaturizer emits ONE sparse word-level TF-IDF matrix
# (shared across the whole zoo), so every estimator here must fit/predict on that sparse matrix -- no family
# may re-featurize raw text (char-ngram / hashing-vectorizer variants would require a DIFFERENT featurizer,
# which is harness-level and out of scope for a catalog-only expansion; documented as an honest constraint).
# Within that path the zoo is still genuinely diverse: a max-margin linear SVM (LinearSVC), two online linear
# learners with different loss surfaces (SGD hinge = SVM-like, SGD log = calibrated logistic), two Bayesian
# baselines (Multinomial + Complement NB, the latter built for imbalanced/long-document text), and a small
# neural MLP over the sparse vector. All are CPU-cheap and accept scipy sparse input directly.
#
# Families WITHOUT predict_proba (LinearSVC, SGD-hinge) are fine: the loop's ECE/proba step is guarded by
# hasattr+try/except (loop.py ~L591) and the certify path uses .predict (labels) only, so proba=None just
# means no calibration signal -- ranking + certification are unaffected.
#
# All worker_safe=False (like the existing two): worker/handler.build_model has no entry for these text
# families, so a remote/GPU run filters them out; they run only on the in-process CPU path.
TEXT_CATALOG = {
    "tfidf+logistic": CatalogEntry(
        "tfidf+logistic", lambda p, s: LogisticRegression(max_iter=2000, C=p["C"]),
        params={"C": ("float", 1e-2, 1e2)}, grid={"C": [0.3, 1.0, 3.0, 10.0, 30.0]},
        prior_gain=0.05, prior_cost=0.5, worker_safe=False),
    "tfidf+multinomial_nb": CatalogEntry(
        "tfidf+multinomial_nb", lambda p, s: MultinomialNB(alpha=p["alpha"]),
        params={"alpha": ("float", 1e-3, 5.0)}, grid={"alpha": [0.1, 0.5, 1.0]},
        prior_gain=0.03, prior_cost=0.4, worker_safe=False),
    "tfidf+complement_nb": CatalogEntry(
        # Complement NB: a Rennie et al. (2003) variant of MultinomialNB designed for text; estimates each
        # class's parameters from the COMPLEMENT of that class, which corrects the multinomial's bias on
        # imbalanced / long documents and frequently beats plain MultinomialNB on topic text (e.g. AG-News).
        "tfidf+complement_nb", lambda p, s: ComplementNB(alpha=p["alpha"]),
        params={"alpha": ("float", 1e-3, 5.0)}, grid={"alpha": [0.1, 0.3, 1.0]},
        prior_gain=0.05, prior_cost=0.4, worker_safe=False),
    "tfidf+linear_svc": CatalogEntry(
        # Linear max-margin SVM (liblinear). A different inductive bias from logistic's log-loss: the hinge
        # margin is a strong, classic TF-IDF text baseline (Joachims 1998) and often the top linear model on
        # sparse text. No predict_proba (the loop guards that path); certify uses .predict labels.
        "tfidf+linear_svc", lambda p, s: LinearSVC(C=p["C"], random_state=s),
        params={"C": ("float", 1e-2, 1e2)}, grid={"C": [0.3, 1.0, 3.0, 10.0]},
        prior_gain=0.06, prior_cost=0.5, worker_safe=False),
    "tfidf+sgd_hinge": CatalogEntry(
        # SGD with hinge loss: an online/streaming linear SVM with explicit L2 strength (alpha). Same loss
        # surface as LinearSVC but a different optimizer + regularization knob, giving the search a distinct
        # bias/variance trade-off. No predict_proba (guarded). class_weight balanced helps skewed splits.
        "tfidf+sgd_hinge", lambda p, s: SGDClassifier(loss="hinge", alpha=p["alpha"], max_iter=2000,
                                                      tol=1e-3, class_weight="balanced", random_state=s),
        params={"alpha": ("float", 1e-6, 1e-2)}, grid={"alpha": [1e-5, 1e-4, 1e-3]},
        prior_gain=0.05, prior_cost=0.4, worker_safe=False),
    "tfidf+sgd_log": CatalogEntry(
        # SGD with log_loss: a calibrated (predict_proba-capable) online logistic regression. Differs from the
        # batch `tfidf+logistic` (liblinear/lbfgs) in optimizer + the alpha L2 parameterization, so it covers
        # a different region of the regularization path and contributes an ECE/calibration signal.
        "tfidf+sgd_log", lambda p, s: SGDClassifier(loss="log_loss", alpha=p["alpha"], max_iter=2000,
                                                    tol=1e-3, random_state=s),
        params={"alpha": ("float", 1e-6, 1e-2)}, grid={"alpha": [1e-5, 1e-4, 1e-3]},
        prior_gain=0.05, prior_cost=0.4, worker_safe=False),
    "tfidf+mlp": CatalogEntry(
        # A small neural MLP over the sparse TF-IDF vector (sklearn MLP accepts scipy-sparse X). The only
        # non-linear family in the text zoo; CPU-cheap with a single 128-unit hidden layer and a capped
        # iteration budget. alpha is the L2 penalty. Gives the search a genuinely different hypothesis class.
        "tfidf+mlp", lambda p, s: MLPClassifier(
            hidden_layer_sizes={"64": (64,), "128": (128,), "128x64": (128, 64)}[p["hidden"]],
            alpha=p["alpha"], max_iter=120, early_stopping=True, random_state=s),
        params={"hidden": ("choice", ["64", "128", "128x64"]), "alpha": ("float", 1e-5, 1e-1)},
        grid={"hidden": ["128"], "alpha": [1e-4, 1e-2]},
        prior_gain=0.06, prior_cost=1.4, worker_safe=False),
}


def catalog_for(kind, task_type):
    """The per-task CATALOG (family -> CatalogEntry). One place; shared by llm_moves.propose_moves and the
    deterministic grid fallback. Text routes to the tfidf catalog; tabular to clf/reg by task_type."""
    if kind == "text":
        return TEXT_CATALOG
    if task_type == "regression":
        return REGRESSION_CATALOG
    return CLASSIFICATION_CATALOG


def runnable_catalog(kind, task_type, provider=None, n_features=None):
    """The catalog filtered to what THIS provider can actually execute. On a worker/GPU provider, drop
    families with no worker/handler.build_model entry (worker_safe=False) so a remote run never dispatches an
    unbuildable family. On the in-process (local CPU) path, the full zoo is runnable. Returns (catalog, dropped)."""
    cat = dict(catalog_for(kind, task_type))
    # FEATURE-ENGINEERING moves: merge transform-wrapped candidates (selection/PCA/scaling + a base estimator,
    # as a sklearn Pipeline) so the recursive PROPOSE step can ENGINEER FEATURES, not just pick models --
    # certified exactly like a model family. GATED to HIGH-DIMENSIONAL tasks (n_features >= 60): on small/clean
    # tasks (digits 64-ish/breast_cancer 30) they add no value and bloat every run (degree-2 poly on 64 feats =
    # 2080 cols -> slow); they earn their cost on high-dim feature-selection problems (madelon-style). combos
    # off keeps the set small/cheap. worker_safe=False, so the worker-path filter below drops them.
    if kind == "tabular" and n_features is not None and n_features >= 60:
        try:
            from .feature_moves import enumerate_feature_candidates
            cat.update(enumerate_feature_candidates(cat, task_type, include_combos=False))
        except Exception:  # noqa: BLE001  feature moves are additive proposal candidates; never break the catalog
            pass
    # GPU/DL arm: inject the torch-MLP family into the RUNNABLE catalog when this provider can execute torch
    # (in-process worker with torch importable, or a remote GPU worker). This is what makes the recursive
    # PROPOSE step actually try a deep model -- runs on cuda via the worker's build_model, certified normally.
    if _torch_runnable(provider) and kind == "tabular":   # tabular DL arm first; text already has tfidf+mlp
        cat["torch_mlp"] = _torch_catalog_entry(task_type)
        if task_type != "regression":                     # CNN classifier: 2D conv on image-shaped inputs
            cat["torch_cnn"] = _torch_cnn_catalog_entry()
    is_worker = getattr(provider, "execution_mode", None) == "worker" if provider is not None else False
    if not is_worker:
        return cat, []
    keep = {k: v for k, v in cat.items() if v.worker_safe}
    dropped = [k for k, v in cat.items() if not v.worker_safe]
    return keep, dropped


def resolve_family(catalog, family, params):
    """FROZEN verify/resolve: map a proposed (family, params) to a real local estimator ctor with CLAMPED
    params. Returns (family_name, ctor(seed)->est, clamped_params) or None for an unknown family. Used by
    both the proposer (to materialize an LLM/grid proposal into a runnable Move) and as the single guard
    that an out-of-catalog family or out-of-range param never reaches an estimator."""
    entry = catalog.get(family)
    if entry is None:
        return None
    clamped = entry.clamp_params(params)
    ctor = lambda s, _e=entry, _p=clamped: _e.build(_p, s)
    return entry.family, ctor, clamped


def _worker_params(clamped):
    """Make a clamped params dict safe for the WORKER build path (worker/handler.build_model). The local
    builder treats max_depth<=0 as 'no limit'; the worker passes max_depth straight to sklearn (where 0 is
    invalid), so a 0 sentinel is translated to None over the wire. Parity in one place."""
    out = dict(clamped)
    if "max_depth" in out and isinstance(out["max_depth"], (int, float)) and out["max_depth"] <= 0:
        out["max_depth"] = None
    return out


def move_from_proposal(catalog, family, params, *, prefix=""):
    """Build a single-family Move from a proposed (family, params), clamped through the catalog. The Move
    name encodes family+clamped params so distinct configs are distinct moves (distinct leaderboard rows /
    case-base keys). Returns None for an unknown family."""
    resolved = resolve_family(catalog, family, params)
    if resolved is None:
        return None
    fam, ctor, clamped = resolved
    entry = catalog[family]
    tag = "|".join(f"{k}={clamped[k]}" for k in sorted(clamped))
    name = f"{fam}|{tag}" if tag else fam
    # the params carried on the Move are WORKER-SAFE (max_depth 0->None) + tagged with the family so the
    # worker's build_model (family.split('|')[0]) and the leaderboard/content-id stay in parity.
    return Move(name, [(name, ctor, dict(_worker_params(clamped), family=fam))],
                prior_gain=entry.prior_gain, prior_cost=entry.prior_cost)


def _grid_configs(entry):
    """All (family, params) cells of one catalog entry's deterministic grid (Cartesian product over grid)."""
    import itertools
    keys = list(entry.grid.keys())
    if not keys:
        return [(entry.family, {})]
    combos = itertools.product(*[entry.grid[k] for k in keys])
    return [(entry.family, dict(zip(keys, vals))) for vals in combos]


class Harness(ABC):
    kind = "tabular"                # tabular | text
    task_type = "binary"           # binary | multiclass | regression
    default_metric = "accuracy"
    higher_is_better = True

    def is_regression(self):
        return self.task_type == "regression"

    @abstractmethod
    def featurizer(self):
        ...

    @abstractmethod
    def moves(self, seeds, provider=None):
        ...

    def catalog(self):
        """The per-task CATALOG (family -> CatalogEntry) this harness draws candidates from. The LLM-guided
        proposer + the deterministic grid fallback are both bounded to this set."""
        return catalog_for(self.kind, self.task_type)

    # ---- target encoding (classification -> indices; regression -> floats) ----------------------
    def encode_targets(self, rows, target_key, labels=None):
        if self.is_regression():
            return np.array([float(r.get(target_key)) for r in rows], dtype=float), None
        labels = labels or sorted({str(r.get(target_key)) for r in rows})
        l2i = {lab: i for i, lab in enumerate(labels)}
        return np.array([l2i[str(r.get(target_key))] for r in rows]), l2i

    def val_score(self, est, Xva, yva):
        """Higher-is-better validation metric used to RANK the fan-out (accuracy for clf, r2 for reg)."""
        pred = est.predict(Xva)
        if self.is_regression():
            ss_res = float(((yva - pred) ** 2).sum())
            ss_tot = float(((yva - yva.mean()) ** 2).sum()) or 1e-9
            return 1.0 - ss_res / ss_tot          # R^2
        return float((pred == yva).mean())        # accuracy

    def fit_score(self, ctor, seed, Xtr, ytr, Xva, yva):
        est = ctor(seed)
        t0 = time.time()
        est.fit(Xtr, ytr)
        latency_ms = (time.time() - t0) * 1000.0
        return self.val_score(est, Xva, yva), latency_ms, est

    def predict_fn(self, est, feat, l2i):
        """Build a row-level predictor for the sealed-test certify (decodes to labels / floats)."""
        if self.is_regression():
            return lambda rows: [float(p) for p in est.predict(feat.transform(rows))]
        inv = {i: lab for lab, i in l2i.items()}
        return lambda rows: [inv[int(p)] for p in est.predict(feat.transform(rows))]


# ----------------------------------------------------------------------------- live harnesses
class TabularClassificationHarness(Harness):
    kind, task_type, default_metric = "tabular", "binary", "accuracy"

    def featurizer(self):
        return TabularFeaturizer()

    def moves(self, seeds, provider=None):
        ms = [
            Move("baseline", [("logistic|C1.0", lambda s: LogisticRegression(max_iter=2000), {"C": 1.0})],
                 prior_gain=0.0, prior_cost=0.3),
            Move("stronger_model",
                 [("random_forest|200", lambda s: RandomForestClassifier(n_estimators=200, random_state=s), {"n": 200}),
                  ("hist_gbm|150", lambda s: HistGradientBoostingClassifier(max_iter=150, random_state=s), {"it": 150}),
                  ("extra_trees|300", lambda s: ExtraTreesClassifier(n_estimators=300, random_state=s), {"n": 300}),
                  ("svc_rbf|C1", lambda s: SVC(C=1.0, probability=True, random_state=s), {"C": 1.0})],
                 prior_gain=0.10, prior_cost=1.5),
            Move("more_capacity",
                 # max_depth carried in params so the WORKER (handler.build_model) builds the SAME depth cap
                 # as the in-process ctor -- otherwise worker mode would silently fit with max_depth=None.
                 [("random_forest|400d12", lambda s: RandomForestClassifier(n_estimators=400, max_depth=12, random_state=s), {"n": 400, "max_depth": 12}),
                  ("hist_gbm|300d8", lambda s: HistGradientBoostingClassifier(max_iter=300, max_depth=8, random_state=s), {"it": 300, "max_depth": 8})],
                 prior_gain=0.04, prior_cost=2.2),
            Move("regularize",
                 [("logistic|C0.1", lambda s: LogisticRegression(max_iter=2000, C=0.1), {"C": 0.1}),
                  ("logistic|C10", lambda s: LogisticRegression(max_iter=2000, C=10.0), {"C": 10.0})],
                 prior_gain=0.02, prior_cost=0.5),
            Move("calibrate",
                 [("calibrated_gbm", _calibrated_gbm, {"method": "isotonic"})],
                 prior_gain=0.03, prior_cost=1.2),
        ]
        if _torch_runnable(provider):
            # GPU family: a torch MLP, dispatched to the worker (cuda) in worker mode. Gated out for
            # providers that can't run torch so a local-CPU goal never proposes an un-runnable move.
            ms.append(Move(
                "deep_model",
                [("torch_mlp|h64", _torch_mlp_ctor("64", 60, 1e-3, 0.1),
                  {"hidden": "64", "epochs": 60, "lr": 0.001, "dropout": 0.1, "weight_decay": 0.0}),
                 ("torch_mlp|h128x64", _torch_mlp_ctor("128x64", 80, 1e-3, 0.1),
                  {"hidden": "128x64", "epochs": 80, "lr": 0.001, "dropout": 0.1, "weight_decay": 0.0})],
                prior_gain=0.08, prior_cost=2.5))
        return ms


class TextClassificationHarness(Harness):
    kind, task_type, default_metric = "text", "binary", "accuracy"

    def __init__(self, text_key="text"):
        self.text_key = text_key

    def featurizer(self):
        return TextFeaturizer(text_key=self.text_key)

    def moves(self, seeds, provider=None):
        ms = [
            Move("baseline", [("tfidf+logistic|C1.0", lambda s: LogisticRegression(max_iter=2000), {"C": 1.0})],
                 prior_gain=0.0, prior_cost=0.3),
            Move("stronger_model",
                 # diverse linear/Bayesian text families over the SAME sparse TF-IDF matrix: a higher-C
                 # logistic, a max-margin linear SVM, and Complement NB -- distinct inductive biases, all
                 # CPU-cheap, all in TEXT_CATALOG. The featurizer is unchanged (no re-vectorization here).
                 [("tfidf+logistic|C3.0", lambda s: LogisticRegression(max_iter=2000, C=3.0), {"C": 3.0}),
                  ("tfidf+linear_svc|C1.0", lambda s: LinearSVC(C=1.0, random_state=s), {"C": 1.0}),
                  ("tfidf+complement_nb|a0.3", lambda s: ComplementNB(alpha=0.3), {"alpha": 0.3})],
                 prior_gain=0.06, prior_cost=0.6),
            Move("online_linear",
                 # online linear learners: SGD hinge (SVM-like, no proba) and SGD log (calibrated logistic),
                 # different optimizer + L2 parameterization than the batch fits above.
                 [("tfidf+sgd_hinge|a1e-4",
                   lambda s: SGDClassifier(loss="hinge", alpha=1e-4, max_iter=2000, tol=1e-3,
                                           class_weight="balanced", random_state=s),
                   {"alpha": 1e-4}),
                  ("tfidf+sgd_log|a1e-4",
                   lambda s: SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-3,
                                           random_state=s),
                   {"alpha": 1e-4})],
                 prior_gain=0.05, prior_cost=0.5),
            Move("regularize",
                 [("tfidf+logistic|C0.3", lambda s: LogisticRegression(max_iter=2000, C=0.3), {"C": 0.3}),
                  ("tfidf+multinomial_nb", lambda s: MultinomialNB(), {})],
                 prior_gain=0.02, prior_cost=0.5),
        ]
        if _torch_runnable(provider):
            # representation-learning lever for hard text: a torch MLP over the (densified) TF-IDF vector,
            # dispatched to the GPU worker. The family name starts with "torch_mlp" so the worker's
            # build_model recognizes it; TF-IDF is featurized locally and TorchMLP densifies the sparse rows.
            # Gated out for providers that can't run torch (so a local-CPU text goal never proposes it).
            ms.append(Move(
                "deep_model",
                [("torch_mlp|tfidf_h128x64", _torch_mlp_ctor("128x64", 60, 1e-3, 0.2),
                  {"hidden": "128x64", "epochs": 60, "lr": 0.001, "dropout": 0.2, "weight_decay": 1e-4})],
                prior_gain=0.07, prior_cost=2.5))
        return ms


class TabularRegressionHarness(Harness):
    kind, task_type, default_metric = "tabular", "regression", "r2"

    def featurizer(self):
        return TabularFeaturizer()

    def moves(self, seeds, provider=None):
        return [
            Move("baseline", [("ridge", lambda s: Ridge(alpha=1.0), {"alpha": 1.0})],
                 prior_gain=0.0, prior_cost=0.3),
            Move("stronger_model",
                 [("random_forest_reg|200", lambda s: RandomForestRegressor(n_estimators=200, random_state=s), {"n": 200}),
                  ("hist_gbm_reg|200", lambda s: HistGradientBoostingRegressor(max_iter=200, random_state=s), {"it": 200})],
                 prior_gain=0.15, prior_cost=1.5),
            Move("more_capacity",
                 [("random_forest_reg|400", lambda s: RandomForestRegressor(n_estimators=400, random_state=s), {"n": 400})],
                 prior_gain=0.04, prior_cost=2.0),
        ]


SUPPORTED_KINDS = ("tabular", "text", "vision")
# only these (kind, task_type) combinations have a BUILT featurizer+harness+certifier path.
SUPPORTED_TASK_TYPES = ("binary", "multiclass", "regression")


class UnsupportedSpec(Exception):
    """No built harness for this (kind, task_type). Raised fail-closed so an unsupported modality/task
    is NEVER silently coerced into tabular accuracy (which would emit a certificate for the wrong problem)."""


def is_supported(kind, task_type):
    """tabular supports binary/multiclass/regression; text supports binary/multiclass; timeseries supports
    forecast (block-bootstrap vertical) and ranking supports ranking (per-query-bootstrap vertical) -- these
    two route to their own modules (vfplatform/timeseries.py, vfplatform/ranking.py), NOT run_goal_loop,
    because their data models are grouped/temporal. Everything else (vision/audio kinds; multilabel) is
    unsupported and declined honestly upstream."""
    if kind == "tabular":
        return task_type in ("binary", "multiclass", "regression")
    if kind == "text":
        return task_type in ("binary", "multiclass")
    if kind == "timeseries":
        return task_type == "forecast"
    if kind == "ranking":
        return task_type == "ranking"
    if kind == "vision":
        return task_type in ("binary", "multiclass")    # image classification, reuses the frozen clf certifier
    return False


def harness_for(kind, task_type, text_key="text"):
    if not is_supported(kind, task_type):
        raise UnsupportedSpec(
            f"no built harness for kind={kind!r}, task_type={task_type!r}. Supported: "
            f"tabular[binary|multiclass|regression], text[binary|multiclass]. Refusing to run it as "
            f"tabular accuracy -- that would certify the wrong problem.")
    if kind == "vision":
        from .vision import VisionClassificationHarness   # lazy: vision.py imports from harness (avoid a cycle)
        return VisionClassificationHarness(task_type=task_type)
    if kind == "text":
        return TextClassificationHarness(text_key=text_key)
    if task_type == "regression":
        return TabularRegressionHarness()
    return TabularClassificationHarness()
