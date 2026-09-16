"""CODE AUTHORERS -- the `code_authorer` hook the regenerative generator calls to author NOVEL source.

WHY THIS EXISTS
---------------
recipe_generator.RecipeGenerator carries an OPTIONAL `code_authorer: Callable[[role, task_shape], src|None]`
and a `code_prob`. When set, the generator mutates a recipe to carry an LLM/template-authored `code_patch`
(a brand-new featurizer or estimator) -- which the arena then EXECUTES (authoring_bridge.run_authored_predict)
and the FROZEN certifier judges. This module supplies two interchangeable authorers behind that one hook:

  * LLMAuthorer    -- OPEN-ENDED. Wraps the existing audited authoring.author_estimator (Claude via the
                      ops.llm_propose harness): the model invents a self-contained sklearn-compatible
                      estimator/featurizer as Python source. NON-BINDING -- the frozen three-stage admission
                      gate (static AST -> spawned rlimited child -> scientific self-test) rejects anything
                      unsafe/non-conforming, and only the frozen certifier decides if it actually works. This
                      is the path that makes discovery genuinely menu-free at the CODE axis: the source was
                      never handed to the system.
  * TemplateAuthorer -- DETERMINISTIC + OFFLINE. A small library of safe, parametrised source templates
                      (kernel lifts, polynomial-interaction featurizers, tree ensembles, prototype/kNN).
                      Used as the reproducible DEFAULT (no network, no API key, byte-stable) and by the
                      hermetic tests. HONESTY: a template library is a code GENERATOR, not unbounded
                      invention; it is the deterministic substitute (mirroring authoring.py's honest DECLINE
                      philosophy), NOT the open-ended claim. The open-ended claim rides on LLMAuthorer.

Both return Python SOURCE (a `def build_estimator(seed): ...`) or None; neither evaluates or promotes.
"""
from __future__ import annotations

import os
import random
from typing import Callable, List, Optional

CodeAuthorer = Callable[[str, str], Optional[str]]


# ============================================================================ deterministic template library
# Every template defines `build_estimator(seed)` taking EXACTLY one arg and threads `seed` into every
# random_state, and imports ONLY from the estimator allowlist (numpy / math / allowed sklearn submodules) so
# it clears authoring.estimator_static_check. Featurizers expose fit/transform (the sandbox pairs them with a
# frozen linear head); classifiers expose fit/predict.

_FEATURIZER_TEMPLATES: List[str] = [
    # RBF random-features lift: standardize, then project into a randomized cos-feature space (Rahimi & Recht).
    '''import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


class _RbfLift:
    def __init__(self, seed, gamma, n_components):
        self.scaler = StandardScaler()
        self.rbf = RBFSampler(gamma=gamma, n_components=n_components, random_state=seed)

    def fit(self, X, y=None):
        self.rbf.fit(self.scaler.fit_transform(np.asarray(X, dtype=float)))
        return self

    def transform(self, X):
        return self.rbf.transform(self.scaler.transform(np.asarray(X, dtype=float)))


def build_estimator(seed):
    return _RbfLift(seed, gamma={gamma}, n_components={ncomp})
''',
    # Nystroem kernel-feature map: a low-rank data-dependent kernel approximation, then a linear head.
    '''import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem


class _NystroemMap:
    def __init__(self, seed, gamma, n_components):
        self.scaler = StandardScaler()
        self.ny = Nystroem(gamma=gamma, n_components=n_components, random_state=seed)

    def fit(self, X, y=None):
        self.ny.fit(self.scaler.fit_transform(np.asarray(X, dtype=float)))
        return self

    def transform(self, X):
        return self.ny.transform(self.scaler.transform(np.asarray(X, dtype=float)))


def build_estimator(seed):
    return _NystroemMap(seed, gamma={gamma}, n_components={ncomp})
''',
    # Pairwise interaction featurizer: standardized columns + their degree-2 interaction terms.
    '''import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


class _Interactions:
    def __init__(self, seed):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, interaction_only={ionly}, include_bias=False)

    def fit(self, X, y=None):
        self.poly.fit(self.scaler.fit_transform(np.asarray(X, dtype=float)))
        return self

    def transform(self, X):
        return self.poly.transform(self.scaler.transform(np.asarray(X, dtype=float)))


def build_estimator(seed):
    return _Interactions(seed)
''',
]

_CLASSIFIER_TEMPLATES: List[str] = [
    # Bagged axis-aligned trees: the textbook nonlinear lever on raw tabular features.
    '''from sklearn.ensemble import RandomForestClassifier


def build_estimator(seed):
    return RandomForestClassifier(n_estimators={ntrees}, max_features="sqrt",
                                  min_samples_leaf={leaf}, random_state=seed, n_jobs=1)
''',
    # Extremely randomized trees: extra variance reduction via random split thresholds.
    '''from sklearn.ensemble import ExtraTreesClassifier


def build_estimator(seed):
    return ExtraTreesClassifier(n_estimators={ntrees}, max_features="sqrt",
                                min_samples_leaf={leaf}, random_state=seed, n_jobs=1)
''',
    # Standardized RBF-kernel SVM: a smooth nonlinear decision surface.
    '''import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def build_estimator(seed):
    return make_pipeline(StandardScaler(), SVC(C={c}, gamma="scale", random_state=seed))
''',
    # Prototype / distance classifier on standardized features.
    '''import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def build_estimator(seed):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors={k}))
''',
]

_REGRESSOR_TEMPLATES: List[str] = [
    # Bagged regression trees: the textbook nonlinear lever on raw tabular features (continuous target).
    '''from sklearn.ensemble import RandomForestRegressor


def build_estimator(seed):
    return RandomForestRegressor(n_estimators={ntrees}, max_features="sqrt",
                                 min_samples_leaf={leaf}, random_state=seed, n_jobs=1)
''',
    # Extremely randomized regression trees: extra variance reduction via random split thresholds.
    '''from sklearn.ensemble import ExtraTreesRegressor


def build_estimator(seed):
    return ExtraTreesRegressor(n_estimators={ntrees}, max_features="sqrt",
                               min_samples_leaf={leaf}, random_state=seed, n_jobs=1)
''',
    # Nystroem kernel feature-map + ridge: a smooth nonlinear regressor (kernel-ridge approximation).
    '''import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def build_estimator(seed):
    return make_pipeline(StandardScaler(),
                         Nystroem(gamma={gamma}, n_components={ncomp}, random_state=seed),
                         Ridge(alpha={alpha}))
''',
    # Prototype / distance regressor on standardized features (distance-weighted kNN).
    '''import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def build_estimator(seed):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors={k}, weights="distance"))
''',
]


class TemplateAuthorer:
    """Deterministic, offline CodeAuthorer. Cycles a small library of safe source templates, filling in
    seed-varied hyper-parameters, so each call returns DISTINCT, sandbox-admissible source. Reproducible
    (byte-stable given the construction seed). This is the default authorer for tests + offline runs."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._fi = 0
        self._ci = 0
        self._ri = 0

    def _featurizer(self) -> str:
        tpl = _FEATURIZER_TEMPLATES[self._fi % len(_FEATURIZER_TEMPLATES)]
        self._fi += 1
        gamma = round(self.rng.choice([0.01, 0.02, 0.05, 0.1]), 4)
        ncomp = self.rng.choice([200, 300, 500])
        ionly = self.rng.choice(["True", "False"])
        return tpl.format(gamma=gamma, ncomp=ncomp, ionly=ionly)

    def _classifier(self) -> str:
        tpl = _CLASSIFIER_TEMPLATES[self._ci % len(_CLASSIFIER_TEMPLATES)]
        self._ci += 1
        ntrees = self.rng.choice([200, 300, 400])
        leaf = self.rng.choice([1, 2, 4])
        c = round(self.rng.choice([1.0, 3.0, 5.0]), 2)
        k = self.rng.choice([10, 15, 25])
        return tpl.format(ntrees=ntrees, leaf=leaf, c=c, k=k)

    def _regressor(self) -> str:
        tpl = _REGRESSOR_TEMPLATES[self._ri % len(_REGRESSOR_TEMPLATES)]
        self._ri += 1
        ntrees = self.rng.choice([200, 300, 400])
        leaf = self.rng.choice([1, 2, 4])
        gamma = round(self.rng.choice([0.01, 0.02, 0.05, 0.1]), 4)
        ncomp = self.rng.choice([200, 300, 500])
        alpha = round(self.rng.choice([0.1, 1.0, 3.0]), 2)
        k = self.rng.choice([10, 15, 25])
        return tpl.format(ntrees=ntrees, leaf=leaf, gamma=gamma, ncomp=ncomp, alpha=alpha, k=k)

    def __call__(self, role: str, task_shape: str) -> Optional[str]:
        if role == "classifier":
            return self._classifier()
        if role == "regressor":
            return self._regressor()
        return self._featurizer()


class LLMAuthorer:
    """Open-ended CodeAuthorer backed by Claude through the audited authoring.author_estimator harness. The
    model invents self-contained sklearn-compatible source; the FROZEN three-stage admission gate rejects
    anything unsafe/non-conforming, and only the frozen certifier decides quality. Returns admitted source
    or None on an honest DECLINE (then the generator simply proposes non-code recipes that round)."""

    def __init__(self, *, n_features: int = 16, n_classes: int = 2, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: float = 120.0, cache_path: Optional[str] = None):
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("ATTESTRA_CODE_MODEL")
        self.timeout = float(timeout)
        self.cache_path = cache_path
        self.calls: List[dict] = []                  # provenance: every author attempt (for the certificate)
        self._failures: dict = {}                    # role -> [prior admission-failure reasons] (revise loop)

    def __call__(self, role: str, task_shape: str) -> Optional[str]:
        from . import authoring as A
        nc = self.n_classes if role == "classifier" else 1
        spec = A.EstimatorSpec(role=role, n_features=self.n_features, n_classes=nc)
        # Feed the model its OWN prior admission-failure reasons (trusted, instruction-side) so it can REVISE --
        # and so each attempt is cache-distinct (identical context would otherwise replay the same rejected
        # code). This is the "LLM interprets the rejection log and tries again" loop; it cannot relax the gate.
        prior = self._failures.get(role, [])
        extra = {"attempt": len(prior) + 1, "task_shape": task_shape}
        if prior:
            extra["prior_admission_failures"] = prior[-3:]
        kwargs = dict(api_key=self.api_key, use_llm=True, timeout=self.timeout, cache_path=self.cache_path,
                      extra_context=extra)
        if self.model:
            kwargs["model"] = self.model
        res = A.author_estimator(spec, **kwargs)
        self.calls.append({"role": role, "task_shape": task_shape, "attempt": extra["attempt"],
                           "authored": bool(res.authored), "used_llm": bool(res.used_llm), "reason": res.reason})
        if res.authored and res.estimator is not None:
            return res.estimator.code
        self._failures.setdefault(role, []).append(res.reason)   # remember -> revise next time
        return None


def make_authorer(*, prefer_llm: Optional[bool] = None, n_features: int = 16, n_classes: int = 2,
                  seed: int = 0) -> CodeAuthorer:
    """Pick the authorer. Default: LLM if ATTESTRA_CODE_LLM=1 (or prefer_llm=True) AND a key is present,
    else the deterministic TemplateAuthorer. Keeping the template path the default makes runs/tests
    reproducible and offline; the LLM path is the open-ended claim and is opted into explicitly."""
    if prefer_llm is None:
        prefer_llm = os.environ.get("ATTESTRA_CODE_LLM", "0") == "1"
    if prefer_llm and os.environ.get("ANTHROPIC_API_KEY"):
        return LLMAuthorer(n_features=n_features, n_classes=n_classes)
    return TemplateAuthorer(seed=seed)


__all__ = ["CodeAuthorer", "TemplateAuthorer", "LLMAuthorer", "make_authorer"]
