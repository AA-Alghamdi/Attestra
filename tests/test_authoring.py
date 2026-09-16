"""Offline tests for the Tier-2 sandboxed LLM-code-authoring layer (NEW module vfplatform/authoring.py).

These tests NEVER call a live LLM. They drive the frozen three-stage admission gate -- static AST allowlist,
isolated subprocess execution (multiprocessing spawn + setrlimit + parent wall-clock timeout), and the
scientific self-test (determinism, valid predictions, row-equivariance / no test access) -- against FIXED
authored-code samples:

  GOOD:    a custom feature-interaction featurizer + logistic head (a genuinely novel inductive bias).
  GOOD:    a custom regressor wrapping a ridge over polynomial-ish interactions.

  MALICIOUS / BROKEN (each must be REJECTED with the right reason):
    - import os                          -> static gate (banned import)
    - open(...)                          -> static gate (banned name 'open')
    - sklearn.model_selection            -> static gate (denied sklearn submodule = split peeking)
    - __subclasses__ escape chain        -> static gate (forbidden dunder attribute)
    - non-deterministic (no seeded RNG)  -> self-test (deterministic case fails)
    - infinite while True (no break)     -> static gate (infinite-loop guard)
    - wrong entrypoint arity             -> static gate (build_estimator must take 1 arg)
    - predict returns wrong length       -> self-test (valid_predictions fails)

It also proves admission is DETERMINISTIC (same code admitted twice -> identical verdict + family) and that
an admitted method becomes a Move-compatible candidate via the existing frozen move_from_proposal (no loop
import / edit). Frozen hashes are re-verified at the end.

Run:  PYTHONPATH=. /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_authoring.py
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import authoring
from vfplatform.authoring import EstimatorSpec, admit


# ===================================================================== FIXED authored-code samples

# ---- GOOD: a custom feature-INTERACTION featurizer (pairwise products of the top-variance features) +
#      a frozen logistic head. This is a genuinely different inductive bias from a plain linear/forest
#      baseline: it learns over explicit second-order interactions. Deterministic (seed threaded). ----
GOOD_FEATURIZER = '''
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def build_estimator(seed):
    class InteractionFeaturizer(BaseEstimator, TransformerMixin):
        def __init__(self, k=4, seed=0):
            self.k = k
            self.seed = seed
        def fit(self, X, y=None):
            X = np.asarray(X, dtype=float)
            var = X.var(axis=0)
            order = np.argsort(-var)
            self.top_ = order[: min(self.k, X.shape[1])]
            return self
        def transform(self, X):
            X = np.asarray(X, dtype=float)
            cols = [X]
            t = self.top_
            for i in range(len(t)):
                for j in range(i, len(t)):
                    cols.append((X[:, t[i]] * X[:, t[j]]).reshape(-1, 1))
            return np.hstack(cols)
    return InteractionFeaturizer(k=4, seed=int(seed))
'''

# ---- GOOD: a custom classifier -- a prototype/nearest-centroid-with-shrinkage method. Deterministic
#      (no RNG at all, so seed-in -> output-out trivially holds). fit/predict surface. ----
GOOD_CLASSIFIER = '''
import numpy as np

def build_estimator(seed):
    class ShrunkCentroid:
        def __init__(self, seed=0):
            self.seed = seed
        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y)
            self.classes_ = np.unique(y)
            mu = X.mean(axis=0)
            self.centroids_ = []
            for c in self.classes_:
                cm = X[y == c].mean(axis=0)
                self.centroids_.append(0.7 * cm + 0.3 * mu)   # shrink toward the global mean
            self.centroids_ = np.asarray(self.centroids_)
            return self
        def predict(self, X):
            X = np.asarray(X, dtype=float)
            d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
            return self.classes_[d.argmin(axis=1)]
    return ShrunkCentroid(seed=int(seed))
'''

# ---- GOOD: a custom regressor -- ridge over the feature matrix (deterministic). ----
GOOD_REGRESSOR = '''
import numpy as np
from sklearn.linear_model import Ridge

def build_estimator(seed):
    class InteractionRidge:
        def __init__(self, alpha=1.0, seed=0):
            self.alpha = alpha
            self.seed = seed
            self._m = Ridge(alpha=alpha)
        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            Z = np.hstack([X, X ** 2])
            self._m.fit(Z, y)
            return self
        def predict(self, X):
            X = np.asarray(X, dtype=float)
            return self._m.predict(np.hstack([X, X ** 2]))
    return InteractionRidge(alpha=1.0, seed=int(seed))
'''

# ---- MALICIOUS: imports os ----
MAL_IMPORT_OS = '''
import os
import numpy as np

def build_estimator(seed):
    os.system("echo pwned")
    class M:
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X), dtype=int)
    return M()
'''

# ---- MALICIOUS: opens a file ----
MAL_OPEN = '''
import numpy as np

def build_estimator(seed):
    class M:
        def fit(self, X, y):
            f = open("/etc/passwd")
            self._d = f.read()
            return self
        def predict(self, X): return np.zeros(len(X), dtype=int)
    return M()
'''

# ---- MALICIOUS: peeks at the split via sklearn.model_selection (denied submodule) ----
MAL_MODEL_SELECTION = '''
import numpy as np
from sklearn.model_selection import train_test_split

def build_estimator(seed):
    class M:
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X), dtype=int)
    return M()
'''

# ---- MALICIOUS: dunder-introspection escape chain to reach the interpreter internals ----
MAL_DUNDER = '''
import numpy as np

def build_estimator(seed):
    cls = ().__class__.__bases__[0]
    subs = cls.__subclasses__()
    class M:
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X), dtype=int)
    return M()
'''

# ---- BROKEN: non-deterministic -- draws fresh randomness on every build, ignoring the seed ----
BROKEN_NONDETERMINISTIC = '''
import numpy as np

def build_estimator(seed):
    class M:
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self._noise = np.random.RandomState().randint(0, 1000000)   # NO seed -> nondeterministic
            return self
        def predict(self, X):
            r = np.random.RandomState()                                 # fresh entropy each predict
            return r.randint(0, max(2, len(self.classes_)), size=len(X))
    return M()
'''

# ---- BROKEN: infinite loop (constant-true while with no break) ----
BROKEN_INFINITE_LOOP = '''
import numpy as np

def build_estimator(seed):
    class M:
        def fit(self, X, y):
            while True:
                x = 1 + 1
            return self
        def predict(self, X): return np.zeros(len(X), dtype=int)
    return M()
'''

# ---- BROKEN: memory bomb -- allocates an enormous array (rlimit on Linux / wall-timeout backstop) ----
BROKEN_MEMORY_BOMB = '''
import numpy as np

def build_estimator(seed):
    class M:
        def fit(self, X, y):
            self._bomb = np.ones((100000, 100000), dtype=np.float64)   # ~80 GB
            return self
        def predict(self, X): return np.zeros(len(X), dtype=int)
    return M()
'''

# ---- BROKEN: wrong entrypoint arity (build_estimator must take exactly 1 arg) ----
BROKEN_ARITY = '''
import numpy as np

def build_estimator(seed, extra):
    class M:
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X), dtype=int)
    return M()
'''

# ---- BROKEN: predict returns the wrong length ----
BROKEN_WRONG_LENGTH = '''
import numpy as np

def build_estimator(seed):
    class M:
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self
        def predict(self, X):
            return np.zeros(3, dtype=int)   # always length 3, ignores X
    return M()
'''


CLF_SPEC = EstimatorSpec(role="classifier", n_features=8, n_classes=3)
REG_SPEC = EstimatorSpec(role="regressor", n_features=8, n_classes=1)
FEAT_SPEC = EstimatorSpec(role="featurizer", n_features=8, n_classes=3)


# ===================================================================== test cases
def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_good_featurizer_admitted_and_predicts():
    rep = admit(GOOD_FEATURIZER, FEAT_SPEC, family="interaction_feat",
                param_space={"k": [2, 4, 6]})
    _assert(rep.admitted, f"good featurizer should be admitted; got: {rep.reason}")
    _assert(rep.factory is not None, "admitted featurizer must carry a factory")
    # the factory must build, fit and predict in-process
    ctor = rep.factory.catalog_entry().builder
    est = ctor({}, 0)
    rng = np.random.RandomState(1)
    X = rng.randn(40, 8)
    y = (X[:, :3].argmax(axis=1)).astype(int)
    est.fit(X, y)
    pred = np.asarray(est.predict(X))
    _assert(pred.shape[0] == 40, "featurizer+head predict must return one row per input")
    _assert(np.all(np.isfinite(pred.astype(float))), "predictions must be finite")
    return "good featurizer ADMITTED + fits/predicts"


def test_good_classifier_admitted_and_predicts():
    rep = admit(GOOD_CLASSIFIER, CLF_SPEC, family="shrunk_centroid")
    _assert(rep.admitted, f"good classifier should be admitted; got: {rep.reason}")
    # all four scientific self-test cases must have passed
    cases = {c["name"]: c["ok"] for c in rep.selftest["cases"]}
    for needed in ("valid_predictions", "deterministic", "row_equivariant"):
        _assert(cases.get(needed) is True, f"self-test case {needed!r} must pass: {rep.selftest}")
    ctor = rep.factory.catalog_entry().builder
    est = ctor({}, 7)
    rng = np.random.RandomState(2)
    X = rng.randn(50, 8)
    y = X[:, :3].argmax(axis=1).astype(int)
    est.fit(X, y)
    pred = np.asarray(est.predict(X))
    _assert(pred.shape[0] == 50 and set(np.unique(pred)).issubset(set(np.unique(y))),
            "classifier predictions must be valid class labels")
    return "good classifier ADMITTED + valid/deterministic/row-equivariant"


def test_good_regressor_admitted():
    rep = admit(GOOD_REGRESSOR, REG_SPEC, family="interaction_ridge")
    _assert(rep.admitted, f"good regressor should be admitted; got: {rep.reason}")
    cases = {c["name"]: c["ok"] for c in rep.selftest["cases"]}
    _assert(cases.get("valid_predictions") is True, "regressor must produce finite predictions")
    _assert(cases.get("deterministic") is True, "regressor must be deterministic")
    return "good regressor ADMITTED"


def test_reject_import_os():
    rep = admit(MAL_IMPORT_OS, CLF_SPEC, family="evil")
    _assert(not rep.admitted, "import os must be rejected")
    _assert("static gate" in rep.reason and ("'os'" in rep.reason or "denied identifier 'os'" in rep.reason),
            f"rejection reason must name the os import; got: {rep.reason}")
    return f"import os REJECTED ({rep.reason[:60]}...)"


def test_reject_open():
    rep = admit(MAL_OPEN, CLF_SPEC, family="evil")
    _assert(not rep.admitted, "open(...) must be rejected")
    _assert("static gate" in rep.reason and "open" in rep.reason,
            f"rejection reason must name 'open'; got: {rep.reason}")
    return f"open() REJECTED ({rep.reason[:60]}...)"


def test_reject_numpy_file_io_exfil():
    """Adversarial-pass regression: numpy's C-extension file I/O (np.savetxt/fromfile/genfromtxt/load/memmap)
    needs NONE of the banned names and was used to exfiltrate the .anthropic_key. All must be REJECTED."""
    head = "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\ndef build_estimator(seed):\n    "
    for attr, call in [("savetxt", 'np.savetxt("/tmp/x", np.array([1.0]))'),
                       ("fromfile", 'np.fromfile(".anthropic_key")'),
                       ("genfromtxt", 'np.genfromtxt(".anthropic_key")'),
                       ("load", 'np.load("x.npy")'),
                       ("memmap", 'np.memmap("x", mode="w+", shape=(2,))')]:
        rep = admit(head + call + "\n    return LogisticRegression()", CLF_SPEC, family="exfil")
        _assert(not rep.admitted, f"np.{attr} file I/O must be rejected")
        _assert(attr in rep.reason, f"rejection must name {attr!r}; got: {rep.reason}")
    return "numpy file-I/O exfil vectors (savetxt/fromfile/genfromtxt/load/memmap) all REJECTED"


def test_reject_model_selection():
    rep = admit(MAL_MODEL_SELECTION, CLF_SPEC, family="evil")
    _assert(not rep.admitted, "sklearn.model_selection must be rejected (split peeking)")
    _assert("static gate" in rep.reason and "model_selection" in rep.reason,
            f"rejection reason must name model_selection; got: {rep.reason}")
    return f"sklearn.model_selection REJECTED ({rep.reason[:60]}...)"


def test_reject_dunder_escape():
    rep = admit(MAL_DUNDER, CLF_SPEC, family="evil")
    _assert(not rep.admitted, "dunder introspection escape must be rejected")
    _assert("static gate" in rep.reason and ("__class__" in rep.reason or "__subclasses__" in rep.reason
                                             or "__bases__" in rep.reason),
            f"rejection reason must name a forbidden dunder attribute; got: {rep.reason}")
    return f"dunder escape REJECTED ({rep.reason[:60]}...)"


def test_reject_nondeterministic():
    rep = admit(BROKEN_NONDETERMINISTIC, CLF_SPEC, family="flaky")
    _assert(not rep.admitted, "non-deterministic method must be rejected")
    _assert("self-test" in rep.reason and "deterministic" in rep.reason,
            f"rejection reason must name the determinism failure; got: {rep.reason}")
    return f"non-deterministic REJECTED ({rep.reason[:60]}...)"


def test_reject_infinite_loop():
    rep = admit(BROKEN_INFINITE_LOOP, CLF_SPEC, family="hang")
    _assert(not rep.admitted, "infinite loop must be rejected")
    # caught statically by the constant-true-while guard (does not even reach execution)
    _assert("static gate" in rep.reason and "while" in rep.reason.lower(),
            f"rejection reason must name the infinite-loop guard; got: {rep.reason}")
    return f"infinite loop REJECTED statically ({rep.reason[:60]}...)"


def test_reject_memory_bomb():
    # static-clean (numpy only) -> reaches isolated execution; rlimit (Linux) or wall-timeout (Darwin)
    # backstop must stop it. Tight wall to keep the test fast.
    spec = EstimatorSpec(role="classifier", n_features=8, n_classes=3, mem_mb=256, wall_s=12.0, cpu_s=6)
    rep = admit(BROKEN_MEMORY_BOMB, spec, family="bomb")
    _assert(not rep.admitted, "memory bomb must be rejected")
    _assert("self-test" in rep.reason, f"memory bomb must fail in the isolated self-test; got: {rep.reason}")
    return f"memory bomb REJECTED ({rep.reason[:70]}...)"


def test_reject_wrong_arity():
    rep = admit(BROKEN_ARITY, CLF_SPEC, family="badarity")
    _assert(not rep.admitted, "wrong entrypoint arity must be rejected")
    _assert("static gate" in rep.reason and "exactly 1 arg" in rep.reason,
            f"rejection reason must name the arity violation; got: {rep.reason}")
    return f"wrong arity REJECTED ({rep.reason[:60]}...)"


def test_reject_wrong_length():
    rep = admit(BROKEN_WRONG_LENGTH, CLF_SPEC, family="badlen")
    _assert(not rep.admitted, "wrong-length predict must be rejected")
    _assert("self-test" in rep.reason and "valid_predictions" in rep.reason,
            f"rejection reason must name the prediction-length failure; got: {rep.reason}")
    return f"wrong-length predict REJECTED ({rep.reason[:60]}...)"


def test_admission_deterministic():
    r1 = admit(GOOD_CLASSIFIER, CLF_SPEC, family="shrunk_centroid")
    r2 = admit(GOOD_CLASSIFIER, CLF_SPEC, family="shrunk_centroid")
    _assert(r1.admitted == r2.admitted is True, "both admissions of the same code must succeed")
    _assert(r1.family == r2.family, "admitted family name must be deterministic")
    _assert(r1.code_digest == r2.code_digest, "code digest must be deterministic")
    p1 = [(c["name"], c["ok"]) for c in r1.selftest["cases"]]
    p2 = [(c["name"], c["ok"]) for c in r2.selftest["cases"]]
    _assert(p1 == p2, "self-test verdicts must be identical across runs")
    # a rejection is also deterministic
    j1 = admit(MAL_IMPORT_OS, CLF_SPEC, family="evil")
    j2 = admit(MAL_IMPORT_OS, CLF_SPEC, family="evil")
    _assert((j1.admitted, j1.reason) == (j2.admitted, j2.reason), "rejection must be deterministic")
    return "admission DETERMINISTIC (accept + reject)"


def test_admitted_becomes_move_compatible_candidate():
    """An admitted method must flow through the EXISTING frozen move_from_proposal -- no loop import/edit."""
    from vfplatform.harness import move_from_proposal, _grid_configs, Move
    rep = admit(GOOD_FEATURIZER, FEAT_SPEC, family="interaction_feat", param_space={"k": [2, 4]})
    _assert(rep.admitted, "featurizer must be admitted to test the bridge")
    catalog = {}
    moves = authoring.authored_moves([rep], catalog, limit=4)
    _assert(len(moves) >= 1, "authored_moves must emit at least one Move")
    _assert(all(isinstance(m, Move) for m in moves), "authored_moves must return harness.Move objects")
    _assert(rep.factory.family in catalog, "the authored family must be registered into the catalog")
    # the CatalogEntry must be worker_safe=False (local in-process only)
    _assert(catalog[rep.factory.family].worker_safe is False,
            "authored CatalogEntry must be worker_safe=False (never crosses the worker wire)")
    # the move's ctor must actually build a fit/predict object
    fam_name, ctor, params = moves[0].families[0]
    est = ctor(0)
    rng = np.random.RandomState(3)
    X = rng.randn(30, 8)
    y = X[:, :3].argmax(axis=1).astype(int)
    est.fit(X, y)
    _assert(np.asarray(est.predict(X)).shape[0] == 30, "move ctor must yield a working estimator")
    return f"admitted method -> {len(moves)} Move-compatible candidate(s), worker_safe=False"


def test_no_loop_import():
    """The module must NOT import the search loop (isolation guarantee)."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "vfplatform", "authoring.py")).read()
    _assert("import loop" not in src and "from .loop" not in src and "from vfplatform.loop" not in src,
            "authoring.py must not import loop.py")
    _assert("sealed" not in src.replace("the sealed", "").replace("sealed test", "").replace("sealed-", "")
            .replace("sealed certify", "").replace("in-memory sealed", "") or "import sealed" not in src,
            "authoring.py must not import sealed.py")
    return "authoring.py imports neither loop.py nor sealed.py"


def test_frozen_hashes_unchanged():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected = {
        os.path.join(root, "vectorforge", "science.py"):
            "b564fba248ea009495fcfc2cb7e2e14a1d1285c2f46165cd9401dfbcd461270d",
        os.path.join(root, "vfplatform", "sealed.py"):
            "30ad62450c53cc4a52ed55e65593352c146549d24e756dcb38532569007c8661",
    }
    for path, want in expected.items():
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        _assert(got == want, f"FROZEN FILE CHANGED: {path}\n  expected {want}\n  got      {got}")
    return "frozen hashes UNCHANGED (science.py b564fba2 / sealed.py 30ad6245)"


TESTS = [
    test_good_featurizer_admitted_and_predicts,
    test_good_classifier_admitted_and_predicts,
    test_good_regressor_admitted,
    test_reject_import_os,
    test_reject_open,
    test_reject_model_selection,
    test_reject_dunder_escape,
    test_reject_nondeterministic,
    test_reject_infinite_loop,
    test_reject_memory_bomb,
    test_reject_wrong_arity,
    test_reject_wrong_length,
    test_reject_numpy_file_io_exfil,
    test_admission_deterministic,
    test_admitted_becomes_move_compatible_candidate,
    test_no_loop_import,
    test_frozen_hashes_unchanged,
]


# ---- pytest collection: expose each as a test_* function (they already are) ----
def main():
    passed = failed = 0
    for t in TESTS:
        try:
            msg = t()
            print(f"  PASS  {t.__name__}: {msg}")
            passed += 1
        except Exception as ex:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(ex).__name__}: {ex}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
