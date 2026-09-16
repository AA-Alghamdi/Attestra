"""Phase 7 verification-beyond-statistics tests.

Asserts the load-bearing property: a leaky/cheating Program that would pass the SEALED CERTIFICATE
is still CAUGHT and refused by the oracle battery, while a clean winner promotes. Run standalone:
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_oracles.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import certify, oracles, sandbox
from frontier.program import Program
from frontier.task import Task


# --------------------------------------------------------------------------- helpers

def _run_fn(task):
    """The integrator-supplied run_fn: predictions-only over the real sandbox (firewall)."""
    def run_fn(prog, Xtr, ytr, Xev):
        r = sandbox.run_program(prog, Xtr, ytr, Xev, kind=task.kind, wall_seconds=60, cpu_seconds=55)
        return r.preds if r.ok else None
    return run_fn


def _certify(task, splits, prog):
    Xse = Task.rows_to_X(splits.sealed_rows)
    Xtr = Task.rows_to_X(splits.train_rows)
    ytr = Task.rows_to_y(splits.train_rows, task.kind)
    r = sandbox.run_program(prog, Xtr, ytr, Xse, kind=task.kind, wall_seconds=60)
    assert r.ok, f"winner must run on sealed: {r.error_kind} {r.error}"
    return certify.certify_on_sealed(task, splits, r.preds)


_LOGREG = ("from sklearn.preprocessing import StandardScaler\n"
           "from sklearn.pipeline import Pipeline\n"
           "from sklearn.linear_model import LogisticRegression\n"
           "def build_estimator():\n"
           "    return Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=2000))])\n")


# --------------------------------------------------------------------------- clean winner passes

def test_clean_winner_promotes():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    task = Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.85, name="bc")
    splits = certify.make_splits(task, seed=0)
    prog = Program(code=_LOGREG, source="seed", label="scale+logreg")
    cert = _certify(task, splits, prog)
    assert cert["certified"], f"breast-cancer logreg should certify (lb={cert['lower_bound']})"

    v = oracles.verify_before_promote(task, splits, prog, cert, _run_fn(task), seed=0)
    failed = [o.name for o in v.oracles if o.severity == "block" and not o.passed]
    assert v.promote, f"clean winner must promote; blocking failures={failed}; reasons={v.reasons}"
    # spot-check the individual oracles agree it is clean
    by = {o.name: o for o in v.oracles}
    assert by["no_label_leak_feature"].passed
    assert by["permuted_label_collapses"].passed
    assert by["beats_trivial_baseline"].passed
    assert by["reproducible"].passed
    assert by["adversarial_self_refutation"].passed
    print(f"[ok] clean winner promotes (lb={cert['lower_bound']}); all blocking oracles green")


# --------------------------------------------------------------------------- leaky winner refused

def _leaky_task(seed=0):
    """A classification task whose LAST feature is a copy of the label (the textbook leak).

    The leaked column makes the problem trivially separable in EVERY split, so the sealed
    certificate is high and SOUND -- which is exactly why statistics alone cannot catch it."""
    rng = np.random.default_rng(seed)
    n = 400
    y = rng.integers(0, 2, size=n)
    # 4 weakly-informative real features + 1 leaked copy of the label as the last column.
    X_real = rng.normal(size=(n, 4)) + 0.15 * y[:, None]
    leak = y.astype(float)[:, None]
    X = np.hstack([X_real, leak])
    return Task(X=X, y=y.astype(str), kind="classification", theta=0.70, name="leaky")


def test_leaky_feature_is_caught():
    task = _leaky_task(seed=0)
    splits = certify.make_splits(task, seed=0)
    # a plain model trivially exploits the leaked column -> sealed certificate PASSES.
    prog = Program(code=_LOGREG, source="seed", label="scale+logreg")
    cert = _certify(task, splits, prog)
    assert cert["certified"], "the leaky pipeline should (wrongly) clear the sealed certificate"
    assert cert["observed"] >= 0.95, f"leak should give near-perfect observed, got {cert['observed']}"

    v = oracles.verify_before_promote(task, splits, prog, cert, _run_fn(task), seed=0)
    assert not v.promote, "a label-leaking winner MUST be refused despite a passing certificate"
    by = {o.name: o for o in v.oracles}
    # the direct artifact scan must flag the leaked column...
    assert not by["no_label_leak_feature"].passed, "feature-leak scan must catch the copied label"
    assert any("4" in str(f[0]) or f[0] == 4 for f in
               by["no_label_leak_feature"].evidence["leaky_features"]), \
        f"should name the leaked feature index 4: {by['no_label_leak_feature'].evidence}"
    # ...and the adversarial pass must independently explain the win as a single-column artifact.
    assert not by["adversarial_self_refutation"].passed, "self-refutation should explain the leak away"
    print(f"[ok] leaky feature caught: cert observed={cert['observed']} certified={cert['certified']} "
          f"-> promote={v.promote}; reasons={v.reasons[:2]}")


def test_row_shuffle_attack_independently_flags_leak():
    """Independent of the feature scan, the adversarial row-shuffle attack must flag the leak.

    Permuting TRAIN labels (the permutation oracle) does NOT recover a feature-encodes-label leak:
    once the training y is shuffled the leaked column no longer correlates with the labels the
    model is trained on, so a permuted-label run correctly COLLAPSES. The leak is instead caught by
    breaking the X<->y pairing while KEEPING the real labels (row-shuffle attack): the leaked
    column still equals the true label, so the refit model keeps predicting well -- the tell that
    it reads a feature, not a learned pairing."""
    task = _leaky_task(seed=1)
    splits = certify.make_splits(task, seed=0)
    prog = Program(code=_LOGREG, source="seed", label="scale+logreg")
    # the permutation oracle should report a clean collapse (it is not the right tool for this leak)
    perm = oracles.oracle_permuted_label_collapses(task, splits, prog, _run_fn(task), seed=0)
    assert perm.passed, f"permuted-label run collapses for a feature-encode leak: {perm.detail}"
    # the adversarial refutation (row-shuffle + single-feature) IS the right tool and must refute
    cert = _certify(task, splits, prog)
    refuted, reasons, _ = oracles.adversarial_self_refutation(
        task, splits, prog, cert, _run_fn(task), seed=0)
    assert refuted, f"adversarial pass must explain the leak away: {reasons}"
    assert any("row-shuffle" in r or "single-feature" in r for r in reasons)
    print(f"[ok] adversarial pass independently flags leak: {reasons[0][:90]}")


# --------------------------------------------------------------------------- baseline / orientation

def test_trivial_baseline_oracle_blocks_no_skill_win():
    """A 'winner' that only matches the majority rate must NOT beat the trivial baseline."""
    # 90% one class; a constant-majority predictor would 'certify' at theta below 0.90.
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.90).astype(int)
    X = rng.normal(size=(n, 5))               # features carry NO signal
    task = Task(X=X, y=y.astype(str), kind="classification", theta=0.80, name="imbalanced")
    splits = certify.make_splits(task, seed=0)
    # a constant-majority "model"
    const_code = ("import numpy as np\n"
                  "from sklearn.base import BaseEstimator, ClassifierMixin\n"
                  "class Const(BaseEstimator, ClassifierMixin):\n"
                  "    def fit(self, X, y):\n"
                  "        v, c = np.unique(y, return_counts=True); self.maj_ = v[c.argmax()]; return self\n"
                  "    def predict(self, X):\n"
                  "        import numpy as np; return np.array([self.maj_]*len(X))\n"
                  "def build_estimator():\n    return Const()\n")
    prog = Program(code=const_code, source="seed", label="const_majority")
    cert = _certify(task, splits, prog)
    res = oracles.oracle_beats_trivial(task, cert, Task.rows_to_y(splits.sealed_rows, task.kind))
    assert not res.passed, f"a majority-rate win must not beat the trivial baseline: {res.detail}"
    print(f"[ok] trivial-baseline oracle blocks no-skill win: {res.detail}")


def test_metric_orientation_oracle():
    task = Task(X=np.zeros((10, 3)), y=np.zeros(10), kind="regression", theta=0.5, metric="r2")
    good = {"metric": "r2", "certified": True, "lower_bound": 0.7, "observed": 0.8}
    assert oracles.oracle_metric_orientation(task, good).passed
    # an unknown / wrong-axis metric in the certificate must be refused
    bad = {"metric": "mse", "certified": True, "lower_bound": 0.7, "observed": 0.8}
    assert not oracles.oracle_metric_orientation(task, bad).passed
    print("[ok] metric-orientation oracle: r2 ok, mse refused")


def test_reproducibility_same_seed_same_digest():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    task = Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.85, name="bc")
    splits = certify.make_splits(task, seed=0)
    prog = Program(code=_LOGREG, source="seed", label="scale+logreg")
    cert = _certify(task, splits, prog)
    res = oracles.oracle_reproducible(task, prog, _run_fn(task), cert, seed=0)
    assert res.passed, f"same seed must reproduce the sealed digest + decision: {res.detail}"
    assert res.evidence["digest_match"] and res.evidence["decision_match"]
    print(f"[ok] reproducibility: {res.detail}")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} oracle tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
