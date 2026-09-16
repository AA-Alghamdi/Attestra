"""End-to-end locks for the AUTONOMOUS /goal FRONT DOOR (vfplatform/goal_solver.py).

`solve(goal_text, data)` is the single entrypoint that turns a free-text goal + arbitrary data into a
GoalCertificate. These tests lock the contract that makes that journey honest -- all offline, deterministic,
and WITHOUT touching the frozen certifier core (the run-time hashes are asserted byte-identical at the end):

  * ACQUISITION -- a data pointer (inline arrays / dict / .npz / .csv / a bundled sklearn name) resolves to a
    dense (X, y, n_classes) with regression vs classification decided from the target, string labels encoded.
  * SPEC INFERENCE -- the certifiable spec (task_type / arena_shape / split / metric) is inferred
    deterministically; an out-of-scope goal (forecasting, ranking) is DECLINED, never coerced into a fake fit.
  * COMPETENCE FLOOR -- theta is set strictly ABOVE the meta-certifier's own trivial baseline, so an
    imbalanced framing the referee would reject as gameable is instead given an honest, non-trivial bar.
  * CERTIFY / IMPROVE -- on a real signal the front door returns a model whose pooled sealed Clopper-Pearson
    lower bound clears the floor (solved), and the regenerative loop can PROMOTE a champion past the seed.
  * NUMERIC FIREWALL -- the certificate's headline bounds are recomputed by the NumericSubstrate from the
    frozen core (single source of truth), and a live adversarial LLM number is recomputed and REFUSED.
"""
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import goal_solver as G  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_EXPECTED = {
    "vectorforge/science.py": "b564fba248ea009495fcfc2cb7e2e14a1d1285c2f46165cd9401dfbcd461270d",
    "vfplatform/sealed.py": "30ad62450c53cc4a52ed55e65593352c146549d24e756dcb38532569007c8661",
}


def _sha256(path):
    with open(os.path.join(_ROOT, path), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ===================================================================== acquisition
def test_acquire_inline_arrays_and_dict_agree():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1])
    Xa, ya, nc, _ = G.acquire((X, y))
    Xb, yb, ncb, _ = G.acquire({"X": X, "y": y})
    assert nc == ncb == 2 and Xa.shape == (6, 2)
    assert np.array_equal(ya, yb)


def test_acquire_encodes_string_labels_to_contiguous_ints():
    X = np.zeros((4, 3))
    y = np.array(["cat", "dog", "cat", "dog"])
    _, ye, nc, _ = G.acquire((X, y))
    assert nc == 2 and set(ye.tolist()) == {0, 1} and ye.dtype.kind in "iu"


def test_acquire_detects_regression_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = rng.normal(size=80)                       # continuous, many distinct -> regression
    _, _, nc, _ = G.acquire((X, y))
    assert nc == 0


def test_acquire_npz_roundtrip(tmp_path):
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = (X[:, 0] > 8).astype(int)
    p = tmp_path / "d.npz"
    np.savez(p, X=X, y=y, n_classes=2, task_hint="unit")
    Xz, yz, nc, th = G.acquire(str(p))
    assert nc == 2 and th == "unit" and np.array_equal(yz, y)


def test_acquire_csv_uses_last_column_as_target(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 1.0, 0.0, 0.0], "label": [0, 0, 1, 1]})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    X, y, nc, _ = G.acquire(str(p))
    assert X.shape == (4, 2) and nc == 2 and y.tolist() == [0, 0, 1, 1]


def test_acquire_bundled_sklearn_name():
    X, y, nc, _ = G.acquire("wine")
    assert X.shape[0] == len(y) == 178 and nc == 3


# ===================================================================== spec inference
def test_infer_spec_balanced_multiclass():
    X, y, nc, _ = G.acquire("wine")
    spec = G.infer_spec(X, y, "classify the wine cultivar", n_classes=nc)
    assert spec.supported and spec.kind == "tabular" and spec.task_type == "multiclass"
    assert spec.arena_shape == "multiclass" and spec.split == "random"
    assert spec.metric in ("accuracy", "macro_f1")


def test_infer_spec_regression_routes_to_r2():
    X, y, nc, _ = G.acquire("diabetes")
    spec = G.infer_spec(X, y, "predict disease progression", n_classes=nc)
    assert spec.supported and spec.task_type == "regression"
    assert spec.arena_shape == "regression" and spec.metric == "r2"


def test_infer_spec_imbalanced_shape_from_majority():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = np.array([0] * 300 + [1] * 100)           # 75% majority -> imbalanced shape
    spec = G.infer_spec(X, y, "detect the rare positive", n_classes=2)
    assert spec.supported and spec.arena_shape == "imbalanced"
    assert spec.majority_fraction == pytest.approx(0.75)


def test_infer_spec_declines_forecasting():
    X, y, nc, _ = G.acquire("wine")
    spec = G.infer_spec(X, y, "forecast the next 7 days of demand", n_classes=nc)
    assert not spec.supported and spec.decline_reason


def test_infer_spec_declines_ranking():
    X, y, nc, _ = G.acquire("wine")
    spec = G.infer_spec(X, y, "learn to rank documents by relevance", n_classes=nc)
    assert not spec.supported and "ranking" in (spec.decline_reason or "")


def test_infer_spec_grouped_split_when_groups_supplied():
    X, y, nc, _ = G.acquire("wine")
    groups = np.arange(len(y)) % 7
    spec = G.infer_spec(X, y, "classify cultivar, generalize across batches", n_classes=nc, groups=groups)
    assert spec.split == "grouped"


# ===================================================================== end-to-end certify
def test_solve_declines_out_of_scope_without_running():
    X, y, _, _ = G.acquire("wine")
    cert = G.solve("forecast next quarter revenue", (X, y), peeks=4, seed=0)
    assert cert.declined and not cert.solved and not cert.refused
    assert cert.decline_reason and cert.champion == "" and cert.peeks_used == 0


def test_solve_certifies_a_real_signal_offline():
    """The headline end-to-end: a real balanced dataset is acquired, framed, run through the regenerative
    loop, and a model is CERTIFIED above the competence floor -- with the numeric audit clean and the
    firewall active. Deterministic + offline."""
    cert = G.solve("classify the wine cultivar from chemical measurements", "wine",
                   peeks=8, seed=0, use_literature=False)
    assert not cert.declined and not cert.refused
    assert cert.solved and cert.pooled_sealed_lb > cert.theta_floor
    assert cert.champion and cert.sealed_acc
    # numeric substrate: the certificate's bounds came from the frozen core, and the firewall is live.
    na = cert.numeric_audit
    assert na["clean"] and na["firewall_held"]
    assert na["single_source_of_truth"]["agreement"]
    assert na["firewall_selftest"]["verdict"] == "contradicted"
    assert na["firewall_selftest"]["refused_by_firewall"]


def test_solve_competence_floor_clears_trivial_baseline_on_imbalanced():
    """An imbalanced binary problem whose train-majority class scores high on the eval rows: the front door
    must set theta ABOVE that trivial baseline so the meta-certifier does not reject the framing as gameable
    (the bug a train-only majority floor would hit)."""
    cert = G.solve("detect malignant tumors from cell measurements", "breast_cancer",
                   peeks=8, seed=0, use_literature=False)
    assert not cert.declined
    assert not cert.refused                       # theta cleared the trivial baseline -> framing accepted
    assert cert.theta_floor > 0.5                 # data-driven, above the balanced default


def test_solve_promotes_a_champion_past_the_seed():
    """The regenerative loop, driven entirely through the front door, lifts the champion beyond the weak
    linear seed on a signal a linear model cannot fully capture (a real, certified improvement)."""
    X, y = _nonlinear_3class(n=1500, seed=0)
    cert = G.solve("classify the three-cluster nonlinear signal", (X, y, 3, "synthetic_signal"),
                   peeks=16, seed=0, code=False, use_literature=False)
    assert not cert.declined and not cert.refused
    assert cert.solved and cert.improved
    assert cert.champion != "raw · linear_probe · head=linear"
    assert cert.numeric_audit["single_source_of_truth"]["agreement"]


def test_solve_is_deterministic():
    a = G.solve("classify the iris species", "iris", peeks=6, seed=0, use_literature=False)
    b = G.solve("classify the iris species", "iris", peeks=6, seed=0, use_literature=False)
    assert a.champion == b.champion
    assert a.sealed_acc == b.sealed_acc
    assert a.pooled_sealed_lb == b.pooled_sealed_lb


def test_solve_certificate_is_json_serializable():
    import json
    cert = G.solve("classify the iris species", "iris", peeks=6, seed=0, use_literature=False)
    blob = json.dumps(cert.to_dict())             # must round-trip (a deliverable artifact)
    assert json.loads(blob)["spec"]["task_type"] == "multiclass"


# ===================================================================== frozen core untouched
def test_frozen_core_is_byte_identical_after_a_full_solve():
    G.solve("classify the iris species", "iris", peeks=6, seed=0, use_literature=False)
    for path, want in FROZEN_EXPECTED.items():
        assert _sha256(path) == want, f"{path} changed -- the frozen certifier core must never be modified"


# ===================================================================== helpers
def _nonlinear_3class(*, n, seed):
    from sklearn.datasets import make_classification
    return make_classification(n_samples=n, n_features=16, n_informative=6, n_redundant=2, n_classes=3,
                               n_clusters_per_class=2, class_sep=0.6, flip_y=0.02, random_state=seed)
