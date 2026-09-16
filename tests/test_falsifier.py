"""Tests for the independent adversarial certificate falsifier (scripts/falsifier.py).

The falsifier is the antidote to self-graded acceptance: it re-materializes the data, re-trains the reported
winner on a FRESH leakage-safe split at a DIFFERENT seed, re-scores on a held-out slice, recomputes the lower
bound via the FROZEN science primitives, and tries to make the claim FAIL. These tests prove it:
  * CONFIRMS an honest, well-clear certificate (breast_cancer, real run);
  * REFUTES an over-stated theta on the SAME data (it is not a rubber stamp);
  * REFUTES a fabricated winner family that cannot actually clear the bar;
  * is INDEPENDENT of the loop's decision (it never imports run_goal_loop's accept logic);
  * does NOT touch the frozen science.py / sealed.py fingerprints.
"""
import hashlib
import importlib
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

falsifier = importlib.import_module("falsifier")
from vfplatform.connectors import load_sklearn


# --------------------------------------------------------------------------- shared real certificate fixture
def _real_cert(dataset_name, theta, threshold=None):
    """Run the REAL loop once to produce a genuine certificate, then hand the falsifier ONLY the claim
    fields (winner family, metric, theta, dataset name). The loop is used ONLY to manufacture a realistic
    certificate to test against; the falsifier never sees the loop's decision."""
    from vfplatform.loop import run_goal_loop
    from vfplatform.harness import harness_for
    spec = load_sklearn(dataset_name)
    h = harness_for(spec["kind"], spec["task_type"])
    res = run_goal_loop(
        spec["records"], f"classify {dataset_name}", harness=h, target_key="target",
        labels=spec["labels"], threshold=(threshold if threshold is not None else theta),
        metric="accuracy", seeds=(0, 1), min_test_n=30, seed=0,
        experiment=f"{dataset_name}-falsify-test", objective="certify", max_rounds=4, llm_enabled=False)
    assert res.certificate is not None, f"loop produced no certificate for {dataset_name}"
    cert = dict(res.certificate)
    cert["dataset"] = f"sklearn:{dataset_name}"
    return cert


@pytest.fixture(scope="module")
def bc_cert():
    # breast_cancer at theta=0.92: a real, comfortably-clearing certificate (n_test ~171).
    return _real_cert("breast_cancer", theta=0.92)


# --------------------------------------------------------------------------- CONFIRM an honest certificate
def test_confirms_honest_breast_cancer(bc_cert):
    out = falsifier.falsify_cert(bc_cert, seeds=(101, 202, 303, 404, 505))
    assert out["verdict"] == "CONFIRMED", out["summary"]
    scored = [t for t in out["trials"] if t.get("status") == "scored"]
    assert len(scored) >= 3, "expected most independent seeds to score cleanly"
    # EVERY scored seed must clear theta for a CONFIRMED verdict.
    assert all(t["clears"] for t in scored)
    # The falsifier's independent bound must be a real number, not echoed from the cert.
    assert out["worst"]["lower_bound"] is not None
    assert out["claim"]["winner_family"] == bc_cert["winner_family"]


# --------------------------------------------------------------------------- REFUTE an over-stated theta
def test_refutes_inflated_theta(bc_cert):
    """Same honest model + data, but the certificate CLAIMS theta=0.999 -- a bar the true lower bound cannot
    clear. An independent re-derivation must REFUTE; a rubber stamp would confirm."""
    out = falsifier.falsify_cert(bc_cert, theta=0.999, seeds=(101, 202, 303, 404, 505))
    assert out["verdict"] == "REFUTED", out["summary"]
    scored = [t for t in out["trials"] if t.get("status") == "scored"]
    assert any(not t["clears"] for t in scored)


def test_refutes_modestly_inflated_theta(bc_cert):
    """theta=0.97 sits ABOVE the achievable lower bound (~0.92-0.96 across seeds) but below observed accuracy
    -- the subtle over-claim the falsifier exists to catch. Must REFUTE."""
    out = falsifier.falsify_cert(bc_cert, theta=0.97, seeds=(101, 202, 303, 404, 505))
    assert out["verdict"] == "REFUTED", out["summary"]


# --------------------------------------------------------------------------- REFUTE a fabricated winner
def test_refutes_fabricated_weak_winner(bc_cert):
    """Swap the winner family to a deliberately weak config (knn with k=64) and keep the honest theta. A
    near-degenerate model should not clear a real 0.92 bar on an independent split -> REFUTED or INCONCLUSIVE,
    never CONFIRMED."""
    cert = dict(bc_cert)
    cert["winner_family"] = "knn|k=64"
    out = falsifier.falsify_cert(cert, seeds=(101, 202, 303, 404, 505))
    assert out["verdict"] != "CONFIRMED", out["summary"]


# --------------------------------------------------------------------------- independence / hygiene
def test_does_not_import_loop_decision():
    """The falsifier module must NOT IMPORT the loop's accept/select-then-bound machinery. It re-derives the
    verdict from frozen primitives only. (Importing run_goal_loop/GoalLoopResult/_val_lower_bound would make
    it a re-run of the same decision, not an independent check.) We parse the AST so docstring PROSE that
    merely names those symbols does not trip the check -- only real imports/references do."""
    import ast
    src = open(os.path.join(_SCRIPTS, "falsifier.py")).read()
    tree = ast.parse(src)
    imported_modules, imported_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
            imported_names |= {a.name for a in node.names}
    # the loop module must never be imported, and no loop decision symbol may be imported by name
    assert "vfplatform.loop" not in imported_modules, "falsifier must not import vfplatform.loop"
    for forbidden in ("run_goal_loop", "GoalLoopResult", "_val_lower_bound", "_diagnose"):
        assert forbidden not in imported_names, f"falsifier must not import loop symbol {forbidden!r}"
    # and the loop module is not referenced by attribute access anywhere (e.g. loop.run_goal_loop)
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "loop" not in referenced


def test_uses_different_split_seed_than_loop(bc_cert):
    """The loop's locked test is the deterministic seed=0 split (its sealed_digest is fixed). The falsifier
    must draw DIFFERENT seeds -- otherwise it would REPLAY the loop's split, not independently re-derive."""
    out = falsifier.falsify_cert(bc_cert, seeds=(101, 202))
    seeds = [t["seed"] for t in out["trials"]]
    assert 0 not in seeds, "falsifier must not reuse the loop's seed=0 split (that would be a replay)"


def test_inconclusive_when_dataset_missing():
    """If the certificate names no dataset and none is supplied, the falsifier cannot re-materialize data and
    must raise FalsifierError (honest setup failure), never silently confirm."""
    with pytest.raises(falsifier.FalsifierError):
        falsifier.falsify_cert({"winner_family": "logistic|C=1.0", "metric": "accuracy", "theta": 0.9})


def test_unknown_metric_rejected():
    with pytest.raises(falsifier.FalsifierError):
        falsifier.falsify_cert({"winner_family": "logistic", "metric": "auroc", "theta": 0.9,
                                "dataset": "sklearn:breast_cancer"})


def test_unknown_family_refused(bc_cert):
    cert = dict(bc_cert)
    cert["winner_family"] = "not_a_real_family|x=1"
    with pytest.raises(falsifier.FalsifierError):
        falsifier.falsify_cert(cert, seeds=(101,))


# --------------------------------------------------------------------------- frozen-core integrity
def test_frozen_fingerprints_unchanged():
    """The falsifier is a read-only checker; building it must not have changed the trust core."""
    def sha(p):
        return hashlib.sha256(open(os.path.join(_REPO, p), "rb").read()).hexdigest()
    assert sha("vectorforge/science.py").startswith("b564fba2"), "science.py fingerprint changed!"
    assert sha("vfplatform/sealed.py").startswith("30ad6245"), "sealed.py fingerprint changed!"


# --------------------------------------------------------------------------- CI gate exit codes
def test_exit_codes_map_verdicts():
    assert falsifier._exit_code("CONFIRMED") == 0
    assert falsifier._exit_code("REFUTED") == 1
    assert falsifier._exit_code("INCONCLUSIVE") == 2
