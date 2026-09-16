"""Behavioral tests for the harness fabric core (Phase 3).

Run standalone:
    cd <repo root> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_harness_base.py

These assert the load-bearing properties, not just import:
  - the registry routes by task-type key (and refuses duplicate keys);
  - TabularHarness.adapt() produces a valid Phase-0 Task and rejects bad data;
  - TabularHarness.self_test() certifies the harness through the FROZEN sealed gate
    (and sets trusted=True only on success);
  - a real dataset routed through the harness reaches a certified result via the Phase-0 engine
    with exactly one sealed peek.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier.harness import (
    Harness,
    HarnessCertificate,
    HarnessRegistry,
    TabularHarness,
    lookup,
    REGISTRY,
)
from frontier.engine import ResearchEngine, EngineConfig
from frontier.task import Task


def test_registry_routes_and_refuses_duplicates():
    reg = HarnessRegistry()
    h = TabularHarness()
    reg.register(h, "tabular", "classification")
    assert reg.lookup("tabular") is h
    assert reg.lookup("classification") is h
    assert "tabular" in reg
    assert set(["classification", "tabular"]).issubset(set(reg.keys()))
    # unknown key raises with a helpful message
    raised = False
    try:
        reg.lookup("vision")
    except KeyError as e:
        raised = "vision" in str(e)
    assert raised, "lookup of unknown key must raise KeyError"
    # duplicate key bound to a DIFFERENT harness must raise (no silent shadowing)
    raised = False
    try:
        reg.register(TabularHarness(), "tabular")
    except KeyError:
        raised = True
    assert raised, "duplicate key for a different harness must raise"
    # re-registering the SAME instance under the same key is idempotent (no raise)
    reg.register(h, "tabular")
    print("[ok] registry routes by key, aliases, refuses duplicate shadowing")


def test_default_registry_has_tabular_under_modality_and_kind_keys():
    for k in ("tabular", "classification", "regression"):
        assert k in REGISTRY, f"default registry missing key {k!r}"
    assert lookup("tabular") is lookup("classification") is lookup("regression")
    print(f"[ok] default registry keys: {REGISTRY.keys()}")


def test_adapt_builds_valid_task_and_picks_right_metric():
    h = TabularHarness()
    X = np.random.RandomState(0).randn(40, 5)
    y_clf = (X[:, 0] > 0).astype(int)
    task = h.adapt(X, y_clf, kind="classification", theta=0.6, name="syn")
    assert isinstance(task, Task)
    assert task.kind == "classification" and task.metric == "accuracy"
    assert task.n_features == 5 and task.theta == 0.6
    # labels are sorted string labels
    assert task.labels == sorted({str(v) for v in y_clf})
    # 1D X is promoted to a column
    t1 = h.adapt(X[:, 0], y_clf, kind="classification", theta=0.5)
    assert t1.n_features == 1
    # regression default metric is r2
    y_reg = X[:, 0] * 2.0 + X[:, 1]
    treg = h.adapt(X, y_reg, kind="regression", theta=0.5)
    assert treg.kind == "regression" and treg.metric == "r2"
    print("[ok] adapt builds valid Task, right-axis metric, 1D->column")


def test_adapt_rejects_bad_data_and_wrong_axis_metric():
    h = TabularHarness()
    X = np.zeros((10, 3))
    y = np.arange(10)
    # non-finite features rejected
    Xbad = X.copy(); Xbad[0, 0] = np.nan
    for bad_call, why in [
        (lambda: h.adapt(Xbad, y[:10].astype(str), kind="classification", theta=0.5), "NaN features"),
        (lambda: h.adapt(X, y[:5], kind="regression", theta=0.5), "X/y mismatch"),
        (lambda: h.adapt(X, y, kind="ranking", theta=0.5), "unknown kind"),
        (lambda: h.adapt(X, y, kind="regression", theta=0.5, metric="accuracy"), "wrong-axis metric"),
    ]:
        raised = False
        try:
            bad_call()
        except (ValueError, KeyError):
            raised = True
        assert raised, f"adapt must reject: {why}"
    print("[ok] adapt rejects NaN, mismatch, unknown kind, wrong-axis metric")


def test_baseline_suite_are_seed_programs_with_recipes():
    h = TabularHarness()
    clf = h.baseline_suite("classification")
    reg = h.baseline_suite("regression")
    assert clf and reg
    for p in clf + reg:
        assert p.source == "seed", "baselines are seeds/fallbacks, never promoters"
        assert "build_estimator" in p.code, "baseline code must define build_estimator()"
        assert p.provenance.get("harness") == "tabular"
    # the regression floor includes a feature-engineering recipe (poly) -- CA-Housing trap fix
    assert any("poly" in p.label for p in reg), f"expected a poly recipe in reg floor: {[p.label for p in reg]}"
    print(f"[ok] baselines: clf={[p.label for p in clf]} reg={[p.label for p in reg]}")


def test_self_test_certifies_through_frozen_gate():
    """The innovation: trust the harness only after it certifies itself on a known-good set."""
    h = TabularHarness()
    assert h.trusted is False, "harness must start untrusted"
    assert h.last_certificate is None
    ok, cert = h.self_test()
    assert ok is True, f"self-test must certify on the known-good dataset: {cert.detail}"
    assert isinstance(cert, HarnessCertificate) and bool(cert) is True
    assert h.trusted is True, "self-test success must flip trusted -> True"
    assert h.last_certificate is cert
    assert cert.dataset == "sklearn_breast_cancer" and cert.metric == "accuracy"
    # the self-test certificate came from the FROZEN sealed certifier: one counted peek,
    # lower bound <= observed, and it cleared the conservative self_theta.
    sc = cert.sealed_cert
    assert sc is not None and sc["peeks"] == 1, "self-test must touch sealed exactly once"
    assert sc["lower_bound"] <= sc["observed"] + 1e-9
    assert sc["lower_bound"] >= cert.self_theta - 1e-9, "certified bound must clear self_theta"
    print(f"[ok] self-test certified: winner={cert.winner_label} "
          f"sealed_lb={sc['lower_bound']:.4f} theta={cert.self_theta} peeks={sc['peeks']}")


def test_untrusted_harness_does_not_certify_a_known_failure():
    """A self-test with an impossible self_theta must NOT certify and must leave trusted=False.

    This guards the gate's honesty: the harness only flips trusted when the frozen certifier
    actually promotes. We subclass to force a self_theta no baseline can clear on the sealed
    lower bound.
    """

    class ImpossibleTabular(TabularHarness):
        key = "tabular_impossible"

        def _self_test_case(self):
            from sklearn.datasets import load_breast_cancer
            d = load_breast_cancer()
            # 0.999 lower-bound on a ~570-sample sealed test is not clearable here.
            return d.data, d.target.astype(str), "classification", "sklearn_breast_cancer", 0.999

    h = ImpossibleTabular()
    ok, cert = h.self_test()
    assert ok is False, "an impossible theta must not certify"
    assert h.trusted is False, "trusted must stay False on self-test failure"
    assert cert.certified is False and "did NOT certify" in cert.detail
    print(f"[ok] honest non-certification: {cert.detail}")


def test_real_dataset_routes_to_certified_result_via_engine():
    """End-to-end: gate the harness, then route a real dataset to a certified Phase-0 result."""
    h = lookup("tabular")
    ok, _ = h.self_test()
    assert ok, "must pass self-test before trusting downstream numbers"

    from sklearn.datasets import load_wine
    d = load_wine()
    # wine: 178 samples, 13 features, 3 classes. A scaled baseline reaches high accuracy; theta
    # 0.80 is conservative for the sealed lower bound but exercises the full route.
    task = h.adapt(d.data, d.target.astype(str), kind="classification", theta=0.80, name="wine")
    test_frac, val_frac = h.split_protocol("classification")
    cfg = EngineConfig(rounds=2, test_frac=test_frac, val_frac=val_frac,
                       wall_seconds=45, cpu_seconds=40)
    proposers = None  # default proposers; harness already validated the route via self_test
    res = ResearchEngine(cfg, proposers=proposers).run(task)
    assert res.winner is not None, "engine must find a runnable winner on wine"
    assert res.certificate is not None and res.certificate["peeks"] == 1
    assert res.certificate["lower_bound"] <= res.certificate["observed"] + 1e-9
    print(f"[ok] real route: winner={res.winner.label} certified={res.certified} "
          f"sealed_lb={res.certificate['lower_bound']:.4f} theta={task.theta}")
    assert res.certified, "wine at theta=0.80 should certify through the frozen gate"


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
    print(f"\n{len(fns) - failed}/{len(fns)} harness tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
