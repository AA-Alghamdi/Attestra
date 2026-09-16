"""Tests for the multi-objective promotion gate.

The central property: the gate can ONLY TIGHTEN. No combination of constraints/measurements may ever
produce certified=True when the frozen base certificate was certified=False.
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import envelope as E
from vfplatform import gate as G


def _env(**kw):
    return E.from_goal(metric=kw.pop("metric", "accuracy"), theta=kw.pop("theta", 0.8),
                       n=kw.pop("n", 200), task_type=kw.pop("task_type", "binary"), **kw)


def test_passing_base_with_no_constraints_stays_certified():
    env = _env()
    dec = G.certified_under_envelope({"certified": True}, G.Measurements(), env)
    assert dec.certified is True and dec.base_certified is True
    G.assert_only_tightens(dec)


def test_failing_base_can_never_be_rescued():
    env = _env(max_latency_ms=1000.0)
    dec = G.certified_under_envelope({"certified": False}, G.Measurements(latency_ms=1.0), env)
    assert dec.certified is False and dec.base_certified is False
    G.assert_only_tightens(dec)


def test_latency_gate_tightens():
    env = _env(max_latency_ms=50.0)
    ok = G.certified_under_envelope({"certified": True}, G.Measurements(latency_ms=30.0), env)
    bad = G.certified_under_envelope({"certified": True}, G.Measurements(latency_ms=80.0), env)
    assert ok.certified is True
    assert bad.certified is False and bad.base_certified is True
    assert bad.constraints["latency"].status == G.FAIL


def test_unmeasured_constraint_blocks_by_default():
    env = _env(max_latency_ms=50.0)
    dec = G.certified_under_envelope({"certified": True}, G.Measurements(latency_ms=None), env)
    assert dec.certified is False
    assert dec.constraints["latency"].status == G.UNMEASURED
    # lenient mode lets it through (still <= base)
    lenient = G.certified_under_envelope({"certified": True}, G.Measurements(latency_ms=None), env,
                                         strict_unmeasured=False)
    assert lenient.certified is True


def test_per_class_recall_floor():
    # 4 escalate rows, model gets 2 right -> recall 0.5 < floor 0.95 -> blocked
    y_true = ["escalate", "escalate", "escalate", "escalate", "goodbye", "goodbye"]
    y_pred = ["escalate", "escalate", "goodbye", "goodbye", "goodbye", "goodbye"]
    env = _env(task_type="multiclass", per_class_recall_floor={"escalate": 0.95})
    dec = G.certified_under_envelope({"certified": True},
                                     G.Measurements(y_true=y_true, y_pred=y_pred), env)
    assert dec.certified is False
    assert dec.constraints["per_class_recall"].status == G.FAIL
    # a model that nails escalate passes
    y_pred2 = ["escalate", "escalate", "escalate", "escalate", "goodbye", "escalate"]
    dec2 = G.certified_under_envelope({"certified": True},
                                      G.Measurements(y_true=y_true, y_pred=y_pred2), env)
    assert dec2.constraints["per_class_recall"].status == G.PASS and dec2.certified is True


def test_subgroup_parity_gap():
    # group A perfect, group B poor -> large accuracy gap
    y_true = ["1"] * 20 + ["1"] * 20
    y_pred = ["1"] * 20 + ["0"] * 20
    groups = ["A"] * 20 + ["B"] * 20
    env = _env(subgroup_parity={"attr": "skin_tone", "max_gap": 0.05, "min_support": 5})
    dec = G.certified_under_envelope(
        {"certified": True}, G.Measurements(y_true=y_true, y_pred=y_pred, subgroup_values=groups), env)
    assert dec.constraints["subgroup_parity"].status == G.FAIL and dec.certified is False


def test_scope_is_recorded():
    env = _env()
    dec = G.certified_under_envelope({"certified": True},
                                     G.Measurements(scope=G.SCOPE_SCOPED), env)
    assert dec.scope == G.SCOPE_SCOPED


def test_only_tightens_property_randomized():
    """Fuzz: across random constraints/measurements/base verdicts, certified ==> base_certified."""
    rng = random.Random(0)
    for _ in range(2000):
        base = rng.random() < 0.5
        env = _env(max_latency_ms=rng.choice([None, 50.0]),
                   max_cost_usd=rng.choice([None, 1.0]),
                   max_ece=rng.choice([None, 0.1]),
                   per_class_recall_floor=rng.choice([None, {"1": 0.9}]),
                   task_type="binary")
        n = 30
        y_true = [rng.choice(["0", "1"]) for _ in range(n)]
        y_pred = [rng.choice(["0", "1"]) for _ in range(n)]
        m = G.Measurements(latency_ms=rng.choice([None, 10.0, 100.0]),
                           cost_usd=rng.choice([None, 0.5, 5.0]),
                           ece=rng.choice([None, 0.05, 0.5]),
                           y_true=y_true, y_pred=y_pred)
        dec = G.certified_under_envelope({"certified": base}, m, env,
                                         strict_unmeasured=rng.choice([True, False]))
        G.assert_only_tightens(dec)
        if dec.certified:
            assert dec.base_certified is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
