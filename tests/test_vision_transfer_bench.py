"""Hermetic tests for the B1 vision-transfer benchmark's PURE logic (no torch, no network, no CIFAR).

The full arena needs torch + an ImageNet download, so it is not a CI test. But the harness logic that makes
the comparison honest -- fixed-width feature keys (so sorted order is stable), an exact paired correctness
vector, and the FROZEN Clopper-Pearson lower bound per arm -- is pure and must be locked. We exercise it on
synthetic separable data and assert the strong-baseline search actually learns and the lower bound is sane.
"""
import importlib.util
import os

import numpy as np

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "benchmark_vision_transfer.py")
_spec = importlib.util.spec_from_file_location("benchmark_vision_transfer", _PATH)
btv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(btv)


def test_records_have_fixedwidth_keys_and_roundtrip():
    X = np.arange(20, dtype=float).reshape(4, 5)
    y = np.array([0, 1, 0, 1])
    recs = btv._records(X, y, np.arange(4))
    keys = sorted(recs[0]["features"].keys())
    assert keys == ["e0", "e1", "e2", "e3", "e4"]   # one digit -> width 1
    X2, y2 = btv._xy(recs)
    assert np.allclose(X2, X) and np.array_equal(y2, y)
    assert [r["rid"] for r in recs] == [0, 1, 2, 3]


def test_lb_is_frozen_clopper_pearson_and_monotone():
    from vectorforge import science
    correct = [1] * 90 + [0] * 10
    assert btv._lb(correct) == round(science.clopper_pearson_lower(90, 100, 0.05), 4)
    # more data at the same rate -> tighter (higher) lower bound
    assert btv._lb([1] * 450 + [0] * 50) > btv._lb(correct)
    assert 0.0 <= btv._lb([1, 0, 1, 0]) <= 1.0


def test_random_search_best_learns_separable_signal():
    rng = np.random.RandomState(0)
    n = 240
    y = (rng.rand(n) > 0.5).astype(int)
    X = np.zeros((n, 6))
    X[:, 0] = y + rng.randn(n) * 0.1            # one strongly informative feature
    X += rng.randn(n, 6) * 0.05
    tr, va, te = slice(0, 140), slice(140, 190), slice(190, n)
    est = btv._random_search_best(np.random.RandomState(1), X[tr], y[tr], X[va], y[va], k=8)
    acc = np.mean(btv._correct(est, X[te], y[te]))
    assert acc > 0.9                            # a strong baseline must crush a separable task
