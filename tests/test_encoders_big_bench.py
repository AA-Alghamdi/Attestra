"""Hermetic tests for the phase #5 BIGGER-ENCODER benchmark's HONESTY-critical logic (no network, no pretrained
weights, no FGVC download -- pure numpy + sklearn on toy embeddings).

The full arena needs CLIP/DINOv2-L + three large frozen encoders + the FGVC dataset, so it is not a CI test. But
two properties make the cross-encoder comparison an honest "select-then-bound" measurement and MUST be locked:
  (1) the sealed TEST rows are derived from (y, seed) ONLY -- they are IDENTICAL across encoders -- so every
      challenger is scored on exactly the champion's sealed rows and the paired McNemar is valid;
  (2) a challenger arm binds the sealed rows exactly once: flipping ONLY the sealed-test labels inverts the
      returned correctness vector elementwise (the tuned-GBM head is selected on val, never on test);
  (3) the on-disk embedding cache is honest: a hit returns the byte-identical matrix without recomputing, and a
      different split size (n) MISSES (so a stale embedding can never be silently reused under a new config).
"""
import importlib.util
import os

import numpy as np
import pytest

pytest.importorskip("sklearn.ensemble")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "scripts", "benchmark_encoders_big.py")
_spec = importlib.util.spec_from_file_location("benchmark_encoders_big", _PATH)
P5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P5)

import scripts.benchmark_backbones as BB  # noqa: E402


def _toy_embeddings(n_per=30, d=16, seed=0):
    """One linearly-separable Gaussian blob per class; disjoint train/val/test ids that both see both classes."""
    rng = np.random.RandomState(seed)
    y = np.array([0] * n_per + [1] * n_per)
    mu = np.zeros(d); mu[0] = 3.0
    sign = np.where(y == 0, -1.0, 1.0)[:, None]
    emb = (rng.randn(2 * n_per, d) + sign * mu).astype(np.float32)
    idx = np.arange(2 * n_per)
    a, b = idx[y == 0], idx[y == 1]
    tr = np.concatenate([a[:20], b[:20]])
    val = np.concatenate([a[20:25], b[20:25]])
    test = np.concatenate([a[25:], b[25:]])
    return emb, y, tr, val, test


def test_sealed_test_rows_are_encoder_independent():
    """The sealed TEST rows must depend on (y, seed) ONLY -- NOT on the embedding values -- so swapping the
    encoder cannot change which rows are held out. This is what makes the cross-encoder McNemar pairing exact:
    every challenger is scored on identical sealed rows as the DINOv2-L champion."""
    rng = np.random.RandomState(1)
    y = np.array([0] * 40 + [1] * 40)
    emb_a = rng.randn(80, 16).astype(np.float32)        # "encoder A" features
    emb_b = rng.randn(80, 64).astype(np.float32) * 9.0  # a totally different "encoder B" (different dim/scale)
    _, _, test_a = BB._sealed_split(emb_a, y, 0)
    _, _, test_b = BB._sealed_split(emb_b, y, 0)
    assert test_a == test_b                              # identical sealed rows -> valid paired comparison
    assert len(test_a) > 0


def test_challenger_arm_is_select_then_bound_sealed_labels_never_leak():
    """A challenger encoder's embeddings go through the SAME tuned-GBM head; flipping ONLY the sealed-test labels
    must invert the returned correctness vector -- proving the sealed rows are bound once and never influence
    training or model selection."""
    emb, y, tr, val, test = _toy_embeddings()
    c1, acc1 = BB._emb_strong_correct(emb, y, tr, val, test, 0)

    y_flip = y.copy()
    y_flip[test] = 1 - y_flip[test]                      # corrupt ONLY the sealed-test labels
    c2, acc2 = BB._emb_strong_correct(emb, y_flip, tr, val, test, 0)

    assert len(c1) == len(test)
    assert set(c1) <= {0, 1}
    assert c2 == [1 - x for x in c1]                     # inverts -> sealed labels are bound-only, no leak
    assert abs(acc1 - (1.0 - acc2)) < 1e-9


def test_emb_cache_is_honest_hit_returns_identical_and_size_change_misses(tmp_path, monkeypatch):
    """The on-disk embedding cache must (a) return the byte-identical matrix on a hit WITHOUT recomputing, and
    (b) MISS when the split size n changes -- so no stale embedding is ever silently reused under a new config."""
    import scripts.benchmark_aircraft as A
    monkeypatch.setattr(P5, "CACHE_DIR", str(tmp_path))

    calls = {"n": 0}
    rng = np.random.RandomState(0)

    def _fake_embed(backbone, pil_imgs):
        calls["n"] += 1
        return rng.randn(len(pil_imgs), 8).astype(np.float32)   # deterministic-enough per call to detect recompute

    monkeypatch.setattr(A, "_open_rgb", lambda paths: list(paths))   # no PIL/torch needed
    monkeypatch.setattr(A, "_embed", _fake_embed)

    paths10 = [f"img_{i}" for i in range(10)]
    e1 = P5._emb_cached("encX", "pairY", object(), paths10)
    assert calls["n"] == 1                                           # computed once
    e2 = P5._emb_cached("encX", "pairY", object(), paths10)
    assert calls["n"] == 1                                           # HIT -> no recompute
    assert np.array_equal(e1, e2)                                   # byte-identical on hit

    P5._emb_cached("encX", "pairY", object(), [f"img_{i}" for i in range(11)])
    assert calls["n"] == 2                                           # different n MISSES -> recomputes
