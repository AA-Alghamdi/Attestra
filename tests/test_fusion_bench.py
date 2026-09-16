"""Hermetic tests for the phase #4 FUSION benchmark's HONESTY-critical logic (no network, no pretrained
weights, no FGVC download -- pure numpy + sklearn on toy embeddings).

The full arena needs CLIP + DINOv2 + the FGVC dataset, so it is not a CI test. But the property that makes the
authored LEARNED featurizer (`_fuse_pls_correct`) an honest "select-then-bound" measurement -- the supervised
PLS projection is fit on the TRAIN rows ONLY, and the sealed TEST rows never influence the projection or the
head -- is pure and MUST be locked. We certify it two ways:
  (1) flipping ONLY the sealed-test labels inverts the returned correctness vector elementwise (test rows are
      bound exactly once and never touch the projection/head fit);
  (2) the PLS projection receives EXACTLY the train-row labels (never val/test), so no val/test leakage can
      enter the learned representation.
"""
import importlib.util
import os

import numpy as np
import pytest

pytest.importorskip("sklearn.cross_decomposition")

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "benchmark_fusion.py")
_spec = importlib.util.spec_from_file_location("benchmark_fusion", _PATH)
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)


def _toy_embeddings(n_per=30, d_dino=16, d_clip=8, seed=0):
    """Two linearly-separable Gaussian blobs in each frozen space; disjoint train/val/test ids that both see
    both classes. Returns (emb_dino, emb_clip, y, tr, val, test)."""
    rng = np.random.RandomState(seed)
    y = np.array([0] * n_per + [1] * n_per)
    mu_d = np.zeros(d_dino); mu_d[0] = 3.0
    mu_c = np.zeros(d_clip); mu_c[0] = 2.5
    sign = np.where(y == 0, -1.0, 1.0)[:, None]
    emb_dino = (rng.randn(2 * n_per, d_dino) + sign * mu_d).astype(np.float32)
    emb_clip = (rng.randn(2 * n_per, d_clip) + sign * mu_c).astype(np.float32)
    idx = np.arange(2 * n_per)
    a, b = idx[y == 0], idx[y == 1]
    tr = np.concatenate([a[:20], b[:20]])
    val = np.concatenate([a[20:25], b[20:25]])
    test = np.concatenate([a[25:], b[25:]])
    return emb_dino, emb_clip, y, tr, val, test


def test_fuse_pls_is_select_then_bound_sealed_labels_never_leak():
    """Flipping ONLY the sealed-test labels must invert the returned correctness vector exactly -- the learned
    PLS projection and the GBM head are fit on train/val only, so test rows are bound exactly once."""
    ed, ec, y, tr, val, test = _toy_embeddings()
    c1, acc1 = F._fuse_pls_correct(ed, ec, y, tr, val, test, seed=0)

    y_flip = y.copy()
    y_flip[test] = 1 - y_flip[test]                       # corrupt ONLY the sealed-test labels
    c2, acc2 = F._fuse_pls_correct(ed, ec, y_flip, tr, val, test, seed=0)

    assert len(c1) == len(test)                           # correctness is exactly the sealed rows
    assert set(c1) <= {0, 1}
    assert c2 == [1 - x for x in c1]                      # inverts -> sealed labels are bound-only, no leak
    assert abs(acc1 - (1.0 - acc2)) < 1e-9


def test_fuse_pls_fits_projection_on_train_rows_only(monkeypatch):
    """The supervised PLS projection must be fit on EXACTLY the train-row labels -- never val or test -- so no
    val/test information can enter the learned representation."""
    import sklearn.cross_decomposition as CD
    ed, ec, y, tr, val, test = _toy_embeddings()

    seen = {}
    real_fit = CD.PLSRegression.fit

    def _spy_fit(self, X, Y):
        seen["nX"] = int(np.asarray(X).shape[0])
        seen["Y"] = np.asarray(Y).ravel().copy()
        return real_fit(self, X, Y)

    monkeypatch.setattr(CD.PLSRegression, "fit", _spy_fit)
    F._fuse_pls_correct(ed, ec, y, tr, val, test, seed=0)

    assert seen["nX"] == len(tr)                                   # projection saw exactly the train rows
    assert np.array_equal(seen["Y"], y[list(tr)].astype(float))    # and exactly the train labels (no val/test)


def test_fuse_concat_correctness_is_sealed_rows_only():
    """The deterministic concat featurizer (concat + same tuned-GBM head) also binds exactly the sealed rows:
    flipping sealed-test labels inverts the correctness vector elementwise."""
    import scripts.benchmark_backbones as BB
    ed, ec, y, tr, val, test = _toy_embeddings()
    Xcat = np.concatenate([ed, ec], axis=1).astype(np.float32)
    c1, acc1 = BB._emb_strong_correct(Xcat, y, tr, val, test, 0)

    y_flip = y.copy()
    y_flip[test] = 1 - y_flip[test]
    c2, acc2 = BB._emb_strong_correct(Xcat, y_flip, tr, val, test, 0)

    assert len(c1) == len(test)
    assert c2 == [1 - x for x in c1]
    assert abs(acc1 - (1.0 - acc2)) < 1e-9
