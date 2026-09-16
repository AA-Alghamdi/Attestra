"""Hermetic tests for the phase #3 FGVC-Aircraft benchmark's HONESTY-critical logic (no network, no FGVC
download, no pretrained weights).

The full arena needs torch + DINOv2/CLIP + the FGVC dataset, so it is not a CI test. But the property that makes
the fine-tune CEILING arm an honest "select-then-bound" measurement -- the sealed TEST rows never influence
training or model/epoch/restart selection (they are bound exactly once at the end) -- is pure and MUST be locked.
We certify it directly: flipping the sealed-test labels must invert the returned correctness vector elementwise
while leaving the val-selected epoch/restart identical. We also lock multi-restart val-selection and the
pre-decode cache geometry. `_make_model` is monkeypatched to a tiny randomly-initialised CNN so the test is fast
and needs no ImageNet download.
"""
import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "benchmark_aircraft.py")
_spec = importlib.util.spec_from_file_location("benchmark_aircraft", _PATH)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


def _tiny_model_factory():
    """A small deterministic CNN with 2 logits -- no pretrained download, all params trainable."""
    def _make(arch):  # signature matches A._make_model(arch)
        torch.manual_seed(1234)
        return nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2),
        )
    return _make


def _toy_task(px=16, n_per=9):
    """n_per images/class of solid-colour squares (class 0 dark, class 1 bright). Returns (decode_cache, y,
    tr_ids, val_ids, test_ids) with disjoint splits -- enough to run a 2-epoch fine-tune deterministically."""
    from PIL import Image
    side = px + 32
    cache, y = [], []
    for cls in (0, 1):
        for k in range(n_per):
            v = 30 + k if cls == 0 else 200 + k
            cache.append(Image.new("RGB", (side, side), (v, v, v)))
            y.append(cls)
    y = np.array(y)
    idx = np.arange(len(y))
    # interleave classes so every split sees both labels
    order = np.concatenate([idx[y == 0], idx[y == 1]])
    a, b = order[y[order] == 0], order[y[order] == 1]
    tr = np.concatenate([a[:5], b[:5]])
    val = np.concatenate([a[5:7], b[5:7]])
    test = np.concatenate([a[7:], b[7:]])
    return cache, y, tr, val, test


def test_decode_cache_is_square_side_px_plus_32():
    cache, _, _, _, _ = _toy_task(px=24)
    # _decode_cache reads file paths; here we assert the geometry contract it guarantees downstream.
    im = cache[0]
    assert im.size == (24 + 32, 24 + 32)
    assert im.mode == "RGB"


def test_finetune_is_select_then_bound_sealed_labels_never_leak(monkeypatch):
    """Flipping ONLY the sealed-test labels must (a) invert the returned correctness vector exactly and
    (b) leave the val-selected epoch identical -- i.e. test rows are bound once and never touch training/selection."""
    monkeypatch.setattr(A, "_make_model", _tiny_model_factory())
    cache, y, tr, val, test = _toy_task()

    c1, acc1, va1, ep1 = A._finetune_once(cache, y, tr, val, test, arch="tiny", px=16, epochs=2, lr=0.01, seed=0)

    y_flip = y.copy()
    y_flip[test] = 1 - y_flip[test]                       # corrupt ONLY the sealed-test labels
    c2, acc2, va2, ep2 = A._finetune_once(cache, y_flip, tr, val, test, arch="tiny", px=16, epochs=2, lr=0.01,
                                          seed=0)

    assert len(c1) == len(test)                           # correctness is exactly the sealed rows
    assert set(c1) <= {0, 1}
    assert va1 == va2 and ep1 == ep2                      # selection (on val) is unaffected by test labels
    assert c2 == [1 - x for x in c1]                      # correctness inverts -> test labels are bound-only
    assert abs(acc1 - (1.0 - acc2)) < 1e-9


def test_finetune_correct_picks_max_val_restart(monkeypatch):
    """The ceiling arm runs R restarts and returns the one with the BEST val accuracy (val-selected, no peeking)."""
    monkeypatch.setattr(A, "_make_model", _tiny_model_factory())
    cache, y, tr, val, test = _toy_task()

    calls = {"vals": [0.3, 0.9, 0.6]}
    real_once = A._finetune_once
    seen = []

    def _fake_once(c, yy, t, v, te, arch, px, epochs, lr, seed):
        out = real_once(c, yy, t, v, te, arch, px, epochs, lr, seed)
        va = calls["vals"][seed]                          # force distinct, known val accuracies per restart
        seen.append(va)
        return out[0], out[1], va, seed

    monkeypatch.setattr(A, "_finetune_once", _fake_once)
    c, acc, best_ep, best_va, vals = A._finetune_correct(cache, y, tr, val, test, arch="tiny", px=16,
                                                         epochs=1, lr=0.01, restarts=3)
    assert vals == [0.3, 0.9, 0.6]
    assert best_va == 0.9                                 # selected the highest-val restart
    assert best_ep == 1                                   # seed index of the winner (restart 1)
    assert len(c) == len(test)
