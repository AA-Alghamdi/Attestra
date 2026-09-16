"""Hermetic locks for the device-agnostic fine-tune ceiling (scripts/gpu_finetune.py).

This is the arm that goes to the GPU tomorrow, so the properties that keep its sealed numbers honest must hold
regardless of device -- and we lock them here on CPU with a tiny synthetic signal (no GPU, no real data):

  (1) device selection: cpu when CUDA is absent, and an explicit override is respected (so the GPU path is a
      pure device swap, not a code change);
  (2) STRICT select-then-bound (the leakage lock): flipping ONLY the sealed-test labels leaves the val-selected
      epoch and best-val IDENTICAL (training + epoch selection never see the sealed labels) and inverts the
      sealed correctness vector elementwise (the predictions are unchanged, only the labels they are scored
      against flip) -- exactly the invariant the frozen Clopper-Pearson bound relies on;
  (3) determinism: identical inputs + seed -> identical sealed correctness (so a certified number is reproducible);
  (4) the arch registry builds the GPU fine-tune backbones with a fresh 2-way head and a non-empty partial
      unfreeze.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.gpu_finetune as G  # noqa: E402


def _tiny_problem(seed=0):
    cache, y = G._synthetic_cache(n_per_class=10, px=32, seed=seed)
    idx = np.arange(len(y))
    rng = np.random.RandomState(7)
    tr, val, test = [], [], []
    for c in (0, 1):
        ci = idx[y == c]
        rng.shuffle(ci)
        tr += list(ci[:6])
        val += list(ci[6:8])
        test += list(ci[8:10])
    return cache, y, np.array(tr), np.array(val), np.array(test)


def _run(cache, y, tr, val, test):
    import torch
    return G.finetune_once(cache, y, tr, val, test, arch="resnet18", px=32, epochs=2, lr=0.01,
                           seed=0, device=torch.device("cpu"))


def test_pick_device_is_cpu_without_cuda_and_respects_override():
    import torch
    if not torch.cuda.is_available():
        assert G.pick_device().type == "cpu"
    assert G.pick_device("cpu").type == "cpu"            # explicit override always honored
    old = os.environ.get("ATTESTRA_DEVICE")
    os.environ["ATTESTRA_DEVICE"] = "cpu"
    try:
        assert G.pick_device().type == "cpu"
    finally:
        if old is None:
            del os.environ["ATTESTRA_DEVICE"]
        else:
            os.environ["ATTESTRA_DEVICE"] = old


def test_finetune_is_strict_select_then_bound():
    """Flipping ONLY the sealed-test labels must not change training/selection, and must invert sealed scoring."""
    cache, y, tr, val, test = _tiny_problem()
    c0, _, va0, ep0 = _run(cache, y, tr, val, test)

    y_flip = y.copy()
    y_flip[test] = 1 - y_flip[test]                      # corrupt ONLY the sealed labels
    c1, _, va1, ep1 = _run(cache, y_flip, tr, val, test)

    assert va0 == va1 and ep0 == ep1, "sealed labels leaked into val selection (best_val/best_ep changed)"
    assert [1 - v for v in c0] == list(c1), "sealed correctness did not invert -> predictions depend on labels"


def test_finetune_is_deterministic():
    cache, y, tr, val, test = _tiny_problem()
    c0, acc0, _, _ = _run(cache, y, tr, val, test)
    c1, acc1, _, _ = _run(cache, y, tr, val, test)
    assert list(c0) == list(c1) and acc0 == acc1


@pytest.mark.parametrize("arch", ["resnet18", "resnet50", "vit_b_16"])
def test_make_model_has_two_way_head_and_nonempty_unfreeze(arch):
    m = G.make_model(arch, n_classes=2)
    trainable = [p for p in m.parameters() if p.requires_grad]
    assert len(trainable) > 0, "partial unfreeze left nothing trainable"
    out_features = [p.shape[0] for n, p in m.named_parameters() if n.endswith("weight") and p.dim() == 2]
    assert 2 in out_features, "no 2-way classification head was attached"
