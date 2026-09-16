"""Tests for foundation-model-style featurizers.

Acceptance (Phase 8): the transfer-flavored image featurizer captures position-INVARIANT shape/edge
structure -- a linear head separates vertical- vs horizontal-edge images far better on its embedding than
on flattened raw pixels (where the signal is positional and a linear model cannot find it). Also: all
featurizers are deterministic, fixed-dimension, L2-normalized, finite; a real backbone/encoder can be
injected; and a short audio clip featurizes well under the 50ms-class budget."""
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import featurizers as FZ


def _stripe_image(orientation: str, size: int, rng: random.Random) -> np.ndarray:
    """A `size`x`size` striped image with RANDOM phase and frequency. orientation 'v' = vertical stripes
    (intensity varies along x), 'h' = horizontal stripes. Random phase/frequency means the discriminative
    signal is the gradient ORIENTATION (a frequency/texture cue), which is not linearly separable in raw
    pixels (a fixed pixel weight averages out across phases) but trivial for an orientation histogram."""
    freq = rng.choice([2, 3, 4])
    phase = rng.uniform(0, 2 * math.pi)
    coords = np.arange(size)
    line = 0.5 + 0.5 * np.cos(2 * math.pi * freq * coords / size + phase)
    if orientation == "v":
        img = np.tile(line.reshape(1, size), (size, 1))
    else:
        img = np.tile(line.reshape(size, 1), (1, size))
    img = img + np.asarray([[rng.gauss(0, 0.02) for _ in range(size)] for _ in range(size)])
    return img


def _linear_cv_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=2000)
    return float(cross_val_score(clf, X, y, cv=4).mean())


def test_image_featurizer_beats_flatten_on_orientation():
    rng = random.Random(0)
    size = 16
    imgs, labels = [], []
    for _ in range(60):
        imgs.append(_stripe_image("v", size, rng)); labels.append(0)
        imgs.append(_stripe_image("h", size, rng)); labels.append(1)
    y = np.asarray(labels)

    flat = np.asarray([im.reshape(-1) for im in imgs])
    fz = FZ.ImageFeaturizer(grid=8, orient_bins=9)
    emb = fz.transform(imgs)

    acc_flat = _linear_cv_accuracy(flat, y)
    acc_emb = _linear_cv_accuracy(emb, y)
    assert emb.shape == (len(imgs), fz.dim)
    assert acc_emb > 0.95                       # orientation embedding nails it
    assert acc_emb > acc_flat + 0.15            # and decisively beats flatten-pixels


def test_featurizers_deterministic_normalized_finite():
    rng = random.Random(1)
    imgs = [_stripe_image("v", 10, rng) for _ in range(5)]
    texts = ["end the call politely goodbye", "escalate to a manager now", "end the call politely goodbye"]
    waves = [np.sin(np.linspace(0, 50, 800)) for _ in range(4)]

    for fz, data in [(FZ.ImageFeaturizer(), imgs), (FZ.TextFeaturizer(dim=128), texts),
                     (FZ.AudioFeaturizer(n_mels=24), waves)]:
        M1 = fz.transform(data)
        M2 = fz.transform(data)
        assert M1.shape[1] == fz.dim
        assert np.allclose(M1, M2)                                    # deterministic
        assert np.isfinite(M1).all()                                 # finite
        norms = np.linalg.norm(M1, axis=1)
        assert np.allclose(norms[norms > 0], 1.0, atol=1e-6)          # L2-normalized


def test_text_similar_docs_closer_than_unrelated():
    fz = FZ.TextFeaturizer(dim=512, word_ngrams=2)
    M = fz.transform(["please escalate this to a manager",
                       "escalate to a manager please",
                       "the weather is sunny today"])
    sim_related = float(M[0] @ M[1])
    sim_unrelated = float(M[0] @ M[2])
    assert sim_related > sim_unrelated + 0.2


def test_backbone_injection_used_when_provided():
    def fake_backbone(inputs):
        return np.asarray([[1.0, 2.0, 3.0] for _ in inputs])
    fz = FZ.ImageFeaturizer(backbone=fake_backbone, backbone_dim=3)
    out = fz.transform([np.zeros((8, 8)), np.ones((8, 8))])
    assert fz.name == "image_backbone" and fz.dim == 3 and out.shape == (2, 3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)             # still L2-normalized


def test_audio_featurization_latency_short_clip():
    fz = FZ.AudioFeaturizer(sample_rate=16000, n_mels=24)
    clip = np.sin(np.linspace(0, 100, 800))                          # 50ms @ 16kHz = 800 samples
    t0 = time.perf_counter()
    M = fz.transform([clip])
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert M.shape == (1, fz.dim)
    assert dt_ms < 50.0                                              # featurization is cheap


def test_factory_dispatch():
    assert FZ.get_featurizer("vision").name.startswith("image")
    assert FZ.get_featurizer("text").name.startswith("text")
    assert FZ.get_featurizer("speech").name.startswith("audio")
    with pytest.raises(ValueError):
        FZ.get_featurizer("smell")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
