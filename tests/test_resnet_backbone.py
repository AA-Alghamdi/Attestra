"""ResnetBackbone plumbing (hermetic: pretrained=False, no download / no network).

Proves the production transfer path is real -- a torchvision CNN with its head removed, mapping mixed raw
image inputs (numpy HxWxC uint8, numpy HxW float, PIL) to a fixed (n, feat_dim) matrix that ImageFeaturizer
L2-normalizes. The frozen-vs-random weights question is irrelevant to plumbing, so we skip the ImageNet
download and use random weights; the benchmark uses pretrained=True.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from vfplatform.featurizers import ImageFeaturizer, ResnetBackbone  # noqa: E402


def _imgs():
    rng = np.random.RandomState(0)
    return [
        (rng.rand(20, 24, 3) * 255).astype(np.uint8),   # HxWxC uint8
        rng.rand(16, 16).astype(float),                  # HxW float (grayscale)
        (rng.rand(30, 10, 3) * 255).astype(np.uint8),   # non-square
    ]


def test_backbone_returns_fixed_dim_matrix():
    bb = ResnetBackbone(arch="resnet18", resize=32, batch_size=2, pretrained=False)
    assert bb.feat_dim == 512
    emb = bb(_imgs())
    assert emb.shape == (3, 512)
    assert np.isfinite(emb).all()


def test_backbone_is_deterministic_eval_mode():
    bb = ResnetBackbone(arch="resnet18", resize=32, pretrained=False)
    a, b = bb(_imgs()), bb(_imgs())
    assert np.allclose(a, b)  # eval mode -> no dropout/batchnorm drift


def test_image_featurizer_uses_backbone_and_l2_normalizes():
    bb = ResnetBackbone(arch="resnet18", resize=32, pretrained=False)
    feat = ImageFeaturizer(backbone=bb, backbone_dim=bb.feat_dim)
    assert feat.name == "image_backbone"
    assert feat.dim == 512
    M = feat.transform(_imgs())
    assert M.shape == (3, 512)
    norms = np.linalg.norm(M, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)  # L2-normalized rows


def test_empty_input_returns_empty_matrix():
    bb = ResnetBackbone(arch="resnet18", resize=32, pretrained=False)
    assert bb([]).shape == (0, 512)


def test_rejects_unknown_arch():
    with pytest.raises(ValueError):
        ResnetBackbone(arch="vgg999", pretrained=False)
