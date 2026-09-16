"""CLIP + timm/DINOv2 backbone plumbing (hermetic: random weights, no download / no network).

Proves the two B2-repr challenger representations are real, frozen, callable inputs -> (n, feat_dim) matrices
with the SAME interface as ResnetBackbone, so ImageFeaturizer consumes them unchanged. The frozen-vs-random
weights question is irrelevant to plumbing, so we skip the pretrained download (CLIP pretrained=None, timm
pretrained=False) and use a tiny image size for speed; the benchmark uses the pretrained weights.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")


def _imgs():
    rng = np.random.RandomState(0)
    return [
        (rng.rand(20, 24, 3) * 255).astype(np.uint8),   # HxWxC uint8
        rng.rand(16, 16).astype(float),                  # HxW float (grayscale)
        (rng.rand(30, 10, 3) * 255).astype(np.uint8),   # non-square
    ]


def test_clip_backbone_returns_fixed_dim_matrix_and_is_deterministic():
    pytest.importorskip("open_clip")
    from vfplatform.featurizers import ImageFeaturizer, ClipImageBackbone
    bb = ClipImageBackbone(model_name="ViT-B-32", pretrained=None, batch_size=2)
    assert bb.feat_dim == 512
    emb = bb(_imgs())
    assert emb.shape == (3, 512)
    assert np.isfinite(emb).all()
    assert np.allclose(emb, bb(_imgs()))            # eval mode -> deterministic
    assert bb([]).shape == (0, 512)
    M = ImageFeaturizer(backbone=bb, backbone_dim=bb.feat_dim).transform(_imgs())
    assert M.shape == (3, 512)
    assert np.allclose(np.linalg.norm(M, axis=1), 1.0, atol=1e-5)   # ImageFeaturizer L2-normalizes


def test_timm_dinov2_backbone_returns_fixed_dim_matrix_and_is_deterministic():
    pytest.importorskip("timm")
    from vfplatform.featurizers import ImageFeaturizer, TimmBackbone
    # patch14 requires img_size divisible by 14; 28 (=2x2 patches) keeps the hermetic test fast.
    bb = TimmBackbone(model_name="vit_small_patch14_dinov2.lvd142m", pretrained=False,
                      batch_size=2, img_size=28)
    assert bb.feat_dim == 384
    emb = bb(_imgs())
    assert emb.shape == (3, 384)
    assert np.isfinite(emb).all()
    assert np.allclose(emb, bb(_imgs()))            # eval mode -> deterministic
    assert bb([]).shape == (0, 384)
    M = ImageFeaturizer(backbone=bb, backbone_dim=bb.feat_dim).transform(_imgs())
    assert M.shape == (3, 384)
    assert np.allclose(np.linalg.norm(M, axis=1), 1.0, atol=1e-5)


def test_siglip_backbone_feat_dim_probe_fallback_is_deterministic():
    # SigLIP's open_clip visual tower is timm-backed and has NO `visual.output_dim`; ClipImageBackbone must
    # fall back to a dummy forward-pass probe to read the embedding width. Hermetic: random weights, no network.
    pytest.importorskip("open_clip")
    pytest.importorskip("timm")
    from vfplatform.featurizers import ImageFeaturizer, ClipImageBackbone
    bb = ClipImageBackbone(model_name="ViT-B-16-SigLIP", pretrained=None, batch_size=2)
    assert bb.feat_dim > 0                          # discovered via probe, not via visual.output_dim
    emb = bb(_imgs())
    assert emb.shape == (3, bb.feat_dim)
    assert np.isfinite(emb).all()
    assert np.allclose(emb, bb(_imgs()))            # eval mode -> deterministic
    assert bb([]).shape == (0, bb.feat_dim)
    M = ImageFeaturizer(backbone=bb, backbone_dim=bb.feat_dim).transform(_imgs())
    assert M.shape == (3, bb.feat_dim)
    assert np.allclose(np.linalg.norm(M, axis=1), 1.0, atol=1e-5)


def test_clip_requires_open_clip_or_raises_importerror():
    # If open_clip is genuinely importable this just constructs; otherwise the constructor must raise
    # ImportError (never a bare ModuleNotFoundError leaking from inside).
    from vfplatform.featurizers import ClipImageBackbone
    try:
        import open_clip  # noqa: F401
    except Exception:
        with pytest.raises(ImportError):
            ClipImageBackbone(pretrained=None)
