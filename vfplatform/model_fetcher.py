"""OPEN BACKBONE DISCOVERY -- the generator pulls models from the published zoo, not a curated list.

WHY THIS EXISTS (the anti-menu, made concrete)
----------------------------------------------
A menu is a hand-written list of ~9 encoders. This module makes the backbone axis OPEN by drawing from the
entire published model zoo:

  * OFFLINE, always available: `timm.list_models(pretrained=True)` exposes ~1700 pretrained architectures
    (DINOv2, CLIP/SigLIP, EVA-02, ConvNeXt, Swin, ViT, ResNet, EfficientNet, ...). The generator filters
    this by a task hint and samples from it -- so the reachable set is "the timm zoo", not a curated nine.
  * ONLINE, when an HF token + internet are present: `huggingface_hub.list_models(...)` extends the pool
    with arbitrary Hub models (image-classification / feature-extraction tagged), ranked by downloads.

The result is a GROWABLE pool. The DiscoveryLedger (vfplatform/recipe.py) records which backbones were
SEED vs DISCOVERED here, so the certificate can prove the champion was discovered, not enumerated.

SAFETY / HONESTY
----------------
* Discovery never downloads weights -- it only lists ids. Weights are fetched by the runner at evaluation
  time, behind the validity cascade's contamination check.
* Offline is the default and is fully functional (timm catalog ships with the package). Online is additive
  and degrades gracefully: any network/auth error falls back to the offline pool, never crashes a run.
"""
from __future__ import annotations

import fnmatch
import os
from typing import List, Optional, Sequence

# Coarse task-hint -> architecture-family globs. A hint biases discovery toward the right part of the zoo
# (fine-grained vision -> self-supervised ViTs & CLIP; texture/medical -> convnets too) WITHOUT enumerating
# specific models. Unknown hints fall back to a broad sample.
_HINT_GLOBS = {
    "fine_grained": ["*dinov2*", "*clip*", "eva02_*", "vit_large*", "vit_base*", "convnext_base*"],
    "distribution_shift": ["*dinov2*", "*clip*", "convnext_*", "vit_*", "resnet50*", "swin_*"],
    "medical": ["convnext_*", "resnet*", "vit_base*", "*dinov2*", "efficientnet_*"],
    "texture": ["convnext_*", "resnet*", "efficientnet_*", "vit_base*"],
    "general": ["vit_*", "convnext_*", "resnet*", "*dinov2*", "*clip*", "swin_*", "eva02_*"],
}


def _timm_pool() -> List[str]:
    """All timm ids with pretrained weights (offline). Empty list if timm is unavailable (never raises)."""
    try:
        import timm
        return list(timm.list_models(pretrained=True))
    except Exception:
        return []


def discover_backbones(task_hint: str = "general", *, limit: int = 24,
                       extra_globs: Optional[Sequence[str]] = None,
                       seed_pool: Optional[Sequence[str]] = None,
                       online: bool = False, online_limit: int = 12) -> List[str]:
    """Return a pool of candidate backbone ids for a task, drawn from the OPEN zoo (not a curated list).

    task_hint biases the architecture families sampled; `limit` caps the offline pool size (deterministic,
    sorted). If `online` and an HF token is present, additionally query the Hub and append (de-duplicated).
    seed_pool ids are always included first so a run is reproducible from its seeds. Never raises."""
    globs = list(_HINT_GLOBS.get(task_hint, _HINT_GLOBS["general"]))
    if extra_globs:
        globs.extend(extra_globs)

    pool: List[str] = list(dict.fromkeys(seed_pool or []))
    catalog = _timm_pool()
    for g in globs:
        hits = sorted(m for m in catalog if fnmatch.fnmatch(m, g))
        for m in hits:
            if m not in pool:
                pool.append(m)
            if len(pool) >= limit:
                break
        if len(pool) >= limit:
            break

    if online:
        for m in _hub_models(task_hint, limit=online_limit):
            if m not in pool:
                pool.append(m)

    return pool[: max(limit, len(seed_pool or []))]


def _hub_models(task_hint: str, *, limit: int = 12) -> List[str]:
    """Query the Hugging Face Hub for image models, ranked by downloads. Returns [] on any error (offline,
    no token, rate limit) so discovery is robust. Only ids are returned; nothing is downloaded here."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        from huggingface_hub import list_models
        tasks = ["image-classification", "image-feature-extraction"]
        out: List[str] = []
        for t in tasks:
            for info in list_models(filter=t, sort="downloads", direction=-1, limit=limit, token=token):
                mid = getattr(info, "id", None) or getattr(info, "modelId", None)
                if mid and mid not in out:
                    out.append(mid)
        return out[:limit]
    except Exception:
        return []


def is_timm_id(model_id: str) -> bool:
    """Whether a model id is loadable as a timm backbone (used by the runner to pick the load path)."""
    try:
        import timm
        return model_id in set(timm.list_models()) or model_id.startswith("hf-hub:")
    except Exception:
        return False


__all__ = ["discover_backbones", "is_timm_id"]
