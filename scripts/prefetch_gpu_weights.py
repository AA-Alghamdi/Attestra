"""#5 -- PRE-FETCH every model weight tomorrow's GPU run needs, TONIGHT, on CPU, so GPU day is zero-download.

Model checkpoints are device-agnostic: the bytes are identical whether they load onto a CPU or a GPU, and a
CPU box can download them just fine. So we fetch them now and verify them, turning tomorrow's first action into
pure compute instead of a multi-gigabyte download wait. Two families:

  * torchvision FINE-TUNE backbones (the ceiling arm uses these end-to-end): resnet50, vit_b_16, and the big one
    a human would actually reach for on a GPU -- vit_l_16 with the SWAG weights (~1.1 GB), which is the only one
    not already on disk. Each is instantiated with its pretrained weights (which downloads + hash-checks the
    checkpoint) and the resulting checkpoint file is recorded.
  * the frozen ENCODERS the arenas embed with (DINOv2-g/L, CLIP-L/B, SigLIP-SO400M, EVA-02-L, the sentence
    encoders) -- already in the HuggingFace cache from the phase-#5 runs; we verify their snapshot dirs exist.

Writes `docs/GPU_WEIGHTS_MANIFEST.json` (path, byte size, source). Run: `python scripts/prefetch_gpu_weights.py`.
NOTE: this box ships a CPU-only torch (`+cpu`); tomorrow install a CUDA build first, then the cached weights are
reused with no re-download.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB_CKPT = os.path.expanduser("~/.cache/torch/hub/checkpoints")
HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")

# torchvision fine-tune backbones the GPU ceiling arm can use (builder attr, weights enum attr).
TORCHVISION_FT = [
    ("resnet50", "ResNet50_Weights", "IMAGENET1K_V2"),
    ("vit_b_16", "ViT_B_16_Weights", "IMAGENET1K_V1"),
    ("vit_l_16", "ViT_L_16_Weights", "IMAGENET1K_SWAG_E2E_V1"),
]

# frozen encoders the arenas embed with (HuggingFace cache dir name under ~/.cache/huggingface/hub).
HF_ENCODERS = [
    "models--timm--vit_giant_patch14_dinov2.lvd142m",
    "models--timm--vit_large_patch14_dinov2.lvd142m",
    "models--timm--vit_small_patch14_dinov2.lvd142m",
    "models--timm--vit_large_patch14_clip_224.openai",
    "models--timm--vit_base_patch32_clip_224.openai",
    "models--timm--vit_so400m_patch14_siglip_224.webli",
    "models--timm--ViT-L-16-SigLIP-256",
    "models--timm--eva02_large_patch14_224.mim_m38m",
    "models--sentence-transformers--all-mpnet-base-v2",
    "models--sentence-transformers--all-MiniLM-L6-v2",
    "models--intfloat--e5-base-v2",
    "models--intfloat--e5-large-v2",
    "models--intfloat--e5-small-v2",
]


def _dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp) and not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def _fetch_torchvision():
    import torchvision.models as M
    fetched = []
    for builder, wenum, variant in TORCHVISION_FT:
        weights = getattr(getattr(M, wenum), variant)
        fname = weights.url.split("/")[-1]
        fpath = os.path.join(HUB_CKPT, fname)
        had = os.path.exists(fpath)
        print(f"[torchvision] {builder} <- {wenum}.{variant}  ({'cached' if had else 'downloading'}) {fname}")
        getattr(M, builder)(weights=weights)        # triggers download + hash check, then loads
        size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
        print(f"             {size/1e6:.1f} MB  {fpath}")
        fetched.append({"arch": builder, "weights": f"{wenum}.{variant}", "file": fname,
                        "path": fpath, "bytes": size, "was_cached": had})
    return fetched


def _verify_encoders():
    out = []
    for name in HF_ENCODERS:
        path = os.path.join(HF_HUB, name)
        present = os.path.isdir(path)
        size = _dir_bytes(path) if present else 0
        flag = "OK" if present else "MISSING"
        print(f"[hf-encoder] {flag:7} {size/1e6:8.1f} MB  {name}")
        out.append({"name": name, "path": path, "bytes": size, "present": present})
    return out


def main():
    import torch
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("NOTE: CPU-only torch on this box -- weights still cache fine; install a CUDA build on GPU day.\n")

    tv = _fetch_torchvision()
    print()
    enc = _verify_encoders()

    missing = [e["name"] for e in enc if not e["present"]]
    manifest = {
        "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
        "torchvision_finetune_weights": tv,
        "hf_encoders": enc,
        "totals": {
            "torchvision_bytes": sum(t["bytes"] for t in tv),
            "hf_encoder_bytes": sum(e["bytes"] for e in enc),
        },
        "missing_encoders": missing,
        "ready": len(missing) == 0 and all(t["bytes"] > 0 for t in tv),
    }
    dst = os.path.join(ROOT, "docs", "GPU_WEIGHTS_MANIFEST.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(manifest, f, indent=2)
    gb = (manifest["totals"]["torchvision_bytes"] + manifest["totals"]["hf_encoder_bytes"]) / 1e9
    print(f"\nwrote {dst}")
    print(f"total cached weights: {gb:.2f} GB   ready={manifest['ready']}"
          + (f"   MISSING={missing}" if missing else ""))
    return 0 if manifest["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
