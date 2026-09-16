# B2-repr-2 - Where is the frozen-feature ceiling? (ViT-L backbones vs the CLIP-B champion)

**Frozen certifier untouched:** `science.py b564fba2` / `sealed.py 30ad6245` - byte-identical before and after
this work (full sha256 recorded in the result JSON). The benchmark only *reads* the frozen Clopper–Pearson
lower bound (`science.clopper_pearson_lower`); it never edits a frozen file and never weakens a gate.

## The question

B2-repr proved the **representation is the lever**: swapping the frozen backbone from resnet18 to CLIP
ViT-B/32 or DINOv2 ViT-S/14 - nothing else changed - won **5/5 FDR** over resnet18 `emb-strong`. The new
champion is **CLIP ViT-B/32**. This phase asks the obvious follow-up before moving arenas: **does pushing to a
bigger / different ViT-L backbone keep climbing, or has the B1 arena hit its frozen-feature ceiling?**

Same B1 arena (five CIFAR-10 confusable binary tasks, `per_class=500`, identical sealed test, n=300/task). The
**baseline is now CLIP ViT-B/32** (the B2-repr champion, *not* resnet18). The split is derived **once** from the
baseline (CLIP-B) embeddings and reused verbatim for every challenger, so all McNemar pairing is on identical
sealed rows; only the frozen backbone changes.

| arm | representation | dim | what it tests |
|-----|----------------|-----|---------------|
| **clip_vitb32** *(baseline)* | OpenAI CLIP image tower (`ViT-B-32-quickgelu`) | 512 | the B2-repr champion |
| resnet18 | ImageNet-supervised CNN, resize 112 | 512 | sanity ref (B1 baseline) |
| dinov2_vits14 | DINOv2 ViT-S/14 (timm) | 384 | sanity ref (the co-winner) |
| **clip_vitl14** | CLIP `ViT-L-14-quickgelu` | 768 | **scale the SAME inductive bias** (CLIP B→L) |
| **dinov2_vitl14** | DINOv2 ViT-L/14 (timm) | 1024 | **scale a different bias** (self-supervised S→L) |
| **siglip_vitl16** | `ViT-L-16-SigLIP-256` (open_clip / timm tower) | 1024 | **a third objective at L scale** (sigmoid loss) |

`script: scripts/benchmark_backbones.py` (`ATTESTRA_ROSTER=big ATTESTRA_BASELINE=clip_vitb32`) ·
`raw JSON: docs/BENCHMARK_BACKBONES_BIG_RESULT.json` (per_class=500, seed=0, sealed n=300/task). Each challenger
is paired against CLIP-B by one-sided exact McNemar (challenger > CLIP-B) on the identical sealed rows, then
Benjamini–Hochberg(α=0.1) across the five-task suite, per backbone.

> Memory-frugal orchestration (7 GB box; three ViT-L towers cannot coexist): **Phase A** loads only the baseline
> backbone, derives each task's split once and records baseline correctness; **Phase B** loads exactly one
> challenger at a time, encodes all five tasks, then frees it (`gc.collect()`) before the next. Results are
> checkpointed to the JSON after every backbone.

## Result

Sealed accuracy, and lift over CLIP-B with its one-sided McNemar p (bold = survives BH-FDR(0.1) with positive
lift):

| task | clip_vitb32 (lb) | resnet18 Δ | dinov2_vits14 Δ | clip_vitl14 Δ | dinov2_vitl14 Δ | siglip_vitl16 Δ |
|------|------------------|------------|-----------------|---------------|-----------------|-----------------|
| cat_vs_dog          | 0.930 | −0.100 (1.00) | −0.017 (.88) | +0.027 (.06) | **+0.067** (.0004) | **+0.037** (.02) |
| automobile_vs_truck | 0.970 | −0.057 (1.00) | −0.010 (.85) | +0.000 (.62) | +0.003 (.50) | +0.017 (.06) |
| deer_vs_horse       | 0.980 | −0.083 (1.00) | −0.003 (.73) | −0.003 (.73) | +0.017 (.06) | **+0.020** (.02) |
| airplane_vs_ship    | 0.983 | −0.090 (1.00) | +0.013 (.11) | +0.003 (.50) | **+0.017** (.03) | +0.013 (.11) |
| bird_vs_frog        | 0.987 | −0.083 (1.00) | +0.007 (.34) | +0.013 (.06) | +0.010 (.19) | +0.007 (.34) |

**BH-FDR(0.1) survivors with positive lift, vs CLIP ViT-B/32:**

| challenger | scales | survivors | mean lift |
|------------|--------|-----------|-----------|
| resnet18      | - (ref)             | 0/5 | −0.083 |
| dinov2_vits14 | - (ref)             | 0/5 | −0.002 |
| **clip_vitl14**   | CLIP **B→L** (same bias) | **0/5** | +0.008 |
| **dinov2_vitl14** | DINOv2 **S→L** (self-sup) | **2/5** (cat_vs_dog, airplane_vs_ship) | +0.023 |
| **siglip_vitl16** | sigmoid-loss, L scale | **2/5** (cat_vs_dog, deer_vs_horse) | +0.019 |

## Honest verdict - the ceiling is essentially reached on this arena (not a flat plateau)

1. **The sanity refs reproduce B2-repr exactly.** With CLIP-B as the baseline, resnet18 sits far below it
   (−0.083, 0/5) and dinov2_vits14 **ties** it (−0.002, 0/5). That is the mirror image of B2-repr, where both
   CLIP-B and DINOv2-S beat resnet18 by ~+0.08 - internally consistent, and confirms CLIP-B ≈ DINOv2-S on this
   arena.
2. **Scaling the *same* inductive bias plateaus.** CLIP ViT-B/32 → ViT-L/14 (512→768 d, a far larger tower)
   yields **0/5** with mean **+0.008** - no FDR-surviving lift on any task. CLIP is **saturated at B/32** here;
   making the *same* representation bigger buys nothing measurable. This echoes B2-repr's resnet18→resnet50
   result (1/5) one level up.
3. **Scaling a *different* bias still breaks through, but only where headroom remains.** DINOv2 **S→L** (2/5,
   +0.023) and SigLIP at L scale (2/5, +0.019) both clear FDR - and the wins are concentrated on **cat_vs_dog**,
   the one task that still had real room (0.93). DINOv2-L turns cat_vs_dog **0.930 → 0.9967** (one error in 300)
   and airplane_vs_ship → 1.000. The lever remains *which* representation, and self-supervised / sigmoid
   pretraining at scale has more headroom than language-aligned CLIP - but it is spending that headroom on the
   last unsaturated task.
4. **The aggregate frozen-feature ceiling on the B1 arena is ~0.99 and now effectively hit.** Four of five tasks
   were already ≥0.97 with CLIP-B; the best ViT-L backbone pushes them to 0.97–1.00 and solves the fifth. There
   is no longer measurable room for a better frozen backbone to demonstrate value on most of this suite - the
   2/5 wins exist only because cat_vs_dog had not yet saturated. **This is the signal to change arenas.**

Every lift is vs a strong baseline (tuned GBM + random search) on its own embeddings, on rows never trained on,
bounded by the frozen certifier, FDR-controlled. No gate was weakened; frozen hashes identical before and after.

## Gate to phase #3

The B1 (CIFAR-10 confusable-pairs) arena is **saturated**: even the best frozen ViT-L features leave no
FDR-measurable room on 4/5 tasks, and the 5th is now at 0.997. Continuing up the backbone axis on *this* arena
cannot produce a falsifiable frontier claim. The evidence-driven next move (phase #3) is a **harder arena with
real headroom above CLIP/DINOv2** - fine-grained / low-shot / out-of-distribution vision - where the best frozen
features still leave a gap, so that "beats a competent human / mid-level engineer" stays measurable and the
representation lever (and, on top of it, authored *featurizers*) can be stress-tested where it actually matters.

`ClipImageBackbone` (now covering the timm-backed SigLIP tower via a dummy-probe `feat_dim` fallback) and
`TimmBackbone` remain real, hermetically tested capability wired into this measured comparison - no dark
modules. The added SigLIP path is covered by `test_siglip_backbone_feat_dim_probe_fallback_is_deterministic` in
`tests/test_transfer_backbones.py`.
