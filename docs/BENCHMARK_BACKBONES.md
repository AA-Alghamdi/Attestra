# B2-repr - Is the representation the lever? (a stronger/different backbone vs resnet18 emb-strong)

**Frozen certifier untouched:** `science.py b564fba2` / `sealed.py 30ad6245` - byte-identical before and after
this work (full sha256 verified; recorded in the result JSON). The benchmark only *reads* the frozen
Clopper–Pearson lower bound (`science.clopper_pearson_lower`); it never edits a frozen file and never weakens a
gate.

## Why this arena exists

Three independent results agree that, **given a fixed representation**, neither the search cycle (audit
tabular 0/4, B1 vision 0/5) nor LLM-authored novel methods (B2 vision 0/5) beat a tuned GBM. The single lever
that *did* move the metric was the **representation**: B1's `emb-strong` (frozen ImageNet resnet18 embeddings)
beat raw pixels **5/5 FDR**, +0.14–0.22 accuracy. This benchmark attacks that lever directly. On the **same B1
arena** (same five CIFAR-10 confusable binary tasks, same `per_class=500`, same identical sealed test, n=300/
task), does a **stronger or different frozen backbone** beat the resnet18 `emb-strong` baseline?

The comparator is exactly B1's deployed representation. Every arm is scored by the **same strong baseline**
(tuned GBM + random search over the catalog) on its **own** embeddings; only the frozen backbone changes:

| arm | representation | dim | family |
|-----|----------------|-----|--------|
| **resnet18** *(baseline)* | ImageNet-supervised CNN, resize 112 | 512 | the B1 comparator |
| **resnet50** | deeper ImageNet-supervised CNN, resize 224 | 2048 | bigger same-family CNN |
| **clip_vitb32** | OpenAI CLIP image tower (`ViT-B-32-quickgelu`) | 512 | **language-aligned** contrastive |
| **dinov2_vits14** | DINOv2 ViT-S/14 (timm), img 224 | 384 | **self-supervised** ViT |

`script: scripts/benchmark_backbones.py` · `raw JSON: docs/BENCHMARK_BACKBONES_RESULT.json`
(per_class=500, seed=0, sealed n=300/task). Each challenger is paired against resnet18 by one-sided exact
McNemar (challenger > resnet18) on the **identical sealed rows**, then Benjamini–Hochberg(α=0.1) across the
five-task suite, **per backbone**. The split is derived once from the baseline embeddings and reused verbatim
for every backbone, so the pairing is on identical rows.

## Result

Sealed accuracy (frozen Clopper–Pearson lower bound), and lift over resnet18 with its one-sided McNemar p:

| task | resnet18 (lb) | resnet50 Δ (p) | clip_vitb32 Δ (p) | dinov2_vits14 Δ (p) |
|------|---------------|----------------|-------------------|---------------------|
| cat_vs_dog          | 0.830 (.790) | +0.030 *(.14)* | **+0.100** (.000018) | **+0.083** (.00012) |
| automobile_vs_truck | 0.913 (.882) | +0.007 *(.42)* | **+0.057** (.00011)  | **+0.047** (.0047)  |
| deer_vs_horse       | 0.897 (.863) | +0.030 *(.088)*| **+0.083** (.000011) | **+0.080** (.0000097) |
| airplane_vs_ship    | 0.893 (.859) | **+0.047** (.014) | **+0.090** (.0000007) | **+0.103** (.0000000040) |
| bird_vs_frog        | 0.903 (.871) | +0.003 *(.50)* | **+0.083** (.0000023) | **+0.090** (.00000023) |

**BH-FDR(0.1) survivors with positive lift, vs resnet18 emb-strong:**

| challenger | survivors | mean lift |
|------------|-----------|-----------|
| resnet50      | **1/5** (airplane_vs_ship only) | +0.023 |
| **clip_vitb32**   | **5/5** (all tasks) | **+0.083** |
| **dinov2_vits14** | **5/5** (all tasks) | **+0.081** |

## Honest verdict

1. **The representation IS the lever, and resnet18 is NOT the ceiling.** Swapping the frozen backbone from
   resnet18 to **CLIP** or **DINOv2** - changing nothing else, same tuned-GBM head, same sealed test - lifts
   sealed accuracy on **every one of the five tasks**, **5/5 surviving FDR**, mean **+0.083 / +0.081**, with
   p-values as low as 4e-09. On cat_vs_dog (the hardest task) CLIP turns 0.83 into 0.93. This is the **first
   FDR-surviving positive frontier result** after three consecutive honest negatives.
2. **It is the *kind* of representation, not merely a bigger one.** resnet50 (a deeper CNN of the **same
   supervised-ImageNet family**, 4× the dimensions) clears FDR on only **1/5** (mean +0.023): scaling the same
   inductive bias barely helps. The large, consistent wins come from a **different inductive bias** -
   language-aligned contrastive (CLIP) and self-supervised (DINOv2) pretraining. The lever is *which
   representation*, not *how big*.
3. **This sharpens the law from B1/B2 instead of contradicting it.** Given a representation, search and
   authoring add nothing (0/5, 0/5); **changing the representation to a better-aligned one adds large,
   certified value (5/5).** The deployable move that beats a mid-level engineer on small/medium vision data is
   **choosing/learning the representation**, not tuning the optimizer or authoring a clever classifier head on
   top of a fixed one.
4. **Positive result, reported under the same discipline as the negatives.** Every lift is vs a strong baseline
   (tuned GBM + random search), on rows never trained on, bounded by the frozen certifier, FDR-controlled. No
   gate was weakened; the frozen hashes are identical before and after.

## What ships, and the gate to the next phase

`ClipImageBackbone` and `TimmBackbone` are real, tested capability (hermetic tests in
`tests/test_transfer_backbones.py`), interface-compatible with `ImageFeaturizer` exactly like `ResnetBackbone`.
They are **wired into a measured comparison** that shows certified value - not dark modules. The proven
recommendation for this task class: **default the transfer representation to a self-supervised / language-
aligned backbone (DINOv2 or CLIP), not a supervised-ImageNet CNN.**

The evidence now points the next experiment at **going further up the representation axis**, since that is the
only lever that has ever moved the metric here:
- **stronger frozen backbones still** (e.g. CLIP/DINOv2 ViT-L, SigLIP) - does the lift keep climbing, or plateau?
- **an authored *featurizer*** (`role="featurizer"`: a learned/combined representation), measured the same way -
  the one remaining way "authoring" could add value, since authoring a *classifier* on a fixed representation
  is now proven (B2) not to.
- **harder arenas with real headroom above CLIP/DINOv2** (fine-grained / medical / low-shot), where even the
  best frozen features leave room - the only place "beats a competent human" stays falsifiable.

Search/authoring **on top of a fixed representation** remains a proven dead end and will not be wired further.
