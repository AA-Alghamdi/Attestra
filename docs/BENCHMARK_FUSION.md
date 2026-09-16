# Phase #4 - The last untested form of authoring: an authored *featurizer* vs the frozen champion (FGVC-Aircraft)

**Frozen certifier untouched:** `science.py b564fba2` / `sealed.py 30ad6245` - byte-identical before and after
this work (full sha256 recorded in the result JSON). The benchmark only *reads* the frozen Clopper–Pearson lower
bound (`science.clopper_pearson_lower`); it never edits a frozen file and never weakens a gate.

## The question

Four phases established one law across two arenas: **given a representation, authoring on top adds nothing.**
Neither the search cycle (B1: 0/5), nor LLM-authored *classifiers* (B2: 0/5), nor end-to-end *fine-tuning*
(#3: 0/10 vs the best frozen rep) beats a tuned GBM head - only *changing* the representation moves the metric
(CLIP/DINOv2 over resnet18: 5/5; DINOv2-L over CLIP-B on the hard arena: 2/10).

That leaves exactly **one untested form of authoring: authoring a REPRESENTATION (a featurizer), not a
classifier.** The hypothesis with a real mechanism: DINOv2-L (self-supervised, fine-grained) and CLIP-B
(language-aligned) plausibly encode **complementary** structure, so a featurizer that *fuses/transforms* them
could beat either alone - which would be the first new authored lever since transfer itself. Phase #4 tests this
on the **identical** FGVC-Aircraft arena and sealed splits as #3, against the **DINOv2-L champion**.

| arm | what it is | what it tests |
|-----|------------|---------------|
| **clip_vitb32** | CLIP ViT-B/32 (512-d) + tuned-GBM head | weak reference |
| **dinov2_vitl14** *(champion)* | DINOv2 ViT-L/14 (1024-d) + tuned-GBM head | the best single frozen rep - the baseline to beat |
| **fuse_concat** | concat[DINOv2-L ‖ CLIP-B] (1536-d) + the SAME tuned-GBM head | authored featurizer: naive fusion of two frozen biases |
| **fuse_pls** | supervised PLS(32) of the per-block-L2 concat, fit on TRAIN rows only, + the SAME head | authored featurizer: a *learned* low-dim representation of the fused space |

`script: scripts/benchmark_fusion.py` · `raw JSON: docs/BENCHMARK_FUSION_RESULT.json` (per_class=100, seed=0,
sealed n=60/task).

### Machinery (identical sealed-split discipline to #3 - verified)

- The train/val/test row-ids are derived **once** from the CLIP-B baseline embeddings
  (`benchmark_backbones._sealed_split`) and reused **verbatim** for every arm - *the same row-ids as phase #3*,
  so all McNemar pairing is on identical sealed rows and the DINOv2-L/CLIP-B columns **reproduce #3 exactly**
  (DINOv2-L vs CLIP-B = 2/10, +0.103, same two pairs - a built-in consistency check).
- Every arm uses the **same** strong tuned-GBM head (`emb_strong` = random-search over GBM/RF/ET/logreg/kNN,
  selected on val), so **only the representation differs**.
- The learned `fuse_pls` featurizer is strictly **select-then-bound**: the PLS projection is fit on the **train
  rows only** (it sees train labels, never val/test), the transform is applied to all rows, and the sealed rows
  are bound exactly once. A hermetic test (`tests/test_fusion_bench.py`) certifies that flipping *only* the
  sealed-test labels inverts the correctness vector elementwise and that PLS.fit receives exactly the train-row
  labels - i.e. no val/test leakage can enter the learned representation.
- Each arm is paired against its comparator by one-sided exact McNemar on the identical sealed rows, then
  Benjamini–Hochberg(α=0.1) across the 10-task suite.

## Result

Sealed accuracy per arm, and the two fusion lifts vs the DINOv2-L champion with one-sided McNemar p
(**bold** = survives BH-FDR(0.1) with positive lift).

| task | clip | dino (champ) | fuse_concat | fuse_pls | concat−dino (p) | pls−dino (p) |
|------|------|------|------|------|------|------|
| 737-700_vs_737-800     | 0.550 | 0.850 | 0.750 | 0.683 | −0.100 (.99) | −0.167 (1.00) |
| 737-300_vs_737-400     | 0.567 | 0.767 | 0.733 | 0.667 | −0.033 (.77) | −0.100 (.95) |
| 747-200_vs_747-400     | 0.633 | 0.750 | 0.750 | 0.767 | +0.000 (.60) | +0.017 (.50) |
| A330-200_vs_A330-300   | 0.600 | 0.750 | 0.517 | 0.550 | −0.233 (1.00) | −0.200 (1.00) |
| A340-300_vs_A340-600   | 0.667 | 0.817 | 0.750 | 0.800 | −0.067 (.93) | −0.017 (.73) |
| CRJ-700_vs_CRJ-900     | 0.650 | 0.750 | 0.800 | 0.733 | +0.050 (.29) | −0.017 (.71) |
| ERJ 135_vs_ERJ 145     | 0.700 | 0.717 | 0.800 | 0.717 | +0.083 (.11) | −0.000 (.61) |
| E-190_vs_E-195         | 0.783 | 0.717 | 0.783 | 0.767 | +0.067 (.21) | +0.050 (.25) |
| 767-300_vs_767-400     | 0.900 | 0.883 | 0.967 | 0.983 | +0.083 (.03) | +0.100 (.02) |
| DHC-8-100_vs_DHC-8-300 | 0.783 | 0.867 | 0.817 | 0.800 | −0.050 (.87) | −0.067 (.93) |
| **mean acc** | **0.683** | **0.787** | **0.767** | **0.747** | **−0.020** | **−0.040** |

**BH-FDR(0.1) survivors with positive lift:**

| comparison | survivors | mean lift |
|------------|-----------|-----------|
| **FUSE-CONCAT vs DINOv2-L (champion)** | **0/10** | −0.020 |
| **FUSE-PLS vs DINOv2-L (champion)** | **0/10** | −0.040 |
| FUSE-CONCAT vs CLIP-B | 2/10 - 737-700/800, 737-300/400 | +0.083 |
| DINOv2-L vs CLIP-B *(sanity, reproduces #3)* | 2/10 - 737-700/800, 737-300/400 | +0.103 |

## Honest verdict - authoring a featurizer does **not** beat the single best frozen encoder; the ceiling *is* the encoder

1. **Both authored featurizers fail to beat the champion (0/10, mean −0.02 and −0.04).** Fusing CLIP-B into
   DINOv2-L - naively (concat) or via a learned supervised projection (PLS) - is, on average, **worse** than
   DINOv2-L alone. The only nominally-significant fusion win is on **767-300/400** (the *easiest* pair, already
   0.88; concat p=.03, pls p=.02), and it does **not** survive BH-FDR (rank-1 threshold 0.01) - and it is not
   where the headroom is.
2. **The mechanism is clear and predicts the result.** On exactly the hard pairs where headroom exists
   (737-700/800, 737-300/400, A330), CLIP-B is **near chance** (0.55–0.60), so adding its 512 dims injects mostly
   noise - fusion *hurts* there by 0.10–0.23. A second representation can only help if it carries complementary
   signal *where the champion is weak*; CLIP-B does not. No amount of authored fusion of *these two* reps can
   manufacture signal that neither encoder contains.
3. **This is the 5th independent confirmation of the same law**, now closing the last gap: given the best
   available representation, neither search (0/5), nor authored classifiers (0/5), nor fine-tuning (0/10), nor an
   authored **featurizer** (0/10 concat, 0/10 PLS) beats a tuned head on the single best frozen rep. The lever is
   the representation, and on low-shot fine-grained vision the **ceiling is set by the pretrained encoder
   itself** - not by anything the autoresearcher can author on top of it.
4. **No escalation to LLM-authored featurizers.** The plan was to escalate to LLM-authored featurizers *only if*
   the deterministic probe showed signal. It shows the opposite (fusion dilutes on the hard pairs), and the
   mechanism - CLIP-B carries no complementary signal where DINOv2-L is weak - means a fancier learned fusion of
   these two reps has no information to exploit. Spending compute there would be chasing noise; the honest call
   (wire-or-delete, negatives-as-deliverables) is to stop and report.

Every lift is vs a strong baseline (tuned GBM + random search) on its own representation, on rows never trained
on, bounded by the frozen certifier, FDR-controlled. No gate was weakened; frozen hashes identical before/after.

## Gate to the next phase

The evidence now points with five confirmations at a single frontier lever: **a better/stronger pretrained
encoder.** Authoring on top of the best frozen rep - classifier, fine-tune, or featurizer - is exhausted as a
source of certified lift on this arena. The pre-registered next move (phase #2 from the #3 gate) is therefore the
right one: measure whether an **even stronger frozen encoder** (e.g. DINOv2-g/1B, EVA-02, SigLIP-SO400M) beats
DINOv2-L on this hard arena under the identical discipline. If it climbs, the frontier is "find/curate the best
representation" (a real, certifiable capability); if even the largest encoders plateau here, the residual gap is
a **data** ceiling (label budget / class count), which reframes the mission toward data acquisition rather than
modeling - itself a first-class, honestly-bounded finding.
