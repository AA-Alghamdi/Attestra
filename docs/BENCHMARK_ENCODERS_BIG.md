# Phase #5 - Does a *bigger* frozen encoder keep climbing, or has the representation lever plateaued? (FGVC-Aircraft)

**Frozen certifier untouched:** `science.py b564fba2` / `sealed.py 30ad6245` - byte-identical before and after
this work (full sha256 recorded in the result JSON). The benchmark only *reads* the frozen Clopper–Pearson lower
bound (`science.clopper_pearson_lower`); it never edits a frozen file and never weakens a gate.

## The question

Five phases established one law across two arenas: **given a representation, authoring on top adds nothing.**
Neither the search cycle (B1: 0/5), nor LLM-authored *classifiers* (B2: 0/5), nor end-to-end *fine-tuning*
(#3: 0/10 vs the best frozen rep), nor an authored *featurizer* (#4: 0/10 concat, 0/10 PLS) beats a tuned GBM
head - only *changing* the representation moves the metric (CLIP/DINOv2 over resnet18: 5/5; DINOv2-L over CLIP-B
on the hard arena: 2/10).

That makes the pre-registered question sharp: **is DINOv2-L the ceiling, or does a still-stronger frozen encoder
keep climbing?** Phase #5 swaps in three larger encoders against the **DINOv2-L champion** on the **identical**
FGVC-Aircraft arena and sealed splits as #3/#4 - only the encoder changes, the tuned-GBM head and the sealed rows
are held fixed.

| arm | what it is | what it tests |
|-----|------------|---------------|
| **clip_vitb32** | CLIP ViT-B/32 (512-d) + tuned-GBM head | weak reference |
| **dinov2_vitl14** *(champion)* | DINOv2 ViT-L/14 (1024-d) + tuned-GBM head | the best single frozen rep - the baseline to beat |
| **siglip_so** | SigLIP SO400M/14 (428M, 1152-d) + the SAME head | a *larger, different* bias (sigmoid language-aligned) |
| **eva02_l** | EVA-02 ViT-L/14 MIM (304M, 1024-d) + the SAME head | a *different* self-supervised bias (masked-image modeling) |
| **dinov2_g** | DINOv2 ViT-g/14 (1.1B, 1536-d) + the SAME head | *more of the winning family* (scale the SSL bias that already won) |

`script: scripts/benchmark_encoders_big.py` · `raw JSON: docs/BENCHMARK_ENCODERS_BIG_RESULT.json`
(per_class=100, seed=0, sealed n=60/task).

### Machinery (identical sealed-split discipline to #3/#4 - verified)

- The train/val/test row-ids are derived **once** from the CLIP-B baseline embeddings
  (`benchmark_backbones._sealed_split`) and reused **verbatim** for every arm - *the same row-ids as phase #3*,
  so all McNemar pairing is on identical sealed rows and the DINOv2-L/CLIP-B columns **reproduce #3 exactly**
  (DINOv2-L vs CLIP-B = 2/10, +0.103, same two pairs - a built-in consistency check).
- Every arm uses the **same** strong tuned-GBM head (`emb_strong` = random-search over GBM/RF/ET/logreg/kNN,
  selected on val), so **only the representation differs**.
- Each challenger is paired against the DINOv2-L champion by one-sided exact McNemar on the identical sealed
  rows, then Benjamini–Hochberg(α=0.1) across the 10-task suite. A hermetic test
  (`tests/test_encoders_big_bench.py`) certifies that the sealed rows are encoder-independent (derived from
  (y, seed) only), that flipping *only* the sealed-test labels inverts the correctness vector elementwise (no
  label leakage), and that the on-disk embedding cache returns byte-identical matrices on a hit and *misses*
  when the split size changes (no stale reuse across encoders).

## Result

Sealed accuracy per arm, and each challenger's lift vs the DINOv2-L champion with one-sided McNemar p
(**bold** = survives BH-FDR(0.1) with positive lift).

| task | clip | dino (champ) | siglip_so | eva02_l | dinov2_g | g−dino (p) |
|------|------|------|------|------|------|------|
| 737-700_vs_737-800     | 0.550 | 0.850 | 0.667 | 0.583 | 0.883 | +0.033 (.38) |
| 737-300_vs_737-400     | 0.567 | 0.767 | 0.700 | 0.533 | 0.767 | −0.000 (.61) |
| 747-200_vs_747-400     | 0.633 | 0.750 | 0.750 | 0.583 | 0.750 | +0.000 (.62) |
| A330-200_vs_A330-300   | 0.600 | 0.750 | 0.483 | 0.583 | 0.733 | −0.017 (.73) |
| A340-300_vs_A340-600   | 0.667 | 0.817 | 0.733 | 0.683 | **0.933** | **+0.117 (.008)** |
| CRJ-700_vs_CRJ-900     | 0.650 | 0.750 | 0.833 | 0.633 | **0.883** | **+0.133 (.019)** |
| ERJ 135_vs_ERJ 145     | 0.700 | 0.717 | 0.850 | 0.750 | **0.917** | **+0.200 (.001)** |
| E-190_vs_E-195         | 0.783 | 0.717 | 0.800 | 0.817 | 0.750 | +0.033 (.38) |
| 767-300_vs_767-400     | 0.900 | 0.883 | 0.983 | 0.950 | 0.933 | +0.050 (.19) |
| DHC-8-100_vs_DHC-8-300 | 0.783 | 0.867 | 0.800 | 0.883 | 0.883 | +0.017 (.50) |
| **mean acc** | **0.683** | **0.787** | **0.760** | **0.700** | **0.825** | **+0.057** |

**BH-FDR(0.1) survivors with positive lift (vs the DINOv2-L champion):**

| comparison | survivors | mean lift |
|------------|-----------|-----------|
| **DINOv2-g (1.1B) vs DINOv2-L** | **3/10** - A340-300/600, CRJ-700/900, ERJ 135/145 | **+0.057** |
| SigLIP-SO400M (428M) vs DINOv2-L | 0/10 | −0.027 |
| EVA-02-L MIM (304M) vs DINOv2-L | 0/10 | −0.087 |
| DINOv2-L vs CLIP-B *(sanity, reproduces #3)* | 2/10 - 737-700/800, 737-300/400 | +0.103 |

## Honest verdict - the representation lever is **still open**: scaling the *winning* family climbs; a bigger *different* family does not

1. **DINOv2-g is the first FDR-surviving positive since transfer itself: 3/10 over the champion, mean +0.057.**
   Scaling the self-supervised family that already won (DINOv2 L→g, 304M→1.1B) breaks through BH-FDR(0.1) on
   three pairs - A340-300/600 (+0.117, p=.008), CRJ-700/900 (+0.133, p=.019), ERJ 135/145 (+0.200, p=.001) - and
   is net-positive on the suite. After five straight "authoring adds nothing" negatives, this confirms the one
   lever that *does* move the metric - the representation - has **not** plateaued at DINOv2-L.
2. **It is specifically *more of the winning bias*, not "a bigger model."** The two *larger-but-different*
   encoders both come in **net-negative**: SigLIP-SO400M (428M, sigmoid language-aligned) at −0.027 and
   EVA-02-L (304M, masked-image modeling) at −0.087, **0/10 each**. Size alone buys nothing; only scaling the
   *specific* self-supervised representation that already won on fine-grained pays off. This sharpens the
   B2-repr law: the *kind* of representation is decisive, and **within the winning kind, scale is a real,
   certifiable lift.**
3. **The wins land on a *different* tier of pairs than the previous rep step - and the hardest pairs are now a
   data ceiling.** DINOv2-L beat CLIP-B on the two *hardest* pairs (the 737 variants, +0.30/+0.20). DINOv2-g beats
   DINOv2-L on the *mid-difficulty* pairs (A340/CRJ/ERJ) while the hardest 737 pairs stay stuck (+0.033/−0.000
   even at 1.1B). Each rung of the representation ladder cracks the next tier of separability; the residual
   hardest pairs no longer respond to *any* encoder we can download, which points to a **data/label ceiling**
   on those specific variants (more labeled examples or finer supervision), not a modeling one.
4. **Wire-or-delete, negatives-and-positives both first-class.** All three challengers are wired into one measured
   comparison with a hermetic leakage test; the two that lost (SigLIP, EVA-02) are reported as honest negatives,
   the one that won (DINOv2-g) is bounded by the frozen certifier exactly once on rows never trained on.

Every lift is vs a strong baseline (tuned GBM + random search) on its own representation, on rows never trained
on, bounded by the frozen certifier, FDR-controlled. No gate was weakened; frozen hashes identical before/after.

## Gate to the next phase

Phase #5 establishes that **"select/scale the best-aligned pretrained representation" is a real, FDR-certifiable
frontier lever** - the autoresearcher climbs by moving DINOv2-L→g, not by authoring on top of a fixed rep. Two
evidence-driven directions follow, and they are complementary:

- **Keep climbing the winning family** - does DINOv2-g→ even larger / an ensemble of DINOv2 scales keep cracking
  the mid pairs, and where does *that* plateau? This continues to locate the frozen-feature ceiling on the pairs
  that still have headroom.
- **Attack the residual hardest pairs as a data problem** - the 737 variants resist every frozen encoder
  (CLIP-B 0.55 → DINOv2-g 0.88 is the rep-driven gain, but L→g adds nothing there). If the lever on those pairs
  is labels/data rather than representation, the honest next move is a **data-ceiling experiment** (vary
  per-class label budget and measure the accuracy curve under the same FDR discipline) - turning "the residual
  gap is a data ceiling" from a hypothesis into a bounded, certified curve.
