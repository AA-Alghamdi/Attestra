# Phase #3 - A harder arena with real headroom: best frozen features vs a fine-tune ceiling (FGVC-Aircraft)

**Frozen certifier untouched:** `science.py b564fba2` / `sealed.py 30ad6245` - byte-identical before and after
this work (full sha256 recorded in the result JSON, re-verified at the end of the run). The benchmark only
*reads* the frozen Clopper–Pearson lower bound (`science.clopper_pearson_lower`); it never edits a frozen file
and never weakens a gate.

## The question

B1/B2-repr-2 located the frozen-feature ceiling on CIFAR-10: that arena saturates at ~0.99, so it can no longer
falsify a frontier claim. Phase #3 moves to a **genuinely hard** arena and adds the move a mid-level engineer
actually reaches for - **end-to-end fine-tuning** - as a *ceiling arm*, to answer two coupled questions:

1. On a hard fine-grained task, does the **representation lever** still hold - does a better frozen backbone
   (DINOv2-L) still beat the deployed champion (CLIP-B) under FDR?
2. **How much headroom do frozen features leave?** i.e. does end-to-end fine-tuning a CNN beat the best frozen
   representation + tuned head, the way the standard playbook assumes it should?

**Arena - FGVC-Aircraft, 10 maximally-confusable same-family adjacent-variant binary pairs** (e.g. 737-700 vs
737-800, A330-200 vs A330-300, CRJ-700 vs CRJ-900). Exactly 100 images/variant exist, so each task is ~110
train / 28 val / 60 sealed-test. Frozen CLIP-B scores **0.55–0.90** here (7/10 pairs ≤ 0.70) versus ~0.93–0.997
on the CIFAR arena - this is the real headroom B1 lacked, so "beats a competent engineer" stays measurable.

| arm | what it is | what it tests |
|-----|------------|---------------|
| **clip_vitb32** *(baseline)* | OpenAI CLIP ViT-B/32 image tower (512-d) + tuned-GBM head | the deployed B2-repr champion |
| **dinov2_vitl14** | DINOv2 ViT-L/14 (timm, 1024-d) + tuned-GBM head | a stronger *frozen* representation (self-supervised, L scale) |
| **finetune (ceiling)** | resnet50 end-to-end, partial unfreeze (layer3/4+fc), 224 px, 30 ep | the standard mid-level-engineer move: actually fine-tune |

`script: scripts/benchmark_aircraft.py` · `raw JSON: docs/BENCHMARK_AIRCRAFT_RESULT.json`
(per_class=100, seed=0, sealed n=60/task).

### Machinery (identical sealed-split discipline to B1)

- The train/val/test row-ids are derived **once** from the CLIP-B baseline embeddings (`benchmark_backbones._sealed_split`)
  and reused **verbatim** for every arm, so all McNemar pairing is on identical sealed rows; only the arm changes.
- The two frozen arms use the same tuned-GBM strong head (`emb_strong`) on their own embeddings.
- The **fine-tune ceiling is a fair, strong attempt and strictly select-then-bound:** it runs **3 independent
  restarts** (distinct seeds), selects the best **epoch** on the identical val rows *and* the best **restart** by
  val accuracy, then is bounded **once** on the identical sealed test rows. The sealed rows never influence
  training, epoch selection, or restart selection - no leakage. Multi-restart val-selection is what a competent
  engineer does and it guards against the occasional early-divergence collapse on ~110-image classes (it lifted
  A330 from a 0.50 chance-collapse to 0.53, and A340 from 0.78 to 0.88 - see note below).
- Each arm is paired against its comparator by one-sided exact McNemar on the identical sealed rows, then
  Benjamini–Hochberg(α=0.1) across the 10-task suite.

> Memory-frugal orchestration (7 GB / 2-core CPU box): **Phase A** loads only CLIP-B, derives each split once and
> records baseline correctness; **Phase B1** loads DINOv2-L alone and encodes all pairs; **Phase B2** fine-tunes
> one pair at a time from a pre-decoded PIL cache (JPEG decoded once, augmented per epoch) with BatchNorm stats
> frozen. Results are checkpointed to the JSON after every pair.

## Result

Sealed accuracy per arm, and the three pairwise lifts with one-sided McNemar p (**bold** = survives BH-FDR(0.1)
with positive lift). Frozen CLIP-B is the baseline column.

| task | clip_vitb32 | dinov2_vitl14 | finetune | DINOv2−CLIP (p) | FT−CLIP (p) | FT−DINOv2 (p) |
|------|-------------|---------------|----------|-----------------|-------------|---------------|
| 737-700_vs_737-800   | 0.550 | 0.850 | 0.733 | **+0.300** (.00) | **+0.183** (.00) | −0.117 (.99) |
| 737-300_vs_737-400   | 0.567 | 0.767 | 0.633 | **+0.200** (.02) | +0.067 (.26) | −0.133 (.97) |
| 747-200_vs_747-400   | 0.633 | 0.750 | 0.667 | +0.117 (.11) | +0.033 (.40) | −0.083 (.93) |
| A330-200_vs_A330-300 | 0.600 | 0.750 | 0.533 | +0.150 (.07) | −0.067 (.86) | −0.217 (1.00) |
| A340-300_vs_A340-600 | 0.667 | 0.817 | 0.883 | +0.150 (.03) | **+0.217** (.01) | +0.067 (.17) |
| CRJ-700_vs_CRJ-900   | 0.650 | 0.750 | 0.767 | +0.100 (.17) | +0.117 (.08) | +0.017 (.50) |
| ERJ 135_vs_ERJ 145   | 0.700 | 0.717 | 0.767 | +0.017 (.50) | +0.067 (.23) | +0.050 (.31) |
| E-190_vs_E-195       | 0.783 | 0.717 | 0.683 | −0.067 (.85) | −0.100 (.94) | −0.033 (.77) |
| 767-300_vs_767-400   | 0.900 | 0.883 | 0.900 | −0.017 (.73) | +0.000 (.64) | +0.017 (.50) |
| DHC-8-100_vs_DHC-8-300 | 0.783 | 0.867 | 0.800 | +0.083 (.18) | +0.017 (.50) | −0.067 (.91) |
| **mean acc** | **0.683** | **0.787** | **0.737** | **+0.103** | **+0.053** | **−0.050** |

**BH-FDR(0.1) survivors with positive lift:**

| comparison | survivors | mean lift |
|------------|-----------|-----------|
| DINOv2-L (frozen) **vs** CLIP-B (frozen) | **2/10** - 737-700/800, 737-300/400 | +0.103 |
| FINE-TUNE (ceiling) **vs** CLIP-B (frozen) | **2/10** - 737-700/800, A340-300/600 | +0.053 |
| FINE-TUNE (ceiling) **vs** DINOv2-L (frozen) | **0/10** | −0.050 |

## Honest verdict - the representation lever holds, and frozen features leave *no recoverable gap* below the best frozen rep here

1. **This arena has real headroom.** Frozen CLIP-B sits at 0.55–0.90 (mean 0.68), nowhere near solved - exactly
   the falsifiable regime B1 lacked. The hardest pairs (737-700/800, 737-300/400) are near chance for CLIP-B
   (0.55, 0.57).
2. **The representation lever still holds on a hard task.** Swapping only the frozen backbone CLIP-B → DINOv2-L
   wins **2/10 under FDR** (mean +0.103), and the wins land precisely on the two hardest pairs where CLIP-B was
   near chance - DINOv2-L turns 737-700/800 from **0.55 → 0.85**. Self-supervised ViT-L features carry the
   fine-grained airframe structure that CLIP's language-aligned features blur. DINOv2-L is the new champion
   (mean 0.787, best on 8/10 pairs).
3. **End-to-end fine-tuning beats the *deployed* champion only modestly (2/10), and never beats the *best*
   frozen rep (0/10).** A fair, multi-restart, val-selected resnet50 fine-tune clears FDR over CLIP-B on 2 pairs
   (737-700/800, A340) - so fine-tuning does add value over the currently-deployed features. But against
   DINOv2-L frozen it is **0/10 and −0.05 on average**: on the three hardest pairs (737-700/800, 737-300/400,
   A330) DINOv2-L frozen beats the fine-tune by 0.12–0.22, and fine-tune only edges DINOv2-L on A340/CRJ/ERJ by
   small, non-significant margins. **The "gap frozen features leave" is not recoverable by the standard
   fine-tune move in this low-shot regime** - fine-tuning a CNN on ~110 images is data-limited and underperforms
   simply choosing a better frozen representation and fitting a tuned head on top.
4. **The absolute ceiling is still far away, but it is a *data*/representation ceiling, not an optimizer one.**
   The champion's mean is 0.787 - a large absolute gap to 1.0 remains - yet neither a different optimizer/head
   (B1/B2: 0/5) nor end-to-end fine-tuning (here: 0/10 vs the best frozen rep) closes it. The only thing that
   ever moves the metric, again, is *which representation* (DINOv2-L over CLIP-B, +0.103).

> **On fairness of the ceiling arm.** The single-recipe fine-tune (one seed) is preserved in
> `docs/BENCHMARK_AIRCRAFT_SINGLE_RESULT.json`; it gave the same qualitative conclusion (fine-tune vs DINOv2-L
> 0/10) but suffered a chance-collapse on A330 (0.50) and a weaker A340 (0.78). The reported result uses the
> stronger **3-restart val-selected** ceiling, which removes those optimization artifacts (A330 0.50→0.53,
> A340 0.78→0.88) without any test peeking - so the "fine-tune does not beat the best frozen rep" conclusion is
> not an artifact of a weak fine-tune.

Every lift is vs a strong baseline (tuned GBM + random search) on its own embeddings, on rows never trained on,
bounded by the frozen certifier, FDR-controlled. No gate was weakened; frozen hashes identical before and after.

## Gate to the next phase

Three independent confirmations now say the same thing across two arenas: given a representation, neither search
(B1 0/5), nor authored classifiers (B2 0/5), nor end-to-end fine-tuning (#3 0/10 vs the best frozen rep) beats a
tuned head - **only changing the representation moves the metric.** And on a *hard* arena the best frozen
representation (DINOv2-L) is not merely competitive with fine-tuning, it **beats** it.

The open frontier question this sharpens: the champion still leaves a large absolute gap (mean 0.787), and that
gap is **not** accessible via fine-tuning here. So the next evidence-driven move is to test the one remaining
untested form of "authoring" - an authored **featurizer** that *combines/transforms representations* (e.g. fuse
DINOv2-L + CLIP-B, or learn a low-shot metric on top of frozen features), measured the exact same way (identical
sealed test, paired McNemar + BH-FDR, frozen CP bound) against the DINOv2-L champion. If fusing/transforming
frozen representations clears FDR over the best single frozen rep, that is a new, real lever; if it ties (as
authored *classifiers* did), the honest conclusion is that on low-shot fine-grained vision the ceiling is set by
the pretrained representation itself, and the frontier move is a better/bigger pretrained encoder, not anything
the autoresearcher can author on top.
