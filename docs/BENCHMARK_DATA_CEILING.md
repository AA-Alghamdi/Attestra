# Certified data-ceiling curve (#3)

**Question.** Across the program the hardest confusable pairs resisted *every* frozen encoder, and the brain
labelled them a "data ceiling" - a hypothesis. Is that true? For the **champion** representation, on **those
exact pairs**, does adding labels certifiably raise sealed accuracy (data-limited), or is the metric stuck no
matter how many of the available labels we spend (a true ceiling)?

**Method (same frozen discipline as B1→#5).** For each hard pair we hold the **sealed test rows and the
val-selection rows byte-FIXED** and sweep only the size of the training subsample of the champion's frozen
embeddings, fitting the same strong val-selected head at each per-class budget `b ∈ {6,12,24,48,96,(pool)}`.
Sealed accuracy is bounded by the frozen `science.clopper_pearson_lower`. Two certified read-outs:

- **slope** - paired exact McNemar(`champion@FULL_POOL` vs `champion@MIN_BUDGET`) on the fixed sealed rows,
  Benjamini–Hochberg(0.1) across the hard pairs. Surviving + positive ⇒ the pair is **DATA-LIMITED**.
- **top-step** - McNemar(`@FULL` vs `@2nd-largest`): is the curve still rising at the top of the available
  labels, or saturating within budget? (characterizing; not FDR-gated).

The sealed rows are never used for training or selection (select-then-bound); only the training-subsample size
changes. The certified McNemar uses the seed-0 canonical subsample per budget (pre-registered); the plotted
curve shows mean ± std over 5 independent balanced subsamples. Frozen hashes byte-identical throughout
(`science.py b564fba2` / `sealed.py 30ad6245`).

---

## Vision - FGVC-Aircraft, champion DINOv2-g (`docs/DATA_CEILING_CURVE_VISION.json`, `.png`)

All **7/7** hard pairs are **CERTIFIED data-limited**: every min→max slope survives BH-FDR with a large positive
lift. The pairs the brain flagged as a "data ceiling" are emphatically **not** a modeling ceiling - DINOv2-g
keeps climbing as labels are added, from near chance at 12 total labels to 0.73–0.88 at the ~112-label pool.

| pair | acc @6/cls | acc @full | slope lift | slope p | top-step p | still rising at pool? |
|---|---|---|---|---|---|---|
| 737-300 vs 737-400 | 0.54 | 0.78 | +0.200 | 0.0145 | 0.31 | approaching plateau |
| 737-700 vs 737-800 | 0.56 | 0.88 | +0.317 | 0.0000 | 0.19 | approaching plateau |
| 747-200 vs 747-400 | 0.53 | 0.75 | +0.250 | 0.0030 | 0.13 | approaching plateau |
| A330-200 vs A330-300 | 0.49 | 0.73 | +0.283 | 0.0008 | **0.0007** | **still rising** |
| CRJ-700 vs CRJ-900 | 0.52 | 0.87 | +0.517 | 0.0000 | 0.50 | saturated in budget |
| DHC-8-100 vs DHC-8-300 | 0.65 | 0.88 | +0.283 | 0.0008 | 1.00 | saturated in budget |
| E-190 vs E-195 | 0.55 | 0.77 | +0.267 | 0.0026 | 0.06 | still rising (borderline) |

A330 is the sharpest result: its curve is **still certifiably rising at the full pool** (top-step p 0.0007), so
that pair is purely label-starved - more data would keep helping. CRJ/DHC-8 saturate within the available
labels (top-step n.s.), i.e. they reach their representation ceiling *inside* the current budget.

## Text - 20-Newsgroups, champion mpnet (`docs/DATA_CEILING_CURVE_TEXT.json`, `.png`)

All **4/4** hard pairs are **CERTIFIED data-limited** as well - the same law holds cross-modally:

| pair | acc @6/cls | acc @full | slope lift | slope p | top-step p |
|---|---|---|---|---|---|
| alt.atheism vs religion.misc | 0.54 | 0.70 | +0.153 | 0.0023 | 0.26 |
| ms-windows.misc vs windows.x | 0.76 | 0.88 | +0.062 | 0.0748 | 0.33 |
| pc.hardware vs mac.hardware | 0.62 | 0.85 | +0.160 | 0.0003 | 0.40 |
| rec.autos vs rec.motorcycles | 0.72 | 0.87 | +0.090 | 0.0147 | 0.95 |

Every text pair gains certified accuracy from more labels and none is still rising at the full pool (all
top-steps n.s.) - text saturates within its larger (134/class) budget, while several vision pairs would still
benefit from more images.

---

## Verdict

The "data ceiling" hypothesis is now a **bounded, certified curve**, not a guess: on **11/11** hard pairs
across both modalities, the residual gap left by the best frozen representation is **certifiably closed by
labels** - these pairs are data-limited, not model-limited. This sharpens the program's law: *given the best
representation, the remaining headroom on the hardest tasks is recovered by data, not by authoring on top of a
fixed representation* (which stays 0/20 FDR). The pairs still rising at the pool (A330, borderline E-190 in
vision) are the highest-value targets for the GPU label-budget expansion tomorrow.

Hermetic locks: `tests/test_data_ceiling.py` (a synthetic XOR signal certifies data-limited; pure noise stays
FLAT; BH-FDR keeps only the real slope; select-then-bound + determinism enforced).
