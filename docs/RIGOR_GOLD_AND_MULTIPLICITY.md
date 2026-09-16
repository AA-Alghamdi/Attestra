# Rigor core: gold confirmation + session-level multiplicity accounting

The autonomous `ReprResearcher` climbs by querying ONE working sealed set many times (best-first over
representations, frozen Tier-3 the sole promoter). Two failure modes survive any amount of GPU compute and are
purely a discipline problem:

1. **Selection bias from sealed-test re-use.** The final champion is the argmax over every certified look, so
   its sealed lower bound is optimistic - it was *selected* on the same rows it is *bounded* on.
2. **No clean, never-peeked confirmation.** A single held-out read, taken once after all climbing, is the only
   number free of that selection.

Both are addressed here without touching the frozen certifier (`vectorforge/science.py b564fba2`,
`vfplatform/sealed.py 30ad6245`, byte-identical before/after every run).

---

## 1. Gold-confirmation tier (never-peeked, read exactly once)

An arena may expose a `gold_measure(name)` that returns correctness vectors on a partition **disjoint from the
working rows** (train / val / sealed) and never read during the climb. After the brain finishes climbing, the
FINAL champion is certified over the START baseline on gold **exactly once**, using the same frozen primitives
(paired McNemar + BH-FDR + Clopper-Pearson). Because the climb never queried gold, this confirmation carries
none of the sealed-test re-use multiplicity.

- **Text arena (`TwentyNewsArena`)** has real corpus headroom (~600–990 docs/class vs 240 used for
  train+val+sealed), so it carves a `GOLD_PER_CLASS=200` partition deterministically from the same shuffle tail
  - guaranteed disjoint from the working docs. The TF-IDF+LSA baseline transformer is fit on the original
  reference texts and applied to gold in the identical lexical space; neural encoders embed gold through the
  frozen backbone; the head is trained on the train rows only and scored once on gold.
- **Vision arena (`FgvcAircraftArena`)** honestly returns `None`: FGVC-Aircraft pools to exactly 100 images per
  variant and the arena's `per_class=100` already consumes every image, so there is no disjoint gold to carve.
  The brain then emits **no** gold confirmation rather than a fabricated (overlapping) one. A vision source with
  image headroom (ImageNet-/CIFAR-scale) would carve gold exactly as the text arena does.

Honesty locks (hermetic, real frozen functions):
- gold is read exactly once per principal, post-hoc (champion then baseline), never inside the loop;
- flipping the champion's gold labels makes the confirmation **fail even though the sealed climb still
  promotes** - proving gold is an independent check that cannot launder a sealed win;
- a domain with no disjoint gold emits `gold_confirmation=None` (honest degradation).

Tests: `tests/test_repr_researcher.py` (post-hoc / independent / absent), `tests/test_text_arena.py` (disjoint
docs, headroom capping, select-then-bound on gold), `tests/test_repr_arena_vision_gold.py` (honest `None`).

---

## 2. Session-level multiplicity accounting (always present)

Every certificate now carries a `multiplicity` block that discloses the selection bias and re-tests the final
win against a family-wise correction over **all** the looks the climb spent:

```
sealed_comparisons          M   = number of frozen Tier-3 certifications spent climbing (== peeks_used)
mcnemar_tests_total             = M * n_tasks   (total paired tests on the working sealed set)
per_comparison_fdr_alpha    a   = the BH-FDR alpha used at each look (0.1)
session_bonferroni_alpha    a/M = family-wise threshold across all M looks (conservative)
champion_vs_start_sealed        = per-task champ-vs-start acc, lift, McNemar p (already-acquired rows; no new look)
fdr_survivors_nominal           = champ-vs-start tasks surviving BH-FDR(a) with positive lift
bonferroni_survivors_session    = champ-vs-start tasks surviving p < a/M with positive lift  (multiplicity-robust)
robust_to_session_multiplicity  = at least one Bonferroni-session survivor
gold_independent_confirmation   = the multiplicity-FREE gold verdict (True/False), or False when no gold
```

`champion_vs_start_sealed` is a **post-hoc summary** computed from correctness vectors already acquired during
the climb (no new encoder is measured, so M is not inflated by the report). The `bonferroni_survivors_session`
column is deliberately conservative: it asks "even if we Bonferroni-correct for *every* sealed look the climb
took, does the final champion still beat the start baseline?" The definitive, selection-free answer remains the
gold confirmation.

Honesty locks (`tests/test_repr_researcher.py`):
- a genuine champion reports `M == peeks_used`, `mcnemar_tests_total == M*n_tasks`,
  `session_bonferroni_alpha == a/M`, and (for a large win) survives the Bonferroni-session correction;
- when nothing certifies (champion stays the start baseline), the report says so honestly - zero per-task lift,
  no survivors - the correction cannot manufacture a win the sealed rows do not contain.

---

## 3. Measured outcome (real runs, frozen hashes byte-identical)

**Vision (`docs/REPR_RESEARCHER_CERTIFICATE.json`, FGVC-Aircraft, 10 pairs):** the brain climbs
`clip_vitb32 → dinov2_vitl14 → fuse[dinov2_vitl14+siglip_so] → dinov2_g` and honest-stops.
- `sealed_comparisons = 7` (70 paired McNemar tests); `session_bonferroni_alpha = 0.0143`.
- champion `dinov2_g` vs start `clip_vitb32` on sealed: **5 FDR-nominal survivors, 4/10 survive the Bonferroni
  correction over all 7 looks** → `robust_to_session_multiplicity = True`.
- `gold_independent_confirmation = False` (this arena has no disjoint gold, by design).

**Text (`docs/REPR_RESEARCHER_TEXT_CERTIFICATE.json`, 20-Newsgroups):** the brain climbs the lexical baseline to
a frozen neural sentence encoder; the champion is **gold-confirmed** on a never-peeked n≈2400 partition (≥1 FDR
survivor, positive lift), and the multiplicity block reports the sealed looks spent plus the Bonferroni-session
re-test. Here `gold_independent_confirmation = True` - the win stands on rows the climb never touched.

Net: the trust story is now end-to-end. The climb discloses exactly how many sealed looks it spent and how
robust the final win is to that re-use, and - where the data allows - a single never-peeked gold read confirms
the champion free of the selection entirely.
