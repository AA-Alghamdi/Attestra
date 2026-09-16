# B1 - Vision-transfer headroom arena (the honest strong-baseline test)

**Frozen certifier untouched:** `science.py b564fba2` / `sealed.py 30ad6245` - identical before and after this
work. The benchmark only *reads* the frozen Clopper–Pearson lower bound (`science.clopper_pearson_lower`); it
never edits a frozen file and never weakens a gate.

## Why this arena exists

The audit's decisive finding was that on the tabular OpenML suite the full recursive cycle does **not** beat a
tuned `HistGradientBoosting` baseline (0/4 FDR survivors) - there is no headroom there, so "the system adds
value" is *unfalsifiable*. B1 builds an arena that **has** headroom and an **honest strong baseline**, so any
later generation/transfer work is measured against a comparator that is hard to beat, not a strawman (logistic).

Arena: five confusable CIFAR-10 binary tasks. Three arms, all scored on an **identical sealed test** (n=300),
paired one-sided exact McNemar, Benjamini–Hochberg (α=0.1) across the suite:

| arm | representation | optimizer |
|-----|----------------|-----------|
| **A1 raw-strong** | raw pixels (3072-d) | tuned GBM + random search over the catalog |
| **A2 emb-strong** | frozen ImageNet resnet18 embeddings (512-d) | the same strong baseline |
| **C cycle-emb**   | the same embeddings | the full `run_goal_loop` (the system) |

`script: scripts/benchmark_vision_transfer.py` · `raw JSON: docs/BENCHMARK_VISION_TRANSFER_RESULT.json`
(per_class=500, seed=0, sealed n=300/task).

## Result

| task | raw-strong (lb) | emb-strong (lb) | cycle (lb) | C>raw | C>embGBM | embGBM>raw |
|------|----------------|-----------------|-----------|-------|----------|------------|
| cat_vs_dog          | 0.610 (.561) | 0.827 (.787) | 0.827 (.787) | +0.217 *(p 1.2e-9)* | +0.000 *(p .56)* | +0.217 *(p 2.3e-9)* |
| automobile_vs_truck | 0.713 (.667) | 0.913 (.882) | 0.913 (.882) | +0.200 *(p 4.6e-11)* | +0.000 *(p .75)* | +0.200 *(p 4.6e-11)* |
| deer_vs_horse       | 0.697 (.650) | 0.897 (.863) | 0.890 (.856) | +0.193 *(p 3.6e-10)* | −0.007 *(p .94)* | +0.200 *(p 8.0e-11)* |
| airplane_vs_ship    | 0.753 (.709) | 0.930 (.901) | 0.910 (.878) | +0.157 *(p 4.7e-8)* | −0.020 *(p .97)* | +0.177 *(p 3.0e-10)* |
| bird_vs_frog        | 0.743 (.699) | 0.887 (.852) | 0.910 (.878) | +0.167 *(p 1.4e-9)* | +0.023 *(p .095)* | +0.143 *(p 1.6e-6)* |

**BH-FDR(0.1) survivors with positive lift:**
- **CYCLE(emb) vs RAW-STRONG: 5/5** - first time the system beats a *strong* baseline (tabular was 0/4).
- **EMB-STRONG vs RAW-STRONG: 5/5** - the headroom is real and is captured by **transfer**.
- **CYCLE(emb) vs EMB-STRONG: 0/5** - given the representation, the cycle adds **nothing** over a strong GBM.

## Honest verdict

1. **The headroom on this arena is captured by REPRESENTATION/TRANSFER, not by the search cycle.** Swapping raw
   pixels for a frozen backbone moves accuracy ~0.61→0.83 / 0.70→0.90 (+0.14 to +0.22, all FDR-surviving). The
   cycle on top of that representation is statistically indistinguishable from a tuned GBM on the same
   embeddings (0/5). This *reproduces the audit's tabular finding on an arena that has headroom* - the cycle is
   confirmed **not** the bottleneck.
2. **This is nonetheless the first proof the system beats a strong baseline**, because transfer is now a real,
   tested capability (`ResnetBackbone` → `ImageFeaturizer`). On small/medium vision data the deployable win is
   the *representation*, exactly as the audit's own commercial examples (biopsy, 100-call-ender) require.
3. **Every claim is vs a STRONG baseline (tuned GBM + random search), never logistic** - the strawman that
   inflated the prior PR6 is structurally impossible here (A1/A2 *are* the strong arms).

## Gate to B2

B1's job was to decide whether more optimizer wiring or a different lever is the path forward. **Decision:
do not wire more search/memory/escalation** (0/5 vs a strong baseline, consistent with tabular). The two levers
with measured or plausible headroom are:

- **Representation/transfer** - *proven* here (+0.14–0.22, FDR-surviving). Make it a first-class capability the
  loop can deploy and certify, including on the commercial examples.
- **B2 novel-method authoring beyond catalog composition** - the only untested path to *exceed* a strong
  baseline given a fixed representation. It must be measured on **this same arena**, vs **emb-strong**, with the
  same paired-McNemar + BH-FDR discipline, and ship only if it produces an FDR-surviving lift over A2.

Until B2 shows an FDR-surviving lift over emb-strong, "the cycle beats a strong baseline given the
representation" remains **unproven**, and we say so.
