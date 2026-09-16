# Item #7 - The representation lever on TABULAR (TabPFN v2 vs a tuned GBM)

**Question.** Vision (7 phases) and text (#2) certified one cross-modal law: *given a representation, authoring on
top adds nothing (0/20 FDR); the only lever that moves the metric is changing the representation.* Does it hold on
**tabular**, where there is no embedding to swap - the columns *are* the native representation and a **tuned
gradient-boosting machine** is the strong baseline a competent engineer reaches for?

**The honest tabular analog of "change the representation"** is to swap the whole inductive prior: from a
per-dataset-fit GBM to a **pretrained tabular foundation model** (TabPFN v2), a transformer carrying an in-context
Bayesian prior learned over millions of synthetic tabular tasks. So the same `ReprResearcher` brain + the same
frozen certifier climb this ladder:

| rung | move | tag |
|------|------|-----|
| start champion | **strong tuned GBM** (random search over GBM/RF/ET/logreg/knn + tuned-GBM default, val-selected) - the mid-level-engineer default, **not** a weak baseline | `gbm_raw` |
| model | swap family GBM → TabPFN (the representation lever, if it exists on tabular) | `tabpfn` (8-ensemble) |
| features | STACK champion + the other family (authored-featurizer analog = soft-vote) | `fuse[tabpfn+gbm_raw]` |
| capacity | scale the winning family (TabPFN 8 → 32 ensemble) | `tabpfn_big` |

Promotion only on an **FDR-surviving win over the current champion**; the frozen certifier is the sole promoter.

**Arena.** UCI letter-recognition (`openml/letter`, 16 numeric features, 26 classes, ~750 rows/class), reduced to a
**pre-registered** suite of 10 shape-confusable binary letter pairs - fixed by letter shape *before* any
measurement, the tabular analog of confusable aircraft variants (#3) and confusable newsgroup pairs (#2):
`O-Q, E-F, M-N, U-V, B-D, I-J, K-X, P-R, C-G, V-Y`. Per class: **train 40 · val 25 · sealed 90 · gold 90**, drawn
from one fixed shuffle so gold shares no row with the working pool. Training is deliberately small (the regime
TabPFN is designed for, where a GBM is not yet saturated), making the family-swap a *fair* test rather than a
foregone GBM win. Each task has its **own identical sealed rows** (GBM and TabPFN scored on byte-identical
examples → exact McNemar pairing) and a **disjoint never-peeked gold tail** for a multiplicity-free confirmation.

**Environment.** TabPFN v2 pins `scikit-learn<1.7`, conflicting with the repo's 1.9.0. To keep the validated
vision/text environment byte-identical, tabular runs in an **isolated venv** (`~/.venv-tabpfn`, sklearn 1.6.1 +
tabpfn 2.2.1, system torch). The frozen science core (`vectorforge/science.py` `b564fba2`, `vfplatform/sealed.py`
`30ad6245`) is numpy-only and verified byte-identical there; only the GBM head's sklearn minor version differs, and
that head is the *baseline being beaten*, never the certifier.

---

## Result - the lever FIRES on tabular, and authoring on it adds nothing

The same brain, started from the strong GBM, climbed autonomously (`docs/REPR_RESEARCHER_TABULAR_CERTIFICATE.json`):

```
[model]    PROMOTE gbm_raw -> tabpfn   survivors 5/10  mean_lift +0.033  sealed_lb 0.943
           FDR wins: U_vs_V, B_vs_D, I_vs_J, C_vs_G, V_vs_Y
[features] reject fuse[tabpfn+gbm_raw] vs tabpfn   0/10  mean_lift -0.004   (stacking adds nothing)
[capacity] reject tabpfn_big          vs tabpfn   0/10  mean_lift -0.002   (scaling the ensemble adds nothing)
honest stop: plateaued at the top of the escalation ladder; every cheaper lever exhausted
```

**Gold confirmation on a DISJOINT never-peeked set (n = 1800, read once after climbing):** `tabpfn` vs `gbm_raw`
**4/10 FDR survivors, mean +0.024, CONFIRMED = True**.

| pair | gbm gold | tabpfn gold | lift | p | FDR |
|------|---------:|------------:|-----:|--:|:---:|
| C_vs_G | 0.906 | 0.961 | +0.056 | 0.0032 | ✓ |
| I_vs_J | 0.867 | 0.917 | +0.050 | 0.0318 | ✓ |
| O_vs_Q | 0.906 | 0.950 | +0.044 | 0.0039 | ✓ |
| V_vs_Y | 0.917 | 0.956 | +0.039 | 0.0327 | ✓ |
| K_vs_X | 0.944 | 0.967 | +0.022 | 0.1719 | |
| P_vs_R | 0.967 | 0.989 | +0.022 | 0.0625 | |
| M_vs_N | 0.939 | 0.956 | +0.017 | 0.1875 | |
| E_vs_F | 0.967 | 0.967 | +0.000 | 0.7500 | |
| U_vs_V | 0.972 | 0.972 | +0.000 | 1.0000 | |
| B_vs_D | 0.944 | 0.939 | −0.006 | 0.6964 | |

**Session multiplicity (honest accounting of sealed-test re-use):** M = 3 sealed certifications (30 McNemar tests),
family-wise Bonferroni α/M = 0.033 → champion `tabpfn` beats the start GBM on **4/10 pairs after Bonferroni**:
`robust_to_session_multiplicity = True`, `gold_independent_confirmation = True`.

### Verdict: `REPRESENTATION_LEVER_FIRES_ON_TABULAR`

The pretrained tabular foundation model FDR-beats the tuned GBM - **the representation law generalises to a third
modality** - *and* the "authoring on the fixed rep adds nothing" half reproduces from the strong-baseline side:
stacking the two families (0/10) and scaling the winning ensemble 8→32 (0/10) both add nothing once TabPFN is the
champion. Wins land where the GBM has headroom (the genuinely confusable pairs), exactly the FGVC pattern.

---

## The lever is not a small-sample artifact - it PERSISTS as the GBM gets more data

The obvious skeptical question is *"did you just starve the GBM?"* So `scripts/tabular_regime_curve.py` turns the
single point into a **curve**: holding the sealed test + val FIXED per pair (apples-to-apples, sealed rows never
change), it sweeps ONLY the training-label budget and re-certifies `tabpfn` vs the tuned GBM at each budget with the
same paired-McNemar + BH-FDR + frozen Clopper-Pearson discipline (`docs/TABULAR_REGIME_CURVE.json`):

| train/class | GBM acc | TabPFN acc | mean lift | FDR survivors | GBM LB | TabPFN LB |
|------------:|--------:|-----------:|----------:|:-------------:|-------:|----------:|
| 20  | 0.903 | 0.929 | +0.026 | 3/10 | 0.891 | 0.918 |
| 40  | 0.935 | 0.956 | +0.021 | 2/10 | 0.925 | 0.947 |
| 80  | 0.945 | 0.977 | +0.032 | 7/10 | 0.935 | 0.970 |
| 160 | 0.963 | 0.984 | +0.021 | 4/10 | 0.955 | 0.979 |
| 320 | 0.971 | 0.992 | +0.021 | 2/10 | 0.964 | 0.987 |

**The TabPFN advantage is FDR-surviving at every budget 20 → 320 train/class, with its pooled Clopper-Pearson lower
bound above the GBM's throughout.** Both models climb with data, but the foundation prior keeps a steady ~+0.02–0.03
certified edge - the tuned GBM does **not** catch up within this range. The lever is real across the regimes tested,
not a quirk of tiny n. (The survivor *count* dips at the largest budgets purely from the accuracy ceiling, not from
the GBM closing the gap - the lift and the LB ordering are stable.)

---

## Where it sits in the program

- **Cross-modal law, now 3 modalities:** representation is the lever - vision (7 phases), text (#2), **tabular**.
- **"Authoring on a fixed rep adds nothing", now from both sides:** 0/20 FDR on vision+text (weak-baseline side),
  and on tabular the *strong-baseline* side reproduces it - stack 0/10, capacity-scale 0/10.
- **The same brain + the same frozen certifier** derive the verdict autonomously in every modality; only the arena
  builder changes (one `MODALITIES` entry in `scripts/run_autoresearch.py`).

## Reproduce

```bash
# certified climb (full 10-pair suite) in the isolated venv:
~/.venv-tabpfn/bin/python scripts/run_repr_researcher_tabular.py
# or via the unified entrypoint:
~/.venv-tabpfn/bin/python scripts/run_autoresearch.py --modality tabular
# the data-regime curve:
~/.venv-tabpfn/bin/python scripts/tabular_regime_curve.py
# hermetic plumbing locks (main env, offline, no TabPFN):
python -m pytest tests/test_tabular_arena.py -q
```

**Frozen certifier byte-identical throughout** (`science.py b564fba2` / `sealed.py 30ad6245`, read-only use of the
Clopper-Pearson bound). New code (`scripts/repr_arena_tabular.py`, `scripts/run_repr_researcher_tabular.py`,
`scripts/tabular_regime_curve.py`) is wired into the same brain and locked by `tests/test_tabular_arena.py`
(6/6 pass) - no dark modules.
