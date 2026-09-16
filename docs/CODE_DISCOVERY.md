# Open-Ended Authored-Code Discovery (the anti-menu capstone)

> *"LLM regenerates proposals. The evaluator runs them. The frozen certifier promotes them."*
> This is the rung where the lever stops being a **backbone the system picks** and becomes **code the
> system writes**. A certified champion that carries a `code_patch` is menu-free at the *code* axis: the
> winning source was never in any list it was handed.

## What changed

The regenerative generator already authored full typed recipes over an open backbone zoo. It now also
authors **novel source** - a featurizer or estimator - through one optional hook:

```
RecipeGenerator(code_authorer=<authorer>, config=GeneratorConfig(code_prob>0, code_roles=("featurizer","classifier")))
```

Two interchangeable authorers sit behind that hook (`vfplatform/code_authoring.py`):

| authorer | what it is | role |
|---|---|---|
| `LLMAuthorer` | wraps the audited `authoring.author_estimator` (Claude via `ops.llm_propose`) - the model **invents** a self-contained sklearn-compatible estimator as Python source | the **open-ended** claim |
| `TemplateAuthorer` | a small deterministic library of safe, parametrised source templates (kernel lifts, interaction featurizers, tree ensembles, prototype/kNN) | reproducible **offline default** + the hermetic tests |

Honesty: a template library is a code *generator*, not unbounded invention - it is the deterministic
substitute (mirroring `authoring.py`'s honest-DECLINE philosophy). The open-ended claim rides on
`LLMAuthorer`; the **rigor** below is identical for either, because nothing the authorer emits is trusted.

The `LLMAuthorer` runs a **revise loop**: when an authored estimator is rejected by the frozen gate, the
rejection reason is fed back into the next request's `extra_context` (trusted, instruction-side), so the
model can *read its own rejection log and try again* - and so each retry is **cache-distinct** (identical
context would otherwise replay the same rejected code). The system prompt also states the gate contract
up-front (no `BaseEstimator` subclassing / `sklearn.utils` / `getattr`; compose sklearn estimators; keep it
light). Neither relaxes the gate - admission still decides.

## The non-negotiable path every authored patch travels

```
author source ─▶ STATIC AST gate (allowlist: numpy/sklearn/math; no os/sys/file/network/dunder)
              ─▶ ISOLATED exec   (spawned, rlimited child; wall-clock timeout)
              ─▶ SCIENTIFIC self-test (determinism + permutation-equivariance on synthetic data)
   [rejected] ─▶ reason fed back to the authorer's next attempt (revise loop) - never reaches a sealed peek
              ─▶ [admitted] EXECUTE on the identical sealed rows  (authoring_bridge.run_authored_predict)
              ─▶ cheap screen ─▶ FROZEN Tier-3 certify (paired McNemar + BH-FDR + Clopper-Pearson)
              ─▶ never-peeked GOLD read (read exactly once)
```

Sealed-blind: only **features** cross into the authored `predict()`; the labels are scored on the trusted
side. Unsafe or non-conforming code is rejected **before it can spend a single sealed peek**. The frozen
certifier (`science.py b564fba2` / `sealed.py 30ad6245`, byte-identical) remains the sole promoter.

## The decisive test: pin the representation, let *only code* move the metric

`scripts/code_arena.py` fixes the representation to **raw tabular features** so the backbone axis is inert
and the **only** thing that can move the metric is code the system authors. This is the honest counterpart
to result **B2** ("authoring on a fixed *pretrained* representation adds nothing", 0/N FDR): on raw features
there *is* headroom for a better model, and the question is whether the system can **discover + certify** it
without being told the answer.

## Results - a real, non-binary task-shape matrix (offline `TemplateAuthorer`, seed 0)

| dataset / shape | classes | seed (linear) | champion | certified lift | gold (never-peeked) | `menu_free` |
|---|---|---|---|---|---|---|
| **covtype / multiclass** | 7 | sealed_lb 0.674 | `raw · linear · +code[classifier]` | **+0.102**, survivors 3/3 | **3/3 FDR, confirmed** (n=4284) | **True** |
| **covtype / noisy_label** (20% train flips) | 7 | sealed_lb 0.628 | `raw · linear · +code[classifier]` | **+0.139**, survivors 3/3 | **3/3 FDR, confirmed** (n=4284) | **True** |
| **digits / multiclass** | 10 | sealed_lb 0.931 | `raw · linear` (unchanged) | - (no certified lift) | 0/3 (honest negative) | False |
| **covtype / imbalanced** (natural long-tail) | 7 | - | *run REFUSED* | - | - | - |

Reading the matrix honestly:

- **The code lever fires and certifies** where there is genuine nonlinear headroom (covtype): the system
  *wrote* a tree-ensemble classifier, it was admitted through the sandbox, **beat the linear champion by
  +0.10–0.14 accuracy on real 7-class data**, and the win **survived a never-peeked gold read** (3/3). The
  certificate carries the authored source (`champion_recipe.code_patch`) so it is auditable + replayable.
- **Noisy labels make the lever *larger*, not smaller** - a tree ensemble is robust to the 20% flipped
  TRAIN labels where a linear head is not (+0.139 vs +0.102), and the sealed/gold sets stay clean.
- **No over-claiming.** On `digits`, raw pixels are already near-linearly separable (linear ≈ 0.95), so the
  authored code earns **no** certified lift and is **not** promoted - the certifier refuses to mint a win
  that isn't there.
- **The metric gate holds.** On heavily `imbalanced` covtype the meta-certifier **REFUSES the whole run**
  (`trivial_baseline` + `label_shuffle` probes fire): raw accuracy is gameable under class imbalance, so the
  referee will not certify on it before a single peek is spent. (Proper imbalanced certification needs a
  balanced metric - tracked as future work; the refusal is the *correct* behavior, not a failure.)

Note: recipe labels now name the estimator - `head=linear` (the seed), `head=gbm` (a *closed-axis menu*
move, HistGradientBoosting), or `+code[<role>]` (an *authored* estimator). So `menu_free=True` on covtype is
the **hard** bar: the authored classifier beat *both* the linear seed **and** the gbm menu ceiling.

## Regression task-shape - the SAME frozen certifier, via tolerance-Bernoulli

Regression has a continuous target, so there is no "exact match" to count. Rather than introduce a new
statistical primitive (which would mean touching the frozen certifier), we turn the continuous target into a
**per-row Bernoulli** and reuse Clopper-Pearson + paired McNemar + BH-FDR **byte-identically**:

```
per-row correctness  :  hit_i = 1[ |ŷ_i − y_i| ≤ τ ]          (a 0/1 outcome per row - exactly what the
                                                               frozen primitives already certify)
tolerance            :  τ = τ_frac · std(y_train),  τ_frac = 0.5
```

`τ` is **pre-registered on the TRAIN target only** (never computed from sealed/gold), so it cannot be snooped
to flatter a model. `θ_floor = 0.5` means a champion must place **>50% of rows within τ**. A constant
(median) predictor lands **below** that floor at τ=0.5·σ (measured: diabetes/california/synthetic all
≈0.47–0.49), so the metric is **not gameable by predicting the center** - the meta-certifier's
trivial-baseline probe (now using the within-τ `metric_fn`) enforces exactly this before a peek is spent.

Heads/roles switch by task-shape automatically: classification uses Logistic/HistGradientBoosting**Classifier**
and authors a `classifier`; regression uses Ridge/HistGradientBoosting**Regressor** and authors a `regressor`
(`RandomForest`/`ExtraTrees`/`Nystroem+Ridge`/distance-kNN regressors, all through the *same* frozen sandbox).

One data-hygiene fix was required (and is **not** the frozen certifier): the data-certificate's kNN
label-DISAGREEMENT and class-balance checks are **classification-only** (a continuous target has ~1.0
discrete-disagreement by construction, and a raw-feature kNN is scale-sensitive - on california its within-τ
miss-rate is *worse* than the median while a scaled GBM clears 0.82). `certify_dataset(is_regression=True)`
skips those inapplicable checks; the **non-negotiable near-duplicate-straddle leak check still runs**, and the
"is this benchmark gameable?" question is carried by the meta-certifier framing gate under the within-τ metric.

| dataset / shape | n (rows) | seed (linear) | champion | certified lift | gold (never-peeked) | `menu_free` |
|---|---|---|---|---|---|---|
| **california / regression** | 20 640, τ=0.582 | sealed_lb 0.645 | `raw · linear · head=gbm` | **+0.158**, survivors 3/3 | **3/3 FDR, confirmed** (n=4284) | False |
| **diabetes / regression** | 442, τ=37.37 | sealed_lb 0.477 | `raw · linear` (unchanged) | - (no certified lift) | 0/3 (honest negative) | False |

Reading it honestly:

- **The nonlinear lever fires and certifies on regression** (california): holding the representation pinned
  to raw features, the gbm head certifies **+0.158** over the linear seed (3/3 FDR) and is **gold-confirmed on
  4 284 never-peeked rows**. An **authored regressor the system wrote** *also* certified **+0.116** (3/3 FDR)
  over the linear seed through the frozen sandbox - the code axis is load-bearing on regression too. The gbm
  menu head out-scored the authored regressor *on this dataset*, so the run ends `menu_free=False` (the same
  honest pattern as the covtype LLM runs - we did **not** drop the legitimate gbm competitor to force a win).
- **No over-claiming** (diabetes): a small, near-linear target (n=442) gives the linear seed no headroom it
  can beat under McNemar at 35-row shards, so nothing is promoted and gold is 0/3 - the honest negative.

## Generality of the *split*, not just the task - grouped & temporal generalization

Every result above is still an **i.i.d. row split** (train/sealed/gold are random draws from the same pool).
The harder, frontier-relevant question is whether a certified lever **generalizes off-distribution**. The
arena now carves three split disciplines (`--split {random,grouped,time}`), reusing the audited leak-safe
logic in `vfplatform/splits.py` and the meta-certifier's `split_leak` probe:

- **`grouped`** - whole **groups** are held out (no group straddles train/val/sealed/gold), so the sealed +
  gold rows belong to groups the model **never trained on** - a covariate-shift / extrapolation test. Groups
  are domain-meaningful: **california → coarse spatial cells** on (Latitude, Longitude) (the canonical
  spatial-CV protocol - held-out cells are unseen regions); **covtype → Elevation bands** (unseen elevation
  regimes).
- **`time`** - forward-chaining: train is the **earliest** block, the sealed shards + gold are strictly
  **later** in time (train-on-past, certify-on-future). No future row can leak into training.

`framing()` re-exposes the `groups`/`times` aligned to the `[train, sealed]` order, so the meta-certifier
**REFUSES the whole run before any sealed peek** if a group straddles a boundary or a training row is later
than a sealed row (proven end-to-end in the locks below).

**The headline real-data result is an honest negative that the certifier gets right.** On california, the
gbm's i.i.d. edge is large - but it **evaporates under spatial extrapolation**:

| split | tol. τ | median | linear (within-τ) | **gbm (within-τ)** | gbm − linear |
|---|---|---|---|---|---|
| `random` (i.i.d.) | 0.582 | 0.421 | 0.663 | **0.821** | **+0.158** (certified, gold-confirmed) |
| `grouped` (held-out regions) | 0.610 | 0.500 | 0.627 | **0.628** | **+0.001** → **nothing certifies, gold 0/3** |

So the nonlinear lever that wins i.i.d. **does not generalize to unseen regions**, and the loop **correctly
refuses to promote it** (no manufactured win under distribution shift - exactly the WILDS lesson). covtype
by elevation band is even harder (severe *label* shift: the held-out bands contain classes barely present in
train), so the linear seed itself sits below `θ_floor` and again nothing is promoted - an honest negative.

**That the machinery can still certify under a grouped split (when a lever genuinely generalizes) is proven
on a controlled group-invariant signal:** when every group shares the same nonlinear target and differs only
by a modest covariate shift, the loop certifies **+0.48** over the linear seed, **gold-confirmed 3/3 on
never-peeked held-out groups** (n=990), and an **authored regressor** is itself frozen-certified under the
grouped split. This rules out "the negatives are just broken/underpowered machinery" - the referee certifies
a generalizing lever and refuses a non-generalizing one, both honestly.

## Suite-level generality - a standard OpenML-CC18 slice, not cherry-picked tasks

Every result above is still a *hand-chosen* dataset. "Use benchmark suites, not cherry-picked tasks" means
the generality claim has to be **suite-level**: the **same** loop (RAW features pinned → the only lever is
code the system authors or the gbm head it can reach → frozen Clopper-Pearson + paired McNemar + BH-FDR →
never-peeked gold), the **same** frozen certifier, run across a published suite, reporting **both** the
certified wins **and** the honest negatives. `scripts/run_cc18_suite.py` runs a diverse slice of
**OpenML-CC18 (study 99)** spanning the task-shape matrix - 3..10-class multiclass, numeric **and**
categorical features, balanced **and** imbalanced binary - and writes one summary to `docs/CC18_SUITE.json`.

Two pieces of machinery make this honest across shapes:
- **Data-driven competence floor** `θ = max(0.5, train-majority + 0.03)` (computed on **train labels only**,
  so it cannot be snooped). On balanced sets it stays 0.5; on imbalanced binary it rises to just above the
  trivial baseline - so "competent" means *beats predicting the majority class*, and raw accuracy is not
  gameable.
- **Automatic de-duplication** of exact-duplicate rows in the loaded data (several CC18 sets - e.g. `segment`
  has 242, `splice` 185 - ship repeated feature vectors). Left in, an identical row can land in both train
  and sealed, which the meta-certifier's near-duplicate **straddle** probe (correctly) treats as leakage and
  **refuses**. Dropping all-but-one representative is the hygiene fix; it touches the data, never the certifier.

**Result (7 datasets, offline `TemplateAuthorer`, seed 0, 14-peek budget, frozen `b564fba2`/`30ad6245`):**

| dataset | shape | classes | θ_floor | lin vs best (i.i.d. probe) | verdict |
|---|---|---|---|---|---|
| `segment` | multiclass | 7 | 0.50 | 0.924 → 0.972 (+0.048) | **CERTIFIED +0.030 (authored code), gold-confirmed 3/3** |
| `optdigits` | multiclass | 10 | 0.50 | 0.968 → 0.981 (+0.013) | **CERTIFIED +0.006 (authored code), gold-confirmed 3/3** |
| `vehicle` | multiclass | 4 | 0.50 | 0.791 → 0.762 (**−0.029**) | honest negative - *linear actually wins*, nothing promoted |
| `mfeat_fourier` | multiclass | 10 | 0.50 | 0.806 → 0.831 (+0.025) | honest negative - gap too small to survive FDR (n≈2k) |
| `splice` | multiclass | 3 | 0.50 | 0.905 → 0.969 (+0.064) | honest negative - balanced subsample shrinks the gap; uncertified |
| `credit_g` | imbalanced | 2 | **0.73** | 0.738 → 0.782 (+0.045) | honest negative - linear ≈ floor; thin headroom, uncertified |
| `pima` | imbalanced | 2 | **0.645** | 0.769 → 0.766 (−0.003) | **REFUSED** - majority baseline (0.688) clears θ → accuracy is gameable |

**totals: 2 certified (both gold-confirmed) · 4 honest negatives · 1 refused.**

The reading is exactly the honest one: a **linear head is a strong tabular baseline**, so authored nonlinear
code only certifies a gold-confirmed win where there is **genuine nonlinear headroom *and* enough sealed power**
(`segment` +0.030; `optdigits` +0.006 at n≈5.6k). Where the gap is small or absent the loop **declines** rather
than manufacture a win (`vehicle` linear wins outright; `mfeat`/`splice`/`credit_g` uncertified), and on `pima`
the data-driven floor + meta-certifier together **refuse** the task because predicting the majority class
already clears the bar - raw accuracy there is not a trustworthy metric. No dataset was dropped, hand-tuned, or
re-framed to get a win.

```bash
PYTHONPATH=. python scripts/run_cc18_suite.py            # full slice -> docs/CC18_SUITE.json
PYTHONPATH=. python scripts/run_cc18_suite.py --only segment,optdigits   # a subset
```

## Open-ended `LLMAuthorer`: what Claude actually did (honest)

With `--llm`, Claude authors the source. Observed on covtype (real API, the revise loop live):

- **The pipeline works end-to-end with a real model.** Claude authored a featurizer that cleared all three
  admission stages and was **frozen-Tier-3 certified +0.035 over the linear seed** (gold-confirmed) in the
  mixed-role run - a brand-new transform the system was never handed.
- **The revise loop works.** A first classifier attempt was *rejected* (it used a removed `multi_class`
  kwarg → the scientific self-test caught the `TypeError`); fed that reason, the **next attempt was
  admitted**. Earlier attempts that imported `sklearn.utils.validation` / called `getattr` (Claude's
  instinct to write a "proper" `BaseEstimator` subclass) were correctly **rejected at the static gate**.
- **The certifier is an impartial referee, not a novelty booster.** Re-run with `--roles classifier`
  (`docs/CODE_DISCOVERY_covtype_multiclass_LLM.json`), Claude's authored classifier was **itself
  frozen-Tier-3 CERTIFIED +0.083 over the linear seed (3/3 survivors)** - it cleared the bar honestly. But in
  the *same round* the tuned gbm head certified **+0.122**, so the certifier (argmax over certified lifts)
  **promoted gbm**, and the run ends `menu_free=False`, `uses_authored_code=False`, gold-confirmed 3/3. This
  is the rigor in action: Claude's code is ranked against the built-in competitor on identical sealed rows
  and **wins only if it is actually better**. We did **not** drop the legitimate gbm competitor to force an
  "LLM-as-champion" headline. Where authored code genuinely *does* out-certify the gbm head, it becomes the
  champion - see the suite-level `segment` (+0.030) and `optdigits` (+0.006) rows above
  (`uses_authored_code=True`, deterministic `TemplateAuthorer`).

LLM certificates are written to `docs/CODE_DISCOVERY_<dataset>_<shape>_LLM.json` (with a per-attempt
`llm_calls` provenance log: role, attempt #, admitted?, used_llm?, exact rejection reason).

## Hermetic locks (`tests/test_code_discovery.py`, offline, real frozen primitives)

1. **sandbox admits + executes** a template featurizer *and* classifier (predictions returned).
2. **sandbox rejects unsafe** - `import os` / `import socket` are blocked at the static gate; never executed.
3. **cascade rejects non-improving code** - on a linearly-separable problem, authored code is *proposed*
   every features round but **none is certified**; the champion keeps no code (`menu_free` stays False).
4. **genuine improvement certifies** - on a nonlinear problem, the authored classifier is frozen-Tier-3
   certified over the linear champion (material lift), gold-confirmed → `menu_free=True`,
   `uses_authored_code=True`, `backbone_is_novel=False` (representation was pinned).
5. **LLM revise loop** (offline, monkeypatched) - a rejected authored estimator's reason is fed into the
   next attempt's `extra_context` (cache-distinct + log-informed); provenance records both attempts.
6. **regression certifies via tolerance-Bernoulli** - on a nonlinear continuous target the linear seed is
   below the within-τ bar and the trivial **median** predictor is below `θ_floor` (not gameable by predicting
   the center); a material lift certifies and is gold-confirmed, and an **authored regressor** reaches a frozen
   certification under the within-τ metric (`test_regression_tolerance_bernoulli_certifies`).
7. **regression data-hygiene admits a continuous target** - the classification kNN-disagreement path would
   *falsely refuse* a clean regression split (noise est >0.5); `is_regression=True` skips the inapplicable
   check while the near-duplicate-straddle leak check still blocks (`test_regression_data_hygiene_admits_…`).
8. **grouped split certifies a generalizing lever + refuses a leak** - whole groups are held out (no
   straddle); on a group-invariant signal a material lift certifies and is **gold-confirmed on never-peeked
   held-out groups** (an authored regressor certifies too); a leaky grouped framing (a sealed row sharing a
   train group) is **refused end-to-end with 0 peeks spent** (`test_grouped_split_certifies_…`).
9. **time split is forward-chained + refuses a temporal leak** - max(train time) ≤ min(sealed time) ≤ gold;
   a training row stamped later than the sealed rows is **refused before any peek**
   (`test_time_split_forward_chaining_and_refuses_temporal_leak`).
10. **CC18 loader encodes, de-dups, and caches** (offline, faked fetch) - one-hot categorical + median-impute
   numeric, exact-duplicate rows dropped, target label-encoded, second call served from disk without touching
   the network (`test_openml_loader_encodes_dedups_and_caches`).
11. **competence floor is data-driven + uncheatable** - `θ = max(0.5, train-majority + margin)` stays 0.5 on
   balanced sets and rises above the trivial baseline on imbalanced ones, computed from train labels only
   (`test_auto_theta_floor_is_data_driven_and_uncheatable`).
12. **frozen-core hashes unchanged** (`b564fba2` / `30ad6245`).

## Reproduce

```bash
# deterministic, offline (the table above):
PYTHONPATH=. python scripts/run_code_discovery.py --dataset covtype --shape multiclass
PYTHONPATH=. python scripts/run_code_discovery.py --dataset covtype --shape noisy_label

# regression task-shape (tolerance-Bernoulli; reuses the frozen certifier unchanged):
PYTHONPATH=. python scripts/run_code_discovery.py --dataset california --shape regression
PYTHONPATH=. python scripts/run_code_discovery.py --dataset diabetes  --shape regression

# grouped / temporal generalization (held-out groups = unseen regions; time = train-past/certify-future):
PYTHONPATH=. python scripts/run_code_discovery.py --dataset california --shape regression --split grouped
PYTHONPATH=. python scripts/run_code_discovery.py --dataset covtype    --shape multiclass --split grouped

# suite-level generality across a standard OpenML-CC18 slice (not cherry-picked) -> docs/CC18_SUITE.json:
PYTHONPATH=. python scripts/run_cc18_suite.py

# open-ended (Claude authors the source; --roles narrows what it authors):
ATTESTRA_CODE_LLM=1 PYTHONPATH=. python scripts/run_code_discovery.py --dataset covtype --shape multiclass \
    --llm --roles classifier --out docs/CODE_DISCOVERY_covtype_multiclass_LLM.json
```

Certificates are written to `docs/CODE_DISCOVERY_<dataset>_<shape>.json` (full climb log + authored source);
LLM runs use the `_LLM.json` suffix and add the `llm_calls` provenance log.
