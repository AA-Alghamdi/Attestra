# B2 - Novel-method authoring vs a strong baseline (given the representation)

**Frozen certifier untouched:** `science.py b564fba2` / `sealed.py 30ad6245` - byte-identical before and after
this work (full sha256 verified). The benchmark only *reads* the frozen Clopper–Pearson lower bound
(`science.clopper_pearson_lower`); it never edits a frozen file and never weakens a gate.

## Why this arena exists

B1 proved on a headroom vision arena that the win is captured by **representation** (frozen resnet18
embeddings, 5/5 FDR over raw pixels) and that the search **cycle adds nothing** over a tuned GBM on those
embeddings (0/5). That leaves exactly one untested lever that could *exceed* a strong baseline **given a fixed
representation**: a full-strength LLM **authoring genuinely novel methods** - not picking catalog cells, not
composing known sklearn parts, but inventing an estimator with a different inductive bias. B2 measures that
lever, on the **same arena, same sealed test, same discipline** as B1.

Two arms, both on the **identical sealed test** B1 used (n=300/task), paired one-sided exact McNemar,
Benjamini–Hochberg (α=0.1) across the suite:

| arm | representation | method |
|-----|----------------|--------|
| **A2 emb-strong** | frozen ImageNet resnet18 embeddings (512-d) | tuned GBM + random search over the catalog (the comparator) |
| **D authored**    | the same embeddings | best-on-**validation** LLM-authored method, bounded **once** on the sealed test (select-then-bound, checks=1) |

`script: scripts/benchmark_authoring.py` · `raw JSON: docs/BENCHMARK_AUTHORING_RESULT.json`
(per_class=500, seed=0, sealed n=300/task, author model `claude-opus-4-8`).

## The admission gate is the only filter (never a quality bar)

The LLM authored **15** candidate methods; the **frozen three-stage admission gate** - (1) static AST allowlist
(denies `os`/`sys`/file/`eval`/`exec`/import-tricks/numpy C-extension I/O), (2) isolated spawned exec under
rlimits with the environment scrubbed, (3) a scientific self-test on synthetic data (valid predictions,
determinism, permutation-equivariance so it cannot read the sealed test, train-only fit) - admitted **8
distinct** ones on **SAFETY + BUILDABILITY + CONTRACT only**. Quality is decided **solely** by the frozen lower
bound on the sealed test; the LLM never promotes. Admitted methods (genuinely different inductive biases, not
catalog cells):

1. `prototype_cosine_gated_logistic` - RBF-softmax cosine similarity to class-conditional KMeans prototypes
2. `random_hyperplane_tessellation_logistic` - soft-sign half-space products forming soft polytope cells
3. `cosine_nystrom_pairwise_intera` - low-rank Nyström cosine-kernel feature map + explicit pairwise products
4. `sparse_random_projection_margi` - committee of univariate Gaussian quadratic discriminants in sparse subspaces
5. `angular_margin_prototype_diffe` - signed contrasts + order-statistic margins of cosine affinity to each class
6. `random_fourier_orthogonal_subs` - orthogonal-block Random Fourier Features (Bochner RBF map) + ensemble
7. `class_conditional_random_subsp` - committee of class-conditional discriminants in random subspaces
8. `copula_rank_spectral_cca_logis` - monotone-invariant Gaussian-copula rank scoring + label-correlation CCA

When a candidate was rejected (e.g. an import outside the allowlist, or a removed sklearn kwarg) the author was
shown **its own rejection reason** and allowed to revise (≤2×) - the frozen gate still decided admission.

## Result

| task | emb-strong (lb) | authored (lb) | winner | D>embGBM | embGBM>D |
|------|-----------------|---------------|--------|----------|----------|
| cat_vs_dog          | 0.830 (.790) | 0.807 (.765) | prototype_cosine_gated_logistic | −0.023 *(p .92)* | +0.023 *(p .15)* |
| automobile_vs_truck | 0.913 (.882) | 0.913 (.882) | cosine_nystrom_pairwise_intera  | +0.000 *(p .59)* | +0.000 *(p .59)* |
| deer_vs_horse       | 0.897 (.863) | 0.910 (.878) | prototype_cosine_gated_logistic | +0.013 *(p .28)* | −0.013 *(p .84)* |
| airplane_vs_ship    | 0.893 (.859) | 0.907 (.874) | cosine_nystrom_pairwise_intera  | +0.013 *(p .27)* | −0.013 *(p .85)* |
| bird_vs_frog        | 0.903 (.871) | 0.887 (.852) | copula_rank_spectral_cca_logis  | −0.017 *(p .89)* | +0.017 *(p .20)* |

**BH-FDR(0.1) survivors with positive lift - AUTHORED vs EMB-STRONG: 0/5.**

## Honest verdict

1. **Novel-method authoring does NOT beat a tuned GBM on these embeddings.** The authored arm *ties* emb-strong:
   lifts span −0.023 to +0.013, every p ≥ 0.27, **0/5** surviving FDR. Two tasks are marginally positive, two
   marginally negative, one exact tie - noise around the strong baseline, not a win.
2. **This is the third independent confirmation of the same law.** Audit (tabular): cycle vs strong GBM 0/4.
   B1 (vision headroom): cycle vs emb-strong 0/5. B2 (vision headroom): authoring vs emb-strong 0/5. **Given a
   fixed strong representation, neither search nor novel-method authoring beats a tuned GBM.** This is now a
   robust, reproduced finding, not a one-off.
3. **The one proven lever remains the REPRESENTATION.** B1: emb-strong vs raw pixels = 5/5 FDR, +0.14–0.22 acc.
   The metric moves when the *representation* changes, not when the *method on top of a fixed representation*
   changes. That is where the remaining headroom is, and where work should go next.
4. **Negative result, reported as a first-class deliverable** - per the audit's rules (3, 4) and §7. No claim of
   value is made without an FDR-surviving lift, and none exists here.

## Gate to the next phase (B2-repr)

Because the metric only moved when the representation moved (B1) and did **not** move for either search (B1) or
authoring (B2) on a fixed representation, the evidence points the next experiment squarely at the
**representation lever**: measure whether a **stronger / different transfer backbone** (e.g. resnet50, a
ViT/CLIP/DINO backbone) - or an **authored *featurizer*** (a learned representation, `role="featurizer"`, rather
than a classifier) - beats the resnet18 `emb-strong` baseline on this same arena, with the same paired-McNemar +
BH-FDR discipline, shipping only on an FDR-surviving lift. Authoring/search on top of a fixed representation is
**not** the path and will not be wired further.
