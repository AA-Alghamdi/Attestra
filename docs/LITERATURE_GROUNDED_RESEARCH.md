# Literature-grounded discovery (the deepest anti-menu)

> Generator regenerates full recipes → validity cascade screens → **frozen Tier-3 is the sole promoter.**
> This layer changes only **where the proposals come from**: not a hand-written list, but what the system
> *reads in the wild*.

## The gap this closes

Until now the proposer's "literature library" was a **hard-coded list of motifs**
(`recipe_generator.literature_library`) - itself a menu a human wrote. A frontier ML researcher's real edge
is *reading the literature*: knowing the current SOTA for a problem and adapting it. This layer replaces the
hand-list with **live retrieval** from the open research surface and feeds the result, with provenance, into
the same governed loop.

```
LiteratureScout(problem)                       vfplatform/literature.py
  ├─ retrieve()  ── arXiv (Atom)  ┐
  │                ── GitHub (repos by stars)  ┤  offline-graceful: any error → [],
  │                ── HF Hub (models by downloads, keyword-expanded) ┤  everything empty → bundled corpus
  │                ── Papers-with-Code (best-effort) ┘
  ├─ extract_backbones()  → concrete Hub ids (verbatim) + family mentions → one representative timm id
  ├─ extract_motifs()     → technique keywords → typed partial genomes (validate_genes drops off-axis)
  └─ llm_motifs()         → (optional) Claude reads the abstracts and proposes motifs/backbones
        │
        ▼  each ingredient carries PROVENANCE {source, ident, url, title}
RecipeGenerator(scout=…)                       vfplatform/recipe_generator.py
  ├─ discover_fn = scout.backbones() ∪ open zoo   (literature ids tagged in the ledger)
  └─ library     = scout.library()                (the RETRIEVED motifs replace the hand-list)
        │
        ▼
validity cascade → FROZEN Tier-3 certifier (sole promoter)   ← unchanged, byte-identical b564fba2 / 30ad6245
        │
        ▼
champion + DiscoveryLedger.novelty() → literature_grounded: bool + literature_source: {…}
```

## The falsifiable claim

`DiscoveryLedger.novelty(champion)` now returns:

- `literature_grounded` - **True iff the certified champion's backbone entered the pool from a retrieved
  source** (not from the seed set), and
- `literature_source` - the exact `{source, ident, url, title}` it traces to.

This is an **honesty instrument**, not a gate (the frozen certifier still promotes). It is *falsifiable*: if
the champion is a seed, or comes from the open zoo rather than the literature, the flag says so. The
`--with-zoo` smoke below shows exactly that honest negative.

## Safety / robustness (why a bad finding can't corrupt a recipe)

- **Offline-graceful.** Every connector returns `[]` on any failure; `retrieve()` falls back to a small
  bundled corpus, so a run never crashes on a flaky API and the test suite is hermetic.
- **Typed firewall.** `validate_genes` drops any gene whose key isn't a recipe axis or whose value is
  off-axis (and clamps numerics). A hallucinated or LLM-proposed technique can at worst be *ignored* - it can
  never inject an illegal recipe. The motif then still has to survive the full cascade and the frozen
  certifier like any other proposal.
- **Frozen core untouched.** `literature.py` imports only stdlib + the typed axes; it never imports or
  modifies the frozen certifier. Hashes verified byte-identical before and after every run.

## Evidence (live retrieval → frozen-certified champion)

`python scripts/run_literature_scout.py` - full result in
[`LITERATURE_GROUNDED_RESULT.json`](LITERATURE_GROUNDED_RESULT.json).

| mode | retrieval | champion | `literature_grounded` | traces to |
|---|---|---|---|---|
| live, literature-only (default) | 16 findings (8 arXiv + 8 HF, real) | `convnext_base.fb_in22k` · linear_probe · logit_ensemble · gbm | **True**, gold-confirmed | arXiv `2409.03543` *"Classification and Object Localization under Distribution Shift"* |
| offline (bundled corpus, deterministic) | 5 corpus findings | `vit_base_patch16_siglip_224.webli` · linear_probe · gbm | **True**, gold-confirmed | corpus `siglip` |
| live, `--with-zoo` (honest negative) | same + open timm zoo | `vit_base_patch14_reg4_dinov2.lvd142m` (from the zoo) | **False** | - zoo backbone out-competed the literature; the flag honestly says "not grounded" |

In the headline run the system **read a real arXiv paper about distribution shift, surfaced ConvNeXt from
it, and ConvNeXt became the frozen-certified, gold-confirmed champion** - traceable back to the paper. The
`--with-zoo` row proves the flag is not rigged: when the open zoo offers a stronger backbone, the champion is
honestly reported as *not* literature-grounded.

## Reproduce

```bash
python scripts/run_literature_scout.py            # live retrieval, literature-only (the headline)
python scripts/run_literature_scout.py --offline  # deterministic, no network (CI-style)
python scripts/run_literature_scout.py --with-zoo  # union the open zoo (honest grounding negative)
python scripts/run_literature_scout.py --llm      # let Claude read the abstracts (needs ANTHROPIC_API_KEY)
python -m pytest tests/test_literature.py -q       # 9 hermetic locks (offline-graceful, typed firewall, anti-menu wiring)
```

## What is *not* claimed

The evaluation arena in the smoke is a CPU stand-in (it rewards the strong-transfer family the literature
surfaces for distribution shift, exactly as `tests/test_recipe_research.py` does) - so this proves the
**plumbing**: a backbone the system read about becomes the frozen-certified champion and is traceable. The
absolute-SOTA evaluation on real images is the GPU run (`scripts/run_wilds_gpu.py`, Gap 2).
