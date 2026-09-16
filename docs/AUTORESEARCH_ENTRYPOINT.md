# Unified autoresearch entrypoint (#4)

**One command, one brain, one cross-modal session certificate.** `scripts/run_autoresearch.py` routes the
identical `vfplatform.repr_researcher.ReprResearcher` policy and the identical frozen certifier
(`science.py b564fba2` / `sealed.py 30ad6245`) across every modality behind a single flag, replacing the two
bespoke per-modality runners that each duplicated ~100 lines of certificate plumbing.

```
python scripts/run_autoresearch.py --modality all       # vision + text, then a cross-modal session certificate
python scripts/run_autoresearch.py --modality vision    # one modality
python scripts/run_autoresearch.py --modality text
```

## What it does

For each requested modality it builds that modality's arena + encoder registry + weak start champion, runs the
autonomous climb, prints the full certificate (research log, promotions/rejections, data-ceiling tasks, the
certified accuracy-vs-cost Pareto front, the never-peeked **gold confirmation**, and the **session
multiplicity** accounting), checks the modality's acceptance properties, and writes a per-modality JSON. Adding
a modality is a single entry in the `MODALITIES` table (arena builder + start champion + acceptance predicate);
nothing else changes because the brain and the certifier are modality-agnostic.

When run over `--modality all` it additionally emits **one cross-modal session certificate**
(`docs/AUTORESEARCH_SESSION.json`) that restates the program's law as machine-checkable booleans over every
modality that ran.

## Verified end-to-end (cached embeddings, both modalities)

| modality | start champion | autonomous champion | peeks | gold confirmed | Bonferroni-robust | acceptance |
|---|---|---|---|---|---|---|
| vision (FGVC-Aircraft) | clip_vitb32 | **dinov2_g** | 7 | n/a (no disjoint gold) | yes (4/10) | 7/7 PASS |
| text (20-Newsgroups) | tfidf_lsa | **mpnet** | 8 | **yes** (5/6, +0.063) | yes (3/6) | 6/6 PASS |

Cross-modal session certificate (`AUTORESEARCH_SESSION.json`):
- `representation_lever_fires_in_every_modality`: **true**
- `champion_robust_to_session_multiplicity_everywhere`: **true**
- `champion_gold_confirmed_where_gold_exists`: **true** (vision honestly reports `gold_confirmed=null` and is
  skipped, rather than fabricating a gold set)
- `all_modalities_accept`: **true**

The vision run autonomously reproduces the seven-phase hand-run verdict (climb to DINOv2-g, reject
SigLIP-SO400M and EVA-02-L, flag the 737 pairs as a data ceiling); the text run satisfies the generality
properties and is gold-confirmed. Frozen hashes byte-identical before and after every run.

## Honesty locks

`tests/test_autoresearch.py` (6 hermetic tests, no caches/network) locks the routing/aggregation layer: the
per-modality acceptance predicates fail when the verdict is not reproduced or not gold-confirmed, and the
session certificate is a faithful AND over modalities that correctly skips arenas with no gold. The brain's
autonomous-promotion invariant and each arena's leak-freeness remain locked by `tests/test_repr_researcher.py`
and `tests/test_text_arena.py`. The original per-modality runners (`run_repr_researcher.py`,
`run_repr_researcher_text.py`) are retained as the canonical acceptance harnesses for their respective
certificates.
