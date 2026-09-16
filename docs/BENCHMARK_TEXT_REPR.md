# #2 - Generality: does "the representation is the lever" hold beyond vision? (20-Newsgroups, TEXT)

**Frozen certifier untouched:** `science.py b564fba2` / `sealed.py 30ad6245` - byte-identical before and after
this work (full sha256 recorded in the result JSONs). The text arena only *reads* the frozen Clopper–Pearson
lower bound (`science.clopper_pearson_lower`) and the frozen suite stats (`mcnemar_pvalue`,
`benjamini_hochberg`); it never edits a frozen file and never weakens a gate.

## The question

Seven vision experiments established **one law across two arenas**: given a fixed representation, *nothing*
authored on top moves the metric - not the search cycle (B1: 0/5), nor LLM-authored classifiers (B2: 0/5), nor
end-to-end fine-tuning (#3: 0/10 vs the best frozen rep), nor an authored featurizer (#4: 0/10). **Only changing
the representation does** (CLIP/DINOv2 over resnet18: 5/5; DINOv2-L over CLIP-B on the hard arena: 2/10; DINOv2-g
over DINOv2-L: 3/10). Every positive so far is *frozen-vision-embedding* transfer on confusable binary pairs.

So the open question is **generality**: is "representation is the lever, authoring on a fixed representation adds
nothing" a real law, or an artifact of vision? This phase re-runs the **identical FDR machinery** on a different
modality - **text** - using the **same frozen certifier** and the **same autonomous brain** (`ReprResearcher`),
changing only the domain (20-Newsgroups confusable pairs) and the representation axis (frozen sentence encoders
vs a weak lexical baseline).

| arm | what it is | role |
|-----|------------|------|
| **tfidf_lsa** | TF-IDF + TruncatedSVD(300), no label supervision in the encoder | weak *lexical* baseline (the "raw-pixels" of text) |
| **minilm** | all-MiniLM-L6-v2 (22M) SBERT | distilled neural sentence encoder |
| **mpnet** | all-mpnet-base-v2 (110M) SBERT | MPNet sentence encoder |
| **e5_small / e5_base / e5_large** | intfloat/e5-{small,base,large}-v2 (33M / 110M / 335M) | a clean *scale ladder* in one contrastive-retrieval family |

`arena: scripts/repr_arena_text.py` · `law benchmark: scripts/benchmark_text_repr.py`
(`docs/BENCHMARK_TEXT_REPR_RESULT.json`) · `autonomous run: scripts/run_repr_researcher_text.py`
(`docs/REPR_RESEARCHER_TEXT_CERTIFICATE.json`). 6 confusable pairs, per_class=240, sealed n=72/task.

### Machinery (identical select-then-bound discipline to the vision arena - verified)

- Every encoder is **frozen**: TF-IDF/LSA is fit on the train docs only (no labels in the transform); the
  sentence transformers are pretrained and never tuned. No label supervision enters the representation.
- The train/val/**sealed** row-ids are derived **once** from the `tfidf_lsa` baseline embeddings
  (`benchmark_backbones._sealed_split`) and reused **verbatim** for every encoder - so all McNemar pairing is on
  **byte-identical sealed rows** and only the representation differs.
- Every arm uses the **same** strong tuned head (`_random_search_best` = random-search over
  GBM/RF/ET/logreg/kNN, fit on train, selected on val). The baseline a challenger must beat is never weaker than
  this tuned search.
- Each comparison is one-sided exact McNemar on the identical sealed rows, then Benjamini–Hochberg(α=0.1) across
  the 6-task suite; the deliverable is bounded by the frozen Clopper–Pearson lower bound. A hermetic test
  (`tests/test_text_arena.py`) certifies the split is encoder-independent and reused verbatim, that `measure()`
  is strictly select-then-bound (flipping *only* the sealed labels inverts the correctness vector elementwise
  while validation correctness is unchanged), and that the arena delegates to the *real* frozen stats.

---

## Panel B - the representation lever (each frozen encoder vs the lexical baseline)

Sealed accuracy per encoder per task (**mpnet** is the champion the system selects; see below):

| task | tfidf_lsa | minilm | **mpnet** | e5_small | e5_base | e5_large |
|------|------|------|------|------|------|------|
| PC.hardware / Mac.hardware   | 0.694 | 0.792 | **0.847** | 0.812 | 0.799 | 0.799 |
| ms-windows.misc / windows.x  | 0.861 | 0.882 | 0.875 | 0.847 | 0.924 | 0.882 |
| atheism / religion.misc      | 0.611 | 0.694 | **0.701** | 0.611 | 0.618 | 0.674 |
| baseball / hockey            | 0.826 | 0.875 | **0.931** | 0.931 | 0.944 | 0.958 |
| autos / motorcycles          | 0.826 | 0.799 | 0.868 | 0.826 | 0.910 | 0.868 |
| electronics / space          | 0.868 | 0.944 | **0.951** | 0.938 | 0.944 | 0.951 |
| **pooled sealed acc**        | **0.781** | **0.831** | **0.862** | **0.828** | **0.857** | **0.855** |

**BH-FDR(0.1) survivors with positive lift (each encoder vs the lexical `tfidf_lsa` baseline):**

| comparison | survivors | mean lift | pooled sealed lb |
|------------|-----------|-----------|------------------|
| **mpnet vs tfidf_lsa** | **4/6** - PC/Mac (p=.001), atheism/religion (p=.052), baseball/hockey (p=.001), electronics/space (p=.006) | **+0.081** | **0.842** |
| e5_base vs tfidf_lsa | 5/6 | +0.075 | 0.835 |
| e5_large vs tfidf_lsa | 3/6 | +0.074 | 0.834 |
| minilm vs tfidf_lsa | 2/6 | +0.050 | 0.809 |
| e5_small vs tfidf_lsa | 3/6 | +0.046 | 0.805 |

**The representation lever fires on text exactly as it did on vision:** every neural sentence encoder clears
BH-FDR(0.1) over the lexical baseline on multiple pairs, with large lifts on the genuinely confusable ones
(PC-vs-Mac hardware 0.694→0.847, baseball-vs-hockey 0.826→0.931). Swapping the representation - lexical →
semantic - is what moves the metric.

## Panel A - authoring on the fixed champion representation (mpnet)

Four authored heads, each vs the **same** strong tuned-search baseline on the **champion mpnet** embeddings,
identical sealed rows, BH-FDR(0.1):

| authored head | survivors | mean lift |
|---------------|-----------|-----------|
| tuned RBF-SVM | **0/6** | +0.021 |
| tuned MLP | **0/6** | −0.020 |
| cosine-prototype | **0/6** | −0.014 |
| whitening + logreg | **0/6** | −0.015 |

**Authoring on a fixed text representation adds nothing - 0/6 across all four heads**, reproducing the vision
result (B1 cycle 0/5, B2 authored classifiers 0/5) on a new modality. Given the representation, the tuned search
is already at the head ceiling; nothing authored on top survives FDR.

---

## The autonomous brain reproduces the verdict on text (no human in the loop)

The **same** `ReprResearcher` policy used for vision, started from the weak lexical champion `tfidf_lsa`, was run
on the text arena. It climbed the representation axis and stopped - entirely on the frozen certifier's decisions
(`docs/REPR_RESEARCHER_TEXT_CERTIFICATE.json`):

- **First move is a family swap, lexical → neural:** it promoted `tfidf_lsa → mpnet` (4/6 survivors, +0.081,
  sealed lb 0.842).
- **The invariant held under fire:** `minilm` (lb 0.809) and `e5_small` (lb 0.805) *also* certified over the
  lexical baseline, but were **superseded by mpnet on certified sealed lower bound (0.842 > 0.809 > 0.805)** -
  the frozen certifier picked the champion, not validation accuracy.
- **It then exhausted every cheaper lever and honestly stopped:** `e5_base` did not certify over mpnet (0/6); it
  escalated `model → features` and tried four fusions (`mpnet+{minilm,e5_small,tfidf_lsa,e5_base}`) - all **0/6**;
  escalated `features → capacity`, found nothing, and stopped ("every cheaper lever exhausted → honest stop").
- It flagged **4 pairs** as a data ceiling (champion still below the competence ceiling) and emitted a certified
  Pareto front (mpnet = most-accurate; tfidf_lsa = cheapest/fastest certified).
- **All 5 generality acceptance checks pass;** frozen hashes byte-identical before and after.

## Honest verdict - the law generalizes beyond vision

1. **Representation is the lever on text too.** Frozen semantic encoders beat the lexical baseline under
   BH-FDR(0.1) (mpnet 4/6 +0.081; e5_base 5/6 +0.075), with the largest gains on the most confusable pairs.
2. **Authoring on a fixed representation adds nothing on text too.** All four authored heads on the champion
   representation are 0/6 - the third modality-independent confirmation that, given the representation, the tuned
   search is already at the head ceiling.
3. **The same autonomous brain climbs the same axis with no human in the loop**, selecting the champion by the
   frozen certified sealed lower bound and stopping honestly - exactly as it did on vision.
4. **One honest nuance on scale.** Within the e5 family, scale helps then plateaus: e5_small→e5_base climbs
   (lb 0.805→0.835, 3→5 survivors) but e5_base→e5_large does not (lb 0.835→0.834). And the single best encoder
   here is mpnet (110M), not the largest e5 (335M) - on this text suite the *kind/quality* of the encoder
   dominates raw size, echoing the vision finding that a *different/larger* bias is not automatically better.

Every lift is vs a strong baseline (tuned search) on its own representation, on rows never trained on, bounded by
the frozen certifier, FDR-controlled. No gate was weakened; frozen hashes identical before/after. This is the
first FDR-certified confirmation of the law **outside vision** - "the representation is the lever, and authoring
on a fixed representation adds nothing" is a cross-modal property, not a vision artifact.
