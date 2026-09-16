# #6 - LLM strategist: a *measurably* better proposer, with the frozen certifier still the sole promoter

## The gap this closes
The audit's sharpest finding was: *"the autoresearcher is currently me, not the system."* Across all seven
hand-run phases I was the strategist - I read each FDR table and decided *"swap to the DINOv2 family, skip
SigLIP/EVA-02, don't bother fusing, then scale to DINOv2-g."* The deterministic policy (`ReprResearcher._propose`)
reproduces that ladder only by **brute force**: it proposes every legal candidate for a rung and spends one
sealed certification on each competent one. A real strategist should reach the same certified champion in
**fewer sealed peeks** by proposing a shortlist. This item makes the *system* do that - and measures whether it
actually helps.

## What ships
- **`vfplatform/llm_strategist.py`**
  - `LLMStrategist` - gives Claude the public state of a rung (move class + its meaning, the champion's
    family/size, what's been tried, and the legal candidates with their metadata) and asks for a ranked
    shortlist that minimises wasted sealed certifications. It sees **no sealed labels**.
  - `StrategistResearcher(ReprResearcher)` - overrides **only** `_propose`; the gate, the certifier, the climb
    and the escalation ladder are inherited unchanged.
- **`scripts/run_llm_strategist.py`** - the falsifiable head-to-head.
- **`tests/test_llm_strategist.py`** - 5 hermetic locks (no network).

## The safety contract (why this can never weaken a certificate)
1. **Strategist proposals are intersected with the policy's legal set.** `StrategistResearcher._propose`
   recomputes the legal proposals with the *unmodified* `ReprResearcher._propose` and keeps only the
   strategist's choices that appear in it. An LLM hallucination is dropped; it can never inject a candidate the
   policy would not allow.
2. **The frozen Tier-3 certifier remains the sole promoter.** The strategist only reorders/prunes *which*
   candidates get a sealed peek; whether any is promoted is decided byte-for-byte by the frozen
   McNemar + BH-FDR + Clopper-Pearson path. A wrong call wastes or saves a peek - it can never mint a promotion.
3. **Graceful degradation.** API failure / unparseable reply / all-illegal ids → the strategist returns the full
   legal set (identical to deterministic). A *deliberate* empty shortlist means "skip this rung" (the
   strategist's most valuable move). Either way the worst case is a worse champion - caught by the
   deterministic baseline that runs alongside - never a false certificate.

## The falsifiable claim and the result
> *The LLM is a better strategist* iff it reaches the **same certified champion** in **strictly fewer sealed
> certifications**. A tie, a regression, or pruning the true winner (→ a worse champion) is an honest negative.

Run on the FGVC-Aircraft arena (10 confusable variant pairs, identical sealed rows as #3/#4/#5, start champion
CLIP ViT-B/32, frozen certifier as sole promoter, `docs/LLM_STRATEGIST_RESULT.json`):

| policy | champion reached | sealed certifications spent |
|---|---|---|
| deterministic | **dinov2_g** | **7** |
| LLM strategist (claude-sonnet-4-5) | **dinov2_g** | **2** |

**Verdict: `LLM_MORE_EFFICIENT` - same certified champion, 5 fewer sealed peeks (7 → 2, −71%).**

The strategist autonomously reproduced the hand-run reasoning, certified:
- **model rung:** kept only `dinov2_vitl14` (self-supervised, the proven fine-grained winner); **pruned**
  `eva02_l` and `siglip_so` ("bigger-but-different families that lose on fine-grained tasks").
- **features rung:** **pruned** the `dinov2_vitl14+clip_vitb32` fusion ("fusion has never certified; it dilutes
  on hard tasks").
- **capacity rung:** spent its second peek on `dinov2_g` ("within-family scaling of the winning family pays") -
  the frozen certifier then promoted it (3/10 FDR survivors), reproducing #5.

## Why this matters for the mission
This is the first time the **system** - not me - runs the search efficiently: it does not just *execute* the
ladder, it *prunes* it with the same priors a frontier researcher would, and the saving is real and measured
(−71% sealed peeks to the identical certified champion). Crucially the trust story is unchanged: every promotion
still rides on the frozen certifier, and the sealed labels never touch the strategist. The LLM made the search
cheaper; it did **not** make any unearned claim.

Frozen certifier byte-identical throughout: `vectorforge/science.py` `b564fba2`, `vfplatform/sealed.py`
`30ad6245`.

## Reproduce
```bash
ATTESTRA_SMOKE=2 python scripts/run_llm_strategist.py     # 2-pair wiring check (saves 2 peeks)
python scripts/run_llm_strategist.py                      # full 10-pair head-to-head (saves 5 peeks)
python -m pytest tests/test_llm_strategist.py -q          # 5 hermetic safety locks
```
