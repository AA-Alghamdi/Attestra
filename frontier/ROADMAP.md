# Frontier autoresearch spine: sequenced roadmap

**This work is additive and separate.** Nothing in `vectorforge/`, `vfplatform/`, or
`attestra/` is edited or deleted. The new `frontier/` package reuses the audited-sound
certifier (`vectorforge.science`, `vfplatform.sealed`) by importing it. Consolidating the
three parallel stacks is a later, explicitly-approved step, not part of this build.

This roadmap is grounded in a five-part audit of the real code on the default branch and the
PR18 / PR19 branches. File:line citations below are to that real tree.

---

## 0. North star

The autoresearcher that **cannot fool itself**: generation and ungameable certification in one
loop. Given a goal plus data plus a verification standard, it returns a result that survived a
held-out sealed certificate, or an honest decline, never a relabeled validation score. It is
**powered by LLMs, not competing with them**: the LLM drives every open-ended decision (what to
try, what code to write, what to read, when to pivot); deterministic code owns only two things,
the numbers and the safety of execution. As base models improve, the system improves for free.

The audit confirmed the rare half is already real: the sealed-test + FDR + numeric-firewall
certifier in `vectorforge/science.py` is sound (Clopper-Pearson matches `scipy.stats.beta.ppf`
to 1e-15; the one-peek guard in `vfplatform/sealed.py` raises on a second peek). The missing
half is generation that is live, sandboxed, and certified in the same path. That is what
`frontier/` builds.

---

## 1. The three integrity bugs the spine fixes first (done in Phase 0)

| # | Bug (audit evidence) | Fix in `frontier/` |
|---|---|---|
| 1 | PR19 certifies on the **validation set** (`attestra/cycle/engine.py:712-714`, `_split_data` makes only train/val), so the certificate is computed on the selection data | True 3-way split; selection touches val only; the winner is certified on a held-out sealed test with one counted peek (`certify.py`, `engine.py`) |
| 2 | PR18/PR19 run LLM code via **in-process `exec` with full builtins** (`att-pr18/vfplatform/generative_researcher.py:329,336`); a crash or OOM takes the loop down | Real out-of-process sandbox: subprocess + `RLIMIT_CPU` + wall-clock timeout + process-group kill (`sandbox.py`) |
| 3 | When PR18's generative path "wins" it emits the **catalog's** certificate under the generative label; generated code is never run on the sealed test (`att-pr18/vfplatform/orchestrator.py:149-167`) | The selected winner **is** the thing certified, generated or seed, same gate (`engine.py`) |

All three are demonstrated by `frontier/tests/test_spine.py` (7/7 passing) and `frontier/demo.py`
(breast-cancer certifies; diabetes at theta=0.40 with the broken-split-bug fixed certifies, and
at a harder theta it honestly declines because the sealed lower bound does not clear theta).

---

## 2. Phased plan

Each phase lists what to build, what sound code to reuse-by-import, the integrity invariant it
must preserve, and an acceptance test.

### Phase 0 - the corrected generative spine  *(BUILT in this branch)*
- **Build:** `Program` (code, not a (family,params) pick) as the proposal unit; `SeedProposer`
  (catalog demoted to seeds/baselines, some carrying feature engineering and target transforms);
  `MutationProposer` (regenerates variants of the champion offline, no LLM needed); pluggable
  `LLMProposer`; the real subprocess sandbox; the certify-on-sealed engine.
- **Reuse:** `science.certify_accuracy` / `certify_regression` / `score_metric` /
  `make_splits` (classification); `sealed.SealedTest` / `certify_on_sealed`.
- **Invariant:** sealed touched once, for the winner; firewall (sandbox returns predictions only).
- **Acceptance:** `test_spine.py` 7/7; demo certifies a real dataset and honestly declines another.

### Phase 1 - feature engineering / preprocessing / target transforms first-class  *(seeded in Phase 0)*
- **Build:** richer recipe space (interactions, PCA, power/quantile transforms, target encoding,
  feature selection, stacking) and a richer mutation operator. The Program-is-code design already
  makes these proposable; this phase broadens the library and the LLM prompt.
- **Kills:** the `n_features >= 60` gate at `vfplatform/harness.py:391-392` that capped California
  Housing at ~0.81 because feature engineering was not proposable for low-dim data.
- **Acceptance:** on California Housing, a feature-engineered recipe is proposed and certified
  above the plain-model floor (the exact gap the owner hit).

### Phase 2 - diagnosis feeds forward  *(channel built in Phase 0; deepen here)*
- **Build:** round N's typed failures + champion already flow into round N+1's proposal context
  (`engine.py`); deepen the diagnosis (per-family error rates, plateau detection, residual
  structure) and let it gate which proposers fire and what the LLM is told.
- **Fixes:** the dead `_relevant` at `vfplatform/loop.py:781` (diagnosis discarded) and the
  only-`gap`-consumed path at PR19 `attestra/cycle/engine.py:395`.
- **Acceptance:** injected failure modes change the next round's proposals measurably.

### Phase 3 - the harness fabric (the "all task types" unlock)
- **Build:** a problem ontology + router (wire the unused `attestra/intake/problem_typing.py`)
  that selects, per task type, the harness, baseline suite, metric, split protocol, and certifier
  hook. Per-modality harnesses (vision/text/timeseries/audio/tabular) that are **self-testing**
  (each certifies itself on a known-good benchmark before its numbers are trusted) and
  **author-able** (when no harness exists for a task type, the LLM writes one and it is certified
  against a held-out known answer before use).
- **Reuse:** the same `certify` path; only the data->(rows,metric,theta) adapter changes.
- **Invariant:** a harness's numbers are trusted only after the harness passes its own sealed
  self-test.
- **Acceptance:** one non-tabular task type runs end to end and certifies through the same gate.

### Phase 4 - the agentic coding loop (executor, not one-shot)
- **Build:** write -> run -> read traceback -> inspect data -> fix -> re-run, with real tools
  (run, read logs, inspect arrays/frames, profile). Turns proposal from one-shot into an agent
  that gets a non-trivial pipeline working.
- **Reuse:** the sandbox as the execution substrate; the error taxonomy from `RunResult`.
- **Acceptance:** a candidate that fails on first run is repaired by the loop and then certified.

### Phase 5 - scientific experiment design + data-centric experimentation
- **Build:** an experiment object `{hypothesis, factor varied, controls held fixed, decision rule}`
  so it answers "is this idea a go?" not just "which model scored highest" (the owner's TTS
  tone-dimensionality example is an ablation). Treat **data** as the thing under test: synthesize,
  request more, restructure, audit quality, run active acquisition, and certify that the data
  change caused the lift.
- **Acceptance:** a controlled A/B with a stated decision rule returns a certified go/no-go.

### Phase 6 - long-horizon experiment management
- **Build:** working checkpoint/resume (the current `attestra/execution/checkpoint.py` resume is
  broken: it loads state then restarts every phase); a cost model; a portfolio with early-kill of
  losing arms (the Thompson portfolio at `attestra/orchestration/portfolio.py` exists but is
  test-only, never called); phase decomposition for week-to-quarter experiments.
- **Acceptance:** a multi-phase run survives a kill mid-experiment and resumes without redoing
  completed phases.

### Phase 7 - verification beyond statistics
- **Build:** sanity oracles (permuted labels must collapse to chance; a trivial baseline must be
  beaten; the metric must be computed on the right axis); seed-controlled re-execution to confirm
  reproducibility; an adversarial self-refutation pass that tries to break its own result before
  promotion.
- **Invariant:** a result promotes only after surviving the oracles and the refutation pass.
- **Acceptance:** a deliberately leaky pipeline is caught and refused.

### Phase 8 - compounding cross-task knowledge
- **Build:** a research knowledge base keyed by problem-type storing what worked and why,
  retrieved to warm-start proposals on **new** task types (transfer). Wire the real LinUCB
  `MetaLearner` (whose recommendations are currently only printed at
  `attestra/orchestration/orchestrator.py:251-259`) into proposal ranking.
- **Acceptance:** run N on a new dataset is measurably faster/better than a cold run by using
  prior-experiment priors.

### Phase 9 - research-artifact output
- **Build:** a generated report per goal: what was tried, ablations, what failed and why, the
  certificate, and exact reproduction instructions. The output is a defensible writeup, not a
  leaderboard row.
- **Acceptance:** the report reproduces the certified number from a clean checkout.

---

## 3. Standing invariant: powered by LLMs, not competing

Every hardcoded heuristic the LLM could make is demoted to a seed/fallback and is **never** the
promotion-bearing or search-bounding decision: the 12-family catalog (`vfplatform/harness.py`),
the `_MOTIF_RULES` / `_ARCH_CANON` keyword tables in retrieval, the architecture templates in
`attestra/execution/architecture_builder.py`. The frozen certifier is the only promoter; the LLM
proposes structure and the deterministic substrate recomputes every number. This is what lets the
system get better with each model release without new code.

---

## 4. Honest ceiling

- **Base-model ceiling.** The reasoning quality is partly set by the base model. That is the
  reason for the "powered by LLMs" framing, not a dodge.
- **Reward hacking.** Generation against a reward signal invites gaming. The sealed certificate
  plus the Phase-7 sanity oracles are the defense; they are load-bearing, not optional.
- **Coverage vs novelty.** "Solves all mid-level ML problems" is a benchmark-coverage claim, not a
  novel-research claim. Keep the two distinct. The defensible, still-extraordinary claim is: the
  first autoresearcher that generalizes across task types and returns only certified results or
  honest declines.

---

## 5. Consolidation (deferred, non-destructive)

The audit found three parallel stacks (`vectorforge/` + `vfplatform/` + `attestra/`), a
byte-identical `science.py` in three locations, and four competing "front door" loops. That is the
real path from vibe-coding to frontier engineering. Per the owner's instruction it is **not** part
of this build: `frontier/` stays separate. When approved, consolidation means porting the sound
`loop.py` selection/memory/certifier discipline into one package, deleting the duplicates, and
pointing `webapp/` and `worker/` at it. Explicit approval required before any deletion.
