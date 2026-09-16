# frontier/

The corrected generative autoresearch spine. **Additive and separate**: it does not edit or
delete anything in `vectorforge/`, `vfplatform/`, or `attestra/`. It reuses the audited-sound
certifier (`vectorforge.science`, `vfplatform.sealed`) by importing it.

See `ROADMAP.md` for the full phased plan. This package is Phase 0 (the spine) plus the seeds of
Phase 1 (feature engineering proposable) and Phase 2 (diagnosis feeds forward).

## What it demonstrates

The three integrity properties the audit found broken in PR18/PR19:
1. **Generation, not a menu.** The proposal unit is a `Program` (arbitrary pipeline code). The
   12-family catalog is demoted to seeds/baselines. Feature engineering and target transforms are
   in-scope by construction, with no `n_features >= 60` gate.
2. **Real sandbox.** Candidate code runs in a separate process with CPU/time limits, not
   in-process `exec`. The sandbox returns predictions only; the parent computes every number
   (numeric firewall).
3. **Certify on a held-out sealed test.** Selection touches the validation split only; the winner
   is certified once on a sealed test (one counted peek). It returns a certified result or an
   honest decline, never a relabeled validation score.

## Run it

From the repo root, with an interpreter that has scikit-learn / scipy / numpy:

```
# integrity tests (7/7)
/Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_spine.py

# end-to-end demo (classification + regression)
/Users/abdullahalghamdi/jax-env-311/bin/python -m frontier.demo
```

Observed on first run: breast-cancer certifies (sealed lower bound 0.9319 > theta 0.90);
diabetes certifies at theta 0.40 (sealed lower bound 0.4828) and honestly declines at a harder
theta. No LLM key needed: the seed + mutation proposers make it generative offline. Wire an
`llm_client` into `EngineConfig` to activate the `LLMProposer`.

## Files

| file | role |
|---|---|
| `program.py` | `Program` (the code-as-proposal unit) and `RunResult` |
| `task.py` | `Task` (goal + data) and row rendering for the certifier |
| `proposers.py` | seed / mutation / LLM proposal sources; recipe -> code |
| `sandbox.py` | out-of-process execution with rlimits + timeout |
| `certify.py` | adapter that imports the sound `science` + `sealed` (never edits them) |
| `engine.py` | the loop: propose -> sandbox -> select-on-val -> certify-on-sealed |
| `demo.py` | live end-to-end demo |
| `tests/test_spine.py` | integrity tests for the properties above |
