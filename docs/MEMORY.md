# Durable Research Memory

The autoresearcher **learns across runs**. Every experiment writes what it learned (which
families worked, which dead-ended, on data shaped like this) into a shared durable store.
A future run on similar - or even different - data warm-starts from that knowledge instead
of cold-starting with the catalog prior. GPU runs compound: every `device="cuda"` outcome
enriches the same store, so the next campaign starts smarter.

## Architecture

```
              +------- in-run realized gains (voi.CaseBase) ------+
              |                                                    |  tier 1: dominates
              |  ResearchMemory (vfplatform/research_memory.py)    |  once observed
              |                                                    |
              +------- cross-dataset warm prior (store) -----------+  tier 2: warm-start
              |                                                    |  from similar past data
              +------- catalog prior (move.prior_gain) -----------+  tier 3: fallback
```

### Three-tier VoI ranking

1. **In-run realized** - if the move ran this session, its empirical gain/cost dominates.
2. **Cross-dataset warm prior** - if a durable store was given AND a similar fingerprint
   has realized data for this family, use that prior (or deprioritize if it dead-ended).
3. **Catalog prior** - the default `move.prior_gain` from the recipe/move definition.

This means: first runs start from catalog (cold). After one campaign, the next run on
similar data inherits what worked. After many campaigns, a library of per-family × per-
dataset-shape realized gains accumulates - the autoresearcher gets better.

## Dataset Fingerprinting

`casebase_store.fingerprint(profile)` produces a bucketed signature:

```
<modality>|<size_bucket>|<feature_bucket>|<n_classes_bucket>|<balance_bucket>
```

Two datasets with the same signature have distance 0 (perfect transfer). Different modality
adds 3.0; different n_classes adds 2.0; etc. `warm_start(fp, k=12, max_distance=6.0)`
finds the k nearest outcomes within max_distance.

## Device Attribution

Every durable outcome carries a `device` field (`"cpu"` or `"cuda"`). This is purely
descriptive provenance - it never affects VoI ranking. It lets a campaign report show:

- "This store has 120 cpu outcomes + 48 cuda outcomes"
- "Top levers on cuda: torch_mlp (+0.18), torch_cnn (+0.12)"
- "The GPU campaign genuinely compounded: 3x richer signal than CPU-only"

## Usage

### With the experiment suite

```bash
# Campaign with durable memory (cross-dataset transfer + device recording):
python scripts/run_experiments.py --memory ./memory_store

# Every experiment warm-starts from prior runs AND writes back.
# The second run of the same campaign is smarter than the first.
```

### With the loop directly

```python
from vfplatform.casebase_store import CaseBaseStore
from vfplatform.loop import run_goal_loop

store = CaseBaseStore(
    outcome_path="./memory/outcomes.jsonl",
    negative_path="./memory/negative.jsonl")

result = run_goal_loop(
    records, goal, ...,
    memory_store=store)  # ← durable, cross-dataset
```

### With the /goal front door

```python
from vfplatform.casebase_store import CaseBaseStore
from vfplatform.goal_solver import solve

store = CaseBaseStore(
    outcome_path="./memory/outcomes.jsonl",
    negative_path="./memory/negative.jsonl")

cert = solve("Beat 90% accuracy on this data", data,
             memory_store=store)  # records champion outcome durably
```

## GPU Compounding

The day you run `--lane gpu --memory ./memory`:

1. The loop detects `device="cuda"` from the provider.
2. Every torch family outcome (torch_mlp, torch_cnn) is recorded with `device="cuda"`.
3. Next run (even on a new dataset): warm-start pulls the cuda outcomes → starts informed.
4. Report shows the device breakdown: you can SEE the GPU investment compounding.

```
$ python scripts/run_experiments.py --memory ./memory --lane gpu

# Report:
#   durable_memory:
#     n_outcomes: 156
#     by_device: {cpu: 80, cuda: 76}
#     top_learned_levers: [(torch_mlp, 0.18), (hist_gbm, 0.12), ...]
#     dead_ends: [logistic, baseline]
```

## Negative Memory

A family that consistently dead-ends (gain <= DEAD_END_GAIN) is recorded in
`negative.jsonl`. On a similar dataset, that family appears in
`ResearchMemory.avoid` and its expected gain is capped at `min(prior * 0.1, 1e-4)`.
This means: once the autoresearcher learns something doesn't work on data like this,
it won't waste budget re-trying.

## Default-Off Contract

`memory_store=None` (the default) means:
- No fingerprinting.
- No warm-start.
- No durable writes.
- Behavior is **byte-identical** to pre-memory code.

This is critical for test stability and the frozen-core invariant.

## Files

| File | Role |
|---|---|
| `vfplatform/research_memory.py` | The `ResearchMemory` adapter (CaseBase subclass) |
| `vfplatform/casebase_store.py` | Durable store: fingerprint, warm_start, record_outcome, summary |
| `vfplatform/loop.py` | Wiring: builds ResearchMemory, records outcomes, forwards in recursion |
| `vfplatform/experiments.py` | Shared store for campaign, passes to both engines |
| `vfplatform/goal_solver.py` | Records /goal champion outcome to durable store |
| `tests/test_research_memory.py` | Unit tests (12) for the adapter |
| `tests/test_experiments.py` | Integration tests (4) for durable memory in campaigns |
