# vfplatform - Ring-1 execution loop + tracking + compute providers

The deterministic loop body's **muscle** for the /goal loop (Architecture v2). Takes a goal + dataset,
fans out many candidate models across a compute provider, scores them on **validation only**, builds a
rich leaderboard, and certifies the validation-winner **once** on the sealed test via the frozen
`science` core. It never promotes - only `science.certify_accuracy` does.

```python
from vfplatform import run_goal_loop, ExperimentStore, LocalCpuProvider
r = run_goal_loop(records, "predict churn", kind="tabular", threshold=0.75,
                  store=ExperimentStore("vf_runs"), providers=[LocalCpuProvider()])
print(r.leaderboard.table()); print(r.decision, r.certificate)
```

## Modules
- `tracking.py` - `ExperimentStore`: experiment→run→{params, step-wise metrics, artifacts, system, tags},
  query + a dashboard JSON roll-up (MLflow / W&B parity, local-first). Each run carries a `science.digest`
  so tracking and the evidence ledger share one root of trust. **Observational - cannot alter a certificate.**
- `providers.py` - `Provider` ABC; `LocalCpuProvider` (runs now, thread pool); `RunPodProvider`
  **gated on `RUNPOD_API_KEY`** - `available()` False without it and `map` raises `ResourceGated`
  (honest decline; **no GPU run is ever faked**). Real RunPod submit/poll/fetch is a deferred follow-up.
- `harness.py` - `Harness` ABC + `TabularClassificationHarness` / `TextClassificationHarness` /
  `TabularRegressionHarness`. Each owns the featurizer, target encoding, validation scorer, and the closed
  **MOVE MENU** (baseline / stronger_model / more_capacity / regularize) the loop proposes among.
- `voi.py` - `CaseBase` (persisted realized val-gains per move) + `voi_rank` (expected gain ÷ cost).
- `checkpoint.py` - `Checkpoint`: auto-passes free CPU moves, raises `CheckpointRequired` before any
  paid/gated provider move (the (4) CHECKPOINT node).
- `sealed.py` - **the moat guard.** `SealedTest` content-addresses the locked test and **counts every
  peek**; `certify_on_sealed` passes `checks = realized peeks` (multiplicity paid, second peek refused) and
  routes each metric to the correct frozen certifier (binomial / bootstrap-clf / `certify_regression`).
  `assert_supported_metric` refuses any metric `science.score_metric` would silently coerce to accuracy.
- `leaderboard.py` - `Run` (family, move, round, seed, val_score, latency, cost_usd, provider) +
  `Leaderboard` ranked by the validation metric. Validation-only.
- `loop.py` - `run_goal_loop`: split (frozen; regression stratified by target quantiles) → audit (frozen,
  blocks first) → **multi-round** baseline → DIAGNOSE → PROPOSE (move menu) → VoI rank → CHECKPOINT →
  EXECUTE fan-out via provider → MEASURE on val → UPDATE best/leaderboard/case-base → certify the winner
  ONCE on the sealed test (enforced counted peek) → narrate. Every run tracked.

## Demonstrated (CPU, `demo_master.py` - 5.0s, 3 subfields)
- **tabular (nonlinear churn)**: baseline logistic 0.57 → **escalates** to `stronger_model` RF 0.94 →
  certified accuracy 0.897 (lower 0.867 > 0.75). Multi-round VoI escalation shown.
- **text sentiment**: baseline tfidf+logistic converges → certified accuracy 0.855 (lower 0.819 > 0.70).
- **regression**: ridge baseline → certified r2 0.998 (lower 0.997 > 0.80, via `certify_regression`).
- each goal: exactly one sealed-test peek (`peeks=1`, enforced); compute `local-cpu=live; runpod-gpu=gated`.
- 14/14 platform tests pass (`test_vfplatform.py`), incl. the moat-guard + VoI + checkpoint suite.

## Honesty ledger (real-on-CPU vs gated)
- REAL now: the multi-round diagnose→propose→VoI→fan-out→measure→update loop, validation leaderboard with
  details (move/round/latency/cost), **enforced** sealed-test certification + metric guard (frozen),
  experiment tracking + dashboard roll-up, provider abstraction + CPU executor, 3 ML subfields, checkpoint.
- GATED: GPU fan-out (RunPod - needs `RUNPOD_API_KEY` + a worker image; honestly declined now).
- NEXT (master plan steps 7, 9-11): re-audit on acquired data, RunPod submit/poll/fetch, live connectors
  (HF/GitHub/OpenML), per-subfield harnesses (vision/time-series/recsys), the literature-search harness,
  a serving/dashboard UI. The moat (`science.py`) is untouched throughout.
