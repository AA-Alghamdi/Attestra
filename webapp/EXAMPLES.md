# Attestera: examples, options, and limits

A practical guide to what you can run, how each input works, and what the system will and won't do.

## The two outcomes (what every run returns)
- **CERTIFIED**: the winning model's metric, on a sealed held-out test, has a statistical **lower bound above your threshold θ**. You get a certificate (observed score, lower bound, n, metric) and a servable model version. Selection happens on validation; the sealed test is peeked **once**.
- **HONEST-STOP**: the validation lower bound never cleared θ, so the system refuses to certify and tells you why (dominant error source, cheapest unblock, suggested next experiment). It does **not** relax the threshold to fake a pass.

## The objective (Certify vs Maximize)
- **Certify** (default): the loop searches only until the validation lower bound clears θ, then stops and issues the certificate. This is the cheapest path to a defensible pass; it does not keep spending once the bar is met.
- **Maximize**: the loop keeps searching for the best achievable score within the run-time/effort budget below, rather than stopping at the first model that clears θ. Use this when you want the strongest model the budget can buy, not just one that passes. The certificate is still issued on the single sealed-test peek, and θ is still never relaxed.

## The run-time / effort budget
- Every run is bounded by a **budget** you set before it starts: a **wall-clock time limit** and an **effort level** (how many models the search is allowed to train and evaluate). The loop will not run forever; when the budget is exhausted it stops and reports the best result so far (certifying if the bound cleared θ, honest-stopping otherwise).
- The budget governs how deep the recursive search goes. A larger budget lets the loop expand more branches of the model zoo and refine more candidates; a small budget keeps the run fast and cheap. The budget is independent of the daily spend cap below, which is a separate hard ceiling on the public deployment.

## Built-in datasets (the dropdown)
| Name | Modality | Task | Rows | Good threshold to try | Notes |
|------|----------|------|------|----------------------|-------|
| `breast_cancer` | tabular | binary classification | 569 | accuracy ≥ 0.95 | easy; usually certifies |
| `wine` | tabular | multiclass (3) | 178 | accuracy ≥ 0.92 | small; certifies |
| `iris` | tabular | multiclass (3) | 150 | accuracy ≥ 0.92 | tiny; certifies |
| `digits` | tabular | multiclass (10) | 1797 | accuracy ≥ 0.95 | harder; may honest-stop at high θ |
| `diabetes` | tabular | regression | 442 | r2 ≥ 0.4 | regression; bound is conservative |
| `timeseries_demo` | time-series | forecast | synthetic | (demo) | block-bootstrap + forward-chaining split |
| `ranking_demo` | ranking | ranking | synthetic | (demo) | NDCG/MAP + per-query bootstrap |

## Your input options
1. **Pick a built-in dataset**: fastest way to see a full run.
2. **Paste JSON rows**: a list of `{"features": {...}, "target": ...}` records.
3. **Upload a training CSV**: the target column is whichever you name in *target column*, else a column literally named `target`, else the last column. Numeric cells become numbers; everything else is treated as categorical.
4. **(Optional) held-out verification set**: your own annotated examples (JSON or CSV). If you supply it, **it becomes the sealed test** the certificate is issued against. It is contamination-checked against the training rows; exact overlap is rejected (the run is blocked, not silently certified).

## The toggles
- **Understand goal with AI**: an LLM reads your free-text goal to infer task type + metric and to read a target % from the text (e.g. "accuracy ≥ 92%"). It is resolver-verified and **non-binding**; if the key is missing or the call fails, it falls back to deterministic parsing (you'll see `used_llm=false`, which is fine). It never sets the certificate.
- **Use GPU (RunPod)**: runs the candidate fan-out on your RunPod serverless endpoint in parallel. Real spend; scales to zero when idle. Falls back to local CPU if the key/endpoint isn't configured. Before dispatching, the GPU lane **health-probes RunPod**; if the workers are unhealthy or unreachable it **falls back to CPU honestly and reports the reason** rather than stalling or failing silently. Transient errors during a dispatch are **retried with backoff** before the lane gives up and degrades to CPU. Two extra GPU options let you tune the run: **max workers** (how many candidates fan out in parallel) and **preferred GPU type** (pin the RunPod hardware class). A **Stop** button halts an in-flight run cleanly at any point.

## How the loop searches (LLM-guided recursive search over a large model zoo)
- The loop no longer sweeps a fixed handful of candidates. It runs an **LLM-guided recursive search over a large model zoo**: the LLM proposes which families and configurations to try next, the system trains and evaluates them on validation, and the results feed back in so the next round of proposals is conditioned on what already worked. Promising branches are expanded recursively; weak ones are pruned.
- The search is bounded entirely by the run-time / effort budget above and stops on the active objective (clears θ under **Certify**, or exhausts the budget under **Maximize**). The LLM steers the search but **never writes or alters a certificate**; it only proposes candidates to evaluate. Selection and certification stay deterministic and local. If the LLM key is missing or a call fails, the search falls back to a deterministic candidate enumeration over the same zoo.

## Threshold + metric
- **Threshold θ**: the bar the lower bound must clear to certify. Higher θ = stricter = more likely to honest-stop.
- **Metric**: `accuracy` (classification), `f1_macro` (imbalanced classes), `r2` (regression). The certifier knows each metric; an unknown metric is rejected, not silently treated as accuracy.

## What it will NOT do (honest limits)
- **Only these modalities are supported:** tabular (binary / multiclass / regression), text (binary / multiclass), time-series forecast, ranking. Anything else (vision, audio, RL, generative, graph, recsys) is an **honest decline**, not a fake result.
- **No threshold relaxation.** If θ can't be cleared, it honest-stops; it never lowers θ to cross zero.
- **The certifier runs locally.** GPU/LLM are "muscle" only; remote compute and the LLM can never write or alter a certificate.
- **One sealed-test peek**, tracked in a durable ledger (multiplicity is accounted for).
- **Spend is capped per day** on the public deployment (default $5/day, 100 GPU runs, 200 AI calls); when a cap is hit, GPU degrades to CPU and AI degrades to deterministic, with a "capped" note.

## Reading a result
- `certificate.observed`: the metric value measured on the sealed test.
- `certificate.lower_bound`: the statistical lower bound (this is what's compared to θ).
- `certificate.n`: sealed-test size.
- `winner.family` / `winner.params`: the model that won, chosen by the recursive search over the model zoo.
- `served_version`: the registered, servable model id (certified runs only).
- On honest-stop: `failure_report.dominant_source`, `.cheapest_unblock`, `.next_experiment`.
