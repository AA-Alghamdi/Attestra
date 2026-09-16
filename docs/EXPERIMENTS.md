# Experiment suites & the basis for GPU runs

This is the reproducible campaign harness: a **declarative JSON manifest** of experiments
(datasets × goals × seeds × budgets) that Attestera runs end-to-end under the **frozen certifier**, producing
one machine-checkable report (a leaderboard + a per-experiment certificate summary). It is the campaign the
autoresearcher runs the day a GPU is connected - and it is **verifiable on CPU before any GPU spend**.

- Library: [`vfplatform/experiments.py`](../vfplatform/experiments.py)
- CLI: [`scripts/run_experiments.py`](../scripts/run_experiments.py)
- Bundled manifest: [`experiments/default_suite.json`](../experiments/default_suite.json)

## TL;DR - one command

```bash
# 1) read-only GPU readiness (NO runs, NO spend): what's runnable today and the exact reason if not
python scripts/run_experiments.py --preflight

# 2) run the bundled suite on CPU now (front door + the in-process worker for the GPU loop lane)
python scripts/run_experiments.py --lane cpu

# 3) the day a GPU is connected - same command, prefer GPU (honest per-lane CPU fallback if unset)
RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...  python scripts/run_experiments.py --lane gpu

# 4) cross-experiment MEMORY: the loop lane learns which families pay off across the campaign
python scripts/run_experiments.py --lane cpu --memory .vf_campaign_memory
```

The frozen certifier (`vectorforge/science.py` + `vfplatform/sealed.py`) is asserted **byte-identical before
and after** every suite; the report records the hashes (`b564fba2` / `30ad6245`).

## Cross-experiment memory (`--memory <dir>`)

By default every experiment **cold-starts**: its Value-of-Information (VoI) move ranking begins from static
priors and is thrown away at the end. Pass `--memory <dir>` and the **loop lane** instead warm-starts VoI from
- and appends its realized `(val-gain / cost)` back to - a shared per-`(kind, task_type)` case-base under that
directory (`casebase_tabular_binary.json`, …). So a campaign **learns from its own experiments**: families that
paid off earlier (e.g. `extra_trees`, `hist_gbm` on nonlinear data) are prioritized earlier in later
experiments, while a linear baseline that never helped sinks. Selection-only - VoI **never** reads the sealed
test, and the frozen certify path is untouched (default `None` == cold start == byte-identical behavior). The
suite is sequential, so a shared case-base is deterministic and race-free. The report carries a
`cross_experiment_memory` summary (moves learned, observations accumulated, the top realized-gain levers).

## Two lanes, one frozen certifier

| lane | engine | what it exercises | GPU? |
|---|---|---|---|
| `goal` | `goal_solver.solve` (the autonomous **/goal front door**) | free-text goal + dataset → a `GoalCertificate` (certified champion / honest non-promotion / honest decline). With `"gpu": true` it also recruits a **torch-MLP head** as a first-class candidate. | optional head |
| `loop` | `loop.run_goal_loop` (the **provider-aware harness engine**) | proposes the GPU torch families (`torch_mlp` / `torch_cnn`); the certifier scores them. On a GPU box they train on **cuda**; with no GPU they device-swap to the **in-process worker on CPU** so the *identical* remote contract (JobSpec → worker → predictions → local frozen certify) is exercised first. | yes |

**Honest gating.** Nothing here spends money on its own. Provider selection probes read-only (RunPod
health + a torch/cuda import check). A GPU job runs only when a real endpoint/pod is configured **and** a
loop-lane experiment is requested. With no GPU connected, `--lane gpu` and `--lane auto` fall back to the
in-process CPU worker per lane - they never fake a GPU run.

## Manifest schema

A manifest is `{"suite": <name>, "experiments": [ <spec>, ... ]}`. Each spec is an
`ExperimentSpec` (unknown keys, e.g. `note`/`_doc`, are ignored, so manifests can self-document):

| field | lanes | meaning |
|---|---|---|
| `name` | both | unique experiment id (use with `--only`) |
| `lane` | both | `"goal"` or `"loop"` |
| `goal` | both | the free-text goal handed to the engine |
| `data` | both | dataset pointer: sklearn name (`wine`,`iris`,`digits`,`breast_cancer`,`diabetes`), `openml://<id>`, an `.npz`, or a `.csv` path |
| `seed` | both | run seed (determinism) |
| `peeks` | goal | sealed-peek budget |
| `gpu` | goal | `true` → recruit the torch-MLP head candidate |
| `use_literature` | goal | ground discovery in the literature scout (slower) |
| `task_type` | loop | `binary` \| `multiclass` \| `regression` |
| `threshold` | loop | certification bar θ |
| `metric` | loop | `accuracy` (clf) / `tolerance` (reg) by default |
| `candidate_seeds` | loop | model seeds explored per round |
| `max_rounds` | loop | search-round budget |
| `objective` | loop | `"maximize"` (explore capacity incl. torch - the GPU-worthy regime) or `"certify"` (stop at the first model clearing θ, cheap) |

## What the bundled suite proves

[`experiments/default_suite.json`](../experiments/default_suite.json):

- `wine_goal` - multiclass i.i.d. → **SOLVED** (sealed lower bound clears the floor).
- `breast_cancer_goal` - imbalanced → **SOLVED** with a data-driven floor (>0.5), not wrongly refused.
- `diabetes_goal` - regression-tolerance → an honest **NOT-SOLVED** is a correct outcome.
- `forecast_decline` - out-of-scope framing → honest **DECLINE** (no run, no peek).
- `wine_goal_gpu_head` - same goal with the **torch-MLP head** enabled (cuda on a GPU box).
- `digits_loop_gpu` - the **loop lane**: `torch_mlp` / `torch_cnn` are proposed & fit (CPU now → cuda on a GPU box).

## GPU day - required secrets & setup

The serverless GPU path needs two values (both honestly gated - missing either keeps the lane on CPU):

| secret | what it is | how to get it |
|---|---|---|
| `RUNPOD_API_KEY` | RunPod account API key (also accepted from a `./.runpod_key` file) | RunPod console → Settings → API Keys |
| `RUNPOD_ENDPOINT_ID` | a serverless endpoint built from the worker image | build & push `worker/` image, create an endpoint (see [`docs/deploy_runpod.md`](deploy_runpod.md)), copy its id |

```bash
export RUNPOD_API_KEY=...           # or: echo "<key>" > .runpod_key   (git-ignored)
export RUNPOD_ENDPOINT_ID=...
python scripts/run_experiments.py --preflight     # should now report ready_for_gpu: true
python scripts/run_experiments.py --lane gpu
```

A keyless **runpod pod** path (`RunPodPodProvider`, driven by `runpodctl`) is also supported for an
interactive GPU box; see `vfplatform/providers.py`. Per-hour cost estimates are surfaced in the preflight so
the loop's cost checkpoint always sees GPU work as paid.

> Note on this sandbox: `runpodctl exec` uses SSH (a high TCP port), which an HTTPS-only sandbox blocks.
> Validate the GPU lane from a normal-egress machine/server; the CPU lane is fully verifiable anywhere.

## Output

The CLI prints a live per-experiment trace + a markdown leaderboard and writes the full report JSON to
`docs/EXPERIMENTS_RESULT.json` (override with `--out`). The report carries the provider/device used, the
read-only preflight, every experiment summary, and the frozen-certifier hashes.
