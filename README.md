# Attestera

**An autonomous machine-learning researcher.** Give it a goal and a target. It proposes candidate models, trains them in parallel, measures, diagnoses its own weaknesses, and proposes better candidates, round after round. It certifies a model only when a single sealed-test peek clears your bar, and stops honestly when it can't.

Live dashboard: **https://lab.abdullahalghamdi.com**

## What it does

- **Recursive self-improvement.** Every round the engine reads its own diagnostics (where it under-fits, which model families stall) and proposes better candidates. A cross-dataset memory carries what it learned into the next problem.
- **Parallel fan-out.** Each round fans dozens of candidate trainings out across parallel workers: remote GPUs when configured, a local CPU pool otherwise. It never fakes hardware; every fallback is reported with its real reason.
- **Sealed-test certification.** Selection happens on validation only. The held-out test is sealed and peeked exactly once, guarded by a lower-bound gate. If the evidence doesn't clear the target, the run ends in an honest decline instead of an inflated number.
- **Literature-grounded proposals.** A retrieval scout queries arXiv and Hugging Face for the dataset's signature and feeds paper titles and abstracts into the proposer.
- **LLM goal understanding.** A language model can read a free-text goal and infer task, metric and target. It is non-binding: it only proposes; the frozen certifier decides.
- **Five modalities, one contract.** Tabular, vision, text, time-series and ranking each route through their own certifier (block-bootstrap for autocorrelated series, grouped splits for ranking) under one interface.
- **Governed autonomy.** A per-day spend cap bounds GPU and LLM cost, degrading to CPU and deterministic proposals instead of overspending.

## Quickstart

```bash
python -m attestra run --goal "classify accurately, accuracy >= 0.92" --data iris --output result.json
```

Data sources: built-ins (`iris`, `wine`, `breast_cancer`, `digits`, `diabetes`), `openml:<id>`, a CSV path or URL. Useful flags: `--metric`, `--threshold`, `--time`, `--no-llm`, `--gpu`.

## Engines

- **catalog**: the full recursive loop: proposal catalog + feature engineering + ASHA hyperparameter search + ensembles, fan-out across workers, per-round diagnosis, one sealed certification.
- **frontier**: the orchestrator engine: menu-free code discovery, sandboxed authored candidates behind a three-stage admission gate, value-of-information experiment ranking, and the same one-peek certification discipline.

## Layout

| Path | Role |
|---|---|
| `attestra/` | The research engine: cycle orchestration, proposals, execution, improvement, memory. |
| `frontier/` | The orchestrator engine: sandboxed code discovery, knowledge base, portfolio search. |
| `vectorforge/`, `vfplatform/` | The frozen certification core (sound statistics, sealed-test discipline) and platform harnesses. Imported verbatim; never edited by the search. |
| `webapp/` | The live dashboard: run console, SSE event streaming, spend governance. |
| `scripts/` | Research harnesses, benchmarks and the one-command test runner. |
| `tests/` | The test suite, including wiring safety nets and sealed-blind adversarial probes. |

## The one invariant

Everything in the search (the fan-out, the recursion, the LLM, the literature) only ever touches **train and validation** data. The sealed test is opened **once**, by frozen code, after the search is over.

## Tests

```bash
python scripts/run_tests.py
```

## Web dashboard

```bash
python3 webapp/server.py     # serves the dashboard on 127.0.0.1:8765
```

Deployment (Docker + Caddy + basic auth + spend cap): see `webapp/DEPLOY.md`.
