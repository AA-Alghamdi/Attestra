---
name: testing-attestra-pipeline
description: Test the Attestera autonomous ML research pipeline end-to-end. Use when verifying orchestrator wiring, module integration, frontier engine, oracle verification, sandbox security, or CLI changes.
---

# Testing the Attestera Pipeline

## Prerequisites

### Secrets Needed
- `PRIME_INTELLECT_API_KEY` - Required for LLM-enabled tests (frontier engine, problem typing, adversarial check, generative proposals). Without it, only catalog-only (no-LLM) tests and frontier offline-degradation tests work.

### Python Environment
- Python 3.12+ with sklearn, numpy, scipy installed
- No torch/pandas required for core pipeline tests (those are optional deps)
- Run from repo root: `/home/ubuntu/Attestera`

## Engine Architecture (3-Tier Routing)

The orchestrator routes to engines in priority order:
1. **Frontier** (primary) - fires when `frontier/` is importable AND engine is `frontier` or `auto` (default). Includes oracles, diagnosis feed-forward, agentic repair, LinUCB, portfolio/ASHA, meta-learner guidance, registry read-back.
2. **Generative** (LLM fallback) - fires when frontier import fails but LLM API key is available.
3. **Catalog** (no-LLM fallback) - fires when `engine="catalog"` or both frontier and LLM are unavailable. Includes parallel pool, codegen, phase-based harness selection.

**CLI flag:** `--engine catalog|frontier|auto` controls routing directly. `auto` (default) tries frontier first.
**Important:** `--no-llm` does NOT bypass frontier to reach catalog. It disables the LLM within frontier (`llm_active=False`). Use `--engine catalog` to force catalog path.

## How to Run Tests

### Unit Tests (fast, no LLM needed)
```bash
cd /home/ubuntu/Attestera

# Integration tests (behavioral, no LLM needed)
python -m pytest tests/test_integration_wiring.py --tb=short -q
# Expected: 28 passed, 0 skipped

# Core attestra tests
python -m pytest tests/test_gaps.py tests/test_attestra_core.py tests/test_e2e_attestra.py --tb=short -q
# Expected: ~83 passed, 5 skipped (1 ENOMEM on constrained VMs)

# Frontier module tests
python -m pytest frontier/tests/ --tb=short -q
# Expected: 219 passed, 3 skipped
```

### Full Test Suite
```bash
python -m pytest tests/ --tb=short -q
# Pre-existing failures: test_gpu_finetune.py (missing torch), test_code_discovery.py (missing pandas),
# test_recipe_research.py (2 novelty assertions). These are NOT regressions.
```

### CLI End-to-End - Frontier Path (needs LLM)
```bash
# Wine - expect CERTIFIED (easy dataset, oracle should approve)
python -m attestra run --goal "Classify wine varieties" --data wine --rounds 5 --time 120 --seed 42

# Iris - may get DO_NOT_CERTIFY (oracle might catch single-feature artifact on petal length)
python -m attestra run --goal "Classify iris species" --data iris --rounds 5 --time 120 --seed 42
```

### CLI End-to-End - Frontier Offline Degradation
```bash
# --no-llm triggers frontier with deterministic proposals only (llm_active=False)
python -m attestra run --goal "Classify wine" --data wine --rounds 5 --time 60 --no-llm --seed 42
# Expect: Stage 3: FRONTIER RESEARCH CYCLE, llm_active=False, likely CERTIFIED
```

### CLI End-to-End - Catalog Path (--engine catalog)
```bash
# CLI flag to force catalog path
python -m attestra run --goal "Classify wine" --data wine --rounds 5 --time 60 --engine catalog --seed 42
# Expect: engine_type='catalog', decision='certified' or 'error' (ENOMEM on small VMs)
```

### Python API - Catalog Fallback
```bash
python3 -c "
import os
from sklearn.datasets import load_wine
from attestra.orchestration.orchestrator import OrchestrateConfig, orchestrate

wine = load_wine()
config = OrchestrateConfig(
    goal='Classify wine varieties',
    X=wine.data, y=wine.target,
    metric='accuracy', threshold=0.80,
    max_rounds=5, time_budget_s=60.0,
    api_key=None,
    use_generative=False,
    use_retrieval=False,
    feature_names=list(wine.feature_names),
    seed=42, verbose=True,
)
result = orchestrate(config)
print(f'Engine: {result.engine_type}, Decision: {result.decision}')
# Expect: engine_type='catalog', decision='certified'
"
```

### Sandbox Security Tests
```python
from attestra.execution.sandbox import run_sandboxed, static_check

# static_check returns StaticCheckReport(ok, violations, has_entrypoint)
# NOT a tuple - access .ok and .violations fields

r = static_check('import os; os.listdir("/")')
assert not r.ok  # blocked

r = static_check('import subprocess; subprocess.run(["ls"])')
assert not r.ok  # blocked

r = static_check('import numpy as np; x = np.array([1,2,3])')
assert r.ok  # allowed

# run_sandboxed may fail with sklearn in constrained environments due to
# rlimit memory caps being too low for shared object loading. This is an
# environment limitation, not a code bug.
```

### Python API Field Verification (13 fields)
```python
import os
from sklearn.datasets import load_wine
from attestra.orchestration.orchestrator import OrchestrateConfig, orchestrate

wine = load_wine()
config = OrchestrateConfig(
    goal='Classify wine varieties',
    X=wine.data, y=wine.target,
    metric='accuracy', threshold=0.80,
    max_rounds=5, time_budget_s=120.0,
    api_key=os.environ.get('PRIME_INTELLECT_API_KEY'),
    use_retrieval=False,
    feature_names=list(wine.feature_names),
    seed=42, verbose=True,
)
result = orchestrate(config)

# Verify all fields
assert result.engine_type == 'frontier'
assert result.plan is not None
assert result.phases_completed > 0
assert result.best_score > 0
assert len(result.best_technique) > 0
assert result.n_proposals > 0
assert result.n_successful >= 1
assert result.decision in ('certified', 'do_not_certify', 'honest_stop')
assert result.certificate is not None
assert result.adversarial is not None and 'oracle_verdict' in result.adversarial
assert result.meta_learner_updated == True
assert result.registry_updated == True
assert result.elapsed_s > 0
```

## What to Verify (Observable Side Effects)

Each module in the pipeline has observable outputs in `OrchestrateResult`:

| Module | Field to Check | Broken if |
|--------|---------------|-----------|
| Problem typing | `result.problem_spec` | is None |
| Adversarial check | `result.adversarial` | is None or missing `oracle_verdict` |
| Oracle verification | `result.adversarial['oracle_verdict']` | missing `promote` key or `oracles` list |
| Experiment manager | `result.phases_completed` | is 0 |
| Data augmentation | stdout "Augmentation:" | missing for classification |
| Meta-learner | `result.meta_learner_updated` | is False |
| FDR controller | `result.fdr_decision` | is None when decision=certified |
| Registry (ledger) | `result.registry_updated` | is False |
| Strategy learner | `result.strategy_updated` | is False |
| Tradeoff engine | `result.tradeoff_analysis` | is None (only with deployment constraints) |
| Checkpoint | `result.checkpoint_saved` | is False (only with checkpoint_dir) |

### Oracle Checks (7 total, all should pass for a valid result)
| Oracle | What it checks |
|--------|---------------|
| metric_orientation | accuracy/loss direction is consistent |
| beats_trivial_baseline | sealed lower_bound beats random chance |
| no_label_leak_feature | no single feature determines the label |
| distribution_drift | train/test feature distributions are similar |
| permuted_label_collapses | permuted-label score collapses to chance |
| reproducible | re-run produces identical certificate |
| adversarial_self_refutation | no single-feature/row-shuffle attack explains the result |

## Key Architecture Notes

- The orchestrator is the single entry point: `attestra/orchestration/orchestrator.py`
- **3-tier engine routing**: frontier (primary) -> generative (LLM fallback) -> catalog (no-LLM fallback)
- Frontier engine: `frontier/core/orchestrator.py` (CoreOrchestrator)
- Catalog engine: `attestra/cycle/engine.py` (ResearchEngine)
- Generative engine: `attestra/cycle/generative.py` (GenerativeEngine)
- The frozen certifier lives in `attestra/core/science.py` - NEVER modify this
- Sandbox: `attestra/execution/sandbox.py` - subprocess isolation with rlimits + AST gate
- Oracle verification: `frontier/oracles.py` - verify_before_promote with 7 checks
- Problem typing uses LLM when API key present, heuristic fallback otherwise
- Augmentation only applies to classification tasks, not regression
- FDR controller only runs when a certificate is produced (decision=certified)
- The CLI is `python -m attestra run` (defined in `attestra/cli/main.py`)
- Both engines (frontier and catalog) now use proper 3-way split: train 60% / val 20% / sealed test 20%

## Adversarial Testing Patterns

When verifying bug fixes, design tests where the broken implementation would produce a **visibly different result**:

| Fix Type | Adversarial Pattern |
|----------|--------------------|
| Dict slice crash | Call `list(dict.items())[:N]` and also verify `dict.items()[:N]` raises `TypeError` |
| Scoping bug | Trigger the edge case (e.g., 0 rounds) and assert no `UnboundLocalError` |
| Infinite loop | Use `signal.alarm(3)` timeout + call counter with `RuntimeError` at >20 calls |
| Source removal (e.g., `or True`) | `inspect.getsource()` + assert pattern NOT in source |
| Security (sandbox) | Submit `import os` code, assert `AST gate rejected` in error message |
| Security (builtins) | `inspect.getsource()` + assert `eval`/`exec`/`compile`/`__import__` NOT present |
| Performance (O(1) vs O(n²)) | Time 1000 operations, assert <5s (O(1) ~ 0.03s, O(n²) ~ 10-50s) |
| Oracle stubs | Call `_run_check()` directly, assert detail length >10 and no "assumed" in text |
| Wiring (prompt evolution) | `register_template` → `record_outcome` → verify `uses`/`successes`/`best_score` |

### Key API Gotchas Discovered During Testing

- **`EnhancedProgram.id`** is a computed `@property` from code hash, NOT a constructor parameter. Create programs with `code='unique_code'` and read `.id` after construction.
- **`StrategyLoopConfig`** uses `max_attempts` (not `max_strategies`). Signature: `(total_budget_s, max_attempts, budget_fraction, min_attempt_budget_s, initial_exploration, exploration_decay, patience, min_improvement)`
- **`run_strategy_loop`** signature: `(config: OrchestrateConfig, loop_config: StrategyLoopConfig, orchestrate_fn: Callable)` - NOT `(X, y, goal, ...)`
- **`ExperimentRegistry.record()`** takes a single `ExperimentRecord` object, NOT keyword args like `task_type=...`
- **`ExperimentRegistry.retrieve()`** signature: `(fingerprint, task_type, top_k)` - NOT `(task_type=..., limit=...)`
- **`PromptEvolver`** method is `get_best_template()` (not `get_best()`)

## LLM Diagnosis Testing (P0 Capability)

### Testing LLM Response Parser
The `_parse_llm_response()` in `frontier/llm_diagnosis.py:283-324` might have header-content alignment issues. LLMs often put content on the same line as the header (e.g., `ROOT CAUSES: The performance...`) rather than on subsequent lines. Test with a real API call:

```python
import os
from attestra.execution.gpu_backend import build_frontier_llm_client
from frontier.llm_diagnosis import _parse_llm_response, llm_diagnose
from frontier.diagnosis import Diagnosis
from frontier.types import Task

api_key = os.environ.get("PRIME_INTELLECT_API_KEY")
llm_client = build_frontier_llm_client(api_key=api_key)

# Build a diagnosis and task, then call llm_diagnose
task = Task(kind="classification", metric="accuracy", n_features=54, n_classes=7, n_train=3500, n_val=750)
diag = Diagnosis(best_score=0.75, n_ok=2, n_fail=1, dominant_error_kind="timeout", plateau=False)

# Provide real val_truth/val_preds to trigger confusion matrix
import numpy as np
val_truth = np.array([0,0,0,1,1,1,2,2,2,2])
val_preds = np.array([0,0,1,1,1,2,0,2,2,2])

result = llm_diagnose(diag, task, [], llm_client=llm_client, val_truth=val_truth, val_preds=val_preds)
# VERIFY: result.available == True
# VERIFY: result.root_causes is non-empty (if empty, parser is dropping content)
# VERIFY: result.repair_strategies is non-empty
# VERIFY: result.confusion_analysis is not None
# VERIFY: result.confusion_analysis.weakest_class is correct
```

### Testing LLM Proposal Quality
LLM proposals might generate wrong task types (e.g., regressors for classification). Test:

```python
from frontier.core.authoring import CoreAuthoringProposer, AuthoringConfig

proposer = CoreAuthoringProposer(client=llm_client, config=AuthoringConfig(n=2))
ctx = {"kind": "classification", "n_features": 54, "n_classes": 7, ...}
proposals = proposer.propose(ctx)
# VERIFY: proposals use Classifier models, not Regressors
# Check proposer.rejections for firewall activity
```

### Stress Testing with LLM Enabled
The repair loop (`_try_repair` in `orchestrator.py`) runs on every failed proposal with LLM calls (4-13s each). This might cause timeouts on real datasets. Budget accordingly:

```python
from frontier.core.orchestrator import CoreOrchestrator, CoreConfig
config = CoreConfig(wall_seconds=300, cpu_seconds=30, llm_client=llm_client, rounds=2, total_seconds=300)
orch = CoreOrchestrator(config)
result = orch.run("classification", X, y, theta=0.70)
# Wine (178 samples): ~165s with LLM
# Covtype (5000 samples): might timeout at 300s+ due to repair loop
```

### Key LLM Diagnosis API Notes
- `llm_diagnose()` returns `LLMDiagnosis` with fields: `available`, `raw_response`, `root_causes`, `targeted_guidance`, `recommended_architectures`, `repair_strategies`, `confusion_analysis`
- `enrich_context_with_llm_diagnosis()` sets 4 context keys: `llm_diagnosis`, `llm_guidance`, `confusion`, `repair_strategies`
- `build_frontier_llm_client()` in `gpu_backend.py` creates the adapter for Prime Intellect inference
- The `CoreConfig` class is in `frontier.core.orchestrator` (not `OrchestrateConfig`)
- `CoreOrchestrator.run(goal, X, y, theta=)` - goal is a string like "classification"
- Result fields: `certified`, `winner`, `winner_val_score`, `decline_reason`, `certificate`

## Common Issues

- **`static_check` API**: Returns `StaticCheckReport(ok, violations, has_entrypoint)` dataclass, NOT a tuple. Access `.ok` and `.violations`, not tuple unpacking.
- **`--no-llm` doesn't reach catalog**: It disables LLM within frontier, not frontier itself. Use `--engine catalog` CLI flag or `engine="catalog"` in `OrchestrateConfig` to force catalog path.
- **`run_sandboxed` memory failures**: In constrained environments, sklearn shared object loading may exceed rlimit memory caps. Increase `mem_mb` parameter or test `static_check` separately.
- **`test_breast_cancer` flakiness**: This test may intermittently fail with `best_score=0.0, decision='error'` under tight time budgets. Re-run to confirm - it's non-deterministic variance, not a regression.
- If `from vectorforge.science` ImportError appears in engine.py, the migration is broken - engine should import from `attestra.core.science`
- If `problem_spec` is None with an API key set, check that `type_problem()` is called in the INTAKE stage
- If `phases_completed` is 0 but decision is not error, the ExperimentManager wiring is broken
- Pre-existing test failures (torch, pandas, novelty assertions) are NOT regressions - they exist on the base branch
- **ENOMEM on constrained VMs**: `test_full_orchestrate` and catalog-path tests may fail with `[Errno 12] Cannot allocate memory` when sandbox forks. This is an environment limitation, not a code bug. The orchestrator correctly returns `decision='error'`.
- **Clean clone verification**: Always verify on a fresh `git clone` - never trust results from a working tree with local `.pyc` caches or uncommitted shims
- **Frontier CLI tests take 2-5 minutes each** - the frontier engine runs 30+ proposals with LLM calls. Budget accordingly.
- **Iris oracle veto is correct behavior** - petal length (feature f3) alone achieves perfect accuracy on Iris, so the adversarial_self_refutation oracle flags it as a single-column artifact. This is the oracle working as designed, not a failure.
