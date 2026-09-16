# 04 - Core integration: the end-to-end CORE loop

**Status:** implementation-ready design. **Scope:** how THE CORE (model authoring, backend-agnostic
execution, sandboxed neural architecture) composes with the platform layer (router/harness, diagnosis,
agentic repair, experiment, data_ops, checkpoint/portfolio/budget, oracles, knowledge, report) into one
coherent loop, using composition only.

**Hard constraint honored throughout:** Phase-0 `engine.py` is FROZEN. We do not edit it. We either wrap
`ResearchEngine` in a thin `CoreOrchestrator`, or we wire the new capabilities in through the two seams
Phase-0 already exposes: the `proposers: list[ProposalSource]` constructor argument and the
`EngineConfig.llm_client` callable. This doc specifies the wrapper plus the wiring contracts, so the
integration model has one target to build against.

Interpreter for any verification: `/Users/abdullahalghamdi/jax-env-311/bin/python`.

---

## 0. Two composition options, and the one we choose

Phase-0 `ResearchEngine.run(task)` already does, per round: build context dict -> gather proposals from
`self.proposers` -> dedup by label/id -> `sandbox.run_program` each -> `score_val` -> track champion ->
after rounds, one sandbox run of the winner on `splits.sealed_rows` -> `certify_on_sealed`. It enforces
invariants 1, 2, 4, 5 from CONTRACT.md by construction.

The new capabilities (LinUCB ranking, portfolio early-kill, agentic repair, oracles, knowledge,
diagnosis depth, harness/router, report) are all things Phase-0's single straight-line `run()` does not
do. There are two ways to add them without editing `engine.py`:

- **Option A - pure proposer/executor wiring.** Inject everything reachable through the two seams. The
  authoring engine, feature proposer, and neural core become `ProposalSource` objects; `llm_client`
  carries the model. This is the smallest possible change, but it cannot add ranking, early-kill,
  oracles, knowledge recording, report, or per-round diagnosis enrichment, because those live OUTSIDE the
  round loop that `engine.py` owns. Option A alone cannot satisfy the task's requirements.

- **Option B - a thin `CoreOrchestrator` that wraps `ResearchEngine`.** The orchestrator owns the outer
  control: router/harness/Task construction, knowledge warm-start, proposal-source assembly, LinUCB
  ranking, portfolio execution, agentic repair, diagnosis enrichment, oracle gating, knowledge recording,
  and report. It uses `ResearchEngine` (or its sound certify path) for the FROZEN parts: split discipline
  and the single sealed certification.

**We choose Option B, with Option A's wiring nested inside it.** Rationale: only the wrapper can host the
modules that operate between rounds and after selection, and only the wrapper can keep the sealed test
touched exactly once while running a richer per-round batch. The Phase-0 engine is reused for what it
proves correct (the split + the one-peek certificate); the orchestrator owns everything generative.

### B.1 How the wrapper avoids editing engine.py but still reuses its certify discipline

There are two sub-modes; the orchestrator picks per deployment:

- **B-delegate (default, lowest risk):** the orchestrator does its own rich per-round search (ranking +
  portfolio + repair + diagnosis), then to certify the winner it calls the FROZEN
  `certify.make_splits` / `sandbox.run_program(winner, ..., X_sealed)` / `certify.certify_on_sealed`
  path directly - the identical three calls `engine.py` lines 164-173 make. The sealed test is touched
  exactly once, in the orchestrator, after oracles clear the winner. `engine.py` is imported for its
  types and for `ResearchEngine` as a fallback, but the rich loop reuses `certify.py`/`sandbox.py`, which
  are also Phase-0 frozen. This keeps a single sealed-peek site under the orchestrator's control, which is
  what we need because oracle gating must happen BEFORE that peek.

- **B-embed (for parity tests):** the orchestrator assembles `proposers=[...]` and
  `EngineConfig(llm_client=...)` and calls `ResearchEngine.run(task)` unchanged, accepting that ranking is
  approximated by proposer ordering, that there is no early-kill or repair, and that oracles run only as a
  post-hoc check on the returned `EngineResult` (it is too late to block the peek, so a failed oracle
  forces a decline of an already-certified number, which is honest but wasteful). B-embed exists so we can
  diff the wrapper's certificate against the frozen engine's certificate on the same Task and prove we did
  not perturb the certify math.

The rest of this doc specifies **B-delegate**, since it is the one that satisfies all the requirements.

---

## 1. The full pipeline

```
goal + raw data + verification standard (theta, metric)
   |
   v
[router.route(goal, X, y, llm_client)] ----> Harness
   |
   v
[harness.self_test()] -- must PASS on its known-good benchmark before its numbers are trusted
   |   (fail -> honest decline: "no trusted harness for this task type")
   v
[harness.build_task(goal, X, y)] ----> Task  (X, y, kind, theta, metric, name)
   |
   v
[knowledge.warm_start(task_signature)] ----> prior recipes/programs + LinUCB state  (transfer)
   |
   v
==================== per-round loop (orchestrator owns it) ==========================
   |  context = base context (Phase-0 keys) enriched by diagnosis.enrich_context(...)
   |
   |  proposal sources, each .propose(context) -> list[Program]:
   |     1. core/authoring.py  (AgenticProposer / authoring engine; needs llm_client)
   |     2. features.py        (FeatureProposer / FeatureMutationProposer)
   |     3. core/neural.py     (sandboxed neural architecture core; emits torch Programs)
   |     4. seed/mutation FLOOR (Phase-0 SeedProposer + MutationProposer; no LLM needed)
   |
   |  pooled, deduped by Program.label / Program.id  (Phase-0 dedup rule)
   |
   |  [knowledge LinUCB ranker].rank(programs, context) ----> ordered programs
   |
   |  [budget.schedule(ordered, round_budget)] ----> the round's batch + per-arm caps
   |
   |  [portfolio.run_batch(batch, run_fn=execute_on_backend, early_kill=...)] :
   |       for each Program -> core/execution.execute(program, backend, splits, budget)
   |          backend in {sklearn, torch}; torch programs route to core/neural sandbox
   |          returns RunResult (preds on VAL only) ; losers killed early
   |
   |  failures (RunResult.ok == False):
   |       [agentic.repair_loop(program, run_result, context)] -> repaired Program | None
   |       repaired program re-enters execute once (bounded); still-failed -> recorded error
   |
   |  score survivors on VAL via certify.score_val ; update champion (best_prog, best_score)
   |
   |  [diagnosis.diagnose(history, trail, task)] ----> Diagnosis (per-source error rates,
   |       plateau, residual structure) ; feeds next round's enrich_context
   |
   |  [knowledge].update(LinUCB reward = val lift per source)  (selection-side learning only)
   |  [checkpoint.save(round_state)]  (resumable)
==================================================================================
   |
   v   (after rounds, or budget exhausted, or plateau)
champion = best_prog on VAL   (no candidate ran -> honest decline now)
   |
   v
[oracles.verify_before_promote(task, splits, winner=champion, cert=None, run_fn)]
   |   permuted-label collapse, trivial-baseline-beaten, metric-axis, seed-repro,
   |   adversarial self-refutation. These re-run the champion on TRAIN/VAL-derived
   |   probes via run_fn; they DO NOT touch sealed_rows.
   |   Verdict.ok == False  -> honest decline ("winner refuted by oracle: <which>")
   v   (Verdict.ok == True)
==== THE ONLY SEALED TOUCH ====
[sandbox.run_program(champion, X_train, y_train, X_sealed, kind)] -> sealed preds
[certify.certify_on_sealed(task, splits, sealed_preds)] -> certificate dict (one counted peek)
==================================
   |
   v
[knowledge.record(task_signature, champion, certificate, diagnosis_trail)]  (durable KB)
   |
   v
[report.render(goal, task, history, diagnosis_trail, certificate, repro)] -> artifact
   |
   v
EngineResult-shaped outcome: certified result OR honest decline (never a relabeled val score)
```

The pipeline is the same shape as Phase-0's `run()`, with five insertions: (a) router/harness/Task in
front; (b) knowledge warm-start before round 1; (c) rank + portfolio + repair + deep diagnosis inside
each round; (d) oracle gate immediately before the single sealed peek; (e) knowledge.record + report
after.

---

## 2. Exact data/contract flow between modules

Notation: `consumes -> returns`. Types reference CONTRACT.md and the roadmap responsibilities.

### 2.1 Front matter (once per goal)

| Module | Consumes | Returns | Context effect |
|---|---|---|---|
| `harness/router.route` | `goal:str, X, y, llm_client` | `Harness` | none yet |
| `Harness.self_test` | (internal known-good benchmark) | `bool` (or raises) | gate: fail -> decline |
| `Harness.build_task` | `goal, X, y` | `Task` (CONTRACT §task.py) | sets `task.kind/metric/theta` |
| `certify.make_splits` | `Task, seed, test_frac, val_frac` | `Splits{train_rows, val_rows, sealed_rows, meta}` | `splits.meta` -> base context |
| `knowledge.warm_start` | task signature `(kind, n_features, metric, name-class)` | `list[Program]` (source `"retrieval"`), prior LinUCB params | seeds round-1 proposal pool; sets `context["prior_best_recipe"]` |

`task_signature` is computed by the orchestrator from `Task` fields only (kind, n_features, metric,
coarse name class). It never includes anything derived from `sealed_rows`. This is what keeps transfer on
the proposal side of invariant 4.

### 2.2 The base context dict (Phase-0 compatible, then enriched)

The orchestrator builds exactly the Phase-0 context keys first, so every existing `ProposalSource`
(including `SeedProposer`, `MutationProposer`, `LLMProposer`) keeps working unchanged:

```
context = {
  "task_kind": task.kind, "n_features": task.n_features, "n_train": len(splits.train_rows),
  "round": r, "tried_labels": set(...), "best_label": ..., "best_score": ...,
  "best_id": ..., "best_recipe": ..., "recent_errors": list[(label, error_kind, msg)],
}
```

Then enrichment, in this order, each writing NEW keys (never overwriting Phase-0 keys that existing
sources read):

1. `knowledge` adds `context["prior_best_recipe"]`, `context["transfer_notes"]`.
2. `diagnosis.enrich_context(context, diagnosis)` adds `context["diagnosis"]` =
   `Diagnosis{per_source_error_rate, plateau:bool, residual_structure, suggested_factors}` and may add
   `context["llm_hint"]` (free text for the authoring prompt).

**Single enrichment site.** Enrichment happens in exactly one place: the top of the round loop in
`CoreOrchestrator._build_context(r, ...)`. No module mutates the context outside that function. Proposal
sources receive it read-only (they copy what they need). This is the rule that keeps the context flow
auditable.

### 2.3 Proposal sources (each round)

All four implement the `ProposalSource` Protocol: `propose(context) -> list[Program]`.

| Source | Consumes from context | Returns | LLM? |
|---|---|---|---|
| `core/authoring.AgenticProposer` | task_kind, n_features, n_train, best_*, recent_errors, diagnosis, llm_hint, transfer_notes | `Program(source="llm", code=<authored module>)` | yes (degrades to []) |
| `features.FeatureProposer` / `FeatureMutationProposer` | task_kind, best_recipe, diagnosis.suggested_factors | `Program(source="feature"/"mutation")` carrying feature-eng / target-transform recipes | no (recipe-driven; LLM optional) |
| `core/neural` proposer | task_kind, n_features, n_train, diagnosis | `Program(source="neural", code=<torch build_estimator>)` | optional |
| `SeedProposer` + `MutationProposer` (FLOOR) | Phase-0 keys only | seed/mutation Programs | no |

Every returned object is a `Program` (CONTRACT §program.py): `code` defines `build_estimator()`,
`source` tags provenance, `provenance` may carry `{"recipe": {...}}` or `{"prompt_chars": n}` or
`{"arch": {...}}`. The orchestrator pools and dedups by `label`/`id` using the exact Phase-0 rule
(engine.py:127-130).

### 2.4 Ranking, scheduling, execution (each round)

| Module | Consumes | Returns |
|---|---|---|
| `knowledge` LinUCB ranker | deduped `list[Program]`, context features (source one-hot, recipe tags, round) | ranked `list[Program]` |
| `budget.schedule` | ranked list, remaining round/global budget, per-program cost estimate | batch to run + per-arm `(wall_seconds, cpu_seconds)` caps |
| `portfolio.run_batch` | batch, `run_fn`, early-kill rule | `list[RunResult]`, with losers killed early |
| `core/execution.execute` (the `run_fn`) | `Program`, backend, `(X_train, y_train, X_val)`, caps | `RunResult` (preds on VAL only) |
| `core/neural` sandbox | torch `Program`, same inputs | `RunResult` |

`core/execution.execute` is the backend-agnostic substrate. It dispatches on the Program: sklearn
programs go to the Phase-0 `sandbox.run_program` runner verbatim; torch programs go to `core/neural`'s
sandboxed runner, which has the same `RunResult` contract (preds only, typed errors, wall seconds). This
is the firewall: in BOTH backends the child returns predictions only; no metric ever crosses the process
boundary.

`portfolio.run_batch`'s `run_fn` signature is `run_fn(program, eval_split) -> RunResult` where
`eval_split` is VAL during search. Early-kill uses only wall-time / partial-progress signals exposed by
the executor, never a sealed number.

### 2.5 Repair (each round, on failures only)

`agentic.repair_loop(program, run_result, context) -> Program | None`. Consumes the failed `Program`, its
`RunResult` (`error_kind`, `error`), and the read-only context (so it sees `recent_errors` and
`diagnosis`). Returns a repaired `Program` (new `id`, `parent_id=program.id`, `source="llm"` or the
original source) or `None` if it cannot fix it within its own bounded budget. The repaired program is
executed once more via the same `run_fn`; a second failure is recorded as a terminal error and fed to
diagnosis. Repair requires the LLM client; with no client it is a no-op returning `None`.

### 2.6 Scoring and selection (each round)

Survivors with `RunResult.ok` are scored by `certify.score_val(task, splits.val_rows, res.preds)` - the
TRUSTED PARENT computes the number, exactly as Phase-0 engine.py:146. Champion update is the Phase-0 rule:
keep the highest VAL score. The orchestrator records a `_Record`-shaped row per candidate into `history`
(reusing the engine's `_Record` fields so `report` and `summary()` stay compatible).

### 2.7 Diagnosis (end of each round)

`diagnosis.diagnose(history, diagnosis_trail, task) -> Diagnosis`. Consumes the accumulated history and
trail (val scores, error kinds per source, champion trajectory) and the Task. Returns the `Diagnosis`
that `enrich_context` will fold into next round's context. It reads VAL-side and history data only; it
never reads `splits.sealed_rows`.

### 2.8 Oracle gate (once, after the rounds, before any sealed touch)

`oracles.verify_before_promote(task, splits, winner, cert, run_fn) -> Verdict`. Per the roadmap signature.
At this call `cert` is `None` (not yet certified). The oracles re-run `winner` through `run_fn` on
probes constructed from TRAIN/VAL only:
- permuted-label run must collapse to chance,
- a trivial baseline must be beaten,
- the metric must be computed on the correct axis,
- seed-controlled re-execution must reproduce the VAL score,
- adversarial self-refutation tries to break the result.

It returns `Verdict{ok:bool, reason:str, evidence:dict}`. **The orchestrator MUST NOT touch
`splits.sealed_rows` until `Verdict.ok` is True.** A `False` verdict is an honest decline carrying the
verdict reason; no peek is spent.

### 2.9 The single sealed touch (once, gated)

Identical to Phase-0 engine.py:164-173, executed in the orchestrator AFTER the oracle clears:
```
X_sealed = Task.rows_to_X(splits.sealed_rows)
final = sandbox.run_program(winner, X_train, y_train, X_sealed, kind=task.kind, caps...)
if not final.ok: -> honest decline ("winner failed on sealed re-fit")
cert = certify.certify_on_sealed(task, splits, final.preds)   # ONE counted peek
```
`splits.sealed_test` is a fresh `SealedTest(max_peeks=1)`; a second peek raises `PeekViolation`. The
orchestrator constructs the sealed test exactly once and never reads `sealed_rows` anywhere else.

### 2.10 Knowledge record + report (once, after certify)

| Module | Consumes | Returns |
|---|---|---|
| `knowledge.record` | task_signature, winner, certificate (or decline reason), diagnosis_trail, LinUCB updates | persists to durable KB |
| `report.render` | goal, task, history, diagnosis_trail, certificate, split_meta, repro command | research artifact (markdown + repro) |

`knowledge.record` stores BOTH certified wins and honest declines (declines are first-class signal for
warm-start and for the LinUCB prior). The certificate it stores is the frozen-certifier certificate, the
only promotion-bearing number.

### 2.11 Where the firewall holds

The firewall is "untrusted code returns predictions only; every decision number is computed by the
trusted parent". It holds at four crossings, all enforced by the same contract:
1. sklearn execution - Phase-0 `sandbox.run_program` (child writes preds.npy, never a metric).
2. torch execution - `core/neural` sandbox (same RunResult contract).
3. VAL scoring - `certify.score_val` runs in the parent on returned preds.
4. SEALED scoring - `certify.certify_on_sealed` runs in the parent on returned preds, one peek.
No proposer, repair loop, harness, or oracle ever receives a number that the trusted parent did not
compute. `report` only formats numbers the parent already computed.

---

## 3. Standing invariants and where each is enforced

From CONTRACT.md §"Standing invariants" and the roadmap §3.

| # | Invariant | Enforced where |
|---|---|---|
| 1 | Only the frozen certifier promotes; sealed touched exactly once, for the winner. | §2.9: the orchestrator has the ONLY `sandbox.run_program(..., X_sealed)` + `certify.certify_on_sealed` site, reached only after the oracle gate. `Splits.sealed_test` is `max_peeks=1`. No other module imports `splits.sealed_rows`. |
| 2 | Untrusted code returns predictions only; every number computed by the trusted parent. | §2.11: both execution backends honor the `RunResult` preds-only contract; all scoring is in `certify.py` in-parent. |
| 3 | Hardcoded heuristics are seeds/fallbacks, never the promotion-bearing or search-bounding decision. | §2.3 floor sources (`SeedProposer`/`MutationProposer`/feature recipes) supply a floor; the promoter is the frozen certifier (inv. 1); the LinUCB ranker only ORDERS proposals, it never prunes the floor and never decides promotion. The neural arch templates are seeds the LLM can replace. |
| 4 | Generalization expands what may be PROPOSED, never what may PROMOTE. | §2.1 `knowledge.warm_start` and the LinUCB ranker act ONLY on the proposal pool and its ordering. `task_signature` excludes sealed data. The promotion path (oracle gate + sealed certify) is untouched by transfer. |
| 5 | Outcomes honest: certified result or honest decline, never a relabeled val score. | The orchestrator returns a certified certificate only from §2.9. Every other exit (no candidate ran, harness self-test failed, oracle refuted, sealed re-fit failed, sealed lower bound below theta) returns a decline with a reason and `certified=False`. The VAL champion score is reported as `winner_val_score`, never as the result. |

Additional self-test invariant (roadmap Phase 3): a harness's numbers are trusted only after
`harness.self_test()` passes. Enforced at §2.1 - a failed self-test short-circuits to a decline before any
Task is built.

---

## 4. Ordering / concurrency constraints and failure handling

### 4.1 Ordering (hard)

1. router/harness/self_test/build_task **before** any split. No Task, no split.
2. `make_splits` **before** round 1. Splits are created once and reused; sealed_rows are partitioned out
   here and never re-split.
3. knowledge warm-start **before** round 1 proposal gather (so priors seed the pool).
4. Within a round, the strict order is: build+enrich context -> gather proposals -> dedup -> rank ->
   schedule -> execute (portfolio) -> repair failures -> score survivors -> update champion -> diagnose
   -> knowledge LinUCB update -> checkpoint. Ranking must precede scheduling (budget needs an order);
   diagnosis must follow scoring (it reads val scores).
5. Oracle gate **after** the last round and **before** the sealed touch. This ordering is load-bearing:
   the oracle must be able to BLOCK the peek.
6. Sealed touch -> knowledge.record -> report. Record and report are after the certificate exists.

### 4.2 Concurrency

- Per-round candidate execution is the only concurrent stage: `portfolio.run_batch` runs the batch with
  early-kill of losing arms. Each arm is an independent subprocess (Phase-0 sandbox / neural sandbox), so
  arms share nothing; concurrency is safe. The budget scheduler sets the max in-flight arms and per-arm
  caps.
- Everything else is sequential. Rounds are sequential (round N+1's context depends on round N's
  diagnosis). The sealed touch is strictly single-threaded and single-shot.
- `checkpoint.save` at the end of each round makes the loop resumable: on resume, completed rounds are
  not redone (the roadmap Phase-6 fix to the broken restart-every-phase resume). Resume rebuilds the
  same `splits` from the saved seed so the sealed partition is identical, and the peek counter (max_peeks
  =1) still applies because the certificate is computed at most once across the whole run.

### 4.3 Degradation when the LLM client is absent (`llm_client is None`)

The system must still produce a certified result or an honest decline; it degrades to OFFLINE FLOORS:

| Capability | LLM present | LLM absent (offline floor) |
|---|---|---|
| `core/authoring.AgenticProposer` | authors full pipelines | `propose()` returns `[]` (Phase-0 LLMProposer rule: do not fabricate) |
| `agentic.repair_loop` | repairs failures | no-op, returns `None`; failures are recorded |
| `features` proposers | LLM-suggested transforms + recipe floor | recipe-driven feature/target transforms only (still real, no LLM) |
| `core/neural` | LLM-authored architectures | template architectures only (seeds), or `[]` if torch unavailable |
| `harness` authoring (no harness exists) | LLM writes + self-tests a harness | decline if no registered harness matches and none can be authored |
| seed/mutation FLOOR | active | active (this is the always-on generative floor) |
| LinUCB ranker, portfolio, budget, oracles, certify | active | active (none need the LLM) |

So with no LLM the loop reduces to: harness/router -> seed+mutation+recipe-feature proposals -> LinUCB
rank -> portfolio execute -> diagnose -> oracle gate -> sealed certify -> report. This is exactly the
Phase-0 generative-offline guarantee, plus ranking, portfolio, oracles, knowledge, and report. The
`llm_active` flag (Phase-0 EngineResult field) is propagated so the report states honestly which paths
were live.

### 4.4 Failure handling

- Harness self-test fail -> decline, no Task built.
- No proposals in a round -> break the loop (Phase-0 rule), proceed to the champion check.
- All candidates fail in a round -> errors recorded, fed to diagnosis; next round still runs.
- No candidate ever succeeded -> hard honest decline ("no candidate executed successfully"), no oracle,
  no peek.
- Sandbox NaN/Inf or crash -> `RunResult.ok=False` with typed `error_kind`; counted, never silently
  swallowed (matches the project rule and the Phase-0 sandbox taxonomy).
- Oracle refutation -> honest decline, no peek spent.
- Winner fails on sealed re-fit -> decline ("winner failed on sealed re-fit"), the one peek is NOT
  consumed by certify (the re-fit failed before certify), so the certificate is simply absent.
- Sealed lower bound below theta -> honest decline carrying the certificate attempt (Phase-0 behavior).
- Budget exhausted mid-run -> stop proposing, take the current champion through the oracle+certify path
  (a certified result on a smaller search is still a valid certified result).

---

## 5. Minimal reference wiring (pseudocode) + sequence diagram

### 5.1 Reference wiring

```python
# core/orchestrator.py  (NEW; composition only; does NOT edit engine.py)
from frontier import certify, sandbox                      # FROZEN Phase-0
from frontier.task import Task
from frontier.program import Program
from frontier.proposers import SeedProposer, MutationProposer
from frontier.core import authoring, neural, execution     # THE CORE
from frontier import features, diagnosis, agentic, oracles, knowledge, report
from frontier import budget, portfolio, checkpoint
from frontier.harness import router

class CoreConfig:
    rounds = 4; seed = 0; test_frac = 0.30; val_frac = 0.20
    wall_seconds = 60.0; cpu_seconds = 55
    llm_client = None          # prompt -> code ; None => offline floors
    max_in_flight = 4

class CoreOrchestrator:
    def __init__(self, cfg=None):
        self.cfg = cfg or CoreConfig()

    def run(self, goal, X, y):
        cfg = self.cfg
        llm = cfg.llm_client

        # --- front matter: router -> harness(+self_test) -> Task
        harness = router.route(goal, X, y, llm_client=llm)
        if not harness.self_test():
            return self._decline("no trusted harness for this task type (self-test failed)")
        task = harness.build_task(goal, X, y)

        splits = certify.make_splits(task, seed=cfg.seed,
                                     test_frac=cfg.test_frac, val_frac=cfg.val_frac)
        X_train = Task.rows_to_X(splits.train_rows)
        y_train = Task.rows_to_y(splits.train_rows, task.kind)
        X_val   = Task.rows_to_X(splits.val_rows)

        sig = self._task_signature(task)          # kind, n_features, metric, coarse name
        prior_programs, ranker = knowledge.warm_start(sig)   # transfer (proposal side only)

        # --- proposal sources: authoring, features, neural, seed/mutation floor
        sources = [
            authoring.AgenticProposer(client=llm),       # [] if llm is None
            features.FeatureProposer(),
            features.FeatureMutationProposer(),
            neural.NeuralProposer(client=llm),           # templates if llm is None / [] if no torch
            SeedProposer(),
            MutationProposer(),
        ]

        history, trail = [], []
        tried = set(); recent_errors = []
        best_score, best_prog = None, None
        diag = None

        def run_fn(program, eval_X):                       # firewall-safe executor
            return execution.execute(program, X_train, y_train, eval_X,
                                     kind=task.kind,
                                     wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)

        for r in range(cfg.rounds):
            ctx = self._build_context(task, splits, r, tried, best_prog, best_score,
                                      recent_errors, sig)
            ctx = diagnosis.enrich_context(ctx, diag) if diag is not None else ctx

            # gather + dedup (Phase-0 rule), prepend transfer warm-start in round 0
            pool, seen = [], set()
            warm = prior_programs if r == 0 else []
            for src_out in [warm] + [s.propose(ctx) for s in sources]:
                for p in src_out:
                    if p.label in tried or p.id in seen:
                        continue
                    seen.add(p.id); pool.append(p)
            if not pool:
                trail.append({"round": r, "note": "no new proposals"}); break

            ordered = ranker.rank(pool, ctx)                       # LinUCB
            batch, caps = budget.schedule(ordered, round_index=r)  # cost model

            results = portfolio.run_batch(batch, run_fn=lambda p: run_fn(p, X_val),
                                          caps=caps, max_in_flight=cfg.max_in_flight)

            for p, res in results:
                tried.add(p.label)
                if not res.ok and llm is not None:
                    fixed = agentic.repair_loop(p, res, ctx)
                    if fixed is not None:
                        res = run_fn(fixed, X_val)
                        p = fixed; tried.add(p.label)
                if res.ok:
                    score = certify.score_val(task, splits.val_rows, res.preds)
                    history.append(_record(p, res, score))
                    if best_score is None or score > best_score:
                        best_score, best_prog = score, p
                    ranker.update(p, ctx, reward=score)            # selection-side learning
                else:
                    recent_errors.append((p.label, res.error_kind, res.error))
                    history.append(_record(p, res, None))

            diag = diagnosis.diagnose(history, trail, task)
            trail.append(self._round_log(r, best_prog, best_score, diag))
            checkpoint.save(self._state(r, splits, history, trail, best_prog, best_score, ranker))

        # --- no candidate ran -> honest decline (no oracle, no peek)
        if best_prog is None:
            return self._decline("no candidate executed successfully", history, trail, splits)

        # --- oracle gate BEFORE the single sealed touch
        verdict = oracles.verify_before_promote(task, splits, winner=best_prog,
                                                cert=None, run_fn=run_fn)
        if not verdict.ok:
            return self._decline(f"winner refuted by oracle: {verdict.reason}",
                                 history, trail, splits, winner=best_prog, val=best_score)

        # ===== THE ONLY SEALED TOUCH (identical to engine.py:164-173) =====
        X_sealed = Task.rows_to_X(splits.sealed_rows)
        final = sandbox.run_program(best_prog, X_train, y_train, X_sealed, kind=task.kind,
                                    wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)
        if not final.ok:
            return self._decline(f"winner failed on sealed re-fit: [{final.error_kind}] {final.error}",
                                 history, trail, splits, winner=best_prog, val=best_score)
        cert = certify.certify_on_sealed(task, splits, final.preds)     # ONE counted peek
        # ==================================================================

        knowledge.record(sig, best_prog, cert, trail, ranker)
        artifact = report.render(goal, task, history, trail, cert, splits.meta)

        return self._result(certified=bool(cert.get("certified")), certificate=cert,
                            winner=best_prog, winner_val_score=best_score,
                            history=history, trail=trail, splits=splits,
                            llm_active=(llm is not None), artifact=artifact)
```

Notes the integration model must honor:
- `_build_context` is the SOLE place context is constructed/enriched (§2.2). Proposal sources get it
  read-only.
- `execution.execute` dispatches sklearn -> Phase-0 `sandbox.run_program`, torch -> `core/neural` sandbox;
  both return `RunResult` (preds only).
- The sealed block is byte-for-byte the Phase-0 discipline; do not move it earlier than the oracle gate.
- B-embed parity test: build `sources` + `EngineConfig(llm_client=llm)`, call
  `ResearchEngine.run(task)`, and assert its certificate equals this orchestrator's certificate on the
  same `(task, seed)` with ranking/portfolio/oracle disabled. This proves we did not perturb certify math.

### 5.2 Sequence diagram (text)

```
User        Orchestrator     Router/Harness   Knowledge   Proposers(4)   LinUCB   Budget   Portfolio   Execution/Neural   Diagnosis   Oracles   Certify(frozen)   Report
 |  goal,X,y     |                 |              |            |            |        |          |              |               |           |            |              |
 |-------------->|                 |              |            |            |        |          |              |               |           |            |              |
 |               | route+self_test |              |            |            |        |          |              |               |           |            |              |
 |               |---------------->| Harness/Task |            |            |        |          |              |               |           |            |              |
 |               |<----------------|              |            |            |        |          |              |               |           |            |              |
 |               | make_splits (train/val/SEALED) [frozen]     |            |        |          |              |               |           |            |              |
 |               | warm_start(sig) |              |            |            |        |          |              |               |           |            |              |
 |               |--------------------------------->prior progs|            |        |          |              |               |           |            |              |
 |  === per round r ============================================================================================================================================== |
 |               | build+enrich context (diagnosis)                          |        |          |              |               |           |            |              |
 |               | propose(context) ----------------------------> Programs   |        |          |              |               |           |            |              |
 |               | dedup by label/id (Phase-0 rule)                          |        |          |              |               |           |            |              |
 |               | rank --------------------------------------------------->ordered |          |              |               |           |            |              |
 |               | schedule ------------------------------------------------------->batch,caps |              |               |           |            |              |
 |               | run_batch (early-kill) ------------------------------------------------------>             |               |           |            |              |
 |               |                 |              |            |            |        |   for each arm: execute |               |           |            |              |
 |               |                 |              |            |            |        |          |--------------> RunResult(preds VAL)        |           |            |              |
 |               | repair_loop on failures (LLM) ; re-run once                                  |              |               |           |            |              |
 |               | score_val(preds) [frozen, in-parent] ; update champion ; ranker.update       |              |               |           |            |              |
 |               | diagnose(history,trail,task) --------------------------------------------------------------------------------> Diagnosis  |            |              |
 |               | checkpoint.save                                                                              |               |           |            |              |
 |  === end rounds ============================================================================================================================================== |
 |               | verify_before_promote(winner, cert=None, run_fn)  [probes on TRAIN/VAL only] ------------------------------------------> Verdict        |              |
 |               |   Verdict.ok == False -> honest decline (NO peek)                                                              |           |            |              |
 |               |   Verdict.ok == True:                                                                                          |           |            |              |
 |               | run_program(winner, X_sealed) [frozen]  ==== ONLY SEALED TOUCH ====                                                       |--> preds   |              |
 |               | certify_on_sealed(preds) [frozen, ONE counted peek] ---------------------------------------------------------------------> certificate |              |
 |               | knowledge.record(sig, winner, cert, trail)                                                                                |            |              |
 |               | report.render(...) ---------------------------------------------------------------------------------------------------------------------> artifact |
 |<--------------| certified result OR honest decline (never a relabeled VAL score)                                                                                   |
```

---

## 6. Integration risks and mitigations

**Risk 1 - A second module reads `splits.sealed_rows` and breaks the one-peek invariant.** The richer
loop has many modules touching `splits` (diagnosis, oracles, portfolio, knowledge). If any of them reads
`sealed_rows` or constructs a second `SealedTest`, the certificate is no longer the single counted peek.
*Mitigation:* the orchestrator is the ONLY holder of `splits`; it passes `train_rows`/`val_rows` (and the
derived `X_train`/`X_val`) to sub-modules, never `sealed_rows`. Oracles receive `run_fn` and the `Task`,
and build their probes from TRAIN/VAL only; their signature takes `splits` but the contract forbids
reading `.sealed_rows`. Add a guard test: wrap `splits.sealed_rows` access with an assertion that the
caller is the orchestrator's sealed block, and assert `certificate["peeks"] == 1` in every end-to-end
test. The `SealedTest(max_peeks=1)` already raises `PeekViolation` on a real second peek, so this is
defense in depth.

**Risk 2 - The LinUCB ranker or budget scheduler silently becomes a search-bounding decision (violates
invariant 3/4).** If ranking prunes the floor proposals out of the batch (e.g. budget too small to ever
run a seed), the promotion-bearing search is implicitly bounded by a learned heuristic, and a cold task
could be denied its floor. *Mitigation:* the budget scheduler reserves a fixed slot for at least one
seed/mutation FLOOR program every round (a "floor guarantee"), so the deterministic floor is always
runnable regardless of LinUCB scores. Ranking only ORDERS; it never removes a candidate from eligibility,
it only defers it to a later round if budget is tight. Knowledge warm-start ADDS prior programs to the
pool; it never removes the floor. Test: on a brand-new task signature with no KB history, assert the
batch contains at least one `source in {"seed","mutation"}` program.

**Risk 3 - Oracle gate ordering regresses to post-hoc (LLM-absent or refactor drift), spending a peek on a
result the oracle would refute.** In B-embed mode, or if a refactor moves the certify call before
`verify_before_promote`, an oracle failure becomes a decline of an already-certified number: honest but it
burned the single peek and may mislead the report. *Mitigation:* B-delegate is the default precisely so
the oracle can block the peek; the oracle gate and the sealed block live in adjacent lines in the
orchestrator with the gate first, and a test asserts that a deliberately leaky pipeline (roadmap Phase-7
acceptance) yields `certified=False` with `certificate is None` and `peeks` unrecorded, proving no peek
was spent. The B-embed path is restricted to parity testing and is documented as not promotion-safe for
oracle-gated runs.
