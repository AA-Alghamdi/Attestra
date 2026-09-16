# INTEGRATION.md - wiring Phases 1–9 into the Phase 0 spine

Goal: stand up the frontier autoresearcher by **composition only**. Do **not** edit any Phase 0 file
(`program.py`, `task.py`, `certify.py`, `sandbox.py`, `proposers.py`, `engine.py`, `__init__.py`,
`demo.py`, `tests/test_spine.py`). Every Phase 1–9 module is additive: it either (a) satisfies the
`ProposalSource` Protocol and is passed into `ResearchEngine(proposers=[...])`, (b) wraps the round
loop from the outside, or (c) consumes/gates around `EngineResult`. The frozen certifier
(`vectorforge.science` + `vfplatform.sealed`, reached via `certify.py`) remains the **only** promoter
and the sealed test is touched **once** for the winner.

Interpreter for all verification: `/Users/abdullahalghamdi/jax-env-311/bin/python`.

The Phase-0 contract these modules bind to (frozen):
- `Program(code, source, label, parent_id, provenance)`, `.id == "source:label:sha12"`.
- `Task(X, y, kind, theta, metric, name)`; `.n_features`, `.labels`; `to_rows()`, `rows_to_X(rows)`,
  `rows_to_y(rows, kind)`.
- `certify.make_splits(task, *, seed, test_frac, val_frac) -> Splits` (`.train_rows/.val_rows/.sealed_rows/.meta`,
  `.sealed_test` = one-peek SealedTest); `certify.score_val(task, val_rows, preds) -> float`;
  `certify.certify_on_sealed(task, splits, sealed_preds) -> dict`.
- `sandbox.run_program(program, X_train, y_train, X_eval, *, kind, wall_seconds, cpu_seconds) -> RunResult`
  (`.ok/.preds/.error/.error_kind/.wall_seconds`) - predictions-only firewall.
- The per-round `context` dict the engine builds (engine.py:109–120): keys `task_kind, n_features,
  n_train, round, tried_labels(set), best_label, best_score, best_id, best_recipe(dict|None),
  recent_errors(list[(label, error_kind, msg)])`.

Two integration surfaces:
- **Surface A - proposer list** (`ResearchEngine(cfg, proposers=[...])`): pure composition, no engine
  edit. Used by features, agentic, knowledge-retrieval, diagnosis-wrapping.
- **Surface B - variant run-loop** (a thin subclass or a hand-rolled driver that re-implements the
  body of `ResearchEngine.run` using the frozen `certify`/`sandbox` calls): required by any module
  that changes *how candidates are scheduled* (portfolio, budget) or *threads state between rounds*
  (checkpoint, knowledge-ranking, diagnosis enrich). Surface B copies engine.run's structure verbatim
  and substitutes the inner loop; it does **not** modify engine.py. The recommended end-to-end driver
  (bottom of this file) is exactly such a variant.

Offline vs. LLM:
- **Run fully offline (no `llm_client`)**: features, diagnosis (deterministic directives),
  agentic (deterministic repair heuristic), experiment (template hypothesis), data_ops, checkpoint,
  portfolio, budget, oracles, knowledge, report, router (shape/dtype typer), harness/tabular,
  harness/text. Every one degrades honestly to a documented fallback.
- **Needs `llm_client: Callable[[str],str]|None` to reach its strong path**: `LLMProposer`,
  `AgenticProposer` (LLM repairer), router (LLM typer), harness **authoring** (LLM author -
  *inactive*, refuses to register, when client is None), experiment `design_experiment`,
  diagnosis `llm_guidance` (advice string, consumed by the LLM proposer). The same single
  `cfg.llm_client` callable is threaded to all of them; `None` is a first-class, honest mode.

---

## Per-module wiring

### Phase 1 - `features.py` (feature engineering as a first-class proposal source)
- **Import**: `from frontier.features import FeatureProposer, FeatureMutationProposer`
- **Surface A.** Add to the proposer list; both satisfy `ProposalSource.propose(context)->list[Program]`.
  Recommended order: after `SeedProposer` (so the plain-model floor is always tried - engine dedups by
  id/label so duplicates are free) and `FeatureMutationProposer` **last** among deterministic sources so
  it mutates whatever champion the cheaper sources found.
- **Argument shapes**: reads `task_kind, n_features, tried_labels, best_recipe, recent_errors` from
  `context`. Emits Programs carrying `provenance={"recipe": <genome>}`; the genome is a strict
  **superset** of the proposers.py recipe (`base/scale/poly/target_log` interpreted identically), so
  champions flow losslessly between `MutationProposer` and `FeatureMutationProposer`.
- **Ordering constraint**: `FeatureMutationProposer` only fires once a `best_recipe` exists (round ≥ 1).
- **Risk**: its genome `recipe_label()` must stay unique per genome or the engine's `tried_labels`
  dedup (engine.py:127) silently drops variants - verified by the feature-phase tests. Poly/PCA counts
  are stored as runtime fractions of `X.shape[1]`, so they never exceed available width after poly
  blow-up; do not "optimize" them to absolute ints. Offline.

### Phase 2 - `diagnosis.py` (round N → round N+1 feed-forward)
- **Import**: `from frontier.diagnosis import diagnose, enrich_context, DiagnosisDrivenProposer`
- **Surface B** (needs the round loop). Two edits to the *driver* (not engine.py):
  1. at construction, gate each proposer: `proposers = [DiagnosisDrivenProposer(p) for p in proposers]`
     (wrapper is backward-compatible: when `context["diagnosis"]` is absent it passes the wrapped
     source through unchanged).
  2. inside the round loop, **after** `context = {...}` is built and **before** proposers run:
     `diag = diagnose(history, trail, task); enrich_context(context, diag)`.
- **Argument shapes**: `diagnose(history: list[_Record], diagnosis_trail: list[dict], task: Task, *,
  val_truth=None, val_preds=None) -> Diagnosis`. Optional residual args enable the regression
  residual-structure signal; pass them after the champion is scored on val. `enrich_context` mutates
  `context` in place, adding `context["diagnosis"]` plus flat convenience keys, never clobbering Phase-0
  keys.
- **Integrity**: re-ranks/gates only what may be **proposed** (invariant 4); never promotes. `llm_guidance`
  is advice appended to the LLM prompt; offline the deterministic `fire_sources`/axis directives still
  steer seeds+mutations.
- **Risk**: `_source_kind` matches by class name (`seed/mutation/llm/retrieval`); a renamed or
  unrecognized proposer defaults to firing always (safe, but the gate becomes a no-op for it). Offline.

### Phase 3 - harness fabric: `harness/base.py`, `harness/tabular.py`, `harness/text.py`, `harness/router.py`, `harness/authoring.py`
- **Imports**:
  `from frontier.harness import lookup, REGISTRY, register, Harness, TabularHarness`;
  `from frontier.harness.router import route`;
  `from frontier.harness.authoring import HarnessRegistry, KnownGoodBenchmark`.
- **Front door (router)**: `harness = route(goal: str, X, y=None, llm_client=cfg.llm_client)` returns a
  `Harness` annotated with `.spec` (a `ProblemSpec`: `kind, modality, metric, theta_hint, risks, blocked`).
  - **Ordering (hard)**: `route` runs **before** `make_splits`. If `harness.spec.blocked` (a `risk`
    with severity `"block"`: reward-hacking / infeasibility / goal-data contradiction) the driver must
    **decline honestly before certifying**, recording `spec.risks` as the decline reason.
  - When the real `REGISTRY` is not yet populated, `route` returns a self-contained `_FallbackHarness`
    carrying the same `.spec`, so wiring is stable; the real harness is selected automatically once the
    registry is populated (it is, at import of `frontier.harness`).
- **Self-test gate (hard, load-bearing)**: `ok, cert = harness.self_test()`. **Trust nothing until
  `ok is True`.** `self_test()` runs the harness's adapt→split→baseline path on a known-good built-in
  dataset through the **real** `ResearchEngine`/frozen gate. If it does not certify, `harness.trusted`
  is False and the driver declines.
- **Adapt**: `task = harness.adapt(X, y, kind=spec.kind, theta=THETA, metric=spec.metric, name=...)`
  returns a frozen `Task`. `theta` is the operator's real threshold; `spec.theta_hint` is advisory only.
- **Split protocol**: `harness.split_protocol() -> (test_frac, val_frac)`; copy into `EngineConfig`.
- **Extra seeds**: `harness.baseline_suite()` returns modality-specific seed Programs; pass them into the
  proposer list (e.g. text baselines) - they are floor/fallback, never promoters.
- **Authoring (LLM, pre-loop)**: for a *novel* modality with no adapter,
  `reg = HarnessRegistry(); authored = reg.author_and_register(task_type=..., benchmark=KnownGoodBenchmark.builtin_text(), llm_client=cfg.llm_client, config=EngineConfig(...))`.
  The LLM-authored `build_task(raw)` is materialized **out of process** via the sandbox; it is registered
  **only** if its reconstructed known-good Task certifies through the frozen engine. With `llm_client=None`
  it returns `status="inactive", accepted=False` and the registry refuses it (honest decline, no fabrication).
  Then `task = authored.harness.to_task(new_raw)`.
- **Risk**: the self-test gate is the single most important invariant of Phase 3 - a buggy harness
  (wrong metric axis, leaky split) would silently corrupt every downstream certificate, so the driver
  must `assert ok`. Router/tabular/text offline; authoring's strong path needs an LLM.

### Phase 4 - `agentic.py` (write → run → read-traceback → fix loop)
- **Import**: `from frontier.agentic import AgenticProposer, repair_loop`
- **Surface A.** `AgenticProposer(llm_client=cfg.llm_client)` is a drop-in `ProposalSource`; or wrap a
  base source: `AgenticProposer(base=MutationProposer(), llm_client=...)`.
- **Behavior**: authors code from the round `context`, then runs each candidate through `repair_loop`
  (out-of-process sandbox, predictions-only) until it RUNS; returns only Programs whose code already
  executed (or the best-effort attempt so the engine still records the typed failure - never a fabricated
  success). Repaired code carries `provenance={"repair": {...}}`.
- **Argument shapes**: `repair_loop(program, X_train, y_train, X_eval, *, kind, ...)`. `AgenticProposer`
  pulls execution arrays from optional context keys `X_train/y_train/X_probe/kind` if the driver adds
  them via `attach_probe_to_context(context, X_train, y_train, X_probe, kind)`; absent, it falls back to a
  synthetic probe matched to `task_kind/n_features`. The engine's own VAL run remains authoritative; the
  probe only drives repair iteration.
- **Integrity**: the repair loop NEVER scores and NEVER touches the sealed test. Offline = deterministic
  repair heuristic (fix import path, define missing name, strip bad kwarg, synthesize missing
  `build_estimator`); LLM client = open-ended repair (strong path).
- **Risk**: if the driver does not `attach_probe_to_context`, repair verifies against a synthetic probe,
  not the real train split - the candidate still re-runs on real VAL in the engine, but repair quality is
  weaker. Prefer attaching the real probe. LLM optional.

### Phase 5a - `experiment.py` (controlled ablation: does THIS factor cause a lift?)
- **Import**: `from frontier.experiment import (Experiment, run_experiment, run_ablation, factor_program,
  scale_factor, poly_factor, noop_factor, design_experiment)`
- **Alongside** the engine (not inside its promote path). An experiment asks **two** questions of one
  locked test, so it owns its own SealedTest with `max_peeks=2`; the per-arm Bonferroni correction is
  still paid via `checks` (arm1 certifies at checks=1, arm2 at checks=2). This is a documented spec
  choice, **not** a relaxation.
- **Call**: `splits = certify.make_splits(task, ...)`; `treatment, control = scale_factor(task.kind, base="svc_rbf")`
  (differ ONLY in the varied factor); `verdict = run_experiment(exp, task, splits, treatment, control,
  sandbox_run=None, wall_seconds=..., cpu_seconds=..., margin=0.0)`. `run_ablation(exp, task, splits,
  baseline=control, levels={name:prog})` sweeps levels vs the SAME control.
- **Verdict**: contrast of the two **certified lower bounds** (conservative), not point estimates; GO iff
  treatment LB > control LB + margin. No-go is a first-class result.
- **Risk**: both arms must share the **same** `splits` object; a fresh `make_splits` per arm would launder
  the multiplicity correction. `design_experiment(goal, factor, client)` authors hypothesis text via LLM,
  template offline. Offline (template) / LLM (hypothesis authoring).

### Phase 5b - `data_ops.py` (data interventions as certifiable A/B)
- **Import**: `from frontier.data_ops import (audit_task, certify_data_change, Deduplicate,
  SMOTEOversample, Mixup, GaussianClassSynthesize, build_intervened_task)`
- **Standalone A/B (headline)**: `report = certify_data_change(base_task, SMOTEOversample(k=5,
  target="balance"), config=EngineConfig(rounds=2), seed=0)`. `report.decision in {"go","no-go"}`,
  `report.caused_lift` is the certified verdict.
- **Protocol (load-bearing)**: split once; apply intervention to **TRAIN ROWS ONLY**; both arms share the
  **identical** val and sealed rows; certify each arm's winner on the same untouched sealed split. A hard
  assert (`_assert_train_only`) guarantees no synthetic row enters val/sealed.
- **Argument shapes**: an intervention is `DataIntervention.apply(train_rows, kind, rng) -> new_train_rows`
  in the `Task.to_rows()` schema (`{"target","features","_x"}`). `audit_task(task) -> DataAudit`
  (imbalance ratio, recommended interventions) is **diagnostic only** - it informs which intervention to
  propose, never promotes.
- **Risk**: prefer `certify_data_change` (shared split) over routing `build_intervened_task` through a
  fresh `ResearchEngine.run` (which re-splits and breaks the controlled comparison). Offline.

### Phase 6a - `portfolio.py` (concurrent ASHA early-kill executor)
- **Import**: `from frontier.portfolio import run_portfolio_round, PortfolioConfig`
- **Surface B** - replaces the engine's inner serial `for p in proposals: sandbox.run_program(...)` block.
- **Call** (per round, after `proposals` built):
  `score_fn = lambda preds: certify.score_val(task, splits.val_rows, preds)` (VAL-only, trusted parent);
  `outcome = run_portfolio_round(proposals, X_train, y_train, X_val, kind=task.kind, score_fn=score_fn,
  config=PortfolioConfig(wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds))`.
- **Map back**: `outcome.records` (list[ArmRecord]) → engine `_Record`; `outcome.best_program/best_score`
  update `(best_prog, best_score)` exactly as the serial loop; `outcome.full_budget_scores` are full-train
  VAL scores (top rung), so selection is identical-in-kind to serial.
- **Integrity**: portfolio receives `(X_train, y_train, X_val)` and `score_fn` only - it **never** sees the
  sealed split and never computes a metric (firewall preserved). ASHA/Thompson decide what to RUN/KILL,
  never what promotes; the FINAL ranking is always at full budget so no early decision promotes. Honest:
  if no arm yields a finite val score, `winner=None` → engine declines.
- **Risk**: the confidence prune is admissible only under its UCB model - keep `policy` defaulting to
  `thompson`/`ucb`; do not prune below `survivors(r)`. Offline.

### Phase 6b - `budget.py` (cost model + budget controller + phase decomposer)
- **Import**: `from frontier.budget import CostModel, BudgetController, BudgetConfig, PhaseDecomposer,
  program_family, program_enhancers`
- **Surface B** - wrap the round/arm loop. Ordering is fixed: **estimate → admit → run → charge(true) →
  record_score**; `end_round` culls; `stop()` checks exhaustion.
  - `cost = CostModel()` (seed coefficients); `est = cost.estimate(program, n_train, n_features)`;
    after the run `cost.observe(program, n_train, n_features, res.wall_seconds, ok=res.ok)` for online
    self-correction.
  - `ctrl = BudgetController(BudgetConfig(total_seconds=3600.0, rounds=cfg.rounds))`; per candidate
    `if not ctrl.admit(p.id, est.seconds): continue`; after run `ctrl.charge(p.id, res.wall_seconds)`,
    `ctrl.record_score(p.id, val_score)`; `ctrl.end_round(r)`; `if ctrl.stop(): break`.
  - `plan = PhaseDecomposer().decompose(total_seconds, n_candidates, kind=task.kind); plan.validate()`
    yields gated screen→refine→confirm phases (validated acyclic, budget-conserving, narrowing).
- **Integrity**: bounds compute only; the single sealed certification of the winner is **outside** the
  budget loop and always runs (the last peek is not optional).
- **Risk**: `program_family` reads `provenance["recipe"]["base"]`; LLM-authored Programs without a recipe
  fall back to a static-analysis family guess from the code string (looser estimate). Offline.

### Phase 7 - `oracles.py` (verification beyond statistics - "cannot fool itself")
- **Import**: `from frontier import oracles`
- **Surface B / gate** - exactly ONE call between "winner certified on sealed" and "promote":
  ```python
  def _run_fn(prog, Xtr, ytr, Xev):
      r = sandbox.run_program(prog, Xtr, ytr, Xev, kind=task.kind,
                              wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)
      return r.preds if r.ok else None        # predictions-only firewall
  verdict = oracles.verify_before_promote(task, splits, best_prog, cert, _run_fn, seed=cfg.seed)
  certified = bool(cert.get("certified")) and verdict.promote
  ```
- **Argument shapes**: `verify_before_promote(task, splits, winner, certificate, run_fn, *, seed=0)
  -> Verdict(promote, reasons, oracles)`. `run_fn(program, X_train, y_train, X_eval) -> list|None`.
- **Behavior**: blocking oracles = metric-orientation, beats-trivial-baseline, no-label-leak-feature,
  permuted-label-collapse, reproducibility, plus adversarial self-refutation; distribution-drift is
  advisory (warn). It can only **VETO** a certified winner (True→False); it never manufactures a
  promotion (invariant 5).
- **Ordering**: AFTER certification (needs the certificate to compare vs trivial floor and to re-derive
  the digest), BEFORE the promotion decision. The cheap oracles do NOT re-peek the winner's sealed test;
  the reproducibility re-run peeks a **freshly rebuilt** sealed split (its own one-peek budget), leaving
  the winner's counted peek untouched.
- **Risk**: this is the firewall against high-but-meaningless certificates (leaked feature, majority-class
  win). Carry `verdict.reasons` into the report. Offline.

### Phase 8 - `knowledge.py` (compounding cross-task flywheel: KB + retrieval + LinUCB)
- **Import**: `from frontier.knowledge import (KnowledgeBase, RetrievalProposer, LinUCBRanker,
  KnowledgeProposer, task_descriptor, task_fingerprint, recipe_descriptor)`
- **Shared state once, persists across tasks**: `kb = KnowledgeBase("frontier_kb.jsonl")` (durable,
  append-only JSONL); `ranker = LinUCBRanker(alpha=1.0); ranker.load_from_kb(kb)`. Or one object:
  `know = KnowledgeProposer(kb)` exposing `propose` + `rank` + `record_outcome`.
- **Surface A (retrieval)**: put `RetrievalProposer(kb)` (or `know`) **FIRST** in the proposer list so
  warm-start seeds lead. `propose(context)` reads `context["task_fingerprint"]`; cold KB → `[]` (Phase-0
  floor unchanged).
- **Surface B (ranking + record)** - driver edits:
  1. add two context keys each round: `context["task_descriptor"] = task_descriptor(task)`;
     `context["task_fingerprint"] = task_fingerprint(task)` (depend only on `Task`).
  2. reorder the deduped proposal list before the sandbox loop: `proposals = ranker.rank(proposals, context)`
     (stable; ties keep input order; empty bandit = identity = Phase-0 order).
  3. after each VAL score: `kb.record(...)` + `ranker.update(context, p, reward=score)`; on failure record
     `val_gain=0, reward=0.0`. Or `know.record_outcome(context, p, val_score=score,
     incumbent_before=prev_best, cost_seconds=res.wall_seconds, ok=res.ok)`.
- **Shapes**: `task_fingerprint -> (FINGERPRINT_DIM,)`, `recipe_descriptor -> (RECIPE_DIM,)`,
  `context_vector -> (CONTEXT_DIM,)` - fixed and shared, so the LinUCB `(A, b)` stats are dimension-stable
  across tasks.
- **Integrity**: records store the realized **VAL gain** (selection-set number), never a sealed/relabeled
  number; the bandit reorders/seeds proposals only, never promotes. Offline.

### Phase 6 - `checkpoint.py` (durable crash-safe resume that does NOT recompute finished rounds)
- **Import**: `from frontier.checkpoint import CheckpointStore, CheckpointedRun, RoundOutcome, RunState`
- **Surface B** - wraps the round loop body:
  ```python
  store = CheckpointStore(path="runs/<task>/<run_id>.json")
  def round_fn(r: int, state: RunState) -> RoundOutcome:
      # read prior champion/payload from state; run ONE round propose->portfolio/sandbox->score->select
      return RoundOutcome(champion_id=best_prog.id if best_prog else None,
                          champion_score=best_score, records=[...],
                          payload={"tried_labels": sorted(tried_labels),
                                   "recent_errors": recent_errors,
                                   "best_prog_id": best_prog.id if best_prog else None})
  runner = CheckpointedRun(store=store, n_rounds=cfg.rounds, round_fn=round_fn,
                           run_id=<run_id>, parent_run_id=<prev if resuming>,
                           on_resume=<rehydrate champion Program from id>)
  final_state = runner.run()      # resumes automatically; skips completed rounds entirely
  ```
- **Ordering contract**: `round_fn` is called in ascending order; a round is marked complete + checkpointed
  (atomic temp-file + fsync + `os.replace` + dir fsync) only after it returns without raising. A crash loses
  at most the in-flight round.
- **Certification is one-shot, not mid-flight checkpointed**: the sealed peek is atomic; `state.payload`
  carries `certified` and `should_certify()` so a resume after certification skips re-peeking (preserves
  the one-peek invariant).
- **Integrity**: pure orchestration/persistence; computes no promotion number. Offline.

### Phase 9 - `report.py` (research artifact)
- **Import**: `from frontier.report import build_report, write_report`
- **Pure consumer of `EngineResult` + `Task`**, called AFTER the run; no engine change, no sealed peek.
  `artifact = build_report(result, task, config=cfg, goal="<text>")` → `artifact.markdown`, `artifact.json_str`;
  `md_path, json_path = write_report(result, task, out_dir, config=cfg, goal=...)`.
- **Behavior**: emits the certificate (sealed lower bound, theta, one counted peek, `sealed_digest`),
  ablations, failures-with-reasons, exact reproduction block (seed, split counts, winning
  `build_estimator()` verbatim). A declined run says DECLINED and carries the best certificate **attempt**,
  never a relabeled val score. The one derived number (LB − theta margin) is a labeled presentation aid.
- **Risk**: the JSON artifact must not leak the `llm_client` object (it is schema-tagged and scrubbed - the
  report tests assert "no client leak"). Offline.

---

## Recommended end-to-end wiring (a Surface-B variant driver)

This is a single driver that composes everything. It re-implements the body of `ResearchEngine.run`
(it does NOT edit engine.py) so it can insert scheduling, ranking, diagnosis, checkpointing, oracle-gating,
and reporting at the right seams. The frozen `certify`/`sandbox` calls and the one-peek invariant are
preserved exactly.

```python
from frontier import certify, sandbox, oracles
from frontier.task import Task
from frontier.engine import EngineConfig, EngineResult, _Record
from frontier.proposers import SeedProposer, MutationProposer, LLMProposer
from frontier.features import FeatureProposer, FeatureMutationProposer
from frontier.agentic import AgenticProposer, attach_probe_to_context
from frontier.diagnosis import diagnose, enrich_context, DiagnosisDrivenProposer
from frontier.knowledge import KnowledgeBase, KnowledgeProposer, task_descriptor, task_fingerprint
from frontier.portfolio import run_portfolio_round, PortfolioConfig
from frontier.budget import CostModel, BudgetController, BudgetConfig
from frontier.checkpoint import CheckpointStore, CheckpointedRun, RoundOutcome
from frontier.report import write_report
from frontier.harness import lookup
from frontier.harness.router import route

def run_frontier(goal, X, y, *, theta, out_dir, run_id,
                 llm_client=None, kb_path="frontier_kb.jsonl", total_seconds=3600.0):
    cfg = EngineConfig(rounds=4, llm_client=llm_client)

    # 1. ROUTER -> typed front door, BEFORE any split.
    harness = route(goal, X, y, llm_client=llm_client)
    if harness.spec.blocked:
        return EngineResult(False, None, None, None, decline_reason=str(harness.spec.risks))

    # 2. HARNESS self-test GATE (trust nothing until True), then adapt -> frozen Task.
    ok, _ = harness.self_test()
    assert ok, "harness failed self-test (untrusted adapter)"
    task = harness.adapt(X, y, kind=harness.spec.kind, theta=theta, metric=harness.spec.metric)
    tf, vf = harness.split_protocol()
    cfg.test_frac, cfg.val_frac = tf, vf

    # 3. SHARED knowledge state (persists across tasks) + bandit warm-start.
    kb = KnowledgeBase(kb_path)
    know = KnowledgeProposer(kb)                      # propose (retrieval) + rank (LinUCB) + record

    # 4. PROPOSERS: retrieval first, then feature/seed/mutation/agentic/llm; each gated by diagnosis.
    base_proposers = [
        know,                                         # warm-start retrieval (cold -> [])
        SeedProposer(), FeatureProposer(),
        MutationProposer(), FeatureMutationProposer(),
        AgenticProposer(llm_client=llm_client),       # author + self-repair
        LLMProposer(llm_client),                      # [] when client is None
    ]
    proposers = [DiagnosisDrivenProposer(p) for p in base_proposers]

    # 5. one frozen split; cost model + budget controller bound compute.
    splits = certify.make_splits(task, seed=cfg.seed, test_frac=cfg.test_frac, val_frac=cfg.val_frac)
    X_train = Task.rows_to_X(splits.train_rows); y_train = Task.rows_to_y(splits.train_rows, task.kind)
    X_val = Task.rows_to_X(splits.val_rows)
    cost = CostModel(); ctrl = BudgetController(BudgetConfig(total_seconds=total_seconds, rounds=cfg.rounds))
    score_fn = lambda preds: certify.score_val(task, splits.val_rows, preds)

    history, trail = [], []
    state = {"best_score": None, "best_prog": None, "tried": set(), "errs": []}

    # 6. RESUMABLE round loop: each round = diagnose -> propose -> rank -> PORTFOLIO early-kill (budget-aware).
    store = CheckpointStore(path=f"{out_dir}/{run_id}.json")
    def round_fn(r, rstate):
        context = {"task_kind": task.kind, "n_features": task.n_features,
                   "n_train": len(splits.train_rows), "round": r,
                   "tried_labels": set(state["tried"]),
                   "best_label": state["best_prog"].label if state["best_prog"] else None,
                   "best_score": state["best_score"],
                   "best_id": state["best_prog"].id if state["best_prog"] else None,
                   "best_recipe": (state["best_prog"].provenance.get("recipe") if state["best_prog"] else None),
                   "recent_errors": list(state["errs"]),
                   "task_descriptor": task_descriptor(task),
                   "task_fingerprint": task_fingerprint(task)}
        diag = diagnose(history, trail, task); enrich_context(context, diag)
        attach_probe_to_context(context, X_train, y_train, X_val, task.kind)   # for agentic repair

        proposals, seen = [], set()
        for src in proposers:
            for p in src.propose(context):
                if p.label in state["tried"] or p.id in seen: continue
                seen.add(p.id); proposals.append(p)
        proposals = know.rank(proposals, context)                              # LinUCB UCB order

        # budget gate (estimate->admit), then concurrent ASHA early-kill on VAL only:
        admitted = [p for p in proposals
                    if ctrl.admit(p.id, cost.estimate(p, len(splits.train_rows), task.n_features).seconds)]
        outcome = run_portfolio_round(admitted or proposals, X_train, y_train, X_val,
                                      kind=task.kind, score_fn=score_fn,
                                      config=PortfolioConfig(wall_seconds=cfg.wall_seconds,
                                                             cpu_seconds=cfg.cpu_seconds))
        for rec in outcome.records:
            state["tried"].add(rec.label)
            if not rec.ok: state["errs"].append((rec.label, rec.error_kind, rec.error))
            history.append(_Record(rec.program_id, rec.label, rec.source, rec.ok,
                                    val_score=rec.val_score, error_kind=rec.error_kind,
                                    error=rec.error, wall_seconds=rec.wall_seconds))
            ctrl.charge(rec.program_id, rec.wall_seconds)
            cost.observe(<prog>, len(splits.train_rows), task.n_features, rec.wall_seconds, ok=rec.ok)
            know.record_outcome(context, <prog>, val_score=(rec.val_score or 0.0),
                                incumbent_before=state["best_score"], cost_seconds=rec.wall_seconds, ok=rec.ok)
        if outcome.best_program is not None and (state["best_score"] is None
                                                 or outcome.best_score > state["best_score"]):
            state["best_score"], state["best_prog"] = outcome.best_score, outcome.best_program
        trail.append({"round": r, "best_out_score": state["best_score"]})
        ctrl.end_round(r)
        return RoundOutcome(champion_id=state["best_prog"].id if state["best_prog"] else None,
                            champion_score=state["best_score"],
                            payload={"best_prog_id": state["best_prog"].id if state["best_prog"] else None})
    CheckpointedRun(store=store, n_rounds=cfg.rounds, round_fn=round_fn, run_id=run_id).run()

    if state["best_prog"] is None:
        result = EngineResult(False, None, None, None, history, trail, splits.meta,
                              decline_reason="no candidate executed successfully")
    else:
        # 7. CERTIFY the single winner on the sealed test (the ONLY sealed peek).
        X_sealed = Task.rows_to_X(splits.sealed_rows)
        final = sandbox.run_program(state["best_prog"], X_train, y_train, X_sealed, kind=task.kind,
                                    wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)
        if not final.ok:
            result = EngineResult(False, None, state["best_prog"], state["best_score"], history, trail,
                                  splits.meta, decline_reason=f"winner failed on sealed re-fit: {final.error}")
        else:
            cert = certify.certify_on_sealed(task, splits, final.preds)
            # 8. ORACLES verify_before_promote -> can only VETO a certified winner.
            def _run_fn(prog, Xtr, ytr, Xev):
                r = sandbox.run_program(prog, Xtr, ytr, Xev, kind=task.kind,
                                        wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)
                return r.preds if r.ok else None
            verdict = oracles.verify_before_promote(task, splits, state["best_prog"], cert, _run_fn, seed=cfg.seed)
            promoted = bool(cert.get("certified")) and verdict.promote
            result = EngineResult(promoted, cert, state["best_prog"], state["best_score"], history, trail,
                                  splits.meta, decline_reason=("" if promoted else "; ".join(verdict.reasons)))

    # 9. REPORT (pure consumer; no sealed peek).
    write_report(result, task, out_dir, config=cfg, goal=goal)
    return result
```

Pipeline realized (matches the requested flow):
`router -> harness (+self-test) -> feature/agentic/knowledge proposers ranked by LinUCB ->
portfolio executor with early-kill + budget -> diagnosis feed-forward -> certify on sealed ->
oracles verify_before_promote -> report`.

### Cross-cutting integration risks (read before wiring)
1. **One sealed peek, owned by the spine.** Only the final winner-certification touches the sealed split
   (step 7). Modules that have their own sealed contact (`experiment` with `max_peeks=2`,
   `oracles` reproducibility with a freshly rebuilt sealed split) use **separate** SealedTest instances
   with their own counted budgets and pay multiplicity via `checks`. Never reuse the winner's SealedTest
   for a second question, and never `make_splits` per arm inside one experiment.
2. **`provenance["recipe"]` is the lingua franca.** features/diagnosis/budget/knowledge all read it.
   Keep `recipe_label()` unique per genome (engine dedup) and keep the features genome a strict superset
   of the proposers recipe. LLM/agentic Programs without a recipe degrade to static-analysis family
   guesses (budget) and label-only KB entries (knowledge) - acceptable but weaker; do not let a missing
   recipe crash a module.
3. **Diagnosis/knowledge context keys must be added in the SAME driver that builds `context`.** The
   stock `ResearchEngine.run` does not add `diagnosis`, `task_descriptor`, `task_fingerprint`, or the
   agentic probe. Wrapping proposers with `DiagnosisDrivenProposer` is safe under stock engine (no-op),
   but `know.rank`, `enrich_context`, and `attach_probe_to_context` require the Surface-B driver. If you
   run under the stock engine, retrieval/seeds still work but ranking/feed-forward/repair-probe are inert.
4. **Self-test / oracle gates are not optional.** Skipping `harness.self_test()` or
   `oracles.verify_before_promote` lets a leaky or mis-axised certificate promote - exactly the failure
   the fabric exists to prevent. Always `assert ok` and always AND the oracle verdict into `certified`.
5. **`llm_client=None` is honest, not broken.** Every LLM path degrades to a documented fallback;
   surface `llm_active` and authoring `status="inactive"` in the report rather than silently faking.
```

## THE CORE

THE CORE (`frontier/core/`, package `frontier.core`, full README at `frontier/core/README.md`) is
the generative-modeling-engine + execution-substrate layer. Like everything else it is **additive**:
no Phase-0 file is edited. Importing it is offline-safe (`import frontier.core` touches no torch / no
LLM / no network). Public symbols are re-exported from `frontier.core.__init__`. Design specs:
`frontier/core/design/01..05`.

### Per-CORE-module: import + call site

| Module | Import | Call site (where the integrator/orchestrator uses it) |
| --- | --- | --- |
| `authoring.py` | `from frontier.core import CoreAuthoringProposer` (also `AuthoringEngine`, `Firewall`, `AuthoringConfig`) | `CoreAuthoringProposer(cfg, llm_client=client)` is a `ProposalSource`. The orchestrator calls `.propose(ctx)` as proposal source **#1** each round; offline it returns `[]` and stamps `status="inactive"`. The `Firewall` sanitizes LLM-authored code (AST allow-list + symbol-registry rename) before any execution. |
| `execution.py` | `from frontier.core import run_program, SKLEARN_SPEC, TORCH_SPEC, LocalSubprocessExecutor, RemotePodExecutor, BackendSpec, SklearnBackend, TorchBackend, Executor` | `run_program(program, X_train, y_train, X_eval, kind=..., backend=SKLEARN_SPEC, executor=LocalSubprocessExecutor())`. With the defaults it is **bit-for-bit identical** to `frontier.sandbox.run_program` (verified). Swap `backend=TORCH_SPEC` + a `RemotePodExecutor` for the GPU path; it declines honestly (`[backend_unavailable]`) if torch is unreachable. Returns the Phase-0 `RunResult` (preds-only, no `score`). |
| `neural.py` | `from frontier.core import NeuralSpec, NASProposer, LLMArchitectProposer` | `NASProposer()` and `LLMArchitectProposer(llm_client=client)` are `ProposalSource`s used as proposal source **#3**. They emit `Program`s rendered from a typed `NeuralSpec`; a sklearn stand-in runs locally and the torch fit is dispatched to the pod, promotion certify-gated. Offline, `NASProposer` still mutates templates deterministically; `LLMArchitectProposer` degrades to `[]`. |
| `sandbox_policy.py` | `from frontier.core import SandboxPolicy, Tier, probe_host` | `SandboxPolicy(requested=Tier.LOCAL, untrusted=True, strict=...)` wraps a candidate run; `.resolve()` degrades downward to host-available tiers (`probe_host()`), `.run(program, ...)` dispatches and stamps **only the guarantees that held** onto the `RunResult`. Used as the policy-aware replacement for a raw sandbox dispatch when isolation tier matters. |
| `orchestrator.py` | `from frontier.core import CoreOrchestrator, CoreConfig, CoreResult` | `CoreOrchestrator(CoreConfig(rounds=..., llm_client=client_or_None)).run(goal, X, y, theta, name)`. This is the top-level CORE entry point; see below. |

### How `CoreOrchestrator` supersedes the stock `ResearchEngine.run`

The stock `ResearchEngine.run(task)` (`engine.py:91`) is a single-pass proposer→portfolio→certify
loop with no per-round enrichment: it does **not** add the `diagnosis` / `task_descriptor` /
`task_fingerprint` / agentic-probe context keys, so `knowledge.rank`, `enrich_context`, and the
repair probe are inert under it (see Risk 3 above). `CoreOrchestrator.run(goal, X, y, theta, name)`
is the Surface-B driver realized as a class. By composition (it imports and never edits
`engine.py`/`certify.py`/`sandbox.py`) it owns the outer control flow the stock engine lacks:

- it is the **sole holder of the `Splits`** and the **only** caller of `certify.certify_on_sealed`
  (one peek, in `_certify_winner`; asserts `sealed_peeks == 1`);
- per round it builds the enriched `context` (the sole enrich site), runs the four proposal sources
  (authoring, features, neural, seed/mutation floor), LinUCB-ranks, budget-admits with a floor
  guarantee, runs the ASHA portfolio, repairs failures, scores VAL in-parent, diagnoses, and records
  KB outcomes;
- it ANDs `oracles.verify_before_promote` into `certified` after the single peek;
- it returns a `CoreResult`, which is **EngineResult-compatible**: `result.as_engine_result()`
  projects onto the exact frozen `EngineResult` shape so `report.build_report` and any EngineResult
  consumer work unchanged. The extra fields (`oracle_verdict`, `sealed_peeks`, `backend_notes`,
  `self_consistency`) are non-promoting provenance and do not pollute the frozen schema.

So an integrator who currently calls `ResearchEngine(proposers=[...]).run(task)` swaps to
`CoreOrchestrator(CoreConfig(...)).run(goal, X, y, theta, name)` to get the richer loop, and calls
`.as_engine_result()` at the boundary if a downstream consumer expects the strict `EngineResult`.

### Which parts need an `llm_client` vs run offline

- **Need a live `llm_client`** (degrade to a documented fallback when `None`): `CoreAuthoringProposer`
  (offline → `[]`, `status="inactive"`), `LLMArchitectProposer` (offline → `[]`), and the
  orchestrator's `agentic.repair_loop` (offline → deterministic-then-give-up).
- **Run fully offline** (no client ever needed): `run_program` + all backends/executors, `NASProposer`
  (deterministic spec mutation), `SandboxPolicy`/`probe_host`, the frozen certify/sealed path, and the
  whole `CoreOrchestrator` loop with `CoreConfig(llm_client=None)`. This offline configuration is the
  default test path and is exercised for real on a sklearn dataset by `frontier/core/demo_core.py`.
  The result's `llm_active` flag states honestly which paths were live.

### CORE integration risk (most important)

**The one-sealed-peek invariant is enforced by the orchestrator, not the type system.** The
`CoreOrchestrator` is the only object holding `splits` and the only legitimate caller of
`certify.certify_on_sealed`. If an integrator reaches around it - calling `run_program`,
`SandboxPolicy.run`, or any proposer with `sealed_rows`, or invoking `certify_on_sealed` outside
`_certify_winner` - the firewall and the peek count are silently defeated. Keep every untrusted
execution on train/val rows only, route the single winner certification through the orchestrator, and
trust the `assert sealed_peeks == 1` / `certificate["peeks"] == 1` checks as the guardrail.
