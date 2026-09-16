# Design 03: Neural Architecture Core

Status: implementation-ready. Additive only. No edits to Phase-0 files
(`program.py`, `task.py`, `certify.py`, `sandbox.py`, `proposers.py`, `engine.py`,
`__init__.py`, `demo.py`, `tests/test_spine.py`) or to `vectorforge.science` /
`vfplatform.sealed`. Everything below is new code under `frontier/core/neural/`,
wired by composition.

## 0. Problem statement and what the audit found

The audit verdict on the legacy `attestra/execution/architecture_builder.py`: it is
"mostly template assembly and never runs." Two failures, both addressed here:

1. Not novel. It picks from a fixed bank of hand-written `nn.Module` templates and
   fills slots. The LLM never authors a genuinely new architecture; the search never
   composes structure it was not pre-given.
2. Never certified. The templates are not built, trained, run on a held-out sealed
   test, and passed through the frozen one-peek certifier. Numbers, where they exist,
   are self-reported.

The Phase-0 spine already solves the certification half for sklearn pipelines: a
`Program` is arbitrary code defining `build_estimator()`, it runs out-of-process in
`sandbox.run_program`, the parent scores on val via `certify.score_val`, and the single
winner is certified once on the sealed test via `certify.certify_on_sealed`. Anything
exposing `fit`/`predict` flows through that path unchanged.

So the neural core's job is precisely: make a trained neural network look like a
`build_estimator()` so it rides the existing Task -> sandbox -> certify spine with zero
spine edits, while making the architecture itself the thing the LLM (or a NAS search)
authors and the certifier promotes.

The hard constraint: local Python has no torch. torch/GPU exist only on Prime Intellect
pods (see design 02, execution substrate). Therefore the entire architecture
representation, code generator, validation, and the certify loop must be demonstrable
locally with an sklearn `MLPClassifier`/`MLPRegressor` stand-in, and the torch path must
be import-gated so it activates only where `import torch` succeeds and is tested on the
pod.

## 1. Architecture representation: `NeuralSpec`

A typed, JSON-serializable, framework-agnostic description of a feed-forward / sequence /
conv network. It is the unit the LLM emits, the NAS samples, diagnosis mutates, and the
knowledge base stores. It never references torch.

New file `frontier/core/neural/spec.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Tuple
import hashlib, json

Activation = Literal["relu", "gelu", "silu", "tanh", "identity"]
Norm       = Literal["none", "batch", "layer"]
BlockKind  = Literal["mlp", "residual_mlp", "conv1d", "attention", "embedding_pool"]

@dataclass
class Block:
    kind: BlockKind
    width: int                       # out features / channels / model dim
    activation: Activation = "relu"
    norm: Norm = "none"
    dropout: float = 0.0             # [0, 0.9]
    residual: bool = False           # add skip connection around this block
    # conv1d only:
    kernel_size: int = 3
    stride: int = 1
    # attention only:
    n_heads: int = 4
    # embedding_pool only (categorical / token inputs):
    vocab_size: int = 0
    pool: Literal["mean", "max", "cls"] = "mean"

@dataclass
class NeuralSpec:
    modality: Literal["tabular", "sequence", "image1d"]
    task_kind: Literal["classification", "regression"]
    in_features: int                 # d for tabular; seq_len for sequence; channels for conv
    out_dim: int                     # n_classes (clf) or 1 (reg)
    blocks: List[Block]
    # training knobs the spec carries (consumed by the runner, design 02):
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20               # early-stop patience on val
    seed: int = 0
    label_smoothing: float = 0.0
    provenance: dict = field(default_factory=dict)  # {"author":"llm"|"nas", "parent": id, ...}

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]
```

### 1.1 Validity (the local, torch-free guarantee)

`spec.py` ships `validate(spec) -> list[str]` returning a list of violations (empty =
valid). It enforces shape-coherence and budget rules that can be checked without building
anything:

- `in_features > 0`, `out_dim >= 1`, `out_dim == 1` iff regression.
- every `width` in `[1, 4096]`; `0 <= dropout < 0.9`; `0 < lr < 1`; `1 <= n_heads`,
  and for attention blocks `width % n_heads == 0` (head-dim integrality).
- conv1d: `kernel_size >= 1`, `stride >= 1`, and the running sequence length after
  the conv stack stays `>= 1` (computed analytically from `in_features`, kernels,
  strides; this is the single most common silent torch crash and is caught here).
- attention/embedding_pool blocks are only legal for `modality in {sequence}`.
- residual blocks require the surrounding width to be constant or the generator must
  insert a linear projection on the skip (the generator handles this; validate only
  flags an impossible request like `residual=True` on the final block to `out_dim`).
- a parameter-count estimate `estimate_params(spec)` (closed form per block) so the
  complexity budget in section 7 is computable before any build.

These checks are pure arithmetic. They are the backbone of local unit tests
(section 5) and they let NAS reject infeasible samples for free.

## 2. Rendering to a `build_estimator()`-compatible wrapper

The spec renders to a Program code string whose `build_estimator()` returns an object
exposing `fit(X, y)` / `predict(X)`. Because the spine only ever calls those two methods
through the sandbox, the wrapper is the entire integration surface. New file
`frontier/core/neural/render.py` exposes `render_program(spec) -> Program`.

The generated code has exactly two branches selected at runtime inside the sandbox child,
so the SAME Program string is portable between the laptop and the pod:

```python
def build_estimator():
    spec = NeuralSpec(**_SPEC_DICT)        # _SPEC_DICT inlined as a literal
    try:
        import torch  # noqa
        return TorchNetEstimator(spec)     # real net, design-02 TorchBackend runner
    except Exception:
        return SklearnMLPStandin(spec)     # torch absent -> exact-interface stand-in
```

Key properties:

- The spec is inlined as a literal dict (no import of `frontier` inside the sandbox
  child, which keeps the candidate self-contained, matching how `proposers.make_code`
  emits self-contained sklearn code today).
- `TorchNetEstimator` and `SklearnMLPStandin` are both emitted into the same code string
  (small, ~120 lines) so there is no dependency on the frontier package being importable
  in the child process. This mirrors the Phase-0 contract that candidate code is a
  standalone module.
- Both classes implement `fit(self, X, y)` and `predict(self, X)` and nothing else that
  the spine touches. The numeric firewall holds: neither returns a metric; the child
  writes predictions, the parent scores.

### 2.1 `SklearnMLPStandin` (local, no torch)

A thin adapter mapping the MLP-relevant parts of a `NeuralSpec` onto
`sklearn.neural_network.MLPClassifier` / `MLPRegressor`:

- `hidden_layer_sizes` = tuple of `block.width` for `kind in {mlp, residual_mlp}`
  (conv/attention blocks are dropped with a recorded note in `provenance`, since sklearn
  cannot express them; the stand-in is a fidelity-reduced surrogate, not a claim of
  equivalence, and section 8 calls this out as a risk).
- `activation`: map `relu->relu`, `gelu/silu->relu` (closest sklearn support), `tanh->tanh`,
  `identity->identity`.
- `alpha` = `weight_decay`; `learning_rate_init` = `lr`; `max_iter` = `max_epochs`;
  `early_stopping=True`, `n_iter_no_change=patience`, `validation_fraction` matched to the
  task's val protocol intent; `batch_size`; `random_state=seed`.
- classification target dtype handling matches the sandbox runner (`y.astype(str)`).

This makes the end-to-end certify loop runnable and demonstrable today: a `NeuralSpec`
becomes a Program, runs in the real subprocess sandbox, is scored on val, and the winner
is certified on the sealed test through the unmodified frozen gate.

### 2.2 `TorchNetEstimator` (import-gated, runs on the pod)

A real `nn.Module` builder plus a self-contained training loop (the protocol in section
3). It is only constructed if `import torch` succeeds, so the laptop never executes it.
Where torch is present, the same Program string builds and trains the genuine network.
The torch module assembly is the novel part: blocks compose into an `nn.Sequential`
spine with explicit residual wiring and (for sequence modality) a final pooling head, all
driven by the spec rather than a fixed template.

## 3. Training protocol (lives in design-02 TorchBackend; specified here)

The training loop is owned by the execution substrate (design 02), but its contract is
fixed here so the spec's training knobs have defined semantics. It runs inside the
sandbox child on the pod and emits predictions only.

1. Data: `X_train, y_train, X_eval` arrive as the sandbox already passes them (float
   arrays for X, object array for y). The estimator converts to tensors:
   `torch.as_tensor(X, dtype=float32)`; for classification, fit a label index map on the
   sorted unique string labels (stable, matches `Task.labels`) and store it so `predict`
   maps argmax back to the original string labels.
2. Internal val carve-out: the runner holds out a fraction of `X_train` (stratified for
   clf, quantile-binned for reg, mirroring `certify._regression_split`) as an INNER
   validation set for early stopping. This inner split is disjoint from the spine's val
   and sealed splits by construction (it only ever sees training rows). Early stopping
   keys on inner-val metric, never on the spine's val and never on sealed.
3. Optimizer/scheduler: `adamw`/`adam`/`sgd` per spec; cosine schedule with linear warmup
   (10% of `max_epochs`); gradient clipping at norm 1.0.
4. Mixed precision: `torch.autocast` + `GradScaler` only when `torch.cuda.is_available()`;
   on CPU the path runs fp32. Gated, never assumed.
5. Determinism: `torch.manual_seed(spec.seed)`, `cudnn.deterministic=True`,
   `cudnn.benchmark=False`, fixed DataLoader worker seeding. Same seed -> same weights ->
   same predictions, which is what Phase-7 reproducibility oracles will re-check.
6. Early stopping: patience on inner-val; restore best weights before predicting.
7. NaN/Inf guard: matching the spine's discipline (sandbox.py comments),
   any NaN/Inf in loss or logits aborts the epoch loop, restores last finite weights, and
   if no finite weights exist the run fails with `error_kind="fit"` so the parent records
   it rather than silently swallowing it.
8. Predictions out: `predict(X_eval)` returns class-label strings (clf) or floats (reg).
   The child writes them; the parent computes every number.

Firewall restated: the runner never computes accuracy/r2/loss-on-val-as-a-score that
leaves the process. It returns predictions. `certify.score_val` (val) and
`certify.certify_on_sealed` (sealed, one peek) compute all promotion-bearing numbers in
the trusted parent.

## 4. The search / authoring loop as a `ProposalSource`

Two new proposers, both emitting `Program` objects so they drop into the engine's
`proposers` list with no engine edit. Both honor the `propose(context) -> list[Program]`
protocol and the `tried_labels` dedup contract already in `engine.py`.

New file `frontier/core/neural/proposers.py`:

### 4.1 `NASProposer` (source="nas", no LLM required)

Samples `NeuralSpec`s from a bounded search space, validates each with `spec.validate`,
discards infeasible/over-budget ones, renders survivors to Programs. Deterministic given
`context["round"]` + `seed` (reproducible search). This makes the neural core generative
offline, exactly as `MutationProposer` makes the sklearn path generative offline.

Diagnosis-conditioned mutation (innovation 1) drives the next sample:

```python
def propose(self, context):
    diag = context.get("diagnosis", {})     # supplied by frontier/diagnosis.enrich_context
    parent = context.get("best_spec")        # the champion spec, if the champion was neural
    if parent is None:
        return [render_program(s) for s in self._sample_seeds(context)]
    muts = []
    if diag.get("plateau"):                  # structural pivot
        muts += [_add_block(parent), _switch_block_kind(parent)]
    if diag.get("underfit"):                 # train and val both weak -> grow capacity
        muts += [_widen(parent), _deepen(parent), _reduce_dropout(parent)]
    if diag.get("overfit"):                  # train >> val -> regularize / shrink
        muts += [_add_dropout(parent), _add_norm(parent), _narrow(parent),
                 _raise_weight_decay(parent)]
    feasible = [m for m in muts if not validate(m)]
    feasible = [m for m in feasible if estimate_params(m) <= context["param_budget"]]
    return [render_program(m) for m in feasible]
```

Underfit/overfit/plateau signals: `frontier/diagnosis.py` already computes `plateau`,
per-family error rates, axis-lift, and (for regression) residual structure, and
`enrich_context` injects them into the context dict. The neural core adds two signals to
that same dict via a tiny helper `neural_fit_signal(history)` that compares a candidate's
inner-train vs spine-val score gap when both are available (train score is surfaced by an
optional extra prediction on a train subsample, see 4.3). Underfit = low val and low
train; overfit = high train, low val; plateau = the existing detector. No new promotion
number is introduced; these are search-steering signals only.

### 4.2 `LLMArchitectProposer` (source="llm")

Reuses the pluggable `client: Callable[[str], str]` contract from `LLMProposer`. The LLM
authors a `NeuralSpec` as JSON (constrained, parseable, validatable) rather than free-form
torch code, then the trusted generator renders it. This is the key safety improvement
over the legacy builder: the LLM's creativity lives in the spec it designs (block kinds,
widths, depths, residual/attention/norm choices, training knobs), while the torch code
that actually runs is emitted by audited `render.py`, not by the model. Three guards:

- Parse: extract the first JSON object; reject on parse failure (recorded as an error,
  fed back into `recent_errors`).
- Validate: `spec.validate` must pass; violations are returned to the LLM next round as
  diagnosis text (the loop is agentic per Phase 4).
- Optionally, an LLM may instead author a full `nn.Module` as code (source="llm",
  free-form). That path is supported but the rendered code STILL runs in the same sandbox
  and STILL must define `build_estimator()`; it has no certification privilege. The spec
  path is preferred and is the default the prompt requests.

Prompt carries: modality, in_features, out_dim, n_train, current champion spec, the
diagnosis (underfit/overfit/plateau + residual notes), the param budget, and the recent
typed errors, mirroring `LLMProposer._prompt`.

### 4.3 Train-score surfacing (for the fit signal)

To distinguish underfit from overfit the parent needs a train-side score without breaking
the firewall. The estimator's `predict` is called by the parent on `X_eval`; to also get a
train estimate, the engine integration (section 6) optionally runs the winning/again-run
candidate on a held-out slice of TRAIN (never val, never sealed) as a second predict, and
the parent scores it. This stays inside the firewall (predictions only) and never touches
val or sealed. It is a search signal, not a certificate input.

## 5. Local verifiability without torch

A dedicated test module `frontier/tests/test_neural_core.py`, runnable today with
`/Users/abdullahalghamdi/jax-env-311/bin/python -m pytest`:

1. Spec validity unit tests: hand-built valid specs pass `validate`; deliberately broken
   specs (conv stack shrinking seq to 0, attention with `width % n_heads != 0`, regression
   with `out_dim=3`, attention on tabular modality) each return the expected violation.
2. Code-generation tests: `render_program(spec).code` (a) `compile()`s without error for a
   battery of specs across all three modalities; (b) defines a callable `build_estimator`
   (mirrors the sandbox runner's own `build_estimator` check); (c) is self-contained
   (no `import frontier`); (d) the param-count estimate matches the realized sklearn MLP's
   coefficient count for the MLP-only specs (sanity on the budget math).
3. sklearn stand-in end-to-end through the REAL spine: build a `NeuralSpec` for
   breast-cancer, render to a Program, run it through `sandbox.run_program` + `score_val`
   + `certify_on_sealed` exactly as `engine.py` does, and assert a certificate dict comes
   back with `certified in {True, False}` and `peeks == 1`. This demonstrates the full
   certify loop for a neural-authored candidate today, no torch.
4. Proposer protocol tests: `NASProposer.propose(context)` returns valid, deduped,
   in-budget Programs; diagnosis-conditioned branches fire on synthetic underfit /
   overfit / plateau contexts (assert widen on underfit, dropout/norm on overfit,
   block-add on plateau).
5. Torch path import-gated and tested where available: every torch test is decorated
   `@pytest.mark.skipif(not _has_torch(), reason="torch only on pod")`. On the pod CI lane
   (design 02) these run: a tiny `NeuralSpec` trains for a few epochs on a synthetic
   separable set, predictions have the right shape and dtype, deterministic re-run with the
   same seed yields identical predictions, and the same Program certifies through the gate.
   The skip is reported, not hidden, so a green local run never masquerades as torch-tested.

A `make_neural_demo()` (added to `frontier/core/neural/__init__.py`, not the frozen
`demo.py`) runs the stand-in end-to-end and prints the certificate, so the loop is
demonstrable now.

## 6. Engine integration by composition (no Phase-0 edit)

The engine accepts `proposers: list[ProposalSource]` in its constructor. Integration is a
call-site choice, exactly like the harness fabric's wiring block:

```python
from frontier.engine import ResearchEngine, EngineConfig
from frontier.core.neural.proposers import NASProposer, LLMArchitectProposer
from frontier.proposers import SeedProposer, MutationProposer, LLMProposer

proposers = [SeedProposer(), MutationProposer(), LLMProposer(client),    # tabular floor
             NASProposer(modality="tabular", param_budget=2_000_000),    # neural search
             LLMArchitectProposer(client)]                               # neural authoring
result = ResearchEngine(EngineConfig(rounds=4, llm_client=client),
                        proposers=proposers).run(task)
```

Two small additions needed in the context dict that the engine builds, supplied WITHOUT
editing `engine.py` by composing a wrapper proposer that enriches context, or by letting
`NASProposer` derive them itself:

- `best_spec`: if the champion Program's `provenance` carries `{"neural_spec": {...}}`
  (the neural proposers set this), `NASProposer` reconstructs the parent spec from it. No
  engine change: provenance already flows because the winner Program is the one the
  proposers emitted.
- `param_budget` and `diagnosis`: `param_budget` is a constructor arg on `NASProposer`;
  `diagnosis` is injected by the existing `frontier.diagnosis.enrich_context` path
  (Phase 2) which the integrator already wires per `diagnosis.py`'s WIRING block. If that
  path is not wired, `NASProposer` degrades gracefully (treats diagnosis as empty and does
  pure-sampling NAS).

The neural core therefore plugs in as additional proposers plus a renderer; it touches no
frozen file. The execution substrate (design 02) supplies the TorchBackend runner that
`TorchNetEstimator` calls; on the laptop the stand-in path is used and design-02 is not
required for the local demo.

## 7. Generalization across modalities

The harness fabric (`frontier/harness/base.py`) already routes a task-type key to a
Harness that adapts raw data into a `Task` and self-certifies. The neural core slots in
behind that fabric: each modality harness declares which `NeuralSpec.modality` it feeds
and what input adapter the spec needs.

- Tabular (`TabularHarness`, exists): modality `"tabular"`. Spec stack is mlp /
  residual_mlp. A TabTransformer variant is expressible as `embedding_pool` (for
  categorical columns) + `attention` blocks once a future `TabularCategoricalHarness`
  marks categorical column indices in the Task; until then tabular specs are MLP/residual.
- Sequence/text: a `SequenceHarness` (sibling of the existing `text.py`) sets modality
  `"sequence"`, supplies `vocab_size`/`seq_len`, and the spec uses `embedding_pool` +
  `attention`/`conv1d` blocks. The harness's `adapt` is responsible for turning raw text
  into the integer/feature arrays the Task carries; the spec sees only shapes.
- Conv/image-as-1d: modality `"image1d"` with conv1d blocks for signals/flattened images;
  a true 2D conv modality is a later spec extension (`conv2d` block kind) gated behind a
  `VisionHarness`.

The contract: the harness supplies the right input adapter and the correct
`in_features`/`out_dim`, and chooses the metric/theta/split (it already does). The neural
core supplies the body. A harness that wants neural candidates passes a
`NASProposer(modality=..., param_budget=...)` in the proposer list it hands the engine
(harness `baseline_suite` stays sklearn seeds as the floor; neural specs are search, not
floor). The harness self-test gate still applies: a harness's numbers are trusted only
after it certifies itself, and a neural candidate is only trusted after the frozen sealed
certifier promotes it.

## 8. Innovations (load-bearing, not decoration)

1. Diagnosis-conditioned architecture mutation. The next architecture is a function of the
   measured underfit/overfit/plateau signal, not a blind sample: underfit grows capacity
   (widen/deepen/less dropout), overfit regularizes (dropout/norm/weight-decay/narrow),
   plateau pivots structure (new block kind, add attention/residual). This makes the search
   directed and sample-efficient, and it reuses the existing `diagnosis.py` signals so the
   neural and sklearn paths share one diagnosis fabric.
2. Complexity-budgeted NAS with early-kill. Every sampled/authored spec must pass
   `estimate_params(spec) <= param_budget` BEFORE it is built (free rejection of bloated
   nets). At run time, the design-02 runner enforces a wall/CPU budget (it already does)
   and an epoch-level early-kill: if inner-val has not improved past a floor by a small
   fraction of `max_epochs`, the run self-terminates and returns its current predictions,
   so a hopeless architecture is killed cheap. Budget is a search-bounding number that is
   literature/first-principles set (a hardware-derived param/FLOP ceiling), never reverse-
   engineered from a target metric, honoring the CONTRACT invariant 3.
3. Certify-gated architecture promotion. An architecture is kept (added to the Phase-8
   knowledge base, surfaced as the champion `best_spec`) ONLY if it was promoted by the
   frozen sealed certifier, or it is retained merely as a val-ranked search node that
   carries no promotion claim. The sealed certificate is the sole gate; a net that wins on
   val but fails the sealed lower bound is recorded as an honest decline, never relabeled.
   This is the exact discipline the spine already enforces, extended to nets: the audit's
   "never runs / never certified" failure cannot recur because the only way a net is kept
   is by surviving the same one-peek gate as every sklearn pipeline.

## 9. Risks and mitigations

- Overfitting the val set via search. Running many architectures and selecting the val-max
  inflates val optimism. Mitigation: the promotion number is the sealed lower bound (one
  peek, Clopper-Pearson / bootstrap), not val; val only ranks. The Phase-7 multiple-arm
  correction and the sealed bound are the defense, and the honest-decline path absorbs the
  case where no net clears theta. Cap the number of distinct nets evaluated per task
  (search-arm budget) to bound selection optimism.
- Compute cost. Training nets is far costlier than sklearn fits. Mitigations: param budget
  + epoch early-kill (innovation 2); the wall/CPU rlimits the sandbox already imposes; NAS
  samples are validated and budget-filtered before any build so no compute is spent on
  infeasible specs; the sklearn stand-in carries all local development so pods are spent
  only on real torch runs.
- Nondeterminism. GPU kernels and DataLoader shuffling can break reproducibility, which
  would make a certificate irreproducible. Mitigation: full deterministic seeding
  (`manual_seed`, `cudnn.deterministic`, seeded workers); a Phase-7 reproducibility oracle
  re-runs the winner with the same seed and asserts identical predictions before promotion;
  any residual nondeterminism is recorded, not hidden.
- Reward hacking. Generated code (especially a free-form `nn.Module` LLM path) could try to
  read the eval labels, the sealed rows, or otherwise game the metric. Mitigations: the
  firewall (the child gets `X_eval` features only, never eval labels, never sealed rows
  during search; sealed is touched once, by the parent, for the winner); the out-of-process
  sandbox with no-network substrate (design 02); the preferred LLM path authors a validated
  SPEC that the audited `render.py` turns into code, so the running code is generator-
  controlled, not model-controlled; Phase-7 sanity oracles (permuted-labels-collapse-to-
  chance, beat-a-trivial-baseline) catch a net that "succeeds" by leakage. The stand-in's
  reduced fidelity (it drops conv/attention) is itself a risk: a spec that certifies via
  the stand-in is NOT a certificate for the torch net; the torch net must be re-certified on
  the pod through the same gate, and `provenance` records which backend produced the
  certified predictions so the two are never conflated.
- Stand-in / torch divergence. The sklearn stand-in is a development surrogate; its
  certificate licenses only the stand-in. Mitigation: `provenance["backend"]` is recorded
  on every certificate; promotion of a torch architecture requires a torch-backend sealed
  certificate, and the local stand-in result is labeled as plumbing-validation only.

## 10. File manifest (all additive)

- `frontier/core/neural/spec.py`: `NeuralSpec`, `Block`, `validate`, `estimate_params`.
- `frontier/core/neural/render.py`: `render_program(spec) -> Program`; emits
  `TorchNetEstimator` (import-gated) and `SklearnMLPStandin` (local) into one self-
  contained code string.
- `frontier/core/neural/proposers.py`: `NASProposer`, `LLMArchitectProposer`,
  `neural_fit_signal`.
- `frontier/core/neural/__init__.py`: exports + `make_neural_demo()`.
- `frontier/tests/test_neural_core.py`: local spec/codegen/stand-in/proposer tests +
  import-gated torch tests.
- Consumes (does not edit): design-02 TorchBackend runner for the training protocol of
  section 3.

Acceptance for this design: (a) `test_neural_core.py` passes locally with no torch,
including a stand-in net certifying through the frozen gate; (b) on a torch-capable pod the
gated tests pass and a NAS/LLM-authored net certifies through the same sealed gate; (c) no
frozen Phase-0 file or `science`/`sealed` file is modified.
