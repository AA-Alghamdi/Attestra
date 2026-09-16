# 01 - The Model-Authoring Engine (core of the autoresearcher)

Status: implementation-ready design. Target builder: an agent that writes
`frontier/core/authoring.py` plus tests, importing only the FROZEN Phase-0 contract.

## 0. Mission and non-negotiables

Kill "model selection from a list." The engine lets an LLM author **arbitrary**
`build_estimator()` code (any sklearn-compatible estimator now, torch later), conditioned on
a structured diagnosis plus retrieved literature, and feeds those Programs into the existing
`ResearchEngine` as just another `ProposalSource`. The deterministic recipe library
(`SeedProposer` + `MutationProposer`) is demoted to a FLOOR/FALLBACK: it guarantees a result
when no client is wired and a non-empty proposal pool every round, but it never out-ranks a
better-validating authored program and it is never the promotion-bearing decision.

Frozen-contract invariants this module MUST preserve (from `frontier/CONTRACT.md`, restated
because they are load-bearing here):

1. Only the frozen certifier promotes. Sealed test touched once, for the winner. This module
   never calls `certify_on_sealed` and never reads `splits.sealed_rows`.
2. Untrusted code returns predictions only; every decision number comes from `science.py` via
   `certify.score_val`. This module never lets authored code report a metric.
3. Hardcoded heuristics are seeds/fallbacks, never the promotion-bearing or search-bounding
   decision.
4. Generalization expands what may be PROPOSED, never what may PROMOTE.
5. Outcomes are honest: a certified result or an honest decline, never a relabeled val score.
   Corollary enforced here: no LLM-claimed number ever influences selection. The LLM emits
   code; the parent runs it in `frontier.sandbox` and scores it on VAL.

**Do not edit any Phase-0 file** (`program.py`, `proposers.py`, `engine.py`, `sandbox.py`,
`certify.py`, `task.py`, `__init__.py`). This module is additive. It composes by being passed
to `ResearchEngine(proposers=[...])`.

Interpreter for all verification: `/Users/abdullahalghamdi/jax-env-311/bin/python`.

---

## 1. AuthoringEngine and CoreAuthoringProposer

### 1.1 File layout

```
frontier/core/
  authoring.py        # AuthoringEngine, CoreAuthoringProposer, prompt builders
  firewall.py         # AST validation, preamble, name correction, leakage strip
  schema.py           # ProgramSpec typed schema + render to prompt directives
  tests/
    test_firewall.py
    test_authoring_degrade.py   # client=None -> [] and engine falls to floor
    test_authoring_live.py      # FakeClient returns canned code -> certifies
```

### 1.2 Pluggable client type

```python
# A client is any callable: prompt(str) -> completion(str). It is the ONLY external
# dependence. Wire vfplatform.resolve_backend / a Prime Intellect client behind this type.
# None => honest degrade: propose() returns [] and the engine runs on the recipe floor.
LLMClient = Callable[[str], str]
```

A richer client (one that also returns token usage / finish_reason) is wrapped to this
signature by the caller; the engine only needs prompt->text. A multi-shot client (used by the
revise loop, section 4) is the optional `ChatLLMClient` protocol:

```python
class ChatLLMClient(Protocol):
    def complete(self, messages: list[dict]) -> str: ...   # messages: {"role","content"}
```

`AuthoringEngine` accepts either; a bare `Callable` is adapted by wrapping
`complete(messages)` as `client("\n\n".join(m["content"] for m in messages))`.

### 1.3 AuthoringEngine

The engine is the stateless-per-call author: given a round context it produces validated,
firewalled Program objects with `source="llm"`. It owns the prompt, the diversity strategy,
parsing, and the firewall handoff. It does NOT run code (that is the sandbox, invoked by
`ResearchEngine`) and does NOT score (that is `certify.score_val`).

```python
@dataclass
class AuthoringConfig:
    n: int = 3                     # candidates authored per round (the "N" in self-consistency)
    temperatures: tuple = (0.2, 0.6, 0.9)   # one per candidate; recycled if n > len
    max_completion_chars: int = 12000
    inject_literature: bool = True
    inject_diagnosis: bool = True
    enable_revise: bool = True     # error-feedback loop (section 4)
    revise_max_attempts: int = 1   # per failed candidate, before handing to agentic repair
    schema_mode: str = "typed"     # "typed" (ProgramSpec directives) | "freeform"

class AuthoringEngine:
    def __init__(self,
                 client: LLMClient | ChatLLMClient | None = None,
                 config: AuthoringConfig | None = None,
                 retriever: "LiteratureRetriever | None" = None,   # knowledge.py, separate module
                 firewall: "Firewall | None" = None):
        self.client = client
        self.cfg = config or AuthoringConfig()
        self.retriever = retriever          # may be None; literature injection then skipped
        self.fw = firewall or Firewall()    # default allow-list firewall (section 3)

    # --- the one method a proposer needs ---
    def author(self, context: dict) -> list[Program]:
        """Author up to cfg.n firewalled Programs for this round. [] if no client."""
        if self.client is None:
            return []
        spec = self._build_spec(context)                 # ProgramSpec (section 7.1)
        motifs = self._retrieve(context) if self.cfg.inject_literature else []
        programs, seen = [], set()
        for i in range(self.cfg.n):
            temp = self.cfg.temperatures[i % len(self.cfg.temperatures)]
            prompt = self._build_prompt(context, spec, motifs, temp_hint=temp, variant=i)
            raw = self._call(prompt, temperature=temp)
            if raw is None:
                continue
            prog = self._materialize(raw, context, spec, motifs, prompt)
            if prog is None or prog.id in seen:
                continue
            seen.add(prog.id)
            programs.append(prog)
        return programs

    def revise(self, failed: Program, run_result, context: dict) -> Program | None:
        """Error-feedback hook (section 4). Re-author given a traceback. None if disabled / no fix."""
        ...

    def combine(self, top_programs: list[Program], context: dict) -> list[Program]:
        """Regeneration: merge top-N programs into a stronger ensemble (section 5)."""
        ...
```

`_materialize` is the firewall pipeline:

```python
def _materialize(self, raw, context, spec, motifs, prompt) -> Program | None:
    code = strip_fences(raw)                          # remove ``` and prose preamble
    code = self.fw.autocorrect_names(code)            # hallucinated symbol -> nearest valid
    code = self.fw.strip_forbidden(code)              # remove network/IO/leakage constructs
    ok, reason = self.fw.validate(code, kind=context["task_kind"])
    if not ok:
        self._reject(reason, code); return None       # logged, never silently dropped
    code = self.fw.add_preamble(code)                 # guarantee symbols are importable
    label = self._label(context, spec)
    return Program(
        code=code, source="llm", label=label,
        provenance={
            "prompt_sha": sha12(prompt),
            "spec": spec.to_dict(),
            "motif_ids": [m.id for m in motifs],
            "temperature": context.get("_temp"),
            "authored_round": context.get("round"),
            "corrections": self.fw.last_corrections,   # audit trail of name fixes / strips
        })
```

### 1.4 CoreAuthoringProposer (the ProposalSource)

Implements the `ProposalSource` Protocol from `proposers.py` (`propose(context) -> list[Program]`).
It is a thin adapter so the engine treats authoring identically to seeds/mutations.

```python
class CoreAuthoringProposer:
    """ProposalSource that authors arbitrary build_estimator() code via an LLM.

    Drop-in for the Phase-0 LLMProposer: same Protocol, richer behavior. When client is None,
    propose() returns [] and the engine reports llm_active=False -- honest degrade to the floor.
    """
    def __init__(self, client=None, config: AuthoringConfig | None = None,
                 retriever=None, firewall=None):
        self.engine = AuthoringEngine(client, config, retriever, firewall)
        self.client = client          # surfaced so engine.llm_active detection still works (*)

    def propose(self, context: dict) -> list[Program]:
        return self.engine.author(context)
```

(*) `engine.py:103` detects an active LLM by `isinstance(p, LLMProposer) and p.client is not None`.
`CoreAuthoringProposer` is NOT an `LLMProposer`, so `llm_active` would read False even when a
client is wired. Two contract-safe options (do NOT edit engine.py):

- **Preferred:** subclass - `class CoreAuthoringProposer(LLMProposer):` and call
  `super().__init__(client=client, n=config.n)` so the existing isinstance + `.client` check
  fires correctly, then override `propose`/`_prompt`. This keeps `llm_active` honest with zero
  Phase-0 edits.
- **Alternative:** expose `.client` (done above) and have the caller set
  `EngineResult.llm_active` by inspecting proposers; brittle, not recommended.

The subclass route is the design choice: `CoreAuthoringProposer(LLMProposer)`. It inherits the
honest-degrade contract (client None -> []) and the `.client` attribute the engine reads.

---

## 2. Prompt architecture

### 2.1 System contract for build_estimator()

A frozen system block, sent verbatim every call. It is the API the model writes against.

```
You author one scikit-learn pipeline as Python code. Output rules:
- Define exactly one function: def build_estimator(): returning a fresh, UNFITTED
  sklearn-compatible estimator (any object with .fit(X, y) and .predict(X)).
- The estimator is fit on a training split and asked to predict an evaluation split by the
  caller. You do NOT fit, score, print, read files, or access the network. You do NOT see y
  beyond what fit() receives.
- You MAY compose: ColumnTransformer, Pipeline, FunctionTransformer, StandardScaler,
  PolynomialFeatures, PCA, SelectKBest, QuantileTransformer, PowerTransformer, target
  transforms via TransformedTargetRegressor, and stacking via StackingClassifier/Regressor.
- Allowed imports: numpy as np, and sklearn.* only (preprocessing, pipeline, compose,
  feature_selection, decomposition, ensemble, linear_model, svm, neighbors, tree,
  naive_bayes, kernel_ridge, gaussian_process, neural_network). No other top-level packages.
- FORBIDDEN: any GridSearchCV/RandomizedSearchCV/cross_val_* that touches the eval split;
  any fit on data other than what build_estimator() returns to the caller; any import of os,
  sys, subprocess, socket, requests, urllib, open(), pickle load of external files, eval/exec.
- For regression you MAY wrap the pipeline in TransformedTargetRegressor for target transforms.
- Set random_state=0 wherever an estimator accepts it. Set n_jobs=1.
- Return ONLY the code. No markdown fences, no commentary.
```

`n_jobs=1` is mandatory because the sandbox sets `RLIMIT_CPU`; parallel backends fork and the
limit applies per-process unpredictably. The firewall (section 3) rewrites `n_jobs=-1` to `1`.

### 2.2 Diagnosis injection

The Phase-0 `context` dict already carries: `task_kind, n_features, n_train, round,
tried_labels, best_label, best_score, best_id, best_recipe, recent_errors`. Phase 2 deepens
diagnosis (per-family error rates, plateau detection, residual structure). The authoring
prompt consumes whatever keys are present and degrades gracefully on absence.

Rendered diagnosis block:

```
TASK: {task_kind}, {n_features} features, {n_train} train rows. Metric: {metric}.
CHAMPION: {best_label} validating at {best_score}.   # omit line if best_score is None
PER-FAMILY (val score, n tried):                      # context["family_ranking"], Phase 2
  hist_gbm: 0.91 (x3)   logreg+scale: 0.88 (x2)   svc_rbf: 0.74 (x1)
PLATEAU: best score has not improved in {context.get("plateau_rounds")} rounds.  # if present
RECENT FAILURES (do not repeat these error modes):
  - poly3+ridge: [fit] LinAlgError singular matrix (degree too high for n_features)
  - llm2_0: [import] No module named 'xgboost' (only sklearn is available)
RESIDUAL HINT: errors concentrate in the upper target quartile.   # context.get("residual_note")
```

Diagnosis -> directive mapping (the engine turns diagnosis into explicit asks, so the model
does not have to infer strategy from raw numbers):

| Diagnosis signal | Injected directive |
|---|---|
| `plateau_rounds >= 2` | "Linear improvements have stalled. Try a structurally different model class or a target transform, not a hyperparameter nudge." |
| best family is linear, kind=regression | "A linear model leads. Probe nonlinearity: gradient boosting, or polynomial / spline features into the linear model." |
| `recent_errors` has `[import]` | "Only numpy and sklearn are importable. Do not import {offending modules}." |
| `recent_errors` has `[fit] LinAlgError` | "A prior pipeline was numerically unstable (singular). Add regularization or scaling; lower polynomial degree." |
| `recent_errors` has `[timeout]` | "A prior pipeline was too slow. Avoid SVC on large n; prefer histogram GBM or linear models; cap n_estimators." |
| `residual_note` present | "Errors concentrate in {region}. Consider a target transform or a model robust to heteroscedasticity." |
| `n_features` large, `n_train` small | "High-dimensional, few rows. Add SelectKBest or PCA, prefer regularized linear / GBM, avoid degree>2 polynomials." |

These directives are heuristics that *shape the prompt*; they are labeled as such in code
comments. They never bound the search (the model may ignore them) and never promote anything.
They are exactly the kind of decision the LLM should eventually make unaided; they are
scaffolding, removable as base models improve, consistent with "powered by LLMs."

### 2.3 Literature motif injection

`retriever.retrieve(context) -> list[Motif]` (the knowledge.py module, separate; treat as a
dependency with this contract):

```python
@dataclass
class Motif:
    id: str                 # e.g. "arxiv:2106.11959" or "skl:hist_gbm_tabular"
    title: str
    claim: str              # one-line takeaway, e.g. "GBDTs still beat MLPs on tabular"
    skeleton: str | None    # OPTIONAL retrieval-grounded code skeleton (section 7.3)
    score: float            # retrieval relevance
```

Injected as grounding, never as code to copy blindly:

```
RELEVANT FINDINGS (use as evidence, adapt; cite the id you used in a leading comment):
  [arxiv:2106.11959] GBDTs remain strong baselines on tabular; tune learning_rate/depth.
  [skl:target_power_transform] PowerTransformer on skewed regression targets often lifts R2.
```

The model is told to put `# motif: <id>` as the first line if it used a finding; the engine
parses that back into provenance for the knowledge ranker (section 6). No motif text is ever
executed; only the model's own authored code runs.

### 2.4 Requesting specific construct families

The typed schema (section 7.1) is rendered into the prompt as an allowed-construct menu so the
model knows the full reachable space. Example rendered directive block:

```
YOU MAY USE ANY OF: feature_engineering{poly, spline, pca, kbest, interactions},
target_transform{log1p, power, quantile}, stacking{StackingRegressor over 2-3 diverse bases},
calibration{CalibratedClassifierCV}. Neural nets via sklearn.neural_network.MLP* are allowed;
torch is NOT yet available (Phase later).
```

When the torch substrate lands, this menu gains `torch_module{...}` and the firewall allow-list
gains a vetted torch import set; no prompt-architecture change otherwise.

### 2.5 Temperature / diversity strategy for N candidates

Goal: N candidates that are genuinely different hypotheses, not N near-duplicates.

- **Temperature ladder:** `temperatures=(0.2, 0.6, 0.9)`. Candidate 0 is the model's
  best-guess (low temp, high reliability, likely to parse and run). Candidates 1..N-1 explore.
- **Variant directives:** each candidate i gets a distinct steering line appended so diversity
  is structural, not just sampling noise:
  - i=0: "Author the single pipeline you are most confident certifies above theta."
  - i=1: "Author a structurally DIFFERENT model class from candidate 0 and from the champion."
  - i=2: "Author a pipeline emphasizing feature engineering / target transform over model swap."
  - i>=3: combination / stacking directive (section 5) once a champion exists.
- **Dedup by Program.id** (sha of code) within a round and against `tried_labels`. If two
  candidates render identical code, only one survives; the engine does not pad.
- **Self-consistency use (section 7.2):** the N programs are not voted in code-space; they are
  all run and scored on VAL, and agreement among the *certified-or-promising* ones is a
  confidence signal surfaced in the result, not a selection shortcut.

---

## 3. Firewall and safety (firewall.py)

The sandbox (`frontier/sandbox.py`) already gives out-of-process isolation: subprocess,
`RLIMIT_CPU`, wall-clock kill, predictions-only return. The firewall here is a STATIC
pre-execution gate. It is defense in depth, not a replacement for the sandbox. Order:
`autocorrect_names -> strip_forbidden -> validate(AST) -> add_preamble`. A program that fails
`validate` is rejected before it ever reaches the sandbox.

### 3.1 AST allow-list validation

```python
ALLOWED_IMPORT_ROOTS = {"numpy", "sklearn"}     # torch added when substrate lands
ALLOWED_NODE_TYPES = {                          # everything else -> reject
    ast.Module, ast.FunctionDef, ast.Return, ast.Assign, ast.AnnAssign, ast.Expr,
    ast.Call, ast.Attribute, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.keyword, ast.arguments, ast.arg,
    ast.Import, ast.ImportFrom, ast.alias, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.Pow, ast.USub, ast.Subscript, ast.Slice, ast.Lambda,
    ast.IfExp, ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.Index,
    ast.ListComp, ast.comprehension,            # comprehensions for feature lists
}
FORBIDDEN_CALL_NAMES = {"eval", "exec", "open", "compile", "__import__", "input",
                        "globals", "locals", "getattr", "setattr", "vars",
                        "exit", "quit"}
FORBIDDEN_ATTR = {"system", "popen", "fork", "remove", "unlink", "rmtree",
                  "urlopen", "request", "get", "post", "Socket", "socket"}
LEAKAGE_CALLS = {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV",
                 "HalvingRandomSearchCV", "cross_val_score", "cross_validate",
                 "cross_val_predict", "learning_curve", "validation_curve"}

def validate(code: str, *, kind: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax: {e}"
    # 1. one build_estimator def, no module-level fit/side effects
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not any(f.name == "build_estimator" for f in funcs):
        return False, "no build_estimator()"
    # 2. walk every node: type allow-list, import roots, forbidden calls/attrs
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODE_TYPES:
            return False, f"forbidden node {type(node).__name__}"
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] \
                not in ALLOWED_IMPORT_ROOTS:
            return False, f"forbidden import {node.module}"
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                    return False, f"forbidden import {a.name}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in FORBIDDEN_CALL_NAMES:
            return False, f"forbidden call {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTR:
            return False, f"forbidden attribute .{node.attr}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in LEAKAGE_CALLS:
            return False, f"leakage construct {node.func.id}"
        # 3. no module-level statements except imports + the function def(s)
    for stmt in tree.body:
        if not isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            return False, f"module-level statement not allowed: {type(stmt).__name__}"
    return True, "ok"
```

Note on `LEAKAGE_CALLS`: a bare `GridSearchCV` *inside* `build_estimator()` returning the
search object as the estimator is actually safe - it cross-validates only on the train split
the caller passes to `.fit`, never on val/sealed (the sandbox only ever hands it train). The
real leakage risk is search that *peeks at the eval split*, which is impossible by
construction because authored code never receives the eval labels (sandbox passes `Xev` with
no `yev`). Therefore the conservative default REJECTS search wrappers (simplest provably-safe
rule), with a documented relaxation toggle `allow_internal_cv=False` that, when True, permits
`GridSearchCV(..., cv=...)` because the eval split is structurally unreachable. Ship with it
False; flag any change to True explicitly as a spec relaxation per the integrity rules.

### 3.2 The import preamble (symbol guarantee)

LLMs frequently reference a symbol they forgot to import. The preamble is a fixed header
prepended after validation that makes the common sklearn symbols available so a missing import
does not cause an avoidable `[import]`/`NameError` failure. It is added only if `validate`
passed, and it never overrides an import the model already wrote (Python re-import is
idempotent). The preamble imports only allow-listed roots, so it cannot widen the safety
surface.

```python
PREAMBLE = (
    "import numpy as np\n"
    "from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion\n"
    "from sklearn.compose import ColumnTransformer, TransformedTargetRegressor\n"
    "from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,\n"
    "    PolynomialFeatures, PowerTransformer, QuantileTransformer, FunctionTransformer,\n"
    "    SplineTransformer, OneHotEncoder)\n"
    "from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold\n"
    "from sklearn.decomposition import PCA, TruncatedSVD\n"
    # estimators are NOT blanket-imported: leaving them to the model keeps the autocorrect
    # signal meaningful (we only fix what the model actually referenced). Optional: import the
    # 12 catalog bases here too if reject rates from missing-estimator-import are high.
)

def add_preamble(code: str) -> str:
    return PREAMBLE + "\n" + code
```

Design choice: preamble imports *helpers* (transformers, pipeline machinery) but not
estimators, so that a model that references `HistGradientBoostingRegressor` without importing
it is corrected by `autocorrect_names` (which appends the exact right import), giving a clean
provenance trail of what was fixed rather than masking it. If reject/repair rates from
missing-estimator-imports are high in practice, fold the catalog estimator imports into the
preamble too (toggle `preamble_includes_estimators`).

### 3.3 Hallucinated-name auto-correction

Two failure modes: (a) a referenced symbol with no import; (b) a near-miss spelling
(`RandomForrestClassifier`, `GradientBoostingRegresor`). Strategy:

```python
# Build a registry of valid sklearn public estimators + their import paths, once, by
# importing sklearn and walking sklearn.utils.all_estimators() plus a curated transformer map.
_REGISTRY: dict[str, str] = build_symbol_registry()   # name -> "from sklearn.x import Name"

def autocorrect_names(code: str) -> tuple[str, list[str]]:
    tree = ast.parse(code)                              # if it won't parse, skip; validate rejects
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    imported = collect_imported_names(tree)
    corrections, header = [], []
    for name in referenced - imported - PYTHON_BUILTINS:
        if name in _REGISTRY:                           # valid symbol, just missing import
            header.append(_REGISTRY[name]); corrections.append(f"import+:{name}")
        else:
            cand = difflib.get_close_matches(name, _REGISTRY, n=1, cutoff=0.85)
            if cand:                                    # near-miss spelling -> rename + import
                code = rename_symbol(code, name, cand[0])
                header.append(_REGISTRY[cand[0]])
                corrections.append(f"rename:{name}->{cand[0]}")
            # else: leave it; validate/sandbox will surface a clean error for the revise loop
    return ("\n".join(header) + "\n" + code if header else code), corrections
```

`cutoff=0.85` is conservative: only obvious typos are auto-renamed; anything ambiguous is left
to fail loudly and inform the revise loop (section 4). All corrections are recorded in
`provenance["corrections"]` so a reviewer can see exactly what the firewall changed. The
firewall never invents an estimator the model did not ask for; it only resolves imports and
fixes near-miss spellings.

### 3.4 Stripping forbidden constructs

`strip_forbidden` rewrites the few mechanical hazards rather than rejecting (higher yield),
then `validate` is the hard gate for anything not mechanically fixable:

- `n_jobs=-1` / `n_jobs=N` -> `n_jobs=1` (RLIMIT_CPU correctness).
- `verbose=...` -> dropped (stdout pollution can corrupt the status-line protocol the sandbox
  parses; the sandbox reads the last stdout line as status).
- `random_state` absent on a known stochastic ctor -> inject `random_state=0` (reproducibility,
  required by Phase-7 seed-controlled re-execution).
- markdown fences, leading prose, trailing prose -> stripped by `strip_fences`.

Anything else dangerous (os import, open(), network attr, leakage wrapper) is NOT rewritten; it
is rejected by `validate` so we never silently "fix" a program into something the model did not
author. Rewrites are limited to the safe-and-mechanical set above and all are logged.

### 3.5 Execution is via frontier.sandbox only

The firewall produces a Program. The Program is executed exactly like any seed/mutation:
`ResearchEngine` calls `sandbox.run_program(p, X_train, y_train, X_eval, kind=...)`. The
AuthoringEngine has no execution path of its own. This is the second integrity layer: even if
a malicious construct slipped the static gate, it runs out-of-process with rlimits and returns
only a `preds.npy`. No authored code shares the parent's memory or can report a number.

---

## 4. Error-feedback / revise loop and agentic handoff

Phase 0's engine appends failures to `recent_errors` and they flow into the next round's
context (good, but coarse - next round, not next attempt). The authoring engine adds a tight
**within-round revise** so a failed candidate's traceback informs an immediate re-author,
before the round's diagnosis is even assembled.

### 4.1 Where the loop lives without editing engine.py

The engine calls `proposer.propose(context)` and then runs each returned Program. To get the
traceback back into the author, the revise loop must run where the RunResult is visible. Two
contract-safe placements:

- **Self-contained (default):** `CoreAuthoringProposer.propose` does a *cheap pre-flight* - it
  runs each authored Program through the sandbox once on a tiny subsample of train
  (`X_train[:k]`, build+fit only) to catch build/fit/import errors, and revises in place before
  returning to the engine. The engine then runs the (already-validated-and-runnable) Programs
  for real on the full val split. This keeps all LLM logic inside the proposer; the engine is
  untouched. Cost: one extra small sandbox call per candidate. Toggle `preflight=True`.

  Problem: the proposer does not have `X_train` in `context`. Fix without editing the contract:
  the caller constructs `CoreAuthoringProposer(client, preflight_data=(X_train_small, y_small))`
  by splitting outside the engine, OR `preflight=False` (default in strict mode) and rely on
  the next-round feedback path. Because the engine builds splits internally, the clean answer
  is placement (b).

- **Wrapper engine (recommended for the revise loop):** ship a thin
  `AuthoringResearchEngine(ResearchEngine)` subclass override of the inner per-candidate run
  that, on `not res.ok`, calls `engine.revise(prog, res, context)` and runs the revision before
  moving on. This subclasses Phase-0 `ResearchEngine` (allowed - composition, not edit) and is
  opt-in. If a builder prefers zero subclassing, the next-round feedback path (already in
  Phase 0) is the floor and is sufficient for Phase 0; the within-round revise is a Phase-4
  deepening.

### 4.2 revise() contract

```python
def revise(self, failed: Program, run_result, context: dict) -> Program | None:
    if not self.cfg.enable_revise or self.client is None:
        return None
    msg = (f"Your previous build_estimator() failed when run.\n"
           f"error_kind={run_result.error_kind}\nerror={run_result.error}\n"
           f"Previous code:\n{failed.code}\n\n"
           "Return a corrected build_estimator() that avoids this error. Same rules apply.")
    # multi-turn if ChatLLMClient, else single prompt = system + msg
    raw = self._call(self._revise_prompt(failed, run_result, context, msg),
                     temperature=0.3)            # low temp: we want a fix, not exploration
    if raw is None:
        return None
    return self._materialize(raw, context, spec=failed.provenance.get("spec"),
                             motifs=[], prompt=msg)
    # provenance records parent_id=failed.id, source stays "llm", label gets "_rev" suffix
```

Error-kind-specific revise hints (mirrors the diagnosis directive table):

| `error_kind` | revise hint appended |
|---|---|
| `import` | "Only numpy and sklearn import. Remove the offending import; use an sklearn equivalent." |
| `build` | "build_estimator() raised at construction. Check constructor arg names against sklearn." |
| `fit` | "Failed during fit. Likely a shape / dtype / numerical issue. Add scaling or imputation; reduce polynomial degree." |
| `timeout` / `cpu` | "Too slow under a CPU limit. Use HistGradientBoosting or a linear model, cap n_estimators, set n_jobs=1." |
| `oom` | "Out of memory. Avoid dense polynomial expansion on many features; use PCA / SelectKBest first." |

### 4.3 Handoff to the agentic repair module (Phase 4)

`revise` is the one-shot fix. The full agentic loop (write -> run -> read traceback -> inspect
data -> fix -> re-run, with tools) lives in the separate `agentic.py`. The handoff contract:

```python
class RepairHandoff(Protocol):
    def repair(self, program: Program, run_result, context: dict,
               sandbox_run: Callable) -> Program | None: ...
```

`AuthoringEngine` holds an optional `repair: RepairHandoff | None`. When set, after
`revise_max_attempts` one-shot revisions fail, the engine calls
`self.repair.repair(prog, res, context, sandbox.run_program)` and uses its result. The agentic
module gets the sandbox runner injected (so it executes through the same firewall) and the same
typed `RunResult` taxonomy. Until `agentic.py` lands, `repair=None` and the loop stops after
the one-shot revise - honest degrade, no fake capability.

---

## 5. Combination / regeneration strategy

Mutation in Phase 0 mutates *recipes* (dicts). The authoring engine mutates *actual code* and
*combines* multiple programs - strictly more expressive.

### 5.1 Champion code mutation

```python
def mutate_code(self, champion: Program, context: dict) -> Program | None:
    """Ask the LLM to improve the champion's literal code, not a recipe abstraction."""
    prompt = (self.SYSTEM + "\n" +
              f"This pipeline is the current best (val {context['best_score']}):\n"
              f"{champion.code}\n\n"
              "Author an improved build_estimator(): change one thing that plausibly raises "
              "the validation metric (add feature engineering, swap/regularize the model, add "
              "a target transform). Keep what is working. Return only code.")
    raw = self._call(prompt, temperature=0.5)
    return self._materialize(raw, context, spec=None, motifs=[], prompt=prompt) if raw else None
```

This is what `best_recipe` could not express: the champion may be an arbitrary authored
pipeline with no recipe dict, and we still mutate it because we mutate its source. `parent_id`
is set to the champion's id for the provenance tree.

### 5.2 Ensemble combination of top-N

After a round, the engine has VAL scores for every program (computed by `certify.score_val`,
the trusted parent). The combiner takes the top-K by VAL score (default K=3, the
*certified-or-promising* set - those that ran and scored, regardless of LLM claims) and asks
for a stack:

```python
def combine(self, top_programs: list[Program], context: dict) -> list[Program]:
    if self.client is None or len(top_programs) < 2:
        return []
    bodies = "\n\n# ---\n".join(f"# candidate {i} (val {p._val})\n{p.code}"
                                for i, p in enumerate(top_programs))
    prompt = (self.SYSTEM + "\n" +
              "Here are the strongest pipelines so far, with their validation scores:\n"
              f"{bodies}\n\n"
              "Author ONE build_estimator() that combines their strengths, e.g. a "
              "StackingRegressor/StackingClassifier over the diverse base estimators with a "
              "simple meta-learner, or a feature-union of their preprocessing. Return only code.")
    raw = self._call(prompt, temperature=0.4)
    p = self._materialize(raw, context, spec=None, motifs=[], prompt=prompt)
    return [p] if p else []
```

Critical firewall point: the combined program is authored fresh and re-validated and re-run.
We never trust the LLM's claim that the ensemble is better - it is just another candidate that
must out-score the champion on VAL and, if it wins, survive the sealed certificate. The val
scores shown in the prompt are the *trusted parent's* numbers, never LLM-reported.

### 5.3 When combination/mutation fire

The proposer emits, per round, in priority order: (1) top-confidence authored candidate, (2)
diversity candidates, (3) champion code-mutation (if a champion exists), (4) ensemble combine
(if >=2 promising programs exist). Seeds and mutations from the floor proposers fill any gap and
guarantee a non-empty pool. Selection is purely by VAL score across the merged pool, so an
authored ensemble only "wins" if it actually validates higher - never because it is fancier.

---

## 6. Composition with the engine (no Phase-0 edits)

### 6.1 Wiring

```python
from frontier.engine import ResearchEngine, EngineConfig
from frontier.proposers import SeedProposer, MutationProposer
from frontier.core.authoring import CoreAuthoringProposer, AuthoringConfig

client = resolve_backend(...)            # vfplatform; or None
proposers = [
    SeedProposer(),                                   # FLOOR: guaranteed non-empty + baseline
    MutationProposer(),                               # FALLBACK: offline generative
    CoreAuthoringProposer(client=client,              # CORE: arbitrary authored code
        config=AuthoringConfig(n=3), retriever=retriever),
]
engine = ResearchEngine(EngineConfig(rounds=4, llm_client=client), proposers=proposers)
result = engine.run(task)
```

`CoreAuthoringProposer(LLMProposer)` (the subclass route from 1.4) means
`engine.llm_active` reads True iff `client is not None`, with zero Phase-0 edits.

### 6.2 Ordering vs seed/mutation

The engine merges proposals across sources and dedups by `Program.id` / `label`, then runs all
and selects by VAL score (`engine.py:125-152`). Ordering in the `proposers` list affects only
which duplicate label is kept on a tie (first wins) and the order of sandbox runs within a
round - not which program is selected. Recommended order: `[Seed, Mutation, CoreAuthoring]`.
Rationale: seeds/mutations are cheap and deterministic, so they establish a champion and a
diagnosis *cheaply* in round 0; by the time the (costly) authoring proposer runs it has a
real champion, real `recent_errors`, and a real `best_recipe`/`family_ranking` to condition on.
Authoring still proposes in round 0 too (the floor just may beat it early), but its strength
compounds as diagnosis accrues. The floor never blocks authoring; it only guarantees a result
exists if authoring yields nothing.

### 6.3 Knowledge LinUCB reorder (Phase 8, separate module)

The knowledge module's `MetaLearner` (LinUCB) would reorder the *merged* proposal pool before
the engine runs them, so that under a per-round candidate budget the highest-expected-lift
programs run first. Contract-safe insertion: a `RankingProposer` *wrapper* that takes the other
proposers, gathers their proposals, asks `MetaLearner.rank(programs, context) -> order`, and
returns the reordered list. Because the engine selects by VAL score regardless of order,
ranking only changes *which programs get sandbox time under a budget*, never which promotes -
preserving invariant 4 (ranking expands/orders what is proposed, never what promotes). Features
for LinUCB: program source, base-family one-hot, has-feature-eng, has-target-transform,
authored-temperature, motif relevance. Reward: VAL-score improvement over champion. This stays
entirely in `knowledge.py`; authoring exposes the provenance fields LinUCB needs and nothing
more.

---

## 7. Innovations (genuinely novel, defensible)

### 7.1 Spec-conditioned program synthesis with a typed ProgramSpec schema

Instead of free-text "write a pipeline," the engine first synthesizes a typed
**ProgramSpec** from the diagnosis, then conditions code generation on it. The spec is a
contract the authored code must satisfy, and it is machine-checkable post-hoc.

```python
@dataclass
class ProgramSpec:
    task_kind: str
    must_handle: list[str]      # e.g. ["scale_sensitive_model", "skewed_target"]
    encourage: list[str]        # e.g. ["nonlinear_model", "feature_selection"]
    forbid: list[str]           # e.g. ["high_degree_poly"]  (from a prior LinAlgError)
    budget: dict                # {"max_fit_seconds": 30, "max_features_after_eng": 5000}
    def to_directives(self) -> str: ...    # renders to the prompt menu (section 2.4)
    def check(self, program_code: str) -> list[str]: ...   # AST checks: violated constraints
```

Why novel: program synthesis for AutoML is usually untyped prompt-to-code or a fixed search
space. A *typed spec derived from live diagnosis* gives (a) a checkable post-condition
(`spec.check` flags a program that used a forbidden construct, feeding the revise loop), and (b)
a stable target the knowledge ranker can featurize. The spec is the bridge between numeric
diagnosis and natural-language authoring, and it is removable scaffolding as models improve.

### 7.2 Self-consistency over multiple authored programs as a calibrated confidence signal

Run all N authored candidates, score each on VAL via the trusted parent. Compute
**structural agreement**: do the top performers converge on the same model family / same
feature-engineering motif? High agreement among high-VAL programs is a (validation-only,
non-promoting) confidence signal reported in the result: "4/5 high-scoring authored pipelines
independently chose gradient boosting with a target log-transform." Low agreement flags an
ill-posed or noisy task. Crucially this never short-circuits the certificate - it is metadata
on top of the sealed gate, and it uses only trusted VAL numbers. This is self-consistency
(known for reasoning) transplanted to *program* space and grounded in held-out scores rather
than the model's own confidence.

### 7.3 Retrieval-grounded code skeletons

A `Motif` may carry an optional `skeleton`: a vetted, firewall-passing code fragment (e.g. the
canonical `TransformedTargetRegressor(PowerTransformer)` block) extracted from a trusted source
(sklearn docs, a cited paper's released code). The prompt offers it as a *fillable skeleton*,
not a finished answer: "Here is a known-good pattern for skewed-target regression; adapt it to
this task." This grounds generation in code that already passes the firewall, raising first-try
run rates while keeping the model in control of composition. Skeletons are themselves run
through `validate` before being offered, so a poisoned skeleton cannot widen the safety surface.
Novel angle: retrieval-augmented *code* generation where the retrieved unit is a
verified-executable skeleton tied to a citable claim, with the citation propagated into
provenance for the Phase-9 report.

---

## 8. Risks and mitigations

| Risk | Vector | Mitigation (load-bearing) |
|---|---|---|
| **Reward hacking via leakage in authored code** | Code that peeks at eval labels, or CVs on the union of splits, to inflate VAL. | Structurally impossible: the sandbox passes `Xev` with NO labels; authored code receives only train (X,y) for `.fit` and unlabeled `Xev` for `.predict`. Plus AST `LEAKAGE_CALLS` rejection of search wrappers by default. Plus invariant 2: VAL/sealed numbers are computed by `science.py` from returned preds, never by authored code. |
| **Memorizing / target leakage through features** | A feature that is a deterministic function of the target (label leakage in the data itself). | Out of scope for the author (it is a data property), but caught by Phase-7 oracles: permuted-label collapse and trivial-baseline-beaten run on the *sealed* gate. The author cannot manufacture this; it can only exploit it if present, and the oracle refuses it. |
| **Sealed-test gaming** | Tuning to the sealed test across rounds. | Impossible by construction: this module never reads `splits.sealed_rows`; the engine touches sealed once, for the winner, via `certify_on_sealed` with `SealedTest(max_peeks=1)` which raises `PeekViolation` on a second peek (audited sound). |
| **Cost (LLM calls, sandbox compute)** | N candidates x rounds x revise x combine. | `AuthoringConfig.n` caps candidates; `revise_max_attempts` caps revises; LinUCB ranker (section 6.3) spends sandbox budget on high-expected-lift programs first; floor proposers are free and run first to establish a champion cheaply; dedup by `Program.id` avoids paying to run identical code. Cost model and early-kill portfolio are Phase 6. |
| **Nondeterminism (sampling, stochastic estimators)** | Different code each run; non-reproducible certificate. | `random_state=0` injected by `strip_forbidden`; `n_jobs=1` enforced; the *authored code and its sha* are stored in `Program.id` and provenance, so the exact winning program is reproducible byte-for-byte. Phase-7 seed-controlled re-execution re-runs the winner and confirms the sealed number reproduces before promotion. Sampling nondeterminism affects *which* programs are proposed, never *whether a proposed program certifies* - that is deterministic given the code. |
| **Hallucinated APIs / malformed code wasting runs** | Model invents `sklearn.magic`. | `autocorrect_names` + preamble fix the common cases; `validate` rejects the rest before any sandbox spend; the revise loop turns a clean typed error into a fix; all corrections logged in provenance. |
| **Silent capability faking** | Pretending the LLM path is active when it is not. | `client=None` -> `propose()` returns `[]` and `engine.llm_active=False` (subclass route). No fabricated proposals. The floor still produces an honest certified-or-declined result. |
| **Forbidden-construct slip-through** | A dangerous node the allow-list missed. | Defense in depth: even a slip runs out-of-process under RLIMIT_CPU + wall timeout + process-group kill, returning only `preds.npy`. The Phase-0 limitation (no network isolation) is documented; production runs the identical runner inside a no-network container / Prime Intellect pod - a substrate swap, not a rewrite. |

### 8.1 What the sealed cert + oracles defend, restated

The authoring engine is deliberately the *untrusted, creative* half. Its entire output is
"candidate code." Every number that could promote a result is computed by the frozen
`science.py` on data the authored code never saw the labels of, and the final promotion is a
single sealed peek the author cannot reach. Therefore even a perfectly adversarial author can,
at worst, waste compute or propose a leaky pipeline - and the Phase-7 oracles (permuted-label
collapse, trivial-baseline-beaten, metric-on-right-axis, adversarial self-refutation) plus the
one-peek sealed bound refuse it. Generation is creative and fallible; certification is frozen
and sound. That separation is the whole design.

---

## 9. Build order for the implementing agent

1. `firewall.py`: `validate`, `add_preamble`, `autocorrect_names`, `strip_forbidden`,
   `build_symbol_registry`. Self-test: feed 20 hand-written good/bad snippets, assert verdicts.
2. `schema.py`: `ProgramSpec` + `to_directives` + `check`. Self-test: spec.check flags a planted
   forbidden construct.
3. `authoring.py`: `AuthoringConfig`, `AuthoringEngine` (author/revise/combine/mutate_code),
   `CoreAuthoringProposer(LLMProposer)`. Use a `FakeClient` returning canned code for tests.
4. Tests: `test_firewall.py` (allow-list, leakage reject, autocorrect, preamble idempotence);
   `test_authoring_degrade.py` (client=None -> [] -> engine certifies on floor, llm_active False);
   `test_authoring_live.py` (FakeClient authors a hist-GBM -> runs in real sandbox -> certifies
   through the same gate as a seed).
5. Verify with `/Users/abdullahalghamdi/jax-env-311/bin/python -m pytest frontier/core/tests`.

Acceptance: with a FakeClient, an authored program that out-validates the floor is selected and
certified on the sealed test, and with `client=None` the system produces the identical
floor-only result it would without this module. No Phase-0 file is modified.
