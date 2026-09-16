# Phase 0 interface contract (FROZEN - build against these exact signatures)

Every Phase 1–9 module plugs into the Phase 0 spine through these types. Do NOT edit Phase 0
files (`program.py`, `task.py`, `certify.py`, `sandbox.py`, `proposers.py`, `engine.py`,
`__init__.py`, `demo.py`, `tests/test_spine.py`). Build NEW modules that import these and
document wiring. Repo root is on `sys.path`; import the sound certifier as
`from vectorforge import science` and `from vfplatform import sealed` (never edit those).

Interpreter for any verification: `/Users/abdullahalghamdi/jax-env-311/bin/python`.

## program.py
```python
@dataclass
class Program:
    code: str            # defines build_estimator() -> unfitted sklearn-like estimator
    source: str          # "seed"|"mutation"|"llm"|"retrieval"|...
    label: str = ""
    parent_id: str|None = None
    provenance: dict = {} # e.g. {"recipe": {...}} for seed/mutation programs
    @property
    def id(self) -> str   # "source:label:sha12"

@dataclass
class RunResult:
    program_id: str
    ok: bool
    preds: list|None = None
    error: str = ""
    error_kind: str = ""  # "timeout"|"import"|"fit"|"build"|"oom"|"cpu"|"other"
    wall_seconds: float = 0.0
```

## task.py
```python
@dataclass
class Task:
    X: np.ndarray         # (n,d) float
    y: np.ndarray
    kind: str             # "classification"|"regression"
    theta: float          # promotion threshold the sealed LOWER bound must clear
    metric: str = ""      # ""-> "accuracy" (clf) / "r2" (reg). also "balanced_accuracy","macro_f1","neg_rmse","neg_mae"
    name: str = "task"
    n_features: int       # property
    labels: list          # property (sorted str labels for clf, [] for reg)
    def to_rows(self) -> list[dict]     # rows: {"target":..., "features":{f0..}, "_x":[...]}
    @staticmethod
    def rows_to_X(rows) -> np.ndarray
    @staticmethod
    def rows_to_y(rows, kind) -> np.ndarray
```

## certify.py  (adapter over the audited-sound science/sealed)
```python
class Splits:                          # train_rows, val_rows, sealed_rows, meta; .sealed_test -> SealedTest(max_peeks=1)
def make_splits(task, *, seed=0, test_frac=0.30, val_frac=0.20) -> Splits
def score_val(task, val_rows, preds) -> float           # selection score on VAL (trusted parent computes it)
def certify_on_sealed(task, splits, sealed_preds) -> dict  # ONE counted peek; returns the sealed certificate dict
# certificate dict keys: observed, n, theta, checks, lower_bound, certified(bool), peeks, sealed_digest, (p_value for accuracy)
```

## sandbox.py  (real out-of-process execution; returns predictions only - firewall)
```python
def run_program(program, X_train, y_train, X_eval, *, kind,
                wall_seconds=60.0, cpu_seconds=55, address_mb=4096) -> RunResult
```

## proposers.py
```python
class ProposalSource(Protocol):
    def propose(self, context: dict) -> list[Program]: ...
class SeedProposer: ...
class MutationProposer: ...
class LLMProposer:
    def __init__(self, client: Callable[[str],str]|None = None, n: int = 2)
def make_code(recipe: dict, kind: str) -> str    # recipe keys: base(str), scale(bool), poly(int), target_log(bool)
def recipe_label(recipe: dict) -> str
_BASES: dict[(kind,name)] -> (import_line, ctor_expr)
# context dict passed to propose(): task_kind, n_features, n_train, round, tried_labels(set),
#   best_label, best_score, best_id, best_recipe(dict|None), recent_errors(list[(label,error_kind,msg)])
```

## engine.py
```python
@dataclass
class EngineConfig:
    rounds=3; seed=0; test_frac=0.30; val_frac=0.20
    wall_seconds=60.0; cpu_seconds=55; llm_client: Callable[[str],str]|None = None
class ResearchEngine:
    def __init__(self, config: EngineConfig|None=None, proposers: list[ProposalSource]|None=None)
    def run(self, task: Task) -> EngineResult
@dataclass
class EngineResult:
    certified: bool; certificate: dict|None; winner: Program|None; winner_val_score: float|None
    history: list; diagnosis_trail: list[dict]; split_meta: dict; decline_reason: str; llm_active: bool
    def summary(self) -> str
```

## Standing invariants (every module must preserve)
1. Only the frozen certifier promotes. The sealed test is touched exactly once, for the winner.
2. The sandbox/untrusted code returns predictions only; every decision number is computed by `science.py`.
3. Hardcoded heuristics are seeds/fallbacks, never the promotion-bearing or search-bounding decision.
4. Generalization expands what may be PROPOSED, never what may PROMOTE.
5. Outcomes are honest: certified result or honest decline, never a relabeled validation score.
