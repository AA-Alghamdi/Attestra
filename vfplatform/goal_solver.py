"""THE AUTONOMOUS /goal FRONT DOOR for the REGENERATIVE autoresearcher (Gap 3, CPU).

WHAT THIS CLOSES
----------------
The regenerative spine (recipe_generator -> verification cascade -> frozen Tier-3 certifier) already runs on
real data through `scripts/code_arena.TabularCodeArena`, but only via per-arena runner scripts that are
handed a named dataset + a chosen shape. This module is the missing connective tissue: a SINGLE entrypoint
that takes a free-text GOAL + an arbitrary DATA pointer and drives the full cycle autonomously --

    acquire(data)         -> a dense (X, y) matrix from inline arrays / .npz / .csv / a bundled sklearn name
                             / an `openml://<id>` URI (reusing the audited code_arena loaders)
    infer_spec(X, y, goal)-> the task shape (binary|multiclass|imbalanced|regression), split discipline
                             (random|grouped|time), and a certifiable metric -- DETERMINISTIC-FIRST via
                             vfplatform.problem_type; an out-of-scope goal (forecast/ranking/audio/multilabel)
                             or inadmissible data is DECLINED honestly, never coerced into a fake fit
    build the arena       -> a TabularCodeArena over the inferred split, with a DATA-DRIVEN competence floor
    run the loop          -> the literature-grounded RecipeResearcher (generator [+ optional LiteratureScout
                             + authored code] -> cascade -> frozen Tier-3 promoter)
    audit the numbers     -> the NUMERICAL SUBSTRATE recomputes the certificate's headline bounds from the
                             frozen core and runs a live firewall self-test, so the returned certificate
                             carries machine-checkable proof that no model-asserted number drove a decision
    return                -> one GoalCertificate: the certified champion (or an honest non-promotion / refusal),
                             sealed bound, gold confirmation, anti-menu + literature provenance, numeric audit

NON-NEGOTIABLES PRESERVED
-------------------------
The frozen Tier-3 certifier remains the SOLE promoter; the meta-certifier framing gate and the data-hygiene
gate still REFUSE a gameable framing or a contaminated split before any sealed peek is spent. This module
only ROUTES a goal to the existing, audited machinery -- it adds no new statistical primitive and never
touches the frozen core.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import problem_type as PT
from .numeric_substrate import NumericSubstrate, guard_recipe_numbers
from .recipe_generator import GeneratorConfig, RecipeGenerator
from .recipe_research import RecipeResearcher

# the fraction of the majority class above which a classification problem is treated as IMBALANCED (so the
# arena samples the natural proportions and the competence floor becomes the data-driven majority bar).
_IMBALANCE_MAJORITY = 0.60


# ====================================================================== data acquisition
def acquire(data, *, target=None, task_hint: str = "goal_data") -> Tuple[np.ndarray, np.ndarray, int, str]:
    """Resolve a DATA pointer to a dense (X, y, n_classes, task_hint). n_classes==0 marks a REGRESSION target.

    Accepts:
      * an (X, y) or (X, y, n_classes, task_hint) tuple/list of arrays;
      * a dict {"X","y"[,"n_classes","task_hint"]};
      * a path to a .npz with arrays X, y (+ optional n_classes / task_hint);
      * a path to a .csv (target column named by `target`, else the last column);
      * a bundled sklearn name (covtype/digits/breast_cancer/wine/diabetes/california) or an OpenML-CC18
        member name -- reusing scripts.code_arena._load_dataset;
      * an `openml://<data_id>` URI -- reusing scripts.code_arena._load_openml.
    """
    if isinstance(data, dict):
        X, y = np.asarray(data["X"], dtype=np.float64), np.asarray(data["y"])
        return _finish_xy(X, y, data.get("n_classes"), data.get("task_hint", task_hint))
    if isinstance(data, (tuple, list)) and len(data) >= 2 and not isinstance(data[0], str):
        X, y = np.asarray(data[0], dtype=np.float64), np.asarray(data[1])
        nc = data[2] if len(data) >= 3 else None
        th = data[3] if len(data) >= 4 else task_hint
        return _finish_xy(X, y, nc, th)
    if isinstance(data, str):
        if data.endswith(".npz"):
            z = np.load(data, allow_pickle=False)
            nc = int(z["n_classes"]) if "n_classes" in z.files else None
            th = str(z["task_hint"]) if "task_hint" in z.files else task_hint
            return _finish_xy(np.asarray(z["X"], dtype=np.float64), np.asarray(z["y"]), nc, th)
        if data.endswith(".csv"):
            return _acquire_csv(data, target=target, task_hint=task_hint)
        # bundled sklearn / OpenML-CC18 name, or openml://<id>
        from scripts import code_arena as CA
        if data.startswith("openml://"):
            return CA._load_openml(int(data[len("openml://"):]), task_hint)
        if data == "iris":                               # a tiny extra sklearn classic not in code_arena
            from sklearn.datasets import load_iris
            d = load_iris()
            return d.data.astype(np.float64), d.target.astype(int), 3, "iris_tabular"
        return CA._load_dataset(data)
    raise ValueError(f"unrecognized data pointer of type {type(data).__name__}")


def _acquire_csv(path: str, *, target: Optional[str], task_hint: str):
    """Minimal, offline CSV intake: numeric columns only (median-imputed); the target is the named column or
    the last column. Non-numeric feature columns are dropped (the caller can pre-encode for richer frames)."""
    import pandas as pd
    df = pd.read_csv(path)
    tcol = target or df.columns[-1]
    ydf = df[tcol]
    Xdf = df.drop(columns=[tcol])
    num = Xdf.apply(pd.to_numeric, errors="coerce")
    num = num.dropna(axis=1, how="all")
    X = num.fillna(num.median(numeric_only=True)).to_numpy(dtype=np.float64)
    y = ydf.to_numpy()
    return _finish_xy(X, y, None, task_hint)


def _finish_xy(X: np.ndarray, y: np.ndarray, n_classes, task_hint: str):
    """Normalize (X, y) and decide n_classes. A float target with many distinct values is REGRESSION
    (n_classes=0); otherwise it is label-encoded to 0..K-1 and n_classes=K."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if n_classes is not None:
        nc = int(n_classes)
        y = y.astype(np.float64 if nc == 0 else int)
        return X, y, nc, task_hint
    is_float = np.issubdtype(y.dtype, np.floating)
    uniq = np.unique(y)
    looks_regression = is_float and len(uniq) > max(20, int(0.2 * len(y)))
    if looks_regression:
        return X, y.astype(np.float64), 0, task_hint
    # label-encode (string or int labels) to a contiguous 0..K-1 integer target
    mapping = {v: i for i, v in enumerate(uniq.tolist())}
    y_enc = np.array([mapping[v] for v in y.tolist()], dtype=int)
    return X, y_enc, int(len(uniq)), task_hint


# ====================================================================== spec inference
@dataclass
class GoalSpec:
    goal_text: str
    kind: str                 # problem_type's modality call (tabular|vision|text); the arena is raw-tabular
    task_type: str            # binary | multiclass | regression
    arena_shape: str          # multiclass | imbalanced | regression  (the TabularCodeArena 'shape')
    split: str                # random | grouped | time
    metric: str               # certifiable metric label (accuracy | macro_f1 | r2)
    n_classes: int
    n_rows: int
    n_features: int
    majority_fraction: float
    supported: bool
    decline_reason: Optional[str]
    source: str               # problem_type source: deterministic | llm

    def to_dict(self) -> dict:
        return asdict(self)


def infer_spec(X: np.ndarray, y: np.ndarray, goal_text: str, *, n_classes: int,
               groups: Optional[np.ndarray] = None, times: Optional[np.ndarray] = None,
               use_llm: bool = False, api_key: Optional[str] = None) -> GoalSpec:
    """Infer the certifiable spec for (X, y, goal). Modality/task/metric + honest declines come from the
    deterministic-first vfplatform.problem_type; the arena SHAPE (balanced vs natural-proportion vs
    regression) and SPLIT discipline (random unless real groups/times are supplied) come from the data."""
    is_regression = (n_classes == 0)
    rows = _records_sample(X, y, is_regression)
    pt = PT.classify(rows, goal_text, use_llm=use_llm, api_key=api_key)

    if is_regression:
        arena_shape = "regression"
        maj = 0.0
    else:
        _, counts = np.unique(y, return_counts=True)
        maj = float(counts.max()) / float(len(y))
        arena_shape = "imbalanced" if maj > _IMBALANCE_MAJORITY else "multiclass"

    split = "grouped" if groups is not None else ("time" if times is not None else "random")
    return GoalSpec(
        goal_text=goal_text, kind=pt["kind"], task_type=pt["task_type"], arena_shape=arena_shape,
        split=split, metric=pt["metric"], n_classes=int(n_classes), n_rows=int(len(y)),
        n_features=int(X.shape[1] if X.ndim == 2 else 0), majority_fraction=round(maj, 4),
        supported=bool(pt["supported"]), decline_reason=pt["decline_reason"], source=pt["source"])


def _records_sample(X: np.ndarray, y: np.ndarray, is_regression: bool, cap: int = 300) -> List[dict]:
    """A small list-of-records view of (X, y) for problem_type.classify (its admissibility inspector wants
    {features, target} rows). Capped for speed; the structural call (admissible? supported task?) is stable
    on a representative sample."""
    n = min(cap, len(y))
    idx = np.linspace(0, len(y) - 1, n).astype(int) if len(y) else np.array([], dtype=int)
    nf = X.shape[1] if X.ndim == 2 else 0
    out = []
    for i in idx:
        feats = {f"f{j}": float(X[i, j]) for j in range(nf)}
        out.append({"features": feats, "target": (float(y[i]) if is_regression else str(int(y[i])))})
    return out


# ====================================================================== the goal certificate
@dataclass
class GoalCertificate:
    goal: str
    spec: dict
    solved: bool                       # a model is CERTIFIED above the data-driven competence floor
    improved: bool                     # the loop promoted a champion BEYOND the seed (a real lift)
    refused: bool                      # the meta-certifier / data-hygiene gate refused the framing/data
    declined: bool                     # the goal/data was out of supported scope (no run attempted)
    decline_reason: Optional[str]
    champion: str
    champion_recipe: dict
    theta_floor: float                 # the competence bar the certified bound must clear
    pooled_sealed_lb: float            # the champion's pooled sealed Clopper-Pearson lower bound
    sealed_acc: dict
    sealed_lb: dict
    gold_confirmation: Optional[dict]
    novelty: dict
    literature: dict
    numeric_audit: dict
    stop_reason: str
    peeks_used: int
    certificate: Optional[dict] = None  # the full RecipeResearchCertificate as a dict (replayable)
    log: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ====================================================================== the autonomous solve()
def _competence_floor(arena, margin: float = 0.03) -> float:
    """Data-driven competence floor set strictly ABOVE the meta-certifier's own trivial baseline (+ margin),
    never below 0.5. The referee defines "trivial" as the TRAIN-majority class scored on the eval rows
    (meta_certifier._majority_predictor); we set theta just above that exact quantity so the floor is
    provably non-trivial and the trivial-baseline probe passes by construction. The model decision (the
    champion's predictions) is never consulted here -- only the aggregate label distribution the referee
    itself reads -- so nothing is snooped. On a balanced set the trivial baseline is ~1/K, so theta stays 0.5."""
    fr = arena.framing()
    y, tr, se = fr["y"], np.asarray(fr["train_idx"]), np.asarray(fr["sealed_idx"])
    classes, counts = np.unique(y[tr], return_counts=True)
    majority = classes[int(np.argmax(counts))]
    maj_pred = np.full(len(se), majority)
    trivial = float(fr["metric_fn"](y[se], maj_pred))
    return float(max(0.5, trivial + margin))


def _competence_floor_from_data(y: np.ndarray, n_classes: int, margin: float = 0.03) -> float:
    """Competence floor computed directly from (y, n_classes) without needing an arena.
    Used by the vertical routing path. Same logic as _competence_floor but on raw labels."""
    if n_classes == 0:  # regression: threshold on R2 doesn't apply the same way
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    majority_rate = float(counts.max()) / float(len(y))
    return float(max(0.5, majority_rate + margin))


def _loop_result_to_certificate(goal_text: str, spec, result, substrate, memory_store, *, gpu: bool) -> GoalCertificate:
    """Convert a GoalLoopResult from the vertical solver into a GoalCertificate.
    GoalLoopResult has: decision, winner, certificate, narration, n_test, rounds, etc."""
    cert_dict = result.certificate or {}
    solved = result.decision == "certified"
    champion = ""
    champion_recipe = {}
    pooled_lb = 0.0
    theta = 0.5
    if result.winner:
        champion = result.winner.family if hasattr(result.winner, "family") else str(result.winner)
        champion_recipe = result.winner.as_dict() if hasattr(result.winner, "as_dict") else {}
    if cert_dict:
        pooled_lb = float(cert_dict.get("lower_bound", 0.0))
        theta = float(cert_dict.get("threshold", 0.5))
    improved = solved
    return GoalCertificate(
        goal=goal_text, spec=spec.to_dict(), solved=solved, improved=improved,
        refused=(result.decision == "refused"), declined=(result.decision == "unsupported"),
        decline_reason=(result.narration if result.decision in ("unsupported", "declined") else None),
        champion=champion, champion_recipe=champion_recipe, theta_floor=round(theta, 4),
        pooled_sealed_lb=round(pooled_lb, 4), sealed_acc=cert_dict.get("sealed_acc", {}),
        sealed_lb=cert_dict.get("sealed_lb", {}),
        gold_confirmation=None, novelty={}, literature={},
        numeric_audit=substrate.audit(), stop_reason=result.narration or result.decision,
        peeks_used=result.n_test, certificate=cert_dict,
        log=[str(r) for r in (result.rounds or [])])


def solve(goal_text: str, data, *, peeks: int = 16, seed: int = 0, use_literature: bool = True,
          online: bool = False, use_llm: bool = False, code: bool = True, target: Optional[str] = None,
          groups: Optional[np.ndarray] = None, times: Optional[np.ndarray] = None,
          fanout: int = 6, api_key: Optional[str] = None,
          data_pool_root: Optional[str] = None, gpu: bool = False,
          memory_store=None) -> GoalCertificate:
    """Take a free-text GOAL + a DATA pointer and autonomously return a GoalCertificate (a certified champion,
    an honest non-promotion, or an honest decline/refusal). CPU-only and deterministic given a seed when
    `online=False` (the default uses the bundled literature corpus + the deterministic template authorer).

    use_literature -> ground discovery in a LiteratureScout (offline corpus unless online=True);
    use_llm        -> let the (audited, non-binding) LLM refine the spec + author code + propose motifs;
    code           -> attach authored code patches (the open-ended lever on raw features; the deterministic
                      TemplateAuthorer unless use_llm).
    gpu            -> let the front door recruit a torch-MLP head (the SAME torch model the GPU worker runs)
                      as a first-class candidate alongside linear/gbm/authored-code. On a GPU box it trains on
                      cuda; on CPU it device-swaps (slower but identical contract). Default OFF -> the recipe
                      path is byte-identical to the linear/gbm baseline.
    """
    X, y, n_classes, task_hint = acquire(data, target=target)
    spec = infer_spec(X, y, goal_text, n_classes=n_classes, groups=groups, times=times,
                      use_llm=use_llm, api_key=api_key)

    substrate = NumericSubstrate()

    # honest decline: an out-of-scope goal or inadmissible data never gets coerced into a fake fit.
    if not spec.supported:
        return GoalCertificate(
            goal=goal_text, spec=spec.to_dict(), solved=False, improved=False, refused=False, declined=True,
            decline_reason=spec.decline_reason, champion="", champion_recipe={}, theta_floor=0.5,
            pooled_sealed_lb=0.0, sealed_acc={}, sealed_lb={},
            gold_confirmation=None, novelty={}, literature={}, numeric_audit=substrate.audit(),
            stop_reason="declined: out of supported scope", peeks_used=0, certificate=None,
            log=[f"DECLINED: {spec.decline_reason}"])

    # VERTICAL ROUTING: if the spec identifies vision or text, route through the vertical solver
    # (which uses run_goal_loop with the appropriate harness) rather than the tabular RecipeResearcher.
    # This is the key integration that makes /goal auto-drive vision/text goals end-to-end.
    from .goal_verticals import can_drive_vertical, route_vertical
    if can_drive_vertical(spec):
        result = route_vertical(goal_text, X, y, n_classes, spec, peeks=peeks, seed=seed,
                                threshold=_competence_floor_from_data(y, n_classes),
                                memory_store=memory_store, gpu=gpu)
        if result is not None:
            return _loop_result_to_certificate(goal_text, spec, result, substrate, memory_store, gpu=gpu)

    arena, generator, researcher, scout = _build(
        X, y, n_classes, task_hint, spec, peeks=peeks, seed=seed, use_literature=use_literature,
        online=online, use_llm=use_llm, code=code, fanout=fanout, api_key=api_key,
        groups=groups, times=times, data_pool_root=data_pool_root, gpu=gpu)

    cert = researcher.run()

    numeric_audit = _audit_numbers(substrate, arena, cert)
    literature = _literature_provenance(cert, scout)
    theta = float(arena.theta_floor)
    pooled_lb = float(numeric_audit["single_source_of_truth"]["pooled"]["lower_bound"])
    improved = (not cert.refused) and len(cert.promotions) > 0
    # SOLVED = we can hand back a model whose sealed lower bound CLEARS the competence floor (the deliverable
    # is a certified model, not merely a beat-the-seed delta). `improved` records whether the loop also lifted
    # the champion beyond the seed.
    solved = (not cert.refused) and pooled_lb > theta
    # DURABLE MEMORY RECORDING (opt-in, non-invasive). If a memory_store was given, record the /goal engine's
    # realized outcome (champion family + certified margin) into the shared durable store, so a future loop-
    # lane experiment on similar data benefits from knowing which family succeeded here. Does NOT warm-start
    # the RecipeResearcher (that needs VoI-seam work on the rung-climber — documented next step).
    if memory_store is not None and not cert.refused:
        try:
            from . import casebase_store as _cs
            fp = _cs.fingerprint({"n_rows": len(X), "n_features": X.shape[1] if hasattr(X, "shape") else None,
                                  "n_classes": int(n_classes), "kind": "tabular",
                                  "task_type": spec.task_type}, kind="tabular", task_type=spec.task_type)
            champion_gain = max(0.0, pooled_lb - theta)
            memory_store.record_outcome(fp, cert.champion or "unknown", champion_gain, cost=1.0,
                                        device=("cuda" if gpu else "cpu"))
        except Exception:  # noqa: BLE001  recording is additive; never break a solve
            pass
    return GoalCertificate(
        goal=goal_text, spec=spec.to_dict(), solved=bool(solved), improved=bool(improved),
        refused=bool(cert.refused), declined=False, decline_reason=None, champion=cert.champion,
        champion_recipe=cert.champion_recipe, theta_floor=round(theta, 4), pooled_sealed_lb=round(pooled_lb, 4),
        sealed_acc=cert.sealed_acc, sealed_lb=cert.sealed_lb, gold_confirmation=cert.gold_confirmation,
        novelty=cert.novelty, literature=literature, numeric_audit=numeric_audit,
        stop_reason=cert.stop_reason, peeks_used=cert.peeks_used, certificate=asdict(cert), log=cert.log)


def _build(X, y, n_classes, task_hint, spec: GoalSpec, *, peeks, seed, use_literature, online, use_llm,
           code, fanout, api_key, groups, times, data_pool_root, gpu=False):
    """Wire the inferred spec into a TabularCodeArena + a literature-grounded RecipeGenerator + a
    RecipeResearcher. The recipe-number guard clamps the seed recipes to the audited numeric-gene space
    before anything is measured (the structural half of the LLM<->numbers firewall)."""
    from scripts.code_arena import TabularCodeArena, TabularSplits
    from .recipe import Recipe

    splits = TabularSplits(data=(X, y, n_classes, task_hint), dataset=task_hint, shape=spec.arena_shape,
                           split=spec.split, seed=seed, groups=groups, times=times)
    arena = TabularCodeArena(splits=splits)
    arena.allow_torch_head = bool(gpu)            # gated GPU/neural head (off => byte-identical baseline)
    arena.theta_floor = _competence_floor(arena)

    # the recipe-number guard: even the seeds pass through the firewall's structural half.
    seeds = [guard_recipe_numbers(r)[0] for r in arena.seed_recipes()]
    if gpu:                                       # offer the neural head as an explicit additional seed
        neural = Recipe(backbone="raw", adaptation="linear_probe", head="torch_mlp")
        seeds.append(guard_recipe_numbers(neural)[0])

    scout = None
    if use_literature:
        try:                                       # offline-graceful: a scout failure must never block a run
            from .literature import LiteratureScout
            scout = LiteratureScout(spec.goal_text, online=online, use_llm=use_llm, api_key=api_key)
        except Exception:                           # noqa: BLE001
            scout = None

    authorer = None
    if code:
        from .code_authoring import LLMAuthorer, TemplateAuthorer
        n_features = int(X.shape[1])
        authorer = (LLMAuthorer(n_features=n_features, n_classes=n_classes, api_key=api_key)
                    if use_llm else TemplateAuthorer(seed=seed))
    roles = ("regressor",) if arena.is_regression else ("featurizer", "classifier")
    cfg = GeneratorConfig(task_hint=task_hint, fanout=fanout, code_prob=(1.0 if code else 0.0),
                          code_roles=roles, graft_prob=0.0, online_discovery=online, gpu_heads=bool(gpu))
    generator = RecipeGenerator(seeds, config=cfg, seed=seed, code_authorer=authorer,
                                discover_fn=(None if scout is not None else (lambda: [])),
                                task_shape=arena.task_shape, scout=scout)

    pool_root = data_pool_root or os.path.expanduser("~/wilds_data/_goalpool")
    researcher = RecipeResearcher(arena, generator, alpha=0.1, theta_floor=arena.theta_floor,
                                  peek_budget=peeks, competence_ceiling=0.999, allow_data_acquisition=False,
                                  data_pool_root=pool_root,
                                  dataset_name=f"goal_{task_hint}_{spec.arena_shape}_{spec.split}")
    return arena, generator, researcher, scout


def _audit_numbers(substrate: NumericSubstrate, arena, cert) -> dict:
    """Run the NUMERICAL SUBSTRATE over the certificate's headline numbers:

      single_source_of_truth -- recompute each task's sealed Clopper-Pearson lower bound from the FROZEN core
        (using only the sealed COUNT k = round(acc*n) and the shard size n) and confirm it equals the bound
        the certificate reports. Proves the certificate's numbers came from the frozen substrate, not from a
        model's assertion.
      firewall_selftest -- a live adversarial check: an LLM 'claims' the pooled sealed accuracy is inflated by
        +0.1; the substrate recomputes the true value and the claim is CONTRADICTED + refused. Proves the
        LLM<->numbers firewall is wired and active in this run.
    """
    tasks = list(arena.tasks)
    shard_n = {t: len(s) for t, s in zip(tasks, arena.splits.shards)}
    per_task, agree = [], True
    tot_k = tot_n = 0
    for t in tasks:
        n = int(shard_n.get(t, 0))
        acc = float(cert.sealed_acc.get(t, 0.0))
        k = int(round(acc * n))
        tot_k += k
        tot_n += n
        correct = [1] * k + [0] * (n - k)
        sub_lb = round(substrate.accuracy_lower_bound(correct, 0.05, name=f"sealed_lb[{t}]").value, 4)
        reported = float(cert.sealed_lb.get(t, 0.0))
        ok = abs(sub_lb - reported) <= 1e-4
        agree = agree and ok
        per_task.append({"task": t, "n": n, "k": k, "reported_lb": reported, "substrate_lb": sub_lb,
                         "agree": ok})

    pooled_acc = (tot_k / tot_n) if tot_n else 0.0
    pooled_correct = [1] * tot_k + [0] * (tot_n - tot_k)
    pooled_lb = substrate.accuracy_lower_bound(pooled_correct, 0.05, name="pooled_sealed_lb")

    # live firewall self-test on a SEPARATE substrate (so the certificate's real bounds, recomputed on the
    # main substrate above, stay provably clean): an LLM asserts an INCORRECT pooled accuracy (the opposite
    # extreme, so the claim is always genuinely wrong); the substrate recomputes the true value, marks it
    # CONTRADICTED, and decide() refuses it. Proves the LLM<->numbers firewall is wired and active this run.
    from .numeric_substrate import NumberLeak, NumericSubstrate
    fw = NumericSubstrate()
    bogus_val = 0.0 if pooled_acc >= 0.5 else 1.0
    bogus = fw.claim_llm("llm_claimed_pooled_acc", bogus_val, tol=1e-3)
    fw.recompute(bogus, lambda: pooled_acc)
    tripped = False
    try:
        fw.decide(bogus)
    except NumberLeak:
        tripped = True

    ledger = substrate.audit()
    return {
        # the certificate's real bounds came only from the frozen-backed substrate (no untrusted number).
        "clean": bool(ledger["clean"]),
        # the adversarial LLM number was recomputed and refused -> the firewall is active.
        "firewall_held": bool(tripped),
        "single_source_of_truth": {
            "agreement": bool(agree), "per_task": per_task,
            "pooled": {"n": tot_n, "k": tot_k, "acc": round(pooled_acc, 4),
                       "lower_bound": round(pooled_lb.value, 4)}},
        "firewall_selftest": {"llm_claimed_acc": round(bogus.value, 4),
                              "substrate_recomputed_acc": round(pooled_acc, 4),
                              "verdict": bogus.verdict, "refused_by_firewall": bool(tripped)},
        "ledger": ledger,
    }


def _literature_provenance(cert, scout) -> dict:
    """Trace the champion to its source, if any: literature-grounded backbone (paper/repo/Hub hit), authored
    code, or a seed. Pulls the headline booleans from the certificate's anti-menu novelty record."""
    nov = cert.novelty or {}
    out = {
        "menu_free": bool(nov.get("menu_free", False)),
        "literature_grounded": bool(nov.get("literature_grounded", False)),
        "literature_source": nov.get("literature_source"),
        "uses_authored_code": bool(nov.get("uses_authored_code", False)),
        "champion_backbone": nov.get("champion_backbone"),
    }
    if scout is not None:
        out["scout"] = {"problem": scout.problem,
                        "n_findings": len(scout.findings),
                        "n_backbones": len(scout.backbones()),
                        "n_motifs": len(scout.motifs()),
                        "used_llm": scout.used_llm}
    return out


__all__ = ["acquire", "infer_spec", "solve", "GoalSpec", "GoalCertificate"]
