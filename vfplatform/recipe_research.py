"""THE REGENERATIVE AUTORESEARCHER -- LLM/generator regenerates recipes, the cascade screens, the FROZEN
Tier-3 certifier promotes. This is the menu-free replacement for the encoder-registry climb.

WHY THIS EXISTS
---------------
`repr_researcher.ReprResearcher` proved the governance half (certify -> promote) is rigorous, but its
proposer enumerated a hand-authored encoder REGISTRY -- a menu. This module keeps the EXACT governance
discipline (frozen Clopper-Pearson lower bound + paired McNemar + BH-FDR + select-then-bound + gold
confirmation + session multiplicity) and swaps the proposer for the OPEN regenerative generator
(vfplatform/recipe_generator.RecipeGenerator over vfplatform/recipe.Recipe). The champion is a full training
RECIPE, the backbone axis is drawn from the published zoo, and code-bearing recipes are admitted through the
frozen authoring sandbox -- so the reachable space is "anything published / authorable", not a fixed list.

THE NON-NEGOTIABLE LOOP (the user's spine, made literal)
--------------------------------------------------------
    generator REGENERATES recipes  (recipe_generator: mutate/recombine/graft + open backbone discovery)
        -> cascade SCREENS cheaply (verification: Tier 0..2 competence; allocate the scarce sealed peeks)
            -> frozen Tier-3 PROMOTES (verification.certify_tier3 -> the frozen paired-FDR sealed compare)
The generator only proposes; the frozen certifier is the sole promoter. A wrong proposal can at most waste a
peek -- it can never mint a certificate.

EVERY PREVIOUSLY-DARK MODULE NOW HAS A MEASURED ROLE HERE (this is the wire-or-delete payoff):
    regeneration   -- the generator's operators (mutate/recombine/graft) + a live QDArchive of evaluated
                      recipes (MAP-Elites diversity over (backbone_family, adaptation, aggregation)).
    search         -- RecipeSearchProblem + BudgetedSearch is an alternate driver (best-first recursion with
                      plateau escalation); selected with driver="search".
    verification   -- the Tier 0..3 cascade is the screening + sole-promotion spine (used by BOTH drivers).
    meta_certifier -- validate_certifier adversarially probes the FRAMING (trivial-baseline / label-shuffle /
                      straddle-leak / split-leak) BEFORE the certifier is trusted to drive search on a novel
                      problem; an untrustworthy framing REFUSES the run (verification leads, not optimism).
    pareto         -- the certified accuracy-vs-cost frontier over the champion chain (the deliverable).
    authoring_bridge / authoring -- code-bearing recipes are admitted through the frozen three-stage sandbox
                      before they may spend a peek; unsafe/un-buildable code is rejected at Tier 0.

CONTRACT: pure policy + the repo's own modules. Reaches the frozen science core ONLY through the arena's
mcnemar / bh / lower_bound (battery + science), exactly like ReprResearcher. Deterministic given a seed +
arena, so it is unit-testable with a synthetic arena (tests/test_recipe_research.py).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import datapool as DP
from . import escalate as ESC
from . import meta_certifier as MC
from .battery import benjamini_hochberg, mcnemar_pvalue
from .pareto import MAXIMIZE, MINIMIZE, ParetoCandidate, ParetoFront
from .recipe import DiscoveryLedger, Recipe, qd_descriptor
from .recipe_generator import RecipeGenerator
from .regeneration import QDArchive
from .repr_researcher import TaskMeasure
from .verification import Candidate, VerificationCascade

try:
    from vectorforge import science as _science
    _CP_LOWER = _science.clopper_pearson_lower
except Exception:                                            # pragma: no cover - frozen core always present
    _CP_LOWER = None

# The escalation rungs, identical semantics to the encoder climb but now over RECIPE axes.
RUNG_MODEL = "model"             # regenerate the BACKBONE (open zoo)
RUNG_FEATURES = "features"       # regenerate augmentation / head / authored featurizer
RUNG_CAPACITY = "capacity"       # escalate adaptation strength / epochs / aggregation
RUNG_DATA = "data_acquisition"   # regenerate the data strategy


# ====================================================================== the arena contract
class RecipeArena(ABC):
    """The domain the autoresearcher operates on, keyed by full Recipes. A concrete arena supplies the
    measurement of a recipe on the identical sealed rows; the frozen statistical primitives have correct
    defaults (battery + science), so a subclass typically implements only `measure` + `seed_recipes`.

    Required:
      tasks                          -> the task names McNemar+BH-FDR runs across (>=1).
      measure(recipe)                -> {task: TaskMeasure} on the identical sealed rows.
      seed_recipes()                 -> the weak starting recipe(s) (the DiscoveryLedger's seed set).
    Optional (sensible defaults):
      mcnemar / bh / lower_bound     -> frozen primitives (override only to inject a test double).
      gold_measure(recipe)           -> {task: 0/1} on a NEVER-PEEKED disjoint gold set, or None.
      cost(recipe)                   -> GPU-cost proxy for the Pareto frontier (default recipe.cost()).
      framing()                      -> dict(X,y,train_idx,sealed_idx,fit_predict_fn,metric_fn[,groups,times])
                                        for the meta-certifier framing probe, or None to skip it.
      task_hint / task_shape         -> bias the generator's discovery + code authoring.
    """

    tasks: List[str] = []
    task_hint: str = "general"
    task_shape: str = "binary"

    @abstractmethod
    def measure(self, recipe: Recipe) -> Dict[str, TaskMeasure]:        # pragma: no cover - interface
        ...

    @abstractmethod
    def seed_recipes(self) -> List[Recipe]:                            # pragma: no cover - interface
        ...

    def mcnemar(self, chal_correct: Sequence[int], base_correct: Sequence[int]) -> float:
        return mcnemar_pvalue(list(chal_correct), list(base_correct))

    def bh(self, pvalues: Sequence[float], alpha: float) -> List[int]:
        return sorted(benjamini_hochberg(list(pvalues), alpha))

    def lower_bound(self, correct: Sequence[int]) -> float:
        c = [int(x) for x in correct]
        k, n = sum(c), len(c)
        if n == 0:
            return 0.0
        if _CP_LOWER is None:                                          # pragma: no cover - frozen core present
            return k / n
        return round(_CP_LOWER(k, n, 0.05), 4)

    def cost(self, recipe: Recipe) -> float:
        return recipe.cost()

    def gold_measure(self, recipe: Recipe) -> Optional[Dict[str, Sequence[int]]]:
        return None

    def framing(self) -> Optional[dict]:
        return None


# ====================================================================== the certified comparison
@dataclass
class RecipeComparison:
    """A challenger recipe's certified suite-level comparison vs the current champion, on identical sealed
    rows. `certified` is the ONLY field the frozen Tier-3 reads; `sealed_lb` (pooled frozen CP lower bound)
    only breaks ties among already-certified winners (argmax, never a fresh test)."""
    survivors: List[str]
    n: int
    mean_lift: float
    sealed_lb: float
    per_task: List[dict]
    certified: bool

    def as_cert(self) -> dict:
        return {"certified": self.certified, "survivors": self.survivors, "n": self.n,
                "mean_lift": round(self.mean_lift, 4), "sealed_lb": round(self.sealed_lb, 4),
                "per_task": self.per_task}


def compare_recipe(arena: RecipeArena, meas: Dict[str, TaskMeasure],
                   champ_correct: Dict[str, Sequence[int]], alpha: float) -> RecipeComparison:
    """Paired McNemar(challenger > champion) per task -> BH-FDR(alpha) -> survivors with positive lift; the
    pooled frozen Clopper-Pearson lower bound is the tie-break. Identical discipline to ReprResearcher."""
    per_task, pvals, lifts = [], [], []
    for t in arena.tasks:
        m = meas[t]
        p = arena.mcnemar(m.sealed_correct, champ_correct[t])
        base_acc = sum(champ_correct[t]) / len(champ_correct[t])
        lift = m.acc - base_acc
        per_task.append({"task": t, "acc": round(m.acc, 4), "base_acc": round(base_acc, 4),
                         "lift": round(lift, 4), "p_gt_champ": round(p, 4)})
        pvals.append(p)
        lifts.append(lift)
    rej = set(arena.bh(pvals, alpha))
    survivors = [arena.tasks[i] for i in range(len(arena.tasks)) if i in rej and lifts[i] > 0]
    mean_lift = sum(lifts) / len(lifts) if lifts else 0.0
    sealed_lb = arena.lower_bound([x for t in arena.tasks for x in meas[t].sealed_correct])
    certified = len(survivors) > 0 and mean_lift > 0
    return RecipeComparison(survivors, len(arena.tasks), mean_lift, sealed_lb, per_task, certified)


# ====================================================================== certificate
@dataclass
class RecipePromotion:
    rung: str
    from_recipe: str
    to_recipe: str
    survivors: List[str]
    mean_lift: float
    sealed_lb: float


@dataclass
class RecipeResearchCertificate:
    champion: str                          # the champion recipe label
    champion_recipe: dict                  # the champion recipe's full fields (replayable)
    promotions: List[RecipePromotion]
    move_class_path: List[str]
    stop_reason: str
    sealed_lb: Dict[str, float]
    sealed_acc: Dict[str, float]
    pareto_front: List[str]
    pareto_report: str
    data_ceiling_tasks: List[str]
    peeks_used: int
    novelty: dict                          # the ANTI-MENU proof (DiscoveryLedger.novelty)
    discovery: dict                        # pool growth: seeds vs proposed vs retrieved
    framing_report: Optional[dict] = None  # the meta-certifier framing verdict (None if no framing supplied)
    data_hygiene: Optional[dict] = None    # the data-pool admission record (contamination/leakage on the data)
    gold_confirmation: Optional[dict] = None
    multiplicity: Optional[dict] = None
    qd_coverage: int = 0                   # MAP-Elites cells filled (recipe-space diversity explored)
    driver: str = "climb"
    refused: bool = False                  # True if the meta-certifier rejected the framing (no search ran)
    log: List[str] = field(default_factory=list)


# ====================================================================== the search-driver adapter
class RecipeSearchProblem:
    """Adapts the open generator + arena to vfplatform.search.SearchProblem so BudgetedSearch can drive the
    recursion. root = current champion; expand = generator.expand (OPEN); make_candidate evaluates a recipe
    and packages a Candidate whose certify_fn is the frozen compare vs the champion."""

    def __init__(self, researcher: "RecipeResearcher", champion: Recipe,
                 champ_correct: Dict[str, Sequence[int]]):
        self._r = researcher
        self._champion = champion
        self._champ_correct = champ_correct
        self._cache: Dict[str, Tuple[Dict[str, TaskMeasure], RecipeComparison, float]] = {}

    def _eval(self, recipe: Recipe):
        sig = recipe.signature()
        if sig not in self._cache:
            meas = self._r._measure(recipe)
            cmp = compare_recipe(self._r.arena, meas, self._champ_correct, self._r.alpha)
            val_pool = self._r._val_pool(meas)
            val_mean = sum(val_pool) / len(val_pool) if val_pool else 0.0
            self._cache[sig] = (meas, cmp, val_mean)
        return self._cache[sig]

    def root(self) -> Recipe:
        return self._champion

    def expand(self, state: Recipe, move_class: str) -> List[Recipe]:
        return [r for r in self._r.generator.expand(state, move_class) if self._r._admit(r)]

    def make_candidate(self, state: Recipe) -> Candidate:
        meas, cmp, _ = self._eval(state)
        val_pool = self._r._val_pool(meas)
        return Candidate(name=state.label(), sanity_ok=True, val_outcomes=val_pool,
                         surrogate_score=None, certify_fn=(lambda c=cmp: c.as_cert()))

    def priority(self, state: Recipe) -> float:
        return self._eval(state)[2]

    def describe(self, state: Recipe) -> str:
        return state.label()


# ====================================================================== the controller
class RecipeResearcher:
    """The regenerative champion-climbing controller. Same governance as ReprResearcher; OPEN proposer."""

    def __init__(self, arena: RecipeArena, generator: RecipeGenerator, *, alpha: float = 0.1,
                 theta_floor: float = 0.5, peek_budget: int = 16, competence_ceiling: float = 0.90,
                 allow_data_acquisition: bool = True, driver: str = "climb", max_rounds: int = 40,
                 data_pool_root: Optional[str] = None, dataset_name: str = "arena"):
        if not arena.tasks:
            raise ValueError("arena must declare at least one task")
        self.arena = arena
        self.data_pool_root = data_pool_root
        self.dataset_name = dataset_name
        self.generator = generator
        self.alpha = float(alpha)
        self.theta_floor = float(theta_floor)
        self.peek_budget = int(peek_budget)
        self.competence_ceiling = float(competence_ceiling)
        self.allow_data_acquisition = bool(allow_data_acquisition)
        self.driver = driver
        self.max_rounds = int(max_rounds)
        self._data_hygiene: Optional[dict] = None
        self.archive = QDArchive()                 # live MAP-Elites archive of evaluated recipes
        self._meas_cache: Dict[str, Dict[str, TaskMeasure]] = {}

    # -- measurement + helpers --------------------------------------------------------------------------
    def _measure(self, recipe: Recipe) -> Dict[str, TaskMeasure]:
        sig = recipe.signature()
        if sig not in self._meas_cache:
            meas = self.arena.measure(recipe)
            self._meas_cache[sig] = meas
            # archive every evaluated recipe by its QD descriptor (diversity bookkeeping)
            val_pool = [float(x) for t in self.arena.tasks for x in meas[t].val_correct]
            fitness = sum(val_pool) / len(val_pool) if val_pool else 0.0
            self.archive.add(recipe.to_genome(), fitness, qd_descriptor(recipe))
        return self._meas_cache[sig]

    def _val_pool(self, meas: Dict[str, TaskMeasure]) -> List[float]:
        return [float(x) for t in self.arena.tasks for x in meas[t].val_correct]

    def _correct(self, meas: Dict[str, TaskMeasure]) -> Dict[str, Sequence[int]]:
        return {t: list(meas[t].sealed_correct) for t in self.arena.tasks}

    def _pooled_lb(self, correct: Dict[str, Sequence[int]]) -> float:
        return self.arena.lower_bound([x for t in self.arena.tasks for x in correct[t]])

    def _admit(self, recipe: Recipe) -> bool:
        """Gate a code-bearing recipe through the frozen authoring sandbox BEFORE it may spend a peek.
        Recipes without a code_patch pass trivially. Wires authoring_bridge / authoring into production."""
        if not recipe.code_patch:
            return True
        try:
            from . import authoring as A
            from . import authoring_bridge as AB
            role = recipe.code_role or "classifier"
            spec = A.EstimatorSpec(role=role, n_features=8,
                                   n_classes=(2 if role == "classifier" else 1))
            report = AB.admit_method(recipe.code_patch, spec, family=recipe.backbone_family())
            return bool(report.admitted)
        except Exception:
            return False

    # -- the framing gate (verification leads on a novel problem) ---------------------------------------
    def _framing_gate(self, framing: Optional[dict], log: List[str]) -> Optional[MC.MetaCertReport]:
        if not framing:
            return None
        probe = {k: framing[k] for k in
                 ("X", "y", "train_idx", "sealed_idx", "fit_predict_fn", "metric_fn", "groups", "times")
                 if k in framing}
        report = MC.validate_certifier(theta=self.theta_floor, **probe)
        log.append("meta-certifier framing probe -> " + report.summary().splitlines()[0])
        return report

    # -- the data-hygiene gate (datapool: refuse a CONTAMINATED dataset before any search) --------------
    def _datapool_gate(self, framing: Optional[dict], log: List[str]) -> Optional[dict]:
        """Admit the arena's dataset into the append-only DataPool, computing its data certificate
        (near-duplicate train/sealed straddle = the non-negotiable leak). A leaky dataset is REFUSED here,
        before the autoresearcher is allowed to spend a single sealed peek on it. Wires datapool into
        production as the data-side complement to the meta-certifier's framing-side gate."""
        if self.data_pool_root is None or not framing:
            return None
        try:
            pool = DP.DataPool(self.data_pool_root)
            # the kNN label-disagreement + class-balance hygiene checks are classification-only; on a
            # continuous target they are meaningless, so the data certificate runs in regression mode
            # (near-duplicate straddle still blocks; the gameable-benchmark checks are the framing gate's).
            is_reg = (str(getattr(self.arena, "task_shape", "")) == "regression")
            ds = pool.add(self.dataset_name, modality=self.arena.task_hint, task_type=self.arena.task_shape,
                          X=framing["X"], y=framing["y"], train_idx=framing["train_idx"],
                          sealed_idx=framing["sealed_idx"], split_method="domain_shift",
                          source=self.dataset_name, require_cert=True, is_regression=is_reg)
            cert = ds.data_certificate
            log.append(f"data-hygiene gate -> ADMITTED {self.dataset_name!r} "
                       f"(near_dup_straddle={cert.get('near_dup_straddle')}, n={ds.n})")
            return {"admitted": True, "dataset": self.dataset_name, "certificate": cert,
                    "pool_counts": pool.counts()}
        except DP.DataPoolError as e:
            log.append(f"data-hygiene gate -> REFUSED {self.dataset_name!r}: {e}")
            return {"admitted": False, "dataset": self.dataset_name, "reason": str(e)}

    # ===================================================================== the run
    def run(self) -> RecipeResearchCertificate:
        log: List[str] = []

        # the framing is featurized once, then probed from two independent angles before any search:
        #   meta-certifier -> is the FRAMING gameable? (trivial-baseline / shuffle / straddle / split leak)
        #   datapool       -> is the DATA contaminated? (near-dup train/sealed straddle = non-negotiable leak)
        framing = self.arena.framing()
        framing_report = self._framing_gate(framing, log)
        data_hygiene = self._datapool_gate(framing, log)
        self._data_hygiene = data_hygiene
        framing_bad = framing_report is not None and not framing_report.trustworthy
        data_bad = data_hygiene is not None and not data_hygiene.get("admitted", True)
        if framing_bad or data_bad:
            # gameable framing OR contaminated data -> REFUSE to search (never let optimism override the referee)
            reason = "framing rejected by meta-certifier" if framing_bad else \
                     "dataset rejected by data-hygiene certificate (contamination)"
            seed = self.generator.ledger
            champ = self.arena.seed_recipes()[0]
            return RecipeResearchCertificate(
                champion=champ.label(), champion_recipe=champ.to_genome(), promotions=[],
                move_class_path=[], stop_reason=reason,
                sealed_lb={}, sealed_acc={}, pareto_front=[], pareto_report="",
                data_ceiling_tasks=list(self.arena.tasks), peeks_used=0,
                novelty=seed.novelty(champ), discovery=self._discovery_dict(),
                framing_report=({"trustworthy": framing_report.trustworthy,
                                 "summary": framing_report.summary()} if framing_report else None),
                data_hygiene=data_hygiene, driver=self.driver, refused=True, log=log)

        cascade = VerificationCascade(theta=self.theta_floor, alpha=0.05)
        champion = self.arena.seed_recipes()[0]
        champ_meas = self._measure(champion)
        champ_correct = self._correct(champ_meas)
        start_recipe = champion
        start_correct = {t: list(champ_correct[t]) for t in self.arena.tasks}
        tried = {champion.signature()}
        promotions: List[RecipePromotion] = []
        peeks_used = 0
        lb_hist: List[Optional[float]] = [self._pooled_lb(champ_correct)]
        move_class = RUNG_MODEL
        move_class_path = [move_class]
        stop_reason = "loop exhausted"
        log.append(f"start champion={champion.label()} sealed_lb={lb_hist[0]:.3f} "
                   f"theta_floor={self.theta_floor} driver={self.driver}")

        if self.driver == "search":
            champion, champ_correct, peeks_used, promotions, move_class_path = self._run_search(
                cascade, champion, champ_correct, tried, log)
        else:
            champion, champ_correct, peeks_used, promotions, move_class_path, stop_reason = self._run_climb(
                cascade, champion, champ_correct, tried, promotions, lb_hist, move_class,
                move_class_path, log)

        return self._finalize(champion, champ_correct, start_recipe, start_correct, promotions,
                              move_class_path, stop_reason, peeks_used, framing_report, log)

    # -- the primary climb driver (round-based, proven discipline) --------------------------------------
    def _run_climb(self, cascade, champion, champ_correct, tried, promotions, lb_hist, move_class,
                   move_class_path, log):
        stop_reason = "loop exhausted"
        peeks_used = 0
        rounds = 0
        while rounds < self.max_rounds:
            rounds += 1
            proposals = [r for r in self.generator.expand(champion, move_class)
                         if r.signature() not in tried]

            eligible: List[Tuple[float, Recipe, RecipeComparison]] = []
            for recipe in proposals:
                if not self._admit(recipe):
                    log.append(f"[{move_class}] {recipe.label()}: REJECTED by authoring sandbox (unsafe/"
                               f"un-buildable code patch) -- never reaches a peek")
                    tried.add(recipe.signature())
                    continue
                meas = self._measure(recipe)
                tried.add(recipe.signature())
                cmp = compare_recipe(self.arena, meas, champ_correct, self.alpha)
                val_pool = self._val_pool(meas)
                val_mean = sum(val_pool) / len(val_pool) if val_pool else 0.0
                cand = Candidate(name=recipe.label(), sanity_ok=True, val_outcomes=val_pool,
                                 surrogate_score=None, certify_fn=(lambda c=cmp: c.as_cert()))
                if cascade.cheap_screen(cand).survived:
                    eligible.append((val_mean, recipe, cmp))

            # spend Tier-3 peeks best-validation-first (select-then-bound). Tier-3 is the SOLE promoter.
            eligible.sort(key=lambda e: (-e[0], e[1].signature()))
            certified: List[Tuple[float, Recipe, RecipeComparison]] = []
            for val_mean, recipe, cmp in eligible:
                if peeks_used >= self.peek_budget:
                    log.append("peek budget exhausted -> stop spending sealed certifications")
                    break
                cand = Candidate(name=recipe.label(), sanity_ok=True,
                                 val_outcomes=self._val_pool(self._measure(recipe)),
                                 surrogate_score=None, certify_fn=(lambda c=cmp: c.as_cert()))
                res = cascade.certify_tier3(cand)
                peeks_used += 1
                if res.promoted:
                    certified.append((cmp.sealed_lb, recipe, cmp))
                    log.append(f"[{move_class}] {recipe.label()}: frozen Tier-3 CERTIFIED over "
                               f"{champion.label()} (survivors {len(cmp.survivors)}/{cmp.n}, "
                               f"mean_lift {cmp.mean_lift:+.3f}, sealed_lb {cmp.sealed_lb:.3f})")

            if certified:
                certified.sort(key=lambda e: (-e[0], e[1].signature()))
                best_lb, best_recipe, best_cmp = certified[0]
                promotions.append(RecipePromotion(move_class, champion.label(), best_recipe.label(),
                                                   best_cmp.survivors, best_cmp.mean_lift, best_lb))
                log.append(f"[{move_class}] PROMOTE {champion.label()} -> {best_recipe.label()} "
                           f"(sealed_lb {best_lb:.3f})")
                champion = best_recipe
                champ_correct = self._correct(self._measure(champion))
                lb_hist.append(self._pooled_lb(champ_correct))
                continue

            lb_hist.append(None)
            decision = ESC.escalation_decision(lb_hist, move_class, ESC.DEFAULT_K,
                                                budget_left=self.peek_budget - peeks_used,
                                                allow_data_acquisition=self.allow_data_acquisition)
            log.append(f"escalate: {decision}")
            if decision.decision == ESC.STOP or decision.to_class is None:
                stop_reason = decision.reason
                break
            move_class = decision.to_class
            move_class_path.append(move_class)
        return champion, champ_correct, peeks_used, promotions, move_class_path, stop_reason

    # -- the alternate search driver (wires search.BudgetedSearch) --------------------------------------
    def _run_search(self, cascade, champion, champ_correct, tried, log):
        from .search import BudgetedSearch
        promotions: List[RecipePromotion] = []
        peeks_used = 0
        move_class_path: List[str] = ["model"]
        sprints = 0
        while peeks_used < self.peek_budget and sprints < self.max_rounds:
            sprints += 1
            problem = RecipeSearchProblem(self, champion, champ_correct)
            search = BudgetedSearch(problem, self.theta_floor,
                                    peek_budget=max(1, self.peek_budget - peeks_used),
                                    cascade=cascade, allow_data_acquisition=self.allow_data_acquisition)
            result = search.run()
            peeks_used += result.peeks_used
            for mc in result.move_class_path:
                if mc not in move_class_path:
                    move_class_path.append(mc)
            if result.certified and result.winner_state is not None:
                winner: Recipe = result.winner_state
                meas = self._measure(winner)
                cmp = compare_recipe(self.arena, meas, champ_correct, self.alpha)
                promotions.append(RecipePromotion("search", champion.label(), winner.label(),
                                                   cmp.survivors, cmp.mean_lift, cmp.sealed_lb))
                log.append(f"[search] sprint#{sprints} CERTIFIED {champion.label()} -> {winner.label()} "
                           f"(peeks now {peeks_used})")
                champion = winner
                champ_correct = self._correct(meas)
                continue
            log.append(f"[search] sprint#{sprints} found no certified improvement -> stop")
            break
        return champion, champ_correct, peeks_used, promotions, move_class_path

    # -- finalize: pareto + gold + multiplicity + novelty -----------------------------------------------
    def _discovery_dict(self) -> dict:
        led = self.generator.ledger
        scout = getattr(self.generator, "scout", None)
        d = {"pool_size": len(self.generator.pool),
             "n_seed_backbones": len(led.seed_backbones),
             "n_proposed_backbones": len(led.proposed_backbones),
             "n_retrieved_backbones": len(led.retrieved_backbones),
             "n_literature_backbones": len(led.literature_backbones),
             "literature_backbones": sorted(led.literature_backbones),
             "seed_backbones": sorted(led.seed_backbones),
             "sample_pool": self.generator.pool[:20]}
        if scout is not None:
            d["literature"] = {"problem": getattr(scout, "problem", None),
                               "n_findings": len(getattr(scout, "findings", [])),
                               "n_motifs": len(scout.motifs()),
                               "motifs": [m.name for m in scout.motifs()],
                               "used_llm": getattr(scout, "used_llm", False)}
        return d

    def _finalize(self, champion, champ_correct, start_recipe, start_correct, promotions, move_class_path,
                  stop_reason, peeks_used, framing_report, log) -> RecipeResearchCertificate:
        champ_lb = {t: self.arena.lower_bound(champ_correct[t]) for t in self.arena.tasks}
        champ_acc = {t: sum(champ_correct[t]) / len(champ_correct[t]) for t in self.arena.tasks}
        data_ceiling = sorted(t for t in self.arena.tasks if champ_acc[t] < self.competence_ceiling)

        # certified Pareto front over the certified deliverables we can always re-measure: the START
        # baseline and the FINAL champion (accuracy lower bound vs recipe cost). Intermediate champions are
        # captured in `promotions`; the frontier reports the endpoints of the climb, both certified.
        pareto_cands: List[ParetoCandidate] = []
        for recipe in [start_recipe, champion]:
            if recipe.label() in {c.name for c in pareto_cands}:
                continue
            correct = self._correct(self._measure(recipe))
            lb = min(self.arena.lower_bound(correct[t]) for t in self.arena.tasks)
            pareto_cands.append(ParetoCandidate(name=recipe.label(), metric_lb=lb,
                                                 latency_ms=self.arena.cost(recipe),
                                                 cost_usd=self.arena.cost(recipe), ece=0.0, certified=True))
        front = ParetoFront(pareto_cands, axes=[("metric_lb", MAXIMIZE), ("cost_usd", MINIMIZE)])
        front_labels = [c.name for c in front.nondominated()]

        # gold confirmation on a never-peeked set (if the arena supplies one)
        gold_confirmation = None
        champ_gold = self.arena.gold_measure(champion)
        if champ_gold is not None:
            base_gold = self.arena.gold_measure(start_recipe)
            gold_confirmation = self._gold_compare(champion, start_recipe, champ_gold, base_gold)
            log.append(f"gold-confirmation (never-peeked n={gold_confirmation['gold_n']}): "
                       f"{champion.label()} vs {start_recipe.label()} -> "
                       f"{len(gold_confirmation['survivors'])}/{gold_confirmation['n_tasks']} FDR survivors, "
                       f"confirmed={gold_confirmation['confirmed']}")

        multiplicity = self._multiplicity(champion, start_recipe, start_correct, champ_correct,
                                          peeks_used, gold_confirmation)

        novelty = self.generator.ledger.novelty(champion)
        log.append(f"ANTI-MENU novelty: menu_free={novelty['menu_free']} "
                   f"(champion backbone {novelty['champion_backbone']!r}, "
                   f"backbone_is_novel={novelty['backbone_is_novel']}, "
                   f"uses_authored_code={novelty['uses_authored_code']}, "
                   f"from_retrieval={novelty['from_retrieval']}, "
                   f"literature_grounded={novelty.get('literature_grounded')})")
        if novelty.get("literature_grounded") and novelty.get("literature_source"):
            src = novelty["literature_source"]
            log.append(f"LITERATURE-GROUNDED: champion backbone traces to "
                       f"[{src.get('source')}:{src.get('ident')}] {src.get('title')!r} ({src.get('url')})")

        return RecipeResearchCertificate(
            champion=champion.label(), champion_recipe=self._recipe_dict(champion), promotions=promotions,
            move_class_path=move_class_path, stop_reason=stop_reason,
            sealed_lb={t: round(champ_lb[t], 4) for t in self.arena.tasks},
            sealed_acc={t: round(champ_acc[t], 4) for t in self.arena.tasks},
            pareto_front=front_labels, pareto_report=front.report(),
            data_ceiling_tasks=data_ceiling, peeks_used=peeks_used,
            novelty=novelty, discovery=self._discovery_dict(),
            framing_report=({"trustworthy": True, "summary": framing_report.summary()}
                            if framing_report is not None else None),
            data_hygiene=self._data_hygiene,
            gold_confirmation=gold_confirmation, multiplicity=multiplicity,
            qd_coverage=self.archive.coverage(), driver=self.driver, refused=False, log=log)

    @staticmethod
    def _recipe_dict(recipe: Recipe) -> dict:
        d = recipe.to_genome()
        d["code_role"] = recipe.code_role
        d["uses_code_patch"] = recipe.code_patch is not None
        d["code_patch"] = recipe.code_patch          # the authored SOURCE (auditable + replayable), or None
        d["label"] = recipe.label()
        return d

    def _gold_compare(self, champion: Recipe, baseline: Recipe,
                      champ_gold: Dict[str, Sequence[int]], base_gold: Dict[str, Sequence[int]]) -> dict:
        per_task, pvals, lifts = [], [], []
        for t in self.arena.tasks:
            cg, bg = list(champ_gold[t]), list(base_gold[t])
            p = self.arena.mcnemar(cg, bg)
            ca = sum(cg) / len(cg) if cg else 0.0
            ba = sum(bg) / len(bg) if bg else 0.0
            per_task.append({"task": t, "champ_gold_acc": round(ca, 4), "base_gold_acc": round(ba, 4),
                             "lift": round(ca - ba, 4), "p_gt_base": round(p, 4), "n": len(cg)})
            pvals.append(p)
            lifts.append(ca - ba)
        rej = set(self.arena.bh(pvals, self.alpha))
        survivors = [self.arena.tasks[i] for i in range(len(self.arena.tasks)) if i in rej and lifts[i] > 0]
        mean_lift = sum(lifts) / len(lifts) if lifts else 0.0
        champ_lb = self.arena.lower_bound([x for t in self.arena.tasks for x in champ_gold[t]])
        base_lb = self.arena.lower_bound([x for t in self.arena.tasks for x in base_gold[t]])
        return {"champion": champion.label(), "baseline": baseline.label(),
                "n_tasks": len(self.arena.tasks),
                "gold_n": sum(len(champ_gold[t]) for t in self.arena.tasks),
                "survivors": survivors, "mean_lift": round(mean_lift, 4),
                "champion_gold_lb": round(champ_lb, 4), "baseline_gold_lb": round(base_lb, 4),
                "per_task": per_task, "confirmed": len(survivors) > 0 and mean_lift > 0}

    def _multiplicity(self, champion: Recipe, start_recipe: Recipe, start_correct, champ_correct,
                      peeks_used: int, gold_confirmation: Optional[dict]) -> dict:
        n_tasks = len(self.arena.tasks)
        m = max(int(peeks_used), 0)
        bonf_alpha = self.alpha / m if m > 0 else self.alpha
        per_task, pvals, lifts = [], [], []
        for t in self.arena.tasks:
            cc, sc = list(champ_correct[t]), list(start_correct[t])
            p = self.arena.mcnemar(cc, sc)
            ca = sum(cc) / len(cc) if cc else 0.0
            sa = sum(sc) / len(sc) if sc else 0.0
            per_task.append({"task": t, "champ_acc": round(ca, 4), "start_acc": round(sa, 4),
                             "lift": round(ca - sa, 4), "p_gt_start": round(p, 4)})
            pvals.append(p)
            lifts.append(ca - sa)
        fdr_rej = set(self.arena.bh(pvals, self.alpha))
        fdr_survivors = [self.arena.tasks[i] for i in range(n_tasks) if i in fdr_rej and lifts[i] > 0]
        bonf_survivors = [self.arena.tasks[i] for i in range(n_tasks)
                          if pvals[i] < bonf_alpha and lifts[i] > 0]
        champ_lb = self.arena.lower_bound([x for t in self.arena.tasks for x in champ_correct[t]])
        gold_ok = bool(gold_confirmation and gold_confirmation.get("confirmed"))
        return {"champion": champion.label(), "baseline": start_recipe.label(), "n_tasks": n_tasks,
                "sealed_comparisons": m, "mcnemar_tests_total": m * n_tasks,
                "per_comparison_fdr_alpha": self.alpha, "session_bonferroni_alpha": round(bonf_alpha, 6),
                "champion_vs_start_sealed": per_task, "fdr_survivors_nominal": fdr_survivors,
                "bonferroni_survivors_session": bonf_survivors, "champion_sealed_lb": round(champ_lb, 4),
                "robust_to_session_multiplicity": len(bonf_survivors) > 0,
                "gold_independent_confirmation": gold_ok}


__all__ = ["RecipeArena", "RecipeComparison", "compare_recipe", "RecipePromotion",
           "RecipeResearchCertificate", "RecipeSearchProblem", "RecipeResearcher",
           "RUNG_MODEL", "RUNG_FEATURES", "RUNG_CAPACITY", "RUNG_DATA"]
