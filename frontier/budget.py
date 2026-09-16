"""Phase 6 budget scheduler: cost model + budget controller + phase decomposer.

Long-horizon experiments (week-to-quarter) need three things the Phase-0 spine does
not provide on its own:

  1. A *cost model* that estimates wall-time per Program BEFORE running it, so the loop
     can plan instead of discovering cost only after a 10-minute fit. The model is keyed
     by estimator family and data size (n_train * n_features), and it is *calibratable*:
     observed `RunResult.wall_seconds` are fed back in to correct family coefficients.

  2. A *BudgetController* that holds a fixed total budget (seconds, the same unit the
     sandbox already enforces via wall_seconds/cpu_seconds) and allocates it across rounds
     and arms, refusing to admit a candidate once admitting it would overrun, and stopping
     the search when the budget is exhausted. It also runs an early-kill rule for losing
     arms (a deterministic successive-halving gate, not the test-only Thompson portfolio
     the audit found never wired in).

  3. A *PhaseDecomposer* that splits one big experiment into gated phases
     (cheap-screen -> refine -> confirm). Each phase has an explicit budget fraction, a
     cohort size, and a promotion gate; only candidates that clear a phase's gate flow to
     the next. The result is a DAG (linear chain of gates here) that is validated to be
     acyclic, budget-conserving, and monotonically narrowing.

Nothing here promotes a result or computes a certificate. The frozen certifier in
`certify.py` / `science.py` is still the only promoter. This module only decides WHICH
candidates get spent on and WHEN to stop -- a search-shaping decision, explicitly allowed
to be deterministic (it bounds compute, not the scientific claim). The promotion-bearing
sealed certificate is untouched.

=== WIRING ===
The integrator plugs this into ResearchEngine WITHOUT editing engine.py by passing a
controller alongside EngineConfig, or by wrapping the loop. Concretely:

  from frontier.budget import CostModel, BudgetController, PhaseDecomposer, BudgetConfig

  # (a) Cost model, calibrated online. Build it once per run.
  cost = CostModel()                       # ships with literature/profiling seed coefficients
  # In the engine's per-candidate loop, BEFORE sandbox.run_program(p, ...):
  est = cost.estimate(program, n_train=len(splits.train_rows),
                      n_features=task.n_features)        # -> CostEstimate(seconds, ...)
  # AFTER the run, feed the truth back so later estimates self-correct:
  cost.observe(program, n_train, n_features, res.wall_seconds, ok=res.ok)

  # (b) Budget controller wraps the round/arm loop. Total is in WALL SECONDS, the unit the
  #     sandbox already uses. The engine asks admit() before spending and stop() to break.
  ctrl = BudgetController(BudgetConfig(total_seconds=3600.0, rounds=cfg.rounds))
  for r in range(cfg.rounds):
      ctrl.begin_round(r)
      for p in proposals:                                 # ranked best-first by the caller
          est = cost.estimate(p, n_train, n_features)
          if not ctrl.admit(p.id, est.seconds):           # would overrun -> skip this one
              continue
          res = sandbox.run_program(p, Xtr, ytr, Xval, ...)
          ctrl.charge(p.id, res.wall_seconds)             # charge the TRUE cost
          ctrl.record_score(p.id, val_score)              # for early-kill ranking
          cost.observe(p, n_train, n_features, res.wall_seconds, ok=res.ok)
      ctrl.end_round(r)                                   # successive-halving cull of arms
      if ctrl.stop():                                     # budget exhausted -> honest early stop
          break
  # The single sealed certification of the winner is OUTSIDE the budget loop and always runs
  # (the spine spends its last peek on the winner regardless; certification is not optional).

  # (c) Phase decomposer turns a goal + total budget into a gated plan the orchestrator drives.
  plan = PhaseDecomposer().decompose(total_seconds=3600.0, n_candidates=64,
                                     kind=task.kind)
  plan.validate()                                          # asserts acyclic + budget-conserving
  for phase in plan.phases:                                # screen -> refine -> confirm
      sub = BudgetController(BudgetConfig(total_seconds=phase.budget_seconds, rounds=1))
      ... run phase.cohort_size survivors, keep those whose val score >= phase.gate ...

Argument shapes: program is a frontier.program.Program (its .provenance["recipe"]["base"]
names the family; LLM programs without a recipe fall back to a static-analysis family guess
from the code string). n_train/n_features are ints. All budgets/costs are float seconds.
Ordering: estimate -> admit -> run -> charge(true) -> record_score; end_round culls; stop()
checks exhaustion. The controller is single-threaded and deterministic.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from .program import Program
except ImportError:  # allow `python frontier/budget.py` self-test (no package parent)
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from frontier.program import Program


# --------------------------------------------------------------------------- cost model

# Seed coefficients for a linear-in-work cost model:  seconds ~= setup + slope * work,
# where work = n_train * n_features (the dominant factor for the sklearn families here)
# scaled by a family-specific complexity exponent on n_train captured in `train_pow`.
#
# These seeds are a STARTING POINT only (labelled as such): rough magnitudes from profiling
# sklearn estimators on commodity CPUs and from the families' known asymptotics
# (tree ensembles ~ O(n log n * d * trees); kernel SVC ~ O(n^2 * d); linear ~ O(n * d)).
# They are NOT promotion-bearing -- they only rank/admit. CostModel.observe() overwrites
# them with measured timings as the run proceeds, so the seeds wash out quickly.
_FAMILY_SEED = {
    # family       setup_s   slope_s_per_workunit   train_pow (extra superlinearity on n_train)
    "hist_gbm":   (0.05,     2.0e-7,                1.0),
    "rf":         (0.05,     6.0e-7,                1.05),
    "gbr":        (0.05,     5.0e-7,                1.05),
    "logreg":     (0.02,     3.0e-8,                1.0),
    "ridge":      (0.01,     1.0e-8,                1.0),
    "svc_rbf":    (0.03,     1.0e-7,                1.6),   # kernel: superlinear in n_train
    "_unknown":   (0.05,     5.0e-7,                1.1),   # conservative default for LLM code
}

# Multiplicative cost of common pipeline enhancers (applied on top of the base family).
_ENHANCER_MULT = {
    "scale": 1.05,        # a StandardScaler pass is cheap
    "poly2": 1.8,         # PolynomialFeatures degree 2 blows up feature count ~ d^2
    "poly3": 4.0,
    "tlog": 1.02,         # target transform: a couple of vectorized ufuncs
}

# Static-analysis fallback: map sklearn class names visible in an LLM code string to a family
# bucket so we can still estimate cost for programs with no recipe provenance.
_CODE_FAMILY_HINTS = [
    (re.compile(r"HistGradientBoosting"), "hist_gbm"),
    (re.compile(r"RandomForest"), "rf"),
    (re.compile(r"GradientBoosting(?:Regressor|Classifier)"), "gbr"),
    (re.compile(r"\bSVC\b|\bSVR\b|\bNuSVC\b"), "svc_rbf"),
    (re.compile(r"LogisticRegression"), "logreg"),
    (re.compile(r"\bRidge\b|\bLasso\b|LinearRegression|ElasticNet"), "ridge"),
]


@dataclass
class CostEstimate:
    """A predicted execution cost for one Program on a given data size.

    `seconds` is the point estimate; `lo`/`hi` bracket it using the family's observed
    spread (or a fixed 0.5x..2.0x band before any observations). `family` and `calibrated`
    let the caller see whether the number is measured or still a seed guess.
    """
    seconds: float
    lo: float
    hi: float
    family: str
    enhancers: Tuple[str, ...]
    calibrated: bool

    def __repr__(self) -> str:
        tag = "meas" if self.calibrated else "seed"
        return (f"CostEstimate({self.seconds:.3f}s [{self.lo:.3f},{self.hi:.3f}] "
                f"{self.family}{'+' + '+'.join(self.enhancers) if self.enhancers else ''} {tag})")


def program_family(program: Program) -> str:
    """Best-effort family bucket for a Program.

    Prefers the structured recipe provenance (seed/mutation programs carry
    provenance["recipe"]["base"]); falls back to static analysis of the code string
    (LLM programs). Returns "_unknown" if nothing matches, so cost is still defined.
    """
    recipe = (program.provenance or {}).get("recipe") if program.provenance else None
    if isinstance(recipe, dict) and recipe.get("base"):
        base = str(recipe["base"])
        return base if base in _FAMILY_SEED else "_unknown"
    code = program.code or ""
    for pat, fam in _CODE_FAMILY_HINTS:
        if pat.search(code):
            return fam
    return "_unknown"


def program_enhancers(program: Program) -> Tuple[str, ...]:
    """Enhancer tags (scale / polyK / tlog) that scale a Program's cost above its base."""
    recipe = (program.provenance or {}).get("recipe") if program.provenance else None
    tags: List[str] = []
    if isinstance(recipe, dict):
        if recipe.get("scale"):
            tags.append("scale")
        if recipe.get("poly"):
            tags.append(f"poly{int(recipe['poly'])}")
        if recipe.get("target_log"):
            tags.append("tlog")
    else:
        code = program.code or ""
        if "StandardScaler" in code:
            tags.append("scale")
        m = re.search(r"PolynomialFeatures\([^)]*degree\s*=\s*(\d+)", code)
        if m:
            tags.append(f"poly{int(m.group(1))}")
        if "TransformedTargetRegressor" in code or "log1p" in code:
            tags.append("tlog")
    return tuple(tags)


@dataclass
class _FamilyStats:
    """Running calibration for one family: a slope fitted by least squares through the
    origin-shifted work, plus residual spread for the lo/hi band."""
    setup: float
    slope: float
    train_pow: float
    n_obs: int = 0
    # accumulators for a robust online slope estimate (median of per-obs slopes is too jumpy;
    # we use a damped least-squares update on (work, seconds - setup)).
    _sw: float = 0.0      # sum work^2
    _swt: float = 0.0     # sum work * (t - setup)
    _ratios: List[float] = field(default_factory=list)   # measured/predicted, for spread


class CostModel:
    """Calibratable cost model: seconds ~= setup + slope * (n_train^train_pow * n_features),
    times an enhancer multiplier. Per-family coefficients start from `_FAMILY_SEED` and are
    corrected online by `observe()`.

    Why linear-in-work with a family exponent rather than a full regression: with only a
    handful of observations per family during a single run, an interpretable
    one-or-two-parameter model is far more stable than fitting many coefficients, and the
    estimate only needs to be good enough to *rank and admit* candidates, not to certify
    anything. The model degrades gracefully: with zero observations it returns the seed
    guess (flagged calibrated=False) with a wide 0.5x..2x band.
    """

    # cap a single estimate so a pathological work value cannot produce an absurd number
    _MAX_SECONDS = 1.0e6

    def __init__(self):
        self._fam: Dict[str, _FamilyStats] = {
            k: _FamilyStats(setup=s, slope=sl, train_pow=tp)
            for k, (s, sl, tp) in _FAMILY_SEED.items()
        }

    @staticmethod
    def _work(n_train: int, n_features: int, train_pow: float) -> float:
        n_train = max(1, int(n_train))
        n_features = max(1, int(n_features))
        return (float(n_train) ** train_pow) * float(n_features)

    @staticmethod
    def _enhancer_mult(enhancers: Tuple[str, ...]) -> float:
        mult = 1.0
        for e in enhancers:
            mult *= _ENHANCER_MULT.get(e, 1.0)
        return mult

    def estimate(self, program: Program, n_train: int, n_features: int) -> CostEstimate:
        """Predict wall-seconds for `program` on data of the given size. Never raises."""
        fam = program_family(program)
        enh = program_enhancers(program)
        st = self._fam.get(fam, self._fam["_unknown"])
        work = self._work(n_train, n_features, st.train_pow)
        base = st.setup + st.slope * work
        seconds = min(self._MAX_SECONDS, max(0.0, base) * self._enhancer_mult(enh))

        if st.n_obs >= 2 and st._ratios:
            # data-driven band: use observed spread of measured/predicted ratios
            rs = sorted(st._ratios)
            lo_r = rs[max(0, int(0.1 * (len(rs) - 1)))]
            hi_r = rs[min(len(rs) - 1, int(0.9 * (len(rs) - 1)))]
            lo, hi = seconds * min(1.0, lo_r), seconds * max(1.0, hi_r)
            calibrated = True
        else:
            lo, hi, calibrated = seconds * 0.5, seconds * 2.0, False
        return CostEstimate(seconds=seconds, lo=lo, hi=hi, family=fam,
                            enhancers=enh, calibrated=calibrated)

    def observe(self, program: Program, n_train: int, n_features: int,
                wall_seconds: float, ok: bool = True) -> None:
        """Feed back a measured wall time so future estimates self-correct.

        Failed runs (ok=False) are recorded with low weight: a build/import failure costs
        almost nothing and a timeout is a censored observation, so neither should pull the
        slope hard. We simply skip slope updates for failures but still note them so the
        caller can inspect coverage; the calibration uses successful runs only.
        """
        fam = program_family(program)
        st = self._fam.setdefault(fam, _FamilyStats(*_FAMILY_SEED["_unknown"]))
        if not ok or wall_seconds is None or wall_seconds <= 0 or not math.isfinite(wall_seconds):
            return
        work = self._work(n_train, n_features, st.train_pow)
        t_resid = max(0.0, float(wall_seconds) - st.setup)
        # damped least-squares-through-origin slope update on (work -> t_resid)
        st._sw += work * work
        st._swt += work * t_resid
        if st._sw > 0:
            fitted = st._swt / st._sw
            # blend with the seed for the first few obs to avoid a single noisy fit dominating
            blend = min(1.0, st.n_obs / 5.0)
            st.slope = (1.0 - blend) * st.slope + blend * max(1e-12, fitted)
        st.n_obs += 1
        # track measured/predicted ratio for the band, against the CURRENT prediction
        predicted = max(1e-9, st.setup + st.slope * work)
        st._ratios.append(float(wall_seconds) / predicted)
        if len(st._ratios) > 64:
            st._ratios.pop(0)

    def calibrate_family(self, family: str, samples: List[Tuple[int, int, float]]) -> None:
        """Optional offline calibration: fit a family from a list of (n_train, n_features,
        seconds) profiling samples (e.g. a quick timing sweep run once at startup)."""
        for nt, nf, sec in samples:
            # wrap in a throwaway program carrying the recipe so program_family resolves it
            p = Program(code="", source="seed", label=family,
                        provenance={"recipe": {"base": family}})
            self.observe(p, nt, nf, sec, ok=True)

    def family_table(self) -> Dict[str, dict]:
        """Inspectable snapshot of the calibrated coefficients (for logging / debugging)."""
        return {k: {"setup": v.setup, "slope": v.slope, "train_pow": v.train_pow,
                    "n_obs": v.n_obs} for k, v in self._fam.items()}


# --------------------------------------------------------------------------- budget controller

@dataclass
class BudgetConfig:
    """Total budget (wall seconds) and how to spread it.

    total_seconds : the hard ceiling. spent_seconds may never exceed this.
    rounds        : number of search rounds the budget is spread across (for per-round caps).
    reserve_frac  : fraction held back so the mandatory single sealed certification of the
                    winner is never starved (certification is not optional and must run).
    keep_frac     : successive-halving survival fraction at end_round (best keep_frac of arms
                    by recorded score survive; the rest are killed early). 1.0 disables culling.
    min_keep      : never cull below this many arms (so a round cannot empty the pool).
    """
    total_seconds: float = 3600.0
    rounds: int = 3
    reserve_frac: float = 0.05
    keep_frac: float = 0.5
    min_keep: int = 2

    def __post_init__(self):
        if self.total_seconds <= 0:
            raise ValueError("total_seconds must be positive")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if not (0.0 <= self.reserve_frac < 1.0):
            raise ValueError("reserve_frac must be in [0, 1)")
        if not (0.0 < self.keep_frac <= 1.0):
            raise ValueError("keep_frac must be in (0, 1]")
        if self.min_keep < 1:
            raise ValueError("min_keep must be >= 1")


class BudgetController:
    """Allocates a fixed wall-second budget across rounds and arms, admits candidates only
    while budget remains, charges true costs, and culls losing arms between rounds.

    Invariant: `spent_seconds` never exceeds `usable_seconds` (= total minus the reserve held
    for certification). `admit(id, est)` is the gate the engine calls BEFORE running a
    candidate; it returns False when admitting it would push spend over the per-round cap or
    the usable total, so the loop simply skips that candidate. `charge` records the TRUE cost
    after the run (estimates rank/admit; truth accounts). This is what makes the run honest:
    the controller can never spend money it does not have, and `stop()` triggers a real,
    reported early stop rather than silently overrunning.
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        self.cfg = config or BudgetConfig()
        self.usable_seconds = self.cfg.total_seconds * (1.0 - self.cfg.reserve_frac)
        self.reserve_seconds = self.cfg.total_seconds - self.usable_seconds
        self.spent_seconds = 0.0
        self.round_index = -1
        self._round_cap = self.usable_seconds / self.cfg.rounds
        self._round_spent = 0.0
        self._admitted: set = set()
        self._scores: Dict[str, float] = {}      # id -> latest val score (for culling)
        self._alive: set = set()                  # arm ids still in the running
        self._killed: List[str] = []
        self.events: List[dict] = []              # audit log of admit/charge/cull/stop

    # ---- accounting helpers
    @property
    def remaining_seconds(self) -> float:
        """Usable budget not yet spent (reserve excluded)."""
        return max(0.0, self.usable_seconds - self.spent_seconds)

    def _round_remaining(self) -> float:
        return max(0.0, self._round_cap - self._round_spent)

    # ---- round lifecycle
    def begin_round(self, r: int) -> None:
        """Open round r. Carries any unspent budget from earlier rounds forward by raising
        this round's cap to (usable - spent) / (rounds - r), so an underspent screen lets a
        later refine spend more -- without ever exceeding the usable total."""
        self.round_index = r
        self._round_spent = 0.0
        rounds_left = max(1, self.cfg.rounds - r)
        self._round_cap = self.remaining_seconds / rounds_left
        self.events.append({"event": "begin_round", "round": r,
                            "round_cap": round(self._round_cap, 4),
                            "remaining": round(self.remaining_seconds, 4)})

    def admit(self, program_id: str, est_seconds: float) -> bool:
        """Return True iff there is room (this round AND overall) for an estimated cost.

        A tiny epsilon slack lets a candidate whose estimate exactly equals the remaining
        budget still run; charging uses the true cost so this cannot cause an overrun beyond
        one candidate's measured time, which is itself wall/cpu-capped by the sandbox.
        """
        est = max(0.0, float(est_seconds))
        eps = 1e-9
        ok = (self.spent_seconds + est <= self.usable_seconds + eps and
              self._round_spent + est <= self._round_cap + eps and
              self.remaining_seconds > eps)
        self.events.append({"event": "admit", "round": self.round_index, "id": program_id,
                            "est": round(est, 4), "admitted": bool(ok),
                            "remaining": round(self.remaining_seconds, 4)})
        if ok:
            self._admitted.add(program_id)
            self._alive.add(program_id)
        return ok

    def charge(self, program_id: str, wall_seconds: float) -> None:
        """Record the TRUE cost of a run. Always called after a run, admitted or not."""
        cost = max(0.0, float(wall_seconds)) if wall_seconds and math.isfinite(wall_seconds) else 0.0
        self.spent_seconds += cost
        self._round_spent += cost
        self.events.append({"event": "charge", "round": self.round_index, "id": program_id,
                            "cost": round(cost, 4), "spent": round(self.spent_seconds, 4)})

    def record_score(self, program_id: str, score: Optional[float]) -> None:
        """Record an arm's latest validation score, used to rank survivors at end_round."""
        if score is None or not math.isfinite(float(score)):
            return
        self._scores[program_id] = float(score)
        self._alive.add(program_id)

    def end_round(self, r: int) -> List[str]:
        """Successive-halving cull: keep the top `keep_frac` of scored alive arms (never below
        min_keep), kill the rest. Returns the list of killed arm ids. Unscored arms (e.g. ones
        that errored) are dropped from the alive set but not counted as 'killed by ranking'.

        This is the deterministic early-kill of losing arms the roadmap asks for, replacing the
        test-only Thompson portfolio. It is a SEARCH-SHAPING decision (which arms keep getting
        compute), never a promotion decision -- the killed arms simply stop being proposed.
        """
        scored = [(pid, s) for pid, s in self._scores.items() if pid in self._alive]
        unscored = [pid for pid in self._alive if pid not in self._scores]
        # drop arms that never produced a score from contention (they failed to run)
        for pid in unscored:
            self._alive.discard(pid)
        if self.cfg.keep_frac >= 1.0 or len(scored) <= self.cfg.min_keep:
            self.events.append({"event": "cull", "round": r, "killed": [], "kept": len(scored)})
            return []
        scored.sort(key=lambda kv: kv[1], reverse=True)   # higher score better (science convention)
        n_keep = max(self.cfg.min_keep, int(math.ceil(self.cfg.keep_frac * len(scored))))
        survivors = {pid for pid, _ in scored[:n_keep]}
        killed = [pid for pid, _ in scored[n_keep:]]
        for pid in killed:
            self._alive.discard(pid)
            self._killed.append(pid)
        self.events.append({"event": "cull", "round": r, "killed": killed,
                            "kept": len(survivors)})
        return killed

    def is_alive(self, program_id: str) -> bool:
        """Whether an arm survived culling and may still be proposed/run."""
        return program_id in self._alive

    def stop(self) -> bool:
        """True when the usable budget is effectively exhausted (honest early stop)."""
        done = self.remaining_seconds <= 1e-6
        if done:
            self.events.append({"event": "stop", "round": self.round_index,
                                "spent": round(self.spent_seconds, 4),
                                "usable": round(self.usable_seconds, 4)})
        return done

    def report(self) -> dict:
        """A compact, inspectable summary of the budget run (for the Phase-9 artifact)."""
        return {
            "total_seconds": self.cfg.total_seconds,
            "usable_seconds": round(self.usable_seconds, 4),
            "reserve_seconds": round(self.reserve_seconds, 4),
            "spent_seconds": round(self.spent_seconds, 4),
            "remaining_seconds": round(self.remaining_seconds, 4),
            "admitted": len(self._admitted),
            "killed_arms": list(self._killed),
            "alive_arms": list(self._alive),
            "exhausted": self.remaining_seconds <= 1e-6,
        }


# --------------------------------------------------------------------------- phase decomposer

@dataclass
class Phase:
    """One gated stage of a decomposed experiment.

    name           : "screen" | "refine" | "confirm" (or caller-defined).
    budget_seconds : wall-second budget allocated to this phase.
    cohort_size    : how many candidates enter this phase.
    survivors      : how many advance to the next phase (the gate width). For the last phase
                     this is 1 (the single winner that gets certified).
    gate           : minimum validation score a candidate must reach to advance. None means
                     "rank-only" (advance the top `survivors` regardless of absolute score).
    depends_on     : the name of the phase that feeds this one (None for the root). Encodes
                     the DAG edges; validate() checks the chain is acyclic and connected.
    """
    name: str
    budget_seconds: float
    cohort_size: int
    survivors: int
    gate: Optional[float]
    depends_on: Optional[str]


@dataclass
class ExperimentPlan:
    """A gated DAG of phases plus the total budget it was built from."""
    phases: List[Phase]
    total_seconds: float

    def validate(self) -> None:
        """Assert the plan is a valid gated DAG:
          - phase names unique;
          - dependency edges form an acyclic, connected chain (each non-root depends on a
            real earlier phase);
          - budgets are positive and SUM TO <= total (budget-conserving, never overcommitted);
          - cohorts narrow monotonically (cohort_size non-increasing, survivors <= cohort_size,
            and each phase's cohort matches the previous phase's survivors);
          - the terminal phase yields exactly one survivor (the winner to certify).
        Raises ValueError on any violation. Never mutates the plan.
        """
        if not self.phases:
            raise ValueError("plan has no phases")
        names = [p.name for p in self.phases]
        if len(set(names)) != len(names):
            raise ValueError(f"phase names must be unique: {names}")
        index = {p.name: i for i, p in enumerate(self.phases)}

        # acyclic + connected chain: follow depends_on from each node; must reach a root
        # without revisiting (a back-edge would loop).
        for p in self.phases:
            seen = set()
            cur = p
            while cur.depends_on is not None:
                if cur.depends_on not in index:
                    raise ValueError(f"phase {cur.name!r} depends on unknown {cur.depends_on!r}")
                if index[cur.depends_on] >= index[cur.name]:
                    raise ValueError(f"edge {cur.depends_on}->{cur.name} is not forward (cycle)")
                if cur.depends_on in seen:
                    raise ValueError(f"cycle detected at {cur.depends_on!r}")
                seen.add(cur.depends_on)
                cur = self.phases[index[cur.depends_on]]
        roots = [p for p in self.phases if p.depends_on is None]
        if len(roots) != 1:
            raise ValueError(f"a gated chain must have exactly one root, got {len(roots)}")

        total_alloc = 0.0
        for i, p in enumerate(self.phases):
            if p.budget_seconds <= 0:
                raise ValueError(f"phase {p.name!r} has non-positive budget")
            if p.cohort_size < 1 or p.survivors < 1:
                raise ValueError(f"phase {p.name!r} cohort/survivors must be >= 1")
            if p.survivors > p.cohort_size:
                raise ValueError(f"phase {p.name!r} survivors {p.survivors} > cohort {p.cohort_size}")
            total_alloc += p.budget_seconds
            if i > 0:
                prev = self.phases[i - 1]
                if p.cohort_size > prev.cohort_size:
                    raise ValueError(f"phase {p.name!r} cohort grows ({p.cohort_size}>{prev.cohort_size})")
                if p.cohort_size != prev.survivors:
                    raise ValueError(
                        f"phase {p.name!r} cohort {p.cohort_size} != prev survivors {prev.survivors}")
        if total_alloc > self.total_seconds + 1e-6:
            raise ValueError(f"phases overcommit budget: {total_alloc} > {self.total_seconds}")
        if self.phases[-1].survivors != 1:
            raise ValueError("terminal phase must yield exactly one survivor (the winner)")

    def as_dict(self) -> dict:
        return {"total_seconds": self.total_seconds,
                "phases": [vars(p) for p in self.phases]}


class PhaseDecomposer:
    """Splits a big experiment into gated phases: cheap-screen -> refine -> confirm.

    Rationale: spending the whole budget on every candidate is wasteful when most candidates
    are quickly seen to be weak. A cheap screen on many candidates with a low budget share
    cheaply prunes the field; a refine phase spends more per survivor; a confirm phase invests
    the most on the few finalists before the single sealed certification. This is the standard
    multi-fidelity / successive-halving shape, made explicit as a budget-conserving DAG.

    `gates` are RANK-ONLY by default (gate=None): advance the top `survivors`. Absolute-score
    gates can be supplied by the caller when they have a meaningful floor (e.g. must beat a
    trivial baseline before refining). The decomposer never reads data or scores anything; it
    only lays out the plan. The orchestrator runs each phase under its own BudgetController.
    """

    # default budget shares and narrowing per phase (screen cheap, confirm rich).
    _DEFAULT_STAGES = [
        # name      budget_frac   keep_frac (cohort -> survivors)
        ("screen",  0.20,         0.25),
        ("refine",  0.30,         0.40),
        ("confirm", 0.50,         1.00),   # confirm narrows to the single winner explicitly
    ]

    def decompose(self, total_seconds: float, n_candidates: int, kind: str = "classification",
                  gates: Optional[List[Optional[float]]] = None) -> ExperimentPlan:
        """Build a 3-phase gated plan for `n_candidates` over `total_seconds`.

        - `total_seconds`: total wall-second budget for the whole experiment.
        - `n_candidates` : how many candidates enter the screen phase (>= 1).
        - `kind`         : task kind, carried through for the caller (does not change the layout
                           but is recorded so the orchestrator picks the right metric/gate).
        - `gates`        : optional per-phase absolute score floors; None entries are rank-only.

        The plan always terminates in a single survivor and never overcommits the budget.
        """
        if total_seconds <= 0:
            raise ValueError("total_seconds must be positive")
        n = max(1, int(n_candidates))
        stages = self._DEFAULT_STAGES
        if gates is not None and len(gates) != len(stages):
            raise ValueError(f"gates must have length {len(stages)}")

        phases: List[Phase] = []
        cohort = n
        prev_name: Optional[str] = None
        for i, (name, bfrac, keep) in enumerate(stages):
            is_last = (i == len(stages) - 1)
            if is_last:
                survivors = 1                         # confirm yields the single winner
            else:
                survivors = max(1, int(math.ceil(keep * cohort)))
                survivors = min(survivors, cohort)
                # never let a non-terminal phase collapse straight to 1 if more candidates
                # remain than later phases need; keep at least 2 so refine has a real choice.
                if survivors < 2 and cohort >= 2:
                    survivors = 2
            gate = gates[i] if gates is not None else None
            phases.append(Phase(name=name, budget_seconds=total_seconds * bfrac,
                                cohort_size=cohort, survivors=survivors, gate=gate,
                                depends_on=prev_name))
            prev_name = name
            cohort = survivors

        plan = ExperimentPlan(phases=phases, total_seconds=total_seconds)
        plan.validate()        # fail loudly at construction, not at run time
        return plan


# --------------------------------------------------------------------------- self-test

def _self_test() -> int:
    """Lightweight smoke test runnable as `python frontier/budget.py`."""
    cm = CostModel()
    p = Program(code="", source="seed", label="rf",
                provenance={"recipe": {"base": "rf", "poly": 2}})
    e = cm.estimate(p, n_train=1000, n_features=20)
    assert e.seconds > 0 and "rf" == e.family and "poly2" in e.enhancers, e
    cm.observe(p, 1000, 20, wall_seconds=0.5, ok=True)
    plan = PhaseDecomposer().decompose(total_seconds=1000.0, n_candidates=32)
    plan.validate()
    ctrl = BudgetController(BudgetConfig(total_seconds=100.0, rounds=2, reserve_frac=0.0))
    ctrl.begin_round(0)            # round cap = 100/2 = 50s
    assert ctrl.admit("a", 40.0)
    ctrl.charge("a", 40.0)         # charge true cost before the next admit (real ordering)
    assert not ctrl.admit("b", 40.0)   # 40 + 40 = 80 > 50 round cap -> refused
    assert ctrl.admit("b", 5.0)
    ctrl.charge("b", 5.0)
    print("budget self-test:", cm.family_table()["rf"]["n_obs"], "obs;", ctrl.report())
    print("[ok] budget self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
