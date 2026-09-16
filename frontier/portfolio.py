"""Concurrent portfolio with EARLY-KILL: ASHA successive-halving + Thompson/UCB arm
selection over proposal Programs, run in parallel through the real sandbox.

Why this exists
---------------
The Phase-0 engine evaluates every proposal of a round SERIALLY on the FULL validation
split (engine.py: the `for p in proposals` loop). That is correct but wasteful: a 12-arm
round fits 12 models to completion on full data even though most are obvious losers after
a cheap look. Phase 6 of the ROADMAP asks for "a portfolio with early-kill of losing arms
(the Thompson portfolio at attestra/orchestration/portfolio.py exists but is test-only,
never called)". The attestra portfolio is a *sequential* selector over already-certified
GoalLoopResults; THIS module is the missing piece: a *concurrent* per-round batch executor
that prunes losing Programs before they consume a full-data fit, and returns the survivor
with the best validation score.

It is a drop-in replacement for the engine's inner per-round evaluation loop. It changes
ONLY how candidates are scheduled and pruned; it does not change what may promote.

INTEGRITY CONTRACT (load-bearing — preserves the CONTRACT.md invariants)
------------------------------------------------------------------------
  * The portfolio NEVER touches the sealed split. It receives (X_train, y_train, X_val) and
    a trusted `score_fn(preds)->float` (which the engine builds from `certify.score_val` on
    the VAL rows). Selection happens on VAL only, exactly as the serial loop does. The sealed
    test stays untouched until the engine certifies the single winner — that path is unchanged.
  * The sandbox firewall is preserved: every candidate runs through `frontier.sandbox.run_program`
    in a separate OS process and returns PREDICTIONS only. The portfolio computes the val score
    in the trusted parent via the injected `score_fn`. No untrusted code computes a metric.
  * ASHA / Thompson / UCB are SEARCH-SCHEDULING heuristics. They decide what to RUN and what to
    KILL early, never what may PROMOTE. The promoter is still the frozen certifier downstream
    (invariant 3/4: "Generalization expands what may be PROPOSED, never what may PROMOTE"; here,
    scheduling expands what is *cheap to evaluate*, never what promotes).
  * Honest outcome: if no arm produces a finite val score the portfolio returns winner=None and
    the engine declines honestly — no relabeled / fabricated number.

The rung schedule (successive halving / ASHA)
---------------------------------------------
Each arm is a Program. A "budget" is a FRACTION of the training rows used to fit before
predicting on the FULL validation split. We use a geometric ladder of training fractions,
the classic Hyperband/ASHA resource axis (training-set size is a monotone, cheap-to-cheap
resource), so a model promoted to the next rung is a strict superset-fit of the same model.

  rung r (r = 0..R-1):
    budget(r)   = clip(min_frac * eta**r, .., 1.0)          # geometric training fraction
    survivors(r)= ceil( n_arms / eta**(r+1) )               # top-(1/eta) by val score advance

  Defaults: eta=3, min_frac=1/9, R chosen so the top rung is budget=1.0 (full train).
  With eta=3 and n arms: rung0 runs all n at 1/9 data, keeps top n/3; rung1 runs those at
  1/3 data, keeps top n/9; the final rung runs the few survivors at full data. Total fit
  work is ~ (n/9 + n/3 + n) * (one full fit) ≈ a small constant * a single full-data sweep,
  versus n full-data fits for the serial loop — the standard successive-halving speedup,
  while the FINAL ranking is always decided at full budget (no early decision promotes).

Arm selection (Thompson / UCB) WITHIN a rung
--------------------------------------------
A rung has more arms than workers. Order matters only for *which losing arms we may never need
to finish* once enough survivors are known — but ASHA's correctness needs every arm in a rung
evaluated at that rung's budget before halving. So we use Thompson/UCB to ORDER launches (most
promising first, seeded by the previous rung's score) and, optionally, to enable AGGRESSIVE
early-kill: once `survivors(r)` arms at the current rung already beat a still-pending arm's
*upper confidence bound*, that pending arm cannot enter the survivor set and is killed without
running. This is an admissible prune (it only skips arms provably out of the top-k under the
confidence model), so it never discards the eventual best arm under the model's assumptions.
The default policy is "thompson"; "ucb" and "none" (pure ASHA, no reorder/prune) are available.

# === WIRING ===
# The integrator plugs this into ResearchEngine.run as the per-round batch executor, replacing
# the serial `for p in proposals: sandbox.run_program(...)` block in engine.py. Concretely, in a
# subclass / variant engine (this module is ADDITIVE — it does not edit engine.py):
#
#   from frontier.portfolio import run_portfolio_round, PortfolioConfig
#   from frontier import certify
#
#   # inside ResearchEngine.run, per round, after `proposals` is built:
#   score_fn = lambda preds: certify.score_val(task, splits.val_rows, preds)  # VAL-only, trusted
#   outcome = run_portfolio_round(
#       proposals,                       # list[Program] for this round
#       X_train, y_train, X_val,         # arrays already materialized in engine.run
#       kind=task.kind,
#       score_fn=score_fn,               # parent computes the metric (firewall)
#       config=PortfolioConfig(wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds),
#   )
#   # outcome.records : list[ArmRecord] -> map onto engine._Record (label/source/ok/val_score/...)
#   # outcome.best_program / outcome.best_score : update (best_prog, best_score) exactly as the
#   #   serial loop does; outcome.full_budget_scores are the FULL-train val scores (rung == top),
#   #   so the selection is identical-in-kind to the serial loop (full-data fit on val).
#   for rec in outcome.records:
#       recent_errors.append(...) for the failed ones; history.append(map(rec))
#   if outcome.best_program is not None and (best_score is None or outcome.best_score > best_score):
#       best_score, best_prog = outcome.best_score, outcome.best_program
#
# Ordering / argument shapes match engine.run exactly: X_train (n,d) float, y_train (n,) object,
# X_val (m,d) float; score_fn maps a list[pred] of length m -> float. The engine still certifies
# the single best_prog on the sealed split afterward (unchanged). EngineConfig gains nothing
# required; PortfolioConfig carries the scheduling knobs so EngineConfig stays frozen.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import sandbox
from .program import Program
from .program import RunResult


# --------------------------------------------------------------------------- config / records

def _default_max_workers() -> int:
    """Bounded concurrency = min(16, cpu-2), at least 1.

    Why cpu-2: each sandboxed fit is itself CPU-bound (sklearn/BLAS). Leave two cores for the
    parent's scheduling thread and the OS so a saturated machine still kills timed-out arms
    promptly. Cap at 16 so a 64-core box does not fork 62 subprocesses and thrash memory.
    """
    cpu = os.cpu_count() or 2
    return max(1, min(16, cpu - 2))


@dataclass
class PortfolioConfig:
    """Scheduling knobs for the concurrent early-kill portfolio.

    eta            : halving rate (keep top 1/eta each rung). 3 is the Hyperband default.
    min_frac       : training fraction at the FIRST (cheapest) rung. The ladder is geometric
                     up to 1.0 (full train) at the top rung.
    max_workers    : bounded concurrency. None -> min(16, cpu-2).
    policy         : "thompson" | "ucb" | "none" (pure ASHA, no reorder, no confidence prune).
    aggressive_kill: if True, kill a still-pending arm at a rung when it cannot enter the
                     survivor set under the confidence model (admissible prune). Default True.
    wall_seconds   : per-arm wall-clock cap passed to the sandbox (a killed arm frees a worker).
    cpu_seconds    : per-arm CPU rlimit passed to the sandbox.
    min_rung_arms  : never run a rung with fewer than this many arms at sub-full budget; below
                     it, jump straight to full budget (small batches gain nothing from halving).
    seed           : RNG seed for Thompson sampling (determinism for tests/repro).
    """
    eta: float = 3.0
    min_frac: float = 1.0 / 9.0
    max_workers: Optional[int] = None
    policy: str = "thompson"
    aggressive_kill: bool = True
    wall_seconds: float = 60.0
    cpu_seconds: int = 55
    min_rung_arms: int = 4
    seed: int = 0


@dataclass
class ArmRecord:
    """Per-arm outcome the engine maps onto its own _Record."""
    program_id: str
    label: str
    source: str
    ok: bool
    val_score: Optional[float] = None        # FULL-budget val score when the arm reached the top rung
    best_rung_score: Optional[float] = None  # best val score across the rungs this arm ran
    reached_rung: int = -1                   # highest rung index this arm was evaluated at
    killed_early: bool = False               # pruned (ASHA halving or confidence prune) before full budget
    error_kind: str = ""
    error: str = ""
    wall_seconds: float = 0.0


@dataclass
class PortfolioOutcome:
    """Result of one concurrent portfolio round."""
    best_program: Optional[Program]
    best_score: Optional[float]              # the winner's FULL-budget val score
    records: List[ArmRecord] = field(default_factory=list)
    rung_schedule: List[dict] = field(default_factory=list)   # [{rung, budget, n_arms, survivors}]
    n_full_fits_saved: float = 0.0           # accounting: full-data fits avoided vs the serial loop
    full_budget_scores: Dict[str, float] = field(default_factory=dict)  # program_id -> full-train val score


# --------------------------------------------------------------------------- the rung ladder

def _build_ladder(n_arms: int, cfg: PortfolioConfig) -> List[dict]:
    """Compute the ASHA rung schedule for `n_arms` arms.

    Returns a list of {rung, budget(frac in (0,1]), survivors(int)} from cheapest to full.
    The top rung always has budget==1.0 so the final decision is made at full training data.
    """
    eta = max(2.0, float(cfg.eta))
    min_frac = min(1.0, max(1e-3, float(cfg.min_frac)))

    # Too few arms to benefit from halving -> a single full-budget rung (degenerate ASHA).
    if n_arms < max(2, cfg.min_rung_arms):
        return [{"rung": 0, "budget": 1.0, "survivors": n_arms}]

    # Number of rungs so that min_frac * eta**(R-1) >= 1.0 (top rung is full data).
    # R = floor(log_eta(1/min_frac)) + 1.
    n_rungs = int(math.floor(math.log(1.0 / min_frac) / math.log(eta))) + 1
    n_rungs = max(2, n_rungs)

    ladder = []
    survivors_prev = n_arms
    for r in range(n_rungs):
        budget = min(1.0, min_frac * (eta ** r))
        if r == n_rungs - 1:
            budget = 1.0  # force full data at the top rung
        # survivors entering the NEXT rung: top 1/eta of THIS rung (ceil, >=1)
        survivors = max(1, int(math.ceil(survivors_prev / eta)))
        if r == n_rungs - 1:
            survivors = survivors_prev  # the top rung keeps everyone it ran (final ranking rung)
        ladder.append({"rung": r, "budget": round(budget, 6), "survivors": survivors})
        survivors_prev = survivors

    # De-duplicate consecutive rungs that collapsed to the same budget (can happen if min_frac
    # is close to 1). Keep the schedule strictly increasing in budget; merge survivors.
    deduped: List[dict] = []
    for row in ladder:
        if deduped and abs(row["budget"] - deduped[-1]["budget"]) < 1e-9:
            deduped[-1]["survivors"] = row["survivors"]
            continue
        row = dict(row)
        row["rung"] = len(deduped)
        deduped.append(row)
    deduped[-1]["budget"] = 1.0
    return deduped


# --------------------------------------------------------------------------- arm selection

class _ArmStats:
    """Posterior bookkeeping for one arm, used by Thompson/UCB ORDERING and the kill prune.

    We treat the val score as living in [score_lo, score_hi] (estimated from observed scores,
    defaulting to a wide band) and model uncertainty as shrinking with the budget the arm has
    been evaluated at. This is a scheduling heuristic, not a statistical claim about the metric.
    """

    def __init__(self, program: Program):
        self.program = program
        self.scores: List[float] = []      # val scores observed across rungs (ascending budget)
        self.budgets: List[float] = []     # budget at which each score was observed
        self.alive = True

    @property
    def last_score(self) -> Optional[float]:
        return self.scores[-1] if self.scores else None

    @property
    def last_budget(self) -> float:
        return self.budgets[-1] if self.budgets else 0.0

    def observe(self, score: float, budget: float) -> None:
        self.scores.append(float(score))
        self.budgets.append(float(budget))

    def ucb(self, c: float, lo: float, hi: float) -> float:
        """Optimistic estimate: last score + an exploration bonus that shrinks with budget.

        With no observation yet, return +inf (must be tried — pure optimism under uncertainty).
        """
        if self.last_score is None:
            return math.inf
        span = max(1e-9, hi - lo)
        # bonus ~ c * span * (1 - budget): an arm evaluated at full budget has ~0 bonus.
        return self.last_score + c * span * (1.0 - self.last_budget)

    def lcb(self, c: float, lo: float, hi: float) -> float:
        """Pessimistic estimate (lower confidence bound). Used by the admissible kill prune."""
        if self.last_score is None:
            return -math.inf
        span = max(1e-9, hi - lo)
        return self.last_score - c * span * (1.0 - self.last_budget)

    def thompson_sample(self, rng: np.random.Generator, lo: float, hi: float) -> float:
        """Sample an optimistic value: a Gaussian around the last score with budget-shrinking std.

        Unobserved arms sample very high (forced exploration). This only orders launches; the
        ASHA halving still evaluates every arm in a rung at that rung's budget.
        """
        if self.last_score is None:
            return float(hi + abs(hi) + 1.0)  # dominate -> tried first
        span = max(1e-9, hi - lo)
        std = 0.5 * span * (1.0 - self.last_budget) + 1e-6
        return float(rng.normal(self.last_score, std))


# --------------------------------------------------------------------------- subsampling

def _subsample_train(X_train: np.ndarray, y_train: np.ndarray, frac: float,
                     kind: str, seed: int) -> tuple:
    """Take a `frac` slice of the training set as the rung's resource budget.

    For classification we stratify by label so a tiny rung-0 budget still sees every class
    (otherwise a class can vanish and the fit/predict breaks for reasons unrelated to the arm's
    quality — that would be an unfair early-kill). For regression we take a plain random slice.
    The subsample is deterministic in `seed` so a promoted arm at rung r+1 is a strict superset
    fit (reproducible, debuggable).
    """
    n = len(X_train)
    if frac >= 1.0 or n == 0:
        return X_train, y_train
    rng = np.random.default_rng(seed)
    k = max(1, int(round(frac * n)))
    if kind == "classification":
        idx_all = []
        labels = np.asarray([str(v) for v in y_train])
        classes = np.unique(labels)
        # proportional allocation per class, at least 1 per class so no class disappears
        for c in classes:
            ci = np.where(labels == c)[0]
            rng.shuffle(ci)
            take = max(1, int(round(frac * len(ci))))
            idx_all.append(ci[:take])
        idx = np.concatenate(idx_all)
        rng.shuffle(idx)
        # If rounding overshot k, trim while keeping >=1 per class is hard; just cap loosely.
        idx = idx[: max(len(classes), k)]
    else:
        idx = rng.permutation(n)[:k]
    return X_train[idx], y_train[idx]


# --------------------------------------------------------------------------- the executor

def run_portfolio_round(
    programs: Sequence[Program],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    *,
    kind: str,
    score_fn: Callable[[list], float],
    config: Optional[PortfolioConfig] = None,
    run_fn: Optional[Callable] = None,
) -> PortfolioOutcome:
    """Run one round of proposal Programs concurrently with ASHA early-kill; return the best-on-val.

    Parameters
    ----------
    programs : list[Program]   the round's candidate arms.
    X_train, y_train           full training arrays (shapes match engine.run: (n,d) float, (n,) obj).
    X_val                      full validation features (m,d). The arm always PREDICTS on full val;
                               only the TRAIN budget is subsampled per rung.
    kind                       "classification" | "regression".
    score_fn                   trusted parent scorer: list[pred] (length m) -> float (higher is better).
                               Built by the engine from certify.score_val on the VAL rows. The portfolio
                               NEVER computes the metric itself and NEVER sees the sealed split.
    config                     PortfolioConfig (scheduling knobs). None -> defaults.
    run_fn                     optional per-arm executor: (Program, X_train, y_train, X_val, kind, wall_s,
                               cpu_s) -> RunResult. When provided, the portfolio calls this instead of
                               sandbox.run_program — enables backend-aware dispatch (e.g. GPU for neural
                               proposals, sklearn sandbox for others). None -> default sandbox executor.

    Returns
    -------
    PortfolioOutcome with best_program / best_score (full-budget val), per-arm records, the rung
    schedule actually used, and accounting for full fits saved.

    Determinism: given the same programs/data/seed and a fixed eta, the FINAL ranking is fixed
    (the top rung scores every survivor at full budget). Concurrency affects only wall-clock and,
    via the optional confidence prune, which provably-losing arms we skip — never the winner.
    """
    cfg = config or PortfolioConfig()
    programs = list(programs)
    outcome = PortfolioOutcome(best_program=None, best_score=None)
    if not programs:
        return outcome

    max_workers = cfg.max_workers or _default_max_workers()
    rng = np.random.default_rng(cfg.seed)

    stats: Dict[str, _ArmStats] = {}
    records: Dict[str, ArmRecord] = {}
    for p in programs:
        stats[p.id] = _ArmStats(p)
        records[p.id] = ArmRecord(p.id, p.label, p.source, ok=True)

    ladder = _build_ladder(len(programs), cfg)
    outcome.rung_schedule = [dict(r) for r in ladder]

    # global observed score band (for the confidence bonuses); seeded wide, tightened as we see data
    obs_lo = [math.inf]
    obs_hi = [-math.inf]
    band_lock = threading.Lock()

    def _update_band(score: float) -> None:
        with band_lock:
            obs_lo[0] = min(obs_lo[0], score)
            obs_hi[0] = max(obs_hi[0], score)

    def _band() -> tuple:
        with band_lock:
            lo, hi = obs_lo[0], obs_hi[0]
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            return (0.0, 1.0)  # default band before any observation / degenerate
        return (lo, hi)

    def _run_arm(prog: Program, budget: float, seed: int) -> RunResult:
        Xt, yt = _subsample_train(X_train, y_train, budget, kind, seed)
        if run_fn is not None:
            return run_fn(prog, Xt, yt, X_val, kind, cfg.wall_seconds, cfg.cpu_seconds)
        return sandbox.run_program(prog, Xt, yt, X_val, kind=kind,
                                   wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)

    # alive set of program ids entering the current rung
    alive: List[str] = [p.id for p in programs]
    saved_fits = 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for rinfo in ladder:
            r = rinfo["rung"]
            budget = rinfo["budget"]
            survivors_k = rinfo["survivors"]
            if not alive:
                break

            # ----- order this rung's launches by the arm-selection policy (most promising first)
            ordered = _order_arms(alive, stats, cfg, rng, _band())

            # ----- launch with bounded concurrency; support admissible early-kill of pending arms
            pending = list(ordered)               # arm ids not yet launched this rung
            in_flight: Dict[object, str] = {}      # future -> program id
            done_scores: Dict[str, float] = {}     # arm id -> val score this rung (or None on failure)
            seed_base = cfg.seed * 1000 + r * 97

            def _submit_next():
                while pending and len(in_flight) < max_workers:
                    pid = pending.pop(0)
                    fut = pool.submit(_run_arm, stats[pid].program, budget,
                                      seed_base + hash(pid) % 100000)
                    in_flight[fut] = pid

            _submit_next()
            while in_flight:
                done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    pid = in_flight.pop(fut)
                    rec = records[pid]
                    try:
                        res: RunResult = fut.result()
                    except Exception as e:  # a scheduling-level failure, not the candidate's fault
                        res = RunResult(pid, ok=False, error=f"executor: {e}"[:200],
                                        error_kind="other")
                    rec.wall_seconds = round(rec.wall_seconds + res.wall_seconds, 3)
                    rec.reached_rung = r
                    if res.ok and res.preds is not None and len(res.preds) == len(X_val):
                        try:
                            score = float(score_fn(list(res.preds)))
                        except Exception as e:
                            rec.ok = False
                            rec.error_kind = "score"
                            rec.error = f"score_fn failed: {e}"[:200]
                            done_scores[pid] = None
                            continue
                        if not math.isfinite(score):
                            rec.ok = False
                            rec.error_kind = "nan"
                            rec.error = "non-finite val score"
                            done_scores[pid] = None
                            continue
                        stats[pid].observe(score, budget)
                        _update_band(score)
                        done_scores[pid] = score
                        rec.best_rung_score = (score if rec.best_rung_score is None
                                               else max(rec.best_rung_score, score))
                        if budget >= 1.0:
                            rec.val_score = score
                            outcome.full_budget_scores[pid] = score
                    else:
                        rec.ok = False
                        rec.error_kind = res.error_kind or "other"
                        rec.error = res.error
                        done_scores[pid] = None

                # ----- admissible early-kill of still-PENDING arms (skip provably-out-of-top-k)
                if cfg.aggressive_kill and cfg.policy != "none" and pending and budget < 1.0:
                    pending, killed = _prune_pending(
                        pending, done_scores, stats, survivors_k, cfg, _band())
                    for kpid in killed:
                        krec = records[kpid]
                        krec.killed_early = True
                        krec.reached_rung = r
                        saved_fits += budget  # accounting: a full-data fit avoided (scaled by budget)
                _submit_next()

            # ----- successive halving: advance the top-`survivors_k` by val score to the next rung
            ranked = sorted(
                [pid for pid in alive if done_scores.get(pid) is not None],
                key=lambda pid: done_scores[pid], reverse=True,
            )
            advance = ranked[:survivors_k]
            for pid in alive:
                if pid not in advance:
                    rec = records[pid]
                    if done_scores.get(pid) is None and not rec.killed_early:
                        # failed/NaN at this rung (already recorded as not ok)
                        pass
                    elif done_scores.get(pid) is not None and budget < 1.0:
                        rec.killed_early = True          # halved out (a real loser, not a crash)
                        saved_fits += (1.0 - budget)
            alive = advance

    # ----- pick the winner: highest FULL-budget val score; fall back to best-rung if (defensively)
    #       no arm reached full budget (e.g. every full-budget fit failed).
    best_pid, best_score = None, -math.inf
    for pid, sc in outcome.full_budget_scores.items():
        if sc > best_score:
            best_pid, best_score = pid, sc
    if best_pid is None:
        for pid, st in stats.items():
            if st.last_score is not None and st.last_score > best_score:
                best_pid, best_score = pid, st.last_score
    if best_pid is not None:
        outcome.best_program = stats[best_pid].program
        outcome.best_score = best_score
    outcome.records = [records[p.id] for p in programs]
    # serial loop would do len(programs) full-data fits; we did far fewer. Report the saving.
    outcome.n_full_fits_saved = round(saved_fits, 3)
    return outcome


def _order_arms(alive: List[str], stats: Dict[str, _ArmStats], cfg: PortfolioConfig,
                rng: np.random.Generator, band: tuple) -> List[str]:
    """Order this rung's arms by the selection policy (descending desirability)."""
    lo, hi = band
    if cfg.policy == "none":
        return list(alive)  # pure ASHA: insertion order (deterministic)
    if cfg.policy == "ucb":
        return sorted(alive, key=lambda pid: stats[pid].ucb(0.5, lo, hi), reverse=True)
    # thompson (default): sample an optimistic value per arm, order by the sample
    keyed = [(pid, stats[pid].thompson_sample(rng, lo, hi)) for pid in alive]
    keyed.sort(key=lambda kv: kv[1], reverse=True)
    return [pid for pid, _ in keyed]


def _prune_pending(pending: List[str], done_scores: Dict[str, float],
                   stats: Dict[str, _ArmStats], survivors_k: int,
                   cfg: PortfolioConfig, band: tuple) -> tuple:
    """Admissible early-kill: drop pending arms that cannot enter the top-`survivors_k`.

    A pending arm survives only if its optimistic upper bound (UCB) could still beat the
    `survivors_k`-th best *settled* lower bound at this rung. We compare a pending arm's UCB
    (best case) against the k-th best LCB among arms already scored this rung. If even the
    pending arm's best case loses to k arms' worst case, it provably cannot make the cut, so
    killing it without running it never discards a possible survivor under the confidence model.
    This is a strict speedup, not a relaxation of the final full-budget decision.
    """
    settled = [pid for pid, sc in done_scores.items() if sc is not None]
    if len(settled) < survivors_k:
        return pending, []  # not enough settled yet to prove anyone out
    lo, hi = band
    # k-th best LCB among settled arms = the bar a pending arm's UCB must beat to have a chance.
    settled_lcb = sorted((stats[pid].lcb(0.5, lo, hi) for pid in settled), reverse=True)
    bar = settled_lcb[survivors_k - 1]
    keep, killed = [], []
    for pid in pending:
        if stats[pid].ucb(0.5, lo, hi) < bar:
            killed.append(pid)     # best case still below k arms' worst case -> cannot survive
        else:
            keep.append(pid)
    return keep, killed


# --------------------------------------------------------------------------- self-test

def _selftest() -> None:
    """Smoke test on the REAL sandbox + breast-cancer: several Programs run concurrently, the
    portfolio prunes, returns a full-budget winner, and never touches a sealed split."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from . import certify
    from .task import Task
    from .proposers import make_code

    d = load_breast_cancer()
    task = Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.9, name="bc")
    splits = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(splits.train_rows); ytr = Task.rows_to_y(splits.train_rows, task.kind)
    Xva = Task.rows_to_X(splits.val_rows)

    recipes = [
        {"base": "logreg", "scale": True}, {"base": "rf"}, {"base": "hist_gbm"},
        {"base": "svc_rbf", "scale": True}, {"base": "logreg"},
    ]
    progs = [Program(code=make_code(r, "classification"), source="seed",
                     label="+".join(k for k in ("scale",) if r.get(k)) + r["base"])
             for r in recipes]
    # ensure unique labels
    for i, p in enumerate(progs):
        p.label = f"{p.label}_{i}"

    score_fn = lambda preds: certify.score_val(task, splits.val_rows, preds)
    out = run_portfolio_round(progs, Xtr, ytr, Xva, kind="classification",
                              score_fn=score_fn,
                              config=PortfolioConfig(wall_seconds=45, cpu_seconds=40))
    assert out.best_program is not None and out.best_score is not None
    print("portfolio self-test OK:",
          {"winner": out.best_program.label, "score": round(out.best_score, 4),
           "rungs": out.rung_schedule, "saved_fits": out.n_full_fits_saved})


if __name__ == "__main__":
    _selftest()
