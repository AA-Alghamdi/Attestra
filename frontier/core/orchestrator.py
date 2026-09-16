"""frontier.core.orchestrator -- THE CORE end-to-end loop (design 04, Option B-delegate).

The CoreOrchestrator is the rich per-round control surface that the FROZEN Phase-0
``ResearchEngine`` does not provide, assembled by COMPOSITION ONLY: it never edits engine.py,
it reuses the frozen ``certify.py`` (the audited sound certifier behind it) for the SINGLE
sealed peek of the winner, and it routes every untrusted candidate through the predictions-only
firewall (Phase-0 ``sandbox`` / the backend-agnostic ``core.execution`` substrate).

================================================================================
=== WIRING ===
================================================================================
HOW THE INTEGRATOR COMPOSES THIS INTO THE SPINE (no Phase-0 edits)

    from frontier.core.orchestrator import CoreOrchestrator, CoreConfig
    orch = CoreOrchestrator(CoreConfig(rounds=4, llm_client=client))  # client may be None
    result = orch.run(goal="classify tumors", X=X, y=y, theta=0.90)
    print(result.summary())          # EngineResult-shaped: certified result OR honest decline

The orchestrator owns the OUTER control flow; the frozen engine owns what it proves correct
(the three-way split discipline + the one-peek certificate). Concretely, per design 04 §0/§1
the pipeline is:

    router.route(goal, X, y, llm)                       # typed front door, BEFORE any split
      -> harness.self_test()  (assert ok)               # trust NOTHING until this passes
      -> harness.adapt(...)  -> Task                     # frozen Task (kind/metric/theta)
      -> certify.make_splits(task)  -> Splits            # the ONE split; sealed partitioned out
      -> knowledge.warm_start (KnowledgeProposer)        # transfer: proposal side only
      == per round ==============================================================
        _build_context (SOLE enrich site) <- diagnosis  # design 04 §2.2
        proposal sources .propose(ctx):
          1. core.authoring.CoreAuthoringProposer        # CORE generative (LLM; [] offline)
          2. features.FeatureProposer/FeatureMutation    # recipe-driven feature/target eng
          3. core.neural NASProposer/LLMArchitect        # neural archs (templates offline)
          4. SeedProposer + MutationProposer  (FLOOR)    # always-on generative floor
          (+ KnowledgeProposer retrieval, round 0)
        dedup by label/id (Phase-0 rule, engine.py:127-130)
        knowledge.rank (LinUCB)                          # ORDERS only; never prunes the floor
        budget.admit (cost model) with a FLOOR GUARANTEE # >=1 seed/mutation always runs
        portfolio.run_portfolio_round (ASHA early-kill)  # firewall: preds-only, VAL-only
        agentic.repair_loop on failures (LLM; offline=deterministic)
        certify.score_val (TRUSTED PARENT) -> champion   # design 04 §2.6
        diagnosis.diagnose -> next round's enrich        # design 04 §2.7
        knowledge.record_outcome (VAL gain)              # selection-side learning
        checkpoint.save (resumable)
      ===========================================================================
      champion -> certify ONCE on sealed (the ONLY peek) # design 04 §2.9
      -> oracles.verify_before_promote ANDED into certified  # design 04 §2.8 / INTEGRATION
      -> knowledge.record + report.build_report

ONE SEALED PEEK, OWNED HERE. The orchestrator is the ONLY holder of ``splits``. It hands
sub-modules ``train_rows``/``val_rows`` and the derived ``X_train``/``X_val``; it NEVER passes
``sealed_rows`` to a proposer, ranker, portfolio, budget, diagnosis, or knowledge call. The
single ``certify.certify_on_sealed`` call site lives in ``_certify_winner`` and is reached only
after the oracle decision; the orchestrator counts that call (``self.sealed_peeks``) and the
end-to-end test asserts it equals 1 and that ``certificate["peeks"] == 1``.

ORACLE ORDERING (honest reconciliation of design 04 §2.8 with the real oracles.py). Design 04
§2.8 sketches an idealized oracle gate that runs with ``cert=None`` BEFORE the peek. The real
``frontier.oracles.verify_before_promote`` is a post-certify VETO: it needs the certificate (it
re-derives the sealed digest and compares the observed score to the trivial floor), exactly as
INTEGRATION.md specifies. Design 04 §0 (B-embed note) and §6 (Risk 3) explicitly acknowledge
this post-hoc reality. We therefore follow the task's stated order and INTEGRATION.md: certify
the winner ONCE, then AND ``verdict.promote`` into ``certified``. This preserves the one-peek
invariant because (a) the winner's certificate is the only ``certify_on_sealed`` on the winner's
split, and (b) the oracle's reproducibility check builds its OWN fresh split with its OWN peek
budget (oracles.py:331), never re-reading the winner's SealedTest. The oracle gate is genuinely
active: a refuted winner flips to an honest decline with the verdict reason, and ``certified``
is False with the un-promoted certificate carried for the report.

FIREWALL (design 04 §2.11) holds at four crossings, none of which this module weakens:
  1. sklearn execution  -> Phase-0 sandbox / core.execution child (writes preds.npy only);
  2. torch execution    -> core.neural sandbox (same RunResult preds-only contract);
  3. VAL scoring        -> certify.score_val, in THIS parent;
  4. SEALED scoring     -> certify.certify_on_sealed, in THIS parent, one peek.
No proposer/ranker/repair/oracle ever receives a number this parent did not compute; the report
only formats numbers already computed.

OFFLINE DEGRADATION (design 04 §4.3). With ``llm_client is None`` the loop reduces to:
harness -> seed+mutation+recipe-feature+neural-template proposals -> LinUCB rank -> portfolio
execute -> diagnose -> certify -> oracles -> report. Every LLM path degrades to a documented
fallback (authoring -> [], repair -> deterministic-then-give-up, neural -> templates). The
``llm_active`` flag is propagated so the report states honestly which paths were live. THIS IS
THE DEFAULT TEST PATH and is exercised for real on a sklearn dataset by demo_core.py and the
test.
================================================================================
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

# Repo root on sys.path so frontier.* / vectorforge.* / vfplatform.* import identically to the
# Phase-0 modules (mirrors frontier/certify.py and frontier/tests/test_spine.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --- FROZEN Phase-0 spine (imported, never edited) ------------------------------------------
from frontier import certify, sandbox                         # noqa: E402
from frontier.task import Task                                # noqa: E402
from frontier.program import Program, RunResult              # noqa: E402
from frontier.proposers import SeedProposer, MutationProposer  # noqa: E402  (the FLOOR)
# EngineConfig/EngineResult/_Record are reused so report.py / summary() stay compatible.
from frontier.engine import EngineResult, _Record            # noqa: E402

# --- THE CORE leaf modules ------------------------------------------------------------------
from frontier.core import execution                          # noqa: E402  (backend-agnostic exec)
from frontier.core.authoring import CoreAuthoringProposer, AuthoringConfig  # noqa: E402
from frontier.core import neural                             # noqa: E402  (neural archs)
# The HONEST sandbox policy front door: advisory AST triage (now covering the file-write escape
# class: ndarray.dump/dumps + scipy.io writers) + probe-derived EnforcedGuarantees stamped on every
# RunResult. It WRAPS the frozen Phase-0 run_program (injected, never edited) and preserves the
# predictions-only firewall. Wiring it here makes the gate ACTUALLY run before each untrusted exec.
from frontier.core.sandbox_policy import (                   # noqa: E402
    SandboxPolicy, Tier, enforced_of,
)
# Best-effort: load the real Tier-2 (uid+netns) substrate so untrusted execution upgrades from
# resource-only LOCAL to kernel-enforced uid/network isolation ON CAPABLE LINUX HOSTS. The module
# self-registers via register_substrate() IFF it can actually create the namespaces here; on macOS or
# a host without unshare/setpriv it is a no-op and the policy keeps degrading honestly to LOCAL. We
# import it in the production entry point (not in sandbox_policy itself) so the unit tests that drive
# SandboxPolicy directly stay hermetic and control their own substrate registration.
try:                                                        # noqa: E402
    from frontier.core import sandbox_linux as _sandbox_linux  # noqa: F401,E402
except Exception:                                           # pragma: no cover - never block startup
    _sandbox_linux = None

# --- platform layer (additive modules; all degrade honestly offline) ------------------------
from frontier import features                                # noqa: E402
from frontier import diagnosis as diag_mod                   # noqa: E402
from frontier import agentic                                 # noqa: E402
from frontier import oracles                                 # noqa: E402
from frontier import knowledge                               # noqa: E402
from frontier import portfolio                               # noqa: E402
from frontier import budget as budget_mod                    # noqa: E402
from frontier import report as report_mod                    # noqa: E402
from frontier import checkpoint as checkpoint_mod            # noqa: E402
from frontier import baseline as baseline_mod                # noqa: E402  (baseline-first floor + gate)
from frontier import llm_diagnosis as llm_diag_mod           # noqa: E402  (LLM-powered diagnosis)
from frontier import intelligence as intel_mod               # noqa: E402  (search intelligence layer)
from frontier.harness import router                          # noqa: E402


# ============================================================================================
# Configuration
# ============================================================================================

@dataclass
class CoreConfig:
    """Knobs for the orchestrated loop. Defaults are honest, conservative, and LLM-free.

    ``llm_client`` is a ``Callable[[str], str]`` (prompt -> code) or None. None is a first-class,
    honest mode: the LLM paths return [] / deterministic fallbacks and ``llm_active`` reports
    False. NOTHING in this config is tuned to any target benchmark (scientific-integrity rule);
    they are plumbing defaults, not modeling choices.
    """
    rounds: int = 4
    seed: int = 0
    test_frac: float = 0.30
    val_frac: float = 0.20
    wall_seconds: float = 60.0
    cpu_seconds: int = 55
    llm_client: Optional[Callable[[str], str]] = None
    total_seconds: float = 3600.0            # global compute budget (budget.BudgetController)
    n_authored: int = 3                      # authored candidates per round (LLM path)
    neural_param_budget: int = 2_000_000     # hardware-derived ceiling; NOT metric-reversed
    enable_neural: bool = True               # neural proposers on (templates offline)
    enable_knowledge: bool = True            # warm-start retrieval + LinUCB ranking
    enable_intelligence: bool = True          # search intelligence (literature, ASHA, features, evolution)
    kb_path: Optional[str] = None            # durable JSONL KB; None -> ephemeral in-memory KB
    checkpoint_path: Optional[str] = None    # resumable round-state; None -> no on-disk ckpt
    report_dir: Optional[str] = None         # if set, write_report persists artifacts here
    portfolio_policy: str = "thompson"       # ASHA reorder policy ("thompson"|"ucb"|"none")
    floor_min_seeds: int = 1                 # FLOOR GUARANTEE: >=N seed/mutation arms per round
    max_repairs_per_round: int = 3           # cap repair attempts per round to avoid bottleneck
    repair_budget_seconds: float = 30.0      # max total wall-clock time for repairs per round
    max_diagnosis_turns: int = 3             # multi-turn diagnosis refinement cap
    subsample_max_rows: int = 10000          # adaptive subsampling threshold for large data
    eval_max_rows: int = 20000               # cap VAL rows used for selection/bound on huge data (sealed test stays full)
    enable_gpu_dispatch: bool = True         # route neural proposals through GPU backend when available
    on_event: Optional[Callable[[dict], None]] = None  # additive live event sink (dashboards); None -> no-op


# ============================================================================================
# Result
# ============================================================================================

@dataclass
class CoreResult:
    """EngineResult-compatible outcome PLUS the orchestrator's richer provenance.

    The base fields match ``frontier.engine.EngineResult`` exactly so ``report.build_report`` and
    any EngineResult consumer work unchanged (``as_engine_result()`` returns the strict shape).
    Extra fields carry the loop's auditable provenance (oracle verdict, sealed-peek count, KB and
    backend notes) without polluting the frozen schema.
    """
    certified: bool
    certificate: Optional[dict]
    winner: Optional[Program]
    winner_val_score: Optional[float]
    history: List[Any] = field(default_factory=list)
    diagnosis_trail: List[dict] = field(default_factory=list)
    split_meta: dict = field(default_factory=dict)
    decline_reason: str = ""
    llm_active: bool = False
    # --- orchestrator extras (non-promoting provenance) ---
    oracle_verdict: Optional[dict] = None
    sealed_peeks: int = 0
    artifact: Optional[Any] = None
    backend_notes: List[str] = field(default_factory=list)
    self_consistency: Optional[dict] = None
    # --- baseline-first / landscape anchor (non-promoting provenance) ---
    floor_certificate: Optional[dict] = None  # the certified baseline floor (baseline.FloorCertificate.to_dict)
    floor_verdict: Optional[dict] = None      # beats_floor verdict (baseline.FloorVerdict.to_dict)

    def as_engine_result(self) -> EngineResult:
        """Project onto the frozen EngineResult shape (so report.py consumes it verbatim)."""
        return EngineResult(
            certified=self.certified, certificate=self.certificate, winner=self.winner,
            winner_val_score=self.winner_val_score, history=list(self.history),
            diagnosis_trail=list(self.diagnosis_trail), split_meta=dict(self.split_meta),
            decline_reason=self.decline_reason, llm_active=self.llm_active,
        )

    def summary(self) -> str:
        s = self.as_engine_result().summary()
        if self.oracle_verdict is not None:
            s += f"\noracle: promote={self.oracle_verdict.get('promote')} "
            reasons = self.oracle_verdict.get("reasons") or []
            if reasons:
                s += "reasons=" + "; ".join(reasons)
        if self.floor_verdict is not None:
            fv = self.floor_verdict
            s += (f"\nbaseline_floor: beats_floor={fv.get('beats_floor')} "
                  f"winner_lb={fv.get('winner_lower_bound')} "
                  f"floor_lb={fv.get('floor_lower_bound')} ('{fv.get('floor_label')}')")
        s += f"\nsealed_peeks={self.sealed_peeks}"
        return s


# ============================================================================================
# The orchestrator
# ============================================================================================

class CoreOrchestrator:
    """Compose THE CORE + platform layer into one end-to-end loop. No Phase-0 edits."""

    def __init__(self, cfg: Optional[CoreConfig] = None,
                 sandbox_policy: Optional[SandboxPolicy] = None):
        self.cfg = cfg or CoreConfig()
        # Audit counter: the number of times the winner's sealed test was certified. The test
        # and the report assert this is exactly 1 for a certified-or-declined-after-peek run.
        self.sealed_peeks = 0
        # The certified baseline floor for the current run (set in run(); consulted at the final
        # decision site and in declines). None outside a run / when no floor could be certified.
        self._floor = None
        # The HONEST sandbox front door. We request the strongest tier (CONTAINER) but resolve()
        # degrades DOWNWARD to whatever this host actually enforces (LOCAL on macOS, with an honest
        # note) and never claims more. untrusted=True (LLM/mutation-authored code), strict=False so an
        # offline CI run on a non-containerized host still executes under Tier-1 resource isolation with
        # a loud warning rather than refusing. The Phase-0 run_program is INJECTED (never edited). Every
        # untrusted-code crossing in this orchestrator goes through `self.sandbox_policy.run(...)`, so the
        # advisory AST gate (now covering ndarray.dump/dumps + scipy.io writers) runs BEFORE each spawn and
        # the enforced guarantees are stamped onto the RunResult. The firewall (predictions only) is
        # preserved end to end -- the policy never receives labels and never computes a metric.
        self.sandbox_policy = sandbox_policy or SandboxPolicy(
            requested=Tier.CONTAINER, untrusted=True, strict=False,
            local_runner=sandbox.run_program,
        )
        # --- RESUME SEAM (additive; default-None => identical behavior to before) -------------
        # `frontier.longrun.ResumableRun` installs these to make the round loop honor a durable
        # checkpoint WITHOUT recomputing finished rounds. Both are None in the default path, so
        # an un-wrapped orchestrator behaves exactly as the frozen design (no skips, no extra
        # persistence). They are NEVER allowed to influence the single sealed peek (which lives
        # outside the round loop), so the certify-once invariant is untouched.
        #   _round_gate(r) -> (skip: bool, champion: Program|None, champion_score: float|None)
        #     called at the TOP of round r; when skip is True the round body (propose/sandbox/
        #     score) is NOT executed and the carried champion is restored instead.
        #   _on_round_complete(r, best_prog, best_score) -> None
        #     called AFTER a round's body finishes, so the wrapper can checkpoint atomically.
        self._round_gate: Optional[Callable[[int], Tuple[bool, Optional[Program], Optional[float]]]] = None
        self._on_round_complete: Optional[Callable[[int, Optional[Program], Optional[float]], None]] = None

    def _emit(self, stage: str, **data) -> None:
        """Best-effort push onto the configured live event sink. NEVER raises and NEVER touches
        the science loop's control flow: a dashboard hiccup must not perturb a certification run.
        The sink only ever receives VAL/provenance fields, never sealed-test data."""
        cb = self.cfg.on_event
        if cb is None:
            return
        try:
            cb({"stage": stage, **data})
        except Exception:  # noqa: BLE001 - an event sink failure is never fatal
            pass

    # ----------------------------------------------------------------- public entrypoint
    def run(self, goal: str, X, y, *, theta: float = 0.0,
            metric: str = "", name: str = "task") -> CoreResult:
        """Drive the full pipeline for one goal. Returns a certified result or an honest decline.

        Parameters
        ----------
        goal   : free-text research goal (typed by the router; steers risks + LLM authoring).
        X, y   : raw data (router infers modality/kind; harness adapts to a Task).
        theta  : the operator's promotion threshold the sealed LOWER bound must clear. This is
                 the verification standard, supplied by the caller -- NEVER reverse-engineered
                 from any reference solution (scientific-integrity rule). 0.0 is a minimal honest
                 bar (beat chance/mean); pass the real threshold for a real run.
        metric : optional metric override; "" -> the harness picks the right-axis metric.
        name   : task name for provenance/reporting.
        """
        cfg = self.cfg
        llm = cfg.llm_client
        self.sealed_peeks = 0
        backend_notes: List[str] = []

        # Real wall-clock deadline. The BudgetController tracks sandbox-reported
        # COMPUTE seconds, which excludes subprocess spawn overhead, LLM repair
        # calls, scoring, and final certification — so it under-counts true
        # elapsed time and the search can overrun total_seconds. This deadline
        # is the hard wall-clock ceiling; the search loop stops opening new
        # rounds once it is crossed (final certification of the current champion
        # still runs, since that is the mandatory single sealed evaluation).
        run_start = time.time()
        wall_deadline = run_start + float(cfg.total_seconds)

        self._emit("intake", goal=goal, theta=float(theta),
                   llm_active=bool(llm), rounds=cfg.rounds)
        # ===== FRONT MATTER: router -> harness(+self_test, assert ok) -> Task ==================
        harness = router.route(goal, X, y, llm_client=llm)
        # Reward-hacking / infeasibility guard rides on the spec: decline BEFORE any split.
        spec = getattr(harness, "spec", None)
        if spec is not None and getattr(spec, "blocked", False):
            return self._decline(f"router blocked the goal: {spec.risks}", llm_active=bool(llm))

        ok, hcert = harness.self_test()
        # HARD GATE (design 04 §3, INTEGRATION Phase-3): trust nothing until self-test passes.
        if not ok:
            detail = getattr(hcert, "detail", "") if hcert is not None else ""
            return self._decline(
                f"no trusted harness for this task type (self-test failed): {detail}",
                llm_active=bool(llm))

        kind = spec.kind if spec is not None else router.type_problem(goal, X, y).kind
        the_metric = metric or (spec.metric if spec is not None else harness.metric_for(kind))
        task = harness.adapt(X, y, kind=kind, theta=float(theta), name=name, metric=the_metric)
        try:
            tf, vf = harness.split_protocol(kind)
        except TypeError:  # some harnesses take no kind arg
            tf, vf = harness.split_protocol()
        cfg.test_frac, cfg.val_frac = tf, vf

        # ===== the ONE split (sealed partitioned out here, never re-split) =====================
        splits = certify.make_splits(task, seed=cfg.seed, test_frac=cfg.test_frac,
                                     val_frac=cfg.val_frac)
        # Large-data guard: cap the VALIDATION rows that every sandboxed arm must predict on, so
        # peak memory stays bounded regardless of dataset size. This is a uniform random subset of
        # an already-random val split, so the accuracy estimate stays unbiased; a SMALLER n only
        # widens the val bound (more conservative), so it can NEVER make certification easier. The
        # SEALED test (the certificate) is left FULL and untouched.
        if cfg.eval_max_rows and len(splits.val_rows) > cfg.eval_max_rows:
            import numpy as _np
            _n_val_full = len(splits.val_rows)
            _idx = _np.random.default_rng(cfg.seed).choice(
                _n_val_full, size=cfg.eval_max_rows, replace=False)
            splits.val_rows = [splits.val_rows[i] for i in sorted(_idx.tolist())]
            backend_notes.append(
                f"capped val selection set from {_n_val_full} to {len(splits.val_rows)} rows "
                f"(memory-bounded; sealed test unchanged)")
        X_train = Task.rows_to_X(splits.train_rows)
        y_train = Task.rows_to_y(splits.train_rows, task.kind)
        X_val = Task.rows_to_X(splits.val_rows)

        # ===== BASELINE-FIRST: certify a landscape-anchoring floor (its OWN sealed discipline) ==
        #     compute_floor builds an INDEPENDENT split (seed+OFFSET) so the floor's sealed
        #     partition is DISJOINT from the winner's; the floor's sealed test is peeked exactly
        #     once there. This does NOT touch self.sealed_peeks (the winner's one-peek budget):
        #     they are different sealed folds, each honestly counted (see baseline.py docstring).
        #     The final `certified` decision below ANDs beats_floor(winner_cert, floor) so a
        #     winner that clears theta but not the floor is reported as "no improvement over
        #     baseline" -- an honest non-win.
        try:
            floor = baseline_mod.compute_floor(
                task, seed=cfg.seed, test_frac=cfg.test_frac, val_frac=cfg.val_frac,
                wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)
        except Exception as e:  # a floor hiccup must never crash the loop; gate then admits winner
            backend_notes.append(f"baseline floor failed to compute: {e!r}")
            floor = None
        self._floor = floor  # consulted at the final decision site and in declines
        self._emit("split", kind=task.kind, metric=the_metric,
                   n_train=len(splits.train_rows), n_val=len(splits.val_rows),
                   floor_certified=floor is not None,
                   floor_label=(getattr(floor, "label", None) if floor is not None else None))

        # ===== adaptive subsampling for large datasets (B2: progressive per-round) =============
        n_train_original = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
        X_train_full, y_train_full = X_train, y_train
        needs_subsampling = n_train_original > cfg.subsample_max_rows
        subsampled = False
        if needs_subsampling:
            # Initial subsample for pre-loop setup; per-round progressive in loop below
            X_train, y_train, subsampled = intel_mod.adaptive_subsample(
                X_train_full, y_train_full, max_rows=cfg.subsample_max_rows, seed=cfg.seed,
                round_idx=0, total_rounds=cfg.rounds)
            backend_notes.append(
                f"subsampled training data from {n_train_original} to "
                f"{X_train.shape[0]} rows (progressive: will scale up in later rounds)")

        # ===== knowledge warm-start (transfer; PROPOSAL side only) =============================
        know: Optional[knowledge.KnowledgeProposer] = None
        if cfg.enable_knowledge:
            kb = knowledge.KnowledgeBase(cfg.kb_path)
            know = knowledge.KnowledgeProposer(kb)

        # ===== search intelligence (literature, ASHA, features, ensemble, evolution) ===========
        intel_state: Optional[intel_mod.IntelligenceState] = None
        if cfg.enable_intelligence:
            try:
                intel_state = intel_mod.init_intelligence(
                    goal, task, llm_client=llm,
                    enable_literature=True, enable_asha=True,
                    enable_feature_eng=True, enable_ensemble=True,
                    enable_evolution=True)
                if intel_state.literature and intel_state.literature.n_findings > 0:
                    backend_notes.append(
                        f"literature scout found {intel_state.literature.n_findings} findings")
            except Exception as e:
                backend_notes.append(f"intelligence init failed (non-fatal): {e!r}")
                intel_state = None

        # ===== proposal sources: authoring, features, neural, seed/mutation FLOOR ==============
        # Order matters only for tie-breaking in dedup/rank; the FLOOR is always present so a
        # cold task is never starved (invariant 3). LLM sources degrade to [] when llm is None.
        sources: List[Any] = []
        if know is not None:
            sources.append(know)                                     # retrieval warm-start (cold->[])
        sources.append(CoreAuthoringProposer(client=llm,
                                             config=AuthoringConfig(n=cfg.n_authored)))
        sources.append(features.FeatureProposer())
        sources.append(features.FeatureMutationProposer())
        if cfg.enable_neural:
            sources.append(neural.NASProposer(modality=_modality_of(spec),
                                              param_budget=cfg.neural_param_budget,
                                              seed=cfg.seed))
            sources.append(neural.LLMArchitectProposer(client=llm,
                                                       modality=_modality_of(spec),
                                                       param_budget=cfg.neural_param_budget,
                                                       n=2))
        sources.append(SeedProposer())                               # FLOOR
        sources.append(MutationProposer())                           # FLOOR

        # honest llm_active: any wired LLM source (authoring subclasses LLMProposer, so the
        # Phase-0 detection rule also reports True via the engine path if ever embedded).
        llm_active = llm is not None

        # ===== budget controller (bounds compute; never promotes) =============================
        cost = budget_mod.CostModel()
        ctrl = budget_mod.BudgetController(
            budget_mod.BudgetConfig(total_seconds=cfg.total_seconds, rounds=cfg.rounds))

        # ===== loop state =====================================================================
        history: List[_Record] = []
        trail: List[dict] = []
        tried: set = set()
        recent_errors: List[Tuple[str, str, str]] = []
        best_score: Optional[float] = None
        best_prog: Optional[Program] = None
        diag = None
        llm_diag = None
        scored_programs: List[Tuple[Program, float]] = []  # for authoring self-consistency

        # Checkpoint store (resumable). We checkpoint AFTER each round so a crash loses at most
        # the in-flight round. The sealed peek is one-shot and lives OUTSIDE the round loop.
        store = (checkpoint_mod.CheckpointStore(cfg.checkpoint_path)
                 if cfg.checkpoint_path else None)

        score_fn = lambda preds: certify.score_val(task, splits.val_rows, preds)  # noqa: E731

        # ===================== PER-ROUND LOOP (orchestrator owns it) ==========================
        for r in range(cfg.rounds):
            # --- (0-pre) WALL-CLOCK DEADLINE: stop opening new rounds once the real elapsed
            #     time crosses total_seconds. Final certification of the current champion still
            #     runs after the loop (mandatory single sealed evaluation).
            if time.time() >= wall_deadline:
                trail.append({"round": r, "note": "wall-clock budget exhausted; stopping search"})
                break

            # --- (0) RESUME GATE: skip rounds already completed in a prior attempt ------------
            #     When a resume wrapper installed _round_gate, a finished round is NOT recomputed:
            #     its proposal/sandbox/score work is bypassed entirely and the durably-recorded
            #     champion is restored so the end-of-loop certification sees the correct winner.
            #     This is the actual fix for "resume restarted from zero". The gate is None in the
            #     default path, so the loop is byte-identical to before when not wrapped.
            if self._round_gate is not None:
                skip, carried_prog, carried_score = self._round_gate(r)
                if skip:
                    if carried_prog is not None and (best_score is None or
                                                     (carried_score is not None and
                                                      carried_score >= best_score)):
                        best_prog, best_score = carried_prog, carried_score
                    trail.append({"round": r, "note": "resumed: round already completed, skipped"})
                    continue

            ctrl.begin_round(r)
            self._emit("round", round=r, total_rounds=cfg.rounds,
                       best_score=round(best_score, 4) if best_score is not None else None)

            # --- (0b) B2: progressive subsampling — scale up training data in later rounds ----
            if needs_subsampling and r > 0:
                X_train, y_train, subsampled = intel_mod.adaptive_subsample(
                    X_train_full, y_train_full, max_rows=cfg.subsample_max_rows,
                    seed=cfg.seed, round_idx=r, total_rounds=cfg.rounds)

            # --- (1) build + enrich context: the SOLE enrichment site (design 04 §2.2) --------
            ctx = self._build_context(task, splits, r, tried, best_prog, best_score,
                                      recent_errors, know)
            if diag is not None:
                diag_mod.enrich_context(ctx, diag)  # adds context["diagnosis"] etc. (additive)
            if llm_diag is not None:
                llm_diag_mod.enrich_context_with_llm_diagnosis(ctx, llm_diag)
            # Intelligence enrichment: literature, feature hints, knowledge read-back, evolution
            if intel_state is not None:
                try:
                    intel_mod.enrich_round_context(
                        ctx, intel_state, task, r, history, kb_path=cfg.kb_path)
                except Exception as e:
                    backend_notes.append(f"intelligence enrich failed (non-fatal): {e!r}")

            # --- (1b) intelligence proposals: feature eng, ensemble, ASHA -------------------
            pool_pre_intel: List[Program] = []
            if intel_state is not None:
                try:
                    intel_proposals = intel_mod.get_intelligence_proposals(
                        intel_state, ctx, task)
                    for p in intel_proposals:
                        if p.label not in tried:
                            pool_pre_intel.append(p)
                except Exception as e:
                    backend_notes.append(f"intelligence proposals failed (non-fatal): {e!r}")
                    pool_pre_intel = []

            # --- (2) gather + dedup proposals (Phase-0 rule, engine.py:127-130) ---------------
            pool: List[Program] = []
            seen: set = set()
            # Inject intelligence-generated proposals first
            for p in pool_pre_intel:
                if p.label not in tried and p.id not in seen:
                    seen.add(p.id)
                    pool.append(p)
            for src in sources:
                try:
                    out = src.propose(ctx) or []
                except Exception as e:  # a broken source never crashes the loop (it degrades)
                    backend_notes.append(f"round{r}: source {type(src).__name__} raised {e!r}")
                    out = []
                for p in out:
                    if p.label in tried or p.id in seen:
                        continue
                    seen.add(p.id)
                    pool.append(p)
            self._emit("propose", round=r, n_proposals=len(pool),
                       n_intelligence=len(pool_pre_intel),
                       labels=[p.label for p in pool[:12]])
            if not pool:
                trail.append({"round": r, "n_proposals": 0, "note": "no new proposals"})
                break  # Phase-0 rule: no proposals -> stop proposing

            # --- (3) LinUCB rank: ORDERS only, never prunes the floor (invariant 3/4) ---------
            ordered = know.rank(pool, ctx) if know is not None else pool

            # --- (4) budget admit with a FLOOR GUARANTEE (design 04 §6 Risk 2) ----------------
            batch = self._admit_with_floor(ordered, ctrl, cost, task, splits)

            # --- (5) portfolio execute: ASHA early-kill on the chosen backend, VAL-only -------
            #     run_fn is the firewall-safe executor (preds-only). Neural proposals dispatch
            #     through the execution substrate with TORCH_SPEC when GPU is available;
            #     sklearn proposals go through the Phase-0 sandbox as before.
            pconf = portfolio.PortfolioConfig(wall_seconds=cfg.wall_seconds,
                                              cpu_seconds=cfg.cpu_seconds,
                                              policy=cfg.portfolio_policy, seed=cfg.seed)
            gpu_run_fn = self._make_gpu_dispatch(task.kind) if cfg.enable_gpu_dispatch else None
            outcome = portfolio.run_portfolio_round(batch, X_train, y_train, X_val,
                                                    kind=task.kind, score_fn=score_fn,
                                                    config=pconf, run_fn=gpu_run_fn)
            id2prog = {p.id: p for p in batch}

            # --- (6) repair failures (agentic; LLM strong, offline deterministic) -------------
            #     Each failed arm gets ONE bounded repair attempt; a repaired program re-enters
            #     execution exactly once via the firewall executor. A still-failed program is an
            #     honest recorded error fed to diagnosis. NEVER a fabricated success.
            #     Budget-gated: max N repairs per round and T seconds total to prevent bottleneck.
            repairs_this_round = 0
            repair_start_time = time.time()
            for rec in outcome.records:
                tried.add(rec.label)
                prog = id2prog.get(rec.program_id)
                ctrl.charge(rec.program_id, rec.wall_seconds)
                if prog is not None:
                    cost.observe(prog, len(splits.train_rows), task.n_features,
                                 rec.wall_seconds, ok=rec.ok)

                if rec.ok and rec.val_score is not None:
                    self._record_success(history, know, ctx, prog, rec, best_score)
                    if best_score is None or rec.val_score > best_score:
                        best_score, best_prog = rec.val_score, prog
                    ctrl.record_score(rec.program_id, rec.val_score)
                    if prog is not None:
                        scored_programs.append((prog, float(rec.val_score)))
                    # Intelligence outcome tracking (prompt evolution + ensemble history)
                    if intel_state is not None and prog is not None:
                        intel_mod.record_intelligence_outcome(
                            intel_state, prog, float(rec.val_score), True)
                        # B1: feed ensemble proposer with family scores for adaptive composition
                        if intel_state.ensemble_proposer is not None:
                            intel_state.ensemble_proposer.record_family_score(
                                prog.label, float(rec.val_score), task.kind)
                    continue

                # failure path: try one repair if within budget
                repair_budget_exhausted = (
                    repairs_this_round >= cfg.max_repairs_per_round
                    or (time.time() - repair_start_time) >= cfg.repair_budget_seconds
                    or time.time() >= wall_deadline  # never repair past the global deadline
                )
                if repair_budget_exhausted:
                    repaired_prog, repaired_res = None, None
                else:
                    repairs_this_round += 1
                    repaired_prog, repaired_res = self._try_repair(prog, rec, ctx, X_train,
                                                                   y_train, X_val, task)
                if repaired_prog is not None and repaired_res is not None and repaired_res.ok:
                    score = certify.score_val(task, splits.val_rows, repaired_res.preds)
                    tried.add(repaired_prog.label)
                    rrec = _Record(repaired_prog.id, repaired_prog.label, repaired_prog.source,
                                   True, val_score=round(score, 4),
                                   wall_seconds=round(repaired_res.wall_seconds, 2))
                    history.append(rrec)
                    if know is not None:
                        self._kb_record(know, ctx, repaired_prog, score, best_score,
                                        repaired_res.wall_seconds, ok=True)
                    if best_score is None or score > best_score:
                        best_score, best_prog = score, repaired_prog
                    scored_programs.append((repaired_prog, float(score)))
                else:
                    recent_errors.append((rec.label, rec.error_kind, rec.error))
                    history.append(_Record(rec.program_id, rec.label, rec.source, False,
                                           error_kind=rec.error_kind, error=rec.error,
                                           wall_seconds=round(rec.wall_seconds, 2)))
                    if know is not None and prog is not None:
                        self._kb_record(know, ctx, prog, 0.0, best_score, rec.wall_seconds,
                                        ok=False)
                    # Intelligence: track repair failures for multi-turn diagnosis
                    if intel_state is not None:
                        intel_mod.record_repair_failure(
                            intel_state, rec.error_kind or '', rec.error or '')
                        intel_mod.record_intelligence_outcome(
                            intel_state, prog, None, False) if prog is not None else None

            # --- (7) diagnosis feed-forward (reads VAL/history only; never sealed) ------------
            diag = diag_mod.diagnose(history, trail, task)
            # LLM-powered deep diagnosis: confusion matrix, root cause, targeted advice
            _val_truth = ([Task.rows_to_y([row], task.kind)[0] for row in splits.val_rows]
                          if splits.val_rows and best_prog else None)
            _val_preds = None
            if _val_truth is not None and best_prog is not None:
                try:
                    _best_res = self.sandbox_policy.run(
                        best_prog, X_train, y_train,
                        Task.rows_to_X(splits.val_rows),
                        kind=task.kind, wall_seconds=cfg.wall_seconds,
                        cpu_seconds=cfg.cpu_seconds)
                    if _best_res.ok:
                        _val_preds = _best_res.preds
                except Exception:
                    pass
            # Multi-turn diagnosis: if prior repairs failed, refine the analysis
            if intel_state is not None and intel_state.repair_failures:
                llm_diag = intel_mod.multi_turn_diagnose(
                    diag, task, history,
                    llm_client=llm, val_truth=_val_truth, val_preds=_val_preds,
                    prior_diagnosis=intel_state.prior_llm_diagnosis,
                    prior_repair_failures=intel_state.repair_failures,
                    max_turns=cfg.max_diagnosis_turns)
                intel_state.prior_llm_diagnosis = llm_diag
                intel_state.repair_failures = []  # reset for next round
            else:
                llm_diag = llm_diag_mod.llm_diagnose(
                    diag, task, history,
                    llm_client=llm, val_truth=_val_truth, val_preds=_val_preds)
                if intel_state is not None:
                    intel_state.prior_llm_diagnosis = llm_diag
            trail.append({"round": r, "n_proposals": len(pool),
                          "best_out_label": best_prog.label if best_prog else None,
                          "best_out_score": round(best_score, 4) if best_score is not None else None,
                          "diag": diag.reason})
            self._emit("measure", round=r,
                       best_label=best_prog.label if best_prog else None,
                       best_score=round(best_score, 4) if best_score is not None else None,
                       diagnosis=diag.reason)

            # --- (8) checkpoint (resumable; the sealed peek is NOT inside the round loop) ------
            if store is not None:
                self._checkpoint_round(store, r, cfg.rounds, best_prog, best_score)
            # Resume-wrapper hook: persist the FULL champion + completed-round lineage atomically
            # so a crash after this point never recomputes round r. Fires only when a resume
            # wrapper installed it; default path is unaffected.
            if self._on_round_complete is not None:
                self._on_round_complete(r, best_prog, best_score)

            ctrl.end_round(r)
            if ctrl.stop():  # budget exhausted -> take the current champion to certify
                trail.append({"round": r, "note": "budget exhausted; stopping search"})
                break

        # ===== no candidate ran -> hard honest decline (no oracle, no peek) ===================
        if best_prog is None:
            return self._decline("no candidate executed successfully", history, trail,
                                 splits, llm_active=llm_active, backend_notes=backend_notes)

        # self-consistency over the high-VAL authored programs (NON-promoting metadata).
        sc = None
        try:
            from frontier.core.authoring import AuthoringEngine
            sc = AuthoringEngine.self_consistency(scored_programs)
        except Exception:
            sc = None

        # ===== THE ONLY SEALED TOUCH (certify the winner ONCE) ================================
        cert, sealed_fail = self._certify_winner(task, splits, best_prog, X_train, y_train)
        if sealed_fail is not None:
            return self._decline(sealed_fail, history, trail, splits, winner=best_prog,
                                 val=best_score, llm_active=llm_active,
                                 backend_notes=backend_notes, self_consistency=sc)

        # ===== oracle gate: AND verify_before_promote into certified (design 04 §2.8) =========
        #     The oracle can only VETO. A refuted winner -> honest decline; certified=False with
        #     the un-promoted certificate carried for the report. The reproducibility oracle uses
        #     its OWN fresh split + peek budget, so the winner's single peek stays at 1.
        def _run_fn(prog: Program, Xtr, ytr, Xev):
            # Through the HONEST policy front door (advisory gate + enforced stamp), not raw run_program.
            r = self.sandbox_policy.run(prog, Xtr, ytr, Xev, kind=task.kind,
                                        wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)
            return r.preds if r.ok else None

        verdict = oracles.verify_before_promote(task, splits, best_prog, cert, _run_fn,
                                                seed=cfg.seed)

        # ===== BASELINE-FIRST GATE: an improvement must BEAT the certified floor ===============
        #     beats_floor compares the winner's certified sealed LOWER bound to the floor's
        #     certified sealed LOWER bound (lower-vs-lower, same statistical discipline as
        #     promotion). A win that clears theta + the oracle but only TIES (or loses to) the
        #     floor is NOT an improvement: we AND beats_floor into `promoted` and report it
        #     honestly as "no improvement over baseline". If no floor could be certified, the
        #     gate admits the winner (it never invents a floor) and says so.
        floor_verdict = baseline_mod.beats_floor(cert, getattr(self, "_floor", None))
        promoted = (bool(cert.get("certified")) and verdict.promote
                    and floor_verdict.beats_floor)

        # ===== knowledge.record + report (after the certificate exists) =======================
        if know is not None:
            try:
                self._kb_record(know, self._build_context(task, splits, cfg.rounds, tried,
                                                          best_prog, best_score, recent_errors,
                                                          know),
                                best_prog, float(best_score), None,
                                0.0, ok=True)  # durable record of the certified/declined winner
            except Exception as e:
                backend_notes.append(f"knowledge.record failed: {e!r}")

        floor = getattr(self, "_floor", None)
        result = CoreResult(
            certified=promoted, certificate=cert, winner=best_prog,
            winner_val_score=round(best_score, 4),
            history=history, diagnosis_trail=trail, split_meta=splits.meta,
            decline_reason=("" if promoted
                            else self._decline_reason_from_verdict(cert, verdict, floor_verdict)),
            llm_active=llm_active,
            oracle_verdict=verdict.to_dict(), sealed_peeks=self.sealed_peeks,
            backend_notes=backend_notes, self_consistency=sc,
            floor_certificate=(floor.to_dict() if floor is not None else None),
            floor_verdict=floor_verdict.to_dict(),
        )

        self._emit("done", certified=promoted,
                   winner_label=best_prog.label if best_prog else None,
                   winner_val_score=round(best_score, 4) if best_score is not None else None,
                   sealed_lower_bound=cert.get("lower_bound") if isinstance(cert, dict) else None,
                   oracle_promote=verdict.promote,
                   beats_floor=floor_verdict.beats_floor,
                   decline_reason=result.decline_reason)
        # report is a pure consumer of the EngineResult-shaped projection (no sealed peek).
        try:
            artifact = report_mod.build_report(result.as_engine_result(), task, goal=goal)
            result.artifact = artifact
            if cfg.report_dir:
                report_mod.write_report(result.as_engine_result(), task, cfg.report_dir,
                                        goal=goal)
        except Exception as e:
            backend_notes.append(f"report.render failed: {e!r}")
        return result

    # =========================================================================================
    # Internal helpers
    # =========================================================================================

    def _build_context(self, task: Task, splits, r: int, tried: set,
                       best_prog: Optional[Program], best_score: Optional[float],
                       recent_errors: List[Tuple[str, str, str]],
                       know: Optional[knowledge.KnowledgeProposer]) -> dict:
        """Build the Phase-0-compatible context, then add the knowledge keys. SOLE build site.

        Diagnosis enrichment is applied by the caller (one place) right after this returns. We
        build EXACTLY the Phase-0 keys first so every existing ProposalSource keeps working, then
        add knowledge keys (additive). We NEVER place anything derived from sealed_rows here.
        """
        ctx = {
            "task_kind": task.kind,
            "n_features": task.n_features,
            "n_train": len(splits.train_rows),
            "round": r,
            "tried_labels": set(tried),
            "best_label": best_prog.label if best_prog else None,
            "best_score": round(best_score, 4) if best_score is not None else None,
            "best_id": best_prog.id if best_prog else None,
            "best_recipe": (best_prog.provenance.get("recipe") if best_prog else None),
            "recent_errors": list(recent_errors),
        }
        # knowledge keys depend ONLY on Task (fingerprint/descriptor exclude sealed data).
        if know is not None:
            ctx["task_descriptor"] = knowledge.task_descriptor(task)
            ctx["task_fingerprint"] = knowledge.task_fingerprint(task)
        return ctx

    def _admit_with_floor(self, ordered: List[Program], ctrl, cost, task, splits) -> List[Program]:
        """Budget-admit candidates while GUARANTEEING the deterministic floor is runnable.

        The budget controller may reject ranked candidates when compute is tight. To prevent a
        learned heuristic from implicitly bounding the promotion-bearing search (design 04 §6
        Risk 2), we reserve at least ``floor_min_seeds`` slots for seed/mutation FLOOR programs
        regardless of admission. Ranking only ORDERS; this method only ensures the floor is never
        starved. Order is preserved; admitted candidates keep their LinUCB rank.
        """
        n_tr, n_feat = len(splits.train_rows), task.n_features
        admitted: List[Program] = []
        for p in ordered:
            est = cost.estimate(p, n_tr, n_feat)
            if ctrl.admit(p.id, est.seconds):
                admitted.append(p)
        # Floor guarantee: ensure >=N seed/mutation programs are present even if budget rejected
        # them (or if admission produced an empty batch on a tiny budget).
        floor = [p for p in ordered if p.source in ("seed", "mutation")]
        present = sum(1 for p in admitted if p.source in ("seed", "mutation"))
        need = max(0, self.cfg.floor_min_seeds - present)
        admitted_ids = {p.id for p in admitted}
        for p in floor:
            if need <= 0:
                break
            if p.id not in admitted_ids:
                admitted.append(p)
                admitted_ids.add(p.id)
                need -= 1
        # If everything was rejected and there is no floor either, fall back to the ranked head
        # so the round still does real work (a certified result on a smaller search is valid).
        if not admitted:
            admitted = list(ordered[: max(1, self.cfg.floor_min_seeds)])
        return admitted

    def _try_repair(self, prog: Optional[Program], rec, ctx: dict, X_train, y_train, X_val,
                    task) -> Tuple[Optional[Program], Optional[RunResult]]:
        """Run ONE bounded repair on a failed candidate. Honest no-op when nothing applies.

        Uses agentic.repair_loop (LLM repairer when a client is wired; deterministic heuristics
        otherwise). We feed the initial RunResult so the loop does not re-run the original. The
        repaired program re-enters execution inside repair_loop's last attempt via the SAME
        firewall sandbox (preds-only). Returns (None, None) when no fix turned it green.
        """
        if prog is None:
            return None, None
        initial = RunResult(rec.program_id, ok=False, error=rec.error,
                            error_kind=rec.error_kind, wall_seconds=rec.wall_seconds)
        try:
            rr = agentic.repair_loop(prog, X_train, y_train, X_val, kind=task.kind,
                                     llm_client=self.cfg.llm_client, k=1,
                                     wall_seconds=self.cfg.wall_seconds,
                                     cpu_seconds=self.cfg.cpu_seconds, context=ctx,
                                     initial=initial)
        except Exception:
            return None, None
        if rr.ok and rr.run is not None and rr.run.ok and rr.repaired:
            return rr.program, rr.run
        return None, None

    def _record_success(self, history: List[_Record], know, ctx, prog, rec,
                        best_score: Optional[float]) -> None:
        """Append the trusted-parent VAL number to history and update the KB/bandit."""
        history.append(_Record(rec.program_id, rec.label, rec.source, True,
                               val_score=round(float(rec.val_score), 4),
                               wall_seconds=round(rec.wall_seconds, 2)))
        if know is not None and prog is not None:
            self._kb_record(know, ctx, prog, float(rec.val_score), best_score,
                            rec.wall_seconds, ok=True)

    def _make_gpu_dispatch(self, kind: str) -> Callable:
        """Build a per-arm run function that routes neural proposals through the GPU backend.

        The returned callable has the portfolio's run_fn signature:
            (Program, X_train, y_train, X_val, kind, wall_s, cpu_s) -> RunResult

        Routing logic:
          - Programs with provenance["neural_spec"] go through execution.run_program with
            TORCH_SPEC when torch is available (probed ONCE at construction, cached).
          - Programs without neural_spec go through sandbox.run_program (same as the
            portfolio default path).

        This preserves the firewall: the child returns predictions only, the trusted parent scores.
        """
        cfg = self.cfg
        # Probe torch availability ONCE, not per-arm (avoids subprocess fork per arm)
        try:
            torch_cap = execution.probe_backend(execution.TORCH_SPEC)
            torch_ok = torch_cap.runnable
        except Exception:
            torch_ok = False

        def _dispatch(prog: Program, Xtr, ytr, Xev, k, wall_s, cpu_s):
            prov = prog.provenance if hasattr(prog, 'provenance') and prog.provenance else {}
            if "neural_spec" in prov and torch_ok:
                try:
                    return execution.run_program(
                        prog, Xtr, ytr, Xev, kind=k,
                        wall_seconds=wall_s, cpu_seconds=int(cpu_s),
                        backend=execution.TORCH_SPEC, seed=cfg.seed)
                except Exception:
                    pass
            return sandbox.run_program(
                prog, Xtr, ytr, Xev, kind=k,
                wall_seconds=wall_s, cpu_seconds=int(cpu_s))

        return _dispatch

    def _kb_record(self, know, ctx, prog, val_score: float, incumbent_before: Optional[float],
                   cost_seconds: float, *, ok: bool) -> None:
        """Record the honest VAL gain into the KB + LinUCB. Never a sealed/relabeled number."""
        try:
            know.record_outcome(ctx, prog, val_score=val_score,
                                incumbent_before=incumbent_before,
                                cost_seconds=float(cost_seconds), ok=ok)
        except Exception:
            pass  # a KB hiccup must never abort the loop (it is selection-side only)

    def _certify_winner(self, task: Task, splits, winner: Program, X_train, y_train
                        ) -> Tuple[Optional[dict], Optional[str]]:
        """THE ONLY SEALED TOUCH. Re-fit the winner, then certify once. Counts the peek.

        Returns (certificate, None) on success, or (None, decline_reason) if the sealed re-fit
        failed (in which case certify is never called and the one peek is NOT consumed). This is
        byte-for-byte the Phase-0 discipline (engine.py:164-173), executed here AFTER selection.
        """
        # The child receives ONLY sealed FEATURES (X_sealed); the sealed LABELS never cross the boundary
        # (they live in splits.sealed_rows here in the trusted parent and are read solely by
        # certify.certify_on_sealed below). Routing through the policy front door runs the advisory gate
        # (a winner carrying ndarray.dump/dumps or a scipy.io writer is rejected as error_kind="policy"
        # BEFORE it touches the sealed features) and stamps the enforced guarantees onto `final`.
        X_sealed = Task.rows_to_X(splits.sealed_rows)
        final = self.sandbox_policy.run(winner, X_train, y_train, X_sealed, kind=task.kind,
                                        wall_seconds=self.cfg.wall_seconds,
                                        cpu_seconds=self.cfg.cpu_seconds)
        if not final.ok:
            return None, (f"winner failed on sealed re-fit: [{final.error_kind}] {final.error}")
        cert = certify.certify_on_sealed(task, splits, final.preds)  # ONE counted peek (parent computes it)
        self.sealed_peeks += 1
        # Stamp the ENFORCED isolation (probe-derived, never the requested tier) into the certificate so a
        # downstream reader can weight the number by the containment that actually held (innovation N1).
        enforced = enforced_of(final)
        if enforced is not None:
            cert = dict(cert)
            cert["sandbox_enforced"] = enforced.as_dict()
        return cert, None

    @staticmethod
    def _decline_reason_from_verdict(cert: dict, verdict, floor_verdict=None) -> str:
        """Human-readable decline reason when the winner certified but did not promote.

        Order of honesty: a winner that fails to clear theta declines on theta; a winner refuted
        by an oracle declines on the oracle; a winner that clears theta + oracle but does NOT beat
        the certified baseline floor declines as an explicit non-win against the floor. The floor
        reason is checked LAST so it only surfaces when the winner was otherwise promotable.
        """
        if not bool(cert.get("certified")):
            return "sealed lower bound does not clear theta (honest decline)"
        reasons = getattr(verdict, "reasons", None) or []
        if not getattr(verdict, "promote", True):
            return ("winner refuted by oracle: " + "; ".join(reasons) if reasons
                    else "winner not promoted by oracle gate")
        if floor_verdict is not None and not floor_verdict.beats_floor:
            return "did not beat baseline floor: " + floor_verdict.reason
        return "winner not promoted"

    def _checkpoint_round(self, store, r: int, n_rounds: int,
                          best_prog: Optional[Program], best_score: Optional[float]) -> None:
        """Persist round-state so a resume skips completed rounds (best-effort; never aborts).

        We update RunState in place (the durable schema): mark round r complete, record the
        champion, and stash the champion id in payload so a resume can rehydrate it. The sealed
        peek is one-shot and lives OUTSIDE this loop, so a resume never re-peeks.
        """
        try:
            state = store.load() or checkpoint_mod.RunState(
                run_id=str(getattr(store, "path", "run")), n_rounds=n_rounds)
            if r not in state.completed_rounds:
                state.completed_rounds.append(r)
            state.champion_id = best_prog.id if best_prog else None
            state.champion_score = best_score
            state.payload = {"best_prog_id": best_prog.id if best_prog else None}
            store.save(state)
        except Exception:
            pass  # checkpoint is orchestration only; a hiccup must not crash the loop

    def _decline(self, reason: str, history: Optional[List] = None,
                 trail: Optional[List[dict]] = None, splits=None, *,
                 winner: Optional[Program] = None, val: Optional[float] = None,
                 llm_active: bool = False, backend_notes: Optional[List[str]] = None,
                 self_consistency: Optional[dict] = None) -> CoreResult:
        """Build an honest decline. certified=False, certificate=None (or carried attempt).

        A decline is a first-class outcome (invariant 5): never a relabeled val score. The VAL
        champion score is reported as ``winner_val_score`` (clearly labeled), never as the result.
        """
        self._emit("done", certified=False,
                   winner_label=winner.label if winner is not None else None,
                   winner_val_score=(round(val, 4) if val is not None else None),
                   decline_reason=reason)
        return CoreResult(
            certified=False, certificate=None, winner=winner,
            winner_val_score=(round(val, 4) if val is not None else None),
            history=list(history or []), diagnosis_trail=list(trail or []),
            split_meta=(splits.meta if splits is not None else {}),
            decline_reason=reason, llm_active=llm_active, sealed_peeks=self.sealed_peeks,
            backend_notes=list(backend_notes or []), self_consistency=self_consistency,
        )


# ============================================================================================
# small free helpers
# ============================================================================================

def _modality_of(spec) -> str:
    """Map a router ProblemSpec modality to the neural core's modality vocabulary.

    The neural proposers understand {tabular, sequence, image1d}; anything else (or absent)
    degrades to "tabular", the safe default for dense numeric features. Never raises.
    """
    mod = getattr(spec, "modality", "tabular") if spec is not None else "tabular"
    if mod in ("tabular", "sequence", "image1d"):
        return mod
    if mod in ("text",):
        return "sequence"
    return "tabular"


__all__ = ["CoreConfig", "CoreResult", "CoreOrchestrator"]
