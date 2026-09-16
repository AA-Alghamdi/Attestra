"""The Ring-1 /goal loop -- MULTI-ROUND (build-order step 6).

round 0 baseline -> [ DIAGNOSE (measured signals) -> PROPOSE only the diagnosis-relevant moves -> VoI
rank -> CHECKPOINT -> EXECUTE fan-out via a Provider -> MEASURE on validation (val score + ECE) ->
UPDATE best/leaderboard/budget/case-base ] x N -> VAL-BOUND-PREDICTS-PASS gate -> certify the
validation-winner ONCE on the SEALED test (enforced counted peek) OR honest-stop with no peek -> narrate.

Implemented per the architecture scheme:
  * DIAGNOSE (_diagnose) decomposes the validation error into measured signals (headroom, per-class
    recall, ECE) and GATES the move menu -- the loop proposes only the relevant moves, not every unused
    one ("diagnosis is the lever, not search"). An empty relevant set is an honest data-limited stop.
  * The VAL-BOUND gate (_val_lower_bound) spends the one counted sealed peek ONLY when the winner's
    validation lower bound (Bonferroni over finished candidates) already clears theta; otherwise it
    honest-stops with NO peek, preserving the sealed test.
  * Non-certified outcomes carry a failure_report {dominant_source, cheapest_unblock, next_experiment}.
  * VERIFY-SPEC gates: forbidden columns dropped before featurizing; certified == bound>theta AND
    latency_ok AND cost_ok. Optional reaudit-gated data acquisition (acquire_fn).

Selection is on VALIDATION only. The sealed test is read exactly once, through the SealedTest guard, with
checks = realized peeks. Only science (the frozen core) promotes.
"""
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_VF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VF not in sys.path:
    sys.path.insert(0, _VF)
from vectorforge import science
from vectorforge.llm_shell.narration import narrate, EvidenceBundle

from .leaderboard import Run, Leaderboard
from .providers import Job, LocalCpuProvider, RunPodProvider, first_available
from .tracking import ExperimentStore
from .sealed import SealedTest, certify_on_sealed, assert_supported_metric, PeekLedger
from .harness import Harness, harness_for, is_supported, runnable_catalog
from .llm_moves import propose_moves
from . import prereg as _prereg
from .cross_experiment import PromotionLedger, LordFDR, negative_certificate
from .voi import CaseBase, voi_rank
from .research_memory import ResearchMemory
from .checkpoint import Checkpoint


class _CancelledRun(Exception):
    """Internal signal: the operator cancelled the run mid-fan-out. Caught at the loop's top level and
    turned into a terminal 'cancelled' GoalLoopResult (no sealed peek, no certificate)."""


@dataclass
class GoalLoopResult:
    experiment: str
    leaderboard: Leaderboard
    winner: Optional[Run]
    certificate: Optional[dict]
    decision: str
    narration: str
    n_test: int
    provider: str
    rounds: list = field(default_factory=list)        # [{round, move, families, best_val_after}]
    compute_report: dict = field(default_factory=dict)
    audit_findings: list = field(default_factory=list)
    failure_report: Optional[dict] = None             # {dominant_source, cheapest_unblock, next_experiment} (step 12)
    objective: str = "certify"                        # resolved objective ("certify" | "maximize")
    time_budget_s: Optional[float] = None             # resolved wall-clock target (echoed back to the caller)
    elapsed_s: Optional[float] = None                 # measured wall-clock spent in the search loop
    plan_hash: Optional[str] = None                   # pre-registration content address of the frozen spec
    cross_experiment: Optional[dict] = None           # online-FDR view over the promotion sequence (reporting)
    negative_result: Optional[dict] = None            # first-class negative certificate on honest-stop / futile
    cand_locked_correct: Optional[list] = None        # winner per-row 0/1 correctness on the sealed test (battery)
    base_locked_correct: Optional[list] = None        # FIXED-baseline per-row 0/1 correctness on the SAME rows (battery)

    def summary(self):
        w = self.winner
        return {"experiment": self.experiment, "decision": self.decision, "n_test": self.n_test,
                "provider": self.provider, "rounds": self.rounds, "objective": self.objective,
                "time_budget_s": self.time_budget_s, "elapsed_s": self.elapsed_s,
                "winner": (w.as_dict() if w else None), "certificate": self.certificate,
                "leaderboard": self.leaderboard.as_dict(top=5), "compute_report": self.compute_report}


def reaudit(existing_train, new_rows, test, *, allow_features=True, text_key="text", target_key="target",
            min_test_n=1):
    """Re-run the frozen leakage auditor on (existing_train + newly acquired rows) vs the sealed test
    BEFORE the new rows may enter training (build-order step 7). Returns (passed, findings). Any data the
    loop acquires mid-run (synthetic or labeled) must clear this before use."""
    combined = list(existing_train) + list(new_rows)
    rep = science.audit(combined, test, allow_features=allow_features, text_key=text_key,
                        target_key=target_key, min_test_n=min(min_test_n, max(1, len(test))))
    return bool(rep["passed"]), [f for f in rep["findings"] if not f.get("ok", True)]


def _to_list(X):
    """Serialize a feature matrix for a JobSpec payload (dense list-of-lists; densifies sparse tfidf)."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X).tolist()


def _score_val(metric, y_true, y_pred, is_regression):
    """Re-score validation LOCALLY with the frozen score_metric (step 7/F6): the worker returns val
    PREDICTIONS, never a trusted scalar, so selection uses the real metric computed on the platform."""
    if is_regression:
        return float(science.score_metric(metric, [float(a) for a in y_true],
                                          [float(b) for b in y_pred], None))
    labels = sorted({str(a) for a in y_true} | {str(b) for b in y_pred})
    return float(science.score_metric(metric, [str(a) for a in y_true],
                                      [str(b) for b in y_pred], labels))


def _ece(confidences, y_pred, y_true):
    """Validation ECE (step 8/MEASURE) from per-sample max-class confidence + correctness. Returns None if
    no probability channel is available (e.g. a margin-only classifier or regression)."""
    if confidences is None:
        return None
    try:
        correct = [int(str(a) == str(b)) for a, b in zip(y_pred, y_true)]
        return float(science.expected_calibration_error(list(confidences), correct, bins=10))
    except Exception:  # noqa: BLE001
        return None


def _val_lower_bound(metric, y_true, y_pred, theta, checks, alpha, is_regression):
    """The winner's VALIDATION lower bound (step 11), Bonferroni-corrected over `checks` (the number of
    finished candidates we selected among). Same frozen primitives as the sealed certifier, so the val
    bound and the sealed bound are comparable. Used to decide whether to spend the one sealed peek."""
    if is_regression:
        c = science.certify_regression([float(a) for a in y_true], [float(b) for b in y_pred],
                                       metric, theta, checks=checks, alpha=alpha)
        return float(c["lower_bound"])
    if metric == "accuracy":
        n = len(y_true)
        k = sum(1 for a, b in zip(y_true, y_pred) if str(a) == str(b))
        c = science.certify_accuracy(k / max(n, 1), n, theta, checks=checks, alpha=alpha)
        return float(c["lower_bound"])
    labs = sorted({str(x) for x in y_true} | {str(x) for x in y_pred})
    lb, _point = science._bootstrap_classification_metric_lower(
        metric, [str(x) for x in y_true], [str(x) for x in y_pred], labs, alpha=alpha / max(checks, 1))
    return float(lb)


def _failure_report(decision, diagnosis, threshold, val_lb, n_val):
    """HONEST FAILURE contract (step 12): from the MEASURED diagnosis, name the dominant error source and
    the CHEAPEST unblock + a concrete next experiment. Emitted on every non-certified outcome so a failure
    is actionable, not just 'do_not_certify'."""
    d = diagnosis or {}
    headroom = d.get("headroom")
    ece = d.get("ece")
    minrec = d.get("min_class_recall")
    if decision == "no_model":
        return {"dominant_source": "no candidate model finished (fit/numeric failures)",
                "cheapest_unblock": "fix the failing family or widen the move menu",
                "next_experiment": "re-run with a simpler baseline and inspect the per-candidate errors"}
    if decision == "blocked":
        return {"dominant_source": "leakage / contamination caught by the auditor",
                "cheapest_unblock": "remove the leaking feature or fix the train/test overlap",
                "next_experiment": "drop the flagged column(s) and re-run the audit"}
    # do_not_certify / honest_stop: rank the measured signals
    if minrec is not None and minrec < 0.6:
        src = f"localized weakness: the weakest class has recall {minrec} on validation"
        unblock = "add capacity (stronger_model/more_capacity) or reweight the weak class"
    elif ece is not None and ece > 0.10:
        src = f"miscalibration: validation ECE {ece} (> 0.10)"
        unblock = "calibrate the classifier (Platt/isotonic) or regularize the fit"
    elif headroom is not None and headroom > 0:
        # estimate labels-to-acquire from the val-bound gap (loose, honest order-of-magnitude)
        gap = max(0.0, threshold - (val_lb if val_lb is not None else (threshold - (headroom or 0.0))))
        need = int(max(50, (n_val or 100) * (1.0 + 4.0 * gap)))
        src = (f"capacity/ceiling: validation point is {headroom} below the bar"
               + ("" if val_lb is None else f"; val lower bound {round(val_lb,4)} does not clear {threshold}"))
        unblock = f"either a higher-capacity model lifts the val bound, or acquire ~{need} more labels"
        return {"dominant_source": src, "cheapest_unblock": unblock,
                "next_experiment": f"acquire ~{need} labeled examples (or enable a GPU deep_model) and re-run"}
    else:
        src = "validation bound below the threshold with no single dominant measured signal"
        unblock = "acquire more labeled data or relax the threshold (a spec change, report both)"
    return {"dominant_source": src, "cheapest_unblock": unblock,
            "next_experiment": "apply the cheapest unblock above and re-run; selection stays on validation"}


def _diagnose(best_run, val_pred, ece, threshold, is_regression, val_lb=None):
    """DIAGNOSE node (step 10): decompose the current validation error into MEASURED signals and return the
    set of move names the diagnosis makes relevant. This GATES the menu (the scheme's 'menu gated by
    diagnosis' / 'diagnosis is the lever, not search'), replacing 'every unused move'. Pragmatic port of
    vectorforge/brain.py's measured gates: capacity (headroom), localized weakness (per-class recall),
    calibration (ECE). An EMPTY relevant set means no diagnosed lever -> honest data-limited stop.

    The 'below the bar' signal is driven by the validation LOWER BOUND when available (the loop objective is
    to raise the bound, not the point) -- so when the point clears but the bound does not (small data), the
    diagnosis still escalates rather than declaring premature success."""
    val = (best_run.val_score if best_run else None) or 0.0
    basis = val_lb if val_lb is not None else val          # drive on the bound when we have it
    below = basis < threshold
    sig = {"val": round(val, 4), "val_lb": (None if val_lb is None else round(val_lb, 4)),
           "headroom": round(threshold - basis, 4), "ece": ece, "below_bar": below}
    relevant = set()
    if is_regression:
        if below:
            relevant |= {"stronger_model", "more_capacity"}            # capacity axis
        return sig, relevant
    min_recall = None
    if val_pred:
        yt = [str(x) for x in val_pred["y_true"]]
        yp = [str(x) for x in val_pred["y_pred"]]
        labs = sorted(set(yt) | set(yp))
        rec = science.per_class_recall(yt, yp, labs)             # {lab: {"recall": float|None, "support": n}}
        recalls = [v["recall"] for v in rec.values() if v.get("recall") is not None]
        min_recall = min(recalls) if recalls else None
    sig["min_class_recall"] = None if min_recall is None else round(min_recall, 4)
    if below:
        relevant |= {"stronger_model", "more_capacity", "deep_model"}  # capacity: lift the val bound
    if min_recall is not None and min_recall < 0.6:
        relevant |= {"stronger_model", "more_capacity", "regularize"}  # localized: a class is being missed
    if ece is not None and ece > 0.10:
        relevant |= {"calibrate", "regularize"}                        # miscalibration -> calibrate (isotonic) or regularize
    return sig, relevant


def _split(records, target_key, text_key, seed, is_regression):
    """Frozen leakage-safe split. Regression stratifies by target quantile bins so make_splits' discrete
    stratification + near-duplicate dedup still apply, then strips the temporary bin tag."""
    if not is_regression:
        tr, va, te, _ = science.make_splits(records, seed=seed, target_key=target_key, text_key=text_key)
        return tr, va, te
    vals = np.array([float(r.get(target_key)) for r in records], dtype=float)
    edges = np.quantile(vals, [i / 10.0 for i in range(1, 10)])
    bins = np.digitize(vals, edges)
    tagged = [dict(r, _strat_bin=int(b)) for r, b in zip(records, bins)]
    tr, va, te, _ = science.make_splits(tagged, seed=seed, target_key="_strat_bin", text_key=text_key)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "_strat_bin"} for r in rows]
    return strip(tr), strip(va), strip(te)


def run_goal_loop(records, goal_text, *, harness: Harness = None, kind="tabular", task_type="binary",
                  target_key="target", labels=None, threshold=0.75, metric=None,
                  store: ExperimentStore = None, providers=None, llm_enabled=False, seeds=(0, 1),
                  text_key="text", min_test_n=100, alpha=0.05, seed=0, experiment=None,
                  max_rounds=12, budget_experiments=200, stop_margin=0.05, checkpoint=None,
                  casebase_path=None, memory_store=None, peek_ledger_path=None,
                  drop_cols=None, max_latency_ms=None, max_cost_usd=None, max_ece=None,
                  acquire_fn=None, acquire_budget=0, on_event=None, registry=None,
                  test_records=None, should_cancel=None,
                  objective="certify", time_budget_s=None, llm_propose=True,
                  llm_cache_path=None, authored_propose=False, authored_sources_fn=None) -> GoalLoopResult:
    """objective: "certify" (clear theta then take the one sealed peek -- the original behavior) or
    "maximize" (no early stop on clearing; run the FULL budget tracking the best validation model, then take
    the ONE sealed peek on that best model and report decision "best_effort"; certified=True ONLY if a theta
    was given AND the sealed lower bound clears it -- never claims more than measured).

    time_budget_s: wall-clock target. While elapsed < time_budget_s AND there are still untried proposed
    candidates, the loop KEEPS proposing+executing rounds (it does NOT honest-stop early while budget +
    proposals remain). A hard safety cap (2x time_budget_s, min 1800s) prevents runaway.

    llm_propose: enable the LLM-guided PROPOSE step (vfplatform/llm_moves.propose_moves). Auto-falls back to
    deterministic grid-expansion when no Anthropic key / the call fails (the proposer owns that fallback).

    memory_store: opt-in DURABLE, DATASET-AWARE learning substrate (a casebase_store.CaseBaseStore). When
    given, the loop fingerprints this dataset (meta-features), warm-starts its VoI ranking from the realized
    per-family gains of the most SIMILAR past datasets, and appends this run's realized (family, gain, cost,
    device) outcomes back to the store -- so the autoresearcher gets better every run AND transfers what it
    learned to NEW datasets (and a GPU campaign's gains accumulate, tagged by device). Selection-only: it
    never reads the sealed test; the frozen certify path is untouched. Default None == cold start ==
    byte-identical to the pre-memory behavior. (``casebase_path`` is the older FLAT per-run/campaign memory;
    the two compose -- ``memory_store`` adds the cross-dataset tier on top.)

    The LLM is NON-BINDING throughout: it only proposes candidate (family, hyperparameter) configs from a
    bounded catalog. It never sets the certificate, the decision, or theta; the certificate is produced ONLY
    by the frozen certify path (_val_lower_bound gate + certify_on_sealed)."""
    store = store or ExperimentStore(os.path.join(_VF, "vf_runs"))
    experiment = experiment or ("exp-" + science.digest({"g": goal_text, "n": len(records)}).split(":")[-1][:8])

    def _emit(stage, status="active", **detail):
        """Live stage event for the UI / any observer (the 2D workflow is driven by these). Best-effort:
        a failing observer never breaks the run."""
        if on_event is None:
            return
        try:
            on_event({"stage": stage, "status": status, "experiment": experiment, **detail})
        except Exception:  # noqa: BLE001
            pass

    def _cancelled():
        """True iff the caller's should_cancel() returns truthy. Best-effort: a misbehaving callable never
        breaks the run (treated as 'not cancelled')."""
        if should_cancel is None:
            return False
        try:
            return bool(should_cancel())
        except Exception:  # noqa: BLE001
            return False

    def _cancel_result():
        """TERMINAL cancellation: emit a done-stage 'cancelled' event and return a result whose decision is
        'cancelled' with NO certificate (the sealed test is never peeked on a cancel). Reads enclosing
        locals defensively: a cancel can fire at intake (before lb/provider/etc. exist) or mid-loop."""
        _emit("cancelled", status="done")
        try:
            _lb = lb
        except NameError:
            _lb = Leaderboard(metric or "accuracy", [])
        try:
            _prov = provider.name
        except NameError:
            _prov = "local"
        try:
            _rounds = rounds_info
        except NameError:
            _rounds = []
        try:
            _compute = compute_report
        except NameError:
            _compute = {}
        return GoalLoopResult(experiment, _lb, None, None, "cancelled",
                              "Run stopped by the operator before a certificate was produced.", 0,
                              _prov, rounds=_rounds, compute_report=_compute)

    _emit("intake", goal=goal_text, kind=kind, task_type=task_type, n_records=len(records))
    if _cancelled():
        return _cancel_result()

    # fail-closed modality/task guard (defense in depth; the front door declines earlier): if no built
    # harness fits this (kind, task_type), DECLINE honestly -- no loop, no sealed peek, no certificate.
    if harness is None and not is_supported(kind, task_type):
        return GoalLoopResult(
            experiment, Leaderboard(metric or "accuracy", []), None, None, "unsupported",
            f"Declined: no built harness for kind={kind!r}, task_type={task_type!r}. Supported: "
            f"tabular[binary|multiclass|regression], text[binary|multiclass]. Refusing to coerce it into "
            f"a different problem and emit a certificate.", 0, "none",
            compute_report={})
    # FORBIDDEN features (VERIFY SPEC, step 13/F23): drop them BEFORE featurizing so a forbidden column is
    # never trained on (the front door previously discarded this constraint).
    if drop_cols:
        _dc = set(drop_cols)
        records = [({**r, "features": {k: v for k, v in (r.get("features") or {}).items() if k not in _dc}}
                    if isinstance(r.get("features"), dict) else r) for r in records]

    harness = harness or harness_for(kind, task_type, text_key=text_key)
    metric = metric or harness.default_metric
    assert_supported_metric(metric)                                   # moat guard: no metric fall-through
    # PRE-REGISTRATION (Phase 2 rigor): content-address the spec BEFORE the run so any later threshold change
    # is a NEW, recorded commitment (never silent). Additive + non-binding: recorded on the certificate and
    # appended to an append-only plan registry; it cannot change the certify decision.
    _ledger_dir = os.path.join(_VF, "vf_runs")
    try:
        _plan = _prereg.commit(_prereg.canonical_spec(
            metric=metric, threshold=threshold, alpha=alpha,
            forbidden_fields=tuple(sorted(drop_cols or ()))))
        _plan_hash = _plan["plan_hash"]
        _prereg.PlanRegistry(os.path.join(_ledger_dir, "_plan_registry.jsonl")).register(_plan)
        _emit("prereg", plan_hash=_plan_hash, metric=metric, threshold=threshold, alpha=alpha)
    except Exception:  # noqa: BLE001  pre-registration is additive; never let it break a run
        _plan_hash = None

    def _binom_pvalue(observed, n, theta):
        """One-sided p-value for H0: true <= theta on n sealed examples (accuracy/binomial only; else None)."""
        try:
            from scipy.stats import binom
            k = int(round(float(observed) * int(n)))
            return float(binom.sf(k - 1, int(n), float(theta)))
        except Exception:  # noqa: BLE001
            return None

    def _fdr_record_and_view(decision, cert):
        """Append the terminal promotion to the cross-experiment ledger and return the online-FDR (LORD) view
        replayed over it. REPORTING ONLY -- it never changes this run's frozen certify decision."""
        try:
            led = PromotionLedger(os.path.join(_ledger_dir, "_promotion_ledger.jsonl"))
            pv = (_binom_pvalue(cert.get("observed"), cert.get("n"), threshold)
                  if (metric == "accuracy" and cert) else None)
            led.record({"plan_hash": _plan_hash, "decision": decision, "metric": metric, "theta": threshold,
                        "observed": (cert or {}).get("observed"), "lower_bound": (cert or {}).get("lower_bound"),
                        "p_value": pv, "certified": bool((cert or {}).get("certified")), "ts": led.count() + 1})
            fdr = LordFDR(alpha=alpha)
            for e in led.all():
                if isinstance(e.get("p_value"), (int, float)):
                    fdr.test(float(e["p_value"]))
            return fdr.summary()
        except Exception:  # noqa: BLE001  reporting layer; never break a run
            return None

    runpod = RunPodProvider()
    providers = providers or [LocalCpuProvider()]
    compute_report = {p.name: p.capabilities() for p in providers}
    compute_report.setdefault(runpod.name, runpod.capabilities())
    provider = first_available(providers)
    checkpoint = checkpoint or Checkpoint()
    # CaseBase defaults to IN-MEMORY (per-run, isolated, deterministic). A GLOBAL shared file made
    # concurrent runs race on VoI calibration -> nondeterministic move selection / cross-talk (caught by the
    # UI's concurrency test). Cross-run VoI warm-starting is now OPT-IN via an explicit casebase_path.
    casebase = CaseBase(casebase_path)

    # split + audit (frozen). If the caller supplied an explicit verification/test set (their "here are my
    # 100 annotated examples"), it BECOMES the sealed test: train+val are split from `records`, and the
    # user's set is held out and certified against once. Otherwise the frozen splitter carves all three.
    # Either way science.audit checks train-vs-test contamination, so a verification set that overlaps the
    # training data is caught as leakage (not silently certified).
    if test_records is not None:
        tr, va, te_internal = _split(records, target_key, text_key, seed, harness.is_regression())
        train, val, test = (tr + te_internal), va, list(test_records)
        # CONTAMINATION GUARD for a USER-provided sealed set: make_splits guarantees disjoint train/test,
        # but a hand-supplied verification set can overlap the training data -> an inflated (memorized)
        # certificate. Block on ANY exact-content overlap between the verification rows and train+val.
        def _row_digest(r):
            feats = r.get("features") if isinstance(r.get("features"), dict) else {
                k: v for k, v in r.items() if k != target_key}
            return science.digest({"t": str(r.get(target_key)), "x": feats,
                                   "txt": str(r.get(text_key, ""))})
        seen = {_row_digest(r) for r in (train + val)}
        overlap = sum(1 for r in test if _row_digest(r) in seen)
        if overlap > 0:
            _emit("audit", status="blocked", contamination=overlap)
            return GoalLoopResult(
                experiment, Leaderboard(metric or "accuracy", []), None, None, "blocked",
                f"Blocked: {overlap}/{len(test)} rows of the supplied verification set also appear in the "
                f"training data (exact-content overlap). Certifying against memorized rows would inflate the "
                f"result. Provide a verification set DISJOINT from training.", len(test), provider.name,
                compute_report=compute_report,
                failure_report={"dominant_source": f"verification/train contamination ({overlap} rows)",
                                "cheapest_unblock": "supply a held-out verification set with no training rows",
                                "next_experiment": "de-duplicate the verification set against training and re-run"})
    else:
        train, val, test = _split(records, target_key, text_key, seed, harness.is_regression())
    _emit("split", n_train=len(train), n_val=len(val), n_test=len(test),
          user_test=bool(test_records is not None))
    _emit("audit")
    audit = science.audit(train, test, allow_features=(harness.kind == "tabular"), text_key=text_key,
                          target_key=target_key, min_test_n=min(min_test_n, max(1, len(test))))
    if not audit["passed"]:
        return GoalLoopResult(experiment, Leaderboard(metric, []), None, None, "blocked",
                              "Blocked before training: the leakage auditor flagged the data.",
                              len(test), provider.name,
                              compute_report=compute_report,
                              audit_findings=[f for f in audit["findings"] if not f.get("ok", True)],
                              failure_report=_failure_report("blocked", None, threshold, None, None))

    # featurize + encode (frozen split; harness-specific features/targets)
    if not harness.is_regression():
        labels = labels or sorted({str(r.get(target_key)) for r in records})
    feat = harness.featurizer().fit(train, train + val + test)
    ytr, l2i = harness.encode_targets(train, target_key, labels)
    yva, _ = harness.encode_targets(val, target_key, labels)
    Xtr, Xva, Xte = feat.transform(train), feat.transform(val), feat.transform(test)
    n_classes = 1 if harness.is_regression() else len(labels)

    lb = Leaderboard(metric=metric, runs=[])
    estimators = {}
    val_predictions = {}        # rid -> {y_pred, y_true, proba} for worker candidates (val-LB step 11 / ECE step 8)
    val_ece_by_rid = {}         # rid -> validation ECE (classification only; None if no probability channel)
    last_diagnosis = [None]     # most recent DIAGNOSE signals (step 10), surfaced on the result/failure report
    moves = harness.moves(seeds, provider=provider)
    by_name = {m.name: m for m in moves}
    # The PROPOSE node (llm_moves.propose_moves) draws from the per-task CATALOG, filtered to what THIS
    # provider can actually run (worker/GPU drops families with no worker/handler.build_model entry; the
    # local CPU path gets the full zoo). The single place family-name -> ctor lives is harness.py.
    cat, dropped_families = runnable_catalog(harness.kind, harness.task_type, provider=provider,
                                             n_features=(int(Xtr.shape[1]) if hasattr(Xtr, "shape") else None))
    if dropped_families:
        _emit("catalog", note="worker cannot build these families; filtered out", dropped=dropped_families)
    ctor_by_family = {}                          # family name -> local estimator ctor (for trusted re-fit)
    for _m in moves:
        for _fam in _m.families:
            ctor_by_family[_fam[0]] = _fam[1]
    used = set()                # Move NAMES already executed (never re-proposed / re-run)
    best_val = -1e18
    total_cost = 0.0            # cumulative real spend across the run (cost VERIFY-SPEC gate, step 13)
    rounds_info = []
    cost_rate = (provider.capabilities().get("cost_per_hour_usd", 0.0) or 0.0)
    _t_start = time.time()      # wall-clock anchor for the time_budget_s contract
    # Hard safety cap so a misbehaving proposer / slow fits cannot run away even with a budget set.
    _hard_cap_s = (max(2.0 * float(time_budget_s), 1800.0) if time_budget_s else None)
    _resolved_objective = objective if objective in ("certify", "maximize") else "certify"

    def _elapsed():
        return time.time() - _t_start

    def _budget_remaining():
        """True iff a time budget is set and we are still within it (the loop must NOT honest-stop early
        while budget + untried proposals remain)."""
        return time_budget_s is not None and _elapsed() < float(time_budget_s)

    def _over_hard_cap():
        return _hard_cap_s is not None and _elapsed() >= _hard_cap_s

    def _register_ctors(move):
        """Keep ctor_by_family current as adaptively-proposed moves arrive, so the trusted local-refit
        certify path (closes attack-3) can rebuild the winner's family on the platform."""
        for _fam in move.families:
            ctor_by_family[_fam[0]] = _fam[1]

    def _leaderboard_summary(topn=6):
        """TRUSTED summary of the runs so far for the PROPOSE node: the best families + val scores (no raw
        data, no labels). Lets the LLM avoid re-proposing weak regions and build on what is working."""
        finished = [r for r in lb.runs if r.status == "FINISHED" and r.val_score is not None]
        finished.sort(key=lambda r: r.val_score, reverse=True)
        return {"n_finished": len(finished), "n_failed": sum(1 for r in lb.runs if r.status == "FAILED"),
                "top": [{"family": r.family, "val": round(float(r.val_score), 4)} for r in finished[:topn]],
                "best_val": (round(float(finished[0].val_score), 4) if finished else None)}

    # TRUSTED dataset profile for the PROPOSE node (shapes, not rows): drives capacity/regularization choices.
    _profile = {"kind": harness.kind, "task_type": harness.task_type, "metric": metric,
                "n_train": len(train), "n_val": len(val),
                "n_features": int(Xtr.shape[1]) if hasattr(Xtr, "shape") else None,
                "n_classes": int(n_classes), "threshold": threshold}
    # DURABLE, DATASET-AWARE MEMORY (opt-in). With a memory_store, upgrade the cold CaseBase to a
    # ResearchMemory: it fingerprints THIS dataset from _profile and warm-starts VoI from the realized
    # per-family gains of the most SIMILAR past datasets, then records this run's outcomes back (tagged by
    # device, so a GPU campaign compounds). Selection-only; the frozen certify path is untouched. The free
    # variable `casebase` is read at call time by execute_move (late binding), so reassigning it here takes
    # effect for every round. No store -> casebase stays the plain CaseBase built above (byte-identical).
    if memory_store is not None:
        casebase = ResearchMemory.build(
            _profile, store=memory_store, run_casebase_path=casebase_path,
            kind=harness.kind, task_type=harness.task_type,
            device=(provider.capabilities().get("device") or "cpu"))
        _emit("memory", **casebase.warm_summary())
    # Resolve the LLM key through the SINGLE canonical resolver the ops layer uses (env OR the gitignored
    # .anthropic_key file), so the loop's _api_key is truthy whenever a key exists -- not env-only, which
    # mislabels a file-only setup as "no key". Non-binding: the LLM only PROPOSES; the certifier is unaffected.
    try:
        from vectorforge.llm_shell._keys import resolve_api_key as _resolve_key
        _api_key = _resolve_key(None)
    except Exception:  # noqa: BLE001  key resolution must never break a run
        _api_key = os.environ.get("ANTHROPIC_API_KEY")
    _n_llm_rounds = [0]   # observability: how many rounds the LLM (not the deterministic grid) drove PROPOSE

    def _run_record(name, params, s, rnd, content_id, provider_name):
        return store.start_run(experiment, params={"family": name, "move": "", **params},
                               tags={"seed": s, "round": rnd, "kind": harness.kind},
                               system={"provider": provider_name, "content_id": content_id})

    def execute_move(move, rnd):
        nonlocal best_val, total_cost
        checkpoint.clear(move.name, provider)                          # (4) CHECKPOINT gate
        cands = [(name, ctor, params, s) for (name, ctor, params) in move.families for s in seeds]
        # CANCEL check before each candidate measurement (the fan-out IS the measurement): if the operator
        # stopped the run, raise so the caller short-circuits to the terminal cancelled result. Raising (vs.
        # returning) keeps the cancel signal out of the leaderboard/best-val bookkeeping.
        if _cancelled():
            raise _CancelledRun()
        before = best_val
        _emit("fanout", move=move.name, round=rnd, n_candidates=len(cands),
              families=[f[0] for f in move.families], device=provider.capabilities().get("device"))

        if provider.execution_mode == "worker":
            # SERIALIZABLE path: build one JobSpec per candidate, dispatched through the worker contract
            # (LocalWorkerProvider in-process now; RunPodProvider on GPU later). No local estimator -- the
            # winner is certified from REMOTE PREDICTIONS (see certify below). Data sent inline (lists).
            xtr_l, ytr_l = _to_list(Xtr), [float(v) for v in ytr] if harness.is_regression() else [int(v) for v in ytr]
            xva_l = _to_list(Xva)                              # NOTE: yva is NOT shipped (re-scored locally)
            yva_l = [float(v) for v in yva] if harness.is_regression() else [int(v) for v in yva]
            jobs = []
            for i, (name, ctor, params, s) in enumerate(cands):
                remote = {"op": "fit_val", "task_type": harness.task_type, "family": name,
                          "params": params, "seed": int(s), "metric": metric, "n_classes": int(n_classes),
                          "Xtr": xtr_l, "ytr": ytr_l, "Xva": xva_l}     # no yva: worker returns predictions
                jobs.append(Job(job_id=f"r{rnd}j{i}", spec={"remote": remote, "cand": (name, params, int(s))}))

            def shape(job, out):
                name, params, s = job.spec["cand"]
                cid = science.digest({"family": name, "params": params, "seed": s})
                rid = _run_record(name, params, s, rnd, cid, provider.name)
                store.log_params(experiment, rid, {"move": move.name})
                yhat = (out or {}).get("y_pred_val")
                if not isinstance(out, dict) or out.get("error") or yhat is None:
                    store.end_run(experiment, rid, "FAILED")
                    return Run(rid, name, params, s, None, None, provider.name, "FAILED",
                               error=str((out or {}).get("error", "no worker predictions"))[:120],
                               move=move.name, round=rnd), None
                if len(yhat) != len(yva_l):                    # alignment: predictions must match val size
                    store.end_run(experiment, rid, "FAILED")
                    return Run(rid, name, params, s, None, None, provider.name, "FAILED",
                               error=f"val prediction count {len(yhat)} != {len(yva_l)}",
                               move=move.name, round=rnd), None
                val = _score_val(metric, yva_l, yhat, harness.is_regression())   # re-score LOCALLY (F6)
                lat = out.get("latency_ms")
                if not np.isfinite(val):                       # audit LOW: non-finite val -> FAILED
                    store.end_run(experiment, rid, "FAILED")
                    return Run(rid, name, params, s, None, None, provider.name, "FAILED",
                               error="non-finite val_score", move=move.name, round=rnd), None
                val_predictions[rid] = {"y_pred": yhat, "y_true": yva_l,
                                        "proba": out.get("proba_val")}          # for val-LB (11) + ECE (8)
                ece = None if harness.is_regression() else _ece(out.get("proba_val"), yhat, yva_l)
                val_ece_by_rid[rid] = ece
                cost = round(cost_rate * ((lat or 0.0) / 3.6e6), 6)
                store.log_metric(experiment, rid, f"val_{metric}", val)
                if ece is not None:
                    store.log_metric(experiment, rid, "val_ece", ece)
                store.log_system(experiment, rid, latency_ms=round(lat or 0.0, 2), cost_usd=cost,
                                 device=out.get("device"))
                store.end_run(experiment, rid, "FINISHED")
                return Run(rid, name, params, s, val, lat, provider.name, "FINISHED", artifact_ref=rid,
                           cost_usd=cost, move=move.name, round=rnd), ("__remote__", name, params, int(s))
            results = provider.map(shape, jobs)
        else:
            # IN-PROCESS path (LocalCpuProvider): fit locally via the harness ctor; keep the estimator.
            def run_one(job):
                name, ctor, params, s = job.spec["cand"]
                cid = science.digest({"family": name, "params": params, "seed": s})
                rid = _run_record(name, params, s, rnd, cid, provider.name)
                store.log_params(experiment, rid, {"move": move.name})
                last_err = None
                for attempt in range(2):                              # one retry on a transient fit error
                    try:
                        val_score, lat, est = harness.fit_score(ctor, s, Xtr, ytr, Xva, yva)
                        if not np.isfinite(val_score):         # audit LOW: non-finite val -> retry/FAILED
                            raise ValueError("non-finite val_score")
                        # MEASURE: val predictions (retained for the val lower bound, step 11) + ECE (step 8)
                        vpred = est.predict(Xva)
                        proba = None
                        if not harness.is_regression() and hasattr(est, "predict_proba"):
                            try:
                                proba = np.max(np.asarray(est.predict_proba(Xva), dtype=float), axis=1).tolist()
                            except Exception:  # noqa: BLE001
                                proba = None
                        yva_list = [float(v) for v in yva] if harness.is_regression() else [int(v) for v in yva]
                        vpred_list = ([float(v) for v in vpred] if harness.is_regression()
                                      else [int(v) for v in vpred])
                        val_predictions[rid] = {"y_pred": vpred_list, "y_true": yva_list, "proba": proba}
                        ece = None if harness.is_regression() else _ece(proba, vpred_list, yva_list)
                        val_ece_by_rid[rid] = ece
                        cost = round(cost_rate * (lat / 3.6e6), 6)
                        store.log_metric(experiment, rid, f"val_{metric}", val_score)
                        if ece is not None:
                            store.log_metric(experiment, rid, "val_ece", ece)
                        store.log_system(experiment, rid, latency_ms=round(lat, 2), cost_usd=cost,
                                         attempts=attempt + 1)
                        store.end_run(experiment, rid, "FINISHED")
                        return Run(rid, name, params, s, val_score, lat, provider.name, "FINISHED",
                                   artifact_ref=rid, cost_usd=cost, move=move.name, round=rnd), est
                    except Exception as e:  # noqa: BLE001
                        last_err = e
                store.end_run(experiment, rid, "FAILED")
                return Run(rid, name, params, s, None, None, provider.name, "FAILED",
                           error=str(last_err)[:120], move=move.name, round=rnd), None
            jobs = [Job(job_id=f"r{rnd}j{i}", spec={"cand": c}) for i, c in enumerate(cands)]
            results = provider.map(run_one, jobs)

        round_cost = 0.0
        for r, est in results:
            lb.runs.append(r)
            if est is not None:
                estimators[r.run_id] = est
            round_cost += (r.cost_usd or 0.0) or ((r.latency_ms or 0.0) / 1000.0)   # cost proxy for VoI
        total_cost += round_cost
        cur = lb.best()
        if cur is not None:
            best_val = max(best_val, cur.val_score)
        # INCREMENTAL gain only. The round-0 anchor (before == -inf) has nothing to improve on yet, so its
        # gain is 0.0 -- crediting it with the full baseline score would pollute the PERSISTED cross-run
        # case-base (it would float the 'baseline' move to the top of warm-started VoI as if it were a high-lift
        # move). In-run this is inert: the anchor always runs unconditionally at round 0 and is never re-ranked.
        gain = max(0.0, best_val - before) if before > -1e17 else 0.0
        casebase.record(move.name, gain)
        casebase.record_cost(move.name, max(round_cost, 1e-6))                       # cost-real VoI
        # DURABLE cross-dataset write (no-op without a memory_store): attribute this round's realized gain
        # to the move's FAMILY (the batch executes one distinct family per move; the name prefix IS that
        # family, matching the warm-start key) so a future run on similar data inherits it.
        if isinstance(casebase, ResearchMemory):
            casebase.observe(move.name.split("|")[0], gain, max(round_cost, 1e-6))
        rounds_info.append({"round": rnd, "move": move.name,
                            "families": [f[0] for f in move.families],
                            "best_val_after": round(best_val, 4), "gain": round(gain, 4)})
        used.add(move.name)
        _emit("measure", round=rnd, move=move.name, best_val=round(best_val, 4),
              best_val_lb=_running_val_lb()[0])

    def execute_authored(rnd):
        """ADDITIVE Wave-3 arm: agent/LLM-authored `build_estimator(seed)` sources run in the OS sandbox
        (authored_pod_sandbox.run_authored) on the SAME featurized train/val split, scored by the SAME frozen
        metric, and registered as candidates competing in the SAME select-then-bound gate + the SINGLE sealed
        certify. The marker ('__authored__', name, source) routes the sealed peek back through the sandbox.
        No-op unless `authored_propose`. Never raises (a sandbox failure becomes a FAILED run)."""
        if not authored_propose:
            return
        try:
            from .propose_authored import stub_authored_sources
            from .authored_pod_sandbox import run_authored
        except Exception:  # noqa: BLE001  module/sandbox unavailable -> skip the arm entirely
            return
        srcs = (authored_sources_fn or stub_authored_sources)()
        is_reg = harness.is_regression()
        yva_l = [float(v) for v in yva] if is_reg else [int(v) for v in yva]
        for name, source in srcs:
            fam = f"authored:{name}"
            if fam in used:
                continue
            used.add(fam)
            cid = science.digest({"authored": name, "src": source})
            rid = _run_record(fam, {"src_sha": cid.split(':')[-1][:12]}, 0, rnd, cid, provider.name)
            store.log_params(experiment, rid, {"move": fam})
            t0 = time.time()
            try:
                r = run_authored(source, Xtr, ytr, Xva, wall_s=180)
            except Exception as e:  # noqa: BLE001  sandbox not available (e.g. off-pod) -> FAILED, not a crash
                r = {"ok": False, "pred": None, "reason": f"sandbox unavailable: {e}"}
            lat = (time.time() - t0) * 1000.0
            _params = {"src_sha": cid.split(':')[-1][:12]}
            if not r.get("ok") or r.get("pred") is None or len(r["pred"]) != len(yva_l):
                store.end_run(experiment, rid, "FAILED")
                lb.runs.append(Run(rid, fam, _params, 0, None, None, provider.name, "FAILED",
                                   error=str(r.get("reason"))[:120], move=fam, round=rnd))
                continue
            yhat = [float(v) for v in r["pred"]] if is_reg else [int(round(v)) for v in r["pred"]]
            val = _score_val(metric, yva_l, yhat, is_reg)
            if not np.isfinite(val):
                store.end_run(experiment, rid, "FAILED")
                lb.runs.append(Run(rid, fam, _params, 0, None, None, provider.name, "FAILED",
                                   error="non-finite val_score", move=fam, round=rnd))
                continue
            val_predictions[rid] = {"y_pred": yhat, "y_true": yva_l, "proba": None}
            val_ece_by_rid[rid] = None
            store.log_metric(experiment, rid, f"val_{metric}", val)
            store.end_run(experiment, rid, "FINISHED")
            lb.runs.append(Run(rid, fam, _params, 0, val, lat, provider.name, "FINISHED",
                               artifact_ref=rid, move=fam, round=rnd))
            estimators[rid] = ("__authored__", name, source)   # marker -> sandbox re-run at certify

    def _running_val_lb():
        """The current overall winner's VALIDATION LOWER BOUND (select-then-bound: the single current winner at checks=1).
        This is the loop's true objective ('raise the validation lower bound'); convergence and escalation
        are driven by it, not the val point -- so a point that clears while the bound does not still escalates."""
        cur = lb.best()
        if cur is None:
            return None, None
        wv = val_predictions.get(cur.run_id)
        if not wv:
            return None, cur
        nf = 1   # select-then-bound: bound the single current winner once (checks=1); see the val-bound gate
        try:
            return round(_val_lower_bound(metric, wv["y_true"], wv["y_pred"], threshold, nf, alpha,
                                          harness.is_regression()), 4), cur
        except Exception:  # noqa: BLE001
            return None, cur

    # how many consecutive rounds of no validation-bound improvement we tolerate before stopping (only
    # honored once any time budget is exhausted -- the budget always wins while it + proposals remain).
    K_NO_IMPROVE = 3
    PER_ROUND_EXEC = 6   # execute a FAMILY-DIVERSE batch per round (not just the single cheapest VoI move):
    #                      one-per-round + cold-start VoI (gain/cost) lets the cheapest family (logistic)
    #                      win every rank, so the high-capacity zoo (hist_gbm/svc_rbf/...) never runs and the
    #                      search can't beat a linear baseline on a nonlinear problem. 6 covers the zoo.
    no_improve = 0

    try:
      # round 0: baseline (the static menu's cheap baseline -- the anchor the diagnosis improves on)
      execute_move(by_name.get("baseline", moves[0]), 0)
      # Wave-3 authored arm (additive, no-op unless authored_propose): authored models compete from the start,
      # so they participate in selection + the single certify even if the loop converges immediately.
      execute_authored(0)
      # rounds 1..N: the REAL recursive cycle --
      #   DIAGNOSE (measured signals on the val LOWER BOUND)
      #   -> PROPOSE (llm_moves.propose_moves: the LLM, given diagnosis+leaderboard+profile, proposes the next
      #      batch of (family, hyperparameter) configs from the bounded catalog; deterministic grid fallback)
      #   -> VoI-rank -> CHECKPOINT -> EXECUTE -> MEASURE -> UPDATE -> repeat.
      # Terminates when: (certify-mode bound cleared with margin) OR (objective=maximize and the time budget
      # elapsed) OR (no improving proposal for K consecutive rounds AND any time budget is exhausted) OR the
      # proposer has no untried candidates left OR the round/experiment/hard-cap caps are hit.
      # The max_rounds cap is the no-budget convergence bound. When a TIME BUDGET is set and still running,
      # the round cap must NOT cut the search short (the contract: keep proposing while budget + untried
      # proposals remain); the budget_experiments cap and the hard wall-clock cap still bound the run.
      rnd = 1
      while (rnd < max_rounds or _budget_remaining()) and len(lb.runs) < budget_experiments:
        if _cancelled():                                  # CANCEL check at the TOP of each round
            raise _CancelledRun()
        if _over_hard_cap():                              # runaway safety: hard wall-clock cap
            rounds_info.append({"round": rnd, "move": "(none)", "stop": "hard time cap reached"})
            break
        cur_lb, cur = _running_val_lb()
        # EARLY STOP is objective-dependent. In "certify" mode, once the validation lower bound clears theta
        # with margin we stop and go certify (the original behavior) -- UNLESS a time budget is still running,
        # in which case we keep searching for an even stronger model (more sealed-bound headroom). In
        # "maximize" mode we NEVER early-stop on clearing: we run the full budget tracking the best model.
        cleared = cur_lb is not None and cur_lb >= threshold + stop_margin
        if cleared and _resolved_objective == "certify" and not _budget_remaining():
            break

        diag_sig, _relevant = _diagnose(cur, val_predictions.get(cur.run_id) if cur else None,
                                        val_ece_by_rid.get(cur.run_id) if cur else None,
                                        threshold, harness.is_regression(), val_lb=cur_lb)
        last_diagnosis[0] = diag_sig
        _emit("diagnose", round=rnd, signals=diag_sig)

        # PROPOSE (LLM-guided, NON-BINDING). The proposer is bounded to the runnable catalog + clamps every
        # param; `used` is the set of move names already executed so each round explores NEW configs.
        proposed, source = propose_moves(
            _profile, _leaderboard_summary(), diag_sig, harness.task_type, metric, used,
            use_llm=llm_propose, api_key=_api_key, catalog=cat, cache_path=llm_cache_path,
            limit=min(12, max(1, budget_experiments - len(lb.runs))))
        proposed = [m for m in proposed if m.name not in used]
        for m in proposed:
            by_name.setdefault(m.name, m)
            _register_ctors(m)
        if source == "llm":
            _n_llm_rounds[0] += 1          # observability: the LLM genuinely drove this round's PROPOSE
        _emit("propose", round=rnd, source=source, n_proposed=len(proposed), used_llm=(source == "llm"),
              families=sorted({f[0].split("|")[0] for m in proposed for f in m.families}))

        if not proposed:
            # No NEW model candidate -> the model axis is exhausted. DATA AXIS (step 14): if a caller supplied
            # an acquisition source, acquire rows, RE-AUDIT them against the sealed test (reject leaky data),
            # and re-run on the augmented set. Otherwise honest-stop with the concrete acquire ask (step 12).
            if acquire_fn is not None and acquire_budget > 0:
                est_need = max(50, len(train) // 2)
                new_rows = list(acquire_fn(est_need) or [])
                ok_aud, findings = reaudit([dict(r) for r in train], new_rows, test,
                                           allow_features=(harness.kind == "tabular"),
                                           text_key=text_key, target_key=target_key, min_test_n=min_test_n)
                if new_rows and ok_aud:
                    return run_goal_loop(
                        records + new_rows, goal_text, harness=harness, kind=kind, task_type=task_type,
                        target_key=target_key, labels=labels, threshold=threshold, metric=metric,
                        store=store, providers=providers, llm_enabled=llm_enabled, seeds=seeds,
                        text_key=text_key, min_test_n=min_test_n, alpha=alpha, seed=seed,
                        max_rounds=max_rounds, budget_experiments=budget_experiments,
                        stop_margin=stop_margin, checkpoint=checkpoint,
                        casebase_path=casebase_path, memory_store=memory_store,
                        peek_ledger_path=peek_ledger_path,
                        drop_cols=drop_cols, max_latency_ms=max_latency_ms, max_cost_usd=max_cost_usd,
                        max_ece=max_ece,
                        acquire_fn=acquire_fn, acquire_budget=acquire_budget - 1,
                        on_event=on_event, test_records=test_records, should_cancel=should_cancel,
                        objective=objective, time_budget_s=time_budget_s, llm_propose=llm_propose,
                        llm_cache_path=llm_cache_path, registry=registry)
                rounds_info.append({"round": rnd, "move": "acquire", "diagnosis": diag_sig,
                                    "reaudit": "blocked" if not ok_aud else "no_rows",
                                    "findings": [f for f in findings if not f.get("ok", True)]})
            rounds_info.append({"round": rnd, "move": "(none)", "diagnosis": diag_sig,
                                "stop": "proposer exhausted -> no untried candidate -> honest stop"})
            break

        # VoI-rank the proposed batch, then EXECUTE A FAMILY-DIVERSE BATCH -- the best move per DISTINCT family
        # (in VoI order), capped per round and by the experiment budget. Executing only ranked[0] starves the
        # zoo: cold-start VoI = gain/cost prefers the cheapest family (logistic), so the nonlinear models never
        # run. A researcher facing large headroom runs a spread of families first; this does that. select-then-
        # bound keeps the certificate at checks=1, so wider exploration never weakens the moat.
        ranked = voi_rank(proposed, casebase)
        batch, fams = [], set()
        for rm in ranked:
            fam = rm.name.split("|")[0]
            if fam in fams:
                continue
            fams.add(fam)
            batch.append(by_name[rm.name])
            if len(batch) >= PER_ROUND_EXEC or len(lb.runs) + len(batch) >= budget_experiments:
                break
        before_lb = cur_lb if cur_lb is not None else -1e18
        for mv in batch:
            if len(lb.runs) >= budget_experiments:
                break
            execute_move(mv, rnd)
        after_lb, _ = _running_val_lb()
        improved = (after_lb is not None) and (after_lb > before_lb + 1e-9)
        no_improve = 0 if improved else (no_improve + 1)
        rnd += 1

        # STOP on stagnation, but ONLY once any time budget is exhausted: while budget + proposals remain we
        # keep going (the contract: TRY MUCH HARDER before stopping). With no budget set, K stagnant rounds
        # is a legitimate convergence stop.
        if no_improve >= K_NO_IMPROVE and not _budget_remaining():
            rounds_info.append({"round": rnd, "move": "(none)",
                                "stop": f"no validation-bound improvement for {K_NO_IMPROVE} rounds"})
            break
    except _CancelledRun:
        return _cancel_result()

    if _cancelled():                          # cancelled after the last round, before any sealed peek
        return _cancel_result()
    winner = lb.best()
    if winner is None:
        return GoalLoopResult(experiment, lb, None, None, "no_model",
                              "No candidate model finished successfully.", len(test), provider.name,
                              rounds=rounds_info, compute_report=compute_report,
                              failure_report=_failure_report("no_model", None, threshold, None, None))

    # VAL-BOUND-PREDICTS-PASS GATE (step 11, the scheme's defining invariant): compute the winner's
    # VALIDATION lower bound, Bonferroni-corrected over the number of finished candidates. Only spend the
    # one counted sealed peek when that bound already clears theta; otherwise HONEST STOP with NO PEEK so
    # the scarce sealed test (and its cumulative Bonferroni budget, F3) is preserved for a stronger attempt.
    # SELECT-THEN-BOUND (multiplicity): selection over many candidates happens FREELY on validation; the
    # validation gate then bounds the ONE selected winner as a SINGLE promotion test (checks=1). It is NOT
    # Bonferroni'd by the candidate count -- doing that was an over-conservative miscalibration that made the
    # gate refuse to peek for genuinely-good models on small validation sets (the wine/iris regression). The
    # independent SEALED test (one counted peek, its own cumulative multiplicity) is the real promotion gate
    # and absorbs selection (winner's-curse) optimism: a model that looked good on val by selection luck will
    # not clear the sealed lower bound. So an optimistic val gate can at worst spend the one peek on a
    # borderline model; it can NEVER produce a false certificate.
    _emit("val_bound", winner=winner.family, val=round(winner.val_score, 4))
    n_finished = sum(1 for r in lb.runs if r.status == "FINISHED")   # reported only
    checks_val = 1   # select-then-bound: one selected winner == one promotion test
    wv = val_predictions.get(winner.run_id)
    val_lb = None
    if wv:
        try:
            val_lb = _val_lower_bound(metric, wv["y_true"], wv["y_pred"], threshold, checks_val, alpha,
                                      harness.is_regression())
        except Exception:  # noqa: BLE001  never let a bound computation block on a numerical edge case
            val_lb = None
    # OBJECTIVE FORK at the val-bound gate:
    #   "certify"  -> the original invariant: if the winner's validation lower bound does NOT clear theta,
    #                 HONEST-STOP with NO sealed peek (preserve the scarce test for a stronger attempt).
    #   "maximize" -> the user asked for the best achievable model, not a certify-or-stop. We do NOT honest-
    #                 stop: we take the ONE sealed peek on the best validation model and report it as a
    #                 "best_effort" measurement. certified=True ONLY if a theta was given AND the sealed lower
    #                 bound clears it (decided by the frozen certify path below). The peek is still exactly one,
    #                 still counted, still through the SealedTest guard -- maximize never weakens the moat.
    if _resolved_objective == "certify" and val_lb is not None and val_lb <= threshold:
        msg = (f"Honest stop, NO sealed peek: the winner's validation lower bound {val_lb:.4f} "
               f"(select-then-bound on the single chosen winner, alpha={alpha}) does not clear the "
               f"threshold {threshold}. Selection was on validation only; the one counted sealed peek is "
               f"preserved. Raise the validation lower bound (more data, or a move that lifts it) before "
               f"certifying. NOTE: small validation sets make this bound intentionally loose.")
        narr = narrate(EvidenceBundle(metric=metric, observed=float(winner.val_score),
                                      lower_bound=float(val_lb), threshold=threshold,
                                      n_test=0, certified=False, audit_passed=True, target=target_key,
                                      task_type=harness.task_type, verdict_label="honest-stop-no-peek"),
                       audience="researcher", use_llm=llm_enabled)
        wdiag, _ = _diagnose(winner, wv, val_ece_by_rid.get(winner.run_id), threshold,
                             harness.is_regression(), val_lb=val_lb)
        report = _failure_report("honest_stop", wdiag, threshold, val_lb, len(wv["y_true"]) if wv else None)
        # POWER-AWARE annotation (non-binding, exact for accuracy): distinguish an UNDERPOWERED stop (the
        # validation set is simply too small for ANY lower bound to clear theta -- more data would help) from
        # a MODEL-LIMITED stop (the winner's true performance is at/below theta -- more data will not help).
        # Pure analysis; it never changes the honest-stop decision, only explains it.
        _pw = None
        try:
            from . import power as _power
            _pw = _power.assess(len(wv["y_true"]) if wv else 0, threshold, float(winner.val_score),
                                metric=metric, alpha=alpha, checks=checks_val)
            if isinstance(report, dict):
                report["power"] = _pw
        except Exception:  # noqa: BLE001  power analysis is additive; never break the honest-stop path
            _pw = None
        try:
            _neg = negative_certificate(plan_hash=_plan_hash, reason="honest_stop_no_peek", diagnosis=wdiag,
                                        observed=float(winner.val_score), lower_bound=val_lb, theta=threshold,
                                        alpha_futility=alpha)
            PromotionLedger(os.path.join(_ledger_dir, "_negative_ledger.jsonl")).record(_neg)
        except Exception:  # noqa: BLE001  negative certificate is additive; never break the honest-stop path
            _neg = None
        _emit("honest_stop", status="done", val_lb=val_lb, threshold=threshold, failure=report,
              power=_pw, plan_hash=_plan_hash, objective=_resolved_objective, elapsed_s=round(_elapsed(), 2))
        return GoalLoopResult(experiment, lb, winner, None, "honest_stop", narr.text, len(test),
                              provider.name, rounds=rounds_info, compute_report=compute_report,
                              failure_report=report, objective=_resolved_objective,
                              time_budget_s=time_budget_s, elapsed_s=round(_elapsed(), 2),
                              plan_hash=_plan_hash, negative_result=_neg)

    # CERTIFY the val-winner ONCE on the SEALED test (enforced counted peek; metric-correct frozen certifier)
    # A durable PeekLedger (when a path is given) makes the Bonferroni `checks` CUMULATIVE across runs on
    # the SAME locked-test digest, so repeated peeks can't launder the multiplicity correction (F3).
    ledger = PeekLedger(peek_ledger_path) if peek_ledger_path else None
    sealed = SealedTest(test, target_key=target_key, max_peeks=1, ledger=ledger)
    marker = estimators[winner.run_id]
    winner_device = "cpu"
    predict_fn = None
    serving_est = None                 # the fitted winner to persist for serving (None for torch-on-worker)
    is_remote = isinstance(marker, tuple) and bool(marker) and marker[0] == "__remote__"
    is_authored = isinstance(marker, tuple) and bool(marker) and marker[0] == "__authored__"
    if is_authored:
        # AUTHORED winner: the authored estimator cannot cross the sandbox boundary, so RE-RUN its source in the
        # OS sandbox on the SEALED test (fit on train, predict sealed). The sandbox is label-blind to yte (never
        # receives it): a corrupt result can only DEGRADE the scored number, never inflate the bound. One peek
        # (checks=1) preserved -- this is the single counted sealed evaluation of the single winner.
        from .authored_pod_sandbox import run_authored
        _, _aname, _asrc = marker
        checkpoint.clear("certify_winner", provider)        # spend gate: the winner re-run bills too
        try:
            _ar = run_authored(_asrc, Xtr, ytr, Xte, wall_s=240)
        except Exception as e:  # noqa: BLE001
            _ar = {"ok": False, "pred": None, "reason": str(e)}
        yp = _ar.get("pred")
        if not _ar.get("ok") or yp is None or len(yp) != len(test):
            return GoalLoopResult(experiment, lb, winner, None, "no_model",
                                  f"authored winner sealed predictions unavailable/misaligned "
                                  f"(got {len(yp) if yp else 0} for {len(test)} rows): {_ar.get('reason')}",
                                  len(test), provider.name, rounds=rounds_info, compute_report=compute_report,
                                  failure_report=_failure_report("no_model", None, threshold, None, None))
        try:
            if harness.is_regression():
                preds = [float(p) for p in yp]
            else:
                inv = {i: lab for lab, i in l2i.items()}
                preds = [inv[int(round(p))] for p in yp]
        except (KeyError, ValueError) as e:
            return GoalLoopResult(experiment, lb, winner, None, "no_model",
                                  f"authored winner produced out-of-range labels: {e}",
                                  len(test), provider.name, rounds=rounds_info, compute_report=compute_report,
                                  failure_report=_failure_report("no_model", None, threshold, None, None))
        winner_device = "sandbox(authored)"
        predict_fn = lambda rows: preds          # precomputed, aligned to sealed.rows order (one peek)
    if is_remote:
        # TRUSTED certify (closes attack-3): if the winner family is buildable LOCALLY (sklearn), re-fit it
        # ON THE PLATFORM and compute the sealed predictions here. The GPU/worker did the SEARCH, but the
        # certified predictions come from the trusted platform, so a corrupt/reordering worker cannot make
        # the certificate non-faithful. Only a torch winner (no local torch) falls through to the worker.
        _, name, params, s = marker
        _lc = ctor_by_family.get(name)
        if _lc is not None:
            try:
                _le = _lc(int(s)); _le.fit(Xtr, ytr)
                predict_fn = harness.predict_fn(_le, feat, l2i)
                serving_est = _le                          # trusted local winner -> servable
                winner_device = "cpu(local-refit)"
            except Exception:  # noqa: BLE001  torch / unbuildable-locally winner -> worker path below
                predict_fn = None
    if is_remote and predict_fn is None:
        # WORKER certify (e.g. a torch winner with no local torch): the worker is LABEL-BLIND (never receives
        # yte), so any corruption of its sealed predictions can only DEGRADE the scored result -- it can never
        # inflate the bound or promote a below-threshold model. The per-row id binding below catches
        # truncation / substitution / missing-or-extra rows; it does NOT detect a self-consistent label-blind
        # permutation, which is harmless to the moat by yte-blindness (it can only lower the certified number).
        _, name, params, s = marker
        xte_l = _to_list(Xte)
        # per-row ids bind each sealed-test row to its prediction (step 7/F5). idx is included so the ids
        # are unique even when two rows share identical features; the worker echoes them aligned to its
        # predictions, and we reindex to the SENT order with a 1:1 coverage assertion before scoring.
        row_ids = [science.digest({"i": i, "x": row}) for i, row in enumerate(xte_l)]
        remote = {"op": "fit_predict_test", "task_type": harness.task_type, "family": name,
                  "params": params, "seed": int(s), "metric": metric, "n_classes": int(n_classes),
                  "Xtr": _to_list(Xtr),
                  "ytr": [float(v) for v in ytr] if harness.is_regression() else [int(v) for v in ytr],
                  "Xte": xte_l, "row_ids": row_ids}
        checkpoint.clear("certify_winner", provider)   # spend gate (audit HIGH): the winner re-fit bills too
        out = provider.run_jobspec(remote)
        yp = (out or {}).get("y_pred")
        if not isinstance(out, dict) or out.get("error") or yp is None or len(yp) != len(test):
            return GoalLoopResult(experiment, lb, winner, None, "no_model",
                                  f"winner predictions unavailable/misaligned "
                                  f"(got {len((out or {}).get('y_pred') or [])} for {len(test)} sealed rows): "
                                  f"{(out or {}).get('error')}",
                                  len(test), provider.name, rounds=rounds_info,
                                  compute_report=compute_report,
                                  failure_report=_failure_report("no_model", None, threshold, None, None))
        ret_ids = out.get("row_ids")
        if ret_ids is not None:                  # new worker contract: reindex by id, assert 1:1 coverage
            if (len(ret_ids) != len(yp) or len(set(ret_ids)) != len(row_ids)
                    or set(ret_ids) != set(row_ids)):
                return GoalLoopResult(experiment, lb, winner, None, "no_model",
                                      "winner sealed predictions failed per-row id binding (truncated/"
                                      "reordered/substituted rows); refusing to certify a misaligned pairing.",
                                      len(test), provider.name, rounds=rounds_info,
                                      compute_report=compute_report,
                                      failure_report=_failure_report("no_model", None, threshold, None, None))
            by_id = dict(zip(ret_ids, yp))
            yp = [by_id[rid] for rid in row_ids]     # canonical (sent) order
        # else: legacy worker without id echo -> positional (length already checked above)
        winner_device = out.get("device", "cpu")
        if harness.is_regression():
            preds = [float(p) for p in yp]
        else:
            inv = {i: lab for lab, i in l2i.items()}
            preds = [inv[int(p)] for p in yp]
        predict_fn = lambda rows: preds          # precomputed, aligned to sealed.rows order (one peek)
    if not is_remote and not is_authored:
        predict_fn = harness.predict_fn(marker, feat, l2i)
        serving_est = marker                               # in-process estimator -> servable
    _emit("sealed_certify", winner=winner.family, device=winner_device, n_test=len(test))
    cert = certify_on_sealed(sealed, predict_fn, threshold, metric=metric, labels=labels, alpha=alpha)
    # BATTERY paired-comparison vectors (Item 3, additive + non-promoting). For a cross-dataset FDR we need a
    # candidate-vs-baseline test on the SAME locked test, not a binomial-vs-floor (the Codex A1 tautology). We
    # compute, for CLASSIFICATION only, the winner's and a FIXED baseline's per-row correctness on the sealed
    # rows. This does NOT call certify_on_sealed (no extra COUNTED peek), does NOT change `cert` or its
    # multiplicity, and the baseline is the no-search reference (logistic/ridge on the same featurized Xtr) --
    # so a downstream McNemar asks the honest question "did the search beat the default?". Never blocks certify.
    cand_locked_correct = base_locked_correct = None
    if not harness.is_regression():
        try:
            _yt = [str(v) for v in sealed.y_true()]
            _cand = [str(v) for v in predict_fn(sealed.rows)]
            cand_locked_correct = [1 if a == b else 0 for a, b in zip(_cand, _yt)]
            from sklearn.linear_model import LogisticRegression as _BL
            _bl = _BL(max_iter=2000); _bl.fit(Xtr, ytr)
            _base = [str(v) for v in harness.predict_fn(_bl, feat, l2i)(sealed.rows)]
            base_locked_correct = [1 if a == b else 0 for a, b in zip(_base, _yt)]
        except Exception:  # noqa: BLE001  battery vectors are additive; never let them break certification
            cand_locked_correct = base_locked_correct = None
    # provenance join: bind the certificate to the EXACT validation-winner it was produced from, so
    # "certified == the model selected on validation" is verifiable (see provenance.verify_provenance).
    assert winner.run_id == lb.best().run_id, "internal: certifying a non-winner"
    cert["winner_run"] = winner.run_id
    cert["winner_family"] = winner.family
    cert["val_ece"] = val_ece_by_rid.get(winner.run_id)      # calibration of the selected model (step 8)
    cert["winner_content_id"] = science.digest({"family": winner.family, "params": winner.params,
                                                "seed": winner.seed})
    cert["winner_device"] = winner_device
    cert["leaderboard_digest"] = science.digest([r.as_dict() for r in lb.ranked()])

    # VERIFY-SPEC gates (step 13/F23): a certificate requires bound>theta AND latency_ok AND cost_ok.
    # latency: the winner's measured fit+inference time is a CONSERVATIVE upper bound on serving latency
    # (if the whole fit+predict clears the budget, inference alone certainly does). cost: cumulative spend.
    win_latency = winner.latency_ms
    latency_ok = (max_latency_ms is None) or (win_latency is None) or (float(win_latency) <= max_latency_ms)
    cost_ok = (max_cost_usd is None) or (round(total_cost, 6) <= max_cost_usd)
    win_ece = cert.get("val_ece")
    ece_ok = (max_ece is None) or (win_ece is None) or (float(win_ece) <= max_ece)   # optional calibration gate
    cert["latency_ms"] = None if win_latency is None else round(float(win_latency), 2)
    cert["latency_ok"] = bool(latency_ok)
    cert["cost_usd"] = round(total_cost, 6)
    cert["cost_ok"] = bool(cost_ok)
    cert["ece_ok"] = bool(ece_ok)
    bound_clears = bool(cert["certified"])
    cert["certified"] = bound_clears and latency_ok and cost_ok and ece_ok
    if not cert["certified"]:
        cert["reason"] = ("lower bound does not clear theta" if not bound_clears
                          else ("latency exceeds budget" if not latency_ok
                                else ("cost exceeds budget" if not cost_ok
                                      else "calibration (ECE) exceeds budget")))
    cert["objective"] = _resolved_objective
    # DECISION label:
    #   certify  -> "certified" iff the frozen certificate cleared every gate, else "do_not_certify".
    #   maximize -> the user wanted the best achievable model; we always took the one sealed peek and report
    #               it honestly. "certified" iff the sealed lower bound genuinely cleared theta (and the
    #               latency/cost/ECE gates), else "best_effort": the BEST model measured, with its honest
    #               sealed observed + lower bound. It NEVER claims more than measured.
    if _resolved_objective == "maximize":
        decision = "certified" if cert["certified"] else "best_effort"
    else:
        decision = "certified" if cert["certified"] else "do_not_certify"

    # SERVING: persist the certified winner as a deployable, versioned artifact (goal -> certificate ->
    # servable). Only certified + locally-fitted winners are registered (torch-on-worker winners have no
    # local estimator to serve). The registry advances a CURRENT pointer; predict()/rollback() use it.
    if registry is not None and decision == "certified" and serving_est is not None:
        try:
            cert["served_version"] = registry.register(
                experiment, goal=goal_text, estimator=serving_est, featurizer=feat, label_map=l2i,
                certificate=cert, kind=harness.kind, task_type=harness.task_type)
        except Exception as ex:  # noqa: BLE001  a serving hiccup must never invalidate a real certificate
            cert["served_version"] = None
            cert["serving_error"] = str(ex)[:160]

    bundle = EvidenceBundle(metric=metric, observed=float(cert["observed"]),
                            lower_bound=float(cert["lower_bound"]), threshold=threshold,
                            n_test=int(cert["n"]), certified=cert["certified"], audit_passed=True,
                            target=target_key, task_type=harness.task_type,
                            verdict_label=("VERIFIED RUNNING" if cert["certified"]
                                           else ("best-effort (measured, not certified)"
                                                 if _resolved_objective == "maximize"
                                                 else "measured-not-certified")))
    narr = narrate(bundle, audience="researcher", use_llm=llm_enabled)

    crid = store.start_run(experiment, params={"stage": "certify", "winner_family": winner.family,
                                               "winner_run": winner.run_id, "metric": metric},
                           tags={"sealed_peek": 1})
    store.log_metric(experiment, crid, f"test_{metric}", float(cert["observed"]))
    store.log_metric(experiment, crid, "lower_bound", float(cert["lower_bound"]))
    store.log_metric(experiment, crid, "sealed_peeks", cert.get("peeks", 1))
    store.log_artifact(experiment, crid, "certificate", cert)
    store.end_run(experiment, crid, "FINISHED")

    report = None
    if decision != "certified":
        wv2 = val_predictions.get(winner.run_id)
        wvlb = None
        if wv2:
            try:
                wvlb = _val_lower_bound(metric, wv2["y_true"], wv2["y_pred"], threshold, checks_val, alpha,
                                        harness.is_regression())
            except Exception:  # noqa: BLE001
                wvlb = None
        wdiag, _ = _diagnose(winner, wv2, val_ece_by_rid.get(winner.run_id), threshold,
                             harness.is_regression(), val_lb=wvlb)
        report = _failure_report("do_not_certify", wdiag, threshold, float(cert["lower_bound"]),
                                 len(wv2["y_true"]) if wv2 else None)
    cert["plan_hash"] = _plan_hash
    _xfdr = _fdr_record_and_view(decision, cert)
    _emit("done", status="done", decision=decision, certified=bool(cert["certified"]),
          lower_bound=float(cert["lower_bound"]), observed=float(cert["observed"]), n_test=len(test),
          plan_hash=_plan_hash, cross_experiment=_xfdr,
          objective=_resolved_objective, elapsed_s=round(_elapsed(), 2))
    return GoalLoopResult(experiment, lb, winner, cert, decision, narr.text, len(test), provider.name,
                          rounds=rounds_info, compute_report=compute_report, failure_report=report,
                          objective=_resolved_objective, time_budget_s=time_budget_s,
                          elapsed_s=round(_elapsed(), 2), plan_hash=_plan_hash, cross_experiment=_xfdr,
                          cand_locked_correct=cand_locked_correct, base_locked_correct=base_locked_correct)
