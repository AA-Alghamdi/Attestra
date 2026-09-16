"""Stage operations: the ML *muscle* a research-program runner calls.

Codex's Program Runner is the orchestration skeleton (durable program object, adaptive loop, planner,
UI, report). This module is the muscle it calls at each stage. Every operation here is:

  * stateless over the wire   -- inputs and outputs are JSON-serializable; models are exchanged as
                                 candidate *specs* (a name + cfg), never as live sklearn objects, so a
                                 stage can be a local call today and an HTTP call tomorrow without change;
  * outcome-scored            -- it returns a uniform Evidence object whose `outcome` is the MEASURED
                                 result (CERTIFIED / IMPROVED / NO_CHANGE / BLOCKED / REFUSED / NEEDS_INPUT),
                                 never just the action word. This is the lesson from the masked-execution
                                 failures: "tried to rebalance" is not the same as "rebalanced and it helped";
  * honest about gates        -- the two operations that were validated in the lab but never enforced as
                                 product gates -- calibrated acquisition and gated synthetic generation --
                                 carry their gate here. `acquire` REFUSES on an uncalibrated model (the
                                 ECE gate; uncertainty sampling on a miscalibrated model selects noise).
                                 `generate` REFUSES unless the synthetic batch survives the leakage/dup
                                 gate against the locked test AND shows a MEASURED held-out lift.

Stage names match the Program Runner's named stages exactly so the two lanes compose:
  profile -> audit (check leakage) -> baseline -> autoresearch -> model_scan -> acquire -> expand -> generate -> certify

Nothing is reinvented: model_scan wraps ml.landscape_scan, audit wraps science.audit, certify wraps
science.certify_accuracy, generate wraps data_engine's measured gated path, acquire wraps the calibration
+ active-acquisition machinery. This module is the *seam*, not a second implementation.
"""

import numpy as np
import sys

from . import ml, science

# ----- gate constants (literature/finding-sourced, not tuned to any target) --------------------------
# ECE gate for acquisition: the calibration finding is that uncertainty acquisition only ever helped on
# calibrated models (SST-2 ECE 0.034 helped; Adult ECE 0.131 did not). 0.10 sits between those regimes
# and is the standard "well-calibrated" rule of thumb. Documented as the acquisition precondition.
ECE_ACQUIRE_MAX = 0.10
# A lift must clear this fraction (in the goal metric) to count as a real improvement, not noise.
MIN_REAL_LIFT = 0.005
AUTORESEARCH_DIR = "/Users/abdullahalghamdi/vectorforge-autoresearch"


# =========================================================================== Evidence object
def _evidence(operation, decision, outcome, *, metric=None, observed=None, lower_bound=None,
              p_value=None, n=None, leakage_passed=None, latency_ms_p95=None,
              evidence=None, next_action=None, summary=""):
    """The uniform record every stage returns. Drops straight into a program-level evidence report:
    what was tried (operation), what happened (decision/outcome), the numbers, and the next human action.
    """
    return {
        "operation": operation,
        "decision": decision,
        "outcome": outcome,                         # MEASURED result, not the action word
        "metric": metric,
        "observed": None if observed is None else round(float(observed), 4),
        "lower_bound": None if lower_bound is None else round(float(lower_bound), 4),
        "p_value": None if p_value is None else float(p_value),
        "n": None if n is None else int(n),
        "leakage_passed": leakage_passed,
        "latency_ms_p95": None if latency_ms_p95 is None else round(float(latency_ms_p95), 3),
        "evidence": evidence or {},
        "next_action": next_action,
        "summary": summary,
    }


# =========================================================================== helpers
def _ctor_table():
    """family name -> a zero-arg estimator factory. The same six strong families ml.candidates uses; kept
    here so a winner produced by one stage (a JSON candidate name like 'hist_gbm|inter') can be rebuilt by
    a later stage from its family prefix + the winner_cfg, without passing a live model over the wire."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import ComplementNB
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    return {
        "logistic": lambda: LogisticRegression(max_iter=2000),
        "logistic_balanced": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
        "linear_svm": lambda: LinearSVC(),
        "complement_nb": lambda: ComplementNB(),
        "hist_gbm": lambda: HistGradientBoostingClassifier(max_iter=200),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1),
    }


def _ctor_for(kind, candidate):
    """Recover the estimator factory for a candidate name by its family prefix (everything before '|')."""
    family = str(candidate).split("|", 1)[0]
    table = _ctor_table()
    if family not in table:
        raise KeyError(f"unknown candidate family {family!r} (from {candidate!r}) for kind {kind!r}")
    return table[family]


def _norm_cfg(cfg):
    """JSON has no tuples: a winner_cfg crossing the wire arrives with ngram as a list, but sklearn's
    TfidfVectorizer requires a tuple ngram_range. Coerce it back so stages are wire-transparent."""
    if isinstance(cfg, dict) and isinstance(cfg.get("ngram"), list):
        return {**cfg, "ngram": tuple(cfg["ngram"])}
    return cfg


def _fit(kind, candidate, cfg, rows, labels):
    cfg = _norm_cfg(cfg)
    ctor = _ctor_for(kind, candidate)
    Xtr, ytr = ml._Xy(kind, rows, cfg)
    pipe = ml.build_pipeline(kind, ctor, cfg)
    pipe.fit(Xtr, ytr)
    return pipe


def _bootstrap_metric_lower(metric, y_true, pred, labels, alpha=0.05, B=2000, seed=0):
    """Lower confidence bound for a general metric (balanced_accuracy / macro_f1) by resampling the
    (y_true, pred) pairs and recomputing the metric. For plain accuracy the caller uses the exact
    Clopper-Pearson bound instead; this is the correct tool for non-mean metrics."""
    yt, yp = np.asarray(y_true, dtype=object), np.asarray(pred, dtype=object)
    n = len(yt)
    if n == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        stats[b] = science.score_metric(metric, list(yt[idx]), list(yp[idx]), labels)
    return float(np.quantile(stats, alpha))


def _predictive_entropy(proba):
    p = np.clip(np.asarray(proba), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


# =========================================================================== STAGE: profile
def stage_profile(kind, train, val, test, labels, metric, **_):
    bal = {l: sum(1 for r in train if str(r.get("target")) == str(l)) for l in labels}
    counts = [c for c in bal.values() if c > 0] or [0]
    imb = (max(counts) / max(min(counts), 1)) if counts else 1.0
    feat_keys = set()
    for r in train[:200]:
        feat_keys |= (set(r.keys()) - {"id", "text", "target"})
        if isinstance(r.get("features"), dict):
            feat_keys |= {f"features.{k}" for k in r["features"]}
    feat_keys.discard("features")
    return _evidence(
        "profile", "profiled", "PASS", metric=metric, n=len(test),
        evidence={"n_train": len(train), "n_val": len(val), "n_test": len(test),
                  "kind": kind, "n_classes": len(labels), "class_balance": bal,
                  "imbalance_ratio": round(imb, 2), "feature_keys": sorted(feat_keys)[:40]},
        summary=f"{kind} task, {len(labels)} classes, imbalance {imb:.1f}:1, "
                f"{len(train)}/{len(val)}/{len(test)} train/val/test")


# =========================================================================== STAGE: audit (check leakage)
def stage_audit(kind, train, val, test, labels, metric, *, min_heldout_n=200, allow_features=None, **_):
    af = bool(allow_features) if allow_features is not None else (kind == "tabular")
    res = science.audit(train, test, min_test_n=min_heldout_n, allow_features=af)
    failed = [f for f in res["findings"] if not f.get("ok", True)]
    passed = res["passed"]
    return _evidence(
        "audit", "clean" if passed else "leakage_detected", "PASS" if passed else "BLOCKED",
        metric=metric, n=len(test), leakage_passed=passed,
        evidence={"findings": res["findings"], "failed_gates": [f["gate"] for f in failed]},
        next_action=None if passed else "fix_or_quarantine_leaking_signal",
        summary="leakage audit clean" if passed
                else f"BLOCKED: {', '.join(f['gate'] for f in failed)}")


# =========================================================================== STAGE: baseline
def stage_baseline(kind, train, val, test, labels, metric, **_):
    """A single honest reference fit (no search) so later stages can be scored as IMPROVED vs NO_CHANGE."""
    cand = "logistic" if kind == "text" else "hist_gbm"
    cfg = {"ngram": (1, 1)} if kind == "text" else {"interactions": False}
    pipe = _fit(kind, cand, cfg, train, labels)
    ev = ml.evaluate(pipe, kind, val, cfg, labels, metric)
    return _evidence(
        "baseline", "reference_set", "PASS", metric=metric, observed=ev["value"],
        n=len(val), latency_ms_p95=ev["latency_ms_p95"],
        evidence={"candidate": cand, "cfg": cfg, "val_metric": ev["value"], "per_class": ev["per_class"]},
        summary=f"baseline {cand}: {metric}={ev['value']} on val")


# =========================================================================== STAGE: autoresearch
def stage_autoresearch(kind, train, val, test, labels, metric, *, seed=None, pool=None, threshold=0.85,
                       label_price=1.0, budget_labels=None, min_heldout_n=200, allow_decompose=True,
                       meta_enabled=False, meta_store=None, acquire_k=300, pilot_k=60, **_):
    """Diagnosis-driven VoI controller, wrapped in the PRODUCT'S stricter rigor gate.

    The autoresearch package decides what experiment to run next: acquire labels, stop on a ceiling,
    decompose a hot slice, or certify. This wrapper keeps product truth discipline:
      * product science.audit runs first and blocks on its normal proxy/NMI/overlap/min-N gate;
      * the controller may propose/certify, but promotion still requires the normal `certify` stage;
      * all controller output is evidence, not a replacement for the product certificate.
    """
    if metric != "accuracy":
        return _evidence(
            "autoresearch", "unsupported_metric", "REFUSED", metric=metric, n=len(test),
            evidence={"reason": "autoresearch controller is currently validated for accuracy goals only"},
            next_action="fall_back_to_solver_stage_dag",
            summary=f"autoresearch refused metric={metric}; use stage DAG/certifier")
    if len(test) < min_heldout_n:
        return _evidence(
            "autoresearch", "underpowered_test", "BLOCKED", metric=metric, n=len(test),
            leakage_passed=False,
            evidence={"reason": f"locked test n={len(test)} < min_heldout_n={min_heldout_n}"},
            next_action="collect_more_held_out_labels",
            summary=f"autoresearch blocked: locked test n={len(test)} < {min_heldout_n}")

    seed_rows = list(seed if seed is not None else train)
    pool_rows = list(pool if pool is not None else [])
    strict_rows = seed_rows + pool_rows
    strict_audit = science.audit(strict_rows, test, min_test_n=min_heldout_n, allow_features=(kind == "tabular"))
    if not strict_audit["passed"]:
        failed = [f for f in strict_audit["findings"] if not f.get("ok", True)]
        return _evidence(
            "autoresearch", "strict_rigor_block", "BLOCKED", metric=metric, n=len(test),
            leakage_passed=False,
            evidence={"findings": strict_audit["findings"], "failed_gates": [f.get("gate") for f in failed]},
            next_action="fix_or_quarantine_leaking_signal",
            summary=f"autoresearch blocked by product rigor gate: {', '.join(str(f.get('gate')) for f in failed)}")

    try:
        if AUTORESEARCH_DIR not in sys.path:
            sys.path.insert(0, AUTORESEARCH_DIR)
        from ar.autoresearch import run_autoresearch
    except Exception as exc:  # noqa: BLE001
        return _evidence(
            "autoresearch", "controller_unavailable", "REFUSED", metric=metric, n=len(test),
            evidence={"error": f"{type(exc).__name__}: {exc}", "path": AUTORESEARCH_DIR},
            next_action="fall_back_to_solver_stage_dag",
            summary="autoresearch controller unavailable; falling back to solver stage DAG")

    # The recursive controller is advisory evidence inside the stage DAG; final promotion still belongs to
    # `stage_certify`. Keep the real locked test invisible here by using validation as the controller's
    # internal certification surface. The product certify stage is the only stage allowed to read test labels.
    controller_test = val
    task = {
        "name": "product_autoresearch_goal",
        "kind": kind,
        "labels": labels,
        "metric": metric,
        "threshold": threshold,
        "truth": "UNKNOWN",
        "seed": seed_rows,
        "pool": pool_rows,
        "val": val,
        "test": controller_test,
        "label_price": label_price,
    }
    budget = int(budget_labels if budget_labels is not None else max(len(pool_rows), acquire_k))
    try:
        result = run_autoresearch(
            task,
            budget=budget,
            seed=0,
            acquire_k=int(acquire_k),
            pilot_k=int(min(pilot_k, max(1, acquire_k))),
            allow_decompose=bool(allow_decompose),
            meta_enabled=bool(meta_enabled),
            meta_store=meta_store,
            meta_record=bool(meta_enabled),
        )
    except Exception as exc:  # noqa: BLE001
        return _evidence(
            "autoresearch", "controller_failed", "REFUSED", metric=metric, n=len(test),
            leakage_passed=True,
            evidence={"error": f"{type(exc).__name__}: {exc}"},
            next_action="fall_back_to_solver_stage_dag",
            summary=f"autoresearch controller failed: {type(exc).__name__}: {exc}")

    certificate = result.get("certificate") or {}
    status = result.get("status")
    certified = bool(certificate.get("certified")) and status == "CERTIFIED"
    stop_detail = result.get("stop_detail") or {}
    dominant_source = stop_detail.get("dominant_source")
    if certified:
        decision, outcome = "certified_controller_path", "CERTIFIED"
    elif status == "STOPPED_INFEASIBLE" and dominant_source == "LEAKAGE":
        decision, outcome = "rigor_or_data_block", "BLOCKED"
    elif status == "STOPPED_INFEASIBLE" and dominant_source == "UNDERPOWERED":
        decision, outcome = "controller_underpowered", "REFUSED"
    elif status == "STOPPED_INFEASIBLE":
        decision, outcome = "stopped_infeasible", "FAILED"
    else:
        decision, outcome = "continue_solver_dag", "NO_CHANGE"

    log = result.get("log") or []
    actions = [str(item.get("action")) for item in log if item.get("action")]
    lower = certificate.get("lower_bound")
    observed = certificate.get("observed") or certificate.get("obs") or certificate.get("accuracy")
    return _evidence(
        "autoresearch", decision, outcome, metric=metric,
        observed=observed, lower_bound=lower, p_value=certificate.get("p_value"),
        n=certificate.get("n") or len(test), leakage_passed=True,
        evidence={
            "status": status,
            "labels_used": result.get("labels_used"),
            "total_cost": result.get("total_cost"),
            "n_labeled_final": result.get("n_labeled_final"),
            "dominant_source": dominant_source,
            "stop_detail": stop_detail,
            "actions": actions[:40],
            "sub_certificates": result.get("sub_certificates", []),
            "certificate": certificate,
            "strict_audit_passed": True,
            "controller": "voi_autoresearch",
            "controller_test_surface": "validation-surrogate",
        },
        next_action=("run_product_certify_stage" if certified else stop_detail.get("cheapest_unblock") or "continue_solver_stage_dag"),
        summary=(f"autoresearch {status}: labels_used={result.get('labels_used')} "
                 f"dominant={dominant_source or 'not-applicable'} actions={','.join(actions[:6]) or 'none'}"))


# =========================================================================== STAGE: model_scan
def stage_model_scan(kind, train, val, test, labels, metric, *, baseline_val=None, **_):
    """The strong-learner muscle: scan families x configs, rank on validation, shortlist, refit, pick a
    winner -- with a latency/metric Pareto. This is what the homegrown TS learners cannot do; it is the
    reason this backend exists. Returns the winner as a (candidate, cfg) spec, not a live model."""
    res = ml.landscape_scan(kind, train, val, test, labels, metric)
    improved = baseline_val is not None and res["val"] >= baseline_val + MIN_REAL_LIFT
    decision = "winner_selected"
    outcome = "IMPROVED" if improved else ("NO_CHANGE" if baseline_val is not None else "PASS")
    return _evidence(
        "model_scan", decision, outcome, metric=metric, observed=res["val"], n=len(val),
        evidence={"winner": res["winner"], "winner_cfg": res["winner_cfg"],
                  "leaderboard": res["leaderboard"], "shortlist": res["shortlist"],
                  "pareto": res["pareto"], "baseline_val": baseline_val},
        summary=f"winner {res['winner']} {metric}={res['val']} on val "
                f"({'+' if improved else ''}{(res['val']-baseline_val):.4f} vs baseline)"
                if baseline_val is not None else f"winner {res['winner']} {metric}={res['val']}")


# =========================================================================== STAGE: acquire (ECE-gated)
def stage_acquire(kind, train, val, test, labels, metric, *, seed_n=300, acquire_k=200, repeats=3, **_):
    """Calibrated active acquisition, PRODUCTIZED WITH ITS GATE.

    The lab finding: uncertainty acquisition only ever helped on *calibrated* models. So this stage first
    measures ECE; if the model is miscalibrated (ECE > ECE_ACQUIRE_MAX) it REFUSES -- uncertainty sampling
    on a model whose confidence is unreliable selects noise, not informative points, and claiming a label
    budget saving there would be dishonest.

    When the gate passes, it MEASURES the value: from a labeled seed it scores val, then compares adding
    `acquire_k` points chosen by predictive entropy (uncertainty) against `acquire_k` random points,
    averaged over `repeats` seeds, and reports the lift with a spread. It only recommends acquisition when
    uncertainty beats random by a real margin. The rest of train stands in as the unlabeled pool so the
    recommendation is measured, never asserted.
    """
    from sklearn.linear_model import LogisticRegression
    # acquisition is a data question; evaluate it with a calibratable probe regardless of the winner.
    cfg = {"ngram": (1, 1)} if kind == "text" else {"interactions": False}

    def fit_proba(rows):
        Xtr, ytr = ml._Xy(kind, rows, cfg)
        clf = LogisticRegression(max_iter=2000)
        pipe = ml.build_pipeline(kind, lambda: clf, cfg)
        pipe.fit(Xtr, ytr)
        return pipe

    Xva, yva = ml._Xy(kind, val, cfg)
    if len(train) <= seed_n + 10:
        return _evidence("acquire", "not_applicable", "NO_CHANGE", metric=metric, n=len(val),
                         evidence={"reason": "train too small to simulate an unlabeled pool"},
                         summary="acquisition not simulated (train too small)")

    rng = np.random.default_rng(0)
    order = rng.permutation(len(train))
    seed_rows = [train[i] for i in order[:seed_n]]
    pool_rows = [train[i] for i in order[seed_n:]]

    seed_pipe = fit_proba(seed_rows)
    try:
        proba_va = seed_pipe.predict_proba(Xva)
        conf = np.max(proba_va, axis=1)
        correct = [p == t for p, t in zip(seed_pipe.predict(Xva), yva)]
        ece = science.expected_calibration_error(conf.tolist(), correct)
    except Exception:                                    # noqa: BLE001  (degenerate proba)
        return _evidence("acquire", "refuse_acquisition", "REFUSED", metric=metric, n=len(val),
                         evidence={"reason": "model exposes no usable probabilities"},
                         next_action="use_a_calibratable_model_before_acquisition",
                         summary="acquisition refused: no usable probabilities")

    base_val = science.score_metric(metric, yva, list(seed_pipe.predict(Xva)), labels)

    # ---- THE ECE GATE -------------------------------------------------------------------------------
    if ece > ECE_ACQUIRE_MAX:
        return _evidence(
            "acquire", "refuse_acquisition", "REFUSED", metric=metric, observed=base_val, n=len(val),
            evidence={"ece": round(ece, 4), "ece_gate": ECE_ACQUIRE_MAX,
                      "reason": "model uncalibrated; uncertainty sampling would select noise, not signal"},
            next_action="calibrate_model_or_request_random_labels",
            summary=f"acquisition REFUSED: ECE {ece:.3f} > {ECE_ACQUIRE_MAX} (uncalibrated)")

    # ---- gate passed: MEASURE uncertainty vs random ------------------------------------------------
    Xpool, _ = ml._Xy(kind, pool_rows, cfg)
    ent = _predictive_entropy(seed_pipe.predict_proba(Xpool))
    unc_idx = list(np.argsort(-ent)[:acquire_k])
    unc_rows = seed_rows + [pool_rows[i] for i in unc_idx]
    unc_val = science.score_metric(metric, yva, list(fit_proba(unc_rows).predict(Xva)), labels)

    rnd_vals = []
    for s in range(repeats):
        ridx = np.random.default_rng(100 + s).choice(len(pool_rows), size=min(acquire_k, len(pool_rows)),
                                                      replace=False)
        rnd_rows = seed_rows + [pool_rows[i] for i in ridx]
        rnd_vals.append(science.score_metric(metric, yva, list(fit_proba(rnd_rows).predict(Xva)), labels))
    rnd_mean = float(np.mean(rnd_vals))
    advantage = unc_val - rnd_mean
    worth_it = advantage > MIN_REAL_LIFT

    return _evidence(
        "acquire", "acquire_by_uncertainty" if worth_it else "no_acquisition_value",
        "IMPROVED" if worth_it else "NO_CHANGE", metric=metric, observed=unc_val, n=len(val),
        evidence={"ece": round(ece, 4), "base_val": round(base_val, 4),
                  "uncertainty_val": round(unc_val, 4), "random_val": round(rnd_mean, 4),
                  "random_spread": [round(v, 4) for v in rnd_vals],
                  "advantage_over_random": round(advantage, 4), "acquire_k": acquire_k},
        next_action=(f"collect {acquire_k} labels selected by uncertainty" if worth_it
                     else "random labels are as good here; no uncertainty premium"),
        summary=(f"calibrated (ECE {ece:.3f}); uncertainty +{advantage:.4f} over random -> acquire"
                 if worth_it else
                 f"calibrated (ECE {ece:.3f}) but uncertainty has no premium over random ({advantage:+.4f})"))


# =========================================================================== STAGE: expand (representation)
def stage_expand(kind, train, val, test, labels, metric, *, baseline_val=None, **_):
    """Representation expansion: wider n-grams (text) or numeric pair interactions (tabular). Measured
    against the baseline so XOR-like structure shows up as a real lift, not just a claimed action."""
    if kind == "text":
        cfg = {"ngram": (1, 2)}
        cand = "logistic"
    else:
        cfg = {"interactions": True}
        cand = "hist_gbm"
    ctor = _ctor_for(kind, cand)
    Xtr, ytr = ml._Xy(kind, train, cfg)
    Xva, yva = ml._Xy(kind, val, cfg)
    pipe = ml.build_pipeline(kind, ctor, cfg); pipe.fit(Xtr, ytr)
    v = science.score_metric(metric, yva, list(pipe.predict(Xva)), labels)
    improved = baseline_val is not None and v >= baseline_val + MIN_REAL_LIFT
    return _evidence(
        "expand", "representation_expanded", "IMPROVED" if improved else "NO_CHANGE",
        metric=metric, observed=v, n=len(val),
        evidence={"cfg": cfg, "candidate": cand, "val": round(v, 4), "baseline_val": baseline_val},
        summary=f"expanded representation {metric}={v:.4f}" +
                (f" (+{v-baseline_val:.4f} vs baseline)" if baseline_val is not None else ""))


# =========================================================================== STAGE: generate (gated)
def stage_generate(goal, train, val, test, *, allow_api=False, **_):
    """Gated synthetic generation as a MEASURED data move. Refuses unless (1) the goal surface permits it
    and spend is approved, (2) an API key is available, (3) generated rows survive the leakage/dup gate
    against the LOCKED test, and (4) they show a measured held-out lift. We never claim a lift we did not
    measure. Wraps data_engine's path so there is exactly one generation implementation."""
    from . import data_engine
    kind, labels, metric = goal.kind, goal.labels, goal.verification.metric
    permitted = goal.surface.synthetic_data and goal.budget.approve_spend
    if not permitted:
        return _evidence("generate", "refuse_generate", "REFUSED", metric=metric, n=len(val),
                         evidence={"reason": "synthetic generation not permitted (surface/budget)"},
                         next_action="approve_synthetic_data_and_spend_to_enable",
                         summary="generation REFUSED: not permitted by surface/budget")
    if not allow_api:
        return _evidence("generate", "refuse_generate", "REFUSED", metric=metric, n=len(val),
                         evidence={"reason": "no API access; generation is a measured paid move, not run"},
                         next_action="provide_api_access_to_run_generation",
                         summary="generation REFUSED: no API access")
    report = data_engine.run_data_engine(goal, train, val, test, allow_api=True)
    move = report.get("move")
    action = report.get("report", {}).get("action", {})
    adopted = move == "GENERATE" and report.get("new_train") is not None
    return _evidence(
        "generate", "generated_adopted" if adopted else "generate_no_lift",
        "IMPROVED" if adopted else "NO_CHANGE", metric=metric, n=len(val),
        evidence={"move": move, "action": action},
        next_action=None if adopted else "request_real_labels (synthetic showed no measured lift)",
        summary=f"generation move={move}, adopted={adopted}")


# =========================================================================== STAGE: certify (locked test, once)
def stage_certify(kind, train, val, test, labels, metric, *, winner, winner_cfg, threshold,
                  alpha=0.05, max_latency_ms=50.0, min_heldout_n=200, checks=1, **_):
    """The promotion gate: refit the winner on train+val, evaluate ONCE on the locked test, and certify
    with a Clopper-Pearson lower bound (accuracy) or a bootstrap lower bound (general metric). Promotion
    requires lower_bound > threshold AND a clean leakage audit AND latency within budget. Anything short
    is an honest WEAK / DO_NOT_CERTIFY, never a relaxed threshold."""
    winner_cfg = _norm_cfg(winner_cfg)
    if len(test) < min_heldout_n:
        return _evidence("certify", "do_not_certify", "FAILED", metric=metric, n=len(test),
                         evidence={"reason": f"locked test n={len(test)} < min_heldout_n={min_heldout_n}"},
                         next_action="collect_more_held_out_labels",
                         summary=f"cannot certify: test n={len(test)} < {min_heldout_n}")
    fit_rows = train + val
    pipe = _fit(kind, winner, winner_cfg, fit_rows, labels)
    ev = ml.evaluate(pipe, kind, test, winner_cfg, labels, metric)
    observed = ev["value"]
    acc = ev["accuracy"]
    n = len(test)
    leak = science.audit(fit_rows, test, min_test_n=min_heldout_n,
                         allow_features=(kind == "tabular"))["passed"]

    if metric == "accuracy":
        cert = science.certify_accuracy(acc, n, threshold, checks=checks, alpha=alpha)
        lower, p = cert["lower_bound"], cert.get("p_value")
    else:
        cert = None
        lower = _bootstrap_metric_lower(metric, ev["y_true"], ev["predictions"], labels, alpha=alpha)
        p = None
    latency_ok = ev["latency_ms_p95"] <= max_latency_ms
    certified = (lower > threshold) and leak and latency_ok
    margin = float(lower) - float(threshold)
    borderline = abs(margin) < 0.02 or (observed >= threshold and not certified)
    if certified:
        decision, outcome = "certified", "CERTIFIED"
    elif borderline:
        decision, outcome = "borderline_weak_evidence", "WEAK"
    elif observed >= threshold:
        decision, outcome = "weak_evidence", "WEAK"
    else:
        decision, outcome = "do_not_certify", "FAILED"

    reason = ("certified: lower bound clears threshold on the locked test" if certified else
              ("leakage audit failed" if not leak else
               ("latency over budget" if not latency_ok else
                "lower confidence bound does not clear the threshold")))
    return _evidence(
        "certify", decision, outcome, metric=metric, observed=observed, lower_bound=lower,
        p_value=p, n=n, leakage_passed=leak, latency_ms_p95=ev["latency_ms_p95"],
        evidence={"winner": winner, "accuracy": acc, "threshold": threshold, "alpha": alpha,
                  "checks": checks, "alpha_per_check": round(alpha / max(1, int(checks)), 6),
                  "borderline": borderline, "margin_to_threshold": round(margin, 4),
                  "certifier_reason": cert.get("reason") if cert else "bootstrap lower bound for non-accuracy metric",
                  "latency_ok": latency_ok, "per_class": ev["per_class"], "reason": reason},
        next_action=("promote" if certified else
                     ("collect_more_held_out_labels_or_improve_model" if outcome == "WEAK" else
                      "model does not meet the bar; report honest failure")),
        summary=f"{decision}: {metric}={observed} lb={round(lower,4)} vs thr={threshold} "
                f"(leak_ok={leak}, latency_ok={latency_ok})")


# the stages a program runner can call, in canonical order, keyed by the Program Runner's stage names.
STAGES = {
    "profile": stage_profile,
    "audit": stage_audit,
    "baseline": stage_baseline,
    "autoresearch": stage_autoresearch,
    "model_scan": stage_model_scan,
    "acquire": stage_acquire,
    "expand": stage_expand,
    "generate": stage_generate,
    "certify": stage_certify,
}
