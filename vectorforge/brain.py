"""The adaptive research controller.

Replaces the single landscape scan (ml.landscape_scan) with a multi-round diagnose -> act loop.
Each round MEASURES signals on the current state, names the single highest-value axis the goal's
surface allows, takes that action, keeps the best-on-validation pipeline, then re-diagnoses. It stops
when validation clears threshold+margin, the budget is exhausted, or no allowed axis improves on
validation (at which point it asks for labels honestly instead of faking a win).

Rigor contract (the whole point of the product):
  * The locked TEST set is NEVER read here. Selection is on VALIDATION only. The caller (runner)
    certifies the winner on test exactly once via science.certify_accuracy / bootstrap_lower.
  * Every decision records the measured EVIDENCE that produced it. No decision is taken without a
    signal behind it.
  * Honest stop: when no axis improves and the bar is still unmet, next_experiment asks for more
    labels / acquisition; it does not relax the threshold or pretend success.

It reuses the learner layer wholesale (ml.candidates / ml.build_pipeline / ml._Xy /
ml.landscape_scan / ml.expand_interactions) and the science layer for every number
(score_metric, per_class_recall, expected_calibration_error). Nothing is reinvented.
"""

from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from . import science, ml

# A round that improves validation by less than this is treated as "no real gain" for the
# stop / memory logic. Chosen well below the certifier's resolution; it is a search-control
# tolerance, not a spec threshold.
_EPS = 1e-4
# Stop early once validation clears threshold by this margin (the spec asked for +0.02): a model
# this far above the bar will survive the multiplicity-corrected lower bound on the locked test.
_STOP_MARGIN = 0.02
# A class-imbalance ratio (max_class_frac / min_class_frac) at or above this is treated as a MEANINGFUL
# imbalance that justifies the class-weighting (rebalance) axis. Below it, label-frequency differences
# are sampling fluctuation, not an imbalance the rebalance lever should claim to fix. 1.5 == a 3:2
# majority:minority share; well above the ~1.03 a near-even split produces, well below a real 3:1 skew.
_IMBALANCE_GATE = 1.5


# =========================================================================== measured signals
def _fit_eval(kind, ctor, cfg, train, val, labels, metric):
    """Fit one pipeline on train, score it on VALIDATION. Returns the fitted pipe, the val score,
    the validation predictions, and predict_proba confidences (or None). Test is never seen."""
    pipe = ml.build_pipeline(kind, ctor, cfg)
    Xtr, ytr = ml._Xy(kind, train, cfg)
    Xva, yva = ml._Xy(kind, val, cfg)
    pipe.fit(Xtr, ytr)
    pred = list(pipe.predict(Xva))
    val_score = science.score_metric(metric, yva, pred, labels)
    conf = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(Xva)
            conf = (np.max(proba, axis=1).tolist(), [p == t for p, t in zip(pred, yva)])
        except Exception:  # noqa: BLE001  (some calibrators degenerate; treat as no-proba)
            conf = None
    return pipe, float(val_score), pred, yva, conf


def _balance(rows, labels):
    c = Counter(str(r.get("target")) for r in rows)
    n = max(sum(c.values()), 1)
    return {l: {"count": c.get(l, 0), "frac": round(c.get(l, 0) / n, 4)} for l in labels}


def _imbalance_ratio(balance):
    fracs = [v["frac"] for v in balance.values() if v["frac"] > 0]
    return round(max(fracs) / min(fracs), 3) if fracs else 1.0


def _ece(conf):
    if not conf:
        return None
    confidences, correct = conf
    return science.expected_calibration_error(confidences, correct)


def _weak_classes(yva, pred, labels, threshold):
    """Per-class recall on validation; a class is weak if its recall sits a clear margin below the
    spec threshold (these are the classes a rebalance / weighting action should target)."""
    pcr = science.per_class_recall(yva, pred, labels)
    weak = []
    for lab, d in pcr.items():
        r = d["recall"]
        if r is not None and r < threshold - 0.05:
            weak.append({"label": lab, "recall": r, "support": d["support"]})
    return weak, pcr


# =========================================================================== axis actions
def _baseline_cfg(kind):
    """A cheap, honest starting point per modality so diagnosis has a real current-best to reason
    about before any expensive search. Logistic on unigrams (text) / base features (tabular)."""
    if kind == "text":
        return ("logistic|ngram1", lambda: LogisticRegression(max_iter=2000), {"ngram": (1, 1)})
    return ("logistic|base", lambda: LogisticRegression(max_iter=2000), {"interactions": False})


def _balanced_family(kind):
    """Class-weight-balanced candidate families for the rebalance axis. These are the same learners
    ml.candidates offers, restricted to their balanced variants so the action is a true rebalance,
    not a disguised model-family search."""
    if kind == "text":
        return [(f"logistic_balanced|ngram{ng[1]}",
                 lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
                 {"ngram": ng}) for ng in [(1, 1), (1, 2)]]
    return [("logistic_balanced|base",
             lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
             {"interactions": False}),
            ("logistic_balanced|inter",
             lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
             {"interactions": True}),
            ("random_forest_balanced|base",
             lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced"),
             {"interactions": False})]


def _representation_family(kind, cur_cfg):
    """The representation axis. Text: widen the n-gram window. Tabular: turn on the categorical
    interaction expansion (ml.expand_interactions) the current cfg is not yet using."""
    if kind == "text":
        wide = (1, 2) if cur_cfg.get("ngram", (1, 1)) == (1, 1) else (1, 3)
        return [(f"logistic|ngram{wide[1]}", lambda: LogisticRegression(max_iter=2000),
                 {"ngram": wide}),
                (f"complement_nb|ngram{wide[1]}", lambda: ComplementNB(), {"ngram": wide})]
    return [("hist_gbm|inter", lambda: HistGradientBoostingClassifier(max_iter=200),
             {"interactions": True}),
            ("logistic|inter", lambda: LogisticRegression(max_iter=2000), {"interactions": True})]


def _calibration_family(kind, cur_ctor, cur_cfg):
    """The calibration axis: wrap the current learner in sklearn isotonic/sigmoid calibration with an
    internal CV (no test, no val leakage; the wrapper holds out internally on train). Returns
    constructors that build a *calibrated* estimator over the same cfg."""
    from sklearn.calibration import CalibratedClassifierCV

    def make(method):
        return lambda: CalibratedClassifierCV(cur_ctor(), method=method, cv=3)

    return [(f"calibrated_{m}", make(m), dict(cur_cfg)) for m in ("isotonic", "sigmoid")]


def _eval_family(kind, family, train, val, labels, metric, evidence_tag):
    """Fit/score a list of (label, ctor, cfg) candidates on validation; return the best one and the
    per-candidate evidence. Never touches test."""
    results, best = [], None
    for label, ctor, cfg in family:
        try:
            pipe, vs, pred, yva, conf = _fit_eval(kind, ctor, cfg, train, val, labels, metric)
            row = {"candidate": label, "val": round(vs, 4), "cfg": cfg, "ctor": ctor,
                   "pipe": pipe, "pred": pred, "yva": yva, "conf": conf}
            results.append({"candidate": label, "val": round(vs, 4)})
            if best is None or vs > best["val"]:
                best = row
        except Exception as e:  # noqa: BLE001
            results.append({"candidate": label, "val": -1.0, "error": str(e)[:80]})
    return best, {"tag": evidence_tag, "candidates": results}


# =========================================================================== the controller
def run_research(goal, train, val, test):  # noqa: C901  (the diagnose->act loop is one cohesive unit)
    """Adaptive multi-round research over a goal's allowed experiment surface.

    Selects entirely on `val`. `test` is accepted only to satisfy the runner's call signature and to
    let us report held-out N honestly in an acquisition ask; it is NEVER scored here.

    Returns: {winner_pipe, winner_cfg, winner_label, best_val, research_state, leaderboard}.
    """
    kind = goal.kind
    labels = goal.labels
    metric = goal.verification.metric
    threshold = goal.verification.threshold
    surface = goal.surface
    max_rounds = max(1, int(goal.budget.max_rounds))
    max_experiments = max(1, int(goal.budget.max_experiments))

    # ---- standing diagnosis: measured once up front, refreshed each round -------------------
    balance = _balance(train, labels)
    imbalance = _imbalance_ratio(balance)
    data_limited = len(train) < 200

    findings = [
        {"id": "class_balance", "evidence": {"balance": balance, "imbalance_ratio": imbalance}},
        {"id": "data_volume", "evidence": {"n_train": len(train), "n_val": len(val),
                                           "data_limited": data_limited}},
    ]
    decisions = []
    memory = []
    experiments = 0

    # ---- round 0: establish a real current-best (a cheap baseline, scored on val) -----------
    blabel, bctor, bcfg = _baseline_cfg(kind)
    best_pipe, base_val, bpred, byva, bconf = _fit_eval(kind, bctor, bcfg, train, val, labels, metric)
    experiments += 1
    best = {"label": blabel, "cfg": bcfg, "val": round(base_val, 4), "pipe": best_pipe,
            "pred": bpred, "yva": byva, "conf": bconf}
    leaderboard = [{"candidate": blabel, "val": best["val"], "axis": "baseline"}]
    findings.append({"id": "baseline_val", "evidence": {"candidate": blabel, "val": best["val"],
                     "metric": metric, "threshold": threshold}})

    hypothesis = (f"A {kind} classifier can reach {metric} >= {threshold} on this goal; the baseline "
                  f"{blabel} scores {best['val']} on validation, so the gap is "
                  f"{round(max(0.0, threshold - best['val']), 4)}.")

    # axes are only available if the surface permits them; map each axis to its gate
    def axis_allowed(axis):
        return {"model_class": surface.model_families,
                "representation": surface.feature_engineering,
                "rebalance": surface.class_weighting,
                "calibration": surface.calibration}[axis]

    tried_axes = set()
    # axes that at some round had measured evidence justifying them (so they are real remaining
    # levers if not yet exhausted). An allowed-but-never-justified axis (e.g. calibration when the
    # model is already well calibrated) is NOT a lever for closing a metric gap and must not be
    # offered as the honest next step.
    justified_axes = set()
    stop_reason = None

    for rnd in range(1, max_rounds + 1):
        # ---- DIAGNOSE on the current best, using only measured validation signals ----------
        weak, pcr = _weak_classes(best["yva"], best["pred"], labels, threshold)
        ece = _ece(best["conf"])
        gap = round(threshold - best["val"], 4)

        diag = {"round": rnd, "best_val": best["val"], "gap": gap, "imbalance_ratio": imbalance,
                "weak_classes": [w["label"] for w in weak], "per_class_recall": pcr,
                "ece": ece, "data_limited": data_limited}

        # already over the line with margin -> stop, certify-ready
        if best["val"] >= threshold + _STOP_MARGIN:
            stop_reason = "val_clears_threshold_plus_margin"
            decisions.append({"round": rnd, "diagnosis": "validation already clears threshold by the "
                              "stop margin; further search cannot help and risks overfitting val",
                              "action": "stop", "evidence": {"best_val": best["val"],
                              "threshold": threshold, "margin": _STOP_MARGIN}})
            break

        if experiments >= max_experiments:
            stop_reason = "experiment_budget_exhausted"
            break

        # ---- CHOOSE the highest-value axis the surface allows, ranked by measured need ------
        # priority: a structural deficit (data) first; then the signal with the strongest evidence.
        candidate_axes = []
        # rebalance is a CLASS-WEIGHTING lever, so it is only the right diagnosis when there is a
        # MEANINGFUL class imbalance (ratio >= _IMBALANCE_GATE). Uniformly-low recall on a (near-)balanced
        # dataset is an underfitting / representation signal, NOT an imbalance signal -- routing it to
        # rebalance would let a class-weighted tree "solve" a balanced task (e.g. XOR) for the wrong
        # stated reason (the model captured the interaction; class weighting did nothing). A trivial
        # 0.507/0.493 fluctuation is not imbalance, so the under-represented class there is not a true
        # minority. Within a genuinely imbalanced regime, the under-represented weak classes (if any)
        # sharpen the severity so rebalance outranks a generic model scan.
        meaningfully_imbalanced = imbalance >= _IMBALANCE_GATE
        minority = ({l for l, v in balance.items()
                     if v["frac"] > 0 and v["frac"] < (1.0 / max(len(labels), 1))}
                    if meaningfully_imbalanced else set())
        imbalance_weak = [w for w in weak if w["label"] in minority]
        if axis_allowed("rebalance") and meaningfully_imbalanced:
            sev = (imbalance - 1.0) + 1.0 * len(imbalance_weak)
            candidate_axes.append(("rebalance", sev,
                {"imbalance_ratio": imbalance, "weak_classes": imbalance_weak,
                 "minority_classes": sorted(minority)}))
        # model_class: a broad family scan; justified whenever we still have a gap to close
        if axis_allowed("model_class") and gap > 0:
            candidate_axes.append(("model_class", 0.5 + gap,
                {"gap": gap, "reason": "broad model-family scan to lift the validation ceiling"}))
        # representation: tabular interaction gain or text n-gram widening; justified with a gap
        if axis_allowed("representation") and gap > 0:
            candidate_axes.append(("representation", 0.4 + gap,
                {"gap": gap, "reason": "expand representation (interactions / wider n-grams)"}))
        # calibration: justified when probabilities are miscalibrated (only helps proba metrics /
        # downstream thresholds, never accuracy directly, so lowest priority and gated on ECE)
        if axis_allowed("calibration") and ece is not None and ece > 0.10:
            candidate_axes.append(("calibration", 0.2,
                {"ece": ece, "reason": "model is miscalibrated; calibrate without changing the decision"}))

        justified_axes |= {a[0] for a in candidate_axes}
        # don't repeat an axis that already failed to improve
        candidate_axes = [a for a in candidate_axes if a[0] not in tried_axes]
        candidate_axes.sort(key=lambda a: -a[1])

        if not candidate_axes:
            stop_reason = "no_allowed_axis_with_evidence"
            break

        axis, _sev, axis_evidence = candidate_axes[0]
        tried_axes.add(axis)

        # ---- ACT on the chosen axis --------------------------------------------------------
        if axis == "model_class":
            scan = ml.landscape_scan(kind, train, val, test, labels, metric)
            experiments += len(scan["leaderboard"])
            for c in scan["leaderboard"]:
                leaderboard.append({**c, "axis": "model_class"})
            # landscape_scan returns the val-best fitted pipe; re-derive its val + diagnostics
            cand_pipe, cand_cfg, cand_val = scan["winner_pipe"], scan["winner_cfg"], scan["val"]
            Xva, yva = ml._Xy(kind, val, cand_cfg)
            cand_pred = list(cand_pipe.predict(Xva))
            cand_conf = None
            if hasattr(cand_pipe, "predict_proba"):
                try:
                    cand_conf = (np.max(cand_pipe.predict_proba(Xva), axis=1).tolist(),
                                 [p == t for p, t in zip(cand_pred, yva)])
                except Exception:  # noqa: BLE001
                    cand_conf = None
            best_cand = {"label": scan["winner"], "cfg": cand_cfg, "val": cand_val,
                         "pipe": cand_pipe, "pred": cand_pred, "yva": yva, "conf": cand_conf}
            action_evidence = {"scanned": len(scan["leaderboard"]), "winner": scan["winner"],
                               "winner_val": cand_val, "top3": scan["leaderboard"][:3]}
        else:
            if axis == "rebalance":
                family = _balanced_family(kind)
            elif axis == "representation":
                family = _representation_family(kind, best["cfg"])
            else:  # calibration
                family = _calibration_family(kind, bctor if best["label"] == blabel else
                                             (lambda: LogisticRegression(max_iter=2000)), best["cfg"])
            bc, ev = _eval_family(kind, family, train, val, labels, metric, axis)
            experiments += len(ev["candidates"])
            for c in ev["candidates"]:
                leaderboard.append({**c, "axis": axis})
            if bc is None:
                stop_reason = f"axis_{axis}_produced_no_valid_candidate"
                decisions.append({"round": rnd, "diagnosis": f"axis={axis} chosen on evidence "
                                  f"{axis_evidence}", "action": f"{axis}: all candidates errored",
                                  "evidence": ev})
                continue
            best_cand = {"label": bc["candidate"], "cfg": bc["cfg"], "val": bc["val"],
                         "pipe": bc["pipe"], "pred": bc["pred"], "yva": bc["yva"], "conf": bc["conf"]}
            action_evidence = {"family": [c["candidate"] for c in ev["candidates"]],
                               "best": bc["candidate"], "best_val": bc["val"]}

        # ---- KEEP best-on-validation; record the decision + the measured evidence ----------
        improved = best_cand["val"] > best["val"] + _EPS
        decisions.append({
            "round": rnd,
            "diagnosis": _diagnosis_text(axis, axis_evidence, diag),
            "action": f"axis={axis}: {best_cand['label']} (val={best_cand['val']})",
            "evidence": {"signal": axis_evidence, "result": action_evidence,
                         "prev_best_val": best["val"], "new_val": best_cand["val"],
                         "kept": improved},
        })
        if improved:
            best = best_cand
            memory.append(f"axis {axis} improved val to {best['val']} via {best['label']}")
        else:
            memory.append(f"axis {axis} did not beat static best (val stayed {best['val']})")

    else:
        stop_reason = stop_reason or "round_budget_exhausted"

    # ---- final re-diagnosis + honest next_experiment ---------------------------------------
    weak, pcr = _weak_classes(best["yva"], best["pred"], labels, threshold)
    ece = _ece(best["conf"])
    met = best["val"] >= threshold

    findings.append({"id": "final_validation", "evidence": {"best_val": best["val"],
                     "threshold": threshold, "met": met, "stop_reason": stop_reason}})
    if ece is not None:
        findings.append({"id": "calibration", "evidence": {"ece": ece,
                         "uncalibrated": ece > 0.10}})
    if weak:
        findings.append({"id": "weak_recall", "evidence": {"classes": weak}})
        for w in weak:
            memory.append(f"class {w['label']} recall low on validation ({w['recall']})")

    next_experiment = _next_experiment(met, best, threshold, weak, len(train), len(test),
                                       goal, tried_axes, justified_axes, stop_reason, ece)

    research_state = {
        "hypothesis": hypothesis,
        "findings": findings,
        "decisions": decisions,
        "memory": memory,
        "next_experiment": next_experiment,
    }
    return {
        "winner_pipe": best["pipe"],
        "winner_cfg": best["cfg"],
        "winner_label": best["label"],
        "best_val": round(best["val"], 4),
        "research_state": research_state,
        "leaderboard": leaderboard,
    }


def _diagnosis_text(axis, axis_evidence, diag):
    if axis == "rebalance":
        return (f"validation imbalance_ratio={diag['imbalance_ratio']} and weak classes "
                f"{diag['weak_classes']}; class weighting should lift minority recall")
    if axis == "model_class":
        return (f"gap of {diag['gap']} to threshold with best_val={diag['best_val']}; a broad "
                f"model-family scan is the highest-value lever on the validation ceiling")
    if axis == "representation":
        return (f"gap of {diag['gap']} remains; the current representation may underfit, so expand "
                f"it (interactions / wider n-grams) before concluding the data is the limit")
    if axis == "calibration":
        return (f"ECE={diag['ece']} indicates miscalibrated probabilities; calibrate the scores")
    return f"axis={axis} on evidence {axis_evidence}"


def _next_experiment(met, best, threshold, weak, n_train, n_test, goal, tried_axes, justified_axes,
                     stop_reason, ece):
    """The honest forward pointer. If the bar is met on val, point at certification. If not, and no
    allowed axis improved, ask for the thing that would actually help (labels / acquisition for the
    weak classes), never a threshold relaxation."""
    if met:
        return {"action": "certify_on_locked_test",
                "reason": f"validation {best['val']} meets the {threshold} bar; promotion is gated by "
                          f"science.certify_accuracy on the locked test, evaluated once",
                "evidence": {"best_val": best["val"], "threshold": threshold,
                             "winner": best["label"]}}
    # bar unmet: be honest about why and what would help. A real remaining lever is an axis that is
    # ALLOWED by the surface, was JUSTIFIED by a measured signal at some round, and was not yet
    # exhausted. An allowed-but-never-justified axis (e.g. calibration on a calibrated model) does
    # not close a metric gap and is not offered.
    allowed = {a for a, ok in [("model_class", goal.surface.model_families),
                               ("representation", goal.surface.feature_engineering),
                               ("rebalance", goal.surface.class_weighting),
                               ("calibration", goal.surface.calibration)] if ok}
    # Calibration cannot move a hard-label metric (it leaves the argmax prediction unchanged), so it
    # is never a lever for closing an accuracy / balanced_accuracy / macro_f1 gap. Excluding it here
    # is what forces the honest "acquire labels" answer instead of a futile calibration suggestion.
    hard_label_metrics = {"accuracy", "balanced_accuracy", "macro_f1"}
    gap_levers = justified_axes & allowed
    if goal.verification.metric in hard_label_metrics:
        gap_levers = gap_levers - {"calibration"}
    untried = sorted(gap_levers - set(tried_axes))
    if untried:
        return {"action": f"try_axis:{untried[0]}",
                "reason": f"validation {best['val']} is below {threshold}; axis {untried[0]} is "
                          f"allowed and not yet exhausted",
                "evidence": {"best_val": best["val"], "untried_axes": untried,
                             "stop_reason": stop_reason}}
    target = [w["label"] for w in weak] or goal.labels
    return {"action": "acquire_labels" if goal.surface.label_requests else "needs_more_evidence",
            "reason": f"every allowed axis was exhausted and validation {best['val']} still trails "
                      f"{threshold}; the limit is data, not search. Acquire labeled examples for "
                      f"{target} rather than relaxing the bar",
            "evidence": {"best_val": best["val"], "threshold": threshold, "weak_classes": weak,
                         "n_train": n_train, "n_test_heldout": n_test, "ece": ece,
                         "stop_reason": stop_reason}}


# =========================================================================== self-test
def _selftest():
    import json
    from pathlib import Path
    from .domain import Goal, VerificationSpec, ExperimentSurface, Budget

    def loadj(d, n):
        return [json.loads(l) for l in (Path(d) / f"{n}.jsonl").read_text().splitlines() if l.strip()]

    ok = True

    # ---- TEXT (SST-2): solvable accuracy goal; expect val to clear a 0.70 bar -------------
    sd = "/Users/abdullahalghamdi/core-ml-acceptance/data"
    tr, va, te = loadj(sd, "train"), loadj(sd, "validation"), loadj(sd, "test")
    gtext = Goal(id="selftest-text", name="sst2", kind="text", labels=["class_a", "class_b"],
                 verification=VerificationSpec(metric="accuracy", threshold=0.70),
                 surface=ExperimentSurface(), budget=Budget(max_rounds=4, max_experiments=60))
    rt = run_research(gtext, tr, va, te)
    rs = rt["research_state"]
    print("=== TEXT (SST-2) ===")
    print(f"  winner={rt['winner_label']}  best_val={rt['best_val']}  (threshold=0.70)")
    print(f"  rounds_decided={len(rs['decisions'])}  memory_facts={len(rs['memory'])}")
    print(f"  next_experiment.action={rs['next_experiment']['action']}")
    cond_text = (
        rt["winner_pipe"] is not None
        and rt["best_val"] >= 0.70                                    # solved via an axis
        and len(rs["decisions"]) >= 1
        and all("evidence" in d and d["evidence"] for d in rs["decisions"])  # every decision cites evidence
        and all(k in rs for k in ("hypothesis", "findings", "decisions", "memory", "next_experiment"))
        and rs["next_experiment"]["action"] == "certify_on_locked_test"
    )
    print(f"  text checks: {'PASS' if cond_text else 'FAIL'}")
    ok = ok and cond_text

    # ---- TABULAR (Adult): imbalanced (~3:1); macro_f1 bar so rebalance axis matters -------
    ad = "/Users/abdullahalghamdi/vectorforge-harnesses/rugged/data"
    atr, ava, ate = loadj(ad, "train"), loadj(ad, "validation"), loadj(ad, "test")
    # balanced_accuracy with a 0.80 bar: the baseline logistic (~0.766 on val) is below the bar,
    # and the imbalance is real (~3:1), so the controller MUST iterate and the rebalance axis
    # (class-weighted families, ~0.824) is the lever that actually closes the gap. This proves the
    # right axis is chosen for the right measured reason, not that any axis happens to pass.
    gtab = Goal(id="selftest-tab", name="adult", kind="tabular", labels=["class_a", "class_b"],
                verification=VerificationSpec(metric="balanced_accuracy", threshold=0.80),
                surface=ExperimentSurface(), budget=Budget(max_rounds=4, max_experiments=80))
    rta = run_research(gtab, atr, ava, ate)
    rsa = rta["research_state"]
    print("=== TABULAR (Adult) ===")
    print(f"  winner={rta['winner_label']}  best_val={rta['best_val']}  (balanced_accuracy threshold=0.80)")
    print(f"  decisions: " + "; ".join(f"r{d['round']}:{d['action']}" for d in rsa['decisions']))
    print(f"  axes used: {sorted({d['action'].split(':')[0].replace('axis=','') for d in rsa['decisions'] if 'axis=' in d['action']})}")
    print(f"  memory: {rsa['memory']}")
    print(f"  next_experiment.action={rsa['next_experiment']['action']}")
    used_rebalance = any("rebalance" in d["action"] for d in rsa["decisions"])
    cond_tab = (
        rta["winner_pipe"] is not None
        and rta["best_val"] >= 0.80                                  # solved
        and used_rebalance                                          # imbalance -> the right axis was tried
        and all("evidence" in d and d["evidence"] for d in rsa["decisions"])
        and len(rsa["findings"]) >= 3
        and rsa["next_experiment"]["action"] == "certify_on_locked_test"
    )
    print(f"  tabular checks: {'PASS' if cond_tab else 'FAIL'}")
    ok = ok and cond_tab

    # ---- HONEST STOP: unsolvable bar must NOT fake a win; must ask for labels --------------
    ghard = Goal(id="selftest-hard", name="sst2-hard", kind="text", labels=["class_a", "class_b"],
                 verification=VerificationSpec(metric="accuracy", threshold=0.999),
                 surface=ExperimentSurface(), budget=Budget(max_rounds=6, max_experiments=120))
    rh = run_research(ghard, tr, va, te)
    nh = rh["research_state"]["next_experiment"]
    print("=== HONEST STOP (threshold=0.999) ===")
    print(f"  best_val={rh['best_val']}  next_experiment.action={nh['action']}")
    print(f"  reason={nh['reason']}")
    # honest: the bar is unmet, so the controller must NOT claim certification-readiness; it must
    # point at acquiring evidence, and the threshold it reports must be the original 0.999 (never
    # silently lowered to manufacture a pass).
    cond_honest = (
        rh["best_val"] < 0.999
        and nh["action"] in ("acquire_labels", "needs_more_evidence")  # never 'certify' when unmet
        and nh["evidence"]["threshold"] == 0.999                       # bar reported unchanged
    )
    print(f"  honesty checks: {'PASS' if cond_honest else 'FAIL'}")
    ok = ok and cond_honest

    print("\n" + ("PASS" if ok else "FAIL") + " brain.py self-test")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
