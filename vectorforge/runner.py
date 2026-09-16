"""The DAG runner: executes a goal end to end through durable, resumable stages.

stages: profile -> audit -> split -> data_engine -> research -> certify -> finalize

Each stage records its result in goal.dag[stage] and the goal is persisted after every stage, so an
interrupted run resumes from the last completed stage. Promotion is gated on the certifier AND the
latency constraint AND a clean leakage audit. A goal that cannot certify returns an honest failure
report; one that is data-limited returns a needs-input request.

The ADAPTIVE loop (this is what makes the runner a research controller, not a single scan):

  * After the leakage-safe split and the post-split audit, the runner asks the DATA ENGINE
    (data_engine.run_data_engine) whether the highest-leverage move is on the DATA, not the model:
    CLEAN (drop exact dups / quarantine a leaking feature), GENERATE (measured synthetic augmentation,
    only when the surface permits synthetic_data AND the budget approves spend), or REQUEST_LABELS
    (data-limited and not honestly manufacturable). The engine is invoked only when the goal is
    data-limited or carries a measurable quality issue; its cleaned/augmented train is RE-AUDITED
    against the locked test before it is adopted, so a data move can never weaken the leakage guarantee.

  * Model search is then the multi-round BRAIN (brain.run_research), which diagnoses on validation,
    names the single highest-value axis the surface allows, acts, keeps the best-on-validation
    pipeline, and re-diagnoses. The brain NEVER reads the locked test. The runner certifies the brain's
    winner on the locked test exactly ONCE, identically to before.

Nothing in the adaptive loop can promote a goal: promotion is still gated solely by
science.certify_accuracy (accuracy) / a bootstrap lower bound (other metrics) on the locked test,
plus the latency constraint and a clean leakage audit.
"""

from collections import Counter

from . import science, ml, store, brain, data_engine
from .domain import (Goal, RUNNING, PASSED, FAILED, BLOCKED, NEEDS_INPUT,
                     Certificate, Deployment)

STAGES = ["profile", "audit", "split", "data_engine", "research", "certify", "finalize"]


def _done(goal, stage):
    return goal.dag.get(stage, {}).get("status") == "done"


def _mark(goal, stage, **data):
    goal.dag[stage] = {"status": "done", **data}
    store.append_log(goal, {"stage": stage, **{k: v for k, v in data.items() if k != "predictions"}})
    store.save(goal)


def run_goal(goal: Goal, resume=True) -> Goal:
    goal.status = RUNNING
    store.save(goal)

    # 1. profile
    if not (resume and _done(goal, "profile")):
        raw = store.read_rows(goal.raw_path)
        bal = dict(Counter(str(r.get("target")) for r in raw))
        _mark(goal, "profile", n=len(raw), classes=len(goal.labels), balance=bal)

    # 2. leakage audit on raw
    if not (resume and _done(goal, "audit")):
        raw = store.read_rows(goal.raw_path)
        rep = science.audit(raw, raw, allow_features=(goal.kind == "tabular"), min_test_n=1)
        # creation-time audit on the whole set (split audit happens after split)
        leaky = [f for f in rep["findings"] if f["gate"] == "feature_target_leakage" and not f["ok"]]
        _mark(goal, "audit", leakage_findings=[f["feature"] for f in leaky], ok=not leaky)
        if leaky and goal.kind == "tabular":
            goal.status = BLOCKED
            goal.failure_report = {"diagnosis": f"leakage detected in features {[f['feature'] for f in leaky]}; "
                                   "quarantine them before training", "blocked": True}
            return store.save(goal)

    # 3. stratified leakage-safe split
    if not (resume and _done(goal, "split")):
        raw = store.read_rows(goal.raw_path)
        tr, va, te, rep = science.make_splits(raw, seed=0)
        store.write_rows(goal.id, "train", tr); store.write_rows(goal.id, "val", va); store.write_rows(goal.id, "test", te)
        goal.split_report = rep
        _mark(goal, "split", **rep)
        # data-limited check -> needs labels
        if rep["counts"]["test"] < goal.verification.min_heldout_n or rep["counts"]["train"] < 60:
            goal.status = NEEDS_INPUT
            goal.needs_input = {"kind": "labels", "ask": f"held-out test n={rep['counts']['test']} is below the required "
                                f"{goal.verification.min_heldout_n}; provide more labeled examples",
                                "current_counts": rep["counts"]}
            return store.save(goal)

    tr = store.read_rows(store._dir(goal.id) / "train.jsonl")
    va = store.read_rows(store._dir(goal.id) / "val.jsonl")
    te = store.read_rows(store._dir(goal.id) / "test.jsonl")

    # post-split leakage audit (train vs locked test)
    split_audit = science.audit(tr, te, allow_features=(goal.kind == "tabular"),
                                min_test_n=goal.verification.min_heldout_n)

    # 3b. ADAPTIVE DATA STAGE: ask the data engine whether the highest-leverage move is on the DATA.
    # Only invoked when the goal is data-limited OR carries a measurable quality issue (exact dups /
    # leaking feature flagged by the post-split audit). Its train is RE-AUDITED before adoption so a
    # data move can never weaken the leakage guarantee. NEEDS_INPUT here is an honest stop.
    if not (resume and _done(goal, "data_engine")):
        data_limited = len(tr) < data_engine.DATA_LIMITED_TRAIN
        # a measurable quality issue: a feature-target leakage finding survived to the post-split audit,
        # or train carries exact-duplicate rows. Both are things the engine can CLEAN.
        leak_findings = [f for f in split_audit["findings"]
                         if f.get("gate") == "feature_target_leakage" and not f.get("ok", True)]
        _deduped, n_dup, _conf = data_engine._exact_duplicates(tr)
        quality_issue = bool(leak_findings) or n_dup > 0

        if data_limited or quality_issue:
            allow_api = bool(goal.surface.synthetic_data and goal.budget.approve_spend)
            de = data_engine.run_data_engine(goal, tr, va, te, allow_api=allow_api)
            move, new_train = de["move"], de["new_train"]
            adopted = False
            adopt_audit_passed = None
            if move in ("CLEAN", "GENERATE") and new_train is not None and new_train is not tr \
                    and len(new_train) > 0:
                # RE-AUDIT the engine's proposed train against the locked test BEFORE adopting it. A
                # cleaned/augmented set is adopted only if it still passes science.audit; otherwise we
                # keep the original train and record the rejection. This is the load-bearing guarantee
                # that a data move cannot smuggle leakage past the certifier.
                re_audit = science.audit(new_train, te, allow_features=(goal.kind == "tabular"),
                                         min_test_n=goal.verification.min_heldout_n)
                if re_audit["passed"]:
                    store.write_rows(goal.id, "train", new_train)
                    tr = new_train
                    adopted = True
                    adopt_audit_passed = True
                    # refresh the post-split audit to reflect the adopted, cleaner train
                    split_audit = re_audit
                else:
                    adopt_audit_passed = False
            _mark(goal, "data_engine", move=move, decision=de["decision"], adopted=adopted,
                  re_audit_passed=adopt_audit_passed, n_train_after=len(tr),
                  report=de.get("report"))

            # honest stop: the engine determined the limit is data (too few labels, nothing safely
            # generable). Surface that as NEEDS_INPUT with the concrete ask; do not fake a model run.
            if move == "REQUEST_LABELS" and data_limited:
                req = de.get("report", {}).get("request") or {}
                goal.needs_input = {"kind": "labels",
                                    "ask": req.get("ask") or de["decision"],
                                    "data_engine_decision": de["decision"],
                                    "per_class": req.get("per_class"),
                                    "current_counts": req.get("current_counts")}
                goal.status = NEEDS_INPUT
                return store.save(goal)
        else:
            _mark(goal, "data_engine", move="NONE",
                  decision="data is clean and not data-limited; no data move taken",
                  adopted=False, re_audit_passed=None, n_train_after=len(tr))

    # the data stage may have rewritten train; re-read so research uses the adopted set, and rebuild
    # the split audit so certification gates on the train that actually trained the winner.
    tr = store.read_rows(store._dir(goal.id) / "train.jsonl")
    split_audit = science.audit(tr, te, allow_features=(goal.kind == "tabular"),
                                min_test_n=goal.verification.min_heldout_n)

    # 4. ADAPTIVE MODEL SEARCH: the multi-round research controller (replaces the single scan).
    # Selects entirely on validation; the locked test is NEVER read here. The winner is certified once
    # below, exactly as the single scan's winner used to be.
    if not (resume and _done(goal, "research")):
        rs = brain.run_research(goal, tr, va, te)
        goal.leaderboard = rs["leaderboard"]
        goal.research_state = rs["research_state"]
        art = store.save_artifact(goal.id, "winner",
                                  {"pipe": rs["winner_pipe"], "cfg": rs["winner_cfg"], "kind": goal.kind})
        _mark(goal, "research", winner=rs["winner_label"], winner_val=rs["best_val"],
              next_experiment=rs["research_state"]["next_experiment"], artifact=art)
    else:
        # resumed past research: rehydrate the durable research state onto the goal object
        goal.research_state = goal.research_state or {}

    # 5. certify the winner on the locked test ONCE
    art = store.load_artifact(goal.dag["research"]["artifact"])
    pipe, cfg = art["pipe"], art["cfg"]
    ev = ml.evaluate(pipe, goal.kind, te, cfg, goal.labels, goal.verification.metric)
    thr = goal.verification.threshold
    metric = goal.verification.metric
    science.assert_certifiable_metric(metric)            # F4 defense-in-depth (also gated at create_goal)
    if metric == "accuracy":
        c = science.certify_accuracy(ev["accuracy"], len(te), thr, checks=1, alpha=goal.verification.alpha)
        certified, lower, pval = c["certified"], c["lower_bound"], c["p_value"]
    elif metric in ("r2", "neg_rmse", "neg_mae"):
        # F27: use the FROZEN regression certifier (finite-sample shrink + Bonferroni), not an inline
        # anti-conservative percentile bootstrap.
        c = science.certify_regression(ev["y_true"], ev["predictions"], metric, thr,
                                       checks=1, alpha=goal.verification.alpha)
        certified, lower, pval = c["certified"], c["lower_bound"], c.get("p_value")
    else:
        # balanced_accuracy / macro_f1: the SAME frozen classification bootstrap lower bound the sealed
        # certifier uses, not a hand-rolled percentile with no shrink.
        lower, _point = science._bootstrap_classification_metric_lower(
            metric, [str(x) for x in ev["y_true"]], [str(x) for x in ev["predictions"]],
            goal.labels, alpha=goal.verification.alpha)
        lower = round(float(lower), 4)
        certified, pval = bool(lower > thr), None

    latency_ok = ev["latency_ms_p95"] <= goal.verification.max_latency_ms
    passed = bool(certified and latency_ok and split_audit["passed"])
    cert = Certificate(decision="certified" if passed else ("blocked" if not split_audit["passed"] else "do_not_certify"),
                       metric=goal.verification.metric, observed=ev["value"], threshold=thr, n=len(te),
                       lower_bound=lower, p_value=pval, latency_ms_p95=ev["latency_ms_p95"],
                       leakage_passed=split_audit["passed"],
                       evidence_strength="CONFIRMED" if passed else ("FAILED" if certified is False else "WEAK"),
                       reason=("certified" if passed else ("leakage" if not split_audit["passed"] else
                               ("latency exceeds budget" if not latency_ok else "lower bound does not clear threshold"))))
    goal.certificate = vars(cert)
    _mark(goal, "certify", observed=ev["value"], lower_bound=lower, certified=passed, latency_ms_p95=ev["latency_ms_p95"])

    # 6. finalize: deploy if passed, else failure report
    if passed:
        endpoint = f"local://goals/{goal.id}/predict"
        smoke = pipe.predict([ml._Xy(goal.kind, te[:1], cfg)[0][0]])
        goal.deployment = vars(Deployment(status="ready", artifact_path=goal.dag["research"]["artifact"],
                                          endpoint=endpoint, smoke_ok=bool(len(smoke))))
        goal.status = PASSED
    else:
        weak = [l for l, d in ev["per_class"].items() if d["recall"] is not None and d["recall"] < thr]
        # an honest forward pointer: prefer the brain's own next_experiment when it asks for evidence,
        # else fall back to the metric-gap recommendation.
        nxt = (goal.research_state or {}).get("next_experiment") or {}
        goal.failure_report = {"diagnosis": cert.reason, "gap": round(thr - ev["value"], 4),
                               "weak_classes": weak, "observed": ev["value"], "lower_bound": lower,
                               "next_experiment": nxt,
                               "recommendation": ("reduce model / faster family for latency" if not latency_ok else
                                                  ("collect more labels for " + ", ".join(weak) if weak else
                                                   "collect more held-out evidence; the gap is within noise"))}
        goal.status = FAILED if split_audit["passed"] else BLOCKED
    _mark(goal, "finalize", status=goal.status)
    return store.save(goal)
