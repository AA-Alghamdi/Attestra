"""Cross-dataset BATTERY over our FROZEN certifier, with a CORRECT candidate-vs-baseline FDR (migration Item 3),
a per-shard PILOT that sets theta = the no-search baseline's score, and a battery-controlled evaluation split that
keeps a better-than-baseline model VISIBLE even when the loop honest-stops (no sealed peek -> no loop-emitted vectors).

Codex's battery fed Benjamini-Hochberg the certifier's binomial-survival-vs-a-fixed-floor p-value -- the WRONG
null: every certified task collapsed to p=0.001 and passed BH by construction (4 tasks were "discoveries" at
lift EXACTLY 0.0). We instead test, PER TASK, H0: the search-winner does NOT beat the no-search baseline on the
SAME held rows, via a one-sided exact McNemar test on two per-row correctness vectors. Then Benjamini-Hochberg
across tasks.

THREE fixes, all PERIPHERAL (battery only -- the frozen certifier and the loop's certify path are untouched):

  1. PILOT-THETA (migration "fix-on-copy"). Before the loop runs, the battery materializes the shard, carves a
     deterministic battery-controlled (train, eval) split with the FROZEN science.make_splits, trains the
     no-search baseline (logistic for classification / ridge for regression) on the battery-train, scores it on
     the battery-eval with the FROZEN science.score_metric, and sets theta = that baseline score. So "certify"
     means "the sealed lower bound BEATS the no-search baseline", not "clear an arbitrary registry floor". The
     registry threshold is kept as a FLOOR: theta = max(registry_floor, pilot_baseline) so a pilot can only
     RAISE the bar, never silently lower a spec the operator wrote down.

  2. BEATS-BASELINE ON HONEST-STOP. The loop emits cand/base locked-test correctness vectors ONLY on CERTIFY
     (loop.py: they default to None on honest_stop, because honest_stop takes NO sealed peek). A model that is
     better-than-baseline but not certifiable would be INVISIBLE to the FDR. So the battery carves its OWN
     evaluation split (the battery-eval above) that the LOOP NEVER SEES (it is held out of the records handed to
     the loop), and on EVERY decision it scores: (a) a battery-trained no-search baseline, and (b) the loop's
     reported winner family, REBUILT by the battery from the frozen catalog (res.winner.family/params/seed) and
     retrained on the battery-train -- both on the same battery-eval rows. This yields a paired correctness pair
     even when the loop honest-stops. It is a SEPARATE, NON-PROMOTING measurement: the frozen certifier remains
     the only promoter; the battery never certifies and never touches the sealed test.

  3. The battery PREFERS the loop's own locked vectors when the loop DID certify (those are scored on the
     loop's sealed test, the gold paired comparison) and falls back to the battery-eval pair otherwise. Both are
     honest paired comparisons on rows the model never trained on; the report records which source was used.

The battery NEVER promotes a model. A task that honest-stops is reported as not-certified but its
beats-baseline status is still computed (and contributes to the FDR) from the battery-eval pair.
"""
from __future__ import annotations

import json
from math import comb


def mcnemar_pvalue(cand_correct, base_correct):
    """One-sided exact McNemar: P(candidate does NOT beat baseline) on the same held rows.
    cand_correct / base_correct are equal-length 0/1 vectors. b = candidate-right & baseline-wrong,
    c = candidate-wrong & baseline-right, n = b + c (discordant pairs). Returns P(X >= b | Binom(n, 0.5))."""
    if not cand_correct or not base_correct or len(cand_correct) != len(base_correct):
        return 1.0
    b = sum(1 for cc, bb in zip(cand_correct, base_correct) if cc == 1 and bb == 0)
    c = sum(1 for cc, bb in zip(cand_correct, base_correct) if cc == 0 and bb == 1)
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, k) for k in range(b, n + 1)) / (2 ** n))


def _abs_err(y_true, y_pred):
    return [abs(float(t) - float(p)) for t, p in zip(y_true, y_pred)]


def regression_paired_pvalue(cand_pred, base_pred, y_true, *, error="abs", n_boot=10000, seed=12345):
    """One-sided paired significance test that the SEARCH WINNER's per-row error is LOWER than the no-search
    baseline's, on the SAME held rows. This is the regression analogue of the one-sided McNemar above
    (H0: the winner does NOT beat the baseline; small p => the winner's per-row error is genuinely lower).

    Inputs are per-row floats: cand_pred / base_pred are the winner / baseline predictions and y_true the
    targets, all aligned and equal-length. We form per-row errors (absolute by default; the FROZEN regression
    metrics neg_rmse/neg_mae are monotone in per-row |error|, and squared error is monotone in |error| too, so
    the ranking of the test is metric-consistent). Let d_i = cand_err_i - base_err_i; the alternative is
    median(d) < 0 (winner error lower).

    Primary test: one-sided Wilcoxon signed-rank (alternative='less') on d -- a paired, distribution-free,
    rank-based test directly comparable to the exact McNemar p (both are one-sided paired tests under the same
    'winner does not beat baseline' null). Fallback (no scipy / all-zero d / Wilcoxon degenerate): a one-sided
    paired BOOTSTRAP on the mean per-row error difference -- resample rows with replacement, recompute
    mean(cand_err) - mean(base_err), and report the fraction of resamples in which the winner does NOT have
    lower mean error (>= 0). Both return P(winner does not beat baseline); 1.0 means no evidence.
    `error` in {'abs','sq'} selects |error| or squared error for the per-row statistic."""
    if not cand_pred or not base_pred or not y_true:
        return 1.0
    n = len(y_true)
    if len(cand_pred) != n or len(base_pred) != n or n < 2:
        return 1.0
    ce = _abs_err(y_true, cand_pred)
    be = _abs_err(y_true, base_pred)
    if error == "sq":
        ce = [e * e for e in ce]
        be = [e * e for e in be]
    d = [c - b for c, b in zip(ce, be)]
    if all(abs(x) < 1e-15 for x in d):
        return 1.0          # identical per-row error -> no claim (mirrors McNemar's no-discordant-pairs -> 1.0)
    # Primary: one-sided Wilcoxon signed-rank, alternative 'less' (median of cand-base difference < 0).
    try:
        from scipy.stats import wilcoxon
        # zero_method='wilcox' drops exact-zero diffs (ties); 'less' tests d < 0 i.e. winner error lower.
        stat = wilcoxon(d, alternative="less", zero_method="wilcox", mode="auto")
        p = float(stat.pvalue)
        if p == p:          # not NaN
            return min(1.0, max(0.0, p))
    except Exception:  # noqa: BLE001  fall back to the bootstrap if scipy is missing or degenerate
        pass
    # Fallback: one-sided paired bootstrap on the mean per-row error difference.
    import numpy as np
    rng = np.random.default_rng(seed)
    darr = np.asarray(d, dtype=float)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = darr[idx].mean(axis=1)
    p = float((means >= 0).mean())          # P(winner does NOT have lower mean error)
    return min(1.0, max(0.0, p))


def benjamini_hochberg(pvalues, alpha=0.1):
    """Indices rejected at FDR <= alpha (BH step-up). Exact."""
    m = len(pvalues)
    if m == 0:
        return set()
    order = sorted(range(m), key=lambda i: pvalues[i])
    kmax = 0
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= alpha * rank / m:
            kmax = rank
    return {order[r] for r in range(kmax)}


# -------------------------------------------------------------------- PILOT + battery-controlled eval helpers
# A small, deterministic, OFFLINE pipeline the BATTERY owns end-to-end (it never calls the frozen certifier or
# the sealed test). It reuses the SAME frozen primitives the loop uses (science.make_splits, the harness
# featurizer + the catalog family builders, science.score_metric) so the pilot baseline and the battery-eval
# scores are computed exactly the way the loop / certifier would compute them.

def _is_regression(spec):
    return spec.get("task_type") == "regression" or (spec.get("metric") in ("r2", "neg_rmse", "neg_mae"))


def _baseline_ctor(spec):
    """The NO-SEARCH baseline: the exact reference the loop scores against on certify (loop.py base vector).
    Classification -> LogisticRegression(max_iter=2000); regression -> Ridge(alpha=1.0)."""
    if _is_regression(spec):
        from sklearn.linear_model import Ridge
        return lambda: Ridge(alpha=1.0)
    from sklearn.linear_model import LogisticRegression
    return lambda: LogisticRegression(max_iter=2000)


def _fit_predict_labels(harness, est_factory, train, eval_rows, target_key, labels):
    """Fit `est_factory()` on `train`, predict on `eval_rows`, return (y_pred_str_or_float, y_true_str_or_float).
    Uses the harness featurizer + target encoding so featurization/decoding match the loop exactly. The
    featurizer is fit on train+eval (its frozen contract is to see the full column domain, never the targets)."""
    import numpy as np
    feat = harness.featurizer().fit(train, train + eval_rows)
    ytr, l2i = harness.encode_targets(train, target_key, labels)
    Xtr, Xev = feat.transform(train), feat.transform(eval_rows)
    est = est_factory()
    est.fit(Xtr, ytr)
    pred = est.predict(Xev)
    if harness.is_regression():
        y_pred = [float(p) for p in pred]
        y_true = [float(r.get(target_key)) for r in eval_rows]
    else:
        inv = {i: lab for lab, i in l2i.items()}
        y_pred = [str(inv[int(p)]) for p in pred]
        y_true = [str(r.get(target_key)) for r in eval_rows]
    return y_pred, y_true


def _battery_split(spec, target_key, seed=12345):
    """Deterministic battery-controlled carve of the materialized records into (loop_records, eval_rows).
    eval_rows is held OUT of what the loop sees so it is unseen by BOTH the battery baseline and the rebuilt
    winner -> an honest paired comparison the battery owns. Uses the FROZEN leakage-safe splitter; the
    'test' slice (30%) becomes the battery-eval, the rest is handed to the loop. Falls back to no holdout
    (empty eval) if the data is too small to carve one."""
    from vectorforge import science
    records = spec["records"]
    is_reg = _is_regression(spec)
    if is_reg:
        import numpy as np
        vals = np.array([float(r.get(target_key)) for r in records], dtype=float)
        edges = np.quantile(vals, [i / 10.0 for i in range(1, 10)])
        bins = np.digitize(vals, edges)
        tagged = [dict(r, _bb=int(b)) for r, b in zip(records, bins)]
        tr, va, te, _ = science.make_splits(tagged, seed=seed, target_key="_bb", text_key="text")
        strip = lambda rows: [{k: v for k, v in r.items() if k != "_bb"} for r in rows]
        tr, va, te = strip(tr), strip(va), strip(te)
    else:
        tr, va, te, _ = science.make_splits(records, seed=seed, target_key=target_key, text_key="text")
    loop_records = list(tr) + list(va)
    eval_rows = list(te)
    if len(eval_rows) < 20 or len(loop_records) < 20:
        return list(records), []     # too small to hold out -> let the loop use all rows; eval pair unavailable
    return loop_records, eval_rows


def _harness_for_spec(spec, target_key):
    from .harness import harness_for
    return harness_for(spec["kind"], spec["task_type"], text_key="text")


def _pilot_theta(spec, target_key, registry_floor, seed=12345):
    """PILOT: train the no-search baseline on a battery-train split, score it on the battery-eval split with the
    FROZEN metric, and return (theta, pilot_info). theta = max(registry_floor, baseline_score): the pilot can
    only RAISE the bar (certify must beat the baseline) and never silently lower the operator's written floor.
    On any failure (or no holdout) it falls back to the registry floor and records why."""
    from vectorforge import science
    try:
        loop_records, eval_rows = _battery_split(spec, target_key, seed=seed)
        if not eval_rows:
            return registry_floor, {"pilot": "no_holdout", "baseline_score": None, "theta_source": "registry_floor"}
        harness = _harness_for_spec(spec, target_key)
        labels = spec.get("labels")
        y_pred, y_true = _fit_predict_labels(harness, _baseline_ctor(spec), loop_records, eval_rows,
                                             target_key, labels)
        metric = spec.get("metric") or harness.default_metric
        if harness.is_regression():
            score = float(science.score_metric(metric, y_true, y_pred, None))
        else:
            labs = sorted(set(y_true) | set(y_pred))
            score = float(science.score_metric(metric, y_true, y_pred, labs))
        theta = max(float(registry_floor), score) if score is not None else float(registry_floor)
        return theta, {"pilot": "ok", "baseline_score": round(score, 4), "registry_floor": float(registry_floor),
                       "theta_source": ("pilot_baseline" if theta > float(registry_floor) else "registry_floor"),
                       "n_eval": len(eval_rows), "n_loop": len(loop_records)}
    except Exception as ex:  # noqa: BLE001  the pilot must never sink a shard; fall back to the floor
        return registry_floor, {"pilot": "error", "error": str(ex)[:160], "baseline_score": None,
                                "theta_source": "registry_floor"}


def _rebuild_winner_ctor(spec, winner, target_key):
    """Rebuild the loop's reported WINNER as a battery-side estimator factory from the FROZEN catalog. The loop
    exposes winner.family (e.g. 'random_forest|n=400|max_depth=12'), winner.params (clamped), winner.seed. The
    catalog key is family.split('|')[0]; resolve_family clamps params + returns a ctor(seed)->estimator. Returns
    a zero-arg factory or None if the family is not catalog-buildable (e.g. a torch-on-worker winner)."""
    from .harness import catalog_for, resolve_family
    if winner is None:
        return None
    fam_key = str(winner.family).split("|")[0]
    cat = catalog_for(spec["kind"], spec["task_type"])
    resolved = resolve_family(cat, fam_key, winner.params or {})
    if resolved is None:
        return None
    _name, ctor, _clamped = resolved
    seed = int(getattr(winner, "seed", 0) or 0)
    return lambda: ctor(seed)


def _battery_eval_pair(spec, res, target_key, loop_records, eval_rows, seed=12345):
    """Battery-controlled paired correctness on the HELD-OUT battery-eval rows (unseen by the loop): the
    no-search baseline vs the loop's REBUILT winner, both retrained on loop_records. Returns
    (cand_correct, base_correct) 0/1 vectors or (None, None) if it cannot be formed (no holdout / no winner /
    non-catalog winner / regression). NON-PROMOTING: this never calls the certifier or the sealed test."""
    if not eval_rows:
        return None, None
    harness = _harness_for_spec(spec, target_key)
    if harness.is_regression():
        return None, None          # classification path: regression uses _battery_eval_regression_pair instead
    winner = getattr(res, "winner", None)
    win_factory = _rebuild_winner_ctor(spec, winner, target_key)
    if win_factory is None:
        return None, None
    labels = spec.get("labels")
    try:
        cand_pred, y_true = _fit_predict_labels(harness, win_factory, loop_records, eval_rows, target_key, labels)
        base_pred, _ = _fit_predict_labels(harness, _baseline_ctor(spec), loop_records, eval_rows,
                                           target_key, labels)
    except Exception:  # noqa: BLE001  battery-eval is additive; never let it sink a shard
        return None, None
    cand_correct = [1 if a == b else 0 for a, b in zip(cand_pred, y_true)]
    base_correct = [1 if a == b else 0 for a, b in zip(base_pred, y_true)]
    return cand_correct, base_correct


def _battery_eval_regression_pair(spec, res, target_key, loop_records, eval_rows):
    """Battery-controlled REGRESSION paired predictions on the HELD-OUT battery-eval rows (unseen by the loop):
    the no-search baseline (Ridge) vs the loop's REBUILT winner, both retrained on loop_records. Returns
    (cand_pred, base_pred, y_true) per-row float vectors, or (None, None, None) if it cannot be formed (no
    holdout / no winner / non-catalog winner / classification). NON-PROMOTING: never calls the certifier or the
    sealed test. The loop emits NO regression sealed vectors (it scores classification correctness only), so the
    battery-eval pair is the sole honest source for the regression beats-baseline test."""
    if not eval_rows:
        return None, None, None
    harness = _harness_for_spec(spec, target_key)
    if not harness.is_regression():
        return None, None, None
    winner = getattr(res, "winner", None)
    win_factory = _rebuild_winner_ctor(spec, winner, target_key)
    if win_factory is None:
        return None, None, None
    labels = spec.get("labels")
    try:
        cand_pred, y_true = _fit_predict_labels(harness, win_factory, loop_records, eval_rows, target_key, labels)
        base_pred, _ = _fit_predict_labels(harness, _baseline_ctor(spec), loop_records, eval_rows,
                                           target_key, labels)
    except Exception:  # noqa: BLE001  battery-eval is additive; never let it sink a shard
        return None, None, None
    return [float(x) for x in cand_pred], [float(x) for x in base_pred], [float(x) for x in y_true]


def _default_runner(task, *, llm_propose=False):
    """Materialize a real dataset, PILOT theta = the no-search baseline score, run the frozen-certifier loop on
    a battery-held-out subset, and return the battery row inputs. Default is deterministic + cheap
    (llm_propose=False). With llm_propose=True the LLM-guided PROPOSE step drives a richer search (needs
    ANTHROPIC_API_KEY); the certifier and the candidate-vs-baseline null are unchanged.

    Vector SOURCE: prefer the loop's own locked vectors (emitted on CERTIFY, scored on the loop's sealed test);
    on honest_stop / do_not_certify with no loop vectors, fall back to the battery-eval pair (winner-refit vs
    baseline on the battery-held-out rows). Both are paired comparisons on rows the model never trained on."""
    from .frontdoor import run as frontdoor_run
    from . import connectors
    spec = connectors.materialize(task["source_uri"], max_rows=task.get("max_rows", 3000))
    metric = task.get("metric") or spec["metric"]
    target_key = spec["target_key"]
    registry_floor = task["threshold"]

    # 1) PILOT: theta = baseline score (>= the operator's registry floor).
    theta, pilot = _pilot_theta(spec, target_key, registry_floor)

    # 2) Battery-controlled holdout: hand the loop only loop_records; keep eval_rows for the beats-baseline pair.
    loop_records, eval_rows = _battery_split(spec, target_key)

    out = frontdoor_run(loop_records, f"classify, {metric} >= {theta}", threshold=theta,
                        kind=spec["kind"], task_type=spec["task_type"], target_key=target_key,
                        labels=spec.get("labels"), metric=metric,
                        llm_enabled=bool(llm_propose), llm_propose=bool(llm_propose))
    res = out["result"]
    cert = res.certificate or {}

    if _is_regression(spec):
        # 3R) REGRESSION beats-baseline: the loop emits NO regression sealed vectors (it scores classification
        # correctness only), so the battery-eval per-row prediction pair is the sole honest source. The paired
        # significance test (regression_paired_pvalue) is applied by run_battery on these per-row vectors.
        cand_pred, base_pred, y_true = _battery_eval_regression_pair(spec, res, target_key, loop_records, eval_rows)
        vec_source = "battery_eval_regression" if (cand_pred and base_pred and y_true) else "none"
        return {"certified": res.decision == "certified", "decision": res.decision,
                "is_regression": True, "cand_pred": cand_pred, "base_pred": base_pred, "y_true": y_true,
                "vec_source": vec_source, "observed": cert.get("observed"),
                "lower_bound": cert.get("lower_bound"), "n_test": res.n_test, "source": spec.get("source"),
                "theta": theta, "pilot": pilot}

    # 3) Vectors: loop's own (certify) first; else the battery-eval pair (honest_stop / do_not_certify).
    cand = getattr(res, "cand_locked_correct", None)
    base = getattr(res, "base_locked_correct", None)
    vec_source = "loop_sealed"
    if not (cand and base):
        cand, base = _battery_eval_pair(spec, res, target_key, loop_records, eval_rows)
        vec_source = "battery_eval" if (cand and base) else "none"

    return {"certified": res.decision == "certified", "decision": res.decision,
            "is_regression": False, "cand_correct": cand, "base_correct": base, "vec_source": vec_source,
            "observed": cert.get("observed"), "lower_bound": cert.get("lower_bound"),
            "n_test": res.n_test, "source": spec.get("source"),
            "theta": theta, "pilot": pilot}


_REGRESSION_METRICS = ("r2", "neg_rmse", "neg_mae")


def _is_regression_task(task, runner_result):
    """Pick the regression paired test iff this is a regression task: the runner can flag it explicitly
    (is_regression) or it is inferred from the metric (r2/neg_rmse/neg_mae), matching _is_regression()."""
    if runner_result.get("is_regression") is not None:
        return bool(runner_result["is_regression"])
    return task.get("metric") in _REGRESSION_METRICS or task.get("task_type") == "regression"


def _lift(cand, base):
    if not cand or not base:
        return None
    return round(sum(cand) / len(cand) - sum(base) / len(base), 4)


def _regression_lift(cand_pred, base_pred, y_true):
    """Mean absolute-error REDUCTION (baseline_mae - winner_mae); positive => winner has lower error."""
    if not cand_pred or not base_pred or not y_true:
        return None
    ce = _abs_err(y_true, cand_pred)
    be = _abs_err(y_true, base_pred)
    return round(sum(be) / len(be) - sum(ce) / len(ce), 4)


def run_battery(registry_path, *, alpha=0.1, runner=None, on_task=None, llm_propose=False):
    """Run every shard in the registry, PILOT theta = the no-search baseline score per shard, compute a
    candidate-vs-baseline McNemar p-value per task, and control FDR across tasks with BH. `runner(task) ->
    {certified, cand_correct, base_correct, theta, pilot, vec_source, ...}` (injectable for tests).
    llm_propose=True drives the richer LLM-guided search in the default runner. Returns a battery report.
    The FDR is over candidate-vs-baseline -- NOT binomial-vs-floor. The battery NEVER promotes; only the frozen
    certifier (inside the loop) promotes."""
    runner = runner or (lambda t: _default_runner(t, llm_propose=llm_propose))
    tasks = json.load(open(registry_path)) if isinstance(registry_path, str) else registry_path
    rows = []
    for t in tasks:
        try:
            r = runner(t)
        except Exception as ex:  # noqa: BLE001  a single shard's failure (e.g. offline) must not sink the battery
            rows.append({**t, "status": "error", "error": str(ex)[:200], "p_value": 1.0,
                         "certified": False, "lift_over_baseline": None})
            if on_task:
                on_task(rows[-1])
            continue
        if _is_regression_task(t, r):
            # REGRESSION: one-sided paired test that the winner's per-row error is LOWER than the baseline's.
            p = regression_paired_pvalue(r.get("cand_pred"), r.get("base_pred"), r.get("y_true"))
            lift = _regression_lift(r.get("cand_pred"), r.get("base_pred"), r.get("y_true"))
            test_name = "regression_paired_error"
        else:
            # CLASSIFICATION: one-sided exact McNemar on per-row correctness.
            p = mcnemar_pvalue(r.get("cand_correct"), r.get("base_correct"))
            lift = _lift(r.get("cand_correct"), r.get("base_correct"))
            test_name = "mcnemar"
        row = {"task_id": t["task_id"], "source": r.get("source"), "metric": t.get("metric"),
               "status": "ok", "decision": r.get("decision"), "certified": bool(r.get("certified")),
               "registry_floor": t.get("threshold"), "theta": r.get("theta"), "pilot": r.get("pilot"),
               "vec_source": r.get("vec_source"), "test": test_name,
               "observed": r.get("observed"), "lower_bound": r.get("lower_bound"), "n_test": r.get("n_test"),
               "lift_over_baseline": lift, "p_value": round(p, 6)}
        rows.append(row)
        if on_task:
            on_task(row)
    rejected = benjamini_hochberg([x["p_value"] for x in rows], alpha=alpha)
    for i, x in enumerate(rows):
        x["beats_baseline_fdr"] = i in rejected
    discoveries = [rows[i]["task_id"] for i in sorted(rejected)]
    return {"alpha": alpha, "m": len(rows), "n_certified": sum(1 for x in rows if x["certified"]),
            "n_discoveries": len(rejected), "discoveries": discoveries, "tasks": rows,
            "theta_rule": "per-shard PILOT: theta = max(registry_floor, no-search baseline score)",
            "null": "candidate-vs-baseline (NOT binomial-vs-floor): classification=one-sided exact McNemar, "
                    "regression=one-sided paired error test (Wilcoxon signed-rank on per-row |error|, "
                    "bootstrap fallback); honest_stop visible via battery-eval"}


def _selftest():
    # Synthetic runners (no network) prove the FDR logic is honest -- the exact case Codex got wrong --
    # AND that an honest-stop model that beats baseline on the battery-eval is still a discovery.
    rng_n = 200

    def make(cand_acc, base_acc):
        # deterministic correctness vectors with the given accuracies (front-loaded 1s)
        cand = [1] * int(cand_acc * rng_n) + [0] * (rng_n - int(cand_acc * rng_n))
        base = [1] * int(base_acc * rng_n) + [0] * (rng_n - int(base_acc * rng_n))
        return cand, base

    def runner(task):
        cand, base = task["_vec"]
        return {"certified": task.get("_certified", True), "decision": task.get("_decision", "certified"),
                "cand_correct": cand, "base_correct": base, "vec_source": task.get("_vsrc", "loop_sealed"),
                "observed": sum(cand) / len(cand), "lower_bound": None, "n_test": len(cand),
                "source": "synthetic", "theta": 0.80, "pilot": {"pilot": "ok", "baseline_score": 0.80}}

    tasks = [
        {"task_id": "identical-to-baseline", "metric": "accuracy", "threshold": 0.5, "_vec": make(0.80, 0.80)},
        {"task_id": "strong-improvement", "metric": "accuracy", "threshold": 0.5, "_vec": make(0.92, 0.78)},
        {"task_id": "tiny-improvement", "metric": "accuracy", "threshold": 0.5, "_vec": make(0.81, 0.80)},
        # honest_stop but beats baseline on the battery-eval pair -> MUST still be a discovery (the fix)
        {"task_id": "honest-stop-but-beats", "metric": "accuracy", "threshold": 0.5,
         "_vec": make(0.90, 0.78), "_certified": False, "_decision": "honest_stop", "_vsrc": "battery_eval"},
    ]
    rep = run_battery(tasks, alpha=0.1, runner=runner)
    by = {r["task_id"]: r for r in rep["tasks"]}
    # THE FIX (Item 3 original): a candidate identical to the baseline is NOT a discovery.
    assert by["identical-to-baseline"]["p_value"] == 1.0, by["identical-to-baseline"]
    assert not by["identical-to-baseline"]["beats_baseline_fdr"], by["identical-to-baseline"]
    # a strong, real improvement IS a discovery
    assert by["strong-improvement"]["p_value"] < 0.01, by["strong-improvement"]
    assert by["strong-improvement"]["beats_baseline_fdr"], by["strong-improvement"]
    # HONEST-STOP VISIBILITY: a not-certified model that beats baseline on the battery-eval still surfaces.
    assert by["honest-stop-but-beats"]["certified"] is False, by["honest-stop-but-beats"]
    assert by["honest-stop-but-beats"]["beats_baseline_fdr"], by["honest-stop-but-beats"]
    assert by["honest-stop-but-beats"]["vec_source"] == "battery_eval", by["honest-stop-but-beats"]
    # exact McNemar spot check: 14 candidate-only wins, 0 baseline-only wins -> p = 0.5**14
    assert abs(mcnemar_pvalue([1] * 14 + [0] * 6, [0] * 14 + [0] * 6) - 0.5 ** 14) < 1e-12

    # REGRESSION beats-baseline (Tier-1 Gap A): a winner with genuinely lower per-row error IS a discovery;
    # an equal-error winner is NOT (p=1.0). The regression null is the one-sided paired error test, fed to BH.
    def reg_runner(task):
        cp, bp, yt = task["_reg"]
        return {"certified": task.get("_certified", False), "decision": task.get("_decision", "honest_stop"),
                "is_regression": True, "cand_pred": cp, "base_pred": bp, "y_true": yt,
                "vec_source": "battery_eval_regression", "observed": None, "lower_bound": None,
                "n_test": len(yt), "source": "synthetic-reg", "theta": None, "pilot": {"pilot": "ok"}}

    rng = __import__("numpy").random.default_rng(7)
    m = 150
    yt = rng.normal(0, 1, m).tolist()
    base_pred = [y + rng.normal(0, 1.0) for y in yt]           # baseline: large per-row error
    win_pred = [y + rng.normal(0, 0.2) for y in yt]            # winner: genuinely lower per-row error
    equal_pred = list(base_pred)                                # equal-error winner: identical to baseline
    reg_tasks = [
        {"task_id": "reg-lower-error", "metric": "neg_rmse", "threshold": -1.0, "_reg": (win_pred, base_pred, yt)},
        {"task_id": "reg-equal-error", "metric": "neg_rmse", "threshold": -1.0, "_reg": (equal_pred, base_pred, yt)},
    ]
    rep_r = run_battery(reg_tasks, alpha=0.1, runner=reg_runner)
    byr = {r["task_id"]: r for r in rep_r["tasks"]}
    assert byr["reg-lower-error"]["test"] == "regression_paired_error", byr["reg-lower-error"]
    assert byr["reg-lower-error"]["p_value"] < 0.01, byr["reg-lower-error"]
    assert byr["reg-lower-error"]["beats_baseline_fdr"], byr["reg-lower-error"]
    assert byr["reg-equal-error"]["p_value"] == 1.0, byr["reg-equal-error"]
    assert not byr["reg-equal-error"]["beats_baseline_fdr"], byr["reg-equal-error"]

    print("battery self-test: PASS",
          {"discoveries": rep["discoveries"], "identical_p": by["identical-to-baseline"]["p_value"],
           "strong_p": by["strong-improvement"]["p_value"],
           "honest_stop_visible": by["honest-stop-but-beats"]["beats_baseline_fdr"],
           "reg_lower_p": byr["reg-lower-error"]["p_value"], "reg_equal_p": byr["reg-equal-error"]["p_value"]})


if __name__ == "__main__":
    _selftest()
