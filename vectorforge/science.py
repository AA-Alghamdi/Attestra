"""The scientific verification spine: certifier, leakage auditor, stratified splitting, calibration,
metrics. Pure numpy + stdlib so every number is reproducible. This is what makes "it only comes back
when verified" a true statement instead of a slogan.
"""

import hashlib
import math
import re
from collections import Counter, defaultdict

import numpy as np

_WORD = re.compile(r"[a-z0-9']+")


# =========================================================================== metrics
def accuracy(y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / max(len(y_true), 1)


def balanced_accuracy(y_true, y_pred, labels):
    recalls = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        n = sum(1 for a in y_true if a == lab)
        recalls.append(tp / n if n else 0.0)
    return float(np.mean(recalls))


def macro_f1(y_true, y_pred, labels):
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append((2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


# the only metrics with a BUILT frozen lower-bound certifier. Anything else must be refused at goal
# construction -- never silently scored as accuracy (which would certify the wrong number, F4).
KNOWN_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "r2", "neg_rmse", "neg_mae")


def assert_certifiable_metric(metric):
    if metric not in KNOWN_METRICS:
        raise ValueError(
            f"metric {metric!r} has no frozen certifier; supported: {KNOWN_METRICS}. Refusing because "
            f"score_metric would otherwise fall through to ACCURACY and certify the wrong quantity.")


def score_metric(metric, y_true, y_pred, labels):
    if metric == "balanced_accuracy":
        return balanced_accuracy(y_true, y_pred, labels)
    if metric == "macro_f1":
        return macro_f1(y_true, y_pred, labels)
    if metric in ("r2", "neg_rmse", "neg_mae"):
        return score_regression_metric(metric, y_true, y_pred)
    if metric != "accuracy":
        raise ValueError(f"unknown metric {metric!r} (supported: {KNOWN_METRICS})")
    return accuracy(y_true, y_pred)


# =========================================================================== regression metrics
# Three conventional regression metrics, all in HIGHER-IS-BETTER orientation so the loop's "lower bound must
# clear the threshold" discipline is identical to classification:
#   r2       -- coefficient of determination 1 - SS_res/SS_tot (1.0 perfect, 0.0 = predicts the mean, <0 worse)
#   neg_rmse -- NEGATED root-mean-squared-error (0 perfect, more negative = worse) so larger is better
#   neg_mae  -- NEGATED mean-absolute-error (same orientation)
# r2 is NOT a mean of per-row quantities (its denominator SS_tot is a property of the whole sample), so the
# regression certifier resamples (y_true, pred) PAIRS and recomputes the whole metric per bootstrap draw
# rather than bootstrapping a per-row value vector. neg_rmse/neg_mae are monotone in per-row errors but rmse
# is a nonlinear (sqrt) function of the mean squared error, so they too are recomputed per draw for exactness.
def r2_score(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0:
        return 0.0
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 0.0:                     # constant target -> R^2 undefined; perfect fit -> 1, else 0
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def neg_rmse(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0:
        return 0.0
    return -float(np.sqrt(np.mean((yt - yp) ** 2)))


def neg_mae(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0:
        return 0.0
    return -float(np.mean(np.abs(yt - yp)))


def score_regression_metric(metric, y_true, y_pred):
    if metric == "r2":
        return r2_score(y_true, y_pred)
    if metric == "neg_rmse":
        return neg_rmse(y_true, y_pred)
    if metric == "neg_mae":
        return neg_mae(y_true, y_pred)
    raise ValueError(f"unknown regression metric {metric!r}")


def per_class_recall(y_true, y_pred, labels):
    out = {}
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        n = sum(1 for a in y_true if a == lab)
        out[lab] = {"recall": round(tp / n, 4) if n else None, "support": n}
    return out


# =========================================================================== certifier
def _log_choose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_binom_pmf(k, n, p):
    if p <= 0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1:
        return 0.0 if k == n else -math.inf
    return _log_choose(n, k) + k * math.log(p) + (n - k) * math.log1p(-p)


def binom_sf(k, n, p):
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    terms = [_log_binom_pmf(i, n, p) for i in range(k, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms)) if math.isfinite(m) else 0.0


def clopper_pearson_lower(k, n, alpha):
    if n <= 0 or k <= 0:
        return 0.0
    lo, hi = 0.0, k / n
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if binom_sf(k, n, mid) <= alpha:
            lo = mid
        else:
            hi = mid
    return lo


def certify_accuracy(observed, n, theta, checks=1, alpha=0.05):
    """Canonical certifier: certified iff Clopper-Pearson lower bound (Bonferroni-corrected for `checks`
    locked-test peeks) clears theta AND the exact binomial one-sided p-value beats alpha/checks."""
    checks = max(1, int(checks))
    a = alpha / checks
    k = max(0, min(n, int(round(observed * n))))
    p = binom_sf(k, n, theta)
    lower = clopper_pearson_lower(k, n, a)
    certified = bool(lower > theta and p < a)
    return {"observed": round(observed, 4), "n": n, "k": k, "theta": round(theta, 4), "checks": checks,
            "alpha_per_check": round(a, 6), "p_value": float(f"{p:.3e}"), "lower_bound": round(lower, 4),
            "certified": certified,
            "reason": "lower bound clears theta after multiplicity correction" if certified
            else "lower confidence bound does not clear theta after paying for all peeks"}


def bootstrap_lower(values, alpha=0.05, B=1500, seed=0):
    """Lower (1-alpha) bound on the mean of a metric via bootstrap (for non-binomial metrics)."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.array([values[rng.integers(0, n, n)].mean() for _ in range(B)])
    return round(float(np.percentile(boots, 100 * alpha)), 4)


# Finite-sample safety factor for the regression percentile-bootstrap lower bound. The plain percentile
# bootstrap is median-biased and ignores skew, so its naive (1-alpha) lower endpoint MILDLY UNDER-COVERS
# at the finite n a locked test actually has: measured worst-cell one-sided coverage was 0.940 for r2
# (n=500, high noise) and ~0.922-0.934 for neg_rmse (rmse is a right-skewed sqrt of resampled squared
# errors) -- below the nominal 0.95. Rather than the textbook bias/skew corrections (a BCa and a
# reverse-/basic-percentile correction were both measured to LOSE to percentile here: BCa worst-cell
# r2 ~0.938, bootstrap-t ~0.92 -- the resampled distribution underestimates sampling variability so the
# acceleration/pivot move the endpoint the wrong way), we apply ONE conservative, monotone, fully
# documented knob: take the percentile at a TIGHTER effective tail probability alpha_eff = alpha * SHRINK.
# The certifier therefore targets a slightly-better-than-nominal one-sided level (alpha=0.05 -> 0.025, a
# 97.5% target) so the EMPIRICAL coverage is >= the nominal 0.95 in every cell. SHRINK=0.5 was chosen by
# a head-to-head sweep (experiments/regression_lb_{design,shrink,confirm}.py): it lifts worst-cell coverage
# to r2 0.966 / neg_rmse 0.957 (gate seed) and r2 0.971 / neg_rmse 0.967 (held-out seed) -- both >= 0.95
# with margin -- at the cost of only a small extra mean lower-bound gap, and a genuine strong-signal
# regression still certifies (california_housing lower 0.5363 > 0.5, synthetic-R2 lower 0.8916 > 0.8; see
# experiments/regression_power_check.py). A finite-sample 1/sqrt(n) shrink was ALSO measured but was too
# weak (worst-cell r2 only ~0.95, neg_rmse ~0.944, FAILED), so we use the fixed factor. This is NOT relaxing
# a spec: it makes the bound MORE conservative.
_REG_FS_SHRINK = 0.5

# Residual-distribution ENVELOPE for the regression bootstrap (F11). MEASURED (coverage study, real sims):
# the fs_shrink=0.5 percentile lower bound is conservative (>=0.95) ONLY for light-tailed, near-symmetric
# residuals; off-envelope coverage collapses (neg_rmse ~0.67, r2 ~0.81 under right-skew) and NO sample-moment
# trick can rescue it (a finite heavy-tailed sample under-reports its own kurtosis -> the gate is fooled by
# exactly the samples it must reject). So the gate is used ONLY to DEFER (refuse to certify) anything outside
# the validated envelope -- it never licenses a heavy-tail certificate. neg_rmse is the most tail-fragile and
# gets a higher n-floor. Within the envelope, measured coverage is r2 ~0.98 / neg_rmse ~0.95 / neg_mae pass.
_REG_MIN_N = 100
_REG_NEG_RMSE_MIN_N = 150
_REG_SKEW_MAX = 0.5
_REG_EXCESS_KURT_MAX = 1.0


def bootstrap_metric_lower(y_true, y_pred, metric, alpha=0.05, B=2000, seed=0, fs_shrink=_REG_FS_SHRINK):
    """One-sided (>= 1-alpha) LOWER confidence bound on a HIGHER-IS-BETTER regression metric, via the
    percentile bootstrap of the SAMPLE METRIC with a finite-sample conservative shrink.

    This is the regression analogue of clopper_pearson_lower: it resamples the (y_true, y_pred) PAIRS with
    replacement B times, recomputes the FULL metric on each resample, and returns a LOWER percentile of the
    bootstrap metric distribution as the lower bound. Resampling whole pairs (not a per-row value vector) is
    required because r2's denominator (SS_tot) and rmse's sqrt are sample-level nonlinear functions -- a
    naive bootstrap of per-row errors would bound the wrong quantity.

    The percentile is taken at the TIGHTER probability alpha_eff = alpha * fs_shrink (default fs_shrink=0.5
    -> alpha_eff = 0.025), a single documented finite-sample safety factor (see _REG_FS_SHRINK above) that
    makes the bound conservative enough to achieve >= 0.95 EMPIRICAL one-sided coverage in every measured
    cell, for BOTH r2 and neg_rmse, without over-conservatizing into never-certifying.

    Returns (lower_bound, point_estimate). For r2 the bound is a higher-is-better R^2 floor; for
    neg_rmse/neg_mae a higher (less-negative) error floor. Empty / degenerate input -> (point, point).

    MEASURED COVERAGE (tests/test_regression_calibration.py, frozen model + giant-eval theta_star, 1000
    trials/cell): with the fs_shrink=0.5 correction the one-sided 95% lower bound has worst-cell coverage
    0.966 (r2) and 0.957 (neg_rmse), mean 0.9702, across n in {200,500,1000} x two noise regimes (gate seed),
    and 0.971 / 0.967 on an independent held-out seed -- conservatively VALID (>= 0.95) in EVERY cell for
    BOTH metrics, where the naive percentile (fs_shrink=1.0) under-covered at 0.940 (r2, n=500/high-noise)
    and 0.922 (neg_rmse). Setting fs_shrink=1.0 recovers the original (anti-conservative) percentile bound
    for reproducing the pre-fix numbers."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    n = len(yt)
    point = score_regression_metric(metric, yt, yp)
    if n <= 1:
        return round(float(point), 6), round(float(point), 6)
    rng = np.random.default_rng(seed)
    boots = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = score_regression_metric(metric, yt[idx], yp[idx])
    alpha_eff = max(0.0, min(alpha, alpha * fs_shrink))
    lower = float(np.percentile(boots, 100.0 * alpha_eff))
    return round(lower, 6), round(float(point), 6)


def certify_regression(y_true, y_pred, metric, theta, checks=1, alpha=0.05, B=2000, seed=0):
    """Canonical REGRESSION certifier, mirroring certify_accuracy's contract and discipline.

    Certified iff the bootstrap LOWER confidence bound on the (higher-is-better) metric clears theta, after a
    Bonferroni correction for `checks` locked-test peeks (alpha/checks). The lower bound is the finite-sample
    conservative percentile bootstrap (bootstrap_metric_lower applies a documented alpha*0.5 tail shrink). The
    metric is recomputed per bootstrap draw. For r2 theta is an R^2 floor (e.g. 0.5); for neg_rmse/neg_mae
    theta is a NEGATED-error floor (e.g. -3.0 means "rmse must be provably below 3.0"). The locked test is
    scored exactly once by the caller; this function only consumes (y_true, y_pred).

    VALIDATED ENVELOPE (F11, measured in tests/test_certifier_coverage.py): one-sided coverage is >= 0.95
    for r2/neg_rmse/neg_mae under LIGHT-TAILED (approx. gaussian/symmetric) residuals (n >= 100, neg_rmse
    n >= 150). Under HEAVY right-skewed / heavy-tailed residuals the percentile bootstrap under-covers
    (measured: neg_rmse ~0.67, r2 ~0.81 under lognormal) and NO sample-moment trick rescues it. So the
    certifier GATES on a measured residual envelope (|skew| <= 0.5, excess_kurt <= 1.0, n-floor) and
    DEFERS -- certified=False, deferred=True, in_envelope=False -- for anything outside it, rather than
    emit an invalid 95% bound. It never promotes out-of-envelope. (classification + the exact
    Clopper-Pearson accuracy path are unaffected.)"""
    checks = max(1, int(checks))
    a = alpha / checks
    n = len(y_true)
    lower, point = bootstrap_metric_lower(y_true, y_pred, metric, alpha=a, B=B, seed=seed)
    # residual moments -> validated-envelope gate (no scipy dependency)
    resid = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    skew = kurt = 0.0
    if resid.size >= 8:
        sd = float(resid.std())
        if sd > 1e-12:
            z = (resid - resid.mean()) / sd
            skew = float(np.mean(z ** 3))
            kurt = float(np.mean(z ** 4) - 3.0)
    min_n = _REG_NEG_RMSE_MIN_N if metric == "neg_rmse" else _REG_MIN_N
    in_envelope = bool(n >= min_n and abs(skew) <= _REG_SKEW_MAX and kurt <= _REG_EXCESS_KURT_MAX)
    bound_clears = bool(lower > theta)
    # DEFER (do not promote) outside the validated light-tailed/symmetric envelope: the bound is not a valid
    # >=0.95 lower bound there (measured under-coverage), so certifying would be an invalid certificate.
    certified = bool(bound_clears and in_envelope)
    deferred = bool(bound_clears and not in_envelope)
    if certified:
        reason = "bootstrap lower bound clears theta after multiplicity correction"
    elif deferred:
        reason = (f"DEFERRED: residuals outside the validated envelope (n={n}/min {min_n}, "
                  f"skew={round(skew, 3)}/max {_REG_SKEW_MAX}, excess_kurt={round(kurt, 3)}/max "
                  f"{_REG_EXCESS_KURT_MAX}); the bootstrap bound is not a valid 95% bound for "
                  f"skewed/heavy-tailed errors -- not certifying an invalid bound.")
    else:
        reason = "bootstrap lower confidence bound does not clear theta after paying for all peeks"
    return {"observed": round(float(point), 4), "n": int(n), "metric": metric,
            "theta": round(float(theta), 4), "checks": checks, "alpha_per_check": round(a, 6),
            "lower_bound": round(float(lower), 4), "certified": certified,
            "residual_skew": round(skew, 3), "residual_excess_kurtosis": round(kurt, 3),
            "in_envelope": in_envelope, "deferred": deferred, "reason": reason}


def _row_value(row, field):
    if field in row:
        return row.get(field)
    features = row.get("features")
    if isinstance(features, dict) and field in features:
        return features.get(field)
    return None


def _normalize_eval_slice(spec, default_metric, default_threshold):
    if isinstance(spec, str):
        return {
            "id": spec,
            "field": "slice",
            "value": spec,
            "metric": default_metric,
            "threshold": default_threshold,
            "min_n": 1,
        }
    if not isinstance(spec, dict):
        ident = str(spec)
        return {
            "id": ident,
            "field": "slice",
            "value": ident,
            "metric": default_metric,
            "threshold": default_threshold,
            "min_n": 1,
        }
    ident = str(spec.get("id") or spec.get("name") or spec.get("slice") or spec.get("value") or "slice")
    field = str(spec.get("field") or "slice")
    value = spec.get("value", spec.get("slice", spec.get("name", ident)))
    return {
        "id": ident,
        "field": field,
        "value": value,
        "metric": str(spec.get("metric") or default_metric),
        "threshold": float(spec.get("threshold", default_threshold)),
        "min_n": max(1, int(spec.get("min_n", spec.get("min_support", spec.get("minExamples", 1))))),
    }


# Additive finite-sample margin for the classification bootstrap lower bound (F8). A tail-probability
# shrink alone (the regression path's approach) PLATEAUS at ~0.93 one-sided coverage in the high-accuracy
# small-n regime, because the bootstrap of a [0,1] metric is boundary-compressed near 1.0 and the lower
# percentile is biased UP. Subtracting c/sqrt(n) from the percentile restores conservative coverage.
# MEASURED (balanced_accuracy, 400 trials/cell, B=400): with c=0.25 worst-cell one-sided coverage is
# 0.970 across true in {0.70,0.80,0.90} x n in {60,100,200,400}, vs 0.887 for the naive percentile
# (c=0.0). This makes the bound MORE conservative (never a false certification); see
# tests/test_certifier_coverage.py. The exact Clopper-Pearson ACCURACY path is unchanged.
_CLF_FS_MARGIN = 0.25


def _bootstrap_classification_metric_lower(metric, y_true, y_pred, labels, alpha=0.05, B=800, seed=0,
                                           fs_margin=_CLF_FS_MARGIN):
    n = len(y_true)
    point = score_metric(metric, y_true, y_pred, labels)
    if n <= 1:
        return round(float(point), 4), round(float(point), 4)
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        boots.append(score_metric(metric, yt[idx], yp[idx], labels))
    raw = float(np.percentile(boots, 100 * alpha))
    lower = max(0.0, raw - fs_margin / math.sqrt(n))      # finite-sample conservative margin (F8)
    return round(lower, 4), round(float(point), 4)


def evaluate_required_eval_slices(required_slices, rows, predictions, labels, default_metric,
                                  default_threshold, alpha=0.05, eval_slice_values=None):
    """Evaluate required locked-test slices as first-class certificate gates.

    Contract shape is intentionally small and source-agnostic:
      * "rare_class" means row["slice"] == "rare_class".
      * {"id": "rare", "field": "segment", "value": "rare", "metric": "macro_f1",
         "threshold": 0.82, "min_n": 30} selects either row[field] or row.features[field].

    Every declared slice must be present with enough support, and its confidence lower bound must clear
    its threshold. The caller has already scored the locked test once; this helper consumes only those
    predictions and cannot create a new test peek.
    """
    specs = [_normalize_eval_slice(s, default_metric, default_threshold) for s in (required_slices or [])]
    if not specs:
        return {"passed": True, "required": 0, "slices": []}

    y_true_all = [r.get("target") for r in rows]
    pred_all = list(predictions)
    sidecar = eval_slice_values if isinstance(eval_slice_values, list) else []
    reports = []
    for index, spec in enumerate(specs):
        mask = [
            i for i, row in enumerate(rows)
            if _eval_slice_value(row, sidecar[i] if i < len(sidecar) else None, spec["field"]) == spec["value"]
        ]
        yt = [y_true_all[i] for i in mask]
        yp = [pred_all[i] for i in mask]
        support = len(mask)
        if support < spec["min_n"]:
            reports.append({
                "id": spec["id"],
                "field": spec["field"],
                "value": spec["value"],
                "support": support,
                "min_n": spec["min_n"],
                "metric": spec["metric"],
                "observed": None,
                "lower_bound": None,
                "threshold": spec["threshold"],
                "passed": False,
                "reason": f"slice support {support} < min_n {spec['min_n']}",
            })
            continue
        observed = score_metric(spec["metric"], yt, yp, labels)
        if spec["metric"] == "accuracy":
            cert = certify_accuracy(observed, support, spec["threshold"], checks=1,
                                    alpha=alpha)
            lower = cert["lower_bound"]
            reason = cert["reason"]
        else:
            lower, _point = _bootstrap_classification_metric_lower(
                spec["metric"], yt, yp, labels, alpha=alpha, seed=17 + index)
            reason = ("slice lower bound clears threshold" if lower > spec["threshold"]
                      else "slice lower confidence bound does not clear threshold")
        passed = bool(lower > spec["threshold"])
        reports.append({
            "id": spec["id"],
            "field": spec["field"],
            "value": spec["value"],
            "support": support,
            "min_n": spec["min_n"],
            "metric": spec["metric"],
            "observed": round(float(observed), 4),
            "lower_bound": round(float(lower), 4),
            "threshold": round(float(spec["threshold"]), 4),
            "passed": passed,
            "reason": reason,
        })
    return {
        "passed": all(s["passed"] for s in reports),
        "required": len(reports),
        "slices": reports,
    }


def _eval_slice_value(row, sidecar, field):
    if isinstance(sidecar, dict) and field in sidecar:
        return sidecar.get(field)
    return _row_value(row, field)


def mean_ci(values, B=1500, seed=0):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.array([values[rng.integers(0, n, n)].mean() for _ in range(B)])
    return round(float(values.mean()), 4), [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


# =========================================================================== calibration
def expected_calibration_error(confidences, correct, bins=10):
    confidences = np.asarray(confidences); correct = np.asarray(correct, dtype=np.float64)
    n = len(confidences)
    if n == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidences > lo) & (confidences <= hi) if b else (confidences >= lo) & (confidences <= hi)
        if mask.sum():
            ece += (mask.sum() / n) * abs(float(confidences[mask].mean()) - float(correct[mask].mean()))
    return round(ece, 4)


# =========================================================================== leakage auditor
# Gate thresholds. The first two (raw MI, single-feature accuracy) catch a feature that strongly or
# exactly predicts the label. The third (NMI_FEAT_MAX) is the *normalized* mutual information relative to
# the feature's OWN entropy, U(L|F)=MI/H(F): the fraction of the feature's variation that is spent
# encoding the label. This is the load-bearing gate against a tuned label-DEPENDENT proxy -- a noisy copy
# of the label whose raw MI (and single-feature accuracy) can be deliberately tuned to sit just under the
# first two gates, but whose ENTIRE reason to exist is the label, so almost all of its entropy is
# label-entropy and MI/H(F) is large. Empirically (Adult/rugged, 7000 rows) every real predictive feature
# has MI/H(F) <= 0.084 (relationship 0.076, marital-status 0.084) while an 76-85% correlated label proxy
# has MI/H(F) in [0.155, 0.328]; an exact label copy has MI/H(F) = 1.0. A gate at 0.12 sits in that gap,
# so the proxy is caught WITHOUT flagging any genuine feature. This separation is a property of what a
# leak IS (a near-function of the label), not a number reverse-engineered from any target's answer.
MI_MAX, SF_ACC_MAX, JACCARD_DUP, DUP_FRAC_MAX = 0.30, 0.80, 0.90, 0.01
NMI_FEAT_MAX = 0.12
FUNCTIONAL_LEAK_MIN_N = 40
FUNCTIONAL_LEAK_ACC_MIN = 0.99
FUNCTIONAL_LEAK_NMI_MIN = 0.99
PROXY_LEAK_MIN_N = 500
PROXY_FEATURE_ENTROPY_MIN = 1.0
# A single-feature high-confidence proxy BLOCKS only when it is near-LOSSLESSLY recoverable on a seeded
# shuffled held-out half (the k=1 analogue of the joint gate's held-out-lossless test). A genuine proxy/leak
# recovers the label ~perfectly out-of-sample; a strong-but-noisy LEGITIMATE feature (e.g. a 90%-accurate
# predictor) does NOT, so it is no longer false-blocked. This replaces the raw mi/sf_acc/nmi-threshold proxy
# block, which over-blocked any feature with sf_acc>0.80 (a normal strong predictor) as "leakage" -- a real
# false-positive that would make the loop refuse to certify a legitimate dominant-feature tabular task.
PROXY_HOLDOUT_LOSSLESS = 0.99


def _entropy_bits(values):
    n = len(values)
    if not n:
        return 0.0
    c = Counter(values)
    return float(-sum((v / n) * math.log2(v / n) for v in c.values()))


def _mi_bits(values, labels):
    n = len(labels)
    if not n:
        return 0.0
    pv, pl, pvl = Counter(values), Counter(labels), Counter(zip(values, labels))
    mi = 0.0
    for (v, l), c in pvl.items():
        pxy, px, py = c / n, pv[v] / n, pl[l] / n
        if pxy > 0:
            mi += pxy * math.log2(pxy / (px * py))
    return float(mi)


def _nmi_feature(values, labels):
    """U(L|F) = I(F;L) / H(F): the coefficient of constraint of the label on the feature -- the fraction
    of the feature's own entropy that is about the label. ~1 for a (near-)copy of the label, small for a
    genuine feature whose variation is mostly NOT label. Returns 0 for a constant feature (H(F)=0)."""
    hf = _entropy_bits(values)
    return (_mi_bits(values, labels) / hf) if hf > 0 else 0.0


def _sf_holdout_recovery(values, labels, *, seed=0, frac=0.5):
    """Single-feature held-out lossless-recovery probe (the k=1 analogue of _joint_leak_verdict's held-out
    test). Seeded-shuffle the rows, fit a cell(value)->majority-label map on one half, score the other half.
    Returns the out-of-sample recovery accuracy in [0,1]. An exact/near-exact label proxy recovers ~1.0; a
    strong-but-noisy LEGITIMATE feature recovers only its true predictive accuracy (e.g. ~0.90). The shuffle
    defeats label-sorted row order; singleton-cell memorization does NOT generalize so it scores low."""
    n = len(labels)
    if n < 4:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    half = int(round(frac * n))
    tr_idx, te_idx = idx[:half], idx[half:]
    if not len(tr_idx) or not len(te_idx):
        return 0.0
    cell_lab = defaultdict(list)
    for i in tr_idx:
        cell_lab[values[i]].append(labels[i])
    tr_map = {cell: Counter(ls).most_common(1)[0][0] for cell, ls in cell_lab.items()}
    gmaj = Counter([labels[i] for i in tr_idx]).most_common(1)[0][0]
    correct = sum(1 for i in te_idx if tr_map.get(values[i], gmaj) == labels[i])
    return correct / len(te_idx)


def _feature_leak_verdict(values, labels, mi, acc, nmi):
    """Return (leaky, reason) for one feature.

    The first version of this live gate used `mi OR single_feature_acc OR nmi`, which is too blunt for
    real tabular data: small capped samples and low-entropy legitimate features can cross a normalized-MI
    threshold even when they are not target leakage. The hard block now requires either:

      * a functional label copy: a single feature recovers >=99% of labels AND almost all of that
        feature's entropy is the label (acc>=0.99 AND nmi>=0.99 -- a near-bijection); or
      * a high-confidence proxy: at n>=PROXY_LEAK_MIN_N, a strongish feature (sf_acc>SF_ACC_MAX) that is
        near-LOSSLESSLY recoverable on a seeded held-out half (recovery>=PROXY_HOLDOUT_LOSSLESS). This is the
        k=1 held-out-lossless discriminator -- it blocks an exact/near-exact proxy (incl. a high-cardinality
        one the functional gate's nmi>=0.99 misses) while NOT false-blocking a strong-but-noisy legitimate
        feature (a 90%-accurate predictor recovers ~0.90 < 0.99). It replaces the old raw mi/sf_acc/nmi-
        threshold proxy block, which over-blocked any feature with sf_acc>0.80 as leakage.

    AUDITOR-SELF-SUFFICIENCY (Cycle 1, moved into the auditor): the EXACT functional copy blocks at ANY n
    -- a feature that is an exact label copy is unambiguous at any row count, and the small-n
    false-positive case lives only on the WEAKER proxy/borderline path (e.g. a legitimate Adult/breast-
    cancer feature at n=11 with acc~1.0 but nmi~0.34, which fails the nmi>=0.99 conjunction and so does NOT
    trip this gate). Previously the any-n guarantee lived only in the loop wrapper
    (ar.autoresearch._leak_blocks); per the convergence thesis (one product CALLS one Python auditor), the
    auditor must enforce it itself so a direct caller (the TS product, a one-shot audit) gets the right
    verdict without re-implementing the loop policy. `FUNCTIONAL_LEAK_MIN_N` is retained for back-compat /
    the proxy-adjacent deferral but is intentionally NOT applied to an exact functional copy.
    """
    n = len(labels)
    entropy = _entropy_bits(values)
    functional = (
        acc >= FUNCTIONAL_LEAK_ACC_MIN
        and nmi >= FUNCTIONAL_LEAK_NMI_MIN
    )
    if functional:
        return True, "functional_label_leak"
    proxy = (
        n >= PROXY_LEAK_MIN_N
        and acc > SF_ACC_MAX                                   # cheap prefilter: only strongish features
        and _sf_holdout_recovery(values, labels) >= PROXY_HOLDOUT_LOSSLESS   # decisive: near-lossless out-of-sample
    )
    if proxy:
        return True, "high_confidence_label_proxy"
    return False, None


def _sf_acc(values, labels):
    buckets = defaultdict(list)
    for v, l in zip(values, labels):
        buckets[v].append(l)
    return sum(Counter(ls).most_common(1)[0][1] for ls in buckets.values()) / len(labels) if labels else 0.0


# --------------------------------------------------------------------------- JOINT (multi-feature) leak
# The per-feature gate above inspects ONE feature at a time, so it MISSES a joint encoding: label = (a XOR b)
# where a,b NOISELESSLY encode the label. Each feature alone is innocent (single_feature_acc ~0.65,
# nmi ~0.004), so no per-feature finding fires -- yet the PAIR is an exact, lossless function of the label.
# That is a genuine train/test leak (a certificate built on it is invalid) and the single-feature gate's
# blind spot.
#
# WHAT A JOINT LEAK *IS* (the property we test, not a number reverse-engineered from any answer): some small
# subset S of features (|S| >= 2) is a NOISELESS DETERMINISTIC FUNCTION of the label -- the discretized joint
# value of S recovers the label exactly AND that recovery GENERALIZES to held-out rows. We require BOTH of:
#
#   (1) EXACT in-sample determinism:  H(label | S) == 0  -- every joint S-cell is label-pure; and
#   (2) LOSSLESS held-out recovery:   a cell->majority-label map fit on one half of the rows recovers the
#       OTHER half's labels with accuracy == 1.0.
#
# WHY THE CONJUNCTION (measured, see experiments/joint_leak_{tournament,boundary,falsepos,conjunction}.py):
#   * (1) alone false-positives on SMALL n / HIGH-cardinality subsets: when binned S-cells are near-singletons
#     even RANDOM labels give H(label|S)=0 (each cell has one row). At n=60 random-noise+random-labels and a
#     genuine sign(x0) rule BOTH read H(label|S)=0. The held-out test (2) defeats this: a memorized singleton
#     map does NOT generalize (random-label holdout <= 0.70).
#   * (2) alone false-positives on a GENUINE high-signal deterministic-ish rule on CONTINUOUS inputs: at large
#     n sign(x0*x1) reaches holdout ~0.99, but binning the continuous boundary leaves H(label|S) > 0, so (1)
#     fails. Requiring (1) AND (2) fires ONLY on a noiseless functional encoding.
#   MEASURED separation (1200 rows, 5 seeds, k in {2,3}): noiseless a-XOR-b leak fires 5/5 (subset (a,b),
#   H(label|S)=0.0, holdout=1.0); a GENUINE noisy XOR (0.18 feature noise) fires 0/5 (closest H(label|S)=0.095,
#   holdout=0.977); the constructed capacity_bound G4 task (XOR, 0.05 feature noise, trees->~1.0) fires 0/5
#   (closest H(label|S)=0.116, holdout=0.973). Even 0.02 feature noise breaks BOTH conditions (H jumps from
#   exactly 0 to >=0.068, holdout drops to <=0.98). So this is a HARD block that does NOT regress G4 and does
#   NOT false-block a genuine tight-noise XOR task.
#
# |S| >= 2 is REQUIRED so this never duplicates or conflicts with the single-feature exact-copy gate
# (_feature_leak_verdict's functional_label_leak handles |S|=1). This GENERALIZES the auditor's existing
# discretize+MI machinery (_qbin, the cell-purity logic behind _sf_acc) to feature subsets; it reuses those
# primitives and weakens no existing gate. It is reported as its own finding and only blocks when the
# conjunction above fires.
JOINT_LEAK_MAX_K = 4            # search subsets of size 2..4 within the feature cap (k=1 is the single-feature
                               # gate's job). Cycle 3 raised this from 3 to close the verified k>=4 arity slip
                               # (a noiseless >=4-feature parity leak). The held-out lossless test guards FPs at
                               # every k; cost is C(<=max_features, k) (e.g. C(12,4)=495), sub-second.
JOINT_LEAK_MIN_N = 80          # need enough rows for a held-out determinism test to be RELIABLE. Cycle-2
                               # property fuzz (tests/test_properties.invariant_joint_leak_blocks) found the
                               # old floor of 60 was too low for k=3 MULTICLASS joint leaks (e.g. label = bit
                               # count -> 4 classes, or base-2 value -> 8 cells): at n in [60,80) a 50/50
                               # holdout split leaves some joint cell entirely out of the train half, so a
                               # GENUINE noiseless leak reads holdout ~0.87 < 1.0 and the gate returns an
                               # UNRELIABLE not-leaky. Measured: worst-case 8-cell encoding-k3 fires 20/20
                               # across seeds at n>=80 (0/20 reliability gap closes exactly at 80); raising the
                               # floor makes the gate DEFER honestly in [60,80) rather than emit a wrong verdict.
                               # This STRENGTHENS the gate (it never returns a less-confident answer); the loop
                               # side refuses to certify a tabular task while the gate is deferring (see
                               # ar.autoresearch _joint_gate_reliable / the certify preconditions).
JOINT_LEAK_MAX_FEATURES = 12   # cap the combinatorial search; if more features, take the highest-MI ones
JOINT_LEAK_HOLDOUT_FRAC = 0.5  # split rows in half: fit the cell->label map on one half, score the other
JOINT_LEAK_SUSPICIOUS_TOKENS = {
    "answer", "class", "copy", "future", "gold", "ground", "groundtruth", "hint", "holdout",
    "label", "leak", "proxy", "route", "routing", "split", "target", "test", "truth", "y"
}


def _joint_leak_verdict(cols, labels, *, max_k=JOINT_LEAK_MAX_K, min_n=JOINT_LEAK_MIN_N,
                        max_features=JOINT_LEAK_MAX_FEATURES, seed=0):
    """Detect a NOISELESS multi-feature (joint) label encoding the per-feature gate misses.

    `cols` is {feature_name: discretized_value_list} (already _qbin'd by the caller), `labels` the parallel
    label list. Returns (leaky: bool, detail: dict). Leaky iff some subset S with 2 <= |S| <= max_k satisfies
    BOTH (1) in-sample H(label|S) == 0 and (2) held-out lossless recovery == 1.0 (see the block comment for
    the measured justification of the conjunction). Below min_n we DEFER (return not-leaky) -- a held-out
    determinism test on too-few rows is unreliable, and the single-feature gate plus structural gates still
    apply; a re-audit once more rows are present catches a real joint leak.

    SEARCH BOUND (XOR-aware; the within-cap path is exhaustive, the wide-table path is a DOCUMENTED LIMIT):
    we ALWAYS scan every pair (k=2) over all features -- C(F,2) is cheap and a marginal-MI prefilter would be
    WRONG here, because the defining property of a joint XOR/parity leak is that each member has ~ZERO marginal
    MI with the label (so a marginal-MI ranking drops exactly the leak members; verified
    experiments/joint_leak_k4_probe.py). When F = len(keys) <= max_features we then FULL-enumerate every subset
    size k in 3..max_k -- no pruning -- because for a true k-parity NO proper subset and NO low-order statistic
    points to the members (parity-learning hardness), so only exhaustive search finds them, and the held-out
    lossless test (2) keeps exhaustive search false-positive-safe at every k (a genuine noisy/continuous rule
    never reaches holdout==1.0; verified experiments/joint_leak_k4_fix_prototype.py: noisy XOR/interaction/
    pure-noise all <1.0 at k<=5). When F > max_features we DECLINE k>=4 and bound k=3 to features in the most
    label-informative PAIRS -- this is XOR-UNSAFE for a true k>=3 parity among many features (no cheap criterion
    points to the members) and is reported as a known limit (`wide_table_high_arity_limit`); k=2 parity is still
    always caught. Cost within the cap is C(F,k) with F<=max_features (e.g. C(12,4)=495), sub-second."""
    keys = sorted(cols)
    n = len(labels)
    if n < min_n or len(keys) < 2:
        return False, {"reason": "deferred: too few rows or <2 features for a joint determinism test",
                       "n": n, "n_features": len(keys)}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    half = int(round(JOINT_LEAK_HOLDOUT_FRAC * n))
    tr_idx, te_idx = idx[:half], idx[half:]
    if not len(tr_idx) or not len(te_idx):
        return False, {"reason": "deferred: degenerate holdout split", "n": n}
    tr_labels = [labels[i] for i in tr_idx]
    gmaj = Counter(tr_labels).most_common(1)[0][0] if tr_labels else None
    closest = {"holdout": 0.0, "H_cond_bits": None, "subset": None, "in_sample_deterministic": None}

    def _eval(subset):
        """(in_sample_deterministic, holdout_lossless_acc, H_cond_bits) for one feature subset."""
        jv = list(zip(*[cols[c] for c in subset]))
        by = defaultdict(list)
        for v, l in zip(jv, labels):
            by[v].append(l)
        h_cond = 0.0
        for ls in by.values():
            c = Counter(ls)
            if len(c) > 1:
                m = len(ls)
                h_cond += (m / n) * (-sum((cnt / m) * math.log2(cnt / m) for cnt in c.values()))
        cell_lab = defaultdict(list)
        for i in tr_idx:
            cell_lab[jv[i]].append(labels[i])
        tr_map = {cell: Counter(ls).most_common(1)[0][0] for cell, ls in cell_lab.items()}
        correct = sum(1 for i in te_idx if tr_map.get(jv[i], gmaj) == labels[i])
        return (h_cond <= 0.0), correct / len(te_idx), h_cond

    import itertools as _it
    # ---- k=2: scan ALL pairs (no prefilter; marginal MI would drop XOR members); cache H(label|pair). ----
    pair_h = {}
    for subset in _it.combinations(keys, 2):
        det, holdout, h_cond = _eval(subset)
        pair_h[subset] = h_cond
        if holdout > closest["holdout"]:
            closest = {"holdout": round(holdout, 5), "H_cond_bits": round(h_cond, 6),
                       "subset": list(subset), "in_sample_deterministic": bool(det)}
        if det and holdout >= 1.0:
            return True, {"subset": list(subset), "k": 2, "H_cond_bits": round(h_cond, 6),
                          "holdout_lossless_acc": round(holdout, 5), "n": n}
    # ---- k in 3..max_k: within the feature cap, FULL-enumerate every k (only exhaustive search finds a
    # parity leak; the held-out lossless test guards FPs at every k). For a WIDE table (> max_features) bound
    # to k=3 over the most label-informative PAIRS -- a documented limit (XOR-unsafe for k>=3 among many
    # features); see the docstring. ----
    wide = len(keys) > max_features
    if max_k >= 3 and len(keys) >= 3:
        if not wide:
            ks, cand = range(3, max_k + 1), (lambda _k: keys)
        else:
            top_pairs = sorted(pair_h, key=lambda p: pair_h[p])[:max_features]
            k3_keys = sorted({f for p in top_pairs for f in p})
            ks, cand = [3], (lambda _k: k3_keys)
        for k in ks:
            ckeys = cand(k)
            if k > len(ckeys):
                continue
            for subset in _it.combinations(ckeys, k):
                det, holdout, h_cond = _eval(subset)
                if holdout > closest["holdout"]:
                    closest = {"holdout": round(holdout, 5), "H_cond_bits": round(h_cond, 6),
                               "subset": list(subset), "in_sample_deterministic": bool(det)}
                if det and holdout >= 1.0:
                    return True, {"subset": list(subset), "k": k, "H_cond_bits": round(h_cond, 6),
                                  "holdout_lossless_acc": round(holdout, 5), "n": n}
    return False, {"reason": "no subset is both exactly deterministic and losslessly recoverable",
                   "closest": closest, "n": n, "searched_k_up_to": (max_k if not wide else 3),
                   "wide_table_high_arity_limit": wide}


def _joint_leak_scope(detail):
    """Return the scope for a deterministic joint signal.

    A lossless joint encoding in user-supplied features is indistinguishable from target leakage using row
    values alone. Earlier builds hard-blocked only when the feature names looked suspicious, which let neutral
    name leaks such as ``a XOR b`` certify. The safer moat rule is name-independent: if the measured conjunction
    in _joint_leak_verdict fires (exact determinism plus shuffled held-out lossless recovery), certification
    stops. Legitimate representation-positive controls must be non-lossless/noisy or carry a future explicit
    provenance waiver outside the certifier path.
    """
    subset = detail.get("subset") if isinstance(detail, dict) else None
    if not isinstance(subset, list):
        return {"hard_block": False, "reason": "no deterministic subset recorded", "suspicious_tokens": []}
    suspicious = []
    for feature in subset:
        tokens = _WORD.findall(str(feature).replace("_", " ").replace(".", " ").lower())
        for token in tokens:
            if token in JOINT_LEAK_SUSPICIOUS_TOKENS:
                suspicious.append(token)
    return {
        "hard_block": True,
        "reason": "lossless_joint_label_encoding",
        "suspicious_tokens": sorted(set(suspicious)),
        "name_independent": True,
        "subset": subset
    }


def _qbin(values, bins=10):
    nums = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            nums.append(None)
    finite = np.array([x for x in nums if x is not None], dtype=np.float64)
    if not finite.size:
        return [str(v) for v in values]
    edges = np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1]) if len(set(finite.tolist())) > bins else np.unique(finite)
    return [f"bin{int(np.digitize([v], edges)[0])}" if v is not None else "nan" for v in nums]


def _is_numeric(values, frac=0.95):
    ok = sum(1 for v in values[:500] if _isfloat(v))
    return ok / max(min(len(values), 500), 1) >= frac


def _isfloat(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def _tokset(t):
    return set(_WORD.findall(str(t).lower()))


def _label_visible_in_text_or_id(label, text):
    """True only when the complete label appears as a token/phrase.

    This avoids false positives for ordinary integer labels: label 1 should not
    match ids like r1 or text like x10. Multi-token labels still match when the
    full phrase is present in order.
    """
    label_text = str(label).strip().lower()
    if not label_text or label_text == "none":
        return False
    label_tokens = _WORD.findall(label_text)
    if not label_tokens:
        return False
    text_tokens = _WORD.findall(str(text).lower())
    if len(label_tokens) == 1:
        return label_tokens[0] in text_tokens
    if len(label_tokens) > len(text_tokens):
        return False
    return any(text_tokens[i:i + len(label_tokens)] == label_tokens
               for i in range(0, len(text_tokens) - len(label_tokens) + 1))


def audit(train, test, *, text_key="text", target_key="target", min_test_n=200, allow_features=False):
    """Measurement-based leakage gates. Returns {passed, findings}. Catches feature/label MI leakage
    (numeric-binned), label-in-text/id, train/test near-duplicate overlap, and min held-out N."""
    findings = []
    labels_tr = [str(r.get(target_key)) for r in train]
    feat_keys = set()
    for r in train:
        feat_keys |= (set(r.keys()) - {"id", text_key, target_key})
        if isinstance(r.get("features"), dict):
            feat_keys |= {f"features.{k}" for k in r["features"]}
    feat_keys.discard("features")
    if feat_keys:
        binned_cols = {}   # feature_name -> discretized value list, reused for the joint (multi-feature) gate
        for key in sorted(feat_keys):
            bare = key.split(".", 1)[1] if key.startswith("features.") else key
            raw = ([r.get("features", {}).get(bare) for r in train] if key.startswith("features.")
                   else [r.get(key) for r in train])
            vals = _qbin(raw) if _is_numeric([str(x) for x in raw]) else [str(x) for x in raw]
            binned_cols[key] = vals
            mi, acc = _mi_bits(vals, labels_tr), _sf_acc(vals, labels_tr)
            nmi = _nmi_feature(vals, labels_tr)
            leaky, leak_reason = _feature_leak_verdict(vals, labels_tr, mi, acc, nmi)
            findings.append({"gate": "feature_target_leakage", "feature": key, "ok": not leaky,
                             "mi_bits": round(mi, 4), "single_feature_acc": round(acc, 4),
                             "nmi_feature": round(nmi, 4),
                             "feature_entropy_bits": round(_entropy_bits(vals), 4),
                             "leak_reason": leak_reason,
                             "policy": "functional-or-high-confidence-proxy"})
        # JOINT (multi-feature) leak gate: a NOISELESS subset encoding the per-feature gate above cannot see
        # (e.g. label = a XOR b). Fires only on the measured conjunction (in-sample H(label|S)==0 AND held-out
        # lossless recovery==1.0); see _joint_leak_verdict. It hard-blocks name-independently because a
        # lossless joint label encoding cannot be distinguished from target leakage by feature values alone.
        joint_leaky, joint_detail = _joint_leak_verdict(binned_cols, labels_tr)
        joint_scope = _joint_leak_scope(joint_detail) if joint_leaky else None
        joint_block = bool(joint_leaky and joint_scope and joint_scope.get("hard_block"))
        findings.append({"gate": "joint_feature_target_leakage", "ok": not joint_block,
                         "leak_reason": "noiseless_joint_label_encoding" if joint_block else None,
                         "detail": joint_detail,
                         "policy": "subset-exact-determinism-and-holdout-lossless name-independent hard block (k in 2..%d)" % JOINT_LEAK_MAX_K})
        if joint_leaky:
            findings.append({"gate": "joint_feature_target_leakage_scope", "ok": True,
                             "warning": not joint_block,
                             "hardBlock": joint_block,
                             "scope": joint_scope,
                             "detail": joint_detail,
                             "policy": "lossless deterministic joint interactions block unless future trusted provenance support is added outside the certifier path"})
        findings.append({"gate": "features_present_must_not_leak",
                         "ok": allow_features and all(f["ok"] for f in findings
                                                      if f["gate"] in ("feature_target_leakage",
                                                                       "joint_feature_target_leakage"))})
    else:
        findings.append({"gate": "no_nontext_features", "ok": True})

    label_vals = sorted(set(labels_tr + [str(r.get(target_key)) for r in test]))
    hits = sum(1 for r in train + test
               if any(_label_visible_in_text_or_id(lv, f"{r.get('id','')} {r.get(text_key,'')}")
                      for lv in label_vals if lv and lv != "None"))
    findings.append({"gate": "label_token_in_text_or_id", "ok": hits == 0, "hits": hits})

    test_tok = [_tokset(r.get(text_key, "")) for r in test]
    inv = defaultdict(list)
    for j, ts in enumerate(test_tok):
        for t in ts:
            inv[t].append(j)
    test_exact = {str(r.get(text_key, "")).strip().lower() for r in test}
    dup = exact = 0
    for r in train:
        if str(r.get(text_key, "")).strip().lower() in test_exact and r.get(text_key):
            exact += 1
        ts = _tokset(r.get(text_key, ""))
        if not ts:
            continue
        cand = set()
        for t in ts:
            cand.update(inv.get(t, ()))
        if any(len(ts & test_tok[j]) / len(ts | test_tok[j]) >= JACCARD_DUP for j in cand if (ts | test_tok[j])):
            dup += 1
    findings.append({"gate": "train_test_overlap", "ok": dup / max(len(test), 1) <= DUP_FRAC_MAX,
                     "near_dup": dup, "exact_dup": exact})
    findings.append({"gate": "min_test_n", "ok": len(test) >= min_test_n, "n": len(test), "threshold": min_test_n})
    return {"passed": all(f.get("ok", True) for f in findings), "findings": findings}


# =========================================================================== stratified leakage-safe split
def make_splits(rows, *, seed=0, test_frac=0.30, val_frac=0.20, text_key="text", target_key="target"):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rows:
        by[str(r.get(target_key))].append(r)
    train, val, test = [], [], []
    for items in by.values():
        idx = rng.permutation(len(items))
        nt = max(1, int(round(test_frac * len(items))))
        nv = max(1, int(round(val_frac * len(items))))
        test += [items[i] for i in idx[:nt]]
        val += [items[i] for i in idx[nt:nt + nv]]
        train += [items[i] for i in idx[nt + nv:]]

    def key(r):
        return r.get(text_key) or "|".join(f"{k}={v}" for k, v in sorted((r.get("features") or {}).items()))
    test_keys = {key(r) for r in test}
    test_tok = [_tokset(r.get(text_key, "")) for r in test]
    inv = defaultdict(list)
    for j, ts in enumerate(test_tok):
        for t in ts:
            inv[t].append(j)

    def leaks(r):
        if key(r) in test_keys:
            return True
        ts = _tokset(r.get(text_key, ""))
        if not ts:
            return False
        cand = set()
        for t in ts:
            cand.update(inv.get(t, ()))
        return any(len(ts & test_tok[j]) / len(ts | test_tok[j]) >= JACCARD_DUP for j in cand if (ts | test_tok[j]))

    before = len(train) + len(val)
    train = [r for r in train if not leaks(r)]
    val = [r for r in val if not leaks(r)]
    bal = lambda s: dict(Counter(str(r.get(target_key)) for r in s))
    return train, val, test, {"counts": {"train": len(train), "val": len(val), "test": len(test)},
                              "leakage_dropped": before - len(train) - len(val),
                              "balance": {"train": bal(train), "val": bal(val), "test": bal(test)}}


def digest(obj):
    return "sha256:" + hashlib.sha256(repr(obj).encode()).hexdigest()[:16]
