"""The OPERATIONALIZATION ADVERSARY -- an intake-time, SPEC-aware adversarial screen -- NEW module.

`admissibility.inspect` is a *structural* gate (constant target, exact-identity leakage, too few rows). It
never sees the SPEC, so it cannot tell whether clearing the threshold is even *meaningful*. This module is the
missing complement: it takes the full spec (metric, threshold theta, and the FORBIDDEN/dropped fields) and
adversarially asks "could theta be hit trivially or gamed?" -- BEFORE the loop spends its one sealed peek.

Three checks, all DETERMINISTIC and all reusing the FROZEN metric definitions (`vectorforge.science`):

  1. TRIVIAL THRESHOLD (severity=decline). The trivial baseline = the score of the metric-optimal CONSTANT
     predictor (majority class for accuracy/balanced_accuracy/macro_f1; the mean for r2/neg_rmse; the median
     for neg_mae). If theta <= that baseline, a constant predictor already "certifies": the goal is degenerate
     and any certificate would be vacuous. This is the only finding that blocks -- and blocking is *more*
     correct, not less (certifying a constant predictor is a bug, not a success).

  2. SINGLE-FEATURE SUFFICIENCY (severity=warn). A cheap depth-2 stump on ONE feature, scored on a holdout
     with the frozen metric. If a single non-forbidden feature alone reaches theta, the goal is likely gameable
     by one proxy / probable target leakage -- the certificate may be real but uninformative. Reported, not
     blocked (one feature is sometimes legitimately predictive).

  3. FORBIDDEN-FIELD SIGNAL (severity=warn). The same probe restricted to the dropped/forbidden columns. If a
     forbidden field alone reaches theta, it confirms the field was correctly excluded as leakage AND flags
     that a surviving feature could be a proxy for it -- worth a human glance.

CONTRACT (matches the project invariants):
  * NON-BINDING and SOFT-EDGE: no estimator is promoted, no sealed peek is taken, no certificate is written.
    The stumps here are throwaway *probes*, explicitly NOT the model under certification.
  * The frozen certifier is untouched and is the only thing that can promote. This screen can only DECLINE a
    degenerate spec or WARN; it can never cause a certificate to be issued.
  * Deterministic given a seed (clock-free). The adversary takes the worst case over a couple of fixed splits.
"""
from __future__ import annotations

import numpy as np

from vectorforge.science import score_metric

# Metrics whose trivial-optimal constant is the MEDIAN (neg_mae) vs the MEAN (r2, neg_rmse).
_REG_METRICS = ("r2", "neg_rmse", "neg_mae")
_PROBE_SEEDS = (0, 1)          # adversary = worst (most gameable) case over a few fixed splits; clock-free


def _is_reg(task_type, metric):
    return task_type == "regression" or metric in _REG_METRICS


def _feature_cols(rows, target_key):
    cols = []
    seen = set()
    for r in rows:
        src = r.get("features", r) if isinstance(r.get("features"), dict) else r
        for k in src:
            if k != target_key and k != "features" and k not in seen:
                seen.add(k)
                cols.append(k)
    return cols


def _raw(r, target_key):
    """The flat attribute dict for a row, whether nested under 'features' or flat."""
    f = r.get("features") if isinstance(r, dict) else None
    return f if isinstance(f, dict) else r


def _targets(rows, target_key):
    return [(_raw(r, target_key).get(target_key) if not isinstance(r.get(target_key, None), (int, float, str))
             else r.get(target_key)) if isinstance(r, dict) else None for r in rows]


def _column(rows, col, target_key):
    return [_raw(r, target_key).get(col) for r in rows]


def _encode_feature(values):
    """Encode one raw feature column into a float vector + a finite mask. Numeric stays numeric (median
    imputed); non-numeric is label-encoded (missing -> its own code). Returns (x[n,1], ok) where ok=False
    means the column is unusable (all-missing or constant after encoding)."""
    n = len(values)
    nums, is_num = [], True
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            nums.append(np.nan)
        else:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                is_num = False
                break
    if is_num:
        arr = np.array(nums, dtype=float)
        if np.all(np.isnan(arr)):
            return None, False
        med = float(np.nanmedian(arr))
        arr = np.where(np.isnan(arr), med, arr)
    else:
        codes, mapping = [], {}
        for v in values:
            key = "\x00NA" if v is None else str(v)
            if key not in mapping:
                mapping[key] = len(mapping)
            codes.append(mapping[key])
        arr = np.array(codes, dtype=float)
    if np.unique(arr).size < 2:
        return None, False                       # constant column: a stump can't use it
    return arr.reshape(-1, 1), True


def _trivial_baseline(y, task_type, metric, labels):
    """Score of the metric-optimal CONSTANT predictor, via the FROZEN scorer (single source of truth)."""
    if _is_reg(task_type, metric):
        ys = np.array([float(v) for v in y], dtype=float)
        const = float(np.median(ys)) if metric == "neg_mae" else float(np.mean(ys))
        from vectorforge.science import score_regression_metric
        return float(score_regression_metric(metric, list(ys), [const] * len(ys)))
    # classification: the majority class is optimal for accuracy; score_metric computes the right value
    # for balanced_accuracy / macro_f1 from a constant-prediction vector too.
    vals, counts = np.unique(np.array([str(v) for v in y]), return_counts=True)
    maj = vals[int(np.argmax(counts))]
    y_str = [str(v) for v in y]
    return float(score_metric(metric, y_str, [maj] * len(y_str), labels or sorted(set(y_str))))


def _single_feature_score(x, y, task_type, metric, labels, seed):
    """Worst-case (max over fixed splits) holdout score of a depth-2 stump on ONE feature, frozen-metric
    scored. A throwaway probe -- never promoted, never certified."""
    from sklearn.model_selection import train_test_split
    reg = _is_reg(task_type, metric)
    best = -np.inf
    for s in _PROBE_SEEDS:
        try:
            strat = None if reg else (np.array([str(v) for v in y]))
            xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=s,
                                                  stratify=strat if (strat is not None
                                                  and min(np.unique(strat, return_counts=True)[1]) >= 2) else None)
            if reg:
                from sklearn.tree import DecisionTreeRegressor
                from vectorforge.science import score_regression_metric
                m = DecisionTreeRegressor(max_depth=2, random_state=seed).fit(xtr, [float(v) for v in ytr])
                pred = m.predict(xte)
                sc = float(score_regression_metric(metric, [float(v) for v in yte], list(pred)))
            else:
                from sklearn.tree import DecisionTreeClassifier
                ytr_s, yte_s = [str(v) for v in ytr], [str(v) for v in yte]
                m = DecisionTreeClassifier(max_depth=2, random_state=seed).fit(xtr, ytr_s)
                pred = [str(v) for v in m.predict(xte)]
                sc = float(score_metric(metric, yte_s, pred, labels or sorted(set(ytr_s) | set(yte_s))))
            best = max(best, sc)
        except Exception:  # noqa: BLE001  a probe that fails to fit simply yields no signal for that column
            continue
    return None if best == -np.inf else best


def screen(records, *, target_key="target", task_type="binary", metric="accuracy", threshold=0.75,
           drop_cols=None, labels=None, seed=0, max_features_probed=60):
    """Adversarially screen the SPEC against the DATA. Non-binding, deterministic, no sealed peek.

    Returns a dict:
        verdict            -- 'clean' | 'warn' | 'decline'
        trivial_baseline   -- score of the metric-optimal constant predictor (the floor theta must clear)
        threshold          -- echoed
        findings           -- [{check, severity, detail, ...}]
        single_feature_best-- {'feature','score'} or None  (strongest single-feature probe, non-forbidden)
    A 'decline' means the spec is degenerate (theta at/below the trivial baseline); the caller should refuse
    to run rather than emit a vacuous certificate. 'warn'/'clean' always proceed.
    """
    rows = list(records or [])
    findings = []
    out = {"verdict": "clean", "trivial_baseline": None, "threshold": float(threshold),
           "findings": findings, "single_feature_best": None}
    if len(rows) < 4:
        return out                                 # admissibility owns the "too few rows" hard decline
    drop = set(drop_cols or ())

    y = _targets(rows, target_key)
    if any(v is None for v in y):
        return out                                 # malformed target -> admissibility / loop handles it

    # ---- Check 1: trivial threshold ------------------------------------------------------------------
    try:
        base = _trivial_baseline(y, task_type, metric, labels)
        out["trivial_baseline"] = round(base, 6)
        if float(threshold) <= base + 1e-12:
            findings.append({
                "check": "trivial_threshold", "severity": "decline",
                "detail": (f"threshold {threshold} is at/below the trivial baseline {round(base, 4)} for "
                           f"metric {metric!r}: a constant predictor already clears it, so any certificate "
                           f"would be vacuous. Raise theta above {round(base, 4)}."),
                "baseline": round(base, 6)})
    except Exception:  # noqa: BLE001  baseline computation must never break intake
        pass

    # ---- Checks 2 & 3: single-feature / forbidden-field sufficiency ----------------------------------
    cols = _feature_cols(rows, target_key)
    probed, best_nf = 0, None
    for c in cols:
        if probed >= max_features_probed:
            break
        x, ok = _encode_feature(_column(rows, c, target_key))
        if not ok:
            continue
        probed += 1
        sc = _single_feature_score(x, y, task_type, metric, labels, seed)
        if sc is None:
            continue
        forbidden = c in drop
        if not forbidden and (best_nf is None or sc > best_nf[1]):
            best_nf = (c, sc)
        if sc >= float(threshold) - 1e-12:
            if forbidden:
                findings.append({
                    "check": "forbidden_field_signal", "severity": "warn", "feature": c, "score": round(sc, 4),
                    "detail": (f"forbidden/dropped field {c!r} ALONE reaches {round(sc, 4)} >= theta {threshold}: "
                               f"it is correctly excluded as leakage -- verify no surviving feature is a proxy.")})
            else:
                findings.append({
                    "check": "single_feature_sufficiency", "severity": "warn", "feature": c, "score": round(sc, 4),
                    "detail": (f"feature {c!r} ALONE reaches {round(sc, 4)} >= theta {threshold}: the goal may be "
                               f"gameable by a single proxy / probable target leakage. Confirm {c!r} is a "
                               f"legitimate predictor before trusting the certificate.")})
    if best_nf is not None:
        out["single_feature_best"] = {"feature": best_nf[0], "score": round(best_nf[1], 4)}

    sev = {f["severity"] for f in findings}
    out["verdict"] = "decline" if "decline" in sev else ("warn" if "warn" in sev else "clean")
    return out


def _selftest():
    rng = np.random.RandomState(0)
    n = 300
    # (a) trivial threshold: 80% class A; accuracy theta below 0.80 must DECLINE.
    y = ["A"] * 240 + ["B"] * 60
    recs = [{"features": {"f": float(rng.randn())}, "target": t} for t in y]
    r = screen(recs, task_type="binary", metric="accuracy", threshold=0.75)
    assert r["verdict"] == "decline" and abs(r["trivial_baseline"] - 0.8) < 1e-6, r
    assert any(f["check"] == "trivial_threshold" for f in r["findings"]), r

    # (b) theta above the trivial baseline: no trivial-threshold decline.
    r2 = screen(recs, task_type="binary", metric="accuracy", threshold=0.95)
    assert not any(f["check"] == "trivial_threshold" for f in r2["findings"]), r2

    # (c) single-feature leak: a feature equal to the (balanced) label clears any theta -> warn.
    yb = (["A"] * 150) + (["B"] * 150)
    leak = [{"features": {"leak": (1.0 if t == "B" else 0.0) + 0.01 * rng.randn(), "noise": float(rng.randn())},
             "target": t} for t in yb]
    r3 = screen(leak, task_type="binary", metric="accuracy", threshold=0.9)
    assert r3["verdict"] in ("warn", "decline"), r3
    assert any(f["check"] == "single_feature_sufficiency" and f["feature"] == "leak" for f in r3["findings"]), r3

    # (d) forbidden field carrying the signal -> warn, distinct check.
    r4 = screen(leak, task_type="binary", metric="accuracy", threshold=0.9, drop_cols=["leak"])
    assert any(f["check"] == "forbidden_field_signal" and f["feature"] == "leak" for f in r4["findings"]), r4
    assert not any(f["check"] == "single_feature_sufficiency" and f["feature"] == "leak"
                   for f in r4["findings"]), r4

    # (e) regression r2: constant (mean) predictor scores 0, so theta<=0 is trivial.
    yr = [float(v) for v in rng.randn(200)]
    rr = [{"features": {"x": float(rng.randn())}, "target": t} for t in yr]
    r5 = screen(rr, task_type="regression", metric="r2", threshold=0.0)
    assert r5["verdict"] == "decline" and any(f["check"] == "trivial_threshold" for f in r5["findings"]), r5
    r6 = screen(rr, task_type="regression", metric="r2", threshold=0.5)
    assert not any(f["check"] == "trivial_threshold" for f in r6["findings"]), r6

    print("intake_adversary self-test: PASS",
          {"trivial_baseline_a": r["trivial_baseline"], "leak_best": r3["single_feature_best"]})


if __name__ == "__main__":
    _selftest()
