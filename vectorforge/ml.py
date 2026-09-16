"""The learner layer: featurizers, real model families, and the model-landscape scan.

This is the "doing" layer at small-classifier scale: it trains a landscape of genuine candidates
(model family x feature config), ranks them on validation, shortlists, and certifies the winner. Uses
real scikit-learn learners (gradient-boosted trees, random forest, SVM, naive Bayes, logistic) so a
correct diagnosis ("rebalance", "expand representation") actually produces a passing model.
"""

import itertools
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor

from . import science


# --------------------------------------------------------------------- featurization
def _is_float(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def expand_interactions(rows, max_pairs=4):
    if not rows or not isinstance(rows[0].get("features"), dict):
        return rows
    cats = [k for k, v in rows[0]["features"].items() if not _is_float(v)]
    pairs = list(itertools.combinations(cats, 2))[:max_pairs]
    out = []
    for r in rows:
        f = dict(r["features"])
        for a, b in pairs:
            f[f"{a}X{b}"] = f"{r['features'].get(a)}&{r['features'].get(b)}"
        out.append({**r, "features": f})
    return out


def _texts(rows):
    return [r.get("text", "") or "" for r in rows]


def _dicts(rows):
    out = []
    for r in rows:
        d = {}
        for k, v in (r.get("features") or {}).items():
            if _is_float(v):
                d[k] = float(v)
            else:
                d[f"{k}={v}"] = 1.0
        out.append(d)
    return out


# --------------------------------------------------------------------- candidate space
def candidates(kind):
    if kind == "text":
        clfs = {"logistic": lambda: LogisticRegression(max_iter=2000),
                "linear_svm": lambda: LinearSVC(),
                "complement_nb": lambda: ComplementNB(),
                "logistic_balanced": lambda: LogisticRegression(max_iter=2000, class_weight="balanced")}
        return [(f"{c}|ngram{ng[1]}", ctor, {"ngram": ng}) for (c, ctor), ng in itertools.product(clfs.items(), [(1, 1), (1, 2)])]
    clfs = {"hist_gbm": lambda: HistGradientBoostingClassifier(max_iter=200),
            "random_forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1),
            "logistic": lambda: LogisticRegression(max_iter=2000),
            "logistic_balanced": lambda: LogisticRegression(max_iter=2000, class_weight="balanced")}
    return [(f"{c}|{'inter' if it else 'base'}", ctor, {"interactions": it}) for (c, ctor), it in itertools.product(clfs.items(), [False, True])]


def build_pipeline(kind, ctor, cfg):
    if kind == "text":
        return make_pipeline(TfidfVectorizer(ngram_range=cfg["ngram"], min_df=1), ctor())
    return make_pipeline(DictVectorizer(sparse=False), MaxAbsScaler(), ctor())


def _Xy(kind, rows, cfg, target_key="target"):
    if kind == "tabular" and cfg.get("interactions"):
        rows = expand_interactions(rows)
    X = _texts(rows) if kind == "text" else _dicts(rows)
    return X, [str(r.get(target_key)) for r in rows]


# --------------------------------------------------------------------- REGRESSION candidate space
# A REGRESSOR landscape parallel to candidates(): real scikit-learn regressors over the SAME DictVectorizer +
# MaxAbsScaler featurizer the tabular classifiers use, so a regression task gets a genuine model scan, not a
# single hard-coded model. The families are the conventional regression workhorses (linear/ridge, ensembles,
# MLP); nothing here is anchored to any dataset. NaN/Inf targets are not produced here (the caller coerces
# numeric targets to finite floats); a fit that raises is handled by the caller exactly like the classifier
# path. Text regression is rare in this stack but supported via a TfidfVectorizer + Ridge head for symmetry.
def regression_candidates(kind):
    if kind == "text":
        regs = {"ridge": lambda: Ridge(alpha=1.0),
                "linear_regression": lambda: LinearRegression()}
        return [(f"{c}|ngram{ng[1]}", ctor, {"ngram": ng})
                for (c, ctor), ng in itertools.product(regs.items(), [(1, 1), (1, 2)])]
    regs = {"hist_gbm_regressor": lambda: HistGradientBoostingRegressor(max_iter=300),
            "random_forest_regressor": lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                                                     random_state=0),
            "ridge": lambda: Ridge(alpha=1.0),
            "linear_regression": lambda: LinearRegression(),
            "mlp_regressor": lambda: MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3,
                                                  max_iter=800, random_state=0)}
    return [(f"{c}|{'inter' if it else 'base'}", ctor, {"interactions": it})
            for (c, ctor), it in itertools.product(regs.items(), [False, True])]


def _Xy_reg(kind, rows, cfg, target_key="target"):
    """Regression featurization: SAME X as classification, but y is a FLOAT vector (continuous target).

    A target that fails to parse as a float maps to NaN; the caller drops/penalizes those rows. This is the
    only difference from _Xy: the target is kept numeric instead of stringified."""
    if kind == "tabular" and cfg.get("interactions"):
        rows = expand_interactions(rows)
    X = _texts(rows) if kind == "text" else _dicts(rows)
    y = []
    for r in rows:
        try:
            y.append(float(r.get(target_key)))
        except (TypeError, ValueError):
            y.append(float("nan"))
    return X, y


def latency_ms(pipe, X, k=120):
    t = []
    for x in X[:k]:
        t0 = time.perf_counter(); pipe.predict([x]); t.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(t, 95)) if t else 0.0


# --------------------------------------------------------------------- the landscape scan
def landscape_scan(kind, train, val, test, labels, metric, *, small_scale=2500, shortlist_k=3):
    """Scan candidates -> cheap validation ranking -> shortlist -> full train -> winner.
    Returns {leaderboard, shortlist, winner, winner_pipe, winner_cfg, val, pareto}. Test is NOT touched
    here (the caller certifies the winner once)."""
    cands = candidates(kind)
    rng = np.random.default_rng(0)
    small = train if len(train) <= small_scale else [train[i] for i in rng.permutation(len(train))[:small_scale]]
    board = []
    for label, ctor, cfg in cands:
        try:
            Xtr, ytr = _Xy(kind, small, cfg)
            Xva, yva = _Xy(kind, val, cfg)
            pipe = build_pipeline(kind, ctor, cfg); pipe.fit(Xtr, ytr)
            v = science.score_metric(metric, yva, pipe.predict(Xva), labels)
            board.append({"candidate": label, "val": round(v, 4), "latency_ms": round(latency_ms(pipe, Xva), 3), "ctor": ctor, "cfg": cfg})
        except Exception as e:  # noqa: BLE001
            board.append({"candidate": label, "val": -1.0, "error": str(e)[:80]})
    board.sort(key=lambda c: -c["val"])
    shortlist = [c for c in board if c["val"] >= 0][:shortlist_k]
    best = None
    for c in shortlist:
        Xtr, ytr = _Xy(kind, train, c["cfg"])
        Xva, yva = _Xy(kind, val, c["cfg"])
        pipe = build_pipeline(kind, c["ctor"], c["cfg"]); pipe.fit(Xtr, ytr)
        v = science.score_metric(metric, yva, pipe.predict(Xva), labels)
        if best is None or v > best["val"]:
            best = {"candidate": c["candidate"], "cfg": c["cfg"], "val": round(v, 4), "pipe": pipe}
    fr = sorted([c for c in board if c["val"] >= 0], key=lambda c: (-c["val"], c["latency_ms"]))
    pareto, bl = [], float("inf")
    for c in fr:
        if c["latency_ms"] < bl:
            pareto.append({"candidate": c["candidate"], "val": c["val"], "latency_ms": c["latency_ms"]}); bl = c["latency_ms"]
    return {"leaderboard": [{k: c[k] for k in ("candidate", "val", "latency_ms") if k in c} for c in board],
            "shortlist": [c["candidate"] for c in shortlist], "winner": best["candidate"],
            "winner_pipe": best["pipe"], "winner_cfg": best["cfg"], "val": best["val"], "pareto": pareto}


def evaluate(pipe, kind, rows, cfg, labels, metric):
    X, y = _Xy(kind, rows, cfg)
    pred = list(pipe.predict(X))
    return {"metric": metric, "value": round(science.score_metric(metric, y, pred, labels), 4),
            "accuracy": round(science.accuracy(y, pred), 4),
            "per_class": science.per_class_recall(y, pred, labels),
            "predictions": pred, "y_true": y, "latency_ms_p95": round(latency_ms(pipe, X, k=200), 3)}
