"""Learning-to-rank vertical (SELF-CONTAINED). The ontology already DECLARES `ranking` as a task type
(vectorforge/llm_shell/_frozen.py:KNOWN_TASK_TYPES, schema.py) but harness.is_supported() returns False for
it -- the audit flagged ranking as the priority missing modality. This module builds the missing path
WITHOUT touching the frozen science.py or sealed.py; the lead wires it into frontdoor/harness (see the
returned wiring edits) so a kind="ranking" goal stops being declined.

WHY A SEPARATE MODULE (the load-bearing reason). Ranking data is GROUPED: a dataset is a list of QUERIES,
each query a list of items with relevance labels, and the metrics (NDCG@k, MAP@k) are computed PER QUERY then
averaged. The items WITHIN a query are NOT exchangeable with items in other queries -- only whole queries are
the i.i.d. unit. So the frozen certifier's resampling unit MUST be the QUERY, not the row. science.py's
bootstrap_lower / _bootstrap_classification_metric_lower / certify_regression all resample ROWS (or
(y,pred) pairs), which is the wrong unit here: it would treat the ~n_items*n_queries items as independent and
massively understate the variance, producing an over-optimistic (anti-conservative, invalid) lower bound.
This module therefore builds a per-QUERY bootstrap that mirrors science.py's discipline exactly:

  * one-sided (1-alpha) LOWER confidence bound (promote ON THE BOUND, never on the point estimate);
  * Bonferroni alpha/checks for the realized number of sealed-test peeks (single-peek discipline);
  * a documented, MONOTONE, conservative finite-sample shrink (tighter effective tail probability), chosen
    by MEASURED coverage (tests/test_ranking.py), NOT reverse-engineered from any target's numbers -- the
    per-query mean of a bounded [0,1] metric is boundary-compressed and median-biased exactly like the
    classification case in science.py, so the naive percentile under-covers and the shrink restores it;
  * an HONEST DEFER when there are too few queries to bound (the grouped analogue of science.py's n<=1
    short-circuit): a bootstrap over <MIN_BOOTSTRAP_QUERIES queries cannot give a trustworthy tail, so we
    refuse rather than emit a confident-looking invalid bound.

Pure numpy + scikit-learn (Ridge / HistGradientBoostingRegressor), no new deps. Every metric is exact and
pinned against a hand-worked example in the test file.
"""
import math
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


# =========================================================================== data model
# A dataset is a list of queries. Each query is {"qid": <hashable>, "items": [{"features": {...},
# "relevance": <int>}, ...]}. The featurizer below builds a per-item numeric matrix from the item feature
# dicts (the same nested `features` contract the tabular harness uses), and a per-query grouping carries the
# item->query membership the metrics and the per-query bootstrap both need.
def _query_items(query):
    return list(query.get("items") or [])


def _query_relevances(query):
    return [int(it.get("relevance", 0)) for it in _query_items(query)]


def validate_dataset(queries):
    """Fail-closed structural check. A ranking dataset must be a non-empty list of queries, each with a
    non-empty item list and an integer relevance per item. Returns the list unchanged or raises ValueError
    (we refuse to silently score a malformed dataset, mirroring harness.UnsupportedSpec fail-closed)."""
    if not isinstance(queries, list) or not queries:
        raise ValueError("ranking dataset must be a non-empty list of queries")
    for qi, q in enumerate(queries):
        items = _query_items(q)
        if not items:
            raise ValueError(f"query {qi} (qid={q.get('qid')!r}) has no items")
        for ii, it in enumerate(items):
            if "relevance" not in it:
                raise ValueError(f"query {qi} item {ii} has no 'relevance' label")
            try:
                int(it["relevance"])
            except (TypeError, ValueError):
                raise ValueError(f"query {qi} item {ii} relevance {it['relevance']!r} is not an int")
    return queries


class RankingFeaturizer:
    """Per-ITEM numeric featurizer over item `features` dicts: numeric columns standardized (zero-mean /
    unit-var, stats fit on the full item union, mirroring models.TabularFeaturizer) and categorical columns
    one-hot over the union vocab. Fit on the union of all items across all (train) queries so the schema is
    stable; transform a query's items to a (n_items, d) matrix whose row order matches the query's item order
    (the metrics rely on that order to map predicted scores back to relevance labels)."""

    def __init__(self):
        self.numeric, self.cats, self.vocab = [], [], {}
        self.mu, self.sd = {}, {}

    @staticmethod
    def _f(item, c):
        return (item.get("features") or {}).get(c)

    def _all_items(self, queries):
        return [it for q in queries for it in _query_items(q)]

    def fit(self, queries):
        items = self._all_items(queries)
        if not items:
            return self
        cols = list((items[0].get("features") or {}).keys())
        self.numeric = [c for c in cols
                        if all(isinstance(self._f(it, c), (int, float)) for it in items
                               if self._f(it, c) is not None)]
        self.cats = [c for c in cols if c not in self.numeric]
        self.vocab = {c: sorted({str(self._f(it, c)) for it in items}) for c in self.cats}
        for c in self.numeric:
            vals = [float(self._f(it, c)) for it in items if isinstance(self._f(it, c), (int, float))]
            arr = np.asarray(vals, dtype=float) if vals else np.zeros(1)
            self.mu[c] = float(arr.mean())
            sd = float(arr.std())
            self.sd[c] = sd if sd > 1e-9 else 1.0          # guard constant columns
        return self

    def transform_items(self, items):
        X = []
        for it in items:
            row = []
            for c in self.numeric:
                v = self._f(it, c)
                v = float(v) if isinstance(v, (int, float)) else self.mu.get(c, 0.0)   # impute mean
                row.append((v - self.mu.get(c, 0.0)) / self.sd.get(c, 1.0))
            for c in self.cats:
                row += [1.0 if str(self._f(it, c)) == lev else 0.0 for lev in self.vocab[c]]
            X.append(row)
        return np.asarray(X, dtype=float) if X else np.zeros((0, 0), dtype=float)

    def stack(self, queries):
        """Flatten all (train) queries into a single (N_items, d) design matrix + parallel relevance vector
        for the pointwise regressor. Returns (X, y)."""
        Xs, ys = [], []
        for q in queries:
            items = _query_items(q)
            Xs.append(self.transform_items(items))
            ys.extend(_query_relevances(q))
        X = np.vstack([x for x in Xs if x.size]) if any(x.size for x in Xs) else np.zeros((0, 0))
        return X, np.asarray(ys, dtype=float)


# =========================================================================== metrics (exact, per query)
# NDCG@k with the STANDARD exponential gain g(rel)=2^rel - 1 and log2 position discount, normalized by the
# IDEAL DCG (the DCG of the relevance-sorted ranking). MAP@k is the mean over relevant items (capped at k) of
# precision@(rank of that item). Both are computed PER QUERY from a score-induced ranking, then AVERAGED over
# queries. A query with no positive relevance has IDCG=0 / no relevant items; by the standard convention its
# NDCG and AP are defined to be 0.0 (it contributes a 0, it is not dropped) -- this is documented so the
# averaged metric is well-defined and the certifier bounds the SAME quantity the menu is ranked on.
def _order_by_scores(scores, seed=0):
    """Return item indices sorted by DESCENDING score. Ties are broken by a DETERMINISTIC seeded permutation
    of the original positions (not by original order), so a model that outputs constant/identical scores
    cannot get a spuriously good (or bad) order from incidental input ordering -- the tie order is random but
    reproducible. This matters for the popularity/constant baselines whose scores are often tied."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    rng = np.random.default_rng(seed)
    tiebreak = rng.permutation(n)
    # lexsort: primary key LAST -> sort by (-score) primary, tiebreak secondary, both ascending
    order = np.lexsort((tiebreak, -scores))
    return order


def dcg_at_k(rels_in_order, k):
    """DCG@k = sum_{i=1..k} (2^rel_i - 1) / log2(i + 1) for the items in the GIVEN order."""
    rels = list(rels_in_order)[:k]
    return float(sum((2.0 ** r - 1.0) / math.log2(i + 2) for i, r in enumerate(rels)))


def ndcg_at_k_for_query(relevances, scores, k, seed=0):
    """NDCG@k for ONE query: order items by predicted score, take DCG@k, divide by the ideal DCG@k (items
    sorted by true relevance descending). Returns 0.0 when the ideal DCG is 0 (no positive relevance)."""
    relevances = list(relevances)
    if not relevances:
        return 0.0
    order = _order_by_scores(scores, seed=seed)
    ranked_rels = [relevances[i] for i in order]
    dcg = dcg_at_k(ranked_rels, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return (dcg / ideal) if ideal > 0 else 0.0


def ap_at_k_for_query(relevances, scores, k, seed=0):
    """Average Precision@k for ONE query, BINARY-relevance convention (rel > 0 counts as relevant). For each
    relevant item appearing in the top-k, accumulate precision@(its rank); divide by min(#relevant, k). This
    is the standard MAP building block. Returns 0.0 when the query has no relevant item."""
    relevances = list(relevances)
    n_rel = sum(1 for r in relevances if r > 0)
    if n_rel == 0:
        return 0.0
    order = _order_by_scores(scores, seed=seed)
    ranked_rels = [relevances[i] for i in order][:k]
    hits = 0
    precisions = []
    for rank, r in enumerate(ranked_rels, start=1):
        if r > 0:
            hits += 1
            precisions.append(hits / rank)
    denom = min(n_rel, k)
    return float(sum(precisions) / denom) if denom > 0 else 0.0


def parse_metric(metric):
    """'ndcg@10' / 'map@5' -> ('ndcg', 10). A bare 'ndcg'/'map' uses the caller-supplied k. Refuses anything
    else fail-closed so a ranking goal never silently scores the wrong metric (the moat rule from sealed.py:
    no metric fall-through)."""
    m = str(metric).strip().lower()
    name, _, ktxt = m.partition("@")
    if name not in ("ndcg", "map"):
        raise ValueError(f"ranking metric {metric!r} not supported; use 'ndcg@k' or 'map@k'")
    k = int(ktxt) if ktxt else None
    return name, k


def per_query_metric(queries, score_fn, metric="ndcg@10", k=None, seed=0):
    """Vector of PER-QUERY metric values (one float per query, in query order). `score_fn(query)->scores`
    returns one predicted score per item in the query's item order. This is the unit the per-query bootstrap
    resamples. The seed feeds the deterministic tie-break so identical-score baselines are reproducible."""
    name, kk = parse_metric(metric)
    kk = kk if kk is not None else k
    if kk is None:
        raise ValueError("k must be supplied either in the metric ('ndcg@10') or via k=")
    out = []
    for qi, q in enumerate(queries):
        rels = _query_relevances(q)
        scores = score_fn(q)
        s = seed + qi                                   # distinct, reproducible tie-break per query
        if name == "ndcg":
            out.append(ndcg_at_k_for_query(rels, scores, kk, seed=s))
        else:
            out.append(ap_at_k_for_query(rels, scores, kk, seed=s))
    return np.asarray(out, dtype=float)


def mean_metric(queries, score_fn, metric="ndcg@10", k=None, seed=0):
    """Mean over queries of the per-query metric (the scalar the certifier lower-bounds)."""
    vals = per_query_metric(queries, score_fn, metric=metric, k=k, seed=seed)
    return float(vals.mean()) if vals.size else 0.0


# =========================================================================== models (honest small menu)
class PointwiseRanker:
    """Pointwise learning-to-rank: fit a REGRESSOR on (item_features -> relevance); the predicted relevance is
    the item score, and sorting items by score induces the ranking. Pointwise (no pairwise/listwise loss) is a
    legitimate, widely used LTR baseline -- we are explicit that it is pointwise and do NOT claim LambdaMART.
    `kind` in {"ridge","hist_gbm"}."""

    def __init__(self, featurizer, kind="ridge", seed=0):
        self.feat = featurizer
        self.kind = kind
        self.seed = seed
        self.model = None

    def fit(self, queries):
        X, y = self.feat.stack(queries)
        if X.size == 0:
            self.model = None
            return self
        if self.kind == "hist_gbm":
            self.model = HistGradientBoostingRegressor(max_iter=200, random_state=self.seed)
        else:
            self.model = Ridge(alpha=1.0)
        self.model.fit(X, y)
        return self

    def score_query(self, query):
        items = _query_items(query)
        if self.model is None or not items:
            return np.zeros(len(items), dtype=float)
        X = self.feat.transform_items(items)
        if X.size == 0:
            return np.zeros(len(items), dtype=float)
        return np.asarray(self.model.predict(X), dtype=float)


class MeanFeatureBaseline:
    """Honest non-learning baseline: score each item by the mean of its standardized numeric feature vector
    (a 'popularity/feature-mean' proxy). It uses NO relevance labels, so it is a true floor -- if the
    pointwise model cannot beat it, there is no learnable signal. Falls back to all-zero scores (pure
    tie-break order) when there are no numeric features."""

    def __init__(self, featurizer):
        self.feat = featurizer

    def fit(self, queries):
        return self

    def score_query(self, query):
        items = _query_items(query)
        if not items:
            return np.zeros(0, dtype=float)
        X = self.feat.transform_items(items)
        if X.size == 0:
            return np.zeros(len(items), dtype=float)
        return X.mean(axis=1)


def build_menu(featurizer, seed=0):
    """The honest model menu: a feature-mean baseline + two pointwise regressors (Ridge, HistGBM). Returns a
    dict name->unfitted model object exposing .fit(queries) and .score_query(query)."""
    return {
        "feature_mean_baseline": MeanFeatureBaseline(featurizer),
        "pointwise_ridge": PointwiseRanker(featurizer, kind="ridge", seed=seed),
        "pointwise_hist_gbm": PointwiseRanker(featurizer, kind="hist_gbm", seed=seed),
    }


# =========================================================================== frozen per-QUERY certifier
# Minimum number of queries below which we DEFER rather than bound. A percentile bootstrap whose resampling
# unit is the query needs enough queries for the empirical tail to mean anything; below this we cannot give a
# trustworthy one-sided bound and refusing is the honest move (the grouped analogue of science.bootstrap_lower
# / bootstrap_metric_lower returning the point estimate at n<=1, and of the joint-leak gate DEFERRING below
# its min_n). 12 is a conservative floor; the measured-coverage test exercises n_queries in {30,80,200}.
MIN_BOOTSTRAP_QUERIES = 12

# Finite-sample conservative tail shrink for the per-query NDCG/MAP bootstrap. The per-query mean of a bounded
# [0,1] metric is boundary-compressed near 1.0 and the percentile lower bound is median-biased UP, EXACTLY the
# failure science.py documents for the [0,1] classification metric (where it uses an additive c/sqrt(n)
# margin). Here we take the percentile at the TIGHTER effective tail probability alpha_eff = alpha * SHRINK
# (the same monotone, documented knob science.bootstrap_metric_lower uses, _REG_FS_SHRINK), targeting a
# slightly-better-than-nominal one-sided level so EMPIRICAL coverage >= nominal. This is NOT relaxing a spec:
# it makes the bound MORE conservative. SHRINK=0.5 (alpha 0.05 -> 0.025) was chosen by the measured-coverage
# sweep in tests/test_ranking.py: it lifts worst-cell one-sided coverage to >= 0.93 across n_queries in
# {30,80,200} for a fixed true-NDCG simulation, where the naive percentile (SHRINK=1.0) under-covers. The
# value is documented, not reverse-engineered from any target's reported number.
_RANK_FS_SHRINK = 0.5


def per_query_bootstrap_lower(per_query_values, alpha=0.05, B=1200, seed=0, fs_shrink=_RANK_FS_SHRINK):
    """One-sided (>= 1-alpha) LOWER confidence bound on the MEAN per-query metric, resampling whole QUERIES
    with replacement (the correct i.i.d. unit for grouped ranking data) and recomputing the mean per draw.

    This is the ranking analogue of science.bootstrap_metric_lower: it returns a LOWER percentile of the
    bootstrap distribution of the mean, at the TIGHTER probability alpha_eff = alpha * fs_shrink (the
    documented finite-sample conservative shrink). Resampling queries -- not items -- is REQUIRED: items
    within a query are not exchangeable across queries, and an item-level bootstrap would understate the
    variance and over-state the bound (an invalid certificate).

    Returns (lower_bound, point_estimate). DEFERS by returning (None, point) when there are fewer than
    MIN_BOOTSTRAP_QUERIES queries -- the caller turns that into an honest 'cannot certify, too few queries'."""
    vals = np.asarray(per_query_values, dtype=np.float64)
    n = len(vals)
    point = float(vals.mean()) if n else 0.0
    if n < MIN_BOOTSTRAP_QUERIES:
        return None, round(point, 6)
    rng = np.random.default_rng(seed)
    boots = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, n)                     # resample whole QUERIES with replacement
        boots[b] = vals[idx].mean()
    alpha_eff = max(0.0, min(alpha, alpha * fs_shrink))
    lower = float(np.percentile(boots, 100.0 * alpha_eff))
    return round(lower, 6), round(point, 6)


def certify_ranking(per_query_values, theta, *, metric="ndcg@10", checks=1, alpha=0.05, B=1200, seed=0):
    """Frozen ranking certifier, mirroring science.certify_regression's contract and discipline.

    Certified iff the per-QUERY bootstrap LOWER confidence bound on the mean metric clears theta, after a
    Bonferroni correction for `checks` locked-test peeks (alpha/checks). Promote ON THE BOUND, never the
    point. The metric is recomputed per (query-resampled) bootstrap draw. The locked TEST queries are scored
    exactly once by the caller; this function only consumes the per-query value vector.

    Returns a certificate dict shaped like the others (observed, n[=n_queries], metric, theta, checks,
    lower_bound, certified, reason). When there are too few queries to bound, it DEFERS honestly:
    certified=False, lower_bound=None, reason names the deficit (no false-confident certificate)."""
    checks = max(1, int(checks))
    a = alpha / checks
    vals = np.asarray(per_query_values, dtype=np.float64)
    n = len(vals)
    lower, point = per_query_bootstrap_lower(vals, alpha=a, B=B, seed=seed)
    if lower is None:
        return {"observed": round(point, 4), "n": int(n), "metric": metric, "theta": round(float(theta), 4),
                "checks": checks, "alpha_per_check": round(a, 6), "lower_bound": None, "certified": False,
                "reason": f"deferred: only {n} queries (< {MIN_BOOTSTRAP_QUERIES}); too few to bound the "
                          f"mean {metric} by a per-query bootstrap -- refusing a confident-looking invalid "
                          f"bound. Acquire more labeled queries."}
    certified = bool(lower > theta)
    return {"observed": round(point, 4), "n": int(n), "metric": metric, "theta": round(float(theta), 4),
            "checks": checks, "alpha_per_check": round(a, 6), "lower_bound": round(float(lower), 4),
            "certified": certified,
            "reason": ("per-query bootstrap lower bound clears theta after multiplicity correction" if certified
                       else "per-query bootstrap lower confidence bound does not clear theta after paying for "
                            "all peeks")}


# =========================================================================== query-level split
def split_queries(queries, *, n_val_queries, n_test_queries, seed=0):
    """Split at the QUERY level (never split items within a query across folds -- that would leak). Returns
    (train, val, test) query lists. The test/val sizes are taken from the tail of a seeded permutation so the
    split is reproducible and the train set gets whatever remains."""
    n = len(queries)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = min(int(n_test_queries), n)
    n_val = min(int(n_val_queries), max(0, n - n_test))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    pick = lambda idx: [queries[i] for i in idx]
    return pick(train_idx), pick(val_idx), pick(test_idx)


# =========================================================================== entry point
def run_ranking_goal(queries, *, threshold, metric="ndcg@10", k=10, n_val_queries=None, n_test_queries=None,
                     alpha=0.05, B=1200, seed=0, on_event=None):
    """Goal entry point for the ranking vertical, shaped like the rest of the platform.

    Pipeline (single-peek discipline on the TEST queries):
      query-level split  ->  featurize (fit schema on TRAIN items)  ->  fit the honest menu on TRAIN  ->
      SELECT the winner on VALIDATION queries by the per-query NDCG/MAP LOWER BOUND (promote on the bound,
      not the point)  ->  CERTIFY the winner ONCE on held-out TEST queries via the per-query bootstrap  ->
      return a certificate dict + winner info.

    Emits on_event stage events in vfplatform/loop.py's shape -- {"stage","status","experiment",**detail} --
    for the UI: stages split / fanout / measure / val_bound / sealed_certify / done.

    The TEST queries are scored exactly ONCE (the single counted peek); `checks` is passed to the certifier so
    Bonferroni multiplicity is paid. Returns a dict:
      {experiment, status, metric, threshold, k, n_queries{train,val,test}, winner, val_table, certificate}.
    """
    import hashlib

    name, parsed_k = parse_metric(metric)
    k = parsed_k if parsed_k is not None else k
    metric_full = f"{name}@{k}"
    experiment = "rank-" + hashlib.sha256(
        repr({"g": "ranking", "n": len(queries), "m": metric_full}).encode()).hexdigest()[:8]

    def _emit(stage, status="active", **detail):
        if on_event is None:
            return
        try:
            on_event({"stage": stage, "status": status, "experiment": experiment, **detail})
        except Exception:  # noqa: BLE001  a failing observer never breaks the run
            pass

    validate_dataset(queries)
    n = len(queries)
    # default split sizes: ~30% test, ~20% val of the remainder, at least 1 each when feasible
    n_test_queries = n_test_queries if n_test_queries is not None else max(1, int(round(0.30 * n)))
    n_val_queries = n_val_queries if n_val_queries is not None else max(1, int(round(0.20 * n)))

    train, val, test = split_queries(queries, n_val_queries=n_val_queries, n_test_queries=n_test_queries,
                                     seed=seed)
    _emit("split", n_train=len(train), n_val=len(val), n_test=len(test), metric=metric_full)

    # honest decline if the split starved a fold (no test/train queries to fit or peek)
    if not train or not test:
        _emit("done", status="declined")
        return {"experiment": experiment, "status": "declined", "metric": metric_full, "threshold": threshold,
                "k": k, "n_queries": {"train": len(train), "val": len(val), "test": len(test)},
                "winner": None, "val_table": [],
                "certificate": {"observed": 0.0, "n": len(test), "metric": metric_full,
                                "theta": round(float(threshold), 4), "checks": 1, "lower_bound": None,
                                "certified": False,
                                "reason": "declined: query-level split left no train or no test queries"}}

    # featurize: fit the per-item schema on TRAIN items only (no val/test peek in the schema)
    feat = RankingFeaturizer().fit(train)

    # fan-out: fit the honest menu on TRAIN
    _emit("fanout", n_models=3)
    menu = build_menu(feat, seed=seed)
    for m in menu.values():
        m.fit(train)

    # measure on VALIDATION; SELECT the winner by the per-query LOWER BOUND (promote on the bound). The val
    # bound is NOT the certificate -- it is the selection criterion; the sealed test is peeked once below.
    _emit("measure", split="val")
    val_table = []
    val_checks = len(menu)                              # we look at the val bound for each menu member
    for mname, model in menu.items():
        pq = per_query_metric(val, model.score_query, metric=metric_full, k=k, seed=seed)
        lower, point = per_query_bootstrap_lower(pq, alpha=alpha / max(1, val_checks), B=B, seed=seed)
        val_table.append({"model": mname, "val_observed": round(point, 4),
                          "val_lower_bound": (round(lower, 4) if lower is not None else None),
                          "n_val_queries": len(val)})

    def _rank_key(row):
        lb = row["val_lower_bound"]
        return (lb if lb is not None else -1.0, row["val_observed"])

    best_row = max(val_table, key=_rank_key)
    winner_name = best_row["model"]
    winner = menu[winner_name]
    _emit("val_bound", winner=winner_name, val=best_row["val_observed"],
          val_lower_bound=best_row["val_lower_bound"])

    # CERTIFY ONCE on the held-out TEST queries (the single counted peek). checks=1: exactly one sealed peek
    # of the test queries happens here. Selection above used only the VAL queries, so the test is unpeeked
    # until this line -- the single-peek discipline holds.
    test_pq = per_query_metric(test, winner.score_query, metric=metric_full, k=k, seed=seed)
    cert = certify_ranking(test_pq, threshold, metric=metric_full, checks=1, alpha=alpha, B=B, seed=seed)
    _emit("sealed_certify", winner=winner_name, n_test=len(test), lower_bound=cert.get("lower_bound"),
          certified=cert.get("certified"))

    status = "certified" if cert.get("certified") else ("deferred" if cert.get("lower_bound") is None
                                                        else "honest_stop")
    _emit("done", status=status)
    return {"experiment": experiment, "status": status, "metric": metric_full, "threshold": threshold, "k": k,
            "n_queries": {"train": len(train), "val": len(val), "test": len(test)},
            "winner": {"model": winner_name, "val_observed": best_row["val_observed"],
                       "val_lower_bound": best_row["val_lower_bound"]},
            "val_table": val_table, "certificate": cert}
