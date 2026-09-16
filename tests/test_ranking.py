"""Measured-coverage + correctness tests for the learning-to-rank vertical (vfplatform/ranking.py).

Mirrors tests/test_certifier_coverage.py: plain asserts, a TESTS list, a run() printing
"  ---- N passed, M failed ----", no pytest. Modest B/trials so the whole file runs < 90s.

What is pinned:
  (a) NDCG@k and MAP@k correctness against a HAND-WORKED example (exact values);
  (b) the per-QUERY bootstrap lower bound achieves >= 0.93 one-sided coverage for a fixed true-NDCG
      simulation across n_queries in {30,80,200} -- AND the naive (shrink=1.0) bound is materially worse at
      the worst cell, so the finite-sample shrink is doing real work (the test is meaningful, not vacuous);
  (c) the bound is CONSERVATIVE: lower_bound <= point estimate;
  plus the honest DEFER below MIN_BOOTSTRAP_QUERIES, promote-on-the-bound selection, and single-peek shape
  of run_ranking_goal.

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_ranking.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vfplatform import ranking as R


# ----------------------------------------------------------------- (a) hand-worked metric correctness
# ONE query, 6 items. Scores [6,5,4,3,2,1] induce the predicted (descending-score) order to be exactly the
# item order, whose relevances are [3,2,3,0,1,2].
#   DCG@6  = sum (2^rel - 1)/log2(rank+1) over predicted order [3,2,3,0,1,2]
#          = 7/1 + 3/1.585 + 7/2 + 0/2.322 + 1/2.585 + 3/2.807 = 13.848263629272981
#   ideal order (relevance-sorted desc) = [3,3,2,2,1,0]
#   IDCG@6 = 7/1 + 7/1.585 + 3/2 + 3/2.322 + 1/2.585 + 0/2.807 = 14.595390756454924
#   NDCG@6 = 13.848263629272981 / 14.595390756454924 = 0.9488107485678985
#   NDCG@3: DCG@3 = 7 + 3/1.585 + 7/2 = 12.3927...; IDCG@3 = 7 + 7/1.585 + 3/2 = 12.9166...
#          NDCG@3 = 0.9594535145926796
# MAP (binary rel>0): predicted-order rels [3,2,3,0,1,2]; relevant at ranks 1,2,3,5,6; n_rel=5.
#   AP@6 = (1/1 + 2/2 + 3/3 + 4/5 + 5/6)/5 = 0.9266666666666665
#   AP@3: top3 [3,2,3] all relevant, denom = min(5,3)=3 -> (1+1+1)/3 = 1.0
_HAND_RELS = [3, 2, 3, 0, 1, 2]
_HAND_SCORES = [6, 5, 4, 3, 2, 1]


def test_ndcg_hand_example_exact():
    n6 = R.ndcg_at_k_for_query(_HAND_RELS, _HAND_SCORES, 6)
    n3 = R.ndcg_at_k_for_query(_HAND_RELS, _HAND_SCORES, 3)
    assert abs(n6 - 0.9488107485678985) < 1e-12, f"NDCG@6 wrong: {n6!r}"
    assert abs(n3 - 0.9594535145926796) < 1e-12, f"NDCG@3 wrong: {n3!r}"


def test_map_hand_example_exact():
    a6 = R.ap_at_k_for_query(_HAND_RELS, _HAND_SCORES, 6)
    a3 = R.ap_at_k_for_query(_HAND_RELS, _HAND_SCORES, 3)
    assert abs(a6 - 0.9266666666666665) < 1e-12, f"AP@6 wrong: {a6!r}"
    assert abs(a3 - 1.0) < 1e-12, f"AP@3 wrong: {a3!r}"


def test_ndcg_perfect_and_empty_conventions():
    # a perfectly-ordered query scores NDCG 1.0; a no-positive-relevance query is 0.0 (counts, not dropped)
    assert abs(R.ndcg_at_k_for_query([2, 1, 0], [3, 2, 1], 3) - 1.0) < 1e-12
    assert R.ndcg_at_k_for_query([0, 0, 0], [3, 2, 1], 3) == 0.0
    assert R.ap_at_k_for_query([0, 0, 0], [3, 2, 1], 3) == 0.0


def test_metric_parse_refuses_unknown():
    ok = False
    try:
        R.parse_metric("auc@10")
    except ValueError:
        ok = True
    assert ok, "parse_metric must refuse an unsupported ranking metric (no silent fall-through)"


# ----------------------------------------------------------------- (b)/(c) per-query bootstrap coverage
def _ndcg_coverage(n_queries, trials, true_ndcg, B, shrink, conc):
    """Empirical P(lower_bound <= true mean NDCG). Per-query NDCG values are drawn from a Beta with mean
    true_ndcg and concentration `conc` -- a bounded [0,1], right-boundary-compressed simulation that is the
    regime where the naive percentile under-covers. Each trial draws n_queries per-query values, resamples
    whole queries to form the bound, and checks coverage of the FIXED population mean true_ndcg."""
    rng = np.random.default_rng(12345)
    a, b = true_ndcg * conc, (1.0 - true_ndcg) * conc
    hits = tot = 0
    for tr in range(trials):
        vals = rng.beta(a, b, size=n_queries)
        lb, _pt = R.per_query_bootstrap_lower(vals, alpha=0.05, B=B, seed=tr, fs_shrink=shrink)
        if lb is None:
            continue
        hits += int(lb <= true_ndcg)
        tot += 1
    return hits / max(tot, 1)


def test_per_query_bootstrap_coverage_conservative():
    # the corrected (shrink=0.5) per-query bound is >= 0.93 one-sided coverage in EVERY cell, for a fixed
    # true NDCG across n_queries in {30,80,200}. (true=0.85 sits in the boundary-compressed regime.)
    for n in (30, 80, 200):
        cov = _ndcg_coverage(n, trials=300, true_ndcg=0.85, B=300, shrink=R._RANK_FS_SHRINK, conc=4.0)
        assert cov >= 0.93, f"per-query coverage under-conservative at n_queries={n}: {cov:.3f}"


def test_naive_bound_is_worse_so_test_is_meaningful():
    # prove the shrink does real work: the naive (shrink=1.0) bound under-covers at the worst cell while the
    # corrected one clears 0.93 -- exactly the pattern science.py pins for its classification/regression bounds.
    naive = min(_ndcg_coverage(n, 300, 0.85, 300, shrink=1.0, conc=4.0) for n in (30, 80, 200))
    fixed = min(_ndcg_coverage(n, 300, 0.85, 300, shrink=R._RANK_FS_SHRINK, conc=4.0) for n in (30, 80, 200))
    assert naive < 0.93 <= fixed, f"expected worst-cell naive {naive:.3f} < 0.93 <= fixed {fixed:.3f}"


def test_bound_is_below_point_estimate():
    # the lower bound must never exceed the point estimate (a one-sided LOWER bound)
    rng = np.random.default_rng(0)
    for _ in range(20):
        vals = rng.beta(3.0, 1.5, size=50)            # n_queries=50 >= MIN_BOOTSTRAP_QUERIES
        lb, pt = R.per_query_bootstrap_lower(vals, alpha=0.05, B=300, seed=7)
        assert lb is not None and lb <= pt + 1e-9, f"lower bound {lb} above point {pt}"


def test_defers_when_too_few_queries():
    # below MIN_BOOTSTRAP_QUERIES the bound DEFERS (None) and the certifier refuses honestly (not certified)
    vals = np.array([0.9, 0.8, 0.95, 0.7])            # 4 queries < MIN_BOOTSTRAP_QUERIES
    lb, pt = R.per_query_bootstrap_lower(vals, alpha=0.05, B=300, seed=0)
    assert lb is None and pt > 0
    cert = R.certify_ranking(vals, theta=0.5, metric="ndcg@10", checks=1)
    assert cert["certified"] is False and cert["lower_bound"] is None and "too few" in cert["reason"]


# ----------------------------------------------------------------- entry-point shape + promote-on-the-bound
def _synthetic_ranking_dataset(n_queries=120, items_per_query=8, signal=2.0, seed=0):
    """Queries where item relevance is a noisy monotone function of a single 'score' feature, so a pointwise
    regressor SHOULD beat the feature-mean baseline -- a learnable ranking signal."""
    rng = np.random.default_rng(seed)
    queries = []
    for q in range(n_queries):
        items = []
        for _ in range(items_per_query):
            x = float(rng.normal())
            noise = float(rng.normal()) * 0.5
            latent = signal * x + noise
            rel = int(np.clip(round(latent + 1.5), 0, 3))     # 0..3 graded relevance
            items.append({"features": {"x": x, "x2": x * x}, "relevance": rel})
        queries.append({"qid": f"q{q}", "items": items})
    return queries


def test_run_ranking_goal_shape_and_single_peek():
    events = []
    queries = _synthetic_ranking_dataset(n_queries=120, seed=1)
    out = R.run_ranking_goal(queries, threshold=0.5, metric="ndcg@5", k=5,
                             n_val_queries=30, n_test_queries=30, B=300, seed=0,
                             on_event=lambda e: events.append(e))
    cert = out["certificate"]
    # certificate carries the canonical fields, shaped like the other verticals
    for key in ("observed", "n", "metric", "theta", "checks", "lower_bound", "certified", "reason"):
        assert key in cert, f"certificate missing field {key!r}"
    assert cert["metric"] == "ndcg@5"
    assert cert["n"] == out["n_queries"]["test"] == 30      # certified n is the # of TEST QUERIES, not items
    assert cert["checks"] == 1                              # single counted peek of the sealed test queries
    # the val/test split is disjoint at the query level and uses the configured sizes
    assert out["n_queries"] == {"train": 60, "val": 30, "test": 30}
    # the canonical stage events fired in order for the UI
    stages = [e["stage"] for e in events]
    for s in ("split", "fanout", "measure", "val_bound", "sealed_certify", "done"):
        assert s in stages, f"missing stage event {s!r}"


def test_run_ranking_goal_promotes_on_the_bound():
    # with a real signal, the pointwise model should be selected over the feature-mean baseline, and the
    # winner is chosen by the VAL LOWER BOUND (promote on the bound). We assert the winner is a pointwise
    # model and that its val lower bound is the max in the table.
    queries = _synthetic_ranking_dataset(n_queries=160, signal=2.5, seed=3)
    out = R.run_ranking_goal(queries, threshold=0.5, metric="ndcg@10", k=10,
                             n_val_queries=40, n_test_queries=40, B=300, seed=0)
    assert out["winner"]["model"].startswith("pointwise"), f"expected a pointwise winner: {out['winner']}"
    lbs = [r["val_lower_bound"] for r in out["val_table"] if r["val_lower_bound"] is not None]
    assert out["winner"]["val_lower_bound"] == max(lbs), "winner must have the max val LOWER BOUND"


TESTS = [test_ndcg_hand_example_exact, test_map_hand_example_exact,
         test_ndcg_perfect_and_empty_conventions, test_metric_parse_refuses_unknown,
         test_per_query_bootstrap_coverage_conservative, test_naive_bound_is_worse_so_test_is_meaningful,
         test_bound_is_below_point_estimate, test_defers_when_too_few_queries,
         test_run_ranking_goal_shape_and_single_peek, test_run_ranking_goal_promotes_on_the_bound]


def run(tests):
    p = f = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}"); f += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {e}"); f += 1
    print(f"  ---- {p} passed, {f} failed ----")
    return p, f


if __name__ == "__main__":
    _, fails = run(TESTS)
    sys.exit(1 if fails else 0)
