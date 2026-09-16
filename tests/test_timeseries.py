"""Time-series forecasting vertical -- MEASURED block-bootstrap coverage + split-leakage regression tests.

This pins the EMPIRICAL one-sided coverage of the block-bootstrap forecast bound so the finite-sample knobs
can't silently regress, and proves the block bootstrap is doing REAL work by showing the naive i.i.d.
bootstrap UNDER-covers on the same autocorrelated series.

A valid one-sided (1-alpha) lower bound must satisfy P(lower_bound <= true_metric) >= 1-alpha = 0.95. We
assert the BLOCK bootstrap reaches >= 0.93 (a small finite-trial / finite-n margin below nominal, exactly as
test_certifier_coverage.py uses a 0.94 acceptance for a 0.95 target) across AR(1) phi in {0.3, 0.6, 0.9},
AND that the NAIVE i.i.d. bootstrap under-covers at phi=0.9 (so the comparison proves the block structure is
load-bearing, not decoration). It also asserts the forward-chaining split never puts a test timestamp before
a train timestamp.

Plain asserts + a main() with a TESTS list + run() printing "---- N passed, M failed ----", matching
tests/test_certifier_coverage.py. Modest n/B/trials so it runs < 90s.

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_timeseries.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectorforge import science
from vfplatform import timeseries as ts


# ----------------------------------------------------------------- AR(1) coverage harness
def _ar1(rng, n, phi, sigma=1.0):
    """A zero-mean AR(1) error series e[t] = phi*e[t-1] + sigma*sqrt(1-phi^2)*z[t] (stationary, unit marginal
    variance when sigma=1). This is the canonical autocorrelated residual the i.i.d. bootstrap mishandles."""
    z = rng.standard_normal(n)
    e = np.empty(n, dtype=np.float64)
    e[0] = z[0]
    innov = sigma * np.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + innov * z[t]
    return e


def _true_neg_rmse(phi, sigma=1.0):
    """The POPULATION value the bound must cover. The forecast error series IS the AR(1) e[t] (the model
    predicts the signal exactly; the residual is the AR(1) noise), so neg_rmse -> -sqrt(E[e^2]) = -sigma as
    n -> inf (stationary AR(1) has marginal variance sigma^2). This is a clean closed form, NOT estimated from
    the same draws used to test coverage, so the coverage measurement is honest."""
    return -float(sigma)


def _coverage(phi, *, n, trials, evaluator, B=300, sigma=1.0, seed0=0):
    """Empirical P(lower_bound <= true_neg_rmse) over `trials` independent AR(1) series of length n.

    evaluator='block' -> ts.block_bootstrap_lower (data-driven circular block); 'iid' -> the same primitive
    forced to block_size=1 (which IS the i.i.d. pair bootstrap), so the two bounds differ ONLY in whether the
    block structure is preserved -- the cleanest possible A/B for 'does the block bootstrap do real work'.
    Deferred draws (too few effective blocks) are counted as COVERED (a deferral never falsely certifies, so
    it cannot break a one-sided lower-bound guarantee); in these cells deferral is rare at the chosen n.
    """
    true_m = _true_neg_rmse(phi, sigma)
    hits = 0
    for tr in range(trials):
        rng = np.random.default_rng(seed0 + tr)
        e = _ar1(rng, n, phi, sigma)
        y_true = e                      # signal == 0 WLOG; the model predicts 0; residual = e (the AR(1))
        y_pred = np.zeros(n, dtype=np.float64)
        if evaluator == "iid":
            bb = ts.block_bootstrap_lower(y_true, y_pred, "neg_rmse", block_size=1, alpha=0.05, B=B,
                                          seed=tr, min_eff_blocks=ts._TS_MIN_EFF_BLOCKS)
        else:
            bb = ts.block_bootstrap_lower(y_true, y_pred, "neg_rmse", block_size=None, alpha=0.05, B=B,
                                          seed=tr, min_eff_blocks=ts._TS_MIN_EFF_BLOCKS)
        if bb["deferred"]:
            hits += 1                   # withholding can never violate a one-sided lower-bound guarantee
        else:
            hits += int(bb["lower"] <= true_m)
    return hits / trials


# ----------------------------------------------------------------- coverage tests
def test_block_bootstrap_covers_under_autocorrelation():
    """The block bootstrap achieves >= 0.93 one-sided coverage at EVERY AR(1) phi in {0.3, 0.6, 0.9}.
    n is scaled up at strong autocorrelation so the effective number of blocks supports a real bound (the
    DEFER guard handles the regime where it cannot; here we test the regime where it CAN)."""
    # n is scaled UP with phi so the effective number of blocks supports a real bound: at phi=0.9 the
    # integrated autocorrelation time is ~19, so n=2000 gives ~50 blocks (the regime where the bound is
    # backable). The DEFER guard handles smaller n at high phi (test_defer_guard_refuses_when_too_few_blocks).
    cells = [(0.3, 400), (0.6, 600), (0.9, 2000)]
    cov = {}
    for phi, n in cells:
        cov[phi] = _coverage(phi, n=n, trials=250, evaluator="block", B=250)
    for phi, _n in cells:
        assert cov[phi] >= 0.93, f"block-bootstrap under-covers at phi={phi}: {cov[phi]:.3f} (< 0.93)"
    _BLOCK_COVERAGE.update(cov)


def test_naive_iid_undercovers_at_high_autocorrelation():
    """The NAIVE i.i.d. (block_size=1) bootstrap UNDER-covers at phi=0.9 on the SAME series the block bound
    covers -- this is what proves the block structure is load-bearing, not decoration. Mirrors
    test_certifier_coverage.test_clf_naive_bound_is_worse_so_test_is_meaningful."""
    n = 2000
    iid = _coverage(0.9, n=n, trials=250, evaluator="iid", B=250)
    block = _BLOCK_COVERAGE.get(0.9) or _coverage(0.9, n=n, trials=250, evaluator="block", B=250)
    _NAIVE_COVERAGE[0.9] = iid
    assert iid < 0.93, f"expected naive i.i.d. bound to UNDER-cover at phi=0.9, got {iid:.3f} (>= 0.93)"
    assert block > iid, f"block coverage {block:.3f} must exceed naive i.i.d. {iid:.3f} (block does real work)"


def test_white_residuals_block_reduces_to_iid():
    """At phi=0 (white residuals) the block bootstrap must NOT over-conservatize: it should cover (>= 0.93)
    AND stay close to the i.i.d. bound, i.e. the block machinery costs ~nothing on truly independent data
    (equality at rho_k=0, as the docstring claims)."""
    block = _coverage(0.0, n=400, trials=200, evaluator="block", B=300)
    iid = _coverage(0.0, n=400, trials=200, evaluator="iid", B=300)
    assert block >= 0.93, f"block bound should still cover white residuals, got {block:.3f}"
    assert block >= iid - 0.03, f"block {block:.3f} should not collapse below iid {iid:.3f} on white noise"


def test_block_size_one_equals_iid_pair_bootstrap():
    """block_size=1 must reduce EXACTLY to an i.i.d. pair bootstrap (each block is one row). Same seed/B ->
    identical lower bound. This pins the 'strict generalization of the frozen i.i.d. bound' invariant."""
    rng = np.random.default_rng(7)
    e = _ar1(rng, 300, 0.6)
    yt, yp = e, np.zeros(300)
    bb1 = ts.block_bootstrap_lower(yt, yp, "neg_rmse", block_size=1, alpha=0.05, B=400, seed=3)
    # an explicit i.i.d. pair bootstrap of the SAME metric with the SAME draws
    rng2 = np.random.default_rng(3)
    boots = np.empty(400)
    for b in range(400):
        idx = rng2.integers(0, 300, 300)
        boots[b] = science.score_regression_metric("neg_rmse", yt[idx], yp[idx])
    iid_lower = round(float(np.percentile(boots, 100.0 * 0.05 * ts._TS_FS_SHRINK)), 6)
    assert abs(bb1["lower"] - iid_lower) < 1e-9, f"block_size=1 ({bb1['lower']}) != iid pair bootstrap ({iid_lower})"
    assert bb1["block_size"] == 1 and not bb1["deferred"]


# ----------------------------------------------------------------- defer guard
def test_defer_guard_refuses_when_too_few_blocks():
    """Strong autocorrelation + small n -> the data-driven block length leaves < min_eff_blocks effective
    blocks -> the certifier must DEFER (deferred=True, certified=False), never emit an anti-conservative
    number. Precedent: science._joint_leak_verdict deferring below min_n."""
    rng = np.random.default_rng(0)
    e = _ar1(rng, 80, 0.9)                       # very autocorrelated, small n -> few effective blocks
    yt, yp = e, np.zeros(80)
    cert = ts.certify_timeseries(yt, yp, "neg_rmse", -2.0, checks=1, alpha=0.05, B=300, seed=1)
    assert cert["deferred"] is True, f"expected defer at phi=0.9/n=80, got eff_blocks={cert['eff_blocks']}"
    assert cert["certified"] is False, "a deferred bound must never certify"
    assert "DEFERRED" in cert["reason"]


# ----------------------------------------------------------------- forward-chaining split leakage
def test_forward_chaining_split_no_future_leak():
    """The forward-chaining split must NEVER put a test timestamp before a train timestamp: max(train idx) <
    min(test idx), with an embargo gap between blocks. We tag rows with their absolute time index and assert
    the ordering invariant directly on the produced blocks."""
    N = 200
    rows = [{"t": i, "target": float(i)} for i in range(N)]      # t is the timestamp; strictly increasing
    train, val, test, info = ts.make_forecast_splits(rows, horizon=1, n_val=30, n_test=30, embargo=5)
    assert train and val and test
    max_train_t = max(r["t"] for r in train)
    min_val_t = min(r["t"] for r in val)
    min_test_t = min(r["t"] for r in test)
    max_val_t = max(r["t"] for r in val)
    assert max_train_t < min_val_t, f"train ({max_train_t}) must be strictly before val ({min_val_t})"
    assert max_val_t < min_test_t, f"val ({max_val_t}) must be strictly before test ({min_test_t})"
    assert max_train_t < min_test_t, f"NO test timestamp before train: train {max_train_t} >= test {min_test_t}"
    # embargo really drops rows between blocks
    assert (min_val_t - max_train_t) > 1, f"embargo gap missing between train and val: {min_val_t - max_train_t}"
    assert info["no_future_leak"] is True


def test_featurizer_is_causal():
    """Every feature must be a function of strictly-PAST target values: perturbing y[t] (and everything after
    the origin) must NOT change the feature vector that PREDICTS y[t]. We build features on a series, then
    corrupt the future and confirm the feature rows are unchanged (only y changes)."""
    base = [{"target": float(v)} for v in range(60)]
    feat = ts.LagWindowFeaturizer(lags=(1, 2, 3), windows=(3,), horizon=1)
    X0, y0, o0 = feat.transform(base)
    corrupt = [dict(r) for r in base]
    # corrupt the LATER half of the targets
    for i in range(30, 60):
        corrupt[i]["target"] = corrupt[i]["target"] + 1000.0
    X1, y1, o1 = feat.transform(corrupt)
    # the feature rows whose ORIGIN (t-h) is before index 30 must be identical (their past wasn't touched)
    safe = [k for k, oi in enumerate(o0) if (oi - feat.horizon) < 30 and (oi - feat.horizon - max(feat.lags) + 1) >= 0]
    assert safe, "expected some origins entirely in the un-corrupted past"
    assert np.allclose(X0[safe], X1[safe]), "a feature changed when only the FUTURE was corrupted -> not causal"


# ----------------------------------------------------------------- end-to-end entry point
def test_run_timeseries_goal_end_to_end():
    """run_timeseries_goal on a clean AR(1)-plus-trend series returns a certificate of the right shape, with a
    single test peek (checks=1), the winner selected on validation, and the split leakage invariant holding."""
    rng = np.random.default_rng(11)
    n = 700
    trend = 0.02 * np.arange(n)
    e = _ar1(rng, n, 0.5, sigma=0.5)
    series = (trend + e).tolist()
    rows = [{"target": float(v)} for v in series]
    events = []
    cert = ts.run_timeseries_goal(rows, horizon=1, threshold=-1.0, metric="neg_rmse",
                                  lags=(1, 2, 3, 7), windows=(3, 7), n_val=120, n_test=120, embargo=1,
                                  B=300, seed=0, on_event=lambda ev: events.append(ev["stage"]))
    for key in ("observed", "n", "metric", "theta", "checks", "lower_bound", "certified", "reason",
                "block_size", "eff_blocks", "deferred", "winner", "val_leaderboard", "decision"):
        assert key in cert, f"certificate missing key {key!r}"
    assert cert["checks"] == 1, f"the future test block must be peeked exactly once, got checks={cert['checks']}"
    assert cert["metric"] == "neg_rmse"
    assert cert["winner"] in ("seasonal_naive", "ridge_lags", "hgb_lags")
    assert cert["split_info"]["no_future_leak"] is True
    # the staged events fire in the loop's shape so the UI can show the run
    for stage in ("split", "fanout", "measure", "val_bound", "sealed_certify", "done"):
        assert stage in events, f"missing stage event {stage!r} (got {events})"
    # the lower bound must be <= the point (a lower bound never exceeds the observed metric)
    if cert["lower_bound"] is not None and cert["observed"] is not None:
        assert cert["lower_bound"] <= cert["observed"] + 1e-6, "lower bound exceeds the observed metric"


def test_metric_guard_refuses_unknown_metric():
    """An unknown/typo'd metric is refused at goal construction, not silently scored as the wrong quantity."""
    raised = False
    try:
        ts.assert_forecast_metric("accuracy")
    except ValueError:
        raised = True
    assert raised, "assert_forecast_metric must refuse a non-forecasting metric"


# coverage numbers shared across the block/naive comparison tests (printed in the summary table)
_BLOCK_COVERAGE = {}
_NAIVE_COVERAGE = {}

TESTS = [
    test_block_bootstrap_covers_under_autocorrelation,
    test_naive_iid_undercovers_at_high_autocorrelation,
    test_white_residuals_block_reduces_to_iid,
    test_block_size_one_equals_iid_pair_bootstrap,
    test_defer_guard_refuses_when_too_few_blocks,
    test_forward_chaining_split_no_future_leak,
    test_featurizer_is_causal,
    test_run_timeseries_goal_end_to_end,
    test_metric_guard_refuses_unknown_metric,
]


def run(tests):
    p = f = 0
    t0 = time.time()
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}"); f += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {e}"); f += 1
    if _BLOCK_COVERAGE:
        print("\n  measured one-sided coverage (neg_rmse, AR(1)), nominal 0.95:")
        print("    phi    block    naive_iid")
        for phi in sorted(set(_BLOCK_COVERAGE) | set(_NAIVE_COVERAGE)):
            b = _BLOCK_COVERAGE.get(phi)
            nv = _NAIVE_COVERAGE.get(phi)
            print(f"    {phi:<5}  {('%.3f' % b) if b is not None else '  -  ':<7}  "
                  f"{('%.3f' % nv) if nv is not None else '  -  '}")
    print(f"\n  ---- {p} passed, {f} failed ({time.time()-t0:.1f}s) ----")
    return p, f


if __name__ == "__main__":
    _, fails = run(TESTS)
    sys.exit(1 if fails else 0)
