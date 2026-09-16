"""Frozen-certifier COVERAGE regression tests (audit step 15, findings F8/F11).

These pin the EMPIRICAL one-sided coverage of the bootstrap lower bounds so the finite-sample corrections
can't silently regress. A valid one-sided (1-alpha) lower bound must satisfy P(lower_bound <= true) >= 0.95.

  * F8 (classification): the balanced_accuracy/macro_f1 bootstrap was anti-conservative (worst-cell
    coverage 0.887 at true=0.90, n=60). The fix is an additive finite-sample margin c/sqrt(n) (c=0.25)
    in _bootstrap_classification_metric_lower; this test asserts the corrected coverage >= 0.94 at the
    previously-weak cells AND that the NAIVE bound (margin=0) is materially worse, so the test is real.
  * F11 (regression): the certify_regression bootstrap (alpha*0.5 tail shrink) is checked for r2/neg_rmse/
    neg_mae under gaussian and right-skewed (lognormal) residuals.

Plain asserts + a main() (no pytest), matching test_vfplatform.py. Modest trials/B so it runs < 60s.
Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_certifier_coverage.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectorforge import science


def _clf_coverage(true_t, n, trials, fs_margin, B=300):
    """Empirical P(lower_bound <= true balanced_accuracy) for symmetric per-class correctness prob true_t."""
    half = n // 2
    yt = ["0"] * half + ["1"] * half
    rng = np.random.default_rng(0)
    hits = 0
    for tr in range(trials):
        yp = [(a if rng.random() < true_t else ("1" if a == "0" else "0")) for a in yt]
        lb, _ = science._bootstrap_classification_metric_lower(
            "balanced_accuracy", yt, yp, ["0", "1"], alpha=0.05, B=B, seed=tr, fs_margin=fs_margin)
        hits += int(lb <= true_t)
    return hits / trials


def _reg_coverage(metric, n, trials, noise, B=300):
    """Empirical P(lower_bound <= true metric) for a fixed linear model under a noise regime."""
    rng = np.random.default_rng(0)
    hits = 0
    # true metric is estimated on a giant sample once (the population value the bound must cover)
    big = 200000
    xb = rng.normal(size=big)
    eb = (rng.normal(size=big) if noise == "gauss" else (rng.lognormal(0, 0.7, size=big) - np.exp(0.7 * 0.7 / 2)))
    ytrue_b = 2.0 * xb
    ypred_b = 2.0 * xb + 0.0          # perfect-mean model; residual is the noise on y
    yt_b = ytrue_b + eb
    true_metric = science.score_regression_metric(metric, yt_b, ypred_b)
    for tr in range(trials):
        x = rng.normal(size=n)
        e = (rng.normal(size=n) if noise == "gauss" else (rng.lognormal(0, 0.7, size=n) - np.exp(0.7 * 0.7 / 2)))
        yt = 2.0 * x + e
        yp = 2.0 * x
        lb, _ = science.bootstrap_metric_lower(yt, yp, metric, alpha=0.05, B=B, seed=tr)
        hits += int(lb <= true_metric)
    return hits / trials


# ----------------------------------------------------------------- F8 classification coverage
def test_clf_bootstrap_coverage_conservative():
    # the two cells that were worst before the fix (high accuracy, small n)
    for t, n in [(0.90, 60), (0.90, 100), (0.80, 60)]:
        cov = _clf_coverage(t, n, trials=300, fs_margin=science._CLF_FS_MARGIN)
        assert cov >= 0.94, f"classification coverage under-conservative at true={t} n={n}: {cov:.3f}"


def test_clf_naive_bound_is_worse_so_test_is_meaningful():
    # prove the margin is doing real work: the naive (margin=0) bound under-covers at the worst cell
    naive = _clf_coverage(0.90, 60, trials=300, fs_margin=0.0)
    fixed = _clf_coverage(0.90, 60, trials=300, fs_margin=science._CLF_FS_MARGIN)
    assert naive < 0.94 < fixed, f"expected naive {naive:.3f} < 0.94 <= fixed {fixed:.3f}"


# ----------------------------------------------------------------- F11 regression coverage
def test_reg_bootstrap_coverage_gauss_in_envelope():
    # the VALIDATED envelope: light-tailed (gaussian) residuals -> coverage >= 0.95 for all 3 metrics
    for metric in ("r2", "neg_rmse", "neg_mae"):
        cov = _reg_coverage(metric, n=200, trials=250, noise="gauss")
        assert cov >= 0.95, f"regression coverage below envelope for {metric}/gauss at n=200: {cov:.3f}"


def test_reg_heavy_skew_defers():
    # OUT of envelope: heavy right-skewed residuals -> the certifier must DEFER (certified=False,
    # in_envelope=False, deferred=True) even with an easy threshold, because the bootstrap bound is not a
    # valid 95% bound there (measured under-coverage ~0.67). It must NOT emit an invalid certificate.
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    resid = rng.lognormal(0, 0.9, size=300) - np.exp(0.9 * 0.9 / 2)   # heavy right skew, zero-mean
    yt = (2.0 * x + resid).tolist()
    yp = (2.0 * x).tolist()
    c = science.certify_regression(yt, yp, "neg_rmse", -1e9, checks=1, alpha=0.05)   # trivially-easy theta
    assert c["in_envelope"] is False and c["deferred"] is True and c["certified"] is False, \
        f"heavy-skew residuals must DEFER, not certify: {c}"
    # and a clean gaussian-residual cert (n>=150) stays IN-envelope and certifies the easy theta
    cg = science.certify_regression((2.0 * x + rng.normal(size=300)).tolist(), yp, "neg_rmse", -1e9)
    assert cg["in_envelope"] is True and cg["certified"] is True


def test_accuracy_cp_path_untouched():
    # the verified Clopper-Pearson accuracy bound must be exactly reproducible (we did not touch it)
    c = science.certify_accuracy(0.82, 50, 0.80, checks=1, alpha=0.05)
    assert abs(c["lower_bound"] - 0.6952) < 0.02 and c["certified"] is False


TESTS = [test_clf_bootstrap_coverage_conservative, test_clf_naive_bound_is_worse_so_test_is_meaningful,
         test_reg_bootstrap_coverage_gauss_in_envelope, test_reg_heavy_skew_defers,
         test_accuracy_cp_path_untouched]


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
