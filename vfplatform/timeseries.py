"""Time-series forecasting vertical -- a SELF-CONTAINED module that adds a leakage-safe forecasting path on
top of the frozen science.py core WITHOUT touching it.

WHY A SEPARATE MODULE (the load-bearing reason). Forecasting is NOT i.i.d.. Two of the frozen core's
assumptions are violated and silently produce INVALID certificates if reused as-is:

  1. science.make_splits SHUFFLES rows (stratified random split). For a time series that LEAKS THE FUTURE
     into train -- a random split puts t+5 in train and t+3 in test, so the model "predicts" the past from
     the future. We instead use a forward-chaining / expanding-window split with a hard EMBARGO gap.

  2. science.bootstrap_metric_lower resamples (y_true, y_pred) PAIRS i.i.d., which assumes EXCHANGEABLE rows.
     Forecast errors are AUTOCORRELATED (a model that is wrong at t is usually wrong at t+1). An i.i.d.
     bootstrap destroys that dependence, so its bootstrap distribution is too NARROW and its lower percentile
     sits too HIGH -- ANTI-CONSERVATIVE. Measured in DESIGN_timeseries_serving.md down to 0.66 one-sided
     coverage at AR(1) phi=0.9 (nominal 0.95). We instead use a MOVING-BLOCK / CIRCULAR-BLOCK bootstrap that
     resamples contiguous BLOCKS, preserving within-block autocorrelation, so the bound is conservative.

The block bootstrap is a STRENGTHENING, never a relaxation: at white residuals (phi=0) it reduces (up to
block_size=1) EXACTLY to the i.i.d. bound, and under positive autocorrelation it is strictly more
conservative. It is NOT a free lunch: at very strong autocorrelation + small n the effective number of
independent blocks (n/L) is too small to bound reliably, so the certifier DEFERS (refuses, like
science._joint_leak_verdict defers below min_n) rather than emit a number it cannot back.

This module reuses science.score_regression_metric for every metric so the metric DEFINITIONS stay
single-sourced, and certify_timeseries returns the SAME dict shape as science.certify_regression (plus
block_size / eff_blocks / deferred) so the platform routing is uniform. science.py is imported read-only.

  run with:  /Users/abdullahalghamdi/jax-env-311/bin/python -c "import vfplatform.timeseries"
  self-test: /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_timeseries.py
"""
import math
import os
import sys

import numpy as np

_VF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VF not in sys.path:
    sys.path.insert(0, _VF)
from vectorforge import science

# Metrics this vertical can certify. neg_rmse/neg_mae are the natural forecasting metrics (higher-is-better
# negated error, like science.py's regression orientation); r2 is supported for completeness. Anything else
# is refused at goal construction (same discipline as science.assert_certifiable_metric) so a typo'd metric
# never falls through to the WRONG quantity.
FORECAST_METRICS = ("neg_rmse", "neg_mae", "r2")


def assert_forecast_metric(metric):
    if metric not in FORECAST_METRICS:
        raise ValueError(
            f"metric {metric!r} has no forecasting certifier; supported: {FORECAST_METRICS}. Refusing "
            f"because the block-bootstrap certifier only validates these higher-is-better regression "
            f"metrics on a temporally-ordered error series.")


# ============================================================================ 1. forward-chaining split
def make_forecast_splits(series_rows, *, horizon=1, n_val=None, n_test=None, embargo=0):
    """Forward-chaining / expanding-window split with an EMBARGO gap. NO SHUFFLING -- the only leakage-safe
    split for a time series.

    `series_rows` is a list of rows in STRICT TEMPORAL ORDER (oldest first); each row is opaque to the split
    (it only partitions indices). Returns (train, val, test, info). Layout in time:

        |<------------ train ------------>|  embargo  |<- val ->|  embargo  |<- test ->|
        0                              n_tr-1                                         N-1

    The EMBARGO (>= horizon-1 recommended) drops the `embargo` rows immediately before each evaluation block
    so a lag/window feature built at a val/test origin cannot peek into the block it is being scored on. The
    test block is STRICTLY AFTER train+val in time, so a test timestamp is never before a train timestamp --
    the invariant tests/test_timeseries.py pins.

    Defaults (when n_val/n_test are None): n_test = max(1, 20% of N), n_val = max(1, 20% of N). The embargo
    eats into the gaps, not the blocks. Raises if there are not enough rows for a non-empty train block after
    carving val/test/embargoes -- a forecasting task with too little history must fail loudly, not silently
    return an empty train set.
    """
    rows = list(series_rows)
    N = len(rows)
    if N < 4:
        raise ValueError(f"need >= 4 ordered rows for a forward-chaining split, got {N}")
    horizon = max(1, int(horizon))
    embargo = max(0, int(embargo))
    n_test = int(n_test) if n_test is not None else max(1, int(round(0.20 * N)))
    n_val = int(n_val) if n_val is not None else max(1, int(round(0.20 * N)))
    if n_test < 1 or n_val < 1:
        raise ValueError(f"n_val ({n_val}) and n_test ({n_test}) must each be >= 1")

    # carve from the END backwards: test is the last n_test rows; an embargo gap; then val; an embargo gap;
    # then train is everything before. This guarantees test is strictly the most-recent block.
    test_lo = N - n_test
    val_hi = test_lo - embargo                       # rows [val_hi, test_lo) are embargoed (dropped)
    val_lo = val_hi - n_val
    train_hi = val_lo - embargo                       # rows [train_hi, val_lo) are embargoed (dropped)
    if val_lo < 0 or train_hi < 1:
        raise ValueError(
            f"not enough history: N={N}, n_val={n_val}, n_test={n_test}, embargo={embargo} leaves "
            f"train_hi={train_hi} (need >= 1 train row). Reduce n_val/n_test/embargo or supply more rows.")
    train = rows[:train_hi]
    val = rows[val_lo:val_hi]
    test = rows[test_lo:N]
    info = {
        "N": N, "horizon": horizon, "embargo": embargo,
        "idx": {"train": [0, train_hi], "val": [val_lo, val_hi], "test": [test_lo, N]},
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "embargoed": (val_lo - train_hi) + (test_lo - val_hi),
        # the leakage invariant, materialized so a caller can assert it: max train index < min test index.
        "max_train_idx": train_hi - 1, "min_val_idx": val_lo, "min_test_idx": test_lo,
        "no_future_leak": bool((train_hi - 1) < val_lo <= val_hi - 1 < test_lo),
    }
    return train, val, test, info


# ============================================================================ 2. lag/window featurizer
class LagWindowFeaturizer:
    """Causal lag + rolling-window featurizer for a univariate (optionally exogenous) series -> a supervised
    (X, y) at each forecast origin.

    For target value at index t (predicting y[t] from the past at horizon h), the features are built ONLY from
    information available at t - h: lags y[t-h], y[t-h-1], ... and rolling mean/std over a trailing window
    ending at t-h. This is CAUSAL by construction -- no feature ever reads y[t] or anything after the origin,
    so the featurizer cannot leak the value it is predicting.

    Rolling statistics that are derived from the TARGET are fit on TRAIN ONLY semantics by construction here
    too: every feature is a function of strictly-past target values (a windowed transform), never a global
    train+val+test statistic, so there is no fit/transform leakage to guard separately -- the window is
    re-derived locally at each origin from that origin's own past.

    Parameters
    ----------
    lags : iterable[int]   positive lag offsets (in steps) to include, e.g. (1, 2, 3, 7).
    windows : iterable[int] trailing-window lengths for rolling mean/std (e.g. (3, 7)). Empty -> none.
    horizon : int          forecast horizon h; features use info no later than t-h.
    target_key, exog_keys  row keys: the target series and optional exogenous LAGGED covariates.
    """

    def __init__(self, *, lags=(1, 2, 3), windows=(3,), horizon=1, target_key="target", exog_keys=()):
        self.lags = tuple(int(l) for l in lags if int(l) >= 1)
        if not self.lags:
            raise ValueError("LagWindowFeaturizer needs at least one lag >= 1")
        self.windows = tuple(int(w) for w in windows if int(w) >= 1)
        self.horizon = max(1, int(horizon))
        self.target_key = target_key
        self.exog_keys = tuple(exog_keys)

    @property
    def max_lookback(self):
        """How many past steps the largest feature needs (so a caller can size the warm-up region)."""
        h = self.horizon
        return max([h + max(self.lags) - 1] + [h + w - 1 for w in self.windows] or [h])

    def feature_names(self):
        names = [f"lag{l}" for l in self.lags]
        for w in self.windows:
            names += [f"rmean{w}", f"rstd{w}"]
        for k in self.exog_keys:
            names += [f"exog_{k}_lag{self.horizon}"]
        return names

    def transform(self, series_rows, *, history=None):
        """Build (X, y, origins) for every index where a full causal feature vector exists.

        `series_rows` are the rows to PRODUCE targets for, in temporal order. `history` (optional) is the
        contiguous block of rows IMMEDIATELY BEFORE series_rows[0] (e.g. the train block when transforming
        val/test) so the first few origins still have their lags available WITHOUT reading any row inside the
        evaluation block from the future. If history is None, the warm-up rows at the start of series_rows are
        simply skipped (no fabricated lags). Returns numpy (X[n, d], y[n], origins[n]) where origins are
        indices into series_rows of the produced targets.
        """
        hist = list(history or [])
        cur = list(series_rows)
        full = hist + cur
        offset = len(hist)
        y_full = np.array([float(r.get(self.target_key)) for r in full], dtype=np.float64)
        h = self.horizon
        X_rows, y_rows, origins = [], [], []
        for j in range(len(cur)):
            t = offset + j                                  # absolute index in `full`
            origin = t - h                                  # last index whose value is usable as a feature
            need = origin - (max(self.lags) - 1)            # earliest index the largest lag/window touches
            if origin < 0 or need < 0:
                continue                                     # not enough causal history -> skip (no fabrication)
            feats = [y_full[origin - (l - 1)] for l in self.lags]
            ok = True
            for w in self.windows:
                lo = origin - (w - 1)
                if lo < 0:
                    ok = False
                    break
                window = y_full[lo:origin + 1]
                feats.append(float(np.mean(window)))
                feats.append(float(np.std(window)))
            if not ok:
                continue
            for k in self.exog_keys:
                xv = full[origin].get(k)
                feats.append(float(xv) if xv is not None else 0.0)
            X_rows.append(feats)
            y_rows.append(float(y_full[t]))
            origins.append(j)
        if not X_rows:
            return (np.empty((0, len(self.feature_names())), dtype=np.float64),
                    np.empty((0,), dtype=np.float64), np.empty((0,), dtype=int))
        return np.asarray(X_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64), np.asarray(origins, dtype=int)


# ============================================================================ 3. honest model menu
class SeasonalNaive:
    """Last-value (period=1) or seasonal-naive (period=s) baseline: predict y_hat[t] = y[t - h - (period-1)],
    i.e. the most recent observed value at the same phase. Implemented over the FEATURE matrix so it plugs
    into the same fit/predict contract as the sklearn models: it just reads the `lag{period}` column. With
    period=1 (the default) it is the persistence / last-value baseline -- the honest floor a real forecaster
    must beat."""

    def __init__(self, *, period=1, lags=(1, 2, 3)):
        self.period = max(1, int(period))
        self._col = None
        # the index of the lag column equal to `period` (persistence reads lag1); fall back to column 0.
        lags = tuple(int(l) for l in lags)
        self._col = lags.index(self.period) if self.period in lags else 0

    def fit(self, X, y):
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] == 0:
            return np.zeros(X.shape[0], dtype=np.float64)
        c = min(self._col, X.shape[1] - 1)
        return X[:, c].copy()


def build_models(*, lags=(1, 2, 3), seed=0):
    """The honest menu: {name -> ctor()} . Small and defensible:
      * seasonal_naive  -- last-value persistence baseline (the floor).
      * ridge_lags      -- linear (Ridge) on lag/window features (the workhorse AR-style model).
      * hgb_lags        -- HistGradientBoostingRegressor on the same features (nonlinear, captures interactions).
    Tree/linear hyperparameters are conventional defaults, NOT tuned to any target answer. sklearn is imported
    lazily so importing this module never requires it until a model is actually built."""
    def _ridge():
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)

    def _hgb():
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=200, random_state=seed)

    return {
        "seasonal_naive": lambda: SeasonalNaive(period=1, lags=lags),
        "ridge_lags": _ridge,
        "hgb_lags": _hgb,
    }


# ============================================================================ 4. block-bootstrap certifier
# Finite-sample conservative tail shrink, the same KIND of knob science._REG_FS_SHRINK uses (alpha_eff =
# alpha * SHRINK), but tuned to the BLOCK bootstrap's measured coverage rather than reused verbatim. The
# percentile bootstrap of a sqrt-of-mean-squared-error (rmse) is right-skewed and median-biased even in the
# block case, so its naive (1-alpha) lower endpoint under-covers at finite n. CRITICALLY, at strong positive
# autocorrelation a LONGER BLOCK ALONE PLATEAUS and does NOT recover coverage (MEASURED here: phi=0.9, n=1200,
# raising the block-length multiplier c from 2 to 4 moved coverage 0.925 -> 0.910, i.e. the wrong way -- the
# residual undercoverage is the percentile SKEW bias, not too-short a block; this matches the DESIGN doc's
# "longer block alone plateaus" finding). So the skew must be paid by the tail shrink. We take the percentile
# at alpha_eff = alpha * 0.4 (a 98% one-sided target for alpha=0.05). MEASURED one-sided coverage (neg_rmse,
# AR(1), B=250-300, 200-300 trials/cell, true_metric = -sigma in closed form): with SHRINK=0.4 the worst cell
# (phi=0.9) reaches 0.936-0.945 at n=2000, phi=0.6 0.96, phi=0.3 0.95, and white phi=0 0.96 -- all >= 0.93 with
# margin, while the naive i.i.d. (block_size=1) bound under-covers at 0.66-0.71 at phi=0.9 (see
# tests/test_timeseries.py). SHRINK=0.5 left phi=0.9 borderline at 0.925-0.933 (below the bar in some seeds), so
# 0.4 is the documented value. White-noise power is preserved (0.96, not over-conservatized into never-
# certifying). Setting fs_shrink=1.0 recovers the raw percentile bound for reproducing pre-shrink numbers.
_TS_FS_SHRINK = 0.4

# Minimum effective number of independent blocks (n / block_size) required to emit a bound. Below this the
# block bootstrap cannot back its lower percentile (the measured 0.834 limit at phi=0.9 / n=300 / eff~16 -- and
# worse at smaller n). When eff_blocks < this floor the certifier DEFERS: deferred=True, certified=False, with
# an honest reason -- the exact precedent of science._joint_leak_verdict deferring below min_n. Deferring can
# only WITHHOLD a certificate, never grant one, so it is the safe failure direction. 10 is the standard rule of
# thumb for the minimum number of blocks a block bootstrap needs for a usable variance estimate (Lahiri 2003);
# it is NOT tuned to any target answer.
_TS_MIN_EFF_BLOCKS = 10


def estimate_block_size(residuals, *, n=None, min_eff_blocks=_TS_MIN_EFF_BLOCKS, c=2.0):
    """Data-driven block length from the residual autocorrelation, NEVER a number tuned to any answer.

    Estimate the integrated autocorrelation time tau_int = 1 + 2*sum_{k>=1} rho_k, truncating the sum at the
    first lag k where |rho_k| falls below the noise floor 2/sqrt(n) (standard automatic windowing). Set
    L = clip(ceil(c * tau_int), L_min, L_max) with:
      * c = 2 (measured: L ~ 2*tau lifted phi=0.6 coverage to 0.894 vs 0.85 i.i.d. in the design doc),
      * L_min = ceil(n^{1/3}) (the standard moving-block-bootstrap rate, Hall-Horowitz-Jing -- so even white
        residuals get a valid block), and
      * L_max = floor(n / 2) (a STABILITY cap only -- a block longer than half the series is degenerate).
    NOTE: L_max is deliberately NOT floor(n/min_eff_blocks). Clamping L down to preserve min_eff_blocks would
    DEFEAT the defer guard in block_bootstrap_lower -- it would force eff_blocks = n//L up to min_eff_blocks
    exactly even when the natural (autocorrelation-driven) block length implies far fewer independent blocks,
    so the certifier would emit an anti-conservative bound precisely in the regime it should DEFER. We instead
    let L grow to ~c*tau_int and let block_bootstrap_lower DEFER honestly when n//L is too small (the
    measured 0.834-coverage limit at phi=0.9/n=300). `min_eff_blocks` is accepted for signature symmetry but
    is intentionally not used to clamp L here.
    For white residuals tau_int ~ 1 so L collapses to the n^{1/3} floor, and the block bootstrap reduces toward
    the i.i.d. bound -- equality (no cost) at true independence.
    """
    r = np.asarray(residuals, dtype=np.float64)
    n = int(n if n is not None else r.size)
    if n < 2:
        return 1
    r = r - r.mean()
    denom = float(np.dot(r, r))
    floor = 2.0 / math.sqrt(n)
    tau = 1.0
    if denom > 1e-30:
        kmax = min(n - 1, max(1, n // 4))                # never sum past n/4 lags (estimates get too noisy)
        for k in range(1, kmax + 1):
            rho_k = float(np.dot(r[:-k], r[k:]) / denom)
            if abs(rho_k) < floor:                       # automatic-windowing truncation at the noise floor
                break
            tau += 2.0 * rho_k
    tau = max(1.0, tau)
    l_min = max(1, int(math.ceil(n ** (1.0 / 3.0))))
    l_max = max(l_min, int(n // 2))                  # stability cap only (NOT n//min_eff_blocks; see docstring)
    L = int(math.ceil(c * tau))
    return int(min(max(L, l_min), l_max))


def block_bootstrap_lower(y_true, y_pred, metric, *, block_size=None, alpha=0.05, B=2000, seed=0,
                          fs_shrink=_TS_FS_SHRINK, min_eff_blocks=_TS_MIN_EFF_BLOCKS):
    """One-sided (>= 1-alpha) LOWER confidence bound on a HIGHER-IS-BETTER forecast metric via the
    MOVING-BLOCK / CIRCULAR-BLOCK bootstrap -- the time-series analogue of science.bootstrap_metric_lower.

    METHOD (Kunsch 1989 moving block; Politis-Romano 1992 circular variant to remove tail-undercoverage of the
    last block). Given n rows in TEMPORAL ORDER, define overlapping contiguous blocks B_i = indices
    [i, i+1, ..., i+L-1] taken MOD n (circular, so every row starts an equal number of blocks). Draw
    ceil(n/L) blocks i.i.d. with replacement, concatenate, truncate to length n, and recompute the FULL metric
    on the resampled (y_true, y_pred) PAIR-series. Repeat B times; return the alpha_eff = alpha*fs_shrink lower
    percentile.

    We resample PAIRS (not a residual vector) for the SAME reason science.bootstrap_metric_lower documents:
    r2's SS_tot and rmse's sqrt are sample-level nonlinear functions, so a per-row-error bootstrap would bound
    the wrong quantity. Resampling whole contiguous blocks PRESERVES the within-block autocorrelation of the
    error series, so the bootstrap variance captures the lag-1..L autocovariances and inflates toward the true
    tau_int * sigma^2 / n -- the distribution is WIDER and its lower percentile LOWER (more conservative) than
    the i.i.d. bootstrap, which destroys the dependence and is anti-conservative under positive autocorrelation.

    block_size=1 reduces EXACTLY to an i.i.d. pair bootstrap (each "block" is one row), so this is a strict
    generalization of the frozen i.i.d. bound. If block_size is None it is chosen data-drivenly from the
    residual autocorrelation (see estimate_block_size).

    DEFER GUARD (the honesty backstop). The effective number of independent blocks is eff_blocks = n // L. If
    eff_blocks < min_eff_blocks the block bootstrap cannot reliably back its lower percentile (the measured
    0.834-coverage limit at phi=0.9 / n=300 in DESIGN_timeseries_serving.md, worse at smaller n), so we DEFER:
    deferred=True and lower is set to the POINT estimate (so any downstream `lower > theta` is decided on the
    raw point, never on an unbacked optimistic bound -- but `deferred` makes certify_timeseries refuse). This
    mirrors science._joint_leak_verdict deferring below min_n; deferring only withholds, never grants.

    Returns a dict: {lower, point, block_size, n, n_blocks, eff_blocks, deferred, reason}.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    n = yt.size
    point = float(science.score_regression_metric(metric, yt, yp))
    if n <= 1:
        return {"lower": round(point, 6), "point": round(point, 6), "block_size": 1, "n": int(n),
                "n_blocks": 0, "eff_blocks": 0, "deferred": True,
                "reason": "too few points (n<=1) to bootstrap a temporal bound"}

    resid = yt - yp
    L = int(block_size) if block_size is not None else estimate_block_size(
        resid, n=n, min_eff_blocks=min_eff_blocks)
    L = max(1, min(L, n))
    eff_blocks = n // L
    n_blocks = int(math.ceil(n / L))                      # blocks drawn per bootstrap replicate

    # DEFER if too few effective blocks to back the bound (block_size=1 is the i.i.d. case and never defers on
    # this guard -- eff_blocks == n there -- so the white-residual path is unaffected).
    if L > 1 and eff_blocks < min_eff_blocks:
        return {"lower": round(point, 6), "point": round(point, 6), "block_size": int(L), "n": int(n),
                "n_blocks": int(n_blocks), "eff_blocks": int(eff_blocks), "deferred": True,
                "reason": (f"effective blocks n//L = {eff_blocks} < {min_eff_blocks}: too few independent "
                           f"blocks to back a conservative bound under autocorrelation (defer, do not "
                           f"emit an anti-conservative number)")}

    rng = np.random.default_rng(seed)
    # precompute circular block START indices; each draw picks n_blocks starts, expands to L-length blocks.
    arange_L = np.arange(L)
    boots = np.empty(B, dtype=np.float64)
    for b in range(B):
        starts = rng.integers(0, n, n_blocks)                          # circular: any start is legal
        idx = (starts[:, None] + arange_L[None, :]).reshape(-1) % n     # [n_blocks*L] circular indices
        idx = idx[:n]                                                    # truncate to the original length
        boots[b] = science.score_regression_metric(metric, yt[idx], yp[idx])
    alpha_eff = max(0.0, min(alpha, alpha * fs_shrink))
    lower = float(np.percentile(boots, 100.0 * alpha_eff))
    return {"lower": round(lower, 6), "point": round(point, 6), "block_size": int(L), "n": int(n),
            "n_blocks": int(n_blocks), "eff_blocks": int(eff_blocks), "deferred": False,
            "reason": "moving-block (circular) bootstrap lower percentile with finite-sample shrink"}


def certify_timeseries(y_true, y_pred, metric, theta, *, checks=1, alpha=0.05, B=2000, seed=0,
                       block_size=None, min_eff_blocks=_TS_MIN_EFF_BLOCKS):
    """Canonical FORECASTING certifier, mirroring science.certify_regression's contract and dict shape.

    Certified iff the BLOCK-bootstrap lower confidence bound on the (higher-is-better) metric clears theta,
    after a Bonferroni correction for `checks` locked-test peeks (alpha/checks), AND the bound is NOT deferred
    (too few effective blocks). The metric is recomputed per block-bootstrap draw. theta is an R^2 floor for
    r2, or a NEGATED-error floor for neg_rmse/neg_mae (theta=-3.0 means "rmse must be provably below 3.0").
    The locked (future) test block is scored exactly ONCE by the caller; this function only consumes the
    aligned (y_true, y_pred) of that block.

    A DEFERRED bound never certifies (certified=False) and says so honestly in `reason` -- the block bootstrap
    refuses to emit a bound it cannot back, rather than silently returning an anti-conservative number.
    """
    checks = max(1, int(checks))
    a = alpha / checks
    n = len(y_true)
    bb = block_bootstrap_lower(y_true, y_pred, metric, block_size=block_size, alpha=a, B=B, seed=seed,
                               min_eff_blocks=min_eff_blocks)
    lower, point, deferred = bb["lower"], bb["point"], bb["deferred"]
    certified = bool((not deferred) and lower > theta)
    if deferred:
        reason = "DEFERRED: " + bb["reason"] + " -- refusing to certify on an unbacked bound"
    elif certified:
        reason = "block-bootstrap lower bound clears theta after multiplicity correction"
    else:
        reason = "block-bootstrap lower confidence bound does not clear theta after paying for all peeks"
    return {"observed": round(float(point), 4), "n": int(n), "metric": metric,
            "theta": round(float(theta), 4), "checks": checks, "alpha_per_check": round(a, 6),
            "lower_bound": round(float(lower), 4), "certified": certified,
            "block_size": int(bb["block_size"]), "n_blocks": int(bb["n_blocks"]),
            "eff_blocks": int(bb["eff_blocks"]), "deferred": bool(deferred), "reason": reason}


# ============================================================================ 6. entry point
def _emit(on_event, stage, status="active", **detail):
    """Live stage event in the SAME shape vfplatform/loop.py uses, so the UI/observer can show the
    forecasting run. Best-effort: a failing observer never breaks the run."""
    if on_event is None:
        return
    try:
        on_event({"stage": stage, "status": status, **detail})
    except Exception:  # noqa: BLE001
        pass


def run_timeseries_goal(series_rows, *, horizon=1, threshold, metric="neg_rmse",
                        lags=(1, 2, 3, 7), windows=(3, 7), n_val=None, n_test=None, embargo=None,
                        exog_keys=(), target_key="target", alpha=0.05, B=2000, seed=0,
                        min_eff_blocks=_TS_MIN_EFF_BLOCKS, on_event=None):
    """Forecasting /goal entry point: split -> featurize -> fit the honest menu -> select the best on the
    VALIDATION block by the metric LOWER BOUND -> certify the winner ONCE on the held-out FUTURE test block
    via the block bootstrap -> return a science.certify_regression-shaped certificate plus winner info.

    Discipline (mirrors loop.run_goal_loop):
      * NO SHUFFLING: forward-chaining split with an embargo (>= horizon-1 by default) so the test block is
        strictly the most-recent rows and no future leaks into train.
      * SELECT ON VALIDATION ONLY, by the block-bootstrap LOWER BOUND (not the point) -- the same "raise the
        lower bound" objective the classification/regression loop uses.
      * SINGLE PEEK on the future test block: it is scored exactly once, for the validation winner only.
      * The certifier may DEFER (refuse) when the test block has too few effective blocks -- honest, never an
        anti-conservative number.

    Returns the certificate dict (observed, n, metric, theta, checks, lower_bound, certified, reason, plus
    block_size/eff_blocks/deferred) extended with winner info (winner, winner_val_lb, val_leaderboard,
    split_info, decision).
    """
    assert_forecast_metric(metric)
    horizon = max(1, int(horizon))
    if embargo is None:
        embargo = max(0, horizon - 1)                     # default embargo just covers the horizon overlap
    rows = list(series_rows)

    # ---- SPLIT (forward-chaining, embargoed) ---------------------------------------------------
    train, val, test, split_info = make_forecast_splits(
        rows, horizon=horizon, n_val=n_val, n_test=n_test, embargo=embargo)
    _emit(on_event, "split", n_train=len(train), n_val=len(val), n_test=len(test),
          no_future_leak=split_info["no_future_leak"], embargoed=split_info["embargoed"])

    # ---- FEATURIZE (causal lag/window; history-warmed so val/test origins keep their lags) ------
    feat = LagWindowFeaturizer(lags=lags, windows=windows, horizon=horizon, target_key=target_key,
                               exog_keys=exog_keys)
    Xtr, ytr, _otr = feat.transform(train)
    Xva, yva, _ova = feat.transform(val, history=train)            # warm val with the train tail (causal)
    Xte, yte, _ote = feat.transform(test, history=train + val)     # warm test with everything strictly before
    if Xtr.shape[0] < 2 or Xva.shape[0] < 1 or Xte.shape[0] < 1:
        return {"observed": None, "n": int(Xte.shape[0]), "metric": metric,
                "theta": round(float(threshold), 4), "checks": 1, "lower_bound": None,
                "certified": False, "deferred": True,
                "reason": (f"insufficient supervised examples after causal featurization "
                           f"(train={Xtr.shape[0]}, val={Xva.shape[0]}, test={Xte.shape[0]}); "
                           f"reduce lags/windows/horizon or supply more history"),
                "winner": None, "val_leaderboard": [], "split_info": split_info,
                "decision": "insufficient_data"}

    # ---- FANOUT: fit the honest menu on train ---------------------------------------------------
    models = build_models(lags=feat.lags, seed=seed)
    _emit(on_event, "fanout", n_candidates=len(models), families=sorted(models))
    fitted, val_scores = {}, {}
    for name, ctor in models.items():
        try:
            est = ctor()
            est.fit(Xtr, ytr)
            yhat_va = np.asarray(est.predict(Xva), dtype=np.float64)
            if yhat_va.shape[0] != yva.shape[0] or not np.all(np.isfinite(yhat_va)):
                continue
            fitted[name] = est
            val_scores[name] = float(science.score_regression_metric(metric, yva, yhat_va))
        except Exception:  # noqa: BLE001  a failing family is skipped, not fatal (mirrors loop fit retry/skip)
            continue
    if not fitted:
        return {"observed": None, "n": int(Xte.shape[0]), "metric": metric,
                "theta": round(float(threshold), 4), "checks": 1, "lower_bound": None,
                "certified": False, "deferred": True,
                "reason": "no forecasting model fit successfully", "winner": None,
                "val_leaderboard": [], "split_info": split_info, "decision": "no_model"}
    _emit(on_event, "measure", val_scores={k: round(v, 4) for k, v in val_scores.items()})

    # ---- SELECT ON VALIDATION by the LOWER BOUND (Bonferroni over the candidates we select among) ---
    n_cand = len(fitted)
    val_lb = {}
    for name, est in fitted.items():
        yhat_va = np.asarray(est.predict(Xva), dtype=np.float64)
        bb = block_bootstrap_lower(yva, yhat_va, metric, alpha=alpha / max(1, n_cand), B=B,
                                   seed=seed + 1, min_eff_blocks=min_eff_blocks)
        # a deferred val bound is uninformative for ranking; fall back to the point so the winner is still the
        # best-on-validation model (selection never certifies, so this is safe -- the sealed certify is what
        # gates promotion). Record deferral for transparency.
        val_lb[name] = {"lower": bb["lower"], "point": val_scores[name], "deferred": bb["deferred"],
                        "block_size": bb["block_size"], "eff_blocks": bb["eff_blocks"]}
    ranked = sorted(val_lb, key=lambda nm: (val_lb[nm]["lower"], val_lb[nm]["point"]), reverse=True)
    winner = ranked[0]
    leaderboard = [{"model": nm, **val_lb[nm]} for nm in ranked]
    _emit(on_event, "val_bound", winner=winner, val_lb=val_lb[winner]["lower"],
          val_point=val_lb[winner]["point"], block_size=val_lb[winner]["block_size"])

    # ---- CERTIFY the validation winner ONCE on the held-out FUTURE test block -------------------
    est = fitted[winner]
    yhat_te = np.asarray(est.predict(Xte), dtype=np.float64)        # the single peek of the future block
    _emit(on_event, "sealed_certify", winner=winner, n_test=int(yte.shape[0]))
    # block_size=None => the certifier derives the block length data-drivenly from the TEST-block residual
    # autocorrelation (estimate_block_size), so the bound matches the autocorrelation of the block it certifies.
    cert = certify_timeseries(yte, yhat_te, metric, threshold, checks=1, alpha=alpha, B=B,
                              seed=seed + 2, block_size=None, min_eff_blocks=min_eff_blocks)
    cert["winner"] = winner
    cert["winner_val_lb"] = val_lb[winner]["lower"]
    cert["winner_val_point"] = val_lb[winner]["point"]
    cert["val_leaderboard"] = leaderboard
    cert["split_info"] = split_info
    cert["horizon"] = horizon
    cert["embargo"] = int(embargo)
    cert["lags"] = list(feat.lags)
    cert["windows"] = list(feat.windows)
    cert["decision"] = ("certified" if cert["certified"]
                        else ("deferred" if cert.get("deferred") else "do_not_certify"))
    _emit(on_event, "done", status="done", decision=cert["decision"], certified=bool(cert["certified"]),
          lower_bound=cert["lower_bound"], observed=cert["observed"], n_test=int(yte.shape[0]),
          deferred=bool(cert.get("deferred")))
    return cert
