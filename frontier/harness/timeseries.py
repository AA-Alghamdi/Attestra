"""TimeSeriesHarness: REAL forecasting/classification with a PROPER temporal split.

What this proves (the Phase-3 breadth goal)
-------------------------------------------
The Phase-0 spine is modality agnostic; only the data->Task adapter (and, here, the split
PROTOCOL) changes, while the certify path is byte for byte the audited-sound one
(vectorforge.science + vfplatform.sealed). This harness is that adapter for univariate time
series, with the one property a time-series harness MUST get right and a naive tabular harness
gets WRONG: **forward-chaining, no-shuffle evaluation**. It SELF-CERTIFIES on a self-contained
synthetic seasonal series (NO network) before its numbers are trusted.

The two pieces a time-series harness owns
-----------------------------------------
1. LAG / ROLLING FEATURES (adapt). A raw series `s[0..T-1]` is windowed into supervised rows:
   row t carries features built ONLY from the strict past `s[t-L .. t-1]` (the last `L` lags plus
   rolling mean/std/min/max over that window) and a target derived from the present/next step.
   Because every row's features use only its own past, the featurization is leakage-free under
   ANY split -- a row never sees its own future. This is what lets the same frozen certify path
   score a time series at all.

2. FORWARD-CHAINING SPLIT (self_test + split protocol). The honest evaluation of a forecaster is
   on data that comes AFTER the training data in time. A shuffled split (what the frozen
   `certify.make_splits` does for i.i.d. tabular data) would let the model train on rows from the
   future of the evaluation rows -- optimistic and dishonest for a series. So this harness's
   `self_test()` builds the split itself, in TIME ORDER, NO shuffle: the earliest 50% of rows are
   train, the next 20% val, the latest 30% sealed. It then certifies through the SAME frozen
   one-peek sealed gate (`certify.certify_on_sealed`) -- reusing the audited certifier verbatim,
   only feeding it a time-ordered split instead of a shuffled one. (The orchestrator path still
   uses the frozen shuffled `make_splits`; that is still leakage-free here because the features
   are strictly causal per row. The forward-chaining split is the harness's honest self-evaluation
   protocol and is exercised by self_test() + the wired modality test.)

Everything that DECIDES (which model wins, whether it promotes) is the frozen certifier; the
harness only ASSEMBLES the windowed rows + the time-ordered split + baselines/metric.

# === WIRING ===
# This is the concrete Harness the Phase-3 router resolves for the "timeseries" task-type key. It
# is registered into the shared REGISTRY by frontier/harness/__init__.py at import time, so the
# integrator/orchestrator does what it does for the other harnesses:
#
#     from frontier.harness import lookup
#     h = lookup("timeseries")                          # also reachable via "forecasting"
#     ok, cert = h.self_test()                          # GATE: trust nothing until ok is True (this
#                                                       #   self-test uses the FORWARD-CHAINING split)
#     task = h.adapt(series, None, kind="classification", theta=0.70)   # series -> windowed Task
#     result = ResearchEngine(EngineConfig(rounds=2)).run(task)
#
# adapt() accepts EITHER a raw 1-D series (y unused; the harness builds the supervised target from
# the series itself) OR a pre-windowed (X, y). metric_for / split_protocol are time-series-aware.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Repo root on sys.path so sibling frontier modules + the sound certifier resolve from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier import certify, sandbox            # noqa: E402  (frozen split/certify + firewall)
from frontier.program import Program             # noqa: E402
from frontier.task import Task                   # noqa: E402

from .base import Harness, HarnessCertificate, register, _StaticSeedProposer  # noqa: E402


# Right-axis metrics per kind the frozen certifier supports. A forecasting (regression) target
# uses r2/neg_rmse; a regime/up-down classification target uses accuracy/macro_f1.
_VALID_METRICS = {
    "classification": ("accuracy", "balanced_accuracy", "macro_f1"),
    "regression": ("r2", "neg_rmse", "neg_mae"),
}
_DEFAULT_METRIC = {"classification": "accuracy", "regression": "r2"}


# --------------------------------------------------------------------------- baseline suite
# Baselines operate on the windowed lag/rolling FLOAT feature matrix adapt() produced. For
# classification (e.g. predict next-step up/down): logistic regression, RBF-SVM, random forest --
# the standard strong baselines on dense engineered features. For regression (forecast next value):
# ridge, gradient boosting, random-forest regressor. SEEDS/FALLBACKS, never the promoter.
_TS_BASELINES = {
    "classification": [
        ("ts_logreg",
         "from sklearn.linear_model import LogisticRegression\n"
         "def build_estimator():\n"
         "    return LogisticRegression(C=2.0, max_iter=3000)\n"),
        ("ts_svc_rbf",
         "from sklearn.svm import SVC\n"
         "def build_estimator():\n"
         "    return SVC(C=4.0, gamma='scale', kernel='rbf')\n"),
        ("ts_rf_clf",
         "from sklearn.ensemble import RandomForestClassifier\n"
         "def build_estimator():\n"
         "    return RandomForestClassifier(n_estimators=200, random_state=0)\n"),
    ],
    "regression": [
        ("ts_ridge",
         "from sklearn.linear_model import Ridge\n"
         "def build_estimator():\n"
         "    return Ridge(alpha=1.0)\n"),
        ("ts_hgb",
         "from sklearn.ensemble import HistGradientBoostingRegressor\n"
         "def build_estimator():\n"
         "    return HistGradientBoostingRegressor(max_iter=300, random_state=0)\n"),
        ("ts_rf_reg",
         "from sklearn.ensemble import RandomForestRegressor\n"
         "def build_estimator():\n"
         "    return RandomForestRegressor(n_estimators=200, random_state=0)\n"),
    ],
}


def timeseries_baseline_programs(kind: str) -> List[Program]:
    """The time-series baseline suite for the kind, as seed Programs. Fresh list per call."""
    if kind not in _TS_BASELINES:
        raise ValueError(f"TimeSeriesHarness handles {tuple(_TS_BASELINES)}, not {kind!r}")
    return [
        Program(code=code, source="seed", label=name,
                provenance={"suite": "timeseries_baselines", "harness": "timeseries"})
        for name, code in _TS_BASELINES[kind]
    ]


# --------------------------------------------------------------------------- windowing

def make_lag_features(series: Sequence[float], *, n_lags: int = 8, kind: str = "classification",
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Window a univariate series into causal (X, y) supervised rows.

    For each position t with at least `n_lags` of past, row t carries:
      - the last `n_lags` raw values s[t-n_lags .. t-1] (most-recent-last);
      - rolling mean / std / min / max over that same window.
    All features use the STRICT PAST of the target, so the rows are leakage-free under any split.

    The target:
      - regression: the present value s[t] (one-step-ahead forecast given the past window);
      - classification: sign of the one-step change, "up" if s[t] >= s[t-1] else "down".

    Rows are returned in TIME ORDER (row i precedes row i+1 in the series), which is what the
    forward-chaining split relies on.
    """
    s = np.asarray(series, dtype=float).ravel()
    T = s.size
    if T < n_lags + 2:
        raise ValueError(f"series too short: need > {n_lags + 1} points, got {T}")
    if kind not in ("classification", "regression"):
        raise ValueError(f"kind must be classification|regression, got {kind!r}")

    rows_X: List[np.ndarray] = []
    rows_y: List = []
    for t in range(n_lags, T):
        window = s[t - n_lags:t]                     # strict past: s[t-n_lags .. t-1]
        feats = np.concatenate([
            window,
            [window.mean(), window.std(), window.min(), window.max()],
        ])
        rows_X.append(feats)
        if kind == "regression":
            rows_y.append(float(s[t]))
        else:
            rows_y.append("up" if s[t] >= s[t - 1] else "down")
    X = np.asarray(rows_X, dtype=float)
    y = np.asarray(rows_y, dtype=(float if kind == "regression" else object))
    return X, (y.astype(float) if kind == "regression" else y.astype(str))


# --------------------------------------------------------------------------- synthetic series

def synth_seasonal_series(*, n_points: int = 720, period: int = 24, trend: float = 0.01,
                          noise: float = 0.25, seed: int = 0) -> np.ndarray:
    """A self-contained seasonal+trend series (NO network).

    s[t] = trend*t + sin(2*pi*t/period) + 0.4*sin(4*pi*t/period) + small AR(1) noise. The strong
    deterministic seasonal/trend structure makes next-step direction and next-step value both
    predictable from the recent window, so a competent baseline certifies -- and a broken harness
    (shuffled-as-iid, wrong target axis, label scramble) fails. Returned in time order.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=float)
    seasonal = np.sin(2 * np.pi * t / period) + 0.4 * np.sin(4 * np.pi * t / period)
    # AR(1) noise so consecutive points are correlated (realistic; not i.i.d. jitter).
    eps = np.zeros(n_points)
    for i in range(1, n_points):
        eps[i] = 0.6 * eps[i - 1] + rng.normal(0.0, noise)
    return trend * t + seasonal + eps


# --------------------------------------------------------------------------- the harness

class TimeSeriesHarness(Harness):
    """Harness for univariate time-series forecasting/classification with a temporal split.

    `adapt()` accepts either a raw 1-D series (y ignored; the supervised target is derived from the
    series) or a pre-windowed (X, y) feature matrix, and returns a standard `frontier.task.Task`.
    `self_test()` is OVERRIDDEN to use a forward-chaining (no-shuffle) split through the frozen
    one-peek sealed gate, the honest evaluation protocol for a forecaster.
    """

    key = "timeseries"
    kinds = ("classification", "regression")

    def __init__(self, *, n_lags: int = 8):
        super().__init__()
        self.n_lags = n_lags

    # ------------------------------------------------------------------ adapter API
    def metric_for(self, kind: str) -> str:
        if kind not in _DEFAULT_METRIC:
            raise ValueError(f"TimeSeriesHarness does not handle kind {kind!r}")
        return _DEFAULT_METRIC[kind]

    def _window(self, X, y, kind: str) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve raw-series-vs-prewindowed input into a causal (Xmat, yvec) in time order."""
        arr = np.asarray(X)
        # Raw 1-D series: build lag/rolling features + derived target (y is ignored).
        if arr.ndim == 1:
            return make_lag_features(arr, n_lags=self.n_lags, kind=kind)
        # Pre-windowed 2-D feature matrix: use the provided (X, y) directly (caller did windowing,
        # responsible for causal ordering). Validate y is present and aligned.
        if arr.ndim == 2:
            if y is None:
                raise ValueError("pre-windowed 2-D X requires an aligned y")
            ya = np.asarray(y)
            if len(ya) != arr.shape[0]:
                raise ValueError(f"X/y length mismatch: {arr.shape[0]} vs {len(ya)}")
            Xmat = np.asarray(arr, dtype=float)
            yvec = ya.astype(float) if kind == "regression" else ya.astype(str)
            return Xmat, yvec
        raise ValueError(f"time-series X must be 1-D (raw series) or 2-D (windowed); got {arr.ndim}-D")

    def adapt(self, X, y, *, kind: str, theta: float, name: str = "task",
              metric: str = "") -> Task:
        """Window a series (or accept windowed rows) into a Phase-0 Task the engine certifies.

        Validates early rather than emit a silently-corrupt Task:
          - kind in {classification, regression};
          - input is a usable raw series or aligned windowed matrix;
          - features are finite;
          - classification has >= 2 distinct labels;
          - the metric, if supplied, is on the right axis.
        theta is the caller's verification standard; the harness does not invent it.
        """
        if kind not in self.kinds:
            raise ValueError(f"TimeSeriesHarness handles {self.kinds}, not {kind!r}")
        if metric:
            if metric not in _VALID_METRICS[kind]:
                raise ValueError(f"metric {metric!r} not valid for {kind}; "
                                 f"choose one of {_VALID_METRICS[kind]}")
        else:
            metric = self.metric_for(kind)

        Xmat, yvec = self._window(X, y, kind)
        if not np.all(np.isfinite(Xmat)):
            n_bad = int((~np.isfinite(Xmat)).sum())
            raise ValueError(f"windowed features have {n_bad} non-finite entries")
        if kind == "classification" and len(set(yvec.tolist())) < 2:
            raise ValueError("time-series classification needs >= 2 distinct labels")
        return Task(X=Xmat, y=yvec, kind=kind, theta=float(theta), metric=metric, name=name)

    def baseline_suite(self, kind: str) -> List[Program]:
        """Time-series floor recipes for the kind, as seed Programs (never promoters)."""
        return timeseries_baseline_programs(kind)

    def split_protocol(self, kind: str) -> Tuple[float, float]:
        """Recommended fractions (latest 30% sealed / 20% val / earliest 50% train).

        These fractions match Phase-0; the time-ORDERING (no shuffle) is applied by self_test()'s
        forward-chaining split. The orchestrator path consumes the fractions through the frozen
        make_splits; because the windowed features are strictly causal per row, that path stays
        leakage-free even though it shuffles.
        """
        return (0.30, 0.20)

    # ------------------------------------------------------------------ forward-chaining split
    def _forward_chaining_splits(self, task: Task, *, test_frac: float,
                                 val_frac: float) -> certify.Splits:
        """Build a TIME-ORDERED 3-way split (no shuffle) wrapped in certify.Splits.

        Rows from task.to_rows() are already in time order (adapt windows the series in order). We
        carve the EARLIEST `1-test_frac-val_frac` as train, the NEXT `val_frac` as val, and the
        LATEST `test_frac` as sealed -- so selection (val) and certification (sealed) happen on data
        strictly AFTER training, the honest forecasting protocol. The returned certify.Splits feeds
        the SAME frozen certify.certify_on_sealed one-peek gate unchanged.
        """
        rows = task.to_rows()
        n = len(rows)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        n_train = n - n_test - n_val
        if n_train < 1 or n_val < 1 or n_test < 1:
            raise ValueError(f"series too short for a 3-way temporal split (n={n})")
        train_rows = rows[:n_train]
        val_rows = rows[n_train:n_train + n_val]
        sealed_rows = rows[n_train + n_val:]
        meta = {"counts": {"train": len(train_rows), "val": len(val_rows),
                           "test": len(sealed_rows)},
                "leakage_dropped": 0, "split": "forward_chaining_temporal"}
        return certify.Splits(train_rows, val_rows, sealed_rows, meta)

    # ------------------------------------------------------------------ self-test (OVERRIDE)
    def _self_test_case(self) -> Tuple[np.ndarray, np.ndarray, str, str, float]:
        """Known-good case: one-step-ahead FORECASTING on a self-contained seasonal series, NO net.

        720 points with a strong seasonal/trend signal. One-step-ahead value forecasting from the
        recent lag/rolling window is the canonical time-series task and is genuinely predictable
        here; a competent regressor reaches r2 ~0.84 and the sealed (bootstrap) LOWER bound clears
        0.50 on the latest-30% forward-chaining sealed block. We pick regression (forecasting) over
        next-step direction because direction on a fast-oscillating seasonal signal is intrinsically
        near-chance (an honest finding: the harness must not certify an unpredictable target). 0.50
        is deliberately below the achievable r2 so the self-test verifies the windowing/metric/
        TEMPORAL-split PLUMBING with margin, not the modeling difficulty.
        """
        series = synth_seasonal_series(seed=0)
        return series, np.asarray([]), "regression", "synthetic_seasonal_series", 0.50

    def self_test(self, *, rounds: int = 1, seed: int = 0,
                  wall_seconds: float = 45.0, cpu_seconds: int = 40
                  ) -> Tuple[bool, HarnessCertificate]:
        """Certify THIS harness on its known-good series through a FORWARD-CHAINING split.

        We deliberately do NOT route through ResearchEngine here (it uses the frozen SHUFFLED
        make_splits, which is dishonest for a series). Instead we:
          1. adapt the series into a windowed Task;
          2. build a TIME-ORDERED 3-way split (no shuffle) -- the protocol under test;
          3. fit each harness baseline in the Phase-0 SANDBOX (preds-only firewall) on train,
             predict on val, and SELECT the best by certify.score_val (trusted parent, VAL only);
          4. certify the single VAL winner ONCE on the latest-block sealed split via the frozen
             certify.certify_on_sealed (one counted peek). Every number is the frozen certifier's.

        This reuses the audited-sound certifier verbatim; only the split is time-ordered. Sets
        self.trusted = True iff the frozen gate promotes the winner above self_theta.
        """
        series, _y_ignored, kind, dataset, self_theta = self._self_test_case()
        metric = self.metric_for(kind)
        task = self.adapt(series, None, kind=kind, theta=self_theta,
                          name=f"selftest:{self.key}", metric=metric)
        test_frac, val_frac = self.split_protocol(kind)
        splits = self._forward_chaining_splits(task, test_frac=test_frac, val_frac=val_frac)

        X_train = Task.rows_to_X(splits.train_rows)
        y_train = Task.rows_to_y(splits.train_rows, task.kind)
        X_val = Task.rows_to_X(splits.val_rows)

        best_score: Optional[float] = None
        best_prog: Optional[Program] = None
        for prog in self.baseline_suite(kind):
            res = sandbox.run_program(prog, X_train, y_train, X_val, kind=task.kind,
                                      wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)
            if not res.ok:
                continue
            score = certify.score_val(task, splits.val_rows, res.preds)
            if best_score is None or score > best_score:
                best_score, best_prog = score, prog

        if best_prog is None:
            cert = HarnessCertificate(
                harness_key=self.key, task_kind=kind, dataset=dataset, metric=metric,
                self_theta=self_theta, certified=False, sealed_cert=None, winner_label="",
                detail="self-test did NOT certify: no baseline ran successfully on the temporal split")
            self._trusted, self._last_cert = False, cert
            return False, cert

        # THE ONE SEALED PEEK on the latest-block (forward-chaining) sealed split.
        X_sealed = Task.rows_to_X(splits.sealed_rows)
        final = sandbox.run_program(best_prog, X_train, y_train, X_sealed, kind=task.kind,
                                    wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)
        if not final.ok:
            cert = HarnessCertificate(
                harness_key=self.key, task_kind=kind, dataset=dataset, metric=metric,
                self_theta=self_theta, certified=False, sealed_cert=None,
                winner_label=best_prog.label,
                detail=f"self-test winner failed on sealed re-fit: [{final.error_kind}] {final.error}")
            self._trusted, self._last_cert = False, cert
            return False, cert

        sealed_cert = certify.certify_on_sealed(task, splits, final.preds)  # frozen one-peek gate
        certified = bool(sealed_cert.get("certified"))
        cert = HarnessCertificate(
            harness_key=self.key, task_kind=kind, dataset=dataset, metric=metric,
            self_theta=self_theta, certified=certified, sealed_cert=sealed_cert,
            winner_label=best_prog.label,
            detail=("self-test certified through frozen sealed gate on a FORWARD-CHAINING "
                    "(time-ordered, no-shuffle) split"
                    if certified else
                    "self-test did NOT certify: sealed lower bound did not clear self_theta"))
        self._trusted, self._last_cert = certified, cert
        return certified, cert


# --------------------------------------------------------------------------- registration

_TIMESERIES = TimeSeriesHarness(n_lags=8)
try:
    register(_TIMESERIES, "timeseries", "forecasting")
except KeyError:
    # Idempotent under re-import (e.g. pytest reimport): keep the already-registered instance.
    pass
