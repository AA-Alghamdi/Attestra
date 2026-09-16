"""Certification adapter.

This module REUSES the audited-sound certifier without modifying it:
  - vectorforge.science : Clopper-Pearson exact bound, pair-bootstrap regression CI,
    score_metric, make_splits (true 3-way split with leakage dedup).
  - vfplatform.sealed   : SealedTest (enforced one-peek counter) + certify_on_sealed.

The audit verified science.clopper_pearson_lower matches scipy.stats.beta.ppf to 1e-15,
that the bootstrap correction is a conservative (interval-widening) heuristic, and that
SealedTest raises PeekViolation on a second uncounted peek. We do not re-implement any of
that. We only:
  (a) make a TRUE three-way split (train / val / sealed) -- selection touches val only;
  (b) compute the val score for selection via score_metric;
  (c) certify the SINGLE selected winner on the held-out sealed test, one counted peek.

This is the exact discipline the live vfplatform/loop.py path uses and that the PR19
attestra/cycle/engine.py path violated (it certified on the validation set).
"""

from __future__ import annotations

import sys
import os
from typing import Sequence

# Import the sound certifier from the existing trees (repo root must be on sys.path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vectorforge import science          # noqa: E402  (sound, imported verbatim)
from vfplatform import sealed             # noqa: E402  (sound, imported verbatim)

from .task import Task


class Splits:
    """A true 3-way split. `sealed` is never exposed during selection."""

    def __init__(self, train_rows, val_rows, sealed_rows, meta):
        self.train_rows = train_rows
        self.val_rows = val_rows
        self.sealed_rows = sealed_rows
        self.meta = meta

    @property
    def sealed_test(self) -> sealed.SealedTest:
        # One peek allowed without a durable ledger -> a second peek raises PeekViolation.
        return sealed.SealedTest(self.sealed_rows, target_key="target", max_peeks=1)


def make_splits(task: Task, *, seed: int = 0, test_frac: float = 0.30,
                val_frac: float = 0.20) -> Splits:
    rows = task.to_rows()
    if task.kind == "classification":
        # science.make_splits stratifies by exact target string (correct for classes) and
        # runs the audited leakage dedup. Reuse it verbatim.
        train, val, sealed_rows, meta = science.make_splits(
            rows, seed=seed, test_frac=test_frac, val_frac=val_frac,
            text_key="text", target_key="target",
        )
    else:
        # For continuous targets, per-unique-value stratification degenerates (every value
        # is its own "class"), so we stratify on target QUANTILE bins instead. The certifier
        # numbers downstream are unchanged; only the split protocol is regression-appropriate.
        train, val, sealed_rows, meta = _regression_split(rows, seed, test_frac, val_frac)
    return Splits(train, val, sealed_rows, meta)


def _regression_split(rows, seed, test_frac, val_frac):
    import numpy as np
    rng = np.random.default_rng(seed)
    y = np.asarray([float(r["target"]) for r in rows], dtype=float)
    n_bins = min(10, max(2, len(rows) // 30))
    edges = np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1))
    bins = np.clip(np.digitize(y, edges[1:-1]), 0, n_bins - 1)
    train, val, test = [], [], []
    for b in range(n_bins):
        idx = np.where(bins == b)[0]
        rng.shuffle(idx)
        n = len(idx)
        nt = int(round(test_frac * n))
        nv = int(round(val_frac * n))
        test += [rows[i] for i in idx[:nt]]
        val += [rows[i] for i in idx[nt:nt + nv]]
        train += [rows[i] for i in idx[nt + nv:]]
    meta = {"counts": {"train": len(train), "val": len(val), "test": len(test)},
            "leakage_dropped": 0, "split": "regression_quantile_stratified"}
    return train, val, test, meta


def score_val(task: Task, val_rows: Sequence, preds: Sequence) -> float:
    """Selection score on the validation split. Trusted parent computes it (firewall)."""
    y_true = Task.rows_to_y(val_rows, task.kind)
    if task.kind == "classification":
        return float(science.score_metric(task.metric, list(map(str, y_true)),
                                           list(map(str, preds)), task.labels))
    return float(science.score_metric(task.metric, [float(v) for v in y_true],
                                      [float(v) for v in preds], None))


def certify_on_sealed(task: Task, splits: Splits, sealed_preds: Sequence) -> dict:
    """Certify the winner on the held-out sealed test via the audited certify_on_sealed.

    `sealed_preds` were produced by ONE sandbox run of the winning Program on the sealed
    features (the only time sealed is touched). We wrap them in a predict_fn so the sound
    one-peek + Clopper-Pearson / bootstrap path runs unchanged.
    """
    st = splits.sealed_test
    if len(sealed_preds) != len(st.rows):
        raise ValueError(f"sealed pred count {len(sealed_preds)} != sealed size {len(st.rows)}")

    # predict_fn returns the precomputed predictions in row order; certify_on_sealed counts
    # the single peek and recomputes the certificate with the metric-correct frozen bound.
    preds_list = list(sealed_preds)
    if task.kind == "classification":
        preds_list = [str(p) for p in preds_list]

    def predict_fn(_rows):
        return preds_list

    labels = list(task.labels) if task.kind == "classification" else None
    cert = sealed.certify_on_sealed(
        st, predict_fn, task.theta, metric=task.metric, labels=labels, alpha=0.05, who="frontier",
    )
    return cert
