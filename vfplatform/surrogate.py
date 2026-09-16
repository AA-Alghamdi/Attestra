"""TIER-0.5 LEARNED VERIFIER -- measurable taste distilled from the certified+negative ledger.

========================================================================================================
STATUS (measured, 2026-06): DEMOTED / NOT WIRED into the certified researcher.
  The surrogate exists to make a large SEARCH affordable by pruning losers early. But search itself was
  measured dead (see search.py: CYCLE vs tuned GBM = 0/5 FDR). A faster way to run a search that adds
  nothing is still nothing. The autonomous researcher (vfplatform/repr_researcher.py) prunes with the
  cheap COMPETENCE screen of verification.VerificationCascade (a fixed val floor), not a learned critic,
  so this module is not on the certified path. Retained for reference only; do not claim value without a
  certified-arena measurement.
========================================================================================================

WHY THIS EXISTS
---------------
Recursion is affordable only if obvious losers are pruned before they cost an expensive evaluation. The
surrogate is a learned critic: trained on the ledger of past outcomes (fingerprint, operator) -> (was it
certified?, what gain?), it predicts "is this candidate worth executing?" so the search does not burn
compute on predictable failures. It is the Tier-0.5 rung of the verification cascade.

THE INVARIANT (D4)
------------------
The surrogate NEVER promotes. It outputs a pass-probability and an expected-gain that ALLOCATE search
(prune at Tier 0.5, rank in the portfolio). Only the frozen Tier-3 sealed certifier mints a certificate.
Because a learned critic can be wrong, its predictions are CALIBRATION-TRACKED against ground truth
(CalibrationTracker): we continuously measure how well its claimed probabilities match observed pass
rates, so its influence can be trusted only to the extent it is calibrated. A miscalibrated surrogate is
visible (high ECE) rather than silently steering the search wrong.

MEMORY COMPOUNDS
----------------
Trained from a CaseBaseStore's durable outcome ledger (from_casebase), the surrogate is how run N starts
smarter than run 1: the operators that worked on similar fingerprints get higher pass-probabilities, so
the cascade reaches the same certified bound in FEWER expensive trials. That compounding edge is the one
a chat model structurally cannot have.

CONTRACT: uses scikit-learn (already a platform dependency) for the logistic/ridge heads, with a constant
fallback when a class is degenerate. Produces no certificate. Deterministic given a seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class OutcomeRecord:
    """One row of the training ledger: an operator applied to a dataset fingerprint, and what happened."""
    fingerprint: dict
    operator: str
    certified: bool
    gain: float = 0.0


def _num(x, default=0.0) -> float:
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


class _FeatureSpace:
    """Fits a stable vocabulary over operators + categorical fingerprint buckets, then maps any
    (fingerprint, operator) to a fixed numeric vector. Unknown categories map to an 'other' slot so the
    transform never errors on a category unseen at fit time."""

    CAT_FIELDS = ("modality", "size_bucket", "feature_bucket", "n_classes_bucket", "balance_bucket")

    def __init__(self):
        self.op_vocab: List[str] = []
        self.cat_vocab: Dict[str, List[str]] = {}
        self._fitted = False

    def fit(self, records: Sequence[OutcomeRecord]) -> "_FeatureSpace":
        ops, cats = set(), {f: set() for f in self.CAT_FIELDS}
        for r in records:
            ops.add(r.operator)
            for f in self.CAT_FIELDS:
                cats[f].add(str(r.fingerprint.get(f, "na")))
        self.op_vocab = sorted(ops)
        self.cat_vocab = {f: sorted(cats[f]) for f in self.CAT_FIELDS}
        self._fitted = True
        return self

    def _onehot(self, value: str, vocab: List[str]) -> List[float]:
        vec = [0.0] * (len(vocab) + 1)            # +1 = 'other' slot
        if value in vocab:
            vec[vocab.index(value)] = 1.0
        else:
            vec[-1] = 1.0
        return vec

    def transform_one(self, fingerprint: dict, operator: str) -> List[float]:
        feats: List[float] = [
            math.log1p(_num(fingerprint.get("n_rows"))),
            math.log1p(_num(fingerprint.get("n_features"))),
            _num(fingerprint.get("n_classes")),
        ]
        feats += self._onehot(operator, self.op_vocab)
        for f in self.CAT_FIELDS:
            feats += self._onehot(str(fingerprint.get(f, "na")), self.cat_vocab[f])
        return feats

    def transform(self, records: Sequence[OutcomeRecord]) -> np.ndarray:
        return np.asarray([self.transform_one(r.fingerprint, r.operator) for r in records], dtype=float)


class CalibrationTracker:
    """Tracks (predicted_prob, observed_outcome) pairs and reports calibration. This is what keeps the
    surrogate honest: its claimed pass-probabilities are continuously checked against reality."""

    def __init__(self):
        self._pred: List[float] = []
        self._obs: List[int] = []

    def record(self, pred_prob: float, outcome: bool) -> None:
        self._pred.append(float(min(max(pred_prob, 0.0), 1.0)))
        self._obs.append(int(bool(outcome)))

    def __len__(self) -> int:
        return len(self._pred)

    def reliability_table(self, bins: int = 10) -> List[dict]:
        """Per-bin (predicted-prob bucket): mean predicted, observed frequency, count. Empty bins omitted."""
        if not self._pred:
            return []
        p = np.asarray(self._pred)
        o = np.asarray(self._obs, dtype=float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        table = []
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (p >= lo) & (p < hi if b < bins - 1 else p <= hi)
            if not mask.any():
                continue
            table.append({"bin": (round(lo, 3), round(hi, 3)), "mean_pred": round(float(p[mask].mean()), 4),
                          "observed_freq": round(float(o[mask].mean()), 4), "count": int(mask.sum())})
        return table

    def ece(self, bins: int = 10) -> Optional[float]:
        """Expected calibration error: sum_b (count_b/N) * |mean_pred_b - observed_freq_b|. None if empty."""
        if not self._pred:
            return None
        n = len(self._pred)
        return float(sum((row["count"] / n) * abs(row["mean_pred"] - row["observed_freq"])
                         for row in self.reliability_table(bins)))


class Surrogate:
    """Learned pass-probability + expected-gain predictor over (fingerprint, operator). Advisory only."""

    def __init__(self, *, seed: int = 0):
        self.seed = int(seed)
        self._space = _FeatureSpace()
        self._clf = None
        self._reg = None
        self._const_prob: Optional[float] = None
        self._const_gain: float = 0.0
        self._fitted = False
        self.calibration = CalibrationTracker()

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, records: Sequence[OutcomeRecord]) -> "Surrogate":
        recs = list(records)
        if not recs:
            raise ValueError("Surrogate.fit needs at least one OutcomeRecord")
        self._space.fit(recs)
        X = self._space.transform(recs)
        y = np.asarray([1 if r.certified else 0 for r in recs], dtype=int)
        g = np.asarray([float(r.gain) for r in recs], dtype=float)
        # pass-probability head
        if len(np.unique(y)) < 2:
            self._const_prob = float(y.mean())     # degenerate ledger -> constant base rate (honest)
        else:
            from sklearn.linear_model import LogisticRegression
            self._clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self.seed)
            self._clf.fit(X, y)
        # expected-gain head
        if np.allclose(g, g[0]):
            self._const_gain = float(g[0])
        else:
            from sklearn.linear_model import Ridge
            self._reg = Ridge(alpha=1.0, random_state=self.seed)
            self._reg.fit(X, g)
        self._fitted = True
        return self

    def predict_pass_prob(self, fingerprint: dict, operator: str) -> float:
        if not self._fitted:
            raise RuntimeError("Surrogate is not fitted")
        if self._clf is None:
            return float(self._const_prob if self._const_prob is not None else 0.5)
        x = np.asarray([self._space.transform_one(fingerprint, operator)], dtype=float)
        return float(self._clf.predict_proba(x)[0, 1])

    def predict_gain(self, fingerprint: dict, operator: str) -> float:
        if not self._fitted:
            raise RuntimeError("Surrogate is not fitted")
        if self._reg is None:
            return float(self._const_gain)
        x = np.asarray([self._space.transform_one(fingerprint, operator)], dtype=float)
        return float(self._reg.predict(x)[0])

    def score_and_track(self, fingerprint: dict, operator: str, *, actual_certified: bool) -> float:
        """Predict the pass-prob AND log it against the realized outcome for calibration tracking. Use this
        in the live loop so the surrogate's calibration is measured on exactly the candidates it scored."""
        p = self.predict_pass_prob(fingerprint, operator)
        self.calibration.record(p, actual_certified)
        return p


def from_casebase(store, *, ledger_records: Optional[Sequence[dict]] = None) -> List[OutcomeRecord]:
    """Adapter: pull OutcomeRecords from a CaseBaseStore's durable outcome ledger. The store persists
    (fingerprint, family, gain, cost, certified?) tuples; we map them to surrogate training rows. If the
    store does not expose raw rows, pass `ledger_records` (a list of dicts with fingerprint/operator/
    certified/gain) directly."""
    rows = ledger_records
    if rows is None:
        rows = _read_store_rows(store)
    out: List[OutcomeRecord] = []
    for r in rows:
        fp = r.get("fingerprint") or r.get("fp") or {}
        op = r.get("operator") or r.get("family") or r.get("approach") or "unknown"
        gain = _num(r.get("gain"))
        # CaseBaseStore outcome rows track realized `gain` but not a certified flag (certification lives in
        # the promotion ledger). When no explicit `certified` is present we PROXY it as "realized a positive
        # gain" -- the honest available signal that this operator helped on data like this.
        cert = bool(r["certified"]) if "certified" in r else (gain > 0.0)
        out.append(OutcomeRecord(fingerprint=fp, operator=str(op), certified=cert, gain=gain))
    return out


def _read_store_rows(store) -> List[dict]:
    """Read a CaseBaseStore's durable outcome JSONL via its public `outcome_path`. Returns [] if the
    ledger file does not exist yet (a fresh store)."""
    import json
    import os
    path = store.outcome_path        # CaseBaseStore exposes outcome_path (see casebase_store.py)
    if not path or not os.path.exists(path):
        return []
    rows: List[dict] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


__all__ = ["Surrogate", "OutcomeRecord", "CalibrationTracker", "from_casebase"]
