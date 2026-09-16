"""Moat-enforcement layer surfaced by the architecture swarm's adversarial stress.

Two recurring moat risks were CONFIRMED against the frozen science.py:
  1. THE "ONE COUNTED PEEK" IS CONVENTION, NOT ENFORCED. certify_accuracy(checks=1) trusts the caller; a
     fan-out / multi-round loop that peeks the sealed test more than once while passing checks=1 launders
     the multiplicity correction and produces an invalid certificate.
  2. score_metric FALLS THROUGH TO ACCURACY for any unrecognized metric string -> an unknown/typo'd metric
     silently certifies on accuracy (a fake certificate for the WRONG metric).

This module closes BOTH at the platform boundary WITHOUT touching the frozen core:
  * SealedTest content-addresses the locked test and COUNTS every evaluation; the certify call passes the
    REALIZED peek count as `checks`, so multiplicity is paid (Bonferroni) and a 2nd uncounted peek is
    refused. "One counted peek" becomes code, not convention.
  * assert_supported_metric refuses any metric science.score_metric would silently coerce to accuracy, and
    certify_on_sealed ROUTES each metric to the correct frozen certifier (binomial for accuracy, bootstrap
    lower bound for balanced_accuracy/macro_f1, certify_regression for r2/neg_rmse/neg_mae).
The frozen science.py is unchanged; this is a guard in front of it.
"""
import json
import os
import sys

_VF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VF not in sys.path:
    sys.path.insert(0, _VF)
from vectorforge import science

KNOWN_CLASSIFICATION = ("accuracy", "balanced_accuracy", "macro_f1")
KNOWN_REGRESSION = ("r2", "neg_rmse", "neg_mae")
SUPPORTED_METRICS = KNOWN_CLASSIFICATION + KNOWN_REGRESSION


class MetricNotCertifiable(Exception):
    """Raised for a metric science.score_metric would silently coerce to accuracy."""


class PeekViolation(Exception):
    """Raised when the sealed test is peeked more than its allowed (counted) budget."""


def assert_supported_metric(metric):
    if metric not in SUPPORTED_METRICS:
        raise MetricNotCertifiable(
            f"metric {metric!r} is not a recognized certifiable metric {SUPPORTED_METRICS}; refusing to "
            f"certify because science.score_metric would silently fall through to ACCURACY (a certificate "
            f"for the wrong metric).")


def _canonical(rows, target_key):
    return science.digest([{"t": str(r.get(target_key)),
                            "x": {k: v for k, v in r.items() if k != target_key}} for r in rows])


class PeekLedger:
    """Durable, append-only count of how many times each sealed-test DIGEST has been peeked, ACROSS
    processes (F3). Bonferroni multiplicity must pay for EVERY historical peek of the same locked test;
    otherwise re-running a goal on the same exact test digest each reports checks=1 and launders the
    correction. JSON file {digest: count}; writes are atomic (tmp + os.replace)."""

    def __init__(self, path):
        self.path = path

    def _read(self):
        try:
            with open(self.path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def count(self, digest):
        return int(self._read().get(digest, 0))

    def record(self, digest):
        d = self._read()
        d[digest] = int(d.get(digest, 0)) + 1
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh)
        os.replace(tmp, self.path)
        return d[digest]


class SealedTest:
    """A content-addressed locked test with an ENFORCED peek ledger. Each evaluation is counted; the
    certify call uses the realized peek count as Bonferroni `checks`.

    With a durable `ledger` (PeekLedger), the count is CUMULATIVE across processes keyed by digest, so
    Bonferroni pays for every historical peek of the same locked test (the production path). Without a
    ledger (default), counting is per-instance with a hard `max_peeks` cap (the unit-test / single-run
    semantics): a second uncounted peek raises PeekViolation."""

    def __init__(self, rows, *, target_key="target", max_peeks=1, ledger=None):
        self.rows = list(rows)
        self.target_key = target_key
        self.max_peeks = int(max_peeks)
        self.digest = _canonical(self.rows, target_key)
        self.ledger = ledger
        self._peeks = 0
        self._cumulative = ledger.count(self.digest) if ledger else 0
        self.log = []

    def peek_count(self):
        # cumulative (incl. prior processes) when a durable ledger is present; else this instance's count
        return self._cumulative if self.ledger else self._peeks

    def _count_peek(self, who):
        self._peeks += 1
        if self.ledger is not None:
            self._cumulative = self.ledger.record(self.digest)   # durable cross-run total -> Bonferroni
        else:
            self._cumulative = self._peeks
            if self._peeks > self.max_peeks:
                raise PeekViolation(
                    f"sealed test {self.digest} peeked {self._peeks} times (max_peeks={self.max_peeks}); "
                    f"refusing silent reuse. A re-peek must be explicitly allowed and is paid for via "
                    f"Bonferroni checks={self._peeks}.")
        self.log.append({"peek": self._peeks, "cumulative": self._cumulative, "who": who})

    def y_true(self):
        return [r.get(self.target_key) for r in self.rows]


def certify_on_sealed(sealed: SealedTest, predict_fn, theta, *, metric="accuracy", labels=None,
                      alpha=0.05, who="certify"):
    """Evaluate `predict_fn` on the sealed rows ONCE (counted), then certify with the metric-correct frozen
    certifier and checks = realized peek count. Returns the certificate dict (with sealed_digest + peeks)."""
    assert_supported_metric(metric)
    y_true = sealed.y_true()
    y_pred = list(predict_fn(sealed.rows))            # the single evaluation of the sealed test
    sealed._count_peek(who)
    if len(y_pred) != len(y_true):                    # audit MEDIUM: refuse a length-mismatched pairing
        raise ValueError(f"prediction count {len(y_pred)} != sealed-test size {len(y_true)}; "
                         f"refusing to certify a length-mismatched (truncated/padded) pairing")
    # NOTE: this guard is LENGTH-ONLY (a backstop). Reordering/substitution is caught UPSTREAM on the
    # worker path by per-row id binding in the loop's certify step (the worker echoes a platform-supplied
    # id per row, reindexed with a 1:1 coverage assertion before predict_fn is built). Do not claim this
    # length check alone rejects reordered pairings.
    checks = sealed.peek_count()                      # honest multiplicity: pay for every peek

    if metric == "accuracy":
        observed = sum(1 for a, b in zip(y_true, y_pred) if str(a) == str(b)) / max(len(y_true), 1)
        cert = science.certify_accuracy(observed, len(y_true), theta, checks=checks, alpha=alpha)
    elif metric in ("balanced_accuracy", "macro_f1"):
        a = alpha / checks
        labs = labels or sorted({str(x) for x in y_true})
        yt = [str(x) for x in y_true]
        yp = [str(x) for x in y_pred]
        lower, point = science._bootstrap_classification_metric_lower(metric, yt, yp, labs, alpha=a)
        cert = {"observed": point, "n": len(yt), "metric": metric, "theta": round(theta, 4),
                "checks": checks, "lower_bound": lower, "certified": bool(lower > theta),
                "reason": ("bootstrap lower bound clears theta after multiplicity correction"
                           if lower > theta else "bootstrap lower bound does not clear theta")}
    else:  # regression
        yt = [float(x) for x in y_true]
        yp = [float(x) for x in y_pred]
        cert = science.certify_regression(yt, yp, metric, theta, checks=checks, alpha=alpha)

    cert["sealed_digest"] = sealed.digest
    cert["peeks"] = checks
    return cert
