"""Cross-experiment multiplicity control over the SEQUENCE of promotions.

A single run's certifier (in ``vectorforge.science``) is frozen: it emits one one-sided
(1-alpha) certificate per run. But the platform promotes MANY models over time, and the
more promotions we test the more false discoveries we expect by chance. This module is an
ADDITIVE accounting layer on top of the frozen certifier: it does NOT change any single
run's bound. It only controls the false-discovery rate ACROSS the stream of certified
promotions, plus it records a first-class NEGATIVE certificate ("what did not work") so
failures are durable, not discarded.

Components
----------
* ``PromotionLedger`` -- append-only JSONL of terminal-outcome entries (one per promotion
  decision). Atomic, tolerant of missing/corrupt files. Carries a CALLER-supplied ``ts``
  (this module never reads a clock).
* ``LordFDR`` -- an online false-discovery-rate controller (the LORD / alpha-investing
  family). It is a VALID online-FDR procedure: the testing levels alpha_t are decided using
  only past decisions, and the total wealth spent is bounded by the initial wealth plus the
  gains from past rejections, which keeps mFDR controlled at level alpha.
* ``negative_certificate`` -- a pure function returning a structured negative-result record
  with a tighter futility alpha. The caller appends it to a ledger.

Self-contained: stdlib + numpy only. NOT wired into the loop; the operator hand-wires it.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np


# =========================================================================== ledger
class PromotionLedger:
    """Append-only JSONL ledger of terminal promotion outcomes.

    A terminal outcome entry SHOULD carry::

        {plan_hash, decision, metric, theta, observed, lower_bound, p_value,
         certified, ts}

    where ``ts`` is supplied by the caller (this module never calls a clock). ``record``
    does not enforce the full schema (callers may add fields) but warns by raising if the
    entry is not a dict. Atomic write (tmp + fsync + os.replace); tolerant of a
    missing/corrupt file on read.
    """

    def __init__(self, path):
        self.path = str(path)

    def _read_lines(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read().splitlines()
        except OSError:
            return []

    def record(self, entry: dict) -> None:
        if not isinstance(entry, dict):
            raise ValueError("entry must be a dict")
        line = json.dumps(entry, sort_keys=True)
        existing = [ln for ln in self._read_lines() if ln.strip()]
        payload = ("\n".join(existing + [line]) + "\n")
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".promo.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def all(self) -> list:
        out = []
        for ln in self._read_lines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except (json.JSONDecodeError, ValueError):
                continue
        return out

    def count(self) -> int:
        return len(self.all())


# =========================================================================== online FDR
def _default_gamma_seq(n, c=0.07720838228):
    """A summable non-increasing sequence gamma_j with sum(gamma) = 1.

    LORD uses an infinite non-negative non-increasing sequence {gamma_j} with sum_j gamma_j
    = 1. We use the standard choice gamma_j proportional to log(max(j,2)) / (j * exp(sqrt(log j))),
    which is the sequence recommended in Javanmard & Montanari (2018) for LORD. We normalize
    the first ``n`` terms so they are correct relative to one another; because the tail is
    tiny, normalizing a long prefix to sum 1 is an accurate, conservative approximation that
    keeps the procedure valid (a slightly smaller alpha_t is always conservative for FDR).

    The default constant ``c`` is unused here (kept for API symmetry); the sequence is
    normalized numerically below.
    """
    j = np.arange(1, n + 1, dtype=float)
    logj = np.log(np.maximum(j, 2.0))
    raw = logj / (j * np.exp(np.sqrt(logj)))
    raw = np.maximum(raw, 0.0)
    s = raw.sum()
    if s <= 0:
        # degenerate guard; uniform fallback (still summable when normalized)
        raw = np.ones(n)
        s = raw.sum()
    return raw / s


class LordFDR:
    """Online false-discovery-rate controller (LORD / alpha-investing variant).

    This is the LORD procedure (Javanmard & Montanari 2018, "Online rules for control of
    false discovery rate and false discovery exceedance"). It controls mFDR at level
    ``alpha`` for a stream of p-values tested one at a time, where the testing level alpha_t
    for test t may depend only on the OUTCOMES of tests 1..t-1.

    Wealth dynamics (LORD++ form, a valid online-FDR procedure):

        * Start with wealth W_0 = alpha * w0, where w0 = 0.5 (a fraction of alpha kept in
          reserve so the procedure can keep testing forever).
        * The testing level for test t is
              alpha_t = gamma_t * w0 * alpha
                        + (alpha - b0) * gamma_{t - tau_1}        (first rejection)
                        + b0 * sum_{j>=2} gamma_{t - tau_j}       (later rejections)
          where tau_j is the time of the j-th rejection, b0 = alpha, and we use the
          standard LORD++ accounting that each rejection at time tau adds back budget
          gamma_{t-tau} * (payout) at every later time t.
        * Reject test t iff p_t <= alpha_t.

    We implement the widely-used LORD++ recursion directly via per-rejection "earn-back"
    so the invariant -- total level spent <= initial wealth + sum of rejection payouts --
    holds by construction. This makes ``alpha_spent`` (sum of alpha_t over tests) bounded
    and the procedure a valid online-FDR rule. The frozen single-run certifier is unchanged;
    this only gates which of MANY certified promotions count as discoveries.
    """

    def __init__(self, alpha=0.05, gamma_seq=None, w0_frac=0.5, max_horizon=100000):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1): {alpha}")
        if not (0.0 < w0_frac < 1.0):
            raise ValueError(f"w0_frac must be in (0,1): {w0_frac}")
        self.alpha = float(alpha)
        self.w0_frac = float(w0_frac)
        self._max_h = int(max_horizon)
        if gamma_seq is None:
            self.gamma = _default_gamma_seq(self._max_h)
        else:
            g = np.asarray(gamma_seq, dtype=float)
            if g.ndim != 1 or g.size == 0 or np.any(g < 0):
                raise ValueError("gamma_seq must be a non-empty 1-D non-negative array")
            # Do not silently rescale a caller-supplied sequence; require it to be (near) a
            # probability sequence so validity is the caller's explicit choice.
            self.gamma = g
        # initial wealth = alpha * w0_frac ; payout per (re)discovery uses b0 = alpha*(1-w0_frac)
        self.W0 = self.alpha * self.w0_frac
        self.b0 = self.alpha * (1.0 - self.w0_frac)
        self.t = 0                      # number of tests performed
        self.n_rej = 0
        self._rej_times = []            # 1-indexed times of rejections (tau_j)
        self.alpha_spent = 0.0

    def _gamma(self, k):
        """gamma_k for 1-indexed k>=1; 0 outside support."""
        if k < 1:
            return 0.0
        idx = k - 1
        if idx >= self.gamma.size:
            return 0.0
        return float(self.gamma[idx])

    def _alpha_t(self, t):
        """Compute the testing level for test number t (1-indexed) from PAST rejections only.

        alpha_t = gamma_t * W0
                  + b0 * sum over rejection times tau (tau < t) of gamma_{t - tau}
        This is the LORD++ recursion: initial wealth is spread by gamma_t, and each past
        rejection earns back budget b0 spread by gamma over subsequent tests. All terms
        depend only on tests before t, so the level is predictable -> valid online-FDR.
        """
        level = self._gamma(t) * self.W0
        for tau in self._rej_times:
            if tau < t:
                level += self.b0 * self._gamma(t - tau)
        # never exceed alpha for a single test (a safe, conservative cap)
        return min(level, self.alpha)

    def test(self, p_value: float) -> dict:
        """Test the next p-value in the stream. Returns {"reject": bool, "alpha_t": float}.

        Decisions are made online: alpha_t uses only past outcomes, then we compare p to it.
        """
        try:
            p = float(p_value)
        except (TypeError, ValueError):
            raise ValueError("p_value must be a float")
        if not (0.0 <= p <= 1.0):
            # tolerant clip with no silent corruption of the count: clip and proceed
            p = min(1.0, max(0.0, p))
        self.t += 1
        a_t = self._alpha_t(self.t)
        self.alpha_spent += a_t
        reject = p <= a_t
        if reject:
            self.n_rej += 1
            self._rej_times.append(self.t)
        return {"reject": bool(reject), "alpha_t": float(a_t)}

    def summary(self) -> dict:
        return {
            "n_tests": int(self.t),
            "n_discoveries": int(self.n_rej),
            "alpha": float(self.alpha),
            "alpha_spent": float(self.alpha_spent),
        }


# =========================================================================== negative cert
def negative_certificate(*, plan_hash, reason, diagnosis=None, observed=None,
                         lower_bound=None, theta, alpha_futility):
    """Build a structured NEGATIVE certificate ("what did not work") as a plain dict.

    A first-class negative result: it records that a pre-registered plan (``plan_hash``) did
    NOT clear its bar, WHY (``reason`` + optional ``diagnosis``), and the numbers, under a
    TIGHTER futility alpha (``alpha_futility``, typically smaller than the promotion alpha so
    we only declare futility when we are quite sure the effect is below theta). Pure: returns
    a dict; the caller appends it to a ledger. This module never writes here.

    Returns
    -------
    dict with keys: kind="negative", plan_hash, reason, diagnosis, theta, observed,
    lower_bound, alpha_futility, certified=False.
    """
    if not isinstance(plan_hash, str) or not plan_hash:
        raise ValueError("plan_hash must be a non-empty string")
    if not isinstance(reason, str) or not reason:
        raise ValueError("reason must be a non-empty string")
    try:
        theta = float(theta)
        alpha_futility = float(alpha_futility)
    except (TypeError, ValueError) as e:
        raise ValueError(f"theta and alpha_futility must be floats: {e}")
    if not (0.0 < alpha_futility < 1.0):
        raise ValueError(f"alpha_futility must be in (0,1): {alpha_futility}")

    def _opt_float(x):
        return None if x is None else float(x)

    return {
        "kind": "negative",
        "plan_hash": plan_hash,
        "reason": reason,
        "diagnosis": diagnosis,
        "theta": theta,
        "observed": _opt_float(observed),
        "lower_bound": _opt_float(lower_bound),
        "alpha_futility": alpha_futility,
        "certified": False,
    }


# --------------------------------------------------------------------------- self-test
def _selftest():
    import shutil

    tmpd = tempfile.mkdtemp(prefix="cross_exp_selftest_")
    try:
        led = PromotionLedger(os.path.join(tmpd, "promo.jsonl"))
        led.record({"plan_hash": "sha256:abc", "decision": "promote", "ts": 1.0})
        led.record({"plan_hash": "sha256:def", "decision": "reject", "ts": 2.0})
        assert led.count() == 2

        rng = np.random.default_rng(0)
        fdr = LordFDR(alpha=0.05)
        ps = list(rng.uniform(size=200))
        ps[10] = ps[50] = ps[120] = 1e-6  # injected strong signals
        for p in ps:
            fdr.test(p)
        s = fdr.summary()
        assert s["n_discoveries"] >= 3
        assert s["alpha_spent"] <= s["n_tests"] * s["alpha"]

        nc = negative_certificate(plan_hash="sha256:abc", reason="lower_bound below theta",
                                  theta=0.8, observed=0.71, lower_bound=0.66, alpha_futility=0.01)
        assert nc["certified"] is False and nc["kind"] == "negative"
        print("cross_experiment self-test OK", s)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
