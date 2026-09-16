"""Anytime-valid e-process (betting supermartingale) for H0: p <= theta on graded-in-[0,1] outcomes.

WHERE THIS BELONGS (and where it does NOT)
------------------------------------------
The autoresearcher PEEKS REPEATEDLY: within a bracket it races validation streams, grades candidates
sequentially, and wants to STOP EARLY once a candidate is clearly above (or hopelessly below) a quality
floor theta. A fixed-n hypothesis test is INVALID under that kind of optional stopping -- repeated
peeking inflates type-I error. An e-process is valid at EVERY stopping time (Ville's inequality), which
is exactly what repeated-peek racing needs.

THIS IS NOT THE PROMOTION GATE. The sole gate that decides whether a model is returned to the user
remains the FROZEN fixed-n Clopper-Pearson lower-bound certifier in vectorforge/science.py, reached only
via vfplatform/sealed.py on a sealed split. This module is ADDITIVE and STANDALONE: it imports nothing
from the certifier or sealed runner, it modifies no certificate, and no promotion path reads it. Its job
is to make RACING/SCHEDULING cheaper and statistically honest under sequential peeking; the locked-test
certificate is still computed once, fixed-n, at the end -- exactly as before.

THE MATH
--------
Outcomes x_1, x_2, ... are graded scores in [0, 1] (per-example correctness, or a bounded reward). Under
H0 the per-step conditional mean is at most theta. Define a betting supermartingale (a capital process):

    E_0 = 1,   E_n = prod_{i=1..n} (1 + lambda_i * (x_i - theta))

where lambda_i is PREDICTABLE (chosen from x_1..x_{i-1} only, never x_i). Since x_i - theta lies in
[-theta, 1-theta], every factor is non-negative iff

    lambda_i in [ -1/(1 - theta) , 1/theta ]                            (0 < theta < 1).

With lambda in that range, E_n >= 0 always. Under H0, E[x_i - theta | past] <= 0, so
E[E_n | past] <= E_{n-1}: E_n is a non-negative supermartingale with E[E_0] = 1. Ville's inequality gives

    P( sup_n E_n >= 1/alpha )  <=  alpha          (under H0),

so the rule "reject H0 the first time E_n >= 1/alpha" controls type-I error at alpha SIMULTANEOUSLY over
all n -- you may peek as often as you like. p_value_anytime := min(1, 1/E_n) is an anytime-valid p-value.

BETTING SCHEDULE (frozen closed form; not tuned to any answer sheet)
--------------------------------------------------------------------
lambdas="grow": a fixed GROW-style fraction. We stake a constant fraction c of the maximal admissible
positive stake, i.e. lambda_i = c / theta with c in (0, 1). This is the closed-form, capped analogue of
the growth-rate-optimal (GRO) prescription -- it maximizes expected log-capital under a fixed alternative
without peeking, and the cap keeps every factor non-negative. The default c = 0.5 is a frozen,
data-independent choice (half-Kelly: the standard variance-robust shrinkage of the log-optimal stake). It
is NOT fit to any benchmark.

lambdas="adaptive": a predictable plug-in. lambda_i adapts using ONLY the running mean of x_1..x_{i-1}
(still predictable, so Ville holds), aiming the stake toward the observed level. Offered as an option;
the default is the frozen "grow" rule so behavior is reproducible and reviewable.

LOWER CONFIDENCE SEQUENCE (always-valid lower bound on p)
---------------------------------------------------------
Inverting the family {E_n(theta')} over candidate floors theta' yields a lower confidence sequence:
L_n = the largest floor theta' that the e-process against "p <= theta'" has already rejected at level
1/alpha on the observed stream. Because (x_i - theta') grows as theta' shrinks, the e-process against
theta' is monotone in theta' -- a single crossing -- which we locate by bisection on the recorded stream.
With probability >= 1 - alpha, L_n <= p for ALL n simultaneously.

Pure + deterministic: no clock, no RNG inside. Any seed/timestamp is the caller's argument.
"""

import math

THETA_EPS = 1e-12           # keep theta strictly inside (0, 1) for the stake caps
DEFAULT_GROW_C = 0.5        # half-Kelly fraction of the maximal admissible positive stake (frozen)


def admissible_lambda_range(theta):
    """Range [lo, hi] of betting fractions keeping (1 + lambda*(x - theta)) >= 0 for all x in [0,1]."""
    t = min(max(theta, THETA_EPS), 1.0 - THETA_EPS)
    return (-1.0 / (1.0 - t), 1.0 / t)


def grow_lambda(theta, c=DEFAULT_GROW_C):
    """Frozen closed-form GROW fraction: constant positive stake = c * (max admissible positive stake)."""
    lo, hi = admissible_lambda_range(theta)
    t = min(max(theta, THETA_EPS), 1.0 - THETA_EPS)
    lam = c / t
    return min(max(lam, lo), hi)


def _log_e_on_stream(xs, theta, alpha, c, early_stop_thr=None):
    """Replay the frozen-grow e-process for H0: p <= theta on a recorded stream xs. Returns log E_n.

    If early_stop_thr is given, returns as soon as log E reaches it (used by the lower-bound scan).
    A wiped factor (<= 0) makes capital 0 -> log E = -inf; that floor is not rejected by growth.
    """
    if not (0.0 < theta < 1.0):
        return float("-inf")
    lam = grow_lambda(theta, c)
    log_e = 0.0
    for x in xs:
        f = 1.0 + lam * (x - theta)
        if f <= 0.0:
            return float("-inf")
        log_e += math.log(f)
        if early_stop_thr is not None and log_e >= early_stop_thr:
            return log_e
    return log_e


class EProcess:
    """Betting-supermartingale e-process for H0: p <= theta with graded outcomes x in [0, 1].

    KEEP fixed-n Clopper-Pearson as the frozen promotion gate. This e-process is for repeated-peek racing,
    where its anytime validity (Ville's inequality) earns the right to stop early without inflating
    type-I error. It produces NO certificate and gates NO promotion.
    """

    def __init__(self, theta, alpha=0.05, lambdas="grow", grow_c=DEFAULT_GROW_C):
        if not (0.0 < theta < 1.0):
            raise ValueError(f"theta must be in (0,1); got {theta}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1); got {alpha}")
        if lambdas not in ("grow", "adaptive"):
            raise ValueError(f"lambdas must be 'grow' or 'adaptive'; got {lambdas!r}")
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.lambdas = lambdas
        self.grow_c = float(grow_c)
        self.log_thr = math.log(1.0 / self.alpha)
        self._lo, self._hi = admissible_lambda_range(self.theta)
        # state
        self.n = 0
        self.log_e = 0.0          # log E_n, accumulated in log-space for numerical stability
        self._sum_x = 0.0         # running sum (past) -> predictable adaptive schedule
        self._xs = []             # recorded stream (for the exact inverted lower-confidence sequence)

    # -- predictable betting fraction (uses only the PAST: state before this step) -----------------
    def _next_lambda(self):
        if self.lambdas == "grow" or self.n == 0:
            return grow_lambda(self.theta, self.grow_c)
        # "adaptive": predictable plug-in from the running mean of the PAST only.
        mean_hat = self._sum_x / self.n
        raw = mean_hat - self.theta
        if raw >= 0.0:
            lam = (raw / max(1.0 - self.theta, THETA_EPS)) * self._hi
        else:
            lam = (raw / max(self.theta, THETA_EPS)) * self._lo  # raw<0, _lo<0 -> lam<0
        return min(max(lam, self._lo), self._hi)

    @property
    def e_value(self):
        return math.exp(self.log_e) if self.log_e != float("-inf") else 0.0

    def update(self, x):
        """Accumulate one graded outcome x in [0, 1]; return the current anytime-valid summary dict."""
        x = float(x)
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"graded outcome must be in [0,1]; got {x}")
        lam = self._next_lambda()                      # predictable: from past state only
        factor = 1.0 + lam * (x - self.theta)
        if factor < 0.0:                               # admissible lambda => factor >= 0; clamp round-off
            factor = 0.0
        # commit state
        self.n += 1
        self._sum_x += x
        self._xs.append(x)
        if factor == 0.0:
            self.log_e = float("-inf")
        elif self.log_e != float("-inf"):
            self.log_e += math.log(factor)
        e = self.e_value
        return {
            "e_value": e,
            "reject": e >= (1.0 / self.alpha),
            "n": self.n,
            "p_value_anytime": min(1.0, (1.0 / e) if e > 0.0 else 1.0),
        }

    # -- always-valid lower confidence sequence on p ----------------------------------------------
    def lower_bound(self):
        """Largest floor theta' the data have already rejected from below = anytime lower bound on p.

        Replays the frozen-grow e-process against H0': p <= theta' on the recorded stream, for a grid of
        theta'. The e-process against theta' is monotone increasing as theta' shrinks, so there is a single
        crossing of 1/alpha; we bisection-search the boundary. With prob >= 1-alpha, the result <= true p
        for all n simultaneously (it is the inverted e-process / running maximum of rejected floors).
        """
        if self.n == 0:
            return 0.0

        def crosses(theta_p):
            return _log_e_on_stream(
                self._xs, theta_p, self.alpha, self.grow_c, early_stop_thr=self.log_thr
            ) >= self.log_thr

        lo, hi = THETA_EPS, 1.0 - THETA_EPS
        if not crosses(lo):
            return 0.0  # not even "p <= 0+" rejected: no informative lower bound yet
        if crosses(hi):
            return hi   # everything rejected (degenerate); clamp to the admissible top
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if crosses(mid):
                lo = mid
            else:
                hi = mid
        return lo


# ============================================================================ self-test stub
if __name__ == "__main__":
    # Minimal smoke check; the real validity/power suite lives in tests/test_eprocess.py.
    ep = EProcess(theta=0.5, alpha=0.05)
    out = None
    for xi in [1.0] * 40:
        out = ep.update(xi)
    print("smoke:", {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()},
          "lb=", round(ep.lower_bound(), 4))
