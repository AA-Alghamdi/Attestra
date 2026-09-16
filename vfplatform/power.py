"""POWER-AWARE experiment design -- "is there even enough data to certify?" -- NEW module.

A frequent, demoralizing failure mode: a genuinely good model HONEST-STOPS not because it is weak but because
the validation set is too small for any lower bound to clear theta (the wine/iris-on-30-rows case). That is an
*underpowered* result, and it is a different finding from "the model is below the bar". This module computes,
for the EXACT frozen accuracy certifier, the statistical power to certify and the minimum sample size needed --
so the loop can tell the user WHICH failure they hit and whether collecting more data would help.

EXACTNESS + single source of truth: for the binomial (accuracy) certifier this is computed exactly from the
FROZEN `certify_accuracy` / `clopper_pearson_lower` / `binom_sf` in `vectorforge.science` -- no re-derivation
of the statistics here, so the power numbers are consistent with the gate by construction. For non-binomial
metrics (balanced_accuracy, macro_f1, the bootstrap regression metrics) an exact power requires simulation;
v1 honestly returns power=None for those rather than fabricate a number (no guessing -- project rule).

CONTRACT: pure analysis. No estimator, no sealed peek, no certificate, no clock. The frozen core is untouched
and is the only thing that promotes; this module can only REPORT (powered vs underpowered, and a target n).
"""
from __future__ import annotations

from vectorforge.science import certify_accuracy, binom_sf

_BINOMIAL_METRICS = ("accuracy",)          # metrics whose certifier is the exact binomial CP path


def _min_k_to_certify(n, theta, alpha, checks):
    """Smallest integer correct-count k in [0, n] for which the FROZEN certify_accuracy certifies at this n.
    Certification is monotone non-decreasing in k (CP lower bound rises, the binomial p-value falls), so a
    binary search is exact. Returns n+1 when not even a perfect model (k=n) can certify at this n."""
    if n <= 0:
        return 1                                   # nothing certifies on no data
    if not certify_accuracy(1.0, n, theta, checks=checks, alpha=alpha)["certified"]:
        return n + 1                               # even k=n fails -> infeasible at this n
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if certify_accuracy(mid / n, n, theta, checks=checks, alpha=alpha)["certified"]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def power_at_n(n, theta, p_assumed, *, alpha=0.05, checks=1):
    """Probability the frozen accuracy certifier CERTIFIES at sample size n, IF the model's true accuracy is
    p_assumed. = P(K >= k* | K ~ Binomial(n, p_assumed)) where k* is the smallest certifying count. Exact.
    Returns 0.0 when no count certifies at this n (e.g. n too small) and ~alpha when p_assumed == theta."""
    n = int(n)
    if n <= 0 or not (0.0 <= p_assumed <= 1.0):
        return 0.0
    kstar = _min_k_to_certify(n, float(theta), float(alpha), int(max(1, checks)))
    if kstar > n:
        return 0.0
    return float(binom_sf(kstar, n, float(p_assumed)))


def min_n_for_power(theta, p_assumed, *, alpha=0.05, checks=1, target_power=0.8, n_max=2_000_000):
    """Smallest n at which projected power to certify >= target_power, assuming true accuracy p_assumed.
    Returns None when p_assumed <= theta (a model that is not truly above theta cannot be certified above it
    at any n -- power asymptotes to <= alpha, never reaching a sensible target) or when n_max is exceeded.

    Power trends upward in n but is not strictly monotone (binomial discreteness causes small wiggles), so we
    return the FIRST n meeting the target and confirm it also holds at n+1, n+2 to skip a one-off dip."""
    theta, p_assumed = float(theta), float(p_assumed)
    if p_assumed <= theta + 1e-12:
        return None
    n = 1
    # expand by doubling to bracket, then walk to the exact first stable crossing (cheap: CP is fast)
    while n <= n_max:
        if power_at_n(n, theta, p_assumed, alpha=alpha, checks=checks) >= target_power:
            # confirm stability across the next two n to avoid a discreteness wiggle
            if all(power_at_n(n + d, theta, p_assumed, alpha=alpha, checks=checks) >= target_power
                   for d in (1, 2)):
                # walk backwards to the true smallest n that (stably) meets the target
                m = n
                while m > 1 and power_at_n(m - 1, theta, p_assumed, alpha=alpha, checks=checks) >= target_power:
                    m -= 1
                return m
        n = n + 1 if n < 64 else int(n * 1.5)      # fine near the bottom, geometric once large
    return None


def assess(n, theta, p_assumed, *, metric="accuracy", alpha=0.05, checks=1, target_power=0.8):
    """Diagnose whether a certify attempt is POWERED at the given sample size. Non-binding analysis.

    Returns a dict:
        metric, n, theta, p_assumed, alpha, checks, target_power
        power            -- probability of certifying at n if true perf == p_assumed (None for non-binomial)
        min_n_for_target -- smallest n reaching target_power (None if p_assumed<=theta or metric non-binomial)
        powered          -- True if power >= target_power; None if not computable for this metric
        note             -- a one-line human explanation (underpowered vs model-limited vs not-computed)
    """
    out = {"metric": metric, "n": int(n), "theta": round(float(theta), 4),
           "p_assumed": (round(float(p_assumed), 4) if p_assumed is not None else None),
           "alpha": alpha, "checks": int(max(1, checks)), "target_power": target_power,
           "power": None, "min_n_for_target": None, "powered": None, "note": ""}
    if metric not in _BINOMIAL_METRICS:
        out["note"] = (f"power not computed for metric {metric!r} in v1 (non-binomial certifier needs "
                       f"simulation); reported only for accuracy.")
        return out
    if p_assumed is None:
        out["note"] = "no assumed performance (p_assumed) available to estimate power."
        return out
    pw = power_at_n(n, theta, p_assumed, alpha=alpha, checks=checks)
    out["power"] = round(pw, 4)
    out["powered"] = bool(pw >= target_power)
    if p_assumed <= theta + 1e-12:
        out["note"] = (f"model-limited, NOT underpowered: assumed accuracy {round(p_assumed,4)} does not exceed "
                       f"theta {round(theta,4)}, so no sample size can certify above theta. Improve the model.")
        return out
    need = min_n_for_power(theta, p_assumed, alpha=alpha, checks=checks, target_power=target_power)
    out["min_n_for_target"] = need
    if out["powered"]:
        out["note"] = (f"powered: at n={int(n)} the projected power to certify theta={round(theta,4)} "
                       f"(assuming true accuracy {round(p_assumed,4)}) is {round(pw,3)} >= {target_power}.")
    else:
        extra = f"; need n>={need} for power {target_power}" if need else ""
        out["note"] = (f"UNDERPOWERED: at n={int(n)} projected power is only {round(pw,3)} < {target_power} "
                       f"(assuming true accuracy {round(p_assumed,4)} > theta {round(theta,4)}){extra}. "
                       f"More evaluation data would help; the model may already be good enough.")
    return out


def _selftest():
    # (a) power rises with n, and at p*=theta power is small (<= a few x alpha).
    p_small = power_at_n(50, 0.9, 0.9)
    p_big = power_at_n(2000, 0.9, 0.95)
    assert p_small < 0.2, p_small
    assert p_big > 0.9, p_big
    assert power_at_n(2000, 0.9, 0.95) >= power_at_n(200, 0.9, 0.95), "power must trend up with n"

    # (b) min_n is finite and monotone: a bigger margin (p* further above theta) needs FEWER samples.
    n_tight = min_n_for_power(0.9, 0.93)
    n_loose = min_n_for_power(0.9, 0.98)
    assert n_tight and n_loose and n_loose < n_tight, (n_loose, n_tight)
    # and the returned n actually meets the target while n-1 does not (it is the true threshold)
    assert power_at_n(n_loose, 0.9, 0.98) >= 0.8 and power_at_n(n_loose - 1, 0.9, 0.98) < 0.8, n_loose

    # (c) p* <= theta -> no n certifies above theta -> min_n is None, assess says model-limited.
    assert min_n_for_power(0.9, 0.9) is None
    a = assess(100, 0.9, 0.88)
    assert a["powered"] is False and "model-limited" in a["note"], a

    # (d) underpowered case: genuinely-good model, tiny n -> powered False with a target n.
    u = assess(30, 0.9, 0.97)
    assert u["powered"] is False and u["min_n_for_target"] and "UNDERPOWERED" in u["note"], u
    # ... and the SAME model with enough data is powered.
    g = assess(u["min_n_for_target"], 0.9, 0.97)
    assert g["powered"] is True, g

    # (e) non-binomial metric: honest None, no fabricated number.
    nb = assess(500, 0.9, 0.95, metric="macro_f1")
    assert nb["power"] is None and "not computed" in nb["note"], nb

    print("power self-test: PASS", {"n_tight@0.93": n_tight, "n_loose@0.98": n_loose,
                                     "underpowered_need": u["min_n_for_target"]})


if __name__ == "__main__":
    _selftest()
