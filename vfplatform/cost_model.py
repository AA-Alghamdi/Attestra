"""A small, explicit COST MODEL -- the seed of cost-aware VoI + scheduling.

This module is pure, deterministic, and clock-free. It encodes a conservative price list for the
device/GPU FLAVORS the platform can dispatch to (the local CPU path, plus the RunPod GPU classes), and
three primitives the portfolio/VoI layers use to reason in DOLLARS rather than seconds:

  * ``cost_per_hour(flavor)``   -- USD/hr for a flavor (0 for CPU).
  * ``estimate_cost(seconds, flavor)`` -- USD for a measured wall-clock duration on a flavor.
  * ``gain_per_cost(gain, usd)`` -- the VoI currency: expected bound LIFT per DOLLAR spent.

IMPORTANT (scientific integrity): the per-hour numbers below are ESTIMATES, not billing truth. They are
conservative upper-ish on-demand bounds taken to match the same intent as the price list already hard-wired
in ``vfplatform.providers.RunPodProvider._GPU_COST`` (so the Checkpoint always sees GPU work as PAID).
They are NOT read from a live pricing API and MUST NOT be cited as actual spend. CPU is modelled as ~0
because the local CPU path bills nothing in this platform (it runs on the operator's own machine).

Nothing here certifies, promotes, or touches the sealed test. It is a reporting/scheduling input only.
"""
from __future__ import annotations

# device/gpu flavor -> {usd_per_hour, note}. Conservative on-demand estimates (USD/hr), NOT billing truth.
# The GPU figures mirror the intent of providers.RunPodProvider._GPU_COST (upper-ish for cost-safety) and
# add the common RunPod on-demand classes (A40/A100/A5000/RTX4090/L4). Documented as estimates everywhere.
FLAVORS: dict = {
    "cpu": {"usd_per_hour": 0.0,
            "note": "local CPU path; bills nothing in this platform (operator's own machine)"},
    "A40": {"usd_per_hour": 0.79,
            "note": "RunPod on-demand estimate; matches providers._GPU_COST['A40']; upper-ish, not billing"},
    "A100": {"usd_per_hour": 1.99,
             "note": "RunPod 80GB on-demand estimate; matches providers._GPU_COST['A100']; upper-ish"},
    "A5000": {"usd_per_hour": 0.44,
              "note": "RunPod RTX A5000 on-demand estimate; upper-ish, not billing truth"},
    "RTX4090": {"usd_per_hour": 0.74,
                "note": "RunPod RTX 4090 on-demand estimate; upper-ish, not billing truth"},
    "L4": {"usd_per_hour": 0.43,
           "note": "RunPod / cloud L4 on-demand estimate; upper-ish, not billing truth"},
}

# Fallback USD/hr for an unknown flavor: a deliberately CONSERVATIVE (high) bound so an unrecognized device
# is never under-priced. Documented as an estimate, like everything else here.
_UNKNOWN_FLAVOR_USD_PER_HOUR = 2.0


def cost_per_hour(flavor: str) -> float:
    """USD/hr for ``flavor``. Known flavors come from FLAVORS; an unknown flavor returns a conservative
    (high) fallback so it is never under-priced. Pure + deterministic."""
    f = FLAVORS.get(str(flavor))
    if f is None:
        return float(_UNKNOWN_FLAVOR_USD_PER_HOUR)
    return float(f["usd_per_hour"])


def estimate_cost(seconds: float, flavor: str) -> float:
    """USD for ``seconds`` of wall-clock on ``flavor``: max(seconds,0)/3600 * cost_per_hour(flavor).

    Negative or non-finite durations are clamped to 0 (a duration cannot cost negative dollars, and a NaN
    duration must not poison a budget). Pure + deterministic; no clock is read here."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        s = 0.0
    if not (s == s) or s in (float("inf"), float("-inf")):   # NaN/Inf guard
        s = 0.0
    if s < 0.0:
        s = 0.0
    return (s / 3600.0) * cost_per_hour(flavor)


def gain_per_cost(gain: float, usd: float, floor: float = 1e-6) -> float:
    """The VoI currency: expected bound LIFT per DOLLAR -- gain / max(usd, floor).

    ``floor`` keeps the ratio finite for free/near-free (CPU) work: a positive gain at ~0 cost yields a
    large-but-finite gain-per-cost rather than a division by zero, which is the correct ordering (free wins
    that lift the bound are maximally cost-efficient). A non-positive gain returns 0.0 (no value, regardless
    of cost). Pure + deterministic."""
    try:
        g = float(gain)
    except (TypeError, ValueError):
        return 0.0
    if not (g == g):                       # NaN gain -> no measurable value
        return 0.0
    if g <= 0.0:
        return 0.0
    try:
        c = float(usd)
    except (TypeError, ValueError):
        c = 0.0
    if not (c == c) or c < 0.0:            # NaN/negative cost -> treat as 0 (use the floor)
        c = 0.0
    fl = float(floor) if (floor and floor > 0.0) else 1e-6
    return g / max(c, fl)


# --------------------------------------------------------------------------- self-test
def _selftest():
    assert cost_per_hour("cpu") == 0.0
    assert cost_per_hour("A100") == 1.99
    assert cost_per_hour("totally-unknown-gpu") == _UNKNOWN_FLAVOR_USD_PER_HOUR
    # a GPU flavor costs strictly more per hour than CPU
    assert cost_per_hour("A40") > cost_per_hour("cpu")

    # estimate_cost: one hour on A100 == its hourly rate; CPU is free; clamps bad durations
    assert abs(estimate_cost(3600.0, "A100") - 1.99) < 1e-12
    assert estimate_cost(3600.0, "cpu") == 0.0
    assert estimate_cost(-5.0, "A40") == 0.0
    assert estimate_cost(float("nan"), "A40") == 0.0
    assert abs(estimate_cost(1800.0, "A40") - 0.79 * 0.5) < 1e-12

    # gain_per_cost: free work with positive gain -> large finite ratio; non-positive gain -> 0
    assert gain_per_cost(0.1, 0.0) == 0.1 / 1e-6
    assert gain_per_cost(0.0, 5.0) == 0.0
    assert gain_per_cost(-0.1, 5.0) == 0.0
    assert abs(gain_per_cost(0.2, 2.0) - 0.1) < 1e-12
    # cheaper dollars for the same gain -> strictly higher gain-per-cost
    assert gain_per_cost(0.1, 1.0) > gain_per_cost(0.1, 10.0)
    print("cost_model self-test OK")


if __name__ == "__main__":
    _selftest()
