"""CERTIFIED PARETO FRONT -- the commercial deliverable is a frontier, not a single model.

WHY THIS EXISTS
---------------
A commercial user states proxies ('50ms', 'accuracy') but their real objective is a TRADEOFF ('fast enough
AND accurate enough AND well-calibrated'). So the deliverable is a certified Pareto front: the set of
non-dominated models plus named picks ('fastest that certifies >= theta', 'most accurate under the latency
budget', 'best-calibrated'). The quality-diversity archive (regeneration.py) is the machine that populates
it; this module reads off the front.

THE INVARIANT
-------------
ONLY certified candidates are eligible for the front. 'certified' means the candidate passed the full
multi-objective gate (gate.certified_under_envelope -> CertDecision.certified), which itself wraps the
frozen Tier-3 certificate and can only tighten it. An uncertified model is never presented as a deliverable.
This module reports; it does not certify or promote.

CONTRACT: stdlib only; consumes already-measured candidates. No estimator, no certifier import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# axis direction: +1 means larger-is-better (metric lower bound), -1 means smaller-is-better (latency/cost/ece)
MAXIMIZE = 1
MINIMIZE = -1


@dataclass
class ParetoCandidate:
    """One measured, gated candidate. `metric_lb` is the certified lower bound on the objective (from the
    frozen cert); the rest are measured operational axes. `certified` is the full multi-objective gate
    verdict."""
    name: str
    metric_lb: float
    latency_ms: float
    cost_usd: float
    ece: float
    certified: bool
    extra: Dict[str, float] = field(default_factory=dict)

    def axis(self, name: str) -> float:
        if name == "metric_lb":
            return self.metric_lb
        if name == "latency_ms":
            return self.latency_ms
        if name == "cost_usd":
            return self.cost_usd
        if name == "ece":
            return self.ece
        if name in self.extra:
            return self.extra[name]
        raise KeyError(f"unknown axis {name!r}")


Axis = Tuple[str, int]   # (axis_name, MAXIMIZE|MINIMIZE)

DEFAULT_AXES: List[Axis] = [("metric_lb", MAXIMIZE), ("latency_ms", MINIMIZE), ("cost_usd", MINIMIZE)]


def _dominates(a: ParetoCandidate, b: ParetoCandidate, axes: Sequence[Axis]) -> bool:
    """a dominates b iff a is no worse on every axis and strictly better on at least one."""
    no_worse_all = True
    strictly_better_one = False
    for name, direction in axes:
        av, bv = a.axis(name) * direction, b.axis(name) * direction   # normalize so larger is better
        if av < bv - 1e-12:
            no_worse_all = False
            break
        if av > bv + 1e-12:
            strictly_better_one = True
    return no_worse_all and strictly_better_one


class ParetoFront:
    """Reads named picks and the non-dominated set off a list of measured candidates."""

    def __init__(self, candidates: Sequence[ParetoCandidate], axes: Optional[Sequence[Axis]] = None):
        self.candidates: List[ParetoCandidate] = list(candidates)
        self.axes: List[Axis] = list(axes) if axes is not None else list(DEFAULT_AXES)

    def certified_only(self) -> List[ParetoCandidate]:
        return [c for c in self.candidates if c.certified]

    def nondominated(self) -> List[ParetoCandidate]:
        """The certified Pareto-optimal set under self.axes."""
        cert = self.certified_only()
        front: List[ParetoCandidate] = []
        for c in cert:
            if not any(_dominates(o, c, self.axes) for o in cert if o is not c):
                front.append(c)
        return front

    # -- named picks (all restricted to certified candidates) -------------------------------------------

    def fastest_certified(self) -> Optional[ParetoCandidate]:
        cert = self.certified_only()
        return min(cert, key=lambda c: c.latency_ms) if cert else None

    def most_accurate(self) -> Optional[ParetoCandidate]:
        cert = self.certified_only()
        return max(cert, key=lambda c: c.metric_lb) if cert else None

    def most_accurate_under_latency(self, budget_ms: float) -> Optional[ParetoCandidate]:
        elig = [c for c in self.certified_only() if c.latency_ms <= budget_ms]
        return max(elig, key=lambda c: c.metric_lb) if elig else None

    def best_calibrated(self) -> Optional[ParetoCandidate]:
        cert = self.certified_only()
        return min(cert, key=lambda c: c.ece) if cert else None

    def cheapest_certified(self) -> Optional[ParetoCandidate]:
        cert = self.certified_only()
        return min(cert, key=lambda c: c.cost_usd) if cert else None

    def picks(self, latency_budget_ms: Optional[float] = None) -> Dict[str, Optional[ParetoCandidate]]:
        """The canonical commercial pick set. Keys map to the user-facing question each answers."""
        out: Dict[str, Optional[ParetoCandidate]] = {
            "fastest_certified": self.fastest_certified(),
            "most_accurate": self.most_accurate(),
            "best_calibrated": self.best_calibrated(),
            "cheapest_certified": self.cheapest_certified(),
        }
        if latency_budget_ms is not None:
            out["most_accurate_under_latency"] = self.most_accurate_under_latency(latency_budget_ms)
        return out

    def report(self, latency_budget_ms: Optional[float] = None) -> str:
        front = self.nondominated()
        lines = [f"Certified Pareto front: {len(front)} of {len(self.candidates)} candidates certified+nondominated"]
        for c in sorted(front, key=lambda x: -x.metric_lb):
            lines.append(f"  {c.name}: metric_lb={c.metric_lb:.3f} latency={c.latency_ms:.1f}ms "
                         f"cost=${c.cost_usd:.4f} ece={c.ece:.3f}")
        for label, pick in self.picks(latency_budget_ms).items():
            lines.append(f"  pick[{label}] = {pick.name if pick else 'none'}")
        return "\n".join(lines)


__all__ = ["MAXIMIZE", "MINIMIZE", "Axis", "DEFAULT_AXES", "ParetoCandidate", "ParetoFront"]
