"""The MULTI-OBJECTIVE PROMOTION GATE -- a conjunctive layer that wraps the frozen certificate.

THE LOAD-BEARING INVARIANT (D2)
-------------------------------
This module NEVER recomputes, relaxes, or replaces the frozen statistical certificate. It takes the
certificate that the frozen core already produced (the Clopper-Pearson / bootstrap lower bound clearing
theta) and ANDs additional MEASURED constraints around it:

    certified_under_envelope  ==  base_cert["certified"]                      # FROZEN, untouched
                                  AND every hard constraint provably holds     # measured, conjunctive

Because the composition is a conjunction, the gate's decision is ALWAYS a subset of the frozen decision:
it can only make promotion STRICTER, never looser. `assert_only_tightens` proves this property, and the
test-suite checks it over randomized inputs. A constraint can therefore never mint a certificate the
frozen gate would not have minted.

This generalizes the latency/cost/ECE gate already inlined in loop.py (loop.py:970-989) into a reusable,
testable function that also enforces per-class recall floors and subgroup parity -- the constraints that
matter for mid-level commercial problems (e.g. "never miss an escalation": recall floor on the escalate
class; "equal across skin tones": subgroup parity) -- and attaches an honest certificate SCOPE (D7).

STRICTNESS ON UNMEASURED CONSTRAINTS (the honest default)
---------------------------------------------------------
If the envelope specifies a constraint but no measurement is supplied for it, the default is to REFUSE to
certify it (`strict_unmeasured=True`): we will not claim a latency/recall/parity bound holds without
having measured it. This is stricter than the legacy loop (which gave unmeasured constraints the benefit
of the doubt) and is the correct posture for a certify-or-honest-fail system. It still only tightens.

CONTRACT: this module imports the frozen `science` for per-class recall and metric scoring only (read
helpers, not the certifier). It produces no certificate of its own; it returns a DECISION about an
already-produced frozen certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .envelope import Constraints, Envelope

# status values for a single constraint check
PASS = "pass"
FAIL = "fail"
NOT_APPLICABLE = "not_applicable"   # the envelope does not impose this constraint
UNMEASURED = "unmeasured"           # the envelope imposes it but no measurement was supplied

# certificate scope tags (D7)
SCOPE_IN_DISTRIBUTION = "in_distribution"
SCOPE_SHIFT_ROBUST = "shift_robust"
SCOPE_SCOPED = "scoped"
SCOPES = (SCOPE_IN_DISTRIBUTION, SCOPE_SHIFT_ROBUST, SCOPE_SCOPED)


@dataclass
class Measurements:
    """Everything measured ON THE SEALED TEST about the winning candidate, used to check constraints.

    The metric bound itself lives in the frozen `base_cert`; these are the EXTRA measured quantities the
    multi-objective gate needs. y_true/y_pred/labels are the sealed-test pairing (already produced for the
    certificate) so per-class recall and subgroup metrics are computed on the SAME locked evaluation -- no
    extra peek. subgroup_values is aligned 1:1 with y_true/y_pred and holds each row's value for the
    subgroup attribute named in constraints.subgroup_parity."""
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    mem_mb: Optional[float] = None
    ece: Optional[float] = None
    y_true: Optional[List] = None
    y_pred: Optional[List] = None
    labels: Optional[List] = None
    subgroup_values: Optional[List] = None
    scope: str = SCOPE_IN_DISTRIBUTION


@dataclass
class ConstraintResult:
    name: str
    status: str
    measured: Optional[float] = None
    threshold: Optional[float] = None
    detail: str = ""

    @property
    def satisfied(self) -> bool:
        # A constraint is satisfied for PROMOTION iff it passed or does not apply. UNMEASURED and FAIL both
        # block promotion (UNMEASURED blocks because we will not assert an unmeasured bound holds).
        return self.status in (PASS, NOT_APPLICABLE)


@dataclass
class CertDecision:
    """The composed promotion decision. `certified` is the conjunction; `base_certified` is the frozen
    certificate's own verdict (always >= `certified`)."""
    certified: bool
    base_certified: bool
    constraints: Dict[str, ConstraintResult] = field(default_factory=dict)
    scope: str = SCOPE_IN_DISTRIBUTION
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "certified": self.certified,
            "base_certified": self.base_certified,
            "scope": self.scope,
            "reason": self.reason,
            "constraints": {k: {"status": v.status, "measured": v.measured,
                                "threshold": v.threshold, "detail": v.detail}
                            for k, v in self.constraints.items()},
        }


def _num_check(name, measured, threshold, *, upper: bool, strict_unmeasured: bool) -> ConstraintResult:
    """Generic numeric constraint. upper=True means measured must be <= threshold (latency/cost/mem/ece)."""
    if threshold is None:
        return ConstraintResult(name, NOT_APPLICABLE)
    if measured is None:
        status = FAIL if strict_unmeasured else PASS
        return ConstraintResult(name, UNMEASURED if status == FAIL else PASS, None, float(threshold),
                                "no measurement supplied for a specified constraint")
    ok = (float(measured) <= float(threshold)) if upper else (float(measured) >= float(threshold))
    return ConstraintResult(name, PASS if ok else FAIL, float(measured), float(threshold),
                            f"measured {measured} {'<=' if upper else '>='} {threshold} -> {ok}")


def _subgroup_gap(metric, y_true, y_pred, subgroup_values, labels, min_support=10):
    """Best-minus-worst gap of `metric` across subgroups with at least `min_support` rows. Returns
    (gap, per_group) or (None, {}) when it cannot be computed (too few groups with support)."""
    from vectorforge import science
    groups: Dict[object, List[int]] = {}
    for i, g in enumerate(subgroup_values):
        groups.setdefault(g, []).append(i)
    per = {}
    for g, idx in groups.items():
        if len(idx) < min_support:
            continue
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        labs = labels or sorted({str(x) for x in yt} | {str(x) for x in yp})
        try:
            per[g] = float(science.score_metric(metric, [str(v) for v in yt], [str(v) for v in yp], labs))
        except Exception:  # noqa: BLE001  fall back to accuracy if the metric needs labels we lack
            per[g] = science.accuracy([str(v) for v in yt], [str(v) for v in yp])
    if len(per) < 2:
        return None, per
    return max(per.values()) - min(per.values()), per


def check_constraints(measurements: Measurements, constraints: Constraints, *,
                      objective_metric: str = "accuracy",
                      strict_unmeasured: bool = True) -> Dict[str, ConstraintResult]:
    """Evaluate each hard constraint to PASS / FAIL / NOT_APPLICABLE / UNMEASURED. Pure measurement; no
    promotion decision here (that is the conjunction in certified_under_envelope)."""
    out: Dict[str, ConstraintResult] = {}
    m = measurements
    out["latency"] = _num_check("latency", m.latency_ms, constraints.max_latency_ms,
                                upper=True, strict_unmeasured=strict_unmeasured)
    out["cost"] = _num_check("cost", m.cost_usd, constraints.max_cost_usd,
                             upper=True, strict_unmeasured=strict_unmeasured)
    out["mem"] = _num_check("mem", m.mem_mb, constraints.max_mem_mb,
                            upper=True, strict_unmeasured=strict_unmeasured)
    out["ece"] = _num_check("ece", m.ece, constraints.max_ece,
                            upper=True, strict_unmeasured=strict_unmeasured)

    # per-class recall floors (classification): EACH listed class must clear its floor on the sealed test.
    if constraints.per_class_recall_floor:
        if m.y_true is None or m.y_pred is None:
            out["per_class_recall"] = ConstraintResult(
                "per_class_recall", UNMEASURED if strict_unmeasured else PASS,
                detail="no sealed predictions supplied to measure per-class recall")
        else:
            from vectorforge import science
            labs = m.labels or sorted({str(x) for x in m.y_true})
            rec = science.per_class_recall([str(v) for v in m.y_true], [str(v) for v in m.y_pred], labs)
            worst = None
            failed = []
            for lab, floor in constraints.per_class_recall_floor.items():
                entry = rec.get(str(lab)) or rec.get(lab) or {}
                r = entry.get("recall")
                if r is None:                      # class absent from sealed test -> cannot measure
                    failed.append((str(lab), None, float(floor)))
                    continue
                if r < float(floor):
                    failed.append((str(lab), float(r), float(floor)))
                if worst is None or (r is not None and r < worst):
                    worst = r
            status = PASS if not failed else FAIL
            out["per_class_recall"] = ConstraintResult(
                "per_class_recall", status, worst, None,
                "all class floors cleared" if status == PASS else f"floors not cleared: {failed}")

    # subgroup parity: best-minus-worst metric gap across subgroups must be <= max_gap.
    if constraints.subgroup_parity:
        sp = constraints.subgroup_parity
        max_gap = float(sp["max_gap"])
        sg_metric = str(sp.get("metric") or objective_metric)
        if m.y_true is None or m.y_pred is None or m.subgroup_values is None:
            out["subgroup_parity"] = ConstraintResult(
                "subgroup_parity", UNMEASURED if strict_unmeasured else PASS, None, max_gap,
                "no subgroup_values / sealed predictions supplied")
        else:
            gap, per = _subgroup_gap(sg_metric, m.y_true, m.y_pred, m.subgroup_values, m.labels,
                                     min_support=int(sp.get("min_support", 10)))
            if gap is None:
                out["subgroup_parity"] = ConstraintResult(
                    "subgroup_parity", UNMEASURED if strict_unmeasured else PASS, None, max_gap,
                    "fewer than 2 subgroups with sufficient support")
            else:
                out["subgroup_parity"] = ConstraintResult(
                    "subgroup_parity", PASS if gap <= max_gap else FAIL, float(gap), max_gap,
                    f"subgroup metric={sg_metric} gap={round(gap,4)} vs max {max_gap}; per_group={per}")
    return out


def certified_under_envelope(base_cert: dict, measurements: Measurements, envelope: Envelope, *,
                             strict_unmeasured: bool = True) -> CertDecision:
    """Compose the FROZEN certificate with the envelope's hard constraints (conjunction; can only
    tighten). `base_cert` is the dict returned by the frozen certifier (must carry boolean 'certified').
    """
    base = bool(base_cert.get("certified", False))
    checks = check_constraints(measurements, envelope.constraints,
                               objective_metric=envelope.objective.metric,
                               strict_unmeasured=strict_unmeasured)
    all_ok = all(c.satisfied for c in checks.values())
    certified = base and all_ok
    if not base:
        reason = "frozen certificate did not clear theta (no constraint can rescue it)"
    elif certified:
        reason = "frozen certificate cleared theta AND all hard constraints hold"
    else:
        blocking = [c.name for c in checks.values() if not c.satisfied]
        reason = f"frozen certificate cleared theta but hard constraints blocked promotion: {blocking}"
    scope = measurements.scope if measurements.scope in SCOPES else SCOPE_IN_DISTRIBUTION
    return CertDecision(certified=certified, base_certified=base, constraints=checks,
                        scope=scope, reason=reason)


def assert_only_tightens(decision: CertDecision) -> None:
    """Invariant guard (D2): the composed decision can never be certified when the frozen base was not.
    Call this anywhere a decision is produced to make the monotonicity property a runtime assertion."""
    if decision.certified and not decision.base_certified:
        raise AssertionError(
            "INVARIANT VIOLATION: multi-objective gate certified a candidate the FROZEN certifier did "
            "not. The gate may only ever AND additional constraints (tighten), never relax the frozen "
            "decision. This is a bug that must be fixed, not worked around.")


__all__ = ["Measurements", "ConstraintResult", "CertDecision", "check_constraints",
           "certified_under_envelope", "assert_only_tightens",
           "PASS", "FAIL", "NOT_APPLICABLE", "UNMEASURED",
           "SCOPE_IN_DISTRIBUTION", "SCOPE_SHIFT_ROBUST", "SCOPE_SCOPED", "SCOPES"]
