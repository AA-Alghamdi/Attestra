"""The PROBLEM ENVELOPE -- a single machine-readable specification of an ML problem.

WHY THIS EXISTS
---------------
Every problem the autoresearcher solves -- a frontier-novel question or a mid-level commercial task
(a 50 ms call-ender on 100 calls, a 10k-image biopsy classifier) -- is the SAME engine with a different
parameterization. The thing that changes per problem is only: (a) what the certifier's gate measures,
(b) which operators are admissible, and (c) the search policy. The Envelope captures (a) and the inputs
to (b)/(c) in one frozen, hashable object so the rest of the system can be problem-agnostic.

The Envelope is DESCRIPTIVE, not a gate. It never promotes anything. It is consumed by:
  * gate.certified_under_envelope  -- turns the constraints into measured, conjunctive promotion checks
    that can only make promotion STRICTER than the frozen certifier (never looser).
  * the search policy               -- reads the data regime / transfer sources to pick operators.
  * intake                          -- elicits/formalizes the real objective from a user's proxy ask.

CONTRACT: pure stdlib + dataclasses. No numpy, no estimator, no certifier import. Frozen, hashable,
serializable. The set of certifiable objective metrics is the SAME frozen list the certifier enforces
(vectorforge.science.KNOWN_METRICS); we mirror it here as a tuple to avoid importing numpy at spec time,
and assert_objective_certifiable cross-checks against the frozen core when available.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Optional, Tuple

# Mirror of vectorforge.science.KNOWN_METRICS (the only metrics with a frozen lower-bound certifier).
# Kept as a literal so building an Envelope never imports numpy; assert_objective_certifiable() does the
# authoritative cross-check against the frozen core.
CERTIFIABLE_METRICS: Tuple[str, ...] = (
    "accuracy", "balanced_accuracy", "macro_f1", "r2", "neg_rmse", "neg_mae",
)
CLASSIFICATION_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
REGRESSION_METRICS = ("r2", "neg_rmse", "neg_mae")

TASK_TYPES = ("binary", "multiclass", "regression", "ranking", "timeseries")
MODALITIES = ("tabular", "text", "vision", "audio", "timeseries")
SHIFTS = ("iid", "temporal", "grouped", "covariate")


class EnvelopeError(ValueError):
    """Raised when an envelope is internally inconsistent (e.g. an uncertifiable objective metric)."""


@dataclass(frozen=True)
class Objective:
    """What 'better' means. `metric` must be a frozen-certifiable metric. `theta` is the bar the
    certifier's LOWER bound must clear. `cost_weights` (optional) makes the objective cost-sensitive
    (e.g. an escalate-miss costs far more than a goodbye-miss in the call-ender example); they are used
    by the proposer/selection and by cost-weighted reporting, NEVER to change the frozen certifier."""
    metric: str
    theta: float
    cost_weights: Optional[Dict[str, float]] = None
    direction: str = "higher_is_better"   # all frozen metrics are higher-is-better by construction

    def __post_init__(self):
        if self.metric not in CERTIFIABLE_METRICS:
            raise EnvelopeError(
                f"objective metric {self.metric!r} is not frozen-certifiable {CERTIFIABLE_METRICS}; "
                f"the certifier would refuse it. Pick a certifiable metric or add a validated variant.")
        if self.direction != "higher_is_better":
            raise EnvelopeError("all frozen metrics are higher-is-better; theta is a lower-bound floor.")


@dataclass(frozen=True)
class Constraints:
    """HARD promotion constraints, every one MEASURED. gate.certified_under_envelope ANDs these around
    the frozen certificate, so they can only shrink the promotion set. A None field is 'no constraint'.

    per_class_recall_floor: {label: min_recall} -- refuse to promote unless EACH listed class clears its
        floor on the sealed test (e.g. {"escalate": 0.95}). The single most important commercial gate.
    subgroup_parity: {"attr": <row field>, "max_gap": <float>, "metric": <optional metric>} -- refuse to
        promote if the gap between the best and worst subgroup on `attr` exceeds max_gap (fairness)."""
    max_latency_ms: Optional[float] = None
    max_cost_usd: Optional[float] = None
    max_mem_mb: Optional[float] = None
    max_ece: Optional[float] = None
    per_class_recall_floor: Optional[Dict[str, float]] = None
    subgroup_parity: Optional[Dict[str, object]] = None

    def __post_init__(self):
        for nm in ("max_latency_ms", "max_cost_usd", "max_mem_mb", "max_ece"):
            v = getattr(self, nm)
            if v is not None and float(v) < 0:
                raise EnvelopeError(f"{nm} must be >= 0; got {v}")
        if self.per_class_recall_floor is not None:
            for lab, fl in self.per_class_recall_floor.items():
                if not (0.0 <= float(fl) <= 1.0):
                    raise EnvelopeError(f"per_class_recall_floor[{lab!r}]={fl} must be in [0,1]")
        if self.subgroup_parity is not None:
            sp = self.subgroup_parity
            if "attr" not in sp or "max_gap" not in sp:
                raise EnvelopeError("subgroup_parity needs at least {'attr':..., 'max_gap':...}")
            if not (0.0 <= float(sp["max_gap"]) <= 1.0):
                raise EnvelopeError("subgroup_parity.max_gap must be in [0,1]")

    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in
                   ("max_latency_ms", "max_cost_usd", "max_mem_mb", "max_ece",
                    "per_class_recall_floor", "subgroup_parity"))


@dataclass(frozen=True)
class DataRegime:
    """The shape of the available data. Drives operator admissibility (small n + transfer source ->
    foundation-model featurizer + tiny head; large n -> train-from-scratch is admissible) and which
    leak-safe split the engine must use (`shift`)."""
    n: int
    modality: str = "tabular"
    task_type: str = "binary"
    n_classes: Optional[int] = None
    label_cost: Optional[float] = None     # cost to acquire one more label -> active-labeling economics
    group_key: Optional[str] = None        # row field that must not straddle splits (patient/agent/customer)
    time_key: Optional[str] = None         # row field giving event order -> forward-chaining split
    shift: str = "iid"

    def __post_init__(self):
        if self.modality not in MODALITIES:
            raise EnvelopeError(f"modality {self.modality!r} not in {MODALITIES}")
        if self.task_type not in TASK_TYPES:
            raise EnvelopeError(f"task_type {self.task_type!r} not in {TASK_TYPES}")
        if self.shift not in SHIFTS:
            raise EnvelopeError(f"shift {self.shift!r} not in {SHIFTS}")
        if self.shift == "grouped" and not self.group_key:
            raise EnvelopeError("shift='grouped' requires a group_key")
        if self.shift == "temporal" and not self.time_key:
            raise EnvelopeError("shift='temporal' requires a time_key")


@dataclass(frozen=True)
class RiskPosture:
    """How costly mistakes are -> whether abstention is a first-class output, and how to weight FP vs FN."""
    abstention_allowed: bool = False
    regulatory: bool = False
    fp_fn_cost_ratio: Optional[float] = None


@dataclass(frozen=True)
class Envelope:
    """The full problem specification. Hashable + serializable; `digest()` content-addresses it so a
    certificate can be bound to the exact envelope it was earned under."""
    objective: Objective
    data_regime: DataRegime
    constraints: Constraints = field(default_factory=Constraints)
    risk_posture: RiskPosture = field(default_factory=RiskPosture)
    transfer_sources: Tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self):
        # cross-field consistency: a classification objective needs a classification task_type, etc.
        is_reg_obj = self.objective.metric in REGRESSION_METRICS
        is_reg_task = self.data_regime.task_type == "regression"
        if is_reg_obj != is_reg_task:
            raise EnvelopeError(
                f"objective metric {self.objective.metric!r} and task_type "
                f"{self.data_regime.task_type!r} disagree on regression-vs-classification.")
        if self.constraints.per_class_recall_floor and is_reg_task:
            raise EnvelopeError("per_class_recall_floor is meaningless for a regression task.")

    # ---- derived helpers used by the policy ------------------------------------------------------
    def is_regression(self) -> bool:
        return self.objective.metric in REGRESSION_METRICS

    def is_small_data(self, threshold: int = 2000) -> bool:
        return self.data_regime.n < threshold

    def wants_transfer(self) -> bool:
        """Small data OR a non-tabular modality => transfer (pretrained features) is the high-leverage
        operator class; search volume alone won't get there (the no-free-lunch boundary)."""
        return bool(self.transfer_sources) or self.is_small_data() or self.data_regime.modality in (
            "text", "vision", "audio")

    def as_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, **kw)

    def with_theta(self, theta: float) -> "Envelope":
        return replace(self, objective=replace(self.objective, theta=float(theta)))

    def digest(self) -> str:
        import hashlib
        return "env:" + hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def assert_objective_certifiable(envelope: Envelope) -> None:
    """Authoritative cross-check against the FROZEN core (imports numpy lazily). Raises if the objective
    metric is not one the frozen certifier accepts -- the single source of truth is the frozen list, this
    function guarantees the mirror in this module never drifts from it."""
    from vectorforge import science
    science.assert_certifiable_metric(envelope.objective.metric)


# ---- builders ------------------------------------------------------------------------------------
def from_goal(*, metric: str, theta: float, n: int, modality: str = "tabular",
              task_type: str = "binary", n_classes: Optional[int] = None,
              max_latency_ms: Optional[float] = None, max_cost_usd: Optional[float] = None,
              max_ece: Optional[float] = None, per_class_recall_floor: Optional[Dict[str, float]] = None,
              subgroup_parity: Optional[Dict[str, object]] = None,
              group_key: Optional[str] = None, time_key: Optional[str] = None, shift: str = "iid",
              cost_weights: Optional[Dict[str, float]] = None, abstention_allowed: bool = False,
              regulatory: bool = False, label_cost: Optional[float] = None,
              transfer_sources: Tuple[str, ...] = (), notes: str = "") -> Envelope:
    """Convenience constructor mirroring the loop's keyword surface (threshold/max_latency_ms/... already
    accepted by run_goal_loop), so an Envelope can be built straight from goal-intake arguments."""
    return Envelope(
        objective=Objective(metric=metric, theta=float(theta), cost_weights=cost_weights),
        data_regime=DataRegime(n=int(n), modality=modality, task_type=task_type, n_classes=n_classes,
                               label_cost=label_cost, group_key=group_key, time_key=time_key, shift=shift),
        constraints=Constraints(max_latency_ms=max_latency_ms, max_cost_usd=max_cost_usd, max_ece=max_ece,
                                per_class_recall_floor=per_class_recall_floor,
                                subgroup_parity=subgroup_parity),
        risk_posture=RiskPosture(abstention_allowed=abstention_allowed, regulatory=regulatory),
        transfer_sources=tuple(transfer_sources), notes=notes,
    )


def from_dict(d: dict) -> Envelope:
    """Rehydrate an Envelope from `as_dict()` output (e.g. read back from a stored certificate)."""
    obj = d["objective"]
    dr = d["data_regime"]
    co = d.get("constraints") or {}
    rp = d.get("risk_posture") or {}
    return Envelope(
        objective=Objective(metric=obj["metric"], theta=obj["theta"],
                            cost_weights=obj.get("cost_weights"),
                            direction=obj.get("direction", "higher_is_better")),
        data_regime=DataRegime(**dr),
        constraints=Constraints(**co),
        risk_posture=RiskPosture(**rp),
        transfer_sources=tuple(d.get("transfer_sources") or ()),
        notes=d.get("notes", ""),
    )


__all__ = ["Envelope", "Objective", "Constraints", "DataRegime", "RiskPosture", "EnvelopeError",
           "from_goal", "from_dict", "assert_objective_certifiable", "CERTIFIABLE_METRICS",
           "CLASSIFICATION_METRICS", "REGRESSION_METRICS", "TASK_TYPES", "MODALITIES", "SHIFTS"]
