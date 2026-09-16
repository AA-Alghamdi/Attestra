"""Core domain objects for the VectorForge autonomous ML execution system.

A Goal is the top-level object. Everything else (splits, data moves, candidates, certificate,
deployment) is a child of it. The Goal carries durable state so a run can resume after interruption.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# Goal lifecycle states.
DRAFT = "draft"
AWAITING_APPROVAL = "awaiting_approval"
RUNNING = "running"
NEEDS_INPUT = "needs_input"          # blocked on a human (labels, decision, spend)
PASSED = "passed"                    # certified
FAILED = "failed"                    # ran to budget/infeasible without certifying
BLOCKED = "blocked"                  # leakage / rigor risk
CANCELLED = "cancelled"


@dataclass
class VerificationSpec:
    """The success contract: how the goal is proven."""
    metric: str = "accuracy"                  # accuracy | balanced_accuracy | macro_f1 | ndcg
    threshold: float = 0.90
    max_latency_ms: float = 50.0
    max_model_size_mb: float = 5.0
    min_heldout_n: int = 200
    alpha: float = 0.05                        # false-certification bound
    required_eval_slices: list = field(default_factory=list)


@dataclass
class ExperimentSurface:
    """What the system is allowed to change."""
    model_families: bool = True
    feature_engineering: bool = True
    class_weighting: bool = True
    calibration: bool = True
    synthetic_data: bool = False               # off by default (costs money / approval)
    active_acquisition: bool = True
    label_requests: bool = True


@dataclass
class Budget:
    max_experiments: int = 60
    max_synth_examples: int = 0
    max_rounds: int = 8
    max_spend_usd: float = 0.0
    approve_spend: bool = False


@dataclass
class Certificate:
    decision: str = "do_not_certify"           # certified | weak_evidence | do_not_certify | blocked
    metric: str = "accuracy"
    observed: float = 0.0
    threshold: float = 0.0
    n: int = 0
    lower_bound: float = 0.0
    p_value: Optional[float] = None
    checks: int = 1
    latency_ms_p95: float = 0.0
    leakage_passed: bool = False
    reason: str = ""
    evidence_strength: str = "ASSERTED"        # CONFIRMED | WEAK | FAILED | ASSERTED


@dataclass
class Deployment:
    status: str = "none"                       # ready | none
    artifact_path: Optional[str] = None
    endpoint: Optional[str] = None
    smoke_ok: bool = False


@dataclass
class Goal:
    id: str
    name: str
    kind: str                                  # text | tabular
    labels: list
    objective: str = ""
    task_desc: str = ""
    verification: VerificationSpec = field(default_factory=VerificationSpec)
    surface: ExperimentSurface = field(default_factory=ExperimentSurface)
    budget: Budget = field(default_factory=Budget)
    label_meaning: Optional[dict] = None       # sealed; the model never sees it, the generator may

    # lifecycle
    status: str = DRAFT
    created_at: str = ""
    updated_at: str = ""

    # data (references to files in the goal's store dir, not inline, for big sets)
    raw_path: Optional[str] = None
    split_report: Optional[dict] = None

    # the research record (durable, resumable)
    plan: Optional[dict] = None
    research_state: dict = field(default_factory=dict)   # hypothesis, findings, decisions, memory
    dag: dict = field(default_factory=dict)              # per-stage state for resume
    leaderboard: list = field(default_factory=list)
    certificate: Optional[dict] = None
    deployment: Optional[dict] = None
    needs_input: Optional[dict] = None                   # the concrete human ask when NEEDS_INPUT
    failure_report: Optional[dict] = None
    evidence_log: list = field(default_factory=list)     # the full audit trail of decisions+evidence

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Goal":
        d = dict(d)
        d["verification"] = VerificationSpec(**d.get("verification", {}))
        d["surface"] = ExperimentSurface(**d.get("surface", {}))
        d["budget"] = Budget(**d.get("budget", {}))
        return Goal(**d)
