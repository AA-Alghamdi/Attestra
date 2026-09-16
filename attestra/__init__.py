"""Attestra: Autonomous ML Research Executor.

Give it a goal and a verification standard. It plans, runs, iterates, and only
comes back with a verified result or an honest failure report.

Architecture:
  INTAKE -> PRE-REGISTRATION -> MODALITY ADAPTER -> EXPERIMENT DESIGN
  -> ORCHESTRATION -> RECURSIVE CYCLE -> CERTIFICATION -> LEDGER -> SELF-IMPROVEMENT

Standing invariants:
  (1) Only the frozen certifier promotes
  (2) Generalization expands what the LLM may propose, never what may promote
"""

__version__ = "0.4.0"

# Core: frozen statistical spine
from .core.science import (
    certify_accuracy,
    certify_regression,
    score_metric,
    make_splits,
    audit,
)

# Orchestration: single entry point
from .orchestration.orchestrator import (
    orchestrate,
    orchestrate_from_text,
    orchestrate_with_strategy_loop,
    OrchestrateConfig,
    OrchestrateResult,
)

# Cycle: the unified research engine
from .cycle.engine import ResearchEngine, CycleResult

# Ledger: experiment registry
from .ledger.registry import ExperimentRecord, ExperimentRegistry

# Profiling: data inspection
from .intake.profiler import profile_data, DataProfile

# Intake: problem typing
from .intake.problem_typing import type_problem, ProblemSpec, ProblemDomain, adversarial_check

# Design: experiment planning
from .design.experiment_plan import design_experiment, ExperimentPlan

# Design: tradeoff engine
from .design.tradeoff import TradeoffEngine, DeploymentConstraints

# Error taxonomy
from .orchestration.error_taxonomy import classify_error, ErrorCategory

# Health monitoring
from .orchestration.health import HealthMonitor, HealthStatus

# Portfolio: parallel execution
from .orchestration.portfolio import Portfolio, PortfolioResult

# Experiment management: complex experiments
from .orchestration.experiment_manager import ExperimentManager, ExperimentComplexity

# Self-improvement
from .improvement.strategy_learner import StrategyLearner
from .improvement.meta_learner import MetaLearner

# Certification: FDR control
from .certification.fdr import FDRController, ExperimentCertificate

# Execution: checkpoint/resume
from .execution.checkpoint import CheckpointManager, ExperimentState

# Augmentation: data augmentation + synthetic generation
from .augmentation.augmentation import DataAugmenter

# Research: literature search
from .research.literature import LiteratureSearch, ResearchProposer

# Adapters
from .adapters.base import get_adapter, available_modalities

# Strategy loop (Loop 2)
from .orchestration.strategy_loop import (
    run_strategy_loop, Strategy, StrategyDiagnosis,
    StrategyLoopResult, StrategyLoopConfig, ESCALATION_LADDER,
)

# Data layer: versioning + data-as-variable experiments
from .data.versioning import DataVersion, DataContract, DataFingerprint, VersionStore
from .data.experiments import DataExperiment, DataTransform, TransformType

# Multi-level routing
from .routing.strategy_router import StrategyRouter, StrategyClass, StrategyRecommendation
from .routing.resource_router import ResourceRouter, ExecutionSubstrate
from .routing.verification_router import VerificationRouter, VerificationSuite

# Parallel execution
from .execution.parallel import ProposalPool, ProposalResult, ParallelASHA

# Self-improvement stack
from .improvement.self_improvement import (
    PromptEvolver, HarnessLibrary, OracleEvolver,
    StrategyGenerator, CompoundingMetrics,
)

# Observability
from .observability.events import EventEmitter, ExperimentEvent, EventType
from .observability.provenance import ProvenanceRecord, ProvenanceTracker

__all__ = [
    # core
    "certify_accuracy", "certify_regression", "score_metric", "make_splits", "audit",
    # orchestration
    "orchestrate", "orchestrate_from_text", "orchestrate_with_strategy_loop",
    "OrchestrateConfig", "OrchestrateResult",
    # strategy loop
    "run_strategy_loop", "Strategy", "StrategyDiagnosis",
    "StrategyLoopResult", "StrategyLoopConfig", "ESCALATION_LADDER",
    # cycle
    "ResearchEngine", "CycleResult",
    # ledger
    "ExperimentRecord", "ExperimentRegistry",
    # profiling
    "profile_data", "DataProfile",
    # intake
    "type_problem", "ProblemSpec", "ProblemDomain", "adversarial_check",
    # design
    "design_experiment", "ExperimentPlan",
    "TradeoffEngine", "DeploymentConstraints",
    # error taxonomy
    "classify_error", "ErrorCategory",
    # health
    "HealthMonitor", "HealthStatus",
    # portfolio
    "Portfolio", "PortfolioResult",
    # experiment management
    "ExperimentManager", "ExperimentComplexity",
    # self-improvement
    "StrategyLearner", "MetaLearner",
    "PromptEvolver", "HarnessLibrary", "OracleEvolver",
    "StrategyGenerator", "CompoundingMetrics",
    # certification
    "FDRController", "ExperimentCertificate",
    # execution
    "CheckpointManager", "ExperimentState",
    "ProposalPool", "ProposalResult", "ParallelASHA",
    # data
    "DataAugmenter",
    "DataVersion", "DataContract", "DataFingerprint", "VersionStore",
    "DataExperiment", "DataTransform", "TransformType",
    # routing
    "StrategyRouter", "StrategyClass", "StrategyRecommendation",
    "ResourceRouter", "ExecutionSubstrate",
    "VerificationRouter", "VerificationSuite",
    # observability
    "EventEmitter", "ExperimentEvent", "EventType",
    "ProvenanceRecord", "ProvenanceTracker",
    # research
    "LiteratureSearch", "ResearchProposer",
    # adapters
    "get_adapter", "available_modalities",
]
