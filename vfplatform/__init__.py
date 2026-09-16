"""VectorForge platform layer (LEGACY — use attestra instead).

This package is maintained for backward compatibility only.
All active development is in the `attestra` package (primary) and `frontier` (engine).

Previously: Ring 1 multi-round /goal loop + tracking + compute providers + sealed-test moat guard.
"""
import warnings as _warnings

_warnings.warn(
    "vfplatform is deprecated. Use `attestra` instead: "
    "from attestra import orchestrate",
    DeprecationWarning,
    stacklevel=2,
)
from .tracking import ExperimentStore, RunRecord
from .providers import (Provider, LocalCpuProvider, LocalWorkerProvider, RunPodProvider,
                        RunPodPodProvider, ResourceGated, Job)
from .leaderboard import Run, Leaderboard
from .harness import (Harness, harness_for, Move, TabularClassificationHarness,
                      TextClassificationHarness, TabularRegressionHarness)
from .voi import CaseBase, voi_rank
from .checkpoint import Checkpoint, CheckpointRequired
from .sealed import (SealedTest, certify_on_sealed, assert_supported_metric, MetricNotCertifiable,
                     PeekViolation, SUPPORTED_METRICS)
from .loop import run_goal_loop, GoalLoopResult, reaudit
from .connectors import load_sklearn, load_openml
from .provenance import verify_provenance, assert_provenance, ProvenanceError
from .frontdoor import run as run_goal

__all__ = ["ExperimentStore", "RunRecord", "Provider", "LocalCpuProvider", "LocalWorkerProvider",
           "RunPodProvider", "RunPodPodProvider", "ResourceGated", "Job", "Run", "Leaderboard", "Harness",
           "harness_for", "Move",
           "TabularClassificationHarness", "TextClassificationHarness", "TabularRegressionHarness",
           "CaseBase", "voi_rank", "Checkpoint", "CheckpointRequired", "SealedTest", "certify_on_sealed",
           "assert_supported_metric", "MetricNotCertifiable", "PeekViolation", "SUPPORTED_METRICS",
           "run_goal_loop", "GoalLoopResult", "reaudit", "load_sklearn", "load_openml",
           "verify_provenance", "assert_provenance", "ProvenanceError", "run_goal"]
