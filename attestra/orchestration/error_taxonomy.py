"""Error taxonomy: classify experiment failures and route to structured recovery.

Every failure in the research cycle is classified into a taxonomy so the system
can take the RIGHT recovery action instead of just logging "failed, moving on."

Taxonomy (ordered by severity):
  INFRASTRUCTURE  - environment/resource issues (OOM, timeout, missing dep)
  DATA            - data quality issues (missing values, leakage, insufficient samples)
  NUMERICAL       - numerical instability (NaN, inf, degenerate distributions)
  MODEL           - model-specific failures (non-convergence, bad architecture for data)
  CODE            - code generation failures (syntax, import, runtime errors)
  EVALUATION      - evaluation issues (metric mismatch, empty predictions)
  BUDGET          - resource exhaustion (time, compute, API calls)

Each category maps to a recovery strategy the cycle can attempt autonomously.
"""
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ErrorCategory(str, Enum):
    INFRASTRUCTURE = "infrastructure"
    DATA = "data"
    NUMERICAL = "numerical"
    MODEL = "model"
    CODE = "code"
    EVALUATION = "evaluation"
    BUDGET = "budget"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    FATAL = "fatal"          # cannot recover, must abort this approach
    RECOVERABLE = "recoverable"  # can try a different strategy
    TRANSIENT = "transient"  # retry may succeed


@dataclass
class ClassifiedError:
    """A classified experiment failure with recovery guidance."""
    category: ErrorCategory
    severity: Severity
    message: str
    original_error: Optional[str] = None
    recovery_actions: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fatal(self) -> bool:
        return self.severity == Severity.FATAL

    @property
    def is_recoverable(self) -> bool:
        return self.severity in (Severity.RECOVERABLE, Severity.TRANSIENT)


# ============================================================================== patterns

_INFRA_PATTERNS = [
    (re.compile(r"out of memory|OOM|CUDA out of memory|MemoryError", re.I), "memory_exhaustion"),
    (re.compile(r"timeout|timed? ?out|deadline exceeded", re.I), "timeout"),
    (re.compile(r"No module named|ModuleNotFoundError|ImportError", re.I), "missing_dependency"),
    (re.compile(r"ConnectionError|ConnectionRefused|NetworkError", re.I), "network_failure"),
    (re.compile(r"PermissionError|Permission denied", re.I), "permission_denied"),
    (re.compile(r"disk full|No space left|OSError.*space", re.I), "disk_full"),
]

_DATA_PATTERNS = [
    (re.compile(r"found (input|array) with 0 (sample|feature)", re.I), "empty_data"),
    (re.compile(r"NaN|nan.*input|input.*nan|missing values|contains? NaN", re.I), "missing_values"),
    (re.compile(r"Found input variables with inconsistent", re.I), "shape_mismatch"),
    (re.compile(r"leakage|leak|contamination", re.I), "data_leakage"),
    (re.compile(r"too few samples|not enough|insufficient.*data|need at least", re.I), "insufficient_data"),
    (re.compile(r"class.*not.*present|unknown label|unseen.*label", re.I), "label_issue"),
]

_NUMERICAL_PATTERNS = [
    (re.compile(r"overflow|underflow|inf|infinity.*encountered", re.I), "overflow"),
    (re.compile(r"singular matrix|LinAlgError|ill-conditioned", re.I), "singular_matrix"),
    (re.compile(r"convergence|did not converge|failed to converge|ConvergenceWarning", re.I), "non_convergence"),
    (re.compile(r"divide by zero|ZeroDivisionError", re.I), "division_by_zero"),
    (re.compile(r"degenerate|constant feature|zero variance", re.I), "degenerate_distribution"),
]

_MODEL_PATTERNS = [
    (re.compile(r"not fitted|NotFittedError|fit.*before.*predict", re.I), "not_fitted"),
    (re.compile(r"predict_proba.*not available|has no.*predict_proba", re.I), "missing_predict_proba"),
    (re.compile(r"multiclass.*not supported|binary.*only", re.I), "task_mismatch"),
    (re.compile(r"max_iter.*reached|IterationLimit", re.I), "iteration_limit"),
    (re.compile(r"negative.*dimension|invalid.*shape", re.I), "architecture_error"),
]

_CODE_PATTERNS = [
    (re.compile(r"SyntaxError", re.I), "syntax_error"),
    (re.compile(r"NameError.*not defined", re.I), "undefined_name"),
    (re.compile(r"TypeError.*argument|TypeError.*expected", re.I), "type_error"),
    (re.compile(r"AttributeError.*has no attribute", re.I), "attribute_error"),
    (re.compile(r"build_estimator.*not.*defined|does not define", re.I), "missing_entry_point"),
    (re.compile(r"IndentationError|TabError", re.I), "indentation_error"),
]

_EVAL_PATTERNS = [
    (re.compile(r"empty.*prediction|no prediction|zero.*predictions", re.I), "empty_predictions"),
    (re.compile(r"metric.*not.*supported|unknown.*metric", re.I), "unsupported_metric"),
    (re.compile(r"shape.*mismatch.*prediction|prediction.*length", re.I), "prediction_shape_mismatch"),
]

# ============================================================================== recovery

_RECOVERY_MAP: Dict[str, List[str]] = {
    # infrastructure
    "memory_exhaustion": ["reduce_batch_size", "reduce_model_size", "subsample_data", "switch_to_lighter_model"],
    "timeout": ["increase_timeout", "simplify_model", "reduce_data_size"],
    "missing_dependency": ["use_available_alternative", "skip_this_approach"],
    "network_failure": ["retry_with_backoff", "use_offline_fallback"],
    "permission_denied": ["skip_this_approach"],
    "disk_full": ["cleanup_checkpoints", "skip_this_approach"],
    # data
    "empty_data": ["check_data_loading", "skip_this_dataset"],
    "missing_values": ["impute_missing", "drop_missing_rows", "use_nan_tolerant_model"],
    "shape_mismatch": ["verify_splits", "realign_data"],
    "data_leakage": ["remove_leaky_features", "rebuild_splits"],
    "insufficient_data": ["reduce_model_complexity", "use_transfer_learning", "augment_data"],
    "label_issue": ["relabel", "filter_unknown_labels", "use_all_labels"],
    # numerical
    "overflow": ["add_scaling", "reduce_learning_rate", "clip_gradients"],
    "singular_matrix": ["add_regularization", "use_pseudoinverse", "add_ridge"],
    "non_convergence": ["increase_iterations", "reduce_learning_rate", "try_different_optimizer"],
    "division_by_zero": ["add_epsilon", "check_for_constant_features"],
    "degenerate_distribution": ["remove_constant_features", "add_noise"],
    # model
    "not_fitted": ["ensure_fit_before_predict", "check_pipeline_order"],
    "missing_predict_proba": ["use_predict_instead", "wrap_with_calibration"],
    "task_mismatch": ["switch_to_compatible_model", "adjust_task_type"],
    "iteration_limit": ["increase_max_iter", "reduce_complexity"],
    "architecture_error": ["simplify_architecture", "check_dimensions"],
    # code
    "syntax_error": ["regenerate_code", "fix_syntax"],
    "undefined_name": ["add_missing_import", "regenerate_code"],
    "type_error": ["fix_argument_types", "regenerate_code"],
    "attribute_error": ["fix_attribute_access", "regenerate_code"],
    "missing_entry_point": ["regenerate_with_entry_point"],
    "indentation_error": ["fix_indentation", "regenerate_code"],
    # evaluation
    "empty_predictions": ["check_model_output", "ensure_predict_called"],
    "unsupported_metric": ["use_supported_metric"],
    "prediction_shape_mismatch": ["reshape_predictions", "check_output_format"],
}


def classify_error(error: BaseException, *, context: Optional[Dict] = None) -> ClassifiedError:
    """Classify an error into the taxonomy and suggest recovery actions.

    Args:
        error: The exception that occurred
        context: Optional context (e.g., which stage, what data shape, what model)

    Returns:
        ClassifiedError with category, severity, and recovery actions
    """
    msg = str(error)
    tb = traceback.format_exception(type(error), error, error.__traceback__)
    full_text = msg + "\n" + "".join(tb[-3:]) if tb else msg

    # Check each pattern category in order
    for patterns, category, severity_default in [
        (_INFRA_PATTERNS, ErrorCategory.INFRASTRUCTURE, Severity.RECOVERABLE),
        (_DATA_PATTERNS, ErrorCategory.DATA, Severity.RECOVERABLE),
        (_NUMERICAL_PATTERNS, ErrorCategory.NUMERICAL, Severity.RECOVERABLE),
        (_MODEL_PATTERNS, ErrorCategory.MODEL, Severity.RECOVERABLE),
        (_CODE_PATTERNS, ErrorCategory.CODE, Severity.RECOVERABLE),
        (_EVAL_PATTERNS, ErrorCategory.EVALUATION, Severity.RECOVERABLE),
    ]:
        for pattern, sub_type in patterns:
            if pattern.search(full_text):
                severity = severity_default
                # Some sub-types are always fatal
                if sub_type in ("permission_denied", "disk_full", "data_leakage"):
                    severity = Severity.FATAL
                # Some are always transient
                if sub_type in ("network_failure", "timeout"):
                    severity = Severity.TRANSIENT
                return ClassifiedError(
                    category=category,
                    severity=severity,
                    message=f"{category.value}/{sub_type}: {msg[:200]}",
                    original_error=full_text[:500],
                    recovery_actions=_RECOVERY_MAP.get(sub_type, ["skip_this_approach"]),
                    context=context or {},
                )

    # Budget errors (check separately since they're often custom)
    if isinstance(error, (TimeoutError,)):
        return ClassifiedError(
            category=ErrorCategory.BUDGET,
            severity=Severity.FATAL,
            message=f"budget/timeout: {msg[:200]}",
            original_error=full_text[:500],
            recovery_actions=["reduce_scope", "increase_budget"],
            context=context or {},
        )

    # Unknown
    return ClassifiedError(
        category=ErrorCategory.UNKNOWN,
        severity=Severity.RECOVERABLE,
        message=f"unknown: {msg[:200]}",
        original_error=full_text[:500],
        recovery_actions=["regenerate_code", "try_different_approach"],
        context=context or {},
    )


def classify_error_string(error_str: str, *, context: Optional[Dict] = None) -> ClassifiedError:
    """Classify from an error string (when we don't have the exception object)."""
    for patterns, category, severity_default in [
        (_INFRA_PATTERNS, ErrorCategory.INFRASTRUCTURE, Severity.RECOVERABLE),
        (_DATA_PATTERNS, ErrorCategory.DATA, Severity.RECOVERABLE),
        (_NUMERICAL_PATTERNS, ErrorCategory.NUMERICAL, Severity.RECOVERABLE),
        (_MODEL_PATTERNS, ErrorCategory.MODEL, Severity.RECOVERABLE),
        (_CODE_PATTERNS, ErrorCategory.CODE, Severity.RECOVERABLE),
        (_EVAL_PATTERNS, ErrorCategory.EVALUATION, Severity.RECOVERABLE),
    ]:
        for pattern, sub_type in patterns:
            if pattern.search(error_str):
                severity = severity_default
                if sub_type in ("permission_denied", "disk_full", "data_leakage"):
                    severity = Severity.FATAL
                if sub_type in ("network_failure", "timeout"):
                    severity = Severity.TRANSIENT
                return ClassifiedError(
                    category=category,
                    severity=severity,
                    message=f"{category.value}/{sub_type}: {error_str[:200]}",
                    original_error=error_str[:500],
                    recovery_actions=_RECOVERY_MAP.get(sub_type, ["skip_this_approach"]),
                    context=context or {},
                )
    return ClassifiedError(
        category=ErrorCategory.UNKNOWN,
        severity=Severity.RECOVERABLE,
        message=f"unknown: {error_str[:200]}",
        original_error=error_str[:500],
        recovery_actions=["regenerate_code", "try_different_approach"],
        context=context or {},
    )


class ErrorTracker:
    """Tracks errors across the research cycle for pattern learning."""

    def __init__(self) -> None:
        self.errors: List[ClassifiedError] = []
        self._counts: Dict[str, int] = {}

    def record(self, error: ClassifiedError) -> None:
        self.errors.append(error)
        key = f"{error.category.value}"
        self._counts[key] = self._counts.get(key, 0) + 1

    @property
    def total(self) -> int:
        return len(self.errors)

    def count_by_category(self) -> Dict[str, int]:
        return dict(self._counts)

    def most_common_category(self) -> Optional[ErrorCategory]:
        if not self._counts:
            return None
        return ErrorCategory(max(self._counts, key=self._counts.get))

    def fatal_count(self) -> int:
        return sum(1 for e in self.errors if e.is_fatal)

    def recovery_suggestions(self) -> List[str]:
        """Aggregate recovery suggestions weighted by frequency."""
        action_counts: Dict[str, int] = {}
        for e in self.errors:
            if e.is_recoverable:
                for action in e.recovery_actions:
                    action_counts[action] = action_counts.get(action, 0) + 1
        return sorted(action_counts, key=action_counts.get, reverse=True)[:5]

    def summary(self) -> Dict[str, Any]:
        return {
            "total_errors": self.total,
            "fatal_errors": self.fatal_count(),
            "by_category": self.count_by_category(),
            "top_recovery_actions": self.recovery_suggestions(),
        }
