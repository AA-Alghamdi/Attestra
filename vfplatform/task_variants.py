"""TASK-TYPE = CERTIFIER VARIANT -- one engine, a different verification discipline per task type.

WHY THIS EXISTS
---------------
The promotion machinery is fixed (select-then-bound -> frozen Tier-3), but the LEAK-SAFE DISCIPLINE around
it must change with the task type: a timeseries problem needs a temporal split + embargo + block-bootstrap
(iid resampling is invalid under autocorrelation); a patient/agent-grouped problem needs whole-group splits;
a ranking problem needs query-grouped splits. This registry maps a task type / data regime to the right
split kind, resampling kind, and embargo requirement, so the same loop instantiates the correct certifier
variant instead of one-size-fits-all.

THE INVARIANT
-------------
A variant can only make verification MORE conservative (temporal/group splits and block-bootstrap are
strictly stricter than iid). It selects discipline; it never relaxes the frozen certifier or widens a bound.
When in doubt the chooser picks the stricter variant (group/temporal over iid).

CONTRACT: stdlib only; reads envelope.DataRegime (no certifier import).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .envelope import DataRegime

SPLIT_STRATIFIED = "stratified"
SPLIT_GROUP = "group"
SPLIT_TEMPORAL = "temporal"

BOOTSTRAP_IID = "iid"
BOOTSTRAP_BLOCK = "block"


@dataclass(frozen=True)
class TaskVariant:
    """The verification discipline for a task type: how to split, how to resample for the bound, whether an
    embargo gap is required, and the default metric."""
    task_type: str
    split_kind: str
    bootstrap_kind: str
    requires_embargo: bool
    default_metric: str
    notes: str


# base registry keyed by envelope TASK_TYPES
REGISTRY: Dict[str, TaskVariant] = {
    "binary": TaskVariant("binary", SPLIT_STRATIFIED, BOOTSTRAP_IID, False, "balanced_accuracy",
                          "class-balanced split; McNemar/iid bootstrap for the bound"),
    "multiclass": TaskVariant("multiclass", SPLIT_STRATIFIED, BOOTSTRAP_IID, False, "balanced_accuracy",
                              "stratified split keeps per-class proportions; per-class recall floors apply"),
    "regression": TaskVariant("regression", SPLIT_STRATIFIED, BOOTSTRAP_IID, False, "r2",
                              "random split; iid bootstrap on the residual metric"),
    "ranking": TaskVariant("ranking", SPLIT_GROUP, BOOTSTRAP_BLOCK, False, "ndcg",
                           "query-grouped split (a query's items never straddle); block-bootstrap by query"),
    "timeseries": TaskVariant("timeseries", SPLIT_TEMPORAL, BOOTSTRAP_BLOCK, True, "smape",
                              "forward-chaining temporal split + embargo gap; block-bootstrap preserves "
                              "autocorrelation (iid resampling is invalid)"),
}


def get_variant(task_type: str) -> TaskVariant:
    if task_type not in REGISTRY:
        raise KeyError(f"no task variant for {task_type!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[task_type]


def variant_for_regime(regime: DataRegime) -> TaskVariant:
    """Pick the discipline from the full data regime, not just the nominal task type. A declared time_key or
    temporal shift forces the TEMPORAL variant; a declared group_key or grouped shift forces the GROUP
    variant -- both strictly stricter than the task type's default. Ties break toward the stricter split."""
    base = get_variant(regime.task_type)
    if regime.time_key is not None or regime.shift == "temporal":
        return TaskVariant(regime.task_type, SPLIT_TEMPORAL, BOOTSTRAP_BLOCK, True, base.default_metric,
                           f"temporal structure declared (time_key/shift) -> {base.notes}")
    if regime.group_key is not None or regime.shift == "grouped":
        return TaskVariant(regime.task_type, SPLIT_GROUP, BOOTSTRAP_BLOCK, False, base.default_metric,
                           f"group structure declared (group_key/shift) -> whole-group split; {base.notes}")
    return base


__all__ = ["SPLIT_STRATIFIED", "SPLIT_GROUP", "SPLIT_TEMPORAL", "BOOTSTRAP_IID", "BOOTSTRAP_BLOCK",
           "TaskVariant", "REGISTRY", "get_variant", "variant_for_regime"]
