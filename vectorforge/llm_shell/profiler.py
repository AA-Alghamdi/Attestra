"""Step 1 -- the FROZEN structural profile. Deterministic, goal-free, no raw rows leave it.

Runs the frozen `ar.compiler.compile` WITHOUT the goal, and projects its inference_report down to a
ProfileView containing ONLY column names + aggregate statistics + the structural candidates. This is
the exact object the LLM prompt is built from. By construction it contains no individual row value,
which quarantines prompt-injection from untrusted data cells: a malicious cell cannot reach the model
because cells are never in the view.
"""
from dataclasses import dataclass, field
from typing import Optional

from . import _frozen


@dataclass
class ProfileView:
    supported: bool
    decline_reason: Optional[str]
    n_rows: Optional[int]
    modality: Optional[str]
    columns: dict                       # colname -> {numeric, n_unique, n_missing, median_tokens, ...}
    candidate_target: Optional[str]     # structural target (the LLM may agree or override)
    target_rule: Optional[str]
    candidate_task_type: Optional[str]  # structural task_type
    valid_metrics: tuple                # ontology valid metrics for the structural task_type
    default_metric: Optional[str]
    labels: Optional[list]
    raw_report: dict = field(default_factory=dict)   # full inference_report, for downstream/debug

    @property
    def column_names(self):
        return list(self.columns.keys())


def profile(file_or_records, *, seed=0, min_test_n=200, max_rows=None):
    """Run the frozen structural pass with NO goal and return a ProfileView.

    Robust to compile() declining or failing the task BUILD: as long as inference produced column
    profiles + a candidate target, we surface them (the LLM can still propose against them). Only a
    truly empty / unparseable input yields supported=False with no columns.
    """
    out = _frozen.compile(file_or_records, goal_hint=None, threshold=None,
                          seed=seed, min_test_n=min_test_n, max_rows=max_rows)
    report = out.get("inference_report", {}) or {}
    columns = report.get("columns", {}) or {}
    candidate_task_type = report.get("task_type")
    return ProfileView(
        supported=bool(out.get("supported")),
        decline_reason=out.get("decline_reason"),
        n_rows=report.get("n_rows"),
        modality=report.get("modality"),
        columns=columns,
        candidate_target=report.get("target_col"),
        target_rule=report.get("target_rule"),
        candidate_task_type=candidate_task_type,
        valid_metrics=_frozen.valid_metrics_for(candidate_task_type),
        default_metric=_frozen.default_metric_for(candidate_task_type),
        labels=report.get("labels"),
        raw_report=report,
    )
