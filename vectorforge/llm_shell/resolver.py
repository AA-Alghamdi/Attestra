"""Step 3 -- the FROZEN resolver: the bound on the LLM. It can only NARROW or REJECT, never invent.

Every field of the (non-binding) GoalProposal is checked against the deterministic ProfileView and the
frozen ontology. The output ResolvedSpec carries:
  * the accepted target / task_type / metric / drop_cols / constraints,
  * a `decisions` log (every accept/reject with a reason -- the audit trail),
  * `needs_human` + `human_review` items (unmapped phrases, conflicts, unenforceable constraints),
and crucially carries NO threshold: the threshold stays with the frozen `ar.spec.propose_spec`.

MOAT INVARIANTS enforced here (by construction, not by trusting the model):
  R1  target must be an EXACT real column, else reject -> structural candidate. LLM cannot name a column.
  R2  task_type: structural inference WINS on conflict when it is ontology-supported; LLM cannot override.
  R3  metric: mapped ONLY into ontology.valid_metrics for the resolved task_type; an unmappable intent
      falls to the ontology default. The LLM can never introduce a metric the certifier can't evaluate.
  R4  forbidden_fields -> drop_cols is SUBTRACTIVE-ONLY and can never include the target; a forbidden
      field that is not a real column is dropped and surfaced. Backstopped independently by science.audit.
  R5  constraints with a real binding are recorded; anything the system can't evaluate is marked
      `unenforceable` and shown to the human -- never silently honored.
  R6  no LLM output reaches the threshold or the certifier. ResolvedSpec has no threshold field.
"""
from dataclasses import dataclass, field
from typing import Optional

from . import _frozen
from .schema import GoalProposal, Constraints
from .profiler import ProfileView


# ----- intent -> metric mapping -------------------------------------------------------------------
# HEURISTIC (labeled as such): a documented, conservative mapping from free-text evaluation intent to a
# metric. It is bounded by R3 -- it can ONLY ever select among the ontology's valid_metrics for the
# resolved task_type; anything it would pick that is not valid is discarded and the ontology default is
# used. It is NOT ground truth and is NOT anchored to any answer sheet; it is a usability convenience
# whose worst case is "wrong-but-valid metric, surfaced to the human", never an uncertifiable metric.
_INTENT_RULES = [
    # (substring triggers, preferred metric, applies-to task_types)
    (("recall", "false negative", "false-negative", "missed", "catch", "sensitivity"),
     "macro_f1", ("binary", "multiclass")),
    (("balanced", "imbalance", "rare class", "across classes", "minority"),
     "balanced_accuracy", ("binary", "multiclass")),
    (("f1", "f-1", "f score", "f-score"), "macro_f1", ("binary", "multiclass")),
    (("accuracy", "accurate", "correct"), "accuracy", ("binary", "multiclass")),
    (("rmse", "root mean square", "squared error"), "neg_rmse", ("regression",)),
    (("mae", "absolute error"), "neg_mae", ("regression",)),
    (("r2", "r^2", "r squared", "variance explained", "explained variance"),
     "r2", ("regression",)),
]


@dataclass
class ResolvedSpec:
    target: Optional[str]
    task_type: Optional[str]
    metric: Optional[str]
    drop_cols: list                      # resolved forbidden fields (subtractive)
    constraints: dict                    # enforceable constraints, with their binding
    unenforceable_constraints: list      # constraints recorded but with no frozen evaluator
    decisions: list                      # [{field, action, detail}] -- the audit trail
    human_review: list                   # phrases/conflicts the human must see
    needs_human: bool
    proposal_source: str                 # llm | fallback
    supported: bool                      # structural support (false => downstream declines anyway)
    # NB: NO threshold field. The threshold is the frozen propose_spec's job (R6).

    def as_dict(self):
        return {
            "target": self.target, "task_type": self.task_type, "metric": self.metric,
            "drop_cols": list(self.drop_cols), "constraints": self.constraints,
            "unenforceable_constraints": list(self.unenforceable_constraints),
            "decisions": list(self.decisions), "human_review": list(self.human_review),
            "needs_human": bool(self.needs_human), "proposal_source": self.proposal_source,
            "supported": bool(self.supported),
        }


def _real_column(name, columns_lower):
    """Resolve a proposed name to the EXACT real column (case-insensitive), else None."""
    if name is None:
        return None
    return columns_lower.get(str(name).strip().lower())


def _derive_task_type(view, target):
    """Frozen structural task_type for a column from its OWN profile stats (mirrors compile()'s
    classification philosophy). Used when the LLM overrides the target, so the type is not stale.
    Non-numeric low-cardinality -> binary/multiclass; numeric high-cardinality -> regression."""
    stats = view.columns.get(target, {}) if isinstance(view.columns.get(target), dict) else {}
    n_unique = stats.get("n_unique", 0) or 0
    numeric = bool(stats.get("numeric", False))
    distinct = stats.get("distinct_frac", 0.0) or 0.0
    if n_unique == 2:
        return "binary"
    if not numeric:
        return "multiclass"
    # numeric: low-cardinality codes are classification; otherwise a continuous regression target
    if n_unique <= 20 and distinct < 0.5:
        return "multiclass"
    return "regression"


def resolve(proposal: GoalProposal, view: ProfileView) -> ResolvedSpec:
    columns = view.column_names
    columns_lower = {str(c).strip().lower(): c for c in columns}
    decisions, human_review = [], []
    needs_human = False

    # ---- R1: target -------------------------------------------------------------------------------
    proposed_target = _real_column(proposal.target, columns_lower)
    if proposal.target is not None and proposed_target is None:
        decisions.append({"field": "target", "action": "reject",
                          "detail": f"proposed target {proposal.target!r} is not a real column; "
                                    f"falling back to structural {view.candidate_target!r}"})
        human_review.append(f"target {proposal.target!r} did not resolve to a column")
        needs_human = True
        target = view.candidate_target
    elif proposed_target is not None:
        target = proposed_target
        if view.candidate_target and proposed_target != view.candidate_target:
            decisions.append({"field": "target", "action": "accept_override",
                              "detail": f"LLM target {proposed_target!r} differs from structural "
                                        f"{view.candidate_target!r}; LLM target is a real column, accepted "
                                        f"with conflict flag"})
            human_review.append(f"target: LLM chose {proposed_target!r}, structural inferred "
                                f"{view.candidate_target!r} -- confirm")
            needs_human = True
        else:
            decisions.append({"field": "target", "action": "accept",
                              "detail": f"{proposed_target!r} (agrees with structural)"})
    else:
        target = view.candidate_target
        decisions.append({"field": "target", "action": "default",
                          "detail": f"no LLM target; using structural {target!r}"})

    # ---- R2: task_type (structural wins when supported) ------------------------------------------
    # If the target was OVERRIDDEN, the structural task_type was inferred for a DIFFERENT column and is
    # stale -- re-derive it for the resolved target from that column's own profile (frozen, no LLM).
    if target == view.candidate_target:
        struct_tt = view.candidate_task_type
    else:
        struct_tt = _derive_task_type(view, target)
        decisions.append({"field": "task_type", "action": "rederive_for_target",
                          "detail": f"target overridden to {target!r}; structural task_type re-derived "
                                    f"as {struct_tt!r} from its profile (was {view.candidate_task_type!r})"})
    llm_tt = proposal.task_type
    if llm_tt and struct_tt and llm_tt != struct_tt:
        if _frozen.task_type_supported(struct_tt):
            task_type = struct_tt
            decisions.append({"field": "task_type", "action": "structural_wins",
                              "detail": f"LLM proposed {llm_tt!r}; structural {struct_tt!r} is "
                                        f"ontology-supported and wins on conflict"})
            human_review.append(f"task_type: LLM said {llm_tt!r}, structural said {struct_tt!r} "
                                f"(structural kept) -- confirm")
            needs_human = True
        else:
            # structural type is unsupported; honor the LLM type only if the ontology knows it
            if llm_tt in _frozen.KNOWN_TASK_TYPES:
                task_type = llm_tt
                decisions.append({"field": "task_type", "action": "accept_llm",
                                  "detail": f"structural {struct_tt!r} unsupported; LLM {llm_tt!r} adopted"})
            else:
                task_type = struct_tt
                decisions.append({"field": "task_type", "action": "keep_structural",
                                  "detail": f"LLM {llm_tt!r} unknown to ontology; kept structural {struct_tt!r}"})
    else:
        task_type = struct_tt
        decisions.append({"field": "task_type", "action": "accept",
                          "detail": f"{task_type!r}"})

    # ---- R3: metric (mapped ONLY into the valid set) ---------------------------------------------
    valid_metrics = _frozen.valid_metrics_for(task_type)
    default_metric = _frozen.default_metric_for(task_type)
    metric = default_metric
    intent = (proposal.metric_intent or "").strip().lower()
    if intent:
        # (a) the intent literally names a valid metric
        named = next((m for m in valid_metrics if m.lower() == intent or m.lower() in intent), None)
        if named:
            metric = named
            decisions.append({"field": "metric", "action": "accept_named",
                              "detail": f"intent {proposal.metric_intent!r} -> valid metric {named!r}"})
        else:
            # (b) heuristic intent mapping, bounded to the valid set
            mapped = None
            for triggers, pref, applies in _INTENT_RULES:
                if task_type in applies and any(t in intent for t in triggers) and pref in valid_metrics:
                    mapped = pref
                    break
            if mapped:
                metric = mapped
                decisions.append({"field": "metric", "action": "map_intent",
                                  "detail": f"intent {proposal.metric_intent!r} -> {mapped!r} "
                                            f"(heuristic, within valid set {list(valid_metrics)})"})
                human_review.append(f"metric: intent {proposal.metric_intent!r} mapped to {mapped!r} "
                                    f"(heuristic) -- confirm")
                needs_human = True
            else:
                metric = default_metric
                decisions.append({"field": "metric", "action": "unmappable_default",
                                  "detail": f"intent {proposal.metric_intent!r} did not map into "
                                            f"{list(valid_metrics)}; using default {default_metric!r}"})
                human_review.append(f"metric intent {proposal.metric_intent!r} could not be honored "
                                    f"(not a certifiable metric for {task_type}); used {default_metric!r}")
                needs_human = True
    else:
        decisions.append({"field": "metric", "action": "default",
                          "detail": f"no intent; ontology default {default_metric!r}"})

    # ---- R4: forbidden_fields -> drop_cols (subtractive, never the target) -----------------------
    drop_cols = []
    for f in proposal.forbidden_fields:
        col = _real_column(f, columns_lower)
        if col is None:
            decisions.append({"field": "forbidden_fields", "action": "drop_unknown",
                              "detail": f"{f!r} is not a real column; ignored"})
            human_review.append(f"forbidden field {f!r} did not resolve to a column")
            needs_human = True
            continue
        if target is not None and col == target:
            decisions.append({"field": "forbidden_fields", "action": "reject_target",
                              "detail": f"refused to forbid the TARGET column {col!r}"})
            human_review.append(f"goal asked to forbid {col!r} but that is the target -- conflict")
            needs_human = True
            continue
        if col not in drop_cols:
            drop_cols.append(col)
            decisions.append({"field": "forbidden_fields", "action": "accept",
                              "detail": f"exclude feature {col!r} (subtractive; backstopped by science.audit)"})

    # ---- R5: constraints (bind to a frozen evaluator or mark unenforceable) ----------------------
    constraints, unenforceable = {}, []
    c: Constraints = proposal.constraints
    if c.fairness is not None:
        gcol = _real_column(c.fairness.group_col, columns_lower)
        if gcol is None:
            unenforceable.append({"kind": "fairness",
                                  "reason": f"group column {c.fairness.group_col!r} is not a real column"})
            human_review.append(f"fairness group {c.fairness.group_col!r} did not resolve -- not enforced")
            needs_human = True
        else:
            # A real group column => a deterministic group-parity MEASUREMENT can be bound at certify time.
            # We record it as enforceable-by-measurement; the gate itself lives downstream (frozen), not here.
            constraints["fairness"] = {"group_col": gcol, "parity_metric": c.fairness.parity_metric,
                                       "enforcement": "measure_group_parity_at_certify (frozen, downstream)"}
            decisions.append({"field": "constraints.fairness", "action": "bind",
                              "detail": f"group_col {gcol!r} bound for downstream parity measurement"})
            human_review.append(f"fairness constraint on {gcol!r} recorded (measured, not yet a hard gate)")
            needs_human = True
    if c.latency_ms is not None:
        constraints["latency_ms"] = {"budget_ms": c.latency_ms,
                                     "enforcement": "check_vs_Artifact.latency_s (frozen, downstream)"}
        decisions.append({"field": "constraints.latency_ms", "action": "record",
                          "detail": f"{c.latency_ms} ms budget recorded for downstream check"})
    if c.cost_usd is not None:
        constraints["cost_usd"] = {"budget_usd": c.cost_usd,
                                   "enforcement": "check_vs_Artifact.cost_usd (frozen, downstream)"}
        decisions.append({"field": "constraints.cost_usd", "action": "record",
                          "detail": f"${c.cost_usd} budget recorded for downstream check"})

    # ---- unmapped phrases: always surfaced (the anti-silence guarantee) --------------------------
    for ph in proposal.unmapped_phrases:
        human_review.append(f"unmapped goal phrase: {ph!r}")
    if proposal.unmapped_phrases:
        needs_human = True

    return ResolvedSpec(
        target=target, task_type=task_type, metric=metric, drop_cols=drop_cols,
        constraints=constraints, unenforceable_constraints=unenforceable,
        decisions=decisions, human_review=human_review, needs_human=needs_human,
        proposal_source=proposal.source, supported=view.supported,
    )
