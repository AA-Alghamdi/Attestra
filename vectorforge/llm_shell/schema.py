"""The GoalProposal contract: the strict shape the LLM must emit, plus validation.

The LLM's ENTIRE output is a GoalProposal and nothing else. It is consumed only by the frozen
resolver (resolver.py) and a human. It never reaches propose_spec, audit, or certify_accuracy.

Design invariants encoded here:
  * There is NO `threshold` field. The threshold is the one number a model is judged against, and
    it is derived by the frozen `ar.spec.propose_spec` (measured) or entered by a human -- never by
    the LLM. Omitting the field from the schema makes "the LLM proposes a threshold" unrepresentable.
  * `unmapped_phrases` is REQUIRED (may be empty). This is the anti-silence guarantee: anything in
    the goal the model could not honor must be listed, converting today's silent keyword-drop into an
    explicit, human-reviewable record.
  * Every field is either NON-BINDING (resolved+human-approved) or DETERMINISTICALLY-VERIFIABLE
    (checked against the profiled columns / ontology). No field is free-form-binding.
"""
from dataclasses import dataclass, field
from typing import Optional


# JSON Schema handed to the Claude API as a tool input_schema. The API validates shape; the frozen
# resolver validates MEANING (membership in real columns / ontology). Both layers are required.
GOAL_PROPOSAL_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["target", "task_type", "metric_intent", "forbidden_fields",
                 "constraints", "unmapped_phrases", "rationale"],
    "properties": {
        "target": {
            "type": ["string", "null"],
            "description": "The column to predict, BY MEANING, resolved to an EXACT column name from "
                           "the provided schema. null if the goal does not name/imply one.",
        },
        "task_type": {
            "type": ["string", "null"],
            "enum": ["binary", "multiclass", "regression", "multilabel", "ranking", None],
            "description": "Task type implied by the goal. null to defer to the structural inference.",
        },
        "metric_intent": {
            "type": ["string", "null"],
            "description": "The user's evaluation INTENT in plain words (e.g. 'minimize missed fraud', "
                           "'balanced across classes', 'r2'). NOT a threshold. The resolver maps this "
                           "into the ontology's valid metric set; it can never introduce a new metric.",
        },
        "forbidden_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns the user said NOT to use (e.g. 'don't use zipcode'). Each must be a "
                           "real column name. Purely subtractive: these are excluded from features.",
        },
        "constraints": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "fairness": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "properties": {
                        "group_col": {"type": "string",
                                      "description": "Real column defining the protected groups."},
                        "parity_metric": {"type": ["string", "null"],
                                          "description": "e.g. 'demographic_parity', 'equalized_odds'."},
                    },
                    "required": ["group_col"],
                },
                "latency_ms": {"type": ["number", "null"],
                               "description": "Max inference latency budget in milliseconds, if stated."},
                "cost_usd": {"type": ["number", "null"],
                             "description": "Max cost budget in USD, if stated."},
            },
            "description": "Structured constraints extracted from prose. Recorded as metadata; each is "
                           "bound by a frozen evaluator where one exists, else marked unenforceable.",
        },
        "unmapped_phrases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "REQUIRED (may be empty). Every phrase in the goal the proposal could NOT "
                           "honor or map to a field. The anti-silence guarantee.",
        },
        "rationale": {
            "type": "string",
            "description": "One short paragraph: how the goal text maps to these fields.",
        },
    },
}

PROMPT_VERSION = "a1-goal-proposal/v1"


@dataclass
class FairnessConstraint:
    group_col: str
    parity_metric: Optional[str] = None


@dataclass
class Constraints:
    fairness: Optional[FairnessConstraint] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None

    @staticmethod
    def from_obj(obj):
        if not isinstance(obj, dict):
            return Constraints()
        f = obj.get("fairness")
        fc = None
        if isinstance(f, dict) and f.get("group_col"):
            fc = FairnessConstraint(group_col=str(f["group_col"]),
                                    parity_metric=(str(f["parity_metric"])
                                                   if f.get("parity_metric") else None))
        lat = obj.get("latency_ms")
        cost = obj.get("cost_usd")
        return Constraints(
            fairness=fc,
            latency_ms=(float(lat) if isinstance(lat, (int, float)) else None),
            cost_usd=(float(cost) if isinstance(cost, (int, float)) else None),
        )

    def as_dict(self):
        return {
            "fairness": ({"group_col": self.fairness.group_col,
                          "parity_metric": self.fairness.parity_metric}
                         if self.fairness else None),
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }


@dataclass
class GoalProposal:
    """The validated, typed LLM output. `source` records who produced it (llm | fallback)."""
    target: Optional[str]
    task_type: Optional[str]
    metric_intent: Optional[str]
    forbidden_fields: list
    constraints: Constraints
    unmapped_phrases: list
    rationale: str
    source: str = "llm"

    @staticmethod
    def validate(obj, *, source="llm"):
        """Coerce a raw dict (LLM tool output) into a GoalProposal. Raises ValueError on a shape that
        cannot be made schema-valid. Tolerant of missing optionals; strict on types and on the
        REQUIRED unmapped_phrases / forbidden_fields being lists.

        This is shape validation only. MEANING validation (column membership, ontology metric set)
        is the frozen resolver's job -- never trusted to the model."""
        if not isinstance(obj, dict):
            raise ValueError(f"GoalProposal must be an object, got {type(obj).__name__}")

        target = obj.get("target")
        if target is not None and not isinstance(target, str):
            raise ValueError("target must be a string or null")

        task_type = obj.get("task_type")
        if task_type is not None and task_type not in (
                "binary", "multiclass", "regression", "multilabel", "ranking"):
            # An out-of-enum task_type is dropped to null rather than rejecting the whole proposal;
            # the resolver would reject it anyway. Record nothing here -- defer to structural.
            task_type = None

        metric_intent = obj.get("metric_intent")
        if metric_intent is not None and not isinstance(metric_intent, str):
            metric_intent = None

        forbidden = obj.get("forbidden_fields", [])
        if not isinstance(forbidden, list):
            raise ValueError("forbidden_fields must be a list")
        forbidden = [str(x) for x in forbidden if isinstance(x, (str, int, float))]

        unmapped = obj.get("unmapped_phrases")
        if unmapped is None or not isinstance(unmapped, list):
            raise ValueError("unmapped_phrases is REQUIRED and must be a list (the anti-silence field)")
        unmapped = [str(x) for x in unmapped]

        rationale = obj.get("rationale", "")
        rationale = str(rationale) if rationale is not None else ""

        return GoalProposal(
            target=(str(target) if target is not None else None),
            task_type=task_type,
            metric_intent=metric_intent,
            forbidden_fields=forbidden,
            constraints=Constraints.from_obj(obj.get("constraints", {})),
            unmapped_phrases=unmapped,
            rationale=rationale,
            source=source,
        )

    def as_dict(self):
        return {
            "target": self.target,
            "task_type": self.task_type,
            "metric_intent": self.metric_intent,
            "forbidden_fields": list(self.forbidden_fields),
            "constraints": self.constraints.as_dict(),
            "unmapped_phrases": list(self.unmapped_phrases),
            "rationale": self.rationale,
            "source": self.source,
        }
