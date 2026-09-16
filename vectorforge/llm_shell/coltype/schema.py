"""The TypeProposal contract the LLM emits for A2 (only for the ambiguous residual columns)."""
from dataclasses import dataclass, field

from .profile_conf import COLUMN_DTYPES

TYPE_PROPOSAL_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["columns"],
    "properties": {
        "columns": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "dtype"],
                "properties": {
                    "name": {"type": "string", "description": "An ambiguous column name from the prompt."},
                    "dtype": {"type": "string", "enum": list(COLUMN_DTYPES),
                              "description": "Proposed type. Validated against the real values; an "
                                             "unsupported claim is dropped to the structural type."},
                    "normalization_hint": {"type": ["string", "null"],
                                           "description": "Optional human note (e.g. 'strip $ and "
                                                          "commas'). Advisory only; not executed."},
                },
            },
        },
    },
}

PROMPT_VERSION = "a2-coltype/v1"


@dataclass
class TypeProposal:
    columns: list            # [{name, dtype, normalization_hint}]
    source: str = "llm"

    @staticmethod
    def validate(obj, *, source="llm"):
        if not isinstance(obj, dict):
            raise ValueError("TypeProposal must be an object")
        cols = []
        for c in (obj.get("columns", []) or []):
            if isinstance(c, dict) and isinstance(c.get("name"), str) and c.get("dtype") in COLUMN_DTYPES:
                cols.append({"name": c["name"], "dtype": c["dtype"],
                             "normalization_hint": (str(c["normalization_hint"])
                                                    if c.get("normalization_hint") else None)})
        return TypeProposal(columns=cols, source=source)

    def as_dict(self):
        return {"columns": list(self.columns), "source": self.source}
