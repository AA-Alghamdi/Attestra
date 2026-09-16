"""The A2 frozen validator: a proposed dtype is accepted only if the column's ACTUAL values support it,
else it is dropped back to the structural dtype. The LLM can relabel within what the data permits; it
can never invent a type the values don't bear out.
"""
from dataclasses import dataclass, field

from .profile_conf import (normalize_numeric, column_values, rule_confidence, structural_dtype,
                           _parses_date, _plain_float, COLUMN_DTYPES)

# acceptance thresholds (frozen)
_NUMERIC_FRAC = 0.90        # >= this fraction must normalize to a number to accept "numeric"
_DATE_FRAC = 0.90
_ID_DISTINCT = 0.95
_CAT_DISTINCT_MAX = 0.60    # categorical must not be near-unique
_TEXT_MIN_TOKENS = 3
_SEQ_LISTLIKE = 0.50


def validate_dtype(dtype, values, stats):
    """Return (ok, reason). Does the column's real values support `dtype`?"""
    n = max(len(values), 1)
    distinct = (stats or {}).get("distinct_frac", 0.0) or 0.0
    median_tok = (stats or {}).get("median_tokens", 0) or 0
    listlike = (stats or {}).get("listlike_frac", 0.0) or 0.0

    if dtype == "numeric":
        frac = sum(1 for v in values if normalize_numeric(v) is not None) / n
        return (frac >= _NUMERIC_FRAC, f"{frac:.2f} of values normalize to a number "
                                       f"(need {_NUMERIC_FRAC})")
    if dtype == "datetime":
        frac = sum(1 for v in values if _parses_date(v)) / n
        return (frac >= _DATE_FRAC, f"{frac:.2f} of values parse as a date (need {_DATE_FRAC})")
    if dtype == "id":
        return (distinct >= _ID_DISTINCT, f"distinct_frac {distinct:.2f} (need >= {_ID_DISTINCT} for id)")
    if dtype == "categorical":
        return (distinct <= _CAT_DISTINCT_MAX,
                f"distinct_frac {distinct:.2f} (need <= {_CAT_DISTINCT_MAX} for categorical)")
    if dtype == "text":
        return (median_tok >= _TEXT_MIN_TOKENS,
                f"median_tokens {median_tok} (need >= {_TEXT_MIN_TOKENS} for text)")
    if dtype == "sequence":
        return (listlike >= _SEQ_LISTLIKE, f"listlike_frac {listlike:.2f} (need >= {_SEQ_LISTLIKE})")
    return (False, f"unknown dtype {dtype!r}")


@dataclass
class ResolvedTyping:
    columns: dict                  # name -> resolved dtype
    decisions: list                # [{name, proposed, resolved, action, reason}]
    needs_human: bool
    ambiguous_columns: list        # names that went to the LLM
    source: str

    def as_dict(self):
        return {"columns": dict(self.columns), "decisions": list(self.decisions),
                "needs_human": self.needs_human, "ambiguous_columns": list(self.ambiguous_columns),
                "source": self.source}

    def column_specs(self):
        """ColumnSpec-shaped dicts the product can lift into ar.source.TypedSchema."""
        return [{"name": n, "dtype": d, "role": "feature"} for n, d in self.columns.items()]


def resolve_typing(records, view, proposal_columns, *, source="llm") -> ResolvedTyping:
    """Build the resolved per-column typing: structural for clear columns, validated-LLM for ambiguous.
    `proposal_columns` is a list of {name, dtype} (may be empty -> pure structural fallback)."""
    proposed = {c["name"]: c["dtype"] for c in (proposal_columns or []) if "name" in c and "dtype" in c}
    columns, decisions, ambiguous = {}, [], []
    needs_human = False

    for col_name, stats in view.columns.items():
        stats = stats if isinstance(stats, dict) else {}
        stats = dict(stats, col=col_name)
        values = column_values(records, col_name)
        conf = rule_confidence(values, stats)
        struct = conf.structural_dtype

        if not conf.ambiguous:
            columns[col_name] = struct
            decisions.append({"name": col_name, "proposed": None, "resolved": struct,
                              "action": "structural", "reason": "rule-confident"})
            continue

        ambiguous.append(col_name)
        prop = proposed.get(col_name)
        if prop is None:
            columns[col_name] = struct
            decisions.append({"name": col_name, "proposed": None, "resolved": struct,
                              "action": "no_proposal_structural", "reason": conf.reason})
            needs_human = True
            continue
        ok, why = validate_dtype(prop, values, stats)
        if ok:
            columns[col_name] = prop
            decisions.append({"name": col_name, "proposed": prop, "resolved": prop,
                              "action": ("accept" if prop != struct else "accept_same"),
                              "reason": why})
            if prop != struct:
                needs_human = True       # a relabel away from the rule is worth a human glance
        else:
            columns[col_name] = struct
            decisions.append({"name": col_name, "proposed": prop, "resolved": struct,
                              "action": "reject_to_structural", "reason": why})
            needs_human = True

    return ResolvedTyping(columns=columns, decisions=decisions, needs_human=needs_human,
                          ambiguous_columns=ambiguous, source=source)
