"""A2 -- LLM column typing on messy/ambiguous data, gated on rule-confidence, behind a frozen validator.

The deterministic profiler types every column first; only the LOW-CONFIDENCE residual (e.g. a currency
column '$1,200' the rule reads as a string, a numeric-looking id, a date column) is sent to the LLM. The
LLM proposes a dtype per ambiguous column (from the product's ColumnSpec vocabulary); a frozen validator
confirms each proposal against the column's ACTUAL values or drops it back to the structural type. Output
feeds the existing two-door `ar.source.build_from_typed_schema` -- A2 changes who AUTHORS the type guess,
never what the certifier sees. Sample values shown to the LLM are quarantined as untrusted data.
"""
from .profile_conf import (normalize_numeric, column_values, structural_dtype, rule_confidence,
                           ColumnConfidence, COLUMN_DTYPES)
from .schema import TYPE_PROPOSAL_JSON_SCHEMA, TypeProposal
from .validator import validate_dtype, resolve_typing, ResolvedTyping
from .typer import type_columns, TypingResult

__all__ = ["normalize_numeric", "column_values", "structural_dtype", "rule_confidence",
           "ColumnConfidence", "COLUMN_DTYPES", "TYPE_PROPOSAL_JSON_SCHEMA", "TypeProposal",
           "validate_dtype", "resolve_typing", "ResolvedTyping", "type_columns", "TypingResult"]
