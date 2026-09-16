"""Frozen per-column rule confidence + the structural dtype + the ambiguous-residual gate.

The numeric normalizer and the confidence bands are documented HEURISTICS (labeled as such). They decide
only WHICH columns go to the LLM and what the deterministic fallback type is; they never decide a
certificate. The validator (validator.py) is the bound on whatever the LLM proposes.
"""
import math
import datetime as _dt
from dataclasses import dataclass

# the product's ColumnSpec dtype vocabulary (ar/source.py)
COLUMN_DTYPES = ("numeric", "categorical", "text", "datetime", "id", "sequence")

_CURRENCY = "$€£¥₹"
_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y")

# confidence bands (heuristic): float-parse fractions outside [LOW, HIGH] are "clear"; inside is muddy.
_PLAIN_NUMERIC_HIGH = 0.95
_FMT_NUMERIC_MIN = 0.60         # >= this fraction normalizes to a number -> a formatted-numeric candidate
_DATE_MIN = 0.60
_ID_DISTINCT_MIN = 0.98
_TEXT_MIN_TOKENS = 3
_CAT_DISTINCT_MAX = 0.50


def normalize_numeric(v):
    """Frozen numeric normalizer: parse a value to float after standard cleanup (currency symbols,
    thousands commas, a trailing %, a k/m/b magnitude suffix, accounting parentheses for negatives).
    Returns float or None. Documented heuristic; identical for all callers (no per-dataset tuning)."""
    s = str(v).strip()
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    while s and s[0] in _CURRENCY:
        s = s[1:].strip()
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1].strip()
    mult = 1.0
    if s and s[-1] in "kmbKMB":
        mult = _SUFFIX[s[-1].lower()]
        s = s[:-1].strip()
    s = s.replace(",", "").strip()
    try:
        f = float(s) * mult
        return -f if neg else f
    except ValueError:
        return None


def _parses_date(v):
    s = str(v).strip()
    for fmt in _DATE_FORMATS:
        try:
            _dt.datetime.strptime(s, fmt)
            return True
        except ValueError:
            continue
    return False


def _plain_float(v):
    try:
        float(str(v).strip())
        return True
    except ValueError:
        return False


def column_values(records, col):
    """Raw non-empty values for a column."""
    out = []
    for r in records:
        v = r.get(col) if isinstance(r, dict) else None
        if v is None:
            continue
        s = v if not isinstance(v, str) else v.strip()
        if s == "":
            continue
        out.append(v)
    return out


@dataclass
class ColumnConfidence:
    name: str
    structural_dtype: str
    ambiguous: bool
    confidence: float            # 1.0 = clearly typed by rules; lower = needs the LLM
    reason: str
    plain_numeric_frac: float
    fmt_numeric_frac: float
    date_frac: float
    distinct_frac: float
    median_tokens: int


def structural_dtype(stats, plain_numeric_frac, fmt_numeric_frac, date_frac):
    """The deterministic dtype the rules assign (and the A2 fallback). Mirrors compile()'s philosophy."""
    listlike = (stats or {}).get("listlike_frac", 0.0) or 0.0
    distinct = (stats or {}).get("distinct_frac", 0.0) or 0.0
    median_tok = (stats or {}).get("median_tokens", 0) or 0
    if listlike >= 0.5:
        return "sequence"
    if plain_numeric_frac >= _PLAIN_NUMERIC_HIGH:
        return "numeric"
    if median_tok >= _TEXT_MIN_TOKENS:
        return "text"
    return "categorical"


def rule_confidence(values, stats) -> ColumnConfidence:
    """Decide the structural dtype and whether the column is ambiguous (-> send to the LLM)."""
    name = (stats or {}).get("col", "") or ""
    n = max(len(values), 1)
    plain = sum(1 for v in values if _plain_float(v)) / n
    fmt = sum(1 for v in values if normalize_numeric(v) is not None) / n
    date = sum(1 for v in values if _parses_date(v)) / n
    distinct = (stats or {}).get("distinct_frac", 0.0) or 0.0
    median_tok = (stats or {}).get("median_tokens", 0) or 0
    struct = structural_dtype(stats, plain, fmt, date)

    ambiguous, reason, conf = False, "clear by rules", 1.0
    # formatted numeric hidden as a string: the rule sees non-numeric, but most values normalize to numbers
    if plain < _PLAIN_NUMERIC_HIGH and fmt >= _FMT_NUMERIC_MIN and fmt > plain + 0.1:
        ambiguous, reason, conf = True, f"formatted-numeric candidate (fmt {fmt:.2f} > plain {plain:.2f})", 0.4
    # date-like strings
    elif plain < _PLAIN_NUMERIC_HIGH and date >= _DATE_MIN:
        ambiguous, reason, conf = True, f"datetime candidate (date_frac {date:.2f})", 0.4
    # numeric but near-unique -> maybe an id; or only 2 values -> maybe a category code
    elif plain >= _PLAIN_NUMERIC_HIGH and distinct >= _ID_DISTINCT_MIN:
        ambiguous, reason, conf = True, f"numeric near-unique -> id candidate (distinct {distinct:.2f})", 0.5
    # muddy middle: a meaningful but not dominant fraction parses as numbers
    elif 0.30 <= fmt < _FMT_NUMERIC_MIN:
        ambiguous, reason, conf = True, f"mixed numeric/non-numeric (fmt {fmt:.2f})", 0.5

    return ColumnConfidence(name=name, structural_dtype=struct, ambiguous=ambiguous, confidence=conf,
                            reason=reason, plain_numeric_frac=round(plain, 3),
                            fmt_numeric_frac=round(fmt, 3), date_frac=round(date, 3),
                            distinct_frac=round(distinct, 3), median_tokens=median_tok)
