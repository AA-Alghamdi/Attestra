"""Deterministic dataset admissibility / inspector gate for ARBITRARY records (NEW module).

This is the front-of-front-door: BEFORE a goal is built and BEFORE any model is fit, profile the raw
records and decide whether a run is even admissible. It is a DETERMINISTIC structural inspector -- no LLM,
no estimator, no sealed peek, no certificate. An `admissible=False` verdict is an HONEST DECLINE (the
platform refuses to run a problem it cannot certify meaningfully); it is NOT, and must never be confused
with, the frozen certificate (which is produced only by vectorforge/science.py via vfplatform/sealed.py).

The platform's record shape (matching connectors.py and the featurizers in models.py) is a list of dicts
    {"features": {col: value, ...}, "target": <label-or-number>}   (target_key defaults to "target")
or a flat dict {col: value, ..., "target": ...}. We support both: a row with a nested "features" dict has
its feature columns read from there; otherwise every non-target key is a feature column.

inspect() answers four operational questions, all from the data alone:
  * what is the target column?            (target_key, else a column literally named 'target', else last col)
  * what task is this?                    (regression vs binary vs multiclass, from the target's dtype+cardinality)
  * what metric should rank it?           (accuracy / f1_macro / r2 -- a HUMAN-FACING suggestion)
  * is it safe to run at all?             (issues that make it inadmissible or merely risky)

Nothing here mutates inputs; the same records in always give the same verdict out (determinism is the
contract -- the inspector is part of the audit trail, not a heuristic that drifts).
"""
from collections import Counter

# DOCUMENTED FLOORS (defensible, not tuned to any answer sheet):
#   * MIN_ROWS: with a frozen 30% sealed test + 20% validation split (science.make_splits), fewer than this
#     leaves a sealed test too small for any non-trivial lower bound to clear a meaningful threshold. 20 rows
#     -> ~6 sealed rows; a Clopper-Pearson lower bound on n<=6 is essentially the whole [0,1] interval, so a
#     run below this floor cannot produce an informative certificate. We decline rather than certify noise.
MIN_ROWS = 20
#   * REGRESSION_MIN_UNIQUE: a numeric target with this many or more distinct values is treated as continuous
#     (regression); at or below it the integers are read as class ids (e.g. digits 0..9 -> multiclass). This
#     is the same boundary connectors.py uses implicitly (label-counting), made explicit and one-place here.
REGRESSION_MIN_UNIQUE = 15
#   * DUP_FRACTION_WARN: if this fraction or more of the rows are EXACT (feature+target) duplicates, the
#     effective sample size is far below n_rows; flagged as risky (not hard-fatal -- some domains repeat).
DUP_FRACTION_WARN = 0.5


def _numeric(v):
    """True iff v is a real number (bools are NOT numeric features -- they are categorical 2-level)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _missing(v):
    if v is None:
        return True
    if isinstance(v, float):
        # NaN != NaN; catch it without importing numpy (stdlib-only contract for the inspector).
        return v != v
    if isinstance(v, str):
        return v.strip() == ""
    return False


def _feature_cols(rows, target_key):
    """The feature columns of a record, supporting BOTH the nested {"features": {...}} shape and the flat
    {col: val, target: ...} shape. Returns the ordered union of columns seen across rows (deterministic)."""
    cols = []
    seen = set()
    for r in rows:
        feats = r.get("features") if isinstance(r.get("features"), dict) else {
            k: v for k, v in r.items() if k != target_key}
        for c in feats.keys():
            if c not in seen:
                seen.add(c)
                cols.append(c)
    return cols


def _feat_value(r, c, target_key):
    feats = r.get("features") if isinstance(r.get("features"), dict) else {
        k: v for k, v in r.items() if k != target_key}
    return feats.get(c)


def _detect_target(rows, target_key):
    """Resolve the target column deterministically: an explicit target_key wins; else a column literally
    named 'target'; else the LAST column of the first row's flat keys (a common CSV convention). Returns the
    chosen key or None if the records carry no usable column."""
    if target_key is not None:
        return target_key
    first = rows[0]
    if isinstance(first.get("features"), dict):
        # nested shape: the target is a sibling of "features"; prefer 'target', else any non-"features" key
        if "target" in first:
            return "target"
        siblings = [k for k in first.keys() if k != "features"]
        return siblings[-1] if siblings else "target"
    keys = list(first.keys())
    if "target" in keys:
        return "target"
    return keys[-1] if keys else None


def _rectangular(rows, target_key):
    """True iff every row exposes the SAME set of feature columns (a rectangular table). A ragged set of
    columns (records that are really heterogeneous JSON) cannot be featurized into a fixed matrix."""
    sigs = set()
    for r in rows:
        feats = r.get("features") if isinstance(r.get("features"), dict) else {
            k: v for k, v in r.items() if k != target_key}
        sigs.add(frozenset(feats.keys()))
    return len(sigs) <= 1


def _row_signature(r, cols, target_key):
    """A hashable (features, target) signature for exact-duplicate counting (stringified, order-stable)."""
    return (tuple(str(_feat_value(r, c, target_key)) for c in cols), str(r.get(target_key)))


def inspect(records, *, target_key=None):
    """Profile `records` and return an admissibility verdict (see module docstring). Deterministic; no LLM,
    no model, no sealed peek. Returns a dict with the documented keys; admissible=False + a clear verdict
    when a HARD issue is present (an honest decline, not a certificate)."""
    issues = []
    # ---- shape guards (before anything else can NPE) ----------------------------------------------
    if not isinstance(records, (list, tuple)) or len(records) == 0:
        return {"admissible": False, "verdict": "empty: no records to inspect",
                "n_rows": 0, "n_features": 0, "schema": {}, "suggested_target": None,
                "suggested_kind": "tabular", "suggested_task_type": "binary",
                "suggested_metric": "accuracy", "issues": ["no records"]}
    if not all(isinstance(r, dict) for r in records):
        return {"admissible": False, "verdict": "non-rectangular: records are not all dicts",
                "n_rows": len(records), "n_features": 0, "schema": {}, "suggested_target": None,
                "suggested_kind": "tabular", "suggested_task_type": "binary",
                "suggested_metric": "accuracy", "issues": ["records are not all dicts"]}

    rows = list(records)
    n_rows = len(rows)
    tkey = _detect_target(rows, target_key)

    rectangular = _rectangular(rows, tkey)
    if not rectangular:
        issues.append("non-rectangular records: feature columns differ across rows (cannot form a matrix)")

    cols = _feature_cols(rows, tkey)
    n_features = len(cols)

    # ---- per-column schema profile (numeric? cardinality? missingness?) ---------------------------
    schema = {}
    for c in cols:
        vals = [_feat_value(r, c, tkey) for r in rows]
        present = [v for v in vals if not _missing(v)]
        is_num = len(present) > 0 and all(_numeric(v) for v in present)
        schema[c] = {"numeric": bool(is_num),
                     "n_unique": len({str(v) for v in present}),
                     "n_missing": int(sum(1 for v in vals if _missing(v)))}
        if schema[c]["n_missing"] == n_rows:
            issues.append(f"column {c!r} is entirely missing")

    # ---- target profile ---------------------------------------------------------------------------
    target_present = tkey is not None and any((tkey in r) for r in rows)
    tvals = [r.get(tkey) for r in rows] if tkey is not None else []
    t_present_vals = [v for v in tvals if not _missing(v)]
    t_missing = int(sum(1 for v in tvals if _missing(v)))
    t_numeric = len(t_present_vals) > 0 and all(_numeric(v) for v in t_present_vals)
    t_unique = len({str(v) for v in t_present_vals})

    if not target_present or len(t_present_vals) == 0:
        issues.append(f"no usable target column (looked for {tkey!r}); cannot define a supervised task")
        task_type = "binary"
        metric = "accuracy"
    elif t_numeric and t_unique >= REGRESSION_MIN_UNIQUE:
        task_type = "regression"
        metric = "r2"
    elif t_unique == 2:
        task_type = "binary"
        metric = "accuracy"
    elif t_unique > 2:
        task_type = "multiclass"
        metric = "f1_macro"          # imbalance-robust default for >2 classes (human-facing suggestion)
    else:
        # exactly one distinct target value
        task_type = "binary"
        metric = "accuracy"

    # ---- HARD issues (these make the run inadmissible) --------------------------------------------
    if target_present and t_unique == 1 and len(t_present_vals) > 0:
        issues.append("constant target: every row has the same label/value (nothing to learn or certify)")
    if t_missing == n_rows and tkey is not None:
        issues.append("target column is entirely missing")

    # trivial leakage: a feature column IDENTICAL (value-for-value) to the target -> a model would 'win'
    # by copying it; the certificate would be meaningless. Detected by exact per-row equality.
    leak_cols = []
    if target_present:
        t_str = [str(v) for v in tvals]
        for c in cols:
            col_str = [str(_feat_value(r, c, tkey)) for r in rows]
            if col_str == t_str:
                leak_cols.append(c)
    for c in leak_cols:
        issues.append(f"trivial leakage: feature {c!r} is identical to the target (drop it before running)")

    if n_rows < MIN_ROWS:
        issues.append(f"too few rows: {n_rows} < documented floor {MIN_ROWS} (sealed test too small for an "
                      f"informative lower bound)")

    if n_features == 0:
        issues.append("no feature columns: cannot featurize")

    # ---- RISKY (soft) issues: duplicate-heavy data ------------------------------------------------
    soft_issues = []
    if n_rows > 0 and n_features > 0 and rectangular:
        sigs = Counter(_row_signature(r, cols, tkey) for r in rows)
        n_dups = n_rows - len(sigs)
        if n_dups / n_rows >= DUP_FRACTION_WARN:
            soft_issues.append(
                f"duplicate-heavy: {n_dups}/{n_rows} rows are exact duplicates (effective n is far smaller; "
                f"the certificate's n will overstate independent evidence)")

    HARD = {"constant target", "trivial leakage", "too few rows", "no usable target", "no feature columns",
            "non-rectangular", "entirely missing", "target column is entirely missing"}

    def _is_hard(msg):
        return any(h in msg for h in HARD)

    hard_issues = [m for m in issues if _is_hard(m)]
    all_issues = issues + soft_issues
    admissible = len(hard_issues) == 0

    if not admissible:
        verdict = "inadmissible: " + "; ".join(hard_issues)
    elif soft_issues:
        verdict = "admissible (with warnings): " + "; ".join(soft_issues)
    else:
        verdict = (f"admissible: {n_rows} rows x {n_features} features, target {tkey!r}, "
                   f"task={task_type}, metric={metric}")

    return {"admissible": bool(admissible), "verdict": verdict,
            "n_rows": int(n_rows), "n_features": int(n_features), "schema": schema,
            "suggested_target": tkey, "suggested_kind": "tabular",
            "suggested_task_type": task_type, "suggested_metric": metric,
            "issues": all_issues}


# --------------------------------------------------------------------------- self-test
def _selftest():
    p = f = 0

    def check(name, cond):
        nonlocal p, f
        if cond:
            print(f"  PASS  {name}"); p += 1
        else:
            print(f"  FAIL  {name}"); f += 1

    # 1) clean multiclass tabular frame -> admissible, multiclass, f1_macro
    rows = [{"features": {"a": float(i), "b": float(i % 4)}, "target": str(i % 3)} for i in range(60)]
    r = inspect(rows)
    check("clean frame admissible", r["admissible"] is True)
    check("detects multiclass", r["suggested_task_type"] == "multiclass")
    check("suggests f1_macro for multiclass", r["suggested_metric"] == "f1_macro")
    check("n_rows/n_features correct", r["n_rows"] == 60 and r["n_features"] == 2)

    # 2) binary
    rb = inspect([{"features": {"a": float(i)}, "target": str(i % 2)} for i in range(40)])
    check("detects binary", rb["suggested_task_type"] == "binary" and rb["suggested_metric"] == "accuracy")

    # 3) regression (many distinct numeric targets)
    rr = inspect([{"features": {"a": float(i)}, "target": float(i) * 1.5} for i in range(40)])
    check("detects regression", rr["suggested_task_type"] == "regression" and rr["suggested_metric"] == "r2")

    # 4) constant target -> inadmissible
    rc = inspect([{"features": {"a": float(i)}, "target": "yes"} for i in range(40)])
    check("constant target inadmissible", rc["admissible"] is False and "constant target" in rc["verdict"])

    # 5) feature == target leak -> inadmissible
    rl = inspect([{"features": {"a": float(i), "leak": str(i % 2)}, "target": str(i % 2)} for i in range(40)])
    check("feature==target leak inadmissible", rl["admissible"] is False and "leakage" in rl["verdict"])

    # 6) too few rows -> inadmissible
    rf = inspect([{"features": {"a": float(i)}, "target": str(i % 2)} for i in range(5)])
    check("too-few-rows inadmissible", rf["admissible"] is False and "too few rows" in rf["verdict"])

    # 7) flat record shape (no nested 'features')
    rflat = inspect([{"a": float(i), "b": float(i % 3), "target": str(i % 2)} for i in range(40)])
    check("flat-shape supported", rflat["admissible"] is True and rflat["n_features"] == 2)

    # 8) empty
    check("empty declines", inspect([])["admissible"] is False)

    # 9) determinism
    check("deterministic", inspect(rows) == inspect(rows))

    print(f"  ---- {p} passed, {f} failed ----")
    return f == 0


if __name__ == "__main__":
    import sys
    print("== admissibility self-test ==")
    sys.exit(0 if _selftest() else 1)
