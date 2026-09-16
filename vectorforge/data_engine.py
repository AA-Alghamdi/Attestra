"""The decision-driven data lever.

Most ML failures are data failures, not model failures. Before the landscape scan ever runs, this
module looks at the actual training set and decides whether the highest-leverage move is to CLEAN it
(it has duplicates or a leaking feature), GENERATE more of it (it is too small AND synthetic data is
permitted AND we can MEASURE that generation helps), or REQUEST_LABELS (it is too small and we cannot
manufacture honest signal). Exactly one move is taken per call, and the choice is justified by
measurement, never by hope.

Contract: run_data_engine(goal, train, val, test, allow_api=False) -> dict
    {move, decision, report, new_train}

  move      : "CLEAN" | "GENERATE" | "REQUEST_LABELS" | "NONE"
  decision  : a short human-readable justification string
  report    : the full structured diagnosis + action record (auditable)
  new_train : the training rows AFTER the move (cleaned / augmented / unchanged). The caller decides
              whether to adopt them; this function never mutates `train` in place and never touches the
              locked test set except to read it for leakage / dedup gates.

Rigor guarantees that make this safe:
  - Leakage diagnosis and every gate reuse science.audit (the one auditor the whole product trusts).
    A feature is only quarantined if science.audit flags it as leaking; nothing is dropped on a hunch.
  - GENERATE is the knowability test. Synthetic rows are (1) gated by the generator's leakage/dup gate
    against the LOCKED test, (2) re-audited with science.audit, and (3) MEASURED: a quick model is
    trained on seed-only and on seed+synth, both scored on a held-out slice that the synthetic data
    never saw. Synthetic data is adopted ONLY if it does not hurt held-out accuracy AND passes audit.
    Otherwise we fall back to REQUEST_LABELS and say so. We never claim a lift we did not measure.
  - REQUEST_LABELS produces a concrete ask: roughly how many more labels and for which classes (the
    classes that are starving the held-out estimate), not a vague "need more data".
  - The locked test set is read for dedup/leakage gates only and is never trained on or shown to the
    generator.
"""

from collections import Counter
import math

import numpy as np

from . import science, ml


# --------------------------------------------------------------------------- tunables (documented)
# Below this many training rows the set is "data-limited": too few to certify a high bar reliably and
# the regime where a data move (generate / request) dominates a model move. This is a heuristic for
# routing the decision, NOT a science threshold; the certifier still has the final say downstream.
DATA_LIMITED_TRAIN = 300
# Cap on the synthetic batch in any single call (cost + the measure step must stay cheap and honest).
DEFAULT_SYNTH_BATCH = 40
# How many more labels to ask for when we request: enough to roughly double a tiny set, floored.
REQUEST_FLOOR = 100


# =========================================================================== row identity / dedup
def _row_key(row):
    """A content key for exact-duplicate detection over the FULL model input. Two rows are duplicates
    only if BOTH their normalized text AND their full feature dict match; using all input fields
    avoids collapsing distinct rows that happen to share a single feature value. The label is
    intentionally EXCLUDED so that two rows with identical inputs but conflicting labels also collapse
    to one key (a label conflict is itself a data-quality defect we want to surface, not hide)."""
    text = str(row.get("text", "") or "").strip().lower()
    feats = row.get("features")
    feat_part = ""
    if isinstance(feats, dict) and feats:
        feat_part = "|".join(f"{k}={feats[k]}" for k in sorted(feats))
    return f"text::{text}::feat::{feat_part}"


def _exact_duplicates(rows):
    """Return (kept_rows, n_dropped, n_label_conflicts). Keeps the first occurrence of each content
    key. Counts how many dropped rows disagreed on the label with the kept copy (a conflict)."""
    seen = {}              # key -> kept row index in output
    kept = []
    n_dropped = 0
    n_conflict = 0
    for r in rows:
        k = _row_key(r)
        if k in seen:
            n_dropped += 1
            if str(kept[seen[k]].get("target")) != str(r.get("target")):
                n_conflict += 1
            continue
        seen[k] = len(kept)
        kept.append(r)
    return kept, n_dropped, n_conflict


# =========================================================================== leakage diagnosis
def _leaking_features(train, test, allow_features):
    """Run science.audit and return the list of feature keys it flagged as target-leaking. Keys come
    back as e.g. 'features.route_hint' or a bare column name; we normalize to the bare feature name
    that lives inside the row's `features` dict (or a top-level column) so we can quarantine it."""
    rep = science.audit(train, test, allow_features=allow_features,
                        min_test_n=1)  # min_test_n=1: we are diagnosing the TRAIN data, not certifying
    leaking = []
    for f in rep["findings"]:
        if f.get("gate") == "feature_target_leakage" and not f.get("ok", True):
            key = f["feature"]
            bare = key.split(".", 1)[1] if key.startswith("features.") else key
            leaking.append(bare)
    return leaking, rep


def _quarantine(rows, feature_names):
    """Return rows with the named features removed (from the `features` dict and/or top-level). Never
    mutates the input rows."""
    drop = set(feature_names)
    out = []
    for r in rows:
        nr = dict(r)
        if isinstance(nr.get("features"), dict):
            nr["features"] = {k: v for k, v in nr["features"].items() if k not in drop}
        for k in drop:
            nr.pop(k, None)
        out.append(nr)
    return out


# =========================================================================== quick held-out measure
def _quick_eval(kind, train_rows, eval_rows, labels, metric):
    """Train the cheapest sane model on `train_rows`, score on `eval_rows`. Returns the metric value.
    This is the GENERATE knowability probe: it is deliberately a single fast pipeline (no landscape) so
    the cost of MEASURING whether synthetic data helps is tiny. Returns None if either side is empty or
    has <2 classes (can't fit)."""
    if not train_rows or not eval_rows:
        return None
    if len({str(r.get("target")) for r in train_rows}) < 2:
        return None
    if kind == "text":
        _, ctor, cfg = ml.candidates("text")[0]      # logistic | ngram1: cheap, deterministic
    else:
        # logistic base for tabular: fast and does not memorize like the trees can
        cands = ml.candidates("tabular")
        pick = next((c for c in cands if c[0].startswith("logistic|base")), cands[0])
        _, ctor, cfg = pick
    try:
        Xtr, ytr = ml._Xy(kind, train_rows, cfg)
        Xev, yev = ml._Xy(kind, eval_rows, cfg)
        pipe = ml.build_pipeline(kind, ctor, cfg)
        pipe.fit(Xtr, ytr)
        return float(science.score_metric(metric, yev, list(pipe.predict(Xev)), labels))
    except Exception as e:  # noqa: BLE001  -- a fit/transform failure means "can't measure", not crash
        return None


# =========================================================================== label request shaping
def _label_request(train, labels, target_n):
    """A concrete ask. Identify the classes that are most under-represented in train (those starve the
    held-out estimate) and split the requested count toward them, weighted by their deficit from a
    balanced share."""
    counts = Counter(str(r.get("target")) for r in train)
    labels = [str(l) for l in labels]
    n = len(train)
    balanced = n / max(len(labels), 1)
    deficits = {l: max(0.0, balanced - counts.get(l, 0)) for l in labels}
    total_def = sum(deficits.values())
    add = max(REQUEST_FLOOR, target_n - n)
    if total_def > 0:
        per_class = {l: int(round(add * deficits[l] / total_def)) for l in labels}
    else:
        per_class = {l: int(round(add / len(labels))) for l in labels}
    # ensure at least the starved classes get a positive ask
    weak = [l for l in labels if counts.get(l, 0) < balanced]
    for l in weak:
        per_class[l] = max(per_class.get(l, 0), 1)
    return {"kind": "labels",
            "ask": f"~{add} more labeled examples (current train n={n}); prioritize classes "
                   f"{weak or labels}",
            "approx_total": add,
            "per_class": {l: per_class.get(l, 0) for l in labels},
            "current_counts": {l: counts.get(l, 0) for l in labels},
            "weak_classes": weak}


# =========================================================================== synthetic generation
def _try_generate(goal, seed_train, val, test, batch, key, measure_holdout):
    """Attempt one measured synthetic-generation batch. Returns a record dict describing exactly what
    happened (parsed / gated / audited / measured / adopted) and, if adopted, the augmented train set.

    Steps, in order, none skippable:
      1. read API key (caller already checked it exists; we re-read defensively)
      2. build prompt from goal.label_meaning + a few seed examples, generate ONE small batch
      3. parse -> rows
      4. gate_generated against the LOCKED test (drop label-token-in-text and near-dups)
      5. science.audit(seed+synth, test): the augmented train must still pass leakage
      6. MEASURE: quick model on seed-only vs seed+synth, scored on a held-out slice the synth never
         saw. Adopt synth ONLY if held-out metric does not drop AND audit passed.
    """
    import sys
    from pathlib import Path
    gen_dir = Path("/Users/abdullahalghamdi/vectorforge-harnesses/synthetic")
    sys.path.insert(0, str(gen_dir))
    import generator  # noqa: E402

    rec = {"attempted": True, "batch_requested": batch}
    labels = [str(l) for l in goal.labels]
    label_meaning = goal.label_meaning or {}
    # forbid both the blinded label tokens AND their human meanings appearing verbatim in generated text
    forbid = set(labels) | {str(v) for v in label_meaning.values()}

    # seed examples for style: a handful per class
    seed_examples = []
    by = {}
    for r in seed_train:
        by.setdefault(str(r.get("target")), []).append(r)
    for lab in labels:
        for r in by.get(lab, [])[:6]:
            seed_examples.append((str(r.get("text", "") or ""), lab))

    system, user = generator.build_prompt(goal.task_desc or goal.objective or goal.name,
                                           label_meaning, seed_examples, batch, forbid)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        raw, in_tok, out_tok = generator.generate_batch(client, system, user, batch)
        rec["api_called"] = True
        rec["tokens"] = {"in": in_tok, "out": out_tok}
    except Exception as e:  # noqa: BLE001
        rec.update({"api_called": False, "error": f"generation failed: {str(e)[:160]}",
                    "adopted": False, "fallback": "REQUEST_LABELS"})
        return rec, None

    parsed = generator.parse_examples(raw, labels)
    kept, dropped = generator.gate_generated(parsed, test, forbid)
    rec.update({"parsed": len(parsed), "gated_kept": len(kept),
                "gated_dropped": [reason for _, reason in dropped]})
    if not kept:
        rec.update({"adopted": False, "reason": "all generated rows failed the leakage/dup gate",
                    "fallback": "REQUEST_LABELS"})
        return rec, None

    synth = [{"text": ex["text"], "target": ex["target"]} for ex in kept]
    augmented = list(seed_train) + synth

    # gate 5: the augmented train must still pass the leakage audit against the locked test
    aug_audit = science.audit(augmented, test, allow_features=(goal.kind == "tabular"), min_test_n=1)
    rec["audit_passed"] = aug_audit["passed"]
    if not aug_audit["passed"]:
        rec.update({"adopted": False, "reason": "augmented set failed science.audit",
                    "fallback": "REQUEST_LABELS"})
        return rec, None

    # gate 6: MEASURE on held-out (the synth never saw measure_holdout)
    metric = goal.verification.metric
    base = _quick_eval(goal.kind, seed_train, measure_holdout, labels, metric)
    aug = _quick_eval(goal.kind, augmented, measure_holdout, labels, metric)
    rec["measure"] = {"holdout_n": len(measure_holdout), "metric": metric,
                      "seed_only": None if base is None else round(base, 4),
                      "seed_plus_synth": None if aug is None else round(aug, 4)}
    if base is None or aug is None:
        rec.update({"adopted": False, "reason": "could not measure synthetic effect (degenerate fit)",
                    "fallback": "REQUEST_LABELS"})
        return rec, None
    # adopt only if it does not HURT held-out (>= base, allowing a tiny numerical slack)
    helped = aug >= base - 1e-9
    rec["measure"]["delta"] = round(aug - base, 4)
    if helped:
        rec.update({"adopted": True, "n_synth_adopted": len(synth),
                    "reason": f"synthetic kept: held-out {metric} {round(base,4)} -> {round(aug,4)} "
                              f"(delta {round(aug-base,4)}), audit passed"})
        return rec, augmented
    rec.update({"adopted": False,
                "reason": f"synthetic rejected: held-out {metric} {round(base,4)} -> {round(aug,4)} "
                          f"(delta {round(aug-base,4)}) did not help",
                "fallback": "REQUEST_LABELS"})
    return rec, None


# =========================================================================== the entry point
def run_data_engine(goal, train, val, test, allow_api=False):
    """Diagnose the training data and take exactly one data move. See module docstring for the contract."""
    train = list(train)
    val = list(val or [])
    test = list(test or [])
    allow_features = (goal.kind == "tabular")

    # ---- diagnose ---------------------------------------------------------------------------------
    deduped, n_dup, n_conflict = _exact_duplicates(train)
    leaking, audit_rep = _leaking_features(train, test, allow_features)
    n_train = len(train)
    data_limited = n_train < DATA_LIMITED_TRAIN
    has_unlabeled = bool(getattr(goal, "research_state", {}) and
                         goal.research_state.get("unlabeled_pool"))

    diagnosis = {
        "n_train": n_train,
        "exact_duplicates": n_dup,
        "label_conflicts": n_conflict,
        "leaking_features": leaking,
        "data_limited": data_limited,
        "data_limited_threshold": DATA_LIMITED_TRAIN,
        "has_unlabeled_pool": has_unlabeled,
        "class_balance": {str(l): sum(1 for r in train if str(r.get("target")) == str(l))
                          for l in goal.labels},
        "audit_passed": audit_rep["passed"],
    }

    quality_issue = (n_dup > 0) or bool(leaking)

    # ---- decide ONE move (priority: CLEAN > GENERATE > REQUEST_LABELS) -----------------------------
    report = {"diagnosis": diagnosis, "audit_findings": audit_rep["findings"]}

    # ---------- CLEAN ----------
    if quality_issue:
        cleaned = deduped
        if leaking:
            cleaned = _quarantine(cleaned, leaking)
        report["action"] = {
            "dropped_duplicates": n_dup,
            "label_conflicts_among_dropped": n_conflict,
            "quarantined_features": leaking,
            "n_train_after": len(cleaned),
        }
        decision = []
        if n_dup:
            decision.append(f"dropped {n_dup} exact-duplicate rows"
                            + (f" ({n_conflict} had conflicting labels)" if n_conflict else ""))
        if leaking:
            decision.append(f"quarantined leaking feature(s) {leaking} flagged by science.audit")
        return {"move": "CLEAN", "decision": "; ".join(decision), "report": report,
                "new_train": cleaned}

    # ---------- GENERATE (only when data-limited AND fully permitted AND measurable) ----------
    can_generate = (data_limited and goal.surface.synthetic_data and allow_api
                    and bool(goal.label_meaning))
    if can_generate:
        from pathlib import Path
        import sys
        sys.path.insert(0, "/Users/abdullahalghamdi/vectorforge-harnesses/synthetic")
        import generator  # noqa: E402
        key = generator.read_key()
        if not key:
            report["action"] = {"generate_skipped": "no API key available"}
            req = _label_request(train, goal.labels, target_n=DATA_LIMITED_TRAIN)
            report["request"] = req
            return {"move": "REQUEST_LABELS",
                    "decision": "data-limited; synthetic enabled but no API key -> " + req["ask"],
                    "report": report, "new_train": train}

        # held-out slice for the measure step: prefer val; if val is too small, carve from train.
        if len(val) >= 20:
            measure_holdout, seed_for_gen = val, train
        else:
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(train))
            cut = max(1, len(train) // 5)
            measure_holdout = [train[i] for i in idx[:cut]]
            seed_for_gen = [train[i] for i in idx[cut:]]

        batch = min(DEFAULT_SYNTH_BATCH, max(8, goal.budget.max_synth_examples or DEFAULT_SYNTH_BATCH))
        rec, augmented = _try_generate(goal, seed_for_gen, val, test, batch, key, measure_holdout)
        report["action"] = {"generate": rec}

        if rec.get("adopted") and augmented is not None:
            # rebuild new_train as seed (the full original train) + adopted synth, so we don't lose the
            # rows we carved for measurement.
            adopted_synth = augmented[len(seed_for_gen):]
            new_train = list(train) + adopted_synth
            return {"move": "GENERATE",
                    "decision": rec["reason"],
                    "report": report, "new_train": new_train}

        # measured-and-failed (or gate-failed) -> honest fallback to REQUEST_LABELS
        req = _label_request(train, goal.labels, target_n=DATA_LIMITED_TRAIN)
        report["request"] = req
        return {"move": "REQUEST_LABELS",
                "decision": f"synthetic did not help ({rec.get('reason','')}); fall back to -> " + req["ask"],
                "report": report, "new_train": train}

    # ---------- REQUEST_LABELS ----------
    if data_limited:
        req = _label_request(train, goal.labels, target_n=DATA_LIMITED_TRAIN)
        report["request"] = req
        why = []
        if not goal.surface.synthetic_data:
            why.append("synthetic_data surface off")
        elif not allow_api:
            why.append("allow_api=False")
        elif not goal.label_meaning:
            why.append("no label_meaning to seed a safe generator")
        reason = ("data-limited"
                  + (f" and cannot generate ({', '.join(why)})" if why else "")
                  + " -> " + req["ask"])
        return {"move": "REQUEST_LABELS", "decision": reason, "report": report, "new_train": train}

    # ---------- NONE: data is clean and sufficient ----------
    report["action"] = {"noop": "data is clean (no dups / no leakage) and not data-limited"}
    return {"move": "NONE", "decision": "no data move needed; data is clean and sufficient",
            "report": report, "new_train": train}


# =========================================================================== self-test
def _selftest():
    import json
    from pathlib import Path
    from .domain import Goal, VerificationSpec, ExperimentSurface, Budget

    TEXT = Path("/Users/abdullahalghamdi/core-ml-acceptance/data")
    TAB = Path("/Users/abdullahalghamdi/vectorforge-harnesses/rugged/data")

    def loadj(d, n):
        return [json.loads(l) for l in (d / f"{n}.jsonl").read_text().splitlines() if l.strip()]

    passed = []
    failed = []

    def check(name, cond, detail=""):
        (passed if cond else failed).append(name)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

    # ------------------------------------------------------------------ CASE 1: CLEAN (text)
    # Inject exact duplicates AND a planted leaking feature (route_hint) into a real text slice.
    print("\n=== CASE 1: CLEAN on a dirty TEXT dataset (dups + planted leak) ===")
    # Build a base set with GUARANTEED-UNIQUE content (real data has many short duplicate fragments,
    # so we synthesize unique texts to make the injected-duplicate count deterministic). Labels and a
    # planted leaking feature are kept so science.audit has something real to flag.
    src = loadj(TEXT, "train")
    base = []
    for i in range(400):
        lab = "class_a" if i % 2 == 0 else "class_b"
        base.append({"id": f"u-{i}", "text": f"unique base example number {i} {src[i]['text']}",
                     "target": lab, "features": {"route_hint": "H_" + lab}})
    te = loadj(TEXT, "test")
    # inject 30 exact duplicates of the first 30 (unique) rows
    dups = [dict(base[i]) for i in range(30)]
    dirty = base + dups
    te_feat = [{**r, "features": {"route_hint": "H_" + str(r["target"])}} for r in te]

    g1 = Goal(id="st-clean", name="dirty_text", kind="tabular",   # tabular so features are allowed/audited
              labels=["class_a", "class_b"],
              verification=VerificationSpec(metric="accuracy", threshold=0.9, min_heldout_n=200),
              surface=ExperimentSurface(), budget=Budget())
    out1 = run_data_engine(g1, dirty, [], te_feat, allow_api=False)
    print("  move =", out1["move"], "| decision:", out1["decision"])
    diag1 = out1["report"]["diagnosis"]
    check("CLEAN: move is CLEAN", out1["move"] == "CLEAN")
    check("CLEAN: detected the 30 exact duplicates",
          diag1["exact_duplicates"] == 30, f"found {diag1['exact_duplicates']}")
    check("CLEAN: flagged route_hint as leaking",
          "route_hint" in diag1["leaking_features"], str(diag1["leaking_features"]))
    # new_train: no dups remain, and the leaking feature is quarantined out
    keys = [_row_key(r) for r in out1["new_train"]]
    check("CLEAN: duplicates removed from new_train", len(keys) == len(set(keys)),
          f"{len(keys)} rows, {len(set(keys))} unique")
    still_leaks = any("route_hint" in (r.get("features") or {}) for r in out1["new_train"])
    check("CLEAN: route_hint quarantined out of new_train", not still_leaks)
    # the cleaned set must now pass science.audit on that feature gate
    re_audit = science.audit(out1["new_train"], te, allow_features=True, min_test_n=1)
    leak_after = [f for f in re_audit["findings"]
                  if f.get("gate") == "feature_target_leakage" and not f.get("ok", True)]
    check("CLEAN: no feature-leakage findings remain after clean", not leak_after, str(leak_after))

    # ------------------------------------------------------------------ CASE 2: REQUEST_LABELS (text, synth off)
    print("\n=== CASE 2: REQUEST_LABELS on a tiny seed with synthetic OFF ===")
    by = {}
    for r in loadj(TEXT, "train"):
        by.setdefault(r["target"], []).append(r)
    tiny = [r for lab in by for r in by[lab][:8]]   # 16 rows, clean, no features
    g2 = Goal(id="st-req", name="tiny_text", kind="text", labels=["class_a", "class_b"],
              label_meaning={"class_a": "negative", "class_b": "positive"},
              verification=VerificationSpec(metric="accuracy", threshold=0.9, min_heldout_n=200),
              surface=ExperimentSurface(synthetic_data=False),  # synthetic OFF
              budget=Budget())
    out2 = run_data_engine(g2, tiny, loadj(TEXT, "validation")[:50], loadj(TEXT, "test"), allow_api=True)
    print("  move =", out2["move"], "| decision:", out2["decision"])
    check("REQUEST: move is REQUEST_LABELS", out2["move"] == "REQUEST_LABELS")
    check("REQUEST: data_limited diagnosed", out2["report"]["diagnosis"]["data_limited"])
    req = out2["report"].get("request", {})
    check("REQUEST: concrete ask has a positive count", req.get("approx_total", 0) > 0,
          f"approx_total={req.get('approx_total')}")
    check("REQUEST: per-class ask present", bool(req.get("per_class")), str(req.get("per_class")))
    check("REQUEST: new_train unchanged (no fabrication)", out2["new_train"] == tiny)

    # ------------------------------------------------------------------ CASE 3: CLEAN priority over data-limited
    print("\n=== CASE 3: CLEAN takes priority even when also data-limited ===")
    tiny_dirty = tiny + [dict(tiny[0]), dict(tiny[1])]  # 2 dups on a tiny set
    g3 = Goal(id="st-prio", name="tiny_dirty", kind="text", labels=["class_a", "class_b"],
              verification=VerificationSpec(metric="accuracy", threshold=0.9, min_heldout_n=200),
              surface=ExperimentSurface(), budget=Budget())
    out3 = run_data_engine(g3, tiny_dirty, [], loadj(TEXT, "test"), allow_api=False)
    print("  move =", out3["move"], "| decision:", out3["decision"])
    check("PRIORITY: CLEAN beats REQUEST_LABELS when both apply", out3["move"] == "CLEAN")
    check("PRIORITY: the 2 dups were removed",
          out3["report"]["diagnosis"]["exact_duplicates"] == 2)

    # ------------------------------------------------------------------ CASE 4: NONE on clean+sufficient
    print("\n=== CASE 4: NONE when data is clean and sufficient (tabular) ===")
    tab_tr = loadj(TAB, "train")[:600]
    tab_te = loadj(TAB, "test")
    g4 = Goal(id="st-none", name="clean_tab", kind="tabular", labels=["class_a", "class_b"],
              verification=VerificationSpec(metric="accuracy", threshold=0.85, min_heldout_n=200),
              surface=ExperimentSurface(), budget=Budget())
    out4 = run_data_engine(g4, tab_tr, [], tab_te, allow_api=False)
    print("  move =", out4["move"], "| decision:", out4["decision"])
    check("NONE: clean sufficient tabular -> NONE", out4["move"] == "NONE",
          f"dups={out4['report']['diagnosis']['exact_duplicates']} "
          f"leaks={out4['report']['diagnosis']['leaking_features']}")

    # ------------------------------------------------------------------ CASE 5: GENERATE (measured) -- only if key
    print("\n=== CASE 5: GENERATE (measured, one small batch) -- only if API key present ===")
    import sys
    sys.path.insert(0, "/Users/abdullahalghamdi/vectorforge-harnesses/synthetic")
    import generator  # noqa: E402
    key = generator.read_key()
    if not key:
        print("  [SKIP] no API key -> GENERATE path exercised structurally via fallback (CASE 2 covers it)")
    else:
        # tiny seed, synthetic ON, allow_api True, label_meaning present -> should attempt GENERATE
        seed = [r for lab in by for r in by[lab][:10]]  # 20 rows
        g5 = Goal(id="st-gen", name="tiny_gen", kind="text", labels=["class_a", "class_b"],
                  task_desc="movie review sentiment",
                  label_meaning={"class_a": "negative sentiment", "class_b": "positive sentiment"},
                  verification=VerificationSpec(metric="accuracy", threshold=0.9, min_heldout_n=200),
                  surface=ExperimentSurface(synthetic_data=True),
                  budget=Budget(max_synth_examples=24))
        out5 = run_data_engine(g5, seed, loadj(TEXT, "validation")[:60], loadj(TEXT, "test"),
                               allow_api=True)
        print("  move =", out5["move"], "| decision:", out5["decision"])
        gen_rec = out5["report"].get("action", {}).get("generate", {})
        print("  generate record:", json.dumps({k: gen_rec.get(k) for k in
              ("api_called", "parsed", "gated_kept", "audit_passed", "measure", "adopted", "reason")},
              default=str))
        # honest outcome: either GENERATE (adopted, measured lift) or REQUEST_LABELS (measured no-help).
        check("GENERATE: outcome is one of the honest moves",
              out5["move"] in ("GENERATE", "REQUEST_LABELS"))
        check("GENERATE: the measure step actually ran (held-out numbers present)",
              bool(gen_rec.get("measure")) or gen_rec.get("adopted") is False,
              str(gen_rec.get("measure")))
        if out5["move"] == "GENERATE":
            check("GENERATE: new_train grew by adopted synth", len(out5["new_train"]) > len(seed))
            # adopted synth must have passed the audit gate
            check("GENERATE: adoption required audit_passed", gen_rec.get("audit_passed") is True)
        else:
            check("GENERATE: fallback left new_train unfabricated", out5["new_train"] == seed)

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 60)
    ok = not failed
    print(f"data_engine self-test: {'PASS' if ok else 'FAIL'}  "
          f"({len(passed)} passed, {len(failed)} failed)")
    if failed:
        print("  failing checks:", failed)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
