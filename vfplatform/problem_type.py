"""Classify an arbitrary (records, goal_text) into a SUPPORTED (kind, task_type, metric) -- NEW module.

This is the routing decision that sits between the inspector (admissibility.inspect, the structural
profile) and the loop (run_goal_loop, which needs a concrete kind+task_type+metric). It is
DETERMINISTIC-FIRST: the structural suggestion comes entirely from admissibility.inspect (no model, no
LLM). The optional LLM pass (use_llm=True) may ONLY refine the task/metric within the supported set and is
NON-BINDING with a deterministic fallback -- it can never promote an unsupported problem into a supported
one, never invent a modality, never set a certificate or threshold. Every certify decision stays with the
frozen core; this module only decides which built harness (if any) a problem routes to.

Supported combos (the only ones with a built featurizer+harness+certifier path, mirroring
vfplatform/harness.is_supported and SUPPORTED_METRICS in vfplatform/sealed.py):
    tabular : binary | multiclass | regression
    text    : binary | multiclass
    vision  : binary | multiclass            (NEW, vfplatform/vision.py -- reuses the frozen clf certifier)
Out of THIS module's scope (declined honestly, with a reason): timeseries/forecast and ranking (their own
modules), audio, multilabel, and anything whose structure the inspector cannot turn into a fixed matrix.

The (kind, metric) returned is always certifiable: classification routes to "accuracy"/"f1_macro"
(f1_macro -> the frozen certifier's "macro_f1"), regression to "r2". We never return a metric the sealed
guard would refuse.
"""
from . import admissibility

# the supported (kind -> task_types) routing table; the single source of truth for THIS module's scope.
SUPPORTED = {
    "tabular": ("binary", "multiclass", "regression"),
    "text": ("binary", "multiclass"),
    "vision": ("binary", "multiclass"),
}

# human-facing metric suggestion -> the certifiable metric name the frozen sealed guard accepts.
# (admissibility suggests "f1_macro"; vfplatform/sealed.SUPPORTED_METRICS spells it "macro_f1".)
_METRIC_CANON = {"f1_macro": "macro_f1", "macro_f1": "macro_f1",
                 "accuracy": "accuracy", "r2": "r2"}

# light lexical cues that a goal/records describe IMAGES (used only to PROMOTE a tabular numeric frame to the
# vision harness when the data is image-shaped). Deterministic and conservative: we only route to vision when
# BOTH the goal hints at images AND the features are an all-numeric, image-pluasible matrix.
_VISION_CUES = ("image", "images", "pixel", "pixels", "photo", "picture", "digit image", "mnist",
                "cifar", "vision", "imagery")
# cues that the problem is OUT OF SCOPE for this module (so we decline with a precise reason rather than
# silently coercing it into tabular accuracy -- which would certify the wrong problem).
_TIMESERIES_CUES = ("forecast", "time series", "timeseries", "time-series", "next day", "next-step",
                    "future value", "trend over time")
_RANKING_CUES = ("rank", "ranking", "rank the", "order by relevance", "search results", "ndcg")
_AUDIO_CUES = ("audio", "speech", "waveform", "spectrogram", "sound clip")
_MULTILABEL_CUES = ("multi-label", "multilabel", "multiple labels per", "tags per")


def _looks_like_images(goal_text, schema, n_features):
    """True iff the goal hints at images AND the feature matrix is plausibly an image (all-numeric, and a
    perfect-square feature count, the digits/MNIST flattened-image convention). Conservative on purpose:
    we never reroute a generic numeric frame to vision without an explicit image hint."""
    g = (goal_text or "").lower()
    if not any(cue in g for cue in _VISION_CUES):
        return False
    if n_features <= 0 or not schema:
        return False
    all_numeric = all(s.get("numeric") for s in schema.values())
    side = int(round(n_features ** 0.5))
    is_square = side * side == n_features and side >= 2
    return all_numeric and is_square


def _scope_decline_reason(goal_text):
    """If the goal clearly describes an OUT-OF-SCOPE problem, return a precise decline reason; else None."""
    g = (goal_text or "").lower()
    if any(c in g for c in _TIMESERIES_CUES):
        return ("looks like time-series forecasting (temporal/grouped data). This module routes only "
                "tabular/text/vision i.i.d. classification+regression; forecasting has its own grouped "
                "block-bootstrap path (vfplatform/timeseries.py). Provide it through the timeseries route.")
    if any(c in g for c in _RANKING_CUES):
        return ("looks like a ranking task (per-query ordering). Ranking has its own per-query-bootstrap "
                "path (vfplatform/ranking.py); it is out of this module's i.i.d. scope.")
    if any(c in g for c in _AUDIO_CUES):
        return ("looks like an audio task. No audio featurizer/harness is built; to support it you would "
                "need an audio featurizer that emits a fixed numeric matrix (e.g. MFCC/spectrogram frames).")
    if any(c in g for c in _MULTILABEL_CUES):
        return ("looks like a multi-label task (multiple labels per row). Only single-label "
                "binary/multiclass classification is built; multi-label needs a per-label certifier.")
    return None


def _refine_with_llm(profile, goal_text, base_kind, base_task, base_metric, *, api_key, cache_path):
    """NON-BINDING LLM refinement: the model may only pick a task_type within the base kind's supported set
    and a metric within the certifiable set. Any failure/rejection falls back to the deterministic choice.
    Routed through the existing vectorforge.llm_shell.ops.llm_propose (read-only import; we add no new LLM
    seam). Returns (task_type, metric, used_llm)."""
    try:
        from vectorforge.llm_shell import ops
    except Exception:  # noqa: BLE001  llm shell unavailable -> deterministic
        return base_task, base_metric, False

    allowed_tasks = list(SUPPORTED[base_kind])
    allowed_metrics = sorted({"accuracy", "macro_f1"} if base_task != "regression" else {"r2"})

    def _verify(raw):
        try:
            tt = (raw or {}).get("task_type")
            mt = (raw or {}).get("metric")
            if tt not in allowed_tasks:
                return False, None, f"task_type {tt!r} not in {allowed_tasks}"
            mt = _METRIC_CANON.get(mt, mt)
            if (tt == "regression") != (base_task == "regression"):
                # the LLM may not flip between regression and classification -- that is a STRUCTURAL fact
                # the deterministic inspector owns (target dtype/cardinality), not a refinement.
                return False, None, "may not change regression-vs-classification (structural)"
            if mt not in allowed_metrics:
                return False, None, f"metric {mt!r} not certifiable for this task"
            return True, {"task_type": tt, "metric": mt}, None
        except Exception as ex:  # noqa: BLE001
            return False, None, str(ex)[:160]

    def _fallback(_reason):
        return {"task_type": base_task, "metric": base_metric}

    schema = {"type": "object",
              "properties": {"task_type": {"type": "string", "enum": allowed_tasks},
                             "metric": {"type": "string", "enum": allowed_metrics}},
              "required": ["task_type", "metric"]}
    req = ops.LLMRequest(
        surface="problem_type.refine",
        system=("You refine the TASK TYPE and METRIC for a supervised ML problem within a FIXED supported "
                "set. You may NOT change a regression problem into classification or vice versa (that is a "
                "structural property of the target). Choose only from the provided enums. Output the tool."),
        trusted_context={"profile": profile, "base_kind": base_kind, "base_task": base_task,
                         "base_metric": base_metric, "allowed_tasks": allowed_tasks,
                         "allowed_metrics": allowed_metrics},
        untrusted_inputs={"goal_text": goal_text or ""},      # goal text is user-controlled -> quarantined
        schema=schema, tool_name="emit_result", prompt_version="v1", force_tool=True)
    p = ops.llm_propose(req, kind="NON_BINDING", verify=_verify, fallback=_fallback,
                        api_key=api_key, use_llm=True, cache_path=cache_path)
    val = p.value or {}
    return val.get("task_type", base_task), _METRIC_CANON.get(val.get("metric"), base_metric), bool(p.used_llm)


def classify(records, goal_text, *, use_llm=False, api_key=None, target_key=None, cache_path=None):
    """Classify (records, goal_text) into a supported (kind, task_type, metric).

    Returns: {"kind", "task_type", "metric", "supported": bool, "decline_reason": str|None,
              "confidence": "high"|"medium"|"low", "source": "deterministic"|"llm"}.

    Deterministic-first (admissibility.inspect drives the structural call); the LLM, if enabled, only
    refines task/metric within the supported set and never promotes. An unsupported problem returns
    supported=False with a concrete decline_reason; we NEVER coerce it into a fake fit."""
    prof = admissibility.inspect(records, target_key=target_key)

    # an inadmissible frame cannot be a supported problem -- decline with the inspector's verdict.
    if not prof["admissible"]:
        return {"kind": prof["suggested_kind"], "task_type": prof["suggested_task_type"],
                "metric": _METRIC_CANON.get(prof["suggested_metric"], prof["suggested_metric"]),
                "supported": False, "decline_reason": f"inadmissible data: {prof['verdict']}",
                "confidence": "high", "source": "deterministic"}

    # explicit out-of-scope problems (forecast/ranking/audio/multilabel) -> honest decline with the reason.
    scope = _scope_decline_reason(goal_text)
    if scope is not None:
        return {"kind": prof["suggested_kind"], "task_type": prof["suggested_task_type"],
                "metric": _METRIC_CANON.get(prof["suggested_metric"], prof["suggested_metric"]),
                "supported": False, "decline_reason": scope, "confidence": "high",
                "source": "deterministic"}

    base_task = prof["suggested_task_type"]
    base_metric = _METRIC_CANON.get(prof["suggested_metric"], prof["suggested_metric"])

    # MODALITY: default tabular. Route to vision iff the goal hints at images AND the matrix is image-shaped
    # (regression-on-images is out of the vision harness's scope -> stays tabular regression honestly).
    kind = "tabular"
    if base_task != "regression" and _looks_like_images(goal_text, prof["schema"], prof["n_features"]):
        kind = "vision"

    confidence = "high" if kind == "tabular" else "medium"
    source = "deterministic"

    if use_llm:
        task, metric, used = _refine_with_llm(prof, goal_text, kind, base_task, base_metric,
                                              api_key=api_key, cache_path=cache_path)
        base_task, base_metric = task, metric
        if used:
            source = "llm"

    # final supported-set guard (defense in depth; the routing above only ever yields supported combos).
    supported = base_task in SUPPORTED.get(kind, ())
    decline_reason = None
    if not supported:
        decline_reason = (f"no built harness for kind={kind!r}, task_type={base_task!r}. Supported: "
                          f"tabular[binary|multiclass|regression], text[binary|multiclass], "
                          f"vision[binary|multiclass].")

    return {"kind": kind, "task_type": base_task, "metric": base_metric, "supported": bool(supported),
            "decline_reason": decline_reason, "confidence": confidence, "source": source}


# --------------------------------------------------------------------------- self-test
def _selftest():
    p = f = 0

    def check(name, cond):
        nonlocal p, f
        if cond:
            print(f"  PASS  {name}"); p += 1
        else:
            print(f"  FAIL  {name}"); f += 1

    # tabular multiclass classification frame -> (tabular, multiclass)
    rows = [{"features": {"a": float(i), "b": float(i % 5)}, "target": str(i % 3)} for i in range(60)]
    r = classify(rows, "classify these rows into their category")
    check("tabular multiclass supported", r["supported"] and r["kind"] == "tabular"
          and r["task_type"] == "multiclass")
    check("metric is certifiable", r["metric"] in ("accuracy", "macro_f1", "r2"))

    # regression frame
    rr = classify([{"features": {"a": float(i)}, "target": float(i) * 2.0} for i in range(40)],
                  "predict the continuous value")
    check("regression supported", rr["supported"] and rr["task_type"] == "regression"
          and rr["metric"] == "r2")

    # image-shaped frame + image goal -> vision
    img = [{"features": {f"p{j}": float((i + j) % 16) for j in range(64)}, "target": str(i % 4)}
           for i in range(60)]
    rv = classify(img, "classify these 8x8 pixel images by digit")
    check("image-shaped frame routes to vision", rv["supported"] and rv["kind"] == "vision"
          and rv["task_type"] == "multiclass")

    # out-of-scope: forecasting -> declined with a reason
    rt = classify(rows, "forecast the next day's value in this time series")
    check("forecast declined", (not rt["supported"]) and rt["decline_reason"] is not None)

    # out-of-scope: ranking -> declined
    rk = classify(rows, "rank the search results by relevance")
    check("ranking declined", (not rk["supported"]) and "ranking" in rk["decline_reason"])

    # inadmissible (constant target) -> declined
    rc = classify([{"features": {"a": float(i)}, "target": "x"} for i in range(40)], "classify")
    check("inadmissible declined", (not rc["supported"]) and "inadmissible" in rc["decline_reason"])

    print(f"  ---- {p} passed, {f} failed ----")
    return f == 0


if __name__ == "__main__":
    import sys
    print("== problem_type self-test ==")
    sys.exit(0 if _selftest() else 1)
