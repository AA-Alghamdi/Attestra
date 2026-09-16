"""Attestera local web app, stdlib-only server (no FastAPI/Flask dependency).

Serves a single-page app with a live 2D workflow diagram wired to the ACTUAL
vfplatform autoresearch loop. The loop's `on_event(event_dict)` callback is the
live event source: each stage event is pushed onto a per-run queue and streamed
to the browser over Server-Sent-Events (SSE).

Endpoints:
  GET  /                 -> webapp/static/index.html (the SPA)
  POST /run              -> {dataset|rows, goal, threshold, metric} -> {run_id}
                            (starts run_goal_loop on a BACKGROUND THREAD)
  GET  /events/{run_id}  -> text/event-stream of each loop event, then a final
                            "result" event (full certificate / failure), then close.

Design notes:
  * stdlib ThreadingHTTPServer: each request is its own thread, so an SSE stream
    that blocks on a queue does not block other requests (POST /run, a 2nd stream).
  * One Queue per run_id. The run thread is the sole producer; the SSE handler is
    the sole consumer. A sentinel (_DONE) closes the stream cleanly.
  * Exceptions in the run thread are caught and emitted as an {"stage":"error"}
    event so the UI never hangs.

Run: /Users/abdullahalghamdi/jax-env-311/bin/python webapp/server.py
"""
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- make the repo importable (server.py lives in webapp/) -----------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vfplatform.connectors import load_sklearn          # noqa: E402
from vfplatform.loop import run_goal_loop               # noqa: E402
from vfplatform.serving import ModelRegistry            # noqa: E402

_REGISTRY = ModelRegistry(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "vf_runs", "_ui_registry"))   # certified UI models -> servable
from vfplatform.providers import LocalCpuProvider       # noqa: E402
import spendcap                                          # noqa: E402  (webapp/ is on sys.path[0] as the script dir)

HOST = os.environ.get("VF_HOST", "127.0.0.1")   # 0.0.0.0 in Docker/containers; 127.0.0.1 for local-only
PORT = int(os.environ.get("VF_PORT", "8765"))
STATIC_DIR = os.path.join(_HERE, "static")
# Finished run results are persisted here so every certificate gets a shareable permalink
# (GET /api/cert/{run_id}); the payload only ever contains what was already streamed to the UI.
CERT_DIR = os.environ.get("VF_CERT_DIR") or os.path.join(_HERE, "certs")
os.makedirs(CERT_DIR, exist_ok=True)


def _persist_result(run_id, payload):
    try:
        rec = dict(payload)
        rec["run_id"] = run_id
        rec["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(os.path.join(CERT_DIR, f"{run_id}.json"), "w") as f:
            json.dump(rec, f)
    except Exception:  # noqa: BLE001  (a persistence hiccup must never break the stream)
        pass


_EVENTS_CAP = 4000                                       # bound the stored stream per run


def _persist_events(run_id, events):
    """Persist the run's full event stream so any past run can be replayed through the diagram
    (GET /api/replay/{run_id}). Only carries what was already streamed to the browser."""
    try:
        with open(os.path.join(CERT_DIR, f"{run_id}.events.json"), "w") as f:
            json.dump({"run_id": run_id, "events": events[:_EVENTS_CAP]}, f)
    except Exception:  # noqa: BLE001
        pass


def _list_runs(limit=50):
    """Summaries of persisted runs (newest first) for the auto-updating verified-runs gallery."""
    out = []
    try:
        names = [n for n in os.listdir(CERT_DIR) if n.endswith(".json") and not n.endswith(".events.json")]
    except OSError:
        return out
    for n in names:
        try:
            with open(os.path.join(CERT_DIR, n)) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        cert = d.get("certificate") or {}
        win = d.get("winner") or {}
        out.append({
            "run_id": d.get("run_id") or n[:-5],
            "saved_at": d.get("saved_at"),
            "decision": d.get("decision"),
            "certified": bool(d.get("certified")),
            "dataset": d.get("dataset"),
            "goal": d.get("goal"),
            "engine": d.get("engine", "catalog"),
            "metric": d.get("metric") or cert.get("metric"),
            "sealed_score": cert.get("observed"),
            "lower_bound": cert.get("lower_bound"),
            "theta": cert.get("theta"),
            "winner_family": win.get("family") or cert.get("winner_family"),
            "provider": d.get("provider"),
            "elapsed_s": d.get("elapsed_s"),
            "has_replay": os.path.exists(os.path.join(CERT_DIR, f"{d.get('run_id') or n[:-5]}.events.json")),
        })
    out.sort(key=lambda r: r.get("saved_at") or "", reverse=True)
    return out[:limit]
BUILTIN_DATASETS = ["breast_cancer", "wine", "iris", "digits", "diabetes"]
VERTICAL_DEMOS = ["timeseries_demo", "ranking_demo"]    # modality verticals (own frozen certifiers)


def _vertical_demo(name):
    """Synthetic demo data + spec for the modality verticals so the UI can show all modalities."""
    import numpy as np
    rng = np.random.default_rng(0)
    if name == "timeseries_demo":
        n = 400; e = rng.normal(size=n); x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.6 * x[t - 1] + e[t]                 # AR(1): autocorrelated -> needs block bootstrap
        rows = [{"target": float(x[t])} for t in range(n)]
        return rows, "timeseries", "forecast", "target", None, "neg_rmse"
    # ranking_demo: per-query item lists with graded relevance
    w = rng.normal(size=4)
    queries = []
    for q in range(140):
        items = []
        for _ in range(8):
            f = rng.normal(size=4)
            rel = int(min(3, max(0, round(float(f @ w) + rng.normal()))))
            items.append({"features": {f"a{j}": float(f[j]) for j in range(4)}, "relevance": rel})
        queries.append({"qid": q, "items": items})
    return queries, "ranking", "ranking", None, None, "ndcg@10"

def _supported_metrics():
    try:
        from vfplatform.sealed import SUPPORTED_METRICS
        return sorted(SUPPORTED_METRICS)
    except Exception:  # noqa: BLE001
        return ["accuracy", "balanced_accuracy", "macro_f1", "r2", "neg_rmse", "neg_mae"]


def _gpu_configured():
    """True if a RunPod GPU lane is *configured* (key + endpoint present). Cheap, no network probe -- the
    deep probe only runs when a user actually requests GPU at run time."""
    epf = os.path.join(_REPO, ".runpod_endpoint")
    ep = os.path.exists(epf) or bool(os.environ.get("RUNPOD_ENDPOINT_ID"))
    kf = os.path.join(_REPO, ".runpod_key")
    key = os.path.exists(kf) or bool(os.environ.get("RUNPOD_API_KEY"))
    return bool(ep and key)


def _meta():
    """Capabilities surface the SPA reads on load: datasets, metrics, engines, modalities, live spend, and
    whether the GPU/LLM lanes are configured. Purely descriptive; never touches the sealed test."""
    return {
        "datasets": {
            "tabular": BUILTIN_DATASETS,
            "verticals": VERTICAL_DEMOS,
            "vision": ["vision_demo"],
        },
        "metrics": _supported_metrics(),
        "engines": ["catalog", "frontier"],
        "modalities": ["tabular", "text", "vision", "timeseries", "ranking"],
        "gpu_configured": _gpu_configured(),
        "llm_configured": _llm_key_present(),
        "spend": spendcap.snapshot(),
    }


_DONE = object()                                        # sentinel: end of a run's event stream

# run_id -> {"queue": Queue, "created": float}. A modest registry; old runs are pruned lazily.
_RUNS = {}
_RUNS_LOCK = threading.Lock()


def _new_run():
    run_id = uuid.uuid4().hex[:12]
    q = queue.Queue()
    cancel = threading.Event()                          # set by POST /cancel; polled by the loop's should_cancel
    with _RUNS_LOCK:
        # lazy prune: drop runs older than 1h to bound memory on a long-lived server
        now = time.time()
        for rid in [k for k, v in _RUNS.items() if now - v["created"] > 3600]:
            _RUNS.pop(rid, None)
        _RUNS[run_id] = {"queue": q, "created": now, "cancel": cancel, "done": False}
    return run_id, q, cancel


def _get_queue(run_id):
    with _RUNS_LOCK:
        rec = _RUNS.get(run_id)
    return rec["queue"] if rec else None


def _cancel_run(run_id):
    """Signal a streaming run to stop. Returns (cancelled: bool, reason: str|None). Unknown/finished runs
    return cancelled=False with a reason (the loop already produced its terminal result)."""
    with _RUNS_LOCK:
        rec = _RUNS.get(run_id)
        if rec is None:
            return False, "unknown run_id"
        if rec.get("done") or rec.get("cancel") is None:
            return False, "run already finished"
        rec["cancel"].set()
    return True, None


def _mark_done(run_id):
    """Mark a run finished so a later /cancel reports honestly, and drop its cancel Event (it can no longer
    do anything). The queue itself is kept until lazy prune so a slow SSE consumer can still drain the tail."""
    with _RUNS_LOCK:
        rec = _RUNS.get(run_id)
        if rec is not None:
            rec["done"] = True
            rec.pop("cancel", None)


def _build_records(body):
    """Resolve the request body into (records, kind, task_type, target_key, labels, metric).

    Either `dataset` (a built-in sklearn name) or `rows` (a list of nested records
    {"features": {...}, "target": ...}). Raises ValueError on bad input -> 400.
    """
    dataset = body.get("dataset")
    rows = body.get("rows")
    csv_text = body.get("csv")
    if csv_text:                                        # uploaded CSV -> records (target_col optional)
        rows = _csv_to_records(csv_text, body.get("target_col"))
    if dataset in VERTICAL_DEMOS:
        recs, kind, task, tkey, labels, metric = _vertical_demo(dataset)
        return recs, kind, task, tkey, labels, metric   # vertical's own metric (neg_rmse / ndcg@10)
    if dataset == "vision_demo":                         # first new modality: image classification (digits-as-images)
        from vfplatform.vision import vision_demo as _vdemo
        d = _vdemo()
        return (d["records"], d["kind"], d["task_type"], d["target_key"], d.get("labels"),
                body.get("metric") or d["metric"])
    if dataset:
        if dataset not in BUILTIN_DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}; available: {BUILTIN_DATASETS + VERTICAL_DEMOS}")
        b = load_sklearn(dataset)
        return (b["records"], b["kind"], b["task_type"], b["target_key"], b.get("labels"),
                body.get("metric") or b.get("metric"))
    if rows:
        if not isinstance(rows, list) or not rows:
            raise ValueError("`rows` must be a non-empty list of records")
        first = rows[0]
        if not isinstance(first, dict) or "features" not in first or "target" not in first:
            raise ValueError('each row must be a dict like {"features": {...}, "target": ...}')
        # ADMISSIBILITY GATE (Phase 3): profile ARBITRARY uploaded data and decline honestly on a hard issue
        # (constant target, a feature identical to the target, too few rows) instead of certifying a broken
        # problem. Degrades gracefully (skips the gate) if the inspector itself errors.
        v = None
        try:
            from vfplatform.admissibility import inspect as _inspect
            v = _inspect(rows, target_key="target")
        except Exception:  # noqa: BLE001
            v = None
        if v is not None and v.get("admissible") is False:
            raise ValueError("dataset not admissible: " + v.get("verdict", "failed admissibility checks")
                             + (("; issues: " + "; ".join(v.get("issues", []))) if v.get("issues") else ""))
        targets = sorted({str(r.get("target")) for r in rows})
        task = (v or {}).get("suggested_task_type")
        if task == "regression":
            return rows, "tabular", "regression", "target", None, (
                body.get("metric") or (v or {}).get("suggested_metric") or "r2")
        if task not in ("binary", "multiclass"):
            is_reg = all(_isfloat(r.get("target")) for r in rows) and len(targets) > 12
            if is_reg:
                return rows, "tabular", "regression", "target", None, body.get("metric") or "r2"
            task = "binary" if len(targets) == 2 else "multiclass"
        return (rows, "tabular", task, "target", targets,
                body.get("metric") or (v or {}).get("suggested_metric") or "accuracy")
    raise ValueError("provide either `dataset` (a built-in name) or `rows` (a list of records)")


def _isfloat(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _threshold_from_goal(goal):
    """Deterministically pull a target threshold from the goal text: '93%' / '92.5 %' -> 0.93/0.925,
    or '>=0.9' / 'at least 0.88' -> 0.9/0.88. Returns None if no clear number (then the UI input stands)."""
    import re
    if not goal:
        return None
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", goal)
    if m:
        v = float(m.group(1)) / 100.0
        return v if 0.0 < v <= 1.0 else None
    m = re.search(r"(?:>=|≥|>|at\s+least|hit|reach|target|threshold|bound|of)\s*(0?\.\d+)", goal, re.I)
    if m:
        v = float(m.group(1))
        return v if 0.0 < v <= 1.0 else None
    return None


def _objective_and_budget_from_goal(goal):
    """Deterministically parse the goal text for a search OBJECTIVE and a wall-clock TIME BUDGET.
      * "(at least|for|run for)? N (min|minute|mins|minutes)" -> time_budget_s = N*60
      * "N (sec|secs|second|seconds)"                          -> time_budget_s = N
      * "highest accuracy" / "maximize" / "best you can" / "as accurate as possible" -> objective="maximize"
    Returns (objective: str|None, time_budget_s: float|None); None means "no signal in the text". Explicit
    POST body fields override these parsed values upstream."""
    import re
    objective, budget = None, None
    if not goal:
        return objective, budget
    g = goal.lower()
    if re.search(r"\b(highest|max(?:imi[sz]e|imal)?|best (?:you can|possible)|"
                 r"as accurate as possible|most accurate)\b", g):
        objective = "maximize"
    m = re.search(r"(?:at\s+least\s+|for\s+|run\s+for\s+|over\s+)?(\d+(?:\.\d+)?)\s*"
                  r"(min(?:ute)?s?|m)\b", g)
    if m:
        budget = float(m.group(1)) * 60.0
    else:
        m = re.search(r"(?:at\s+least\s+|for\s+|run\s+for\s+|over\s+)?(\d+(?:\.\d+)?)\s*"
                      r"(sec(?:ond)?s?|s)\b", g)
        if m:
            budget = float(m.group(1))
    return objective, budget


def _llm_infer_spec(goal, records, default_metric):
    """LLM intake (resolver-verified) -> {task_type, metric, target, used_llm, needs_human}. The LLM proposes
    against the real columns; the deterministic resolver narrows/rejects. Returns None on any failure (so the
    caller falls back to the dropdown spec) -- 'delete the LLM -> deterministic'."""
    try:
        from vectorforge.llm_shell.intake import intake
        from vfplatform.sealed import SUPPORTED_METRICS
        if intake is None:                       # ar/intake unavailable (slim deploy) -> deterministic
            return None
        ic = intake(records, goal, use_llm=True)
        rs = ic.resolved
        metric = rs.metric if rs.metric in SUPPORTED_METRICS else default_metric
        contract = getattr(ic, "contract", {}) or {}
        return {"task_type": rs.task_type or None, "metric": metric, "target": rs.target,
                "used_llm": bool(contract.get("used_llm")), "needs_human": bool(rs.needs_human)}
    except Exception:  # noqa: BLE001
        return None


def _llm_key_present():
    """True only when a usable Anthropic key is configured, so the spend cap charges the LLM lane ONLY when a
    paid call will actually be made (no key -> intake falls back to free deterministic inference -> no charge)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    try:
        from vectorforge.llm_shell._keys import resolve_api_key
        k = resolve_api_key(None)
        return bool(k) and "PASTE" not in (k or "")
    except Exception:  # noqa: BLE001
        return False


def _csv_to_records(csv_text, target_col=None):
    """Parse uploaded CSV text into nested records {"features": {...}, "target": ...}. The target column is
    `target_col` if given, else a column literally named 'target', else the LAST column. Numeric cells are
    coerced to float; everything else stays a string (the featurizer one-hots categoricals)."""
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if not rows:
        raise ValueError("CSV has no data rows")
    cols = list(rows[0].keys())
    tcol = target_col or ("target" if "target" in cols else cols[-1])
    if tcol not in cols:
        raise ValueError(f"target column {tcol!r} not found; columns: {cols}")
    out = []
    for r in rows:
        feats = {}
        for c in cols:
            if c == tcol:
                continue
            v = r[c]
            feats[c] = float(v) if _isfloat(v) else v
        tv = r[tcol]
        out.append({"features": feats, "target": float(tv) if _isfloat(tv) else tv})
    return out


def _records_to_xy(records, target_key="target"):
    """Project nested records {"features": {...}, "target": ...} onto a dense numeric (X, y) for the
    frontier CoreOrchestrator, which consumes ndarray X + label vector y (not the catalog's record list).
    Categorical feature cells are one-hot encoded via DictVectorizer so arbitrary uploads still run.
    Returns (X: np.ndarray, y: np.ndarray)."""
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    feats = [dict(r.get("features", {})) for r in records]
    y = np.array([r.get(target_key) for r in records])
    X = DictVectorizer(sparse=False).fit_transform(feats)
    return np.asarray(X, dtype=float), y


def _run_frontier_thread(run_id, q, records, goal, threshold, metric, kind, task_type, target_key,
                         labels, cancel_event=None, time_budget_s=None, use_gpu=False, dataset=None):
    """Background worker for the FRONTIER engine path: convert records -> (X, y), drive the
    CoreOrchestrator with its on_event sink wired to the SSE queue, then push a terminal 'result'
    payload + the _DONE sentinel. Mirrors _run_loop_thread's contract so the SPA is engine-agnostic.

    The frontier engine streams its own stage vocabulary (intake/split/round/propose/measure/done);
    the sink only ever carries VAL/provenance fields, never sealed-test labels (orchestrator invariant)."""

    ev_log = []

    def on_event(ev):
        try:
            if len(ev_log) < _EVENTS_CAP:
                ev_log.append(ev)
            q.put({"type": "event", "data": ev})
        except Exception:  # noqa: BLE001
            pass

    try:
        from frontier.core.orchestrator import CoreOrchestrator, CoreConfig
        on_event({"stage": "engine", "status": "active", "engine": "frontier"})
        # The frontier engine dispatches neural proposals through the LOCAL torch execution substrate
        # (orchestrator._make_gpu_dispatch); it has no remote GPU lane. Report the compute honestly so the
        # UI's fan-out panel shows a device instead of staying blank when this engine is selected.
        cuda_ok, gpu_reason = False, None
        try:
            import torch  # noqa: F401
            cuda_ok = bool(torch.cuda.is_available())
            if use_gpu and not cuda_ok:
                gpu_reason = "no CUDA device on this host; frontier trains on the local CPU substrate"
        except ImportError:
            if use_gpu:
                gpu_reason = "torch is not installed on this host; frontier trains on the local CPU substrate"
        on_event({"stage": "provider", "status": "active",
                  "provider": "frontier-local", "gpu": cuda_ok,
                  "gpu_requested": bool(use_gpu), "gpu_available": cuda_ok,
                  "gpu_unavailable_reason": (gpu_reason if (use_gpu and not cuda_ok) else None),
                  "gpu_workers_max": 1})
        X, y = _records_to_xy(records, target_key=target_key or "target")
        total_seconds = float(time_budget_s) if time_budget_s else 180.0
        cfg = CoreConfig(rounds=4, seed=0, llm_client=None,
                         total_seconds=total_seconds,
                         enable_neural=True, enable_knowledge=True, enable_intelligence=True,
                         on_event=on_event)
        res = CoreOrchestrator(cfg).run(goal=goal, X=X, y=y,
                                        theta=float(threshold), name=f"ui-{run_id}")
        cert = res.certificate
        winner = res.winner
        win_payload = None
        if winner is not None:
            win_payload = {"family": getattr(winner, "label", None),
                           "params": getattr(winner, "spec", None)}
        payload = {
            "decision": ("certified" if res.certified else "declined"),
            "certified": bool(res.certified),
            "certificate": cert,
            "failure_report": ({"reason": res.decline_reason} if not res.certified else None),
            "winner": win_payload,
            "winner_val_score": res.winner_val_score,
            "rounds": res.diagnosis_trail,
            "provider": "frontier-local",
            "modality": kind,
            "engine": "frontier",
            "decline_reason": res.decline_reason,
            "sealed_peeks": res.sealed_peeks,
            "dataset": dataset,
            "goal": goal,
            "metric": metric,
        }
        _persist_result(run_id, payload)
        _persist_events(run_id, ev_log)
        q.put({"type": "result", "data": payload})
    except Exception as ex:  # noqa: BLE001
        tb = traceback.format_exc()
        q.put({"type": "event", "data": {"stage": "error", "status": "done",
                                         "error": str(ex), "traceback": tb[-1500:]}})
        q.put({"type": "result", "data": {"decision": "error", "certified": False,
                                          "error": str(ex), "certificate": None,
                                          "failure_report": None, "winner": None}})
    finally:
        _mark_done(run_id)
        q.put(_DONE)


def _gpu_provider(workers_max=None, gpu_type=""):
    """Build a RunPod GPU provider from the gitignored .runpod_endpoint + .runpod_key. Returns a TUPLE
    (provider_or_None, reason_or_None): the provider when the GPU lane is configured AND a deep probe says it
    is usable; otherwise (None, reason) so the caller falls back to CPU and the UI says so honestly.

    `workers_max` caps the CLIENT-SIDE concurrent submit/poll fan-out. `gpu_type` is the user's preferred
    GPU. HONEST NOTE: for the SERVERLESS endpoint the GPU type is fixed at the ENDPOINT (built when the
    worker image was deployed), so per-request gpu_type is NOT honorable here. We pass it to the provider's
    `gpu` field (used only for cost labeling) and ECHO it back to the UI; we do not pretend it changes the
    hardware actually allocated.

    The deep check is provider.probe(): cheap presence (available()) is NOT enough, since a configured key can
    still be expired, the endpoint unreachable, or all workers unhealthy. probe() returns
    {"available": bool, "reason": str|None, "health": <summary dict>}. We surface its reason verbatim so the
    UI can show an honest 'GPU unavailable: <reason>. Running on CPU.' note. Never raises (any failure ->
    (None, reason))."""
    try:
        from vfplatform.providers import RunPodProvider
        epf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".runpod_endpoint")
        ep = open(epf).read().strip() if os.path.exists(epf) else os.environ.get("RUNPOD_ENDPOINT_ID")
        kw = {"endpoint_id": ep, "workers_max": workers_max}
        if gpu_type:
            kw["gpu"] = gpu_type
        p = RunPodProvider(**kw)
        probe = p.probe()
        if probe.get("available"):
            return p, None
        return None, (probe.get("reason") or "RunPod not available")
    except Exception as ex:  # noqa: BLE001
        return None, f"RunPod probe failed: {ex}"


def _run_loop_thread(run_id, q, records, goal, threshold, metric, kind, task_type, target_key, labels,
                     test_records=None, use_gpu=False, use_llm_intake=False,
                     gpu_workers_max=3, gpu_type="", cancel_event=None,
                     objective="certify", time_budget_s=None, dataset=None):
    """Background worker: drive run_goal_loop, pushing every on_event onto q, then a terminal
    'result' payload, then the _DONE sentinel. Any exception becomes an error event.
    `test_records` (the user's own held-out verification set) becomes the sealed test when supplied.
    `use_gpu` opts into parallel RunPod GPU fan-out (real spend); falls back to CPU if unconfigured.
    `gpu_workers_max` caps parallel RunPod workers (and LocalCpuProvider.max_workers on the CPU path).
    `gpu_type` is the user's preferred RunPod GPU (set at the endpoint for serverless; echoed honestly).
    `cancel_event` is a threading.Event; when set the loop stops at the next round/candidate boundary."""

    should_cancel = (lambda: cancel_event.is_set()) if cancel_event is not None else None

    ev_log = []

    def on_event(ev):
        # best-effort copy onto the stream; never let a queue hiccup break the science loop
        try:
            if len(ev_log) < _EVENTS_CAP:
                ev_log.append(ev)
            q.put({"type": "event", "data": ev})
        except Exception:  # noqa: BLE001
            pass

    try:
        from vfplatform.frontdoor import run as frontdoor_run    # routes tabular/text + verticals uniformly
        # AI goal understanding: let the LLM infer task/metric/target from the free-text goal (resolver-
        # verified) + a deterministic threshold parse from the goal text. Overrides the dropdowns; falls
        # back to them if the LLM/intake is unavailable. Only for the i.i.d. modalities.
        llm_capped, cap_reason = False, None
        if use_llm_intake and kind in ("tabular", "text"):
            # spend backstop: charge the LLM lane ONLY when a paid call will really happen (key present). If
            # the daily cap is hit, degrade to the deterministic dropdown spec + threshold parse -- honest, free.
            if _llm_key_present():
                allowed, cap_reason, _snap = spendcap.try_charge("llm")
                llm_capped = not allowed
            inf = None if llm_capped else _llm_infer_spec(goal, records, metric)
            tg = _threshold_from_goal(goal)
            if inf:
                task_type = inf.get("task_type") or task_type
                metric = inf.get("metric") or metric
            if tg is not None:
                threshold = tg
            on_event({"stage": "ai_intake", "status": "active", "ai_intake": True,
                      "used_llm": bool(inf and inf.get("used_llm")),
                      "capped": llm_capped, "cap_reason": cap_reason,
                      "inferred": {"task_type": task_type, "metric": metric, "threshold": threshold,
                                   "target": (inf or {}).get("target")}})
        # spend backstop: charge the GPU lane ONLY when a RunPod provider is actually live (otherwise the
        # toggle is already a free CPU fallback). If the daily cap is hit, degrade to CPU with an honest note.
        provider, gpu_capped, gpu_cap_reason = None, False, None
        gpu_unavailable_reason = None      # set when GPU was requested but a deep probe rejected the lane
        try:
            _workers_cap = max(1, int(gpu_workers_max))
        except (TypeError, ValueError):
            _workers_cap = 3
        if use_gpu:
            gp, gpu_unavailable_reason = _gpu_provider(workers_max=_workers_cap, gpu_type=gpu_type)
            if gp is not None:
                allowed, gpu_cap_reason, _snap = spendcap.try_charge("gpu")
                if allowed:
                    provider = gp
                else:
                    gpu_capped = True
                    gpu_unavailable_reason = gpu_cap_reason   # capped is also an honest "not on GPU" reason
        # CPU fallback (or CPU-only run): cap local parallelism by the same gpu_workers_max knob so the UI's
        # "max parallel workers" control means something on both lanes.
        provider = provider or LocalCpuProvider(max_workers=_workers_cap)
        gpu_available = provider.name == "runpod-gpu"
        # HONEST GPU-type echo: for the serverless endpoint the GPU type is fixed at the endpoint, so we
        # record what the user REQUESTED rather than claiming we set it per request (the UI shows
        # "(set at the RunPod endpoint)" next to it). gpu_requested/gpu_available/gpu_unavailable_reason let
        # the UI show an HONEST "GPU unavailable: <reason>. Running on CPU." note (we never fake GPU activity).
        on_event({"stage": "provider", "status": "active", "provider": provider.name,
                  "gpu": gpu_available,
                  "gpu_requested": bool(use_gpu),
                  "gpu_available": gpu_available,
                  "gpu_unavailable_reason": (gpu_unavailable_reason if (use_gpu and not gpu_available) else None),
                  "gpu_type_requested": gpu_type or "",
                  "gpu_workers_max": _workers_cap,
                  "capped": gpu_capped, "cap_reason": gpu_cap_reason})   # tell the UI which compute is running
        # LLM proposals are gated by the same user toggle + spend cap as intake: with the toggle off (or the
        # daily cap hit) the loop must use the free deterministic proposal catalog even when a key is present.
        llm_ok = bool(use_llm_intake) and not llm_capped and _llm_key_present()
        kw = dict(providers=[provider], on_event=on_event, should_cancel=should_cancel,
                  llm_propose=llm_ok,
                  objective=objective, time_budget_s=time_budget_s)
        if provider.name == "runpod-gpu":
            # ticking the GPU box IS the spend consent -> clear the Checkpoint spend-gate for this run
            from vfplatform.checkpoint import Checkpoint
            kw["checkpoint"] = Checkpoint(approve_spend=True)
        if kind in ("tabular", "text"):                          # i.i.d. loop accepts these extras
            kw.update(test_records=test_records, registry=_REGISTRY, experiment=f"ui-{run_id}")
        out = frontdoor_run(records, goal, threshold=float(threshold), kind=kind, task_type=task_type,
                            target_key=target_key, metric=metric, labels=labels, **kw)
        res = out["result"]
        winner = getattr(res, "winner", None)
        win_payload = None
        if winner is not None:
            win_payload = winner if isinstance(winner, dict) else {
                "family": getattr(winner, "family", None), "params": getattr(winner, "params", None)}
        cert = getattr(res, "certificate", None)
        payload = {
            "decision": getattr(res, "decision", None),
            "certified": bool(cert and cert.get("certified")),
            "certificate": cert,
            "failure_report": getattr(res, "failure_report", None),
            "winner": win_payload,
            "rounds": getattr(res, "rounds", []),
            "n_test": getattr(res, "n_test", None),
            "provider": getattr(res, "provider", "local"),
            "modality": getattr(res, "modality", kind),
            "objective": getattr(res, "objective", objective),
            "time_budget_s": getattr(res, "time_budget_s", time_budget_s),
            "elapsed_s": getattr(res, "elapsed_s", None),
            "served_version": (cert or {}).get("served_version") if cert else None,
            "dataset": dataset,
            "goal": goal,
            "metric": metric,
            "engine": "catalog",
        }
        _persist_result(run_id, payload)
        _persist_events(run_id, ev_log)
        q.put({"type": "result", "data": payload})
    except Exception as ex:  # noqa: BLE001
        tb = traceback.format_exc()
        q.put({"type": "event", "data": {"stage": "error", "status": "done",
                                         "error": str(ex), "traceback": tb[-1500:]}})
        q.put({"type": "result", "data": {"decision": "error", "certified": False,
                                          "error": str(ex), "certificate": None,
                                          "failure_report": None, "winner": None}})
    finally:
        _mark_done(run_id)        # a later /cancel now reports honestly; the run's cancel Event is dropped
        q.put(_DONE)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep-alive so SSE can stream chunked

    def log_message(self, fmt, *args):     # quieter, single-line access log
        sys.stderr.write("[vf] %s - %s\n" % (self.address_string(), fmt % args))

    # ---- helpers --------------------------------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self._send_json({"error": f"not found: {os.path.basename(path)}"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- routing --------------------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
            return
        if path == "/healthz":          # liveness + today's spend-vs-cap, for the reverse proxy / operator
            self._send_json({"ok": True, "spend": spendcap.snapshot()})
            return
        if path == "/health":
            self._send_json({"ok": True, "datasets": BUILTIN_DATASETS})
            return
        if path == "/api/meta":         # capabilities surface for the SPA (datasets, metrics, engines, spend)
            self._send_json(_meta())
            return
        if path.startswith("/api/cert/"):    # certificate permalink: the persisted result of a past run
            rid = path[len("/api/cert/"):]
            if not (rid and all(c in "0123456789abcdef" for c in rid) and len(rid) <= 32):
                self._send_json({"error": "bad run id"}, status=400)
                return
            self._send_file(os.path.join(CERT_DIR, f"{rid}.json"), "application/json")
            return
        if path == "/api/runs":              # persisted-run summaries for the auto-updating gallery
            self._send_json({"runs": _list_runs()})
            return
        if path.startswith("/api/replay/"):  # a past run's stored event stream, for diagram replay
            rid = path[len("/api/replay/"):]
            if not (rid and all(c in "0123456789abcdef" for c in rid) and len(rid) <= 32):
                self._send_json({"error": "bad run id"}, status=400)
                return
            self._send_file(os.path.join(CERT_DIR, f"{rid}.events.json"), "application/json")
            return
        if path.startswith("/events/"):
            self._stream_events(path[len("/events/"):])
            return
        # static passthrough (css/js if split out later)
        safe = os.path.normpath(path).lstrip("/\\")
        candidate = os.path.join(STATIC_DIR, safe)
        if os.path.isfile(candidate) and candidate.startswith(STATIC_DIR):
            ctype = ("text/css" if candidate.endswith(".css")
                     else "application/javascript" if candidate.endswith(".js")
                     else "application/octet-stream")
            self._send_file(candidate, ctype)
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/run", "/cancel"):
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError) as ex:
            self._send_json({"error": f"bad JSON body: {ex}"}, status=400)
            return
        if path == "/cancel":
            run_id = (body.get("run_id") or "").strip()
            if not run_id:
                self._send_json({"cancelled": False, "reason": "`run_id` is required"}, status=400)
                return
            cancelled, reason = _cancel_run(run_id)
            self._send_json({"cancelled": cancelled} if cancelled
                            else {"cancelled": False, "reason": reason})
            return
        goal = (body.get("goal") or "").strip()
        if not goal:
            self._send_json({"error": "`goal` is required"}, status=400)
            return
        threshold = body.get("threshold", 0.9)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            self._send_json({"error": "`threshold` must be a number"}, status=400)
            return
        try:
            records, kind, task_type, target_key, labels, metric = _build_records(body)
        except ValueError as ex:
            self._send_json({"error": str(ex)}, status=400)
            return

        # optional user-supplied verification set (their own held-out annotated examples) -> sealed test
        test_rows = body.get("test_rows")
        if body.get("test_csv"):
            try:
                test_rows = _csv_to_records(body["test_csv"], body.get("target_col"))
            except ValueError as ex:
                self._send_json({"error": f"verification CSV: {ex}"}, status=400)
                return
        if test_rows is not None and (not isinstance(test_rows, list) or not test_rows):
            self._send_json({"error": "`test_rows`, if given, must be a non-empty list of records"}, status=400)
            return

        use_gpu = bool(body.get("use_gpu"))
        use_llm_intake = bool(body.get("use_llm_intake"))
        # GPU options (all optional, backward-compatible): max parallel workers + preferred GPU type.
        try:
            gpu_workers_max = int(body.get("gpu_workers_max", 3))
        except (TypeError, ValueError):
            self._send_json({"error": "`gpu_workers_max` must be an integer"}, status=400)
            return
        if gpu_workers_max < 1:
            self._send_json({"error": "`gpu_workers_max` must be >= 1"}, status=400)
            return
        gpu_type = str(body.get("gpu_type") or "")

        # OBJECTIVE + TIME BUDGET: parse from the goal text, then let explicit body fields OVERRIDE the parse.
        parsed_obj, parsed_budget = _objective_and_budget_from_goal(goal)
        objective = body.get("objective") or parsed_obj or "certify"
        if objective not in ("certify", "maximize"):
            self._send_json({"error": "`objective` must be 'certify' or 'maximize'"}, status=400)
            return
        time_budget_s = body.get("time_budget_s", None)
        if time_budget_s is None:
            time_budget_s = parsed_budget
        if time_budget_s is not None:
            try:
                time_budget_s = float(time_budget_s)
            except (TypeError, ValueError):
                self._send_json({"error": "`time_budget_s` must be a number"}, status=400)
                return
            if time_budget_s <= 0:
                self._send_json({"error": "`time_budget_s` must be > 0"}, status=400)
                return

        engine = str(body.get("engine") or "catalog").lower()
        if engine not in ("catalog", "frontier"):
            self._send_json({"error": "`engine` must be 'catalog' or 'frontier'"}, status=400)
            return

        run_id, q, cancel_event = _new_run()
        ds_name = body.get("dataset") or ("uploaded" if (body.get("rows") or body.get("csv")) else None)
        if engine == "frontier":
            t = threading.Thread(
                target=_run_frontier_thread,
                args=(run_id, q, records, goal, threshold, metric, kind, task_type, target_key, labels),
                kwargs={"cancel_event": cancel_event, "time_budget_s": time_budget_s, "use_gpu": use_gpu,
                        "dataset": ds_name},
                daemon=True,
            )
        else:
            t = threading.Thread(
                target=_run_loop_thread,
                args=(run_id, q, records, goal, threshold, metric, kind, task_type, target_key, labels),
                kwargs={"test_records": test_rows, "use_gpu": use_gpu, "use_llm_intake": use_llm_intake,
                        "gpu_workers_max": gpu_workers_max, "gpu_type": gpu_type, "cancel_event": cancel_event,
                        "objective": objective, "time_budget_s": time_budget_s, "dataset": ds_name},
                daemon=True,
            )
        t.start()
        # gpu_available must reflect the DEEP probe (key present is not enough: it can be expired, the
        # endpoint unreachable, or all workers unhealthy). Only probe when the user actually requested GPU;
        # otherwise report False without a network round-trip. _gpu_provider returns (provider, reason).
        gpu_probe_available, gpu_probe_reason = (False, None)
        if use_gpu:
            _gp, gpu_probe_reason = _gpu_provider(workers_max=gpu_workers_max, gpu_type=gpu_type)
            gpu_probe_available = _gp is not None
        self._send_json({"run_id": run_id, "kind": kind, "task_type": task_type,
                         "metric": metric, "threshold": threshold, "n_records": len(records),
                         "n_verification": (len(test_rows) if test_rows else 0),
                         "objective": objective, "time_budget_s": time_budget_s,
                         "gpu_requested": use_gpu, "gpu_available": gpu_probe_available,
                         "gpu_unavailable_reason": (gpu_probe_reason if (use_gpu and not gpu_probe_available) else None),
                         "gpu_workers_max": gpu_workers_max, "gpu_type_requested": gpu_type,
                         "spend": spendcap.snapshot()})

    # ---- SSE stream -----------------------------------------------------------------------------
    def _sse_write(self, event, data):
        chunk = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
        # chunked transfer encoding (HTTP/1.1, no Content-Length): size in hex, CRLF, body, CRLF
        self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
        self.wfile.flush()

    def _stream_events(self, run_id):
        q = _get_queue(run_id)
        if q is None:
            self._send_json({"error": f"unknown run_id {run_id!r}"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self._sse_write("open", {"run_id": run_id})
            while True:
                try:
                    item = q.get(timeout=1.0)
                except queue.Empty:
                    # heartbeat comment keeps the connection (and any proxy) alive. The comment MUST end
                    # with its own newline: an SSE comment is ":<text>\n"; without the trailing \n it glues
                    # onto the next "event:" line and swallows that event on slow gaps (chunk body ":\n").
                    self.wfile.write(b"2\r\n:\n\r\n")
                    self.wfile.flush()
                    continue
                if item is _DONE:
                    self._sse_write("close", {"run_id": run_id})
                    break
                self._sse_write(item["type"], item["data"])
            # terminate the chunked stream
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away mid-stream; nothing to clean up


def main():
    os.makedirs(STATIC_DIR, exist_ok=True)
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"Attestera web app running at {url}")
    print(f"  datasets: {', '.join(BUILTIN_DATASETS)}")
    print("  Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
