"""The single front door (build-order item 1): 'type a goal + a dataset' -> inferred spec -> the loop.

Wires Ring-2 intake (A1) + column typing (A2) in front of the Ring-1 platform loop so the user supplies
only a goal string and rows. Flat raw rows are typed (A2) and nested; connector bundles that are already
nested + carry hints skip inference. The inferred spec is NON-binding -- the frozen loop + sealed-test
guard bound everything downstream.
"""
import os
import sys

_VF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VF not in sys.path:
    sys.path.insert(0, _VF)

from dataclasses import dataclass, field
from typing import Optional

from .loop import run_goal_loop, GoalLoopResult
from .sealed import SUPPORTED_METRICS
from .harness import is_supported
from .leaderboard import Leaderboard


@dataclass
class VerticalResult:
    """Normalized result for the modality verticals (ranking / time-series) so EVERY front-door result --
    tabular/text (GoalLoopResult) or a vertical -- exposes the same .decision / .certificate / .winner /
    .n_test surface. The UI and any caller can then render all modalities uniformly."""
    decision: str
    certificate: Optional[dict]
    winner: object
    n_test: int
    provider: str = "local-vertical"
    modality: str = ""
    failure_report: Optional[dict] = None
    raw: dict = field(default_factory=dict)


def _normalize_vertical(kind, raw):
    """Map a vertical's native dict onto the common VerticalResult surface (no loss: the full native dict is
    kept in .raw). ranking exposes status+nested certificate; time-series is itself the certificate+decision."""
    if kind == "ranking":
        cert = raw.get("certificate") or {}
        nq = raw.get("n_queries")
        n_test = (int(cert.get("n")) if isinstance(cert.get("n"), (int, float))
                  else (int(nq) if isinstance(nq, (int, float))
                        else int((nq or {}).get("test", 0) if isinstance(nq, dict) else 0)))
        return VerticalResult(decision=raw.get("status") or ("certified" if cert.get("certified")
                                                             else "do_not_certify"),
                              certificate=cert, winner=raw.get("winner"),
                              n_test=n_test, modality="ranking", raw=raw)
    # time-series: the dict IS the certificate (+ a decision field)
    cert = {k: raw.get(k) for k in ("observed", "n", "metric", "theta", "checks", "alpha_per_check",
                                    "lower_bound", "certified", "reason", "block_size", "n_blocks",
                                    "eff_blocks", "deferred")}
    return VerticalResult(decision=raw.get("decision") or ("certified" if raw.get("certified")
                                                          else "do_not_certify"),
                          certificate=cert, winner=raw.get("winner"),
                          n_test=int(raw.get("n") or 0), modality="timeseries", raw=raw)


def _is_nested(records):
    return bool(records) and isinstance(records[0], dict) and "features" in records[0]


def run(records, goal_text, *, threshold, llm_enabled=False, providers=None, store=None,
        kind=None, task_type=None, target_key=None, metric=None, labels=None,
        text_key="text", min_test_n=100, max_rounds=12, seeds=(0, 1), should_cancel=None,
        objective="certify", time_budget_s=None, llm_propose=True, **loop_kw):
    """Goal + dataset -> certified artifact or honest failure. Returns {spec_inferred, result}.

    objective ("certify"|"maximize"), time_budget_s (wall-clock target), and llm_propose (enable the
    LLM-guided PROPOSE step) are forwarded to the i.i.d. run_goal_loop. They do not apply to the ranking /
    time-series verticals (those have their own modules + certifiers) and are ignored there."""
    spec_inferred = {}
    nested = _is_nested(records)

    # ---- MODALITY VERTICALS: grouped (ranking) / temporal (forecast) data routes to its OWN module with
    # its OWN frozen certifier (per-query / block bootstrap), NOT the i.i.d. run_goal_loop. Dispatched here,
    # before inference, because these data models are not flat i.i.d. rows. ------------------------------
    if kind == "ranking":
        from vfplatform.ranking import run_ranking_goal
        rk = {k: loop_kw[k] for k in ("k", "n_val_queries", "n_test_queries", "on_event", "seed")
              if k in loop_kw}
        raw = run_ranking_goal(records, threshold=threshold, metric=(metric or "ndcg@10"), **rk)
        return {"spec_inferred": {"kind": "ranking", "task_type": "ranking", "metric": metric or "ndcg@10"},
                "result": _normalize_vertical("ranking", raw)}
    if kind == "timeseries":
        from vfplatform.timeseries import run_timeseries_goal
        ts_metric = metric if metric in ("neg_rmse", "neg_mae", "r2") else "neg_rmse"
        ts_rows = records if nested else [{"target": r.get(target_key or "target")} for r in records]
        ts_kw = {k: loop_kw[k] for k in ("horizon", "lags", "windows", "n_val", "n_test", "embargo",
                                         "exog_keys", "seed", "B", "on_event") if k in loop_kw}
        raw = run_timeseries_goal(ts_rows, threshold=threshold, metric=ts_metric,
                                  target_key=(target_key or "target"), **ts_kw)
        return {"spec_inferred": {"kind": "timeseries", "task_type": "forecast", "metric": ts_metric},
                "result": _normalize_vertical("timeseries", raw)}

    # ---- infer the spec when hints are not supplied (raw user data) -----------------------------
    if not (kind and task_type and target_key):
        from vectorforge.llm_shell.intake import intake
        from vectorforge.llm_shell.profiler import profile
        ic = intake(records, goal_text, use_llm=llm_enabled)
        rs = ic.resolved
        target_key = target_key or (rs.target or "target")
        task_type = task_type or (rs.task_type or "binary")
        if metric is None and rs.metric in SUPPORTED_METRICS:
            metric = rs.metric
        view = profile(records)
        kind = kind or view.modality                  # honest: do NOT coerce vision/audio/etc. to tabular
        # VERIFY-SPEC: carry forbidden columns + latency/cost constraints from the resolver into the loop
        # (F23: previously dropped on the floor). The loop enforces them; certified == bound AND lat AND cost.
        inferred_drop = list(getattr(rs, "drop_cols", None) or [])
        _cons = getattr(rs, "constraints", None) or {}
        if inferred_drop:
            loop_kw.setdefault("drop_cols", inferred_drop)
        if _cons.get("max_latency_ms") is not None:
            loop_kw.setdefault("max_latency_ms", _cons["max_latency_ms"])
        if _cons.get("max_cost_usd") is not None:
            loop_kw.setdefault("max_cost_usd", _cons["max_cost_usd"])
        spec_inferred = {"target": rs.target, "task_type": rs.task_type, "metric": rs.metric,
                         "modality": view.modality, "needs_human": rs.needs_human,
                         "drop_cols": inferred_drop, "constraints": _cons,
                         "contract_digest": ic.contract_digest}

    # ---- honest decline for unsupported modality / task type (no coercion, no loop, no peek) -----
    if not is_supported(kind, task_type):
        declined = GoalLoopResult(
            "exp-declined", Leaderboard(metric or "accuracy", []), None, None, "unsupported",
            f"Declined: kind={kind!r}, task_type={task_type!r} has no built path. Supported: "
            f"tabular[binary|multiclass|regression], text[binary|multiclass]. The data is NOT coerced "
            f"into a tabular accuracy run, and no certificate is emitted. (Unsupported modalities/tasks "
            f"are an explicit honest decline until their vertical is built.)", 0, "none")
        return {"spec_inferred": spec_inferred, "result": declined}

    # ---- OPERATIONALIZATION ADVERSARY: spec-aware intake screen (additive, non-binding, no peek) --
    # Runs on the RAW records (column names still match the resolved target_key / forbidden fields) BEFORE
    # the loop spends its one sealed peek. Always attached to spec_inferred for transparency. A 'decline'
    # verdict means the spec is DEGENERATE (theta at/below the metric's trivial baseline -> a constant
    # predictor would 'certify'); in certify mode we refuse rather than emit a vacuous certificate -- a
    # spec error, explicitly distinct from a model honest-stop. Maximize mode never declines (theta is only
    # a target there). Any screen error is swallowed -> treated as clean (it must never break a real run).
    try:
        from . import intake_adversary
        _adv = intake_adversary.screen(
            records, target_key=(target_key or "target"), task_type=task_type,
            metric=(metric or ("r2" if task_type == "regression" else "accuracy")),
            threshold=threshold, drop_cols=loop_kw.get("drop_cols"), labels=labels)
    except Exception:  # noqa: BLE001  the adversary is a soft edge; never let it break intake
        _adv = {"verdict": "clean", "findings": [], "trivial_baseline": None}
    spec_inferred["adversary"] = _adv
    if _adv.get("verdict") == "decline" and objective == "certify":
        msg = ("Declined (degenerate spec): " + "; ".join(
            f["detail"] for f in _adv["findings"] if f.get("severity") == "decline")
            + " No loop was run and no sealed peek was taken -- fix the spec, not the model.")
        declined = GoalLoopResult("exp-declined", Leaderboard(metric or "accuracy", []), None, None,
                                  "declined_spec", msg, 0, "none")
        return {"spec_inferred": spec_inferred, "result": declined}

    # ---- shape the rows for the loop ------------------------------------------------------------
    if kind == "text":
        # Lift the text to a top-level "text" field whether rows are NESTED (connector bundle
        # {"features":{"text":...},"target":...}) or FLAT (raw upload). Detect text BEFORE the `nested`
        # passthrough: connector-nested text used to fall through to `loop_rows = records` and reach the loop
        # unflattened, so TextFeaturizer saw no text and the leakage auditor mis-fired on spurious features.
        if records and isinstance(records[0].get("features"), dict):
            loop_rows = [{"text": str((r.get("features") or {}).get(text_key)
                          or next((v for v in (r.get("features") or {}).values() if isinstance(v, str)), "")),
                          "target": r.get("target")} for r in records]
        else:
            from vectorforge.llm_shell.coltype import type_columns
            from vectorforge.llm_shell.profiler import profile
            typed = type_columns(records, profile(records), use_llm=llm_enabled)
            textcol = next((k for k, dt in typed.columns.items() if dt == "text"), None) or text_key
            loop_rows = [{"text": str(r.get(textcol, "")), "target": r.get(target_key)} for r in records]
    elif nested:
        loop_rows = records                                   # connector bundle: already {"features",target}
    else:
        # tabular raw: A2-type, clean (normalize numerics, drop id/datetime), nest under "features"
        from vectorforge.llm_shell.coltype import type_columns, normalize_numeric
        from vectorforge.llm_shell.profiler import profile
        typed = type_columns(records, profile(records), use_llm=llm_enabled)
        spec_inferred["column_types"] = typed.columns

        def feats(r):
            out = {}
            for k, v in r.items():
                if k == target_key:
                    continue
                dt = typed.columns.get(k, "categorical")
                if dt == "numeric":
                    nv = normalize_numeric(v)
                    if nv is not None:
                        out[k] = nv
                elif dt in ("id", "datetime"):
                    continue
                else:
                    out[k] = v
            return out
        loop_rows = [{"features": feats(r), "target": r.get(target_key)} for r in records]

    # Cumulative cross-run peek accounting (F3) is available but OPT-IN: a caller deploying against a
    # fixed locked test passes loop_kw["peek_ledger_path"] to make Bonferroni `checks` accumulate across
    # runs of the same digest. It is NOT a default here because a global persistent ledger makes otherwise
    # independent runs non-deterministic; the single-run default reports peeks=1.
    result = run_goal_loop(loop_rows, goal_text, kind=kind, task_type=task_type, target_key="target",
                           labels=labels, threshold=threshold, metric=metric, store=store,
                           providers=providers, llm_enabled=llm_enabled, seeds=seeds, text_key=text_key,
                           min_test_n=min_test_n, max_rounds=max_rounds, should_cancel=should_cancel,
                           objective=objective, time_budget_s=time_budget_s, llm_propose=llm_propose,
                           **loop_kw)
    return {"spec_inferred": spec_inferred, "result": result}
