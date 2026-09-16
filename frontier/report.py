"""Phase 9 -- research-artifact output.

Turn an `EngineResult` (the full provenance the spine already carries: history,
diagnosis_trail, certificate, winner.code, split_meta) into two artifacts:

  (a) a markdown RESEARCH REPORT -- goal, plan, what was tried, ablations,
      failures with reasons, the certificate with the sealed LOWER bound and theta,
      and EXACT reproduction instructions (seed, split counts, sealed_digest, and
      the winning build_estimator() code verbatim); and
  (b) a machine-readable JSON ARTIFACT carrying the same facts in a stable schema.

Why this matters (per the roadmap's North Star): the output of an autoresearcher
is a defensible writeup that a reviewer can REPRODUCE, not a leaderboard row. The
report is the human-facing proof that the loop "cannot fool itself": it states the
sealed certificate, the one counted peek, the sealed_digest that pins the held-out
set, and the exact code that produced the number. If the run did not clear theta,
the report says DECLINED in plain language and carries the best certificate ATTEMPT
-- never a relabeled validation score (Standing Invariant #5).

This module computes NO promotion-bearing numbers. It only READS what the frozen
certifier already produced (`EngineResult.certificate`) and re-presents it. The
single "derived" number it shows (sealed lower-bound margin over theta) is a plain
subtraction of two already-certified fields, clearly labeled as a presentation aid,
and is never used to decide anything.

# === WIRING ===
# The integrator calls this AFTER `ResearchEngine.run(task)` returns. Nothing in the
# engine changes; this is a pure consumer of EngineResult + Task.
#
#   from frontier.engine import ResearchEngine, EngineConfig
#   from frontier.report import build_report, write_report
#
#   cfg    = EngineConfig(seed=0, rounds=3)          # the seed/fracs go INTO the report
#   result = ResearchEngine(cfg).run(task)
#
#   # one call -> both artifacts as strings:
#   artifact = build_report(result, task, config=cfg, goal="<free-text goal>")
#   print(artifact.markdown)                          # the human report
#   print(artifact.json_str)                          # the machine artifact (stable schema)
#
#   # or persist both next to each other:
#   md_path, json_path = write_report(result, task, out_dir, config=cfg, goal=...)
#
# Argument shapes (all already produced by Phase 0):
#   result : EngineResult  (engine.py)  -- certified, certificate, winner, winner_val_score,
#                                          history(list[_Record]), diagnosis_trail(list[dict]),
#                                          split_meta(dict), decline_reason(str), llm_active(bool)
#   task   : Task          (task.py)    -- kind, metric, theta, name, n_features
#   config : EngineConfig  (engine.py)  -- OPTIONAL; only its reproduction-relevant fields
#                                          (seed, rounds, test_frac, val_frac, wall/cpu) are read.
#                                          If omitted, the report records "config: not supplied"
#                                          and the seed shown is taken from split_meta when present.
#
# Ordering: call build_report/write_report ONCE per run, after the engine returns. It
# performs no I/O on the sealed test and triggers no certifier peek (the certificate
# was already minted inside engine.run); it is safe to call repeatedly on the same
# result without affecting the one-peek invariant.
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, List, Optional


# --------------------------------------------------------------------------- #
# small, dependency-free helpers (no promotion numbers computed here)
# --------------------------------------------------------------------------- #
def _rec_to_dict(rec: Any) -> dict:
    """Normalize a history record (engine._Record dataclass) to a plain dict.

    We avoid importing the private _Record type; any dataclass or already-dict
    record is accepted so the report stays decoupled from engine internals.
    """
    if isinstance(rec, dict):
        return dict(rec)
    if is_dataclass(rec):
        return asdict(rec)
    # last resort: pull known attributes
    keys = ("program_id", "label", "source", "ok", "val_score",
            "error_kind", "error", "wall_seconds")
    return {k: getattr(rec, k, None) for k in keys}


def _fmt_num(x: Optional[float], nd: int = 4) -> str:
    """Format an optional float for the markdown table; '-' for None."""
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _md_escape(s: str) -> str:
    """Escape pipe characters so free-text never breaks a markdown table cell."""
    return str(s).replace("|", "\\|").replace("\n", " ")


# --------------------------------------------------------------------------- #
# the artifact bundle
# --------------------------------------------------------------------------- #
@dataclass
class ReportArtifact:
    """Both renderings of one run, plus the structured payload they share.

    `markdown`  : human-readable research report (str)
    `json_str`  : machine-readable artifact, pretty-printed JSON (str)
    `payload`   : the dict that `json_str` serializes (handy for programmatic asserts)
    """
    markdown: str
    json_str: str
    payload: dict


# --------------------------------------------------------------------------- #
# core: assemble the structured payload from an EngineResult
# --------------------------------------------------------------------------- #
def _collect_payload(result: Any, task: Any, *, config: Any, goal: str) -> dict:
    """Read everything the report needs off the (already-finished) EngineResult.

    No certifier call, no sealed peek: `result.certificate` was minted by the engine.
    """
    history = [_rec_to_dict(r) for r in getattr(result, "history", []) or []]
    ok_records = [r for r in history if r.get("ok")]
    failed_records = [r for r in history if not r.get("ok")]

    cert = getattr(result, "certificate", None)
    declined = not bool(getattr(result, "certified", False))

    winner = getattr(result, "winner", None)
    winner_code = getattr(winner, "code", None) if winner is not None else None
    winner_label = getattr(winner, "label", None) if winner is not None else None
    winner_source = getattr(winner, "source", None) if winner is not None else None
    winner_id = getattr(winner, "id", None) if winner is not None else None
    winner_prov = getattr(winner, "provenance", {}) if winner is not None else {}

    split_meta = dict(getattr(result, "split_meta", {}) or {})

    # reproduction knobs: prefer the supplied EngineConfig, fall back to split_meta.
    repro = _reproduction_block(config, split_meta, cert)

    # ablations = the per-candidate val scores that LOST to the winner. These are the
    # natural factor-variation table: each row varied the model/feature recipe while the
    # split, seed, and certifier were held fixed.
    ablations = _ablation_rows(ok_records, winner_label)

    failures = _failure_rows(failed_records)

    # presentation-only margin: sealed lower bound minus theta. Labeled as derived.
    margin = None
    if cert is not None and cert.get("lower_bound") is not None and cert.get("theta") is not None:
        try:
            margin = round(float(cert["lower_bound"]) - float(cert["theta"]), 6)
        except (TypeError, ValueError):
            margin = None

    payload = {
        "schema": "frontier.report/v1",
        "goal": goal or f"Certify a model for task '{getattr(task, 'name', 'task')}' "
                        f"at theta={getattr(task, 'theta', None)}.",
        "task": {
            "name": getattr(task, "name", "task"),
            "kind": getattr(task, "kind", None),
            "metric": getattr(task, "metric", None),
            "theta": getattr(task, "theta", None),
            "n_features": getattr(task, "n_features", None),
        },
        "outcome": {
            "status": "DECLINED" if declined else "CERTIFIED",
            "certified": not declined,
            "decline_reason": getattr(result, "decline_reason", "") or "",
            "winner_val_score": getattr(result, "winner_val_score", None),
            "llm_active": bool(getattr(result, "llm_active", False)),
        },
        "winner": {
            "label": winner_label,
            "source": winner_source,
            "id": winner_id,
            "provenance": winner_prov,
            "build_estimator_code": winner_code,
        } if winner is not None else None,
        "certificate": cert,                    # verbatim from the frozen certifier
        "sealed_lower_bound_margin_over_theta": margin,   # DERIVED display aid only
        "candidates_tried": len(history),
        "candidates_ok": len(ok_records),
        "ablations": ablations,
        "failures": failures,
        "diagnosis_trail": list(getattr(result, "diagnosis_trail", []) or []),
        "split_meta": split_meta,
        "reproduction": repro,
    }
    return payload


def _reproduction_block(config: Any, split_meta: dict, cert: Optional[dict]) -> dict:
    """Everything a clean checkout needs to reproduce the certified number.

    Seed + split fractions + split counts + sealed_digest fully pin the data; the
    winning code (carried separately in the winner block) pins the model. We record
    them here in one place so the markdown's "Reproduction" section is a single read.
    """
    block: dict = {}
    if config is not None:
        block["config_supplied"] = True
        for fld in ("seed", "rounds", "test_frac", "val_frac", "wall_seconds", "cpu_seconds"):
            if hasattr(config, fld):
                block[fld] = getattr(config, fld)
        # never leak a live LLM client object into an artifact; record only its presence
        block["llm_client_supplied"] = getattr(config, "llm_client", None) is not None
    else:
        block["config_supplied"] = False
        # fall back to whatever the split recorded
        if "seed" in split_meta:
            block["seed"] = split_meta["seed"]

    block["split_counts"] = split_meta.get("counts")
    block["split_protocol"] = split_meta.get("split")
    block["leakage_dropped"] = split_meta.get("leakage_dropped")
    block["sealed_digest"] = (cert or {}).get("sealed_digest")
    block["interpreter"] = "/Users/abdullahalghamdi/jax-env-311/bin/python"
    return block


def _ablation_rows(ok_records: List[dict], winner_label: Optional[str]) -> List[dict]:
    """One row per successfully-run candidate: label, source, val score, is_winner.

    Sorted by val score descending so the winner sits on top and the gap to each
    alternative is visible -- the ablation question "did this factor help?" answered
    by held-fixed-split comparison.
    """
    rows = []
    for r in ok_records:
        rows.append({
            "label": r.get("label"),
            "source": r.get("source"),
            "val_score": r.get("val_score"),
            "wall_seconds": r.get("wall_seconds"),
            "is_winner": (r.get("label") == winner_label),
        })
    rows.sort(key=lambda d: (d["val_score"] is not None, d["val_score"]), reverse=True)
    return rows


def _failure_rows(failed_records: List[dict]) -> List[dict]:
    """One row per candidate that did NOT run: label, error_kind, reason.

    Failures are first-class (a rigorous failure is a result). We surface the typed
    error_kind so a reader sees WHY each attempt died (timeout/import/fit/...).
    """
    rows = []
    for r in failed_records:
        rows.append({
            "label": r.get("label"),
            "source": r.get("source"),
            "error_kind": r.get("error_kind") or "other",
            "reason": (r.get("error") or "").strip(),
        })
    return rows


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def _render_markdown(p: dict) -> str:
    """Render the structured payload to a defensible markdown report."""
    L: List[str] = []
    task = p["task"]
    out = p["outcome"]
    cert = p.get("certificate")
    declined = not out["certified"]

    # ---- header + outcome banner (honest, up top) ----
    L.append(f"# Research report: {task['name']}")
    L.append("")
    banner = "DECLINED" if declined else "CERTIFIED"
    L.append(f"**Outcome: {banner}.**  "
             f"Task kind `{task['kind']}`, metric `{task['metric']}`, "
             f"promotion threshold theta = `{task['theta']}`.")
    if declined:
        reason = out.get("decline_reason") or "the sealed lower bound did not clear theta"
        L.append("")
        L.append(f"> This run is an HONEST DECLINE: {reason}. "
                 f"The number below is the best certificate ATTEMPT on the held-out sealed "
                 f"test, not a passing result and not a relabeled validation score.")
    L.append("")

    # ---- goal ----
    L.append("## Goal")
    L.append("")
    L.append(p["goal"])
    L.append("")

    # ---- plan ----
    L.append("## Plan")
    L.append("")
    L.append(textwrap.dedent(f"""\
        Propose candidate models as executable `build_estimator()` programs (seed catalog,
        offline mutations of the champion, and LLM-authored code when a client is wired),
        run each in an out-of-process sandbox on the TRAIN split and score its predictions
        on the VALIDATION split, select the single best by `{task['metric']}`, then certify
        that one winner ONCE on a held-out SEALED test via the frozen Clopper-Pearson /
        bootstrap certifier. The sealed test is read exactly one time, for the winner.
        LLM path active during this run: {out['llm_active']}.""").rstrip())
    L.append("")

    # ---- what was tried ----
    L.append("## What was tried")
    L.append("")
    L.append(f"{p['candidates_tried']} candidate program(s) proposed; "
             f"{p['candidates_ok']} executed successfully.")
    L.append("")
    if p["ablations"]:
        L.append("### Ablations (held-fixed split; each row varies the model/feature recipe)")
        L.append("")
        L.append("| candidate | source | val score | wall (s) | winner |")
        L.append("|---|---|---|---|---|")
        for a in p["ablations"]:
            mark = "yes" if a["is_winner"] else ""
            L.append(f"| {_md_escape(a['label'])} | {_md_escape(a['source'])} | "
                     f"{_fmt_num(a['val_score'])} | {_fmt_num(a['wall_seconds'], 2)} | {mark} |")
        L.append("")

    # ---- failures ----
    L.append("## Failures and why")
    L.append("")
    if p["failures"]:
        L.append("| candidate | source | error_kind | reason |")
        L.append("|---|---|---|---|")
        for f in p["failures"]:
            reason = _md_escape(f["reason"])[:200] or "(no message)"
            L.append(f"| {_md_escape(f['label'])} | {_md_escape(f['source'])} | "
                     f"`{_md_escape(f['error_kind'])}` | {reason} |")
    else:
        L.append("None: every proposed candidate executed.")
    L.append("")

    # ---- the certificate ----
    L.append("## Certificate (held-out sealed test)")
    L.append("")
    if cert is None:
        L.append("No certificate was minted: " +
                 (out.get("decline_reason") or "no candidate executed successfully") + ".")
    else:
        L.append("The promotion-bearing number is the **sealed LOWER bound** vs theta. "
                 "It is computed by the frozen certifier on a sealed test touched exactly "
                 f"once (peeks = {cert.get('peeks')}).")
        L.append("")
        L.append("| field | value |")
        L.append("|---|---|")
        order = ["observed", "lower_bound", "theta", "n", "k", "checks", "peeks",
                 "alpha_per_check", "p_value", "metric", "certified",
                 "sealed_digest", "reason"]
        shown = set()
        for key in order:
            if key in cert:
                L.append(f"| {key} | {_md_escape(cert[key])} |")
                shown.add(key)
        for key, val in cert.items():       # any extra keys the certifier added
            if key not in shown:
                L.append(f"| {key} | {_md_escape(val)} |")
        L.append("")
        margin = p.get("sealed_lower_bound_margin_over_theta")
        if margin is not None:
            sign = "above" if margin > 0 else ("below" if margin < 0 else "at")
            L.append(f"_Derived display aid (not a decision): sealed lower bound is "
                     f"`{margin}` {sign} theta._")
            L.append("")

    # ---- reproduction ----
    L.append("## Reproduction")
    L.append("")
    repro = p["reproduction"]
    L.append("Reproducing the number above requires: the seed + split fractions (which pin "
             "the data partition), the split counts and `sealed_digest` (which pin the exact "
             "held-out set), and the winning `build_estimator()` code (which pins the model).")
    L.append("")
    L.append("| knob | value |")
    L.append("|---|---|")
    for key in ("config_supplied", "seed", "rounds", "test_frac", "val_frac",
                "wall_seconds", "cpu_seconds", "llm_client_supplied",
                "split_protocol", "leakage_dropped", "sealed_digest", "interpreter"):
        if key in repro and repro[key] is not None:
            L.append(f"| {key} | {_md_escape(repro[key])} |")
    if repro.get("split_counts"):
        L.append(f"| split_counts | {_md_escape(json.dumps(repro['split_counts']))} |")
    L.append("")

    L.append("### Exact steps")
    L.append("")
    seed = repro.get("seed", 0)
    rounds = repro.get("rounds", 3)
    test_frac = repro.get("test_frac", 0.30)
    val_frac = repro.get("val_frac", 0.20)
    L.append("```bash")
    L.append("# from a clean checkout, repo root on sys.path:")
    L.append(f"{repro.get('interpreter', 'python')} - <<'PY'")
    L.append("import sys, os; sys.path.insert(0, os.getcwd())")
    L.append("from frontier.engine import ResearchEngine, EngineConfig")
    L.append("from frontier.task import Task")
    L.append("# rebuild the SAME Task (same X, y, kind, metric, theta) used for this report,")
    L.append(f"# then run with the recorded knobs (LLM client off => deterministic seeds+mutations):")
    L.append(f"cfg = EngineConfig(seed={seed}, rounds={rounds}, "
             f"test_frac={test_frac}, val_frac={val_frac})")
    L.append("res = ResearchEngine(cfg).run(task)   # 'task' = the rebuilt Task above")
    L.append("print(res.summary())")
    L.append("PY")
    L.append("```")
    L.append("")
    L.append("The certified number is reproduced when the printed `sealed_digest` matches "
             f"`{repro.get('sealed_digest')}` and the sealed lower bound vs theta matches the "
             "certificate above. (With the LLM client off, seed + mutation proposals are "
             "deterministic, so the winner and its sealed certificate reproduce exactly.)")
    L.append("")

    # ---- the winning code, verbatim ----
    L.append("## Winning build_estimator() (verbatim)")
    L.append("")
    win = p.get("winner")
    if win is None or not win.get("build_estimator_code"):
        L.append("No winning code is available (no candidate executed successfully).")
    else:
        L.append(f"Label `{win['label']}`, source `{win['source']}`, id `{win['id']}`.")
        if win.get("provenance"):
            L.append("")
            L.append(f"Provenance: `{_md_escape(json.dumps(win['provenance']))}`")
        L.append("")
        L.append("```python")
        L.append(win["build_estimator_code"].rstrip("\n"))
        L.append("```")
    L.append("")

    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_report(result: Any, task: Any, *, config: Any = None,
                 goal: str = "") -> ReportArtifact:
    """Build both the markdown report and the JSON artifact from a finished run.

    Parameters
    ----------
    result : EngineResult   the object returned by `ResearchEngine.run(task)`.
    task   : Task           the task that was run (for goal/metric/theta context).
    config : EngineConfig   optional; its reproduction-relevant fields are recorded.
    goal   : str            optional free-text goal; a sensible default is synthesized.

    Returns
    -------
    ReportArtifact with `.markdown`, `.json_str`, and `.payload`.

    This function performs NO certifier call and NO sealed peek -- it only reads the
    certificate the engine already minted, so it is safe to call any number of times.
    """
    payload = _collect_payload(result, task, config=config, goal=goal)
    md = _render_markdown(payload)
    json_str = json.dumps(payload, indent=2, sort_keys=False, default=str)
    return ReportArtifact(markdown=md, json_str=json_str, payload=payload)


def write_report(result: Any, task: Any, out_dir: str, *, config: Any = None,
                 goal: str = "", basename: Optional[str] = None) -> tuple:
    """Build and persist both artifacts under `out_dir`.

    Returns (markdown_path, json_path). The basename defaults to the task name
    (sanitized). Creates `out_dir` if needed. Pure file I/O; no certifier interaction.
    """
    artifact = build_report(result, task, config=config, goal=goal)
    os.makedirs(out_dir, exist_ok=True)
    name = basename or "".join(
        ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(getattr(task, "name", "task"))
    ) or "report"
    md_path = os.path.join(out_dir, f"{name}.report.md")
    json_path = os.path.join(out_dir, f"{name}.report.json")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(artifact.markdown)
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(artifact.json_str)
    return md_path, json_path
