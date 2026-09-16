"""Phase 9 report tests.

Run the REAL Phase-0 engine on a real sklearn dataset and assert the generated
research artifact contains the certificate, the winning code, and a reproduction
block -- and that a DECLINED run is reported honestly as declined.

    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_report.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier.engine import ResearchEngine, EngineConfig
from frontier.report import build_report, write_report
from frontier.task import Task


def _bc_task(theta: float) -> Task:
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return Task(X=d.data, y=d.target.astype(str), kind="classification",
                theta=theta, name="breast_cancer")


def _run(theta: float):
    cfg = EngineConfig(rounds=2, wall_seconds=45, cpu_seconds=40, seed=0)
    task = _bc_task(theta)
    res = ResearchEngine(cfg).run(task)
    return res, task, cfg


def test_certified_report_contains_certificate_code_and_repro():
    """On a real dataset at an achievable theta the report must carry the sealed
    certificate, the winning build_estimator() code verbatim, and a reproduction block."""
    res, task, cfg = _run(theta=0.85)
    assert res.winner is not None and res.certificate is not None

    art = build_report(res, task, config=cfg, goal="Certify a breast-cancer classifier.")
    md = art.markdown
    payload = art.payload

    # --- certificate present in BOTH renderings, with the sealed LOWER bound + theta ---
    cert = res.certificate
    assert payload["certificate"] == cert, "JSON artifact must carry the certifier dict verbatim"
    assert "Certificate (held-out sealed test)" in md
    assert "lower_bound" in md and "theta" in md
    assert str(cert["lower_bound"]) in md, "markdown must show the sealed lower bound value"
    assert str(cert["sealed_digest"]) in md, "markdown must pin the sealed_digest"
    assert "peeks" in md and str(cert["peeks"]) in md

    # --- winning code present VERBATIM ---
    assert res.winner.code.strip() in md, "winning build_estimator() must appear verbatim"
    assert payload["winner"]["build_estimator_code"] == res.winner.code
    assert "def build_estimator" in md

    # --- reproduction block: seed, split counts, sealed_digest, exact steps ---
    assert "## Reproduction" in md
    repro = payload["reproduction"]
    assert repro["seed"] == cfg.seed
    assert repro["split_counts"] == res.split_meta.get("counts")
    assert repro["sealed_digest"] == cert["sealed_digest"]
    assert "EngineConfig(" in md and "ResearchEngine(cfg).run" in md
    assert repro["interpreter"].endswith("jax-env-311/bin/python")

    # --- honest CERTIFIED banner only when actually certified ---
    if res.certified:
        assert "Outcome: CERTIFIED" in md
        assert payload["outcome"]["status"] == "CERTIFIED"

    # --- report did not recompute the promotion number: margin is a plain subtraction ---
    margin = payload["sealed_lower_bound_margin_over_theta"]
    assert abs(margin - (cert["lower_bound"] - cert["theta"])) < 1e-9

    print(f"[ok] certified report: certified={res.certified} "
          f"lb={cert['lower_bound']} theta={cert['theta']} digest={cert['sealed_digest'][:10]}...")


def test_declined_report_is_marked_declined():
    """At an impossible theta the run honestly declines; the report must say DECLINED
    and must NOT present a passing result -- it carries the best certificate ATTEMPT."""
    res, task, cfg = _run(theta=0.999999)
    assert not res.certified, "theta this high must not certify on a finite sealed test"

    art = build_report(res, task, config=cfg)
    md = art.markdown
    payload = art.payload

    assert "Outcome: DECLINED" in md, "declined run must be banner-flagged as DECLINED"
    assert "HONEST DECLINE" in md
    assert payload["outcome"]["status"] == "DECLINED"
    assert payload["outcome"]["certified"] is False
    # the certificate attempt is still shown (its lower bound is below theta)
    if res.certificate is not None:
        assert res.certificate["certified"] is False
        assert res.certificate["lower_bound"] <= res.certificate["theta"]
        assert "lower_bound" in md
    print(f"[ok] declined report: status={payload['outcome']['status']} "
          f"reason='{payload['outcome']['decline_reason']}'")


def test_failures_and_ablations_surface():
    """Ablation rows (the candidates that lost) and any typed failures appear in the report."""
    res, task, cfg = _run(theta=0.85)
    art = build_report(res, task, config=cfg)
    payload = art.payload
    md = art.markdown

    assert payload["candidates_tried"] >= payload["candidates_ok"] >= 1
    assert len(payload["ablations"]) == payload["candidates_ok"]
    # exactly one ablation row flagged winner, and it sorts to the top
    winners = [a for a in payload["ablations"] if a["is_winner"]]
    assert len(winners) == 1
    assert payload["ablations"][0]["is_winner"], "winner must sort to the top by val score"
    assert "## What was tried" in md and "## Failures and why" in md
    # failures table is internally consistent with the typed error kinds the engine recorded
    assert len(payload["failures"]) == payload["candidates_tried"] - payload["candidates_ok"]
    print(f"[ok] tried={payload['candidates_tried']} ok={payload['candidates_ok']} "
          f"failures={len(payload['failures'])}")


def test_json_artifact_is_valid_and_roundtrips():
    """The machine-readable artifact must be valid JSON with the stable schema tag."""
    res, task, cfg = _run(theta=0.85)
    art = build_report(res, task, config=cfg)
    parsed = json.loads(art.json_str)        # must not raise
    assert parsed["schema"] == "frontier.report/v1"
    assert parsed["certificate"] == res.certificate
    assert parsed["reproduction"]["sealed_digest"] == res.certificate["sealed_digest"]
    # no live LLM client object leaked into the artifact
    assert parsed["reproduction"]["llm_client_supplied"] is False
    print("[ok] JSON artifact valid + schema-tagged + no client leak")


def test_write_report_persists_both_files():
    """write_report drops a .md and a .json next to each other."""
    res, task, cfg = _run(theta=0.85)
    with tempfile.TemporaryDirectory() as d:
        md_path, json_path = write_report(res, task, d, config=cfg)
        assert os.path.exists(md_path) and os.path.exists(json_path)
        with open(md_path) as fh:
            assert "# Research report" in fh.read()
        with open(json_path) as fh:
            assert json.load(fh)["schema"] == "frontier.report/v1"
    print("[ok] write_report persisted both artifacts")


def test_no_config_degrades_honestly():
    """With no EngineConfig supplied the report still builds and records the absence."""
    res, task, _ = _run(theta=0.85)
    art = build_report(res, task)            # config omitted
    assert art.payload["reproduction"]["config_supplied"] is False
    assert "## Reproduction" in art.markdown
    print("[ok] missing config degrades honestly")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} report tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
