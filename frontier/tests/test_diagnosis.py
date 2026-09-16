"""Behavioral tests for the Phase-2 diagnosis-feed-forward module.

Asserts that diagnosis is computed correctly from a synthetic history AND that its directives
actually CHANGE which proposals fire (the acceptance criterion: "injected failure modes change
the next round's proposals measurably"). Run standalone:

    cd <repo root> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_diagnosis.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier.diagnosis import (
    diagnose, enrich_context, DiagnosisDrivenProposer,
    _parse_recipe_from_label,
)
from frontier.proposers import SeedProposer, MutationProposer
from frontier.task import Task


# --------------------------------------------------------------------------- helpers

def _reg_task():
    """A tiny real-ish regression task (diabetes) just to supply kind/metric to diagnose."""
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    return Task(X=d.data, y=d.target.astype(float), kind="regression", theta=0.40, name="diab")


def _rec(label, source, ok, val_score=None, error_kind="", error=""):
    """A plain-dict stand-in for engine._Record (diagnose accepts dicts via _RecordView)."""
    return {"label": label, "source": source, "ok": ok, "val_score": val_score,
            "error_kind": error_kind, "error": error}


# --------------------------------------------------------------------------- tests

def test_label_parse_roundtrip():
    r = _parse_recipe_from_label("tlog+poly2+scale+ridge")
    assert r["base"] == "ridge"
    assert r.get("target_log") is True and r.get("scale") is True and r.get("poly") == 2
    assert _parse_recipe_from_label("hist_gbm") == {"base": "hist_gbm"}
    assert _parse_recipe_from_label("llm0_1")["base"] == "llm0_1"
    print("[ok] label->recipe parse")


def test_dominant_error_kind():
    hist = [
        _rec("a+rf", "seed", ok=False, error_kind="timeout"),
        _rec("b+svc_rbf", "seed", ok=False, error_kind="timeout"),
        _rec("c+logreg", "seed", ok=False, error_kind="import"),
        _rec("d+ridge", "seed", ok=True, val_score=0.5),
    ]
    diag = diagnose(hist, [], _reg_task())
    assert diag.dominant_error_kind == "timeout", diag.dominant_error_kind
    assert abs(diag.dominant_error_share - 2 / 3) < 1e-9
    assert "TIMED OUT" in diag.directives["llm_guidance"]
    print(f"[ok] dominant error: {diag.dominant_error_kind} share={diag.dominant_error_share:.2f}")


def test_plateau_detection_and_seed_drop():
    # three rounds, best-on-val flat -> plateau -> seeds should be dropped from fire_sources
    trail = [
        {"round": 0, "best_out_score": 0.50},
        {"round": 1, "best_out_score": 0.50},
        {"round": 2, "best_out_score": 0.50},
    ]
    hist = [_rec("scale+ridge", "seed", ok=True, val_score=0.50)]
    diag = diagnose(hist, trail, _reg_task())
    assert diag.plateau is True and diag.plateau_span >= 2, (diag.plateau, diag.plateau_span)
    assert "seed" not in diag.directives["fire_sources"], diag.directives["fire_sources"]
    assert "mutation" in diag.directives["fire_sources"]
    print(f"[ok] plateau span={diag.plateau_span} fire={diag.directives['fire_sources']}")

    # contrast: an improving trail is NOT a plateau and keeps seeds
    trail_up = [
        {"round": 0, "best_out_score": 0.40},
        {"round": 1, "best_out_score": 0.55},
        {"round": 2, "best_out_score": 0.62},
    ]
    diag_up = diagnose(hist, trail_up, _reg_task())
    assert diag_up.plateau is False
    assert "seed" in diag_up.directives["fire_sources"]
    print("[ok] improving trail keeps seeds firing")


def test_family_rank_and_avoid_failed_only_bases():
    hist = [
        _rec("hist_gbm", "seed", ok=True, val_score=0.62),
        _rec("scale+ridge", "seed", ok=True, val_score=0.48),
        _rec("rf", "seed", ok=False, error_kind="timeout"),   # rf only ever failed
    ]
    diag = diagnose(hist, [], _reg_task())
    fams = [f.family for f in diag.family_rank]
    assert fams[0] == "hist_gbm", fams                         # best ok family ranks first
    assert "hist_gbm" in diag.directives["prefer_bases"]
    assert "rf" in diag.directives["avoid_bases"]             # failed-only base avoided
    print(f"[ok] family rank={fams} prefer={diag.directives['prefer_bases']} "
          f"avoid={diag.directives['avoid_bases']}")


def test_axis_lift_exploit_and_avoid():
    # 'scale' helps (with-scale runs higher), 'poly' hurts (with-poly runs lower)
    hist = [
        _rec("scale+ridge", "seed", ok=True, val_score=0.60),
        _rec("scale+lasso", "seed", ok=True, val_score=0.58),
        _rec("ridge", "seed", ok=True, val_score=0.40),
        _rec("lasso", "seed", ok=True, val_score=0.38),
        _rec("poly2+ridge", "seed", ok=True, val_score=0.30),
    ]
    diag = diagnose(hist, [], _reg_task())
    assert diag.axis_lift["scale"] is not None and diag.axis_lift["scale"] > 0
    assert "scale" in diag.directives["exploit_axes"], diag.directives["exploit_axes"]
    assert diag.axis_lift["poly"] is not None and diag.axis_lift["poly"] < 0
    assert "poly" in diag.directives["avoid_axes"], diag.directives["avoid_axes"]
    print(f"[ok] axis lift scale={diag.axis_lift['scale']:+.3f} poly={diag.axis_lift['poly']:+.3f}")


def test_residual_summary_regression():
    task = _reg_task()
    rng = np.random.default_rng(0)
    yt = rng.uniform(10, 300, size=200)
    # heteroscedastic + skewed residuals: error scales with magnitude
    resid = (yt / yt.mean()) * np.abs(rng.normal(0, 20, size=200))
    yp = yt - resid
    diag = diagnose([_rec("hist_gbm", "seed", ok=True, val_score=0.5)], [], task,
                    val_truth=yt, val_preds=yp)
    assert "skew=" in diag.residual_summary and "corr=" in diag.residual_summary
    # honest degradation when no residuals supplied
    diag_none = diagnose([_rec("hist_gbm", "seed", ok=True, val_score=0.5)], [], task)
    assert diag_none.residual_summary == ""
    print(f"[ok] residual summary: '{diag.residual_summary}' (empty without residuals)")


def test_enrich_context_is_additive():
    task = _reg_task()
    diag = diagnose([_rec("scale+ridge", "seed", ok=True, val_score=0.5)], [], task)
    ctx = {"task_kind": "regression", "tried_labels": set(), "best_recipe": None, "round": 1}
    out = enrich_context(ctx, diag)
    assert out is ctx                                    # in place
    assert "diagnosis" in ctx and "diag_guidance" in ctx and "diag_summary" in ctx
    # Phase-0 keys untouched
    assert ctx["task_kind"] == "regression" and "best_recipe" in ctx
    print(f"[ok] enrich additive: {ctx['diag_summary']}")


def test_directives_change_which_proposals_fire():
    """ACCEPTANCE: the same wrapped proposers produce DIFFERENT proposals under different
    diagnoses -- a plateau gates the seed source off; an avoid-base directive thins mutation."""
    task = _reg_task()

    seed_p = DiagnosisDrivenProposer(SeedProposer())
    mut_p = DiagnosisDrivenProposer(MutationProposer())

    base_ctx = {
        "task_kind": "regression", "n_features": task.n_features, "n_train": 200, "round": 1,
        "tried_labels": set(),
        "best_recipe": {"base": "ridge", "scale": True}, "best_id": "x",
    }

    # (1) no diagnosis -> backward-compatible: seed fires its full library
    ctx0 = dict(base_ctx)
    seeds_baseline = seed_p.propose(ctx0)
    assert len(seeds_baseline) >= 3, "seed source should fire its library with no diagnosis"

    # (2) plateau diagnosis -> seed source gated OFF, mutation still fires
    plateau_trail = [
        {"round": 0, "best_out_score": 0.5}, {"round": 1, "best_out_score": 0.5},
        {"round": 2, "best_out_score": 0.5},
    ]
    diag_plateau = diagnose([_rec("scale+ridge", "seed", ok=True, val_score=0.5)],
                            plateau_trail, task)
    ctx_plateau = enrich_context(dict(base_ctx), diag_plateau)
    seeds_plateau = seed_p.propose(ctx_plateau)
    muts_plateau = mut_p.propose(ctx_plateau)
    assert seeds_plateau == [], "plateau must gate the seed source off"
    assert len(muts_plateau) > 0, "mutation must still fire on plateau"
    assert seeds_plateau != seeds_baseline, "directives measurably changed seed proposals"

    # (3) avoid-base directive -> mutation output is thinned (no proposal on the avoided base)
    #     Build a diagnosis where 'hist_gbm' only ever failed, so it is avoided; mutation would
    #     otherwise swap-base into hist_gbm.
    hist_fail = [
        _rec("scale+ridge", "seed", ok=True, val_score=0.5),
        _rec("hist_gbm", "seed", ok=False, error_kind="timeout"),
        _rec("hist_gbm", "mutation", ok=False, error_kind="timeout"),
    ]
    # not a plateau (only 1 scored round) so seeds still allowed; isolate the avoid-base effect
    diag_avoid = diagnose(hist_fail, [{"round": 0, "best_out_score": 0.5}], task)
    assert "hist_gbm" in diag_avoid.directives["avoid_bases"]
    ctx_avoid = enrich_context(dict(base_ctx), diag_avoid)

    muts_unfiltered = MutationProposer().propose(dict(base_ctx))   # raw, no gating
    muts_filtered = mut_p.propose(ctx_avoid)                       # gated
    raw_bases = {_parse_recipe_from_label(m.label)["base"] for m in muts_unfiltered}
    filt_bases = {_parse_recipe_from_label(m.label)["base"] for m in muts_filtered}
    assert "hist_gbm" in raw_bases, "raw mutation would propose hist_gbm (precondition)"
    assert "hist_gbm" not in filt_bases, "avoid-base directive must thin out hist_gbm proposals"
    assert len(muts_filtered) < len(muts_unfiltered), "gating measurably reduced proposals"

    print(f"[ok] directives change proposals: seeds {len(seeds_baseline)}->{len(seeds_plateau)} "
          f"(plateau), mutation bases {sorted(raw_bases)}->{sorted(filt_bases)} (avoid hist_gbm)")


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
    print(f"\n{len(fns) - failed}/{len(fns)} diagnosis tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
