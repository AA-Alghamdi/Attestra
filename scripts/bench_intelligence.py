"""Multi-dataset benchmark: prove Phase-B/C intelligence proposals actually fire.

Instruments frontier.intelligence.get_intelligence_proposals to record, per round
and per dataset, how many feature-eng / ensemble / ASHA-HP proposals were generated
(and that they entered the pool), plus the final certify/decline + oracle verdict.

LLM-free run (deterministic, hermetic) so the numbers are reproducible offline.
"""
import json
import time
from collections import defaultdict

import numpy as np
from sklearn import datasets as skd

import frontier.intelligence as intel_mod
from frontier.core.orchestrator import CoreOrchestrator, CoreConfig

# ---- instrument intelligence proposals (record labels by kind, per round) ----
_REC = {"rounds": []}
_orig = intel_mod.get_intelligence_proposals


def _kind(label: str) -> str:
    if label.startswith("feat_"):
        return "feature_eng"
    if label.startswith("ensemble_"):
        return "ensemble"
    if label.startswith("asha_"):
        return "asha_hp"
    return "other"


def _spy(state, context, task):
    out = _orig(state, context, task)
    by = defaultdict(list)
    for p in out:
        by[_kind(p.label)].append(p.label)
    _REC["rounds"].append({k: v for k, v in by.items()})
    return out


intel_mod.get_intelligence_proposals = _spy


def _load(name):
    if name == "breast_cancer":
        d = skd.load_breast_cancer(); return d.data, d.target, 0.90, "classify tumors as malignant/benign"
    if name == "wine":
        d = skd.load_wine(); return d.data, d.target, 0.80, "classify wine cultivar from chemistry"
    if name == "digits":
        d = skd.load_digits(); return d.data, d.target, 0.85, "classify handwritten digits"
    if name == "iris":
        d = skd.load_iris(); return d.data, d.target, 0.85, "classify iris species"
    if name == "synthetic_hard":
        X, y = skd.make_classification(n_samples=3000, n_features=50, n_informative=12,
                                       n_redundant=10, n_classes=4, class_sep=0.8,
                                       flip_y=0.05, random_state=0)
        return X, y, 0.55, "classify a hard synthetic 4-class tabular problem"
    raise ValueError(name)


def run_one(name):
    X, y, theta, goal = _load(name)
    _REC["rounds"] = []
    events = []
    cfg = CoreConfig(rounds=4, seed=0, total_seconds=900.0, wall_seconds=45.0,
                     cpu_seconds=40, llm_client=None,  # hermetic, no LLM
                     on_event=lambda e: events.append((e.get("stage"), e)))
    t0 = time.time()
    orch = CoreOrchestrator(cfg)
    res = orch.run(goal=goal, X=X, y=y, theta=theta, name=name)
    dt = time.time() - t0
    rounds_intel = _REC["rounds"]
    totals = defaultdict(int)
    for rd in rounds_intel:
        for k, labs in rd.items():
            totals[k] += len(labs)
    ov = res.oracle_verdict or {}
    return {
        "dataset": name, "n": int(X.shape[0]), "features": int(X.shape[1]),
        "theta": theta, "certified": bool(res.certified),
        "winner_val": (round(res.winner_val_score, 4) if res.winner_val_score is not None else None),
        "decline_reason": res.decline_reason or "",
        "oracle_promote": ov.get("promote"),
        "intel_rounds": [{k: len(v) for k, v in rd.items()} for rd in rounds_intel],
        "intel_totals": dict(totals),
        "wall_seconds": round(dt, 1),
    }


def main():
    names = ["iris", "wine", "breast_cancer", "digits", "synthetic_hard"]
    results = []
    for nm in names:
        print(f"\n===== {nm} =====", flush=True)
        try:
            r = run_one(nm)
        except Exception as e:
            r = {"dataset": nm, "error": repr(e)}
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)
    with open("scripts/bench_intelligence_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWROTE scripts/bench_intelligence_results.json", flush=True)


if __name__ == "__main__":
    main()
