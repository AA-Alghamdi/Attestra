"""Phase 4 — large-data (100K+ rows) end-to-end scale verification.

Drives the full CoreOrchestrator pipeline on a 150K-row synthetic tabular
dataset to prove the engine completes without OOM/timeout at true scale, and
records peak resident memory (self + sandbox children) and wall-clock time.

The orchestrator's own knobs carry the scale:
  - progressive subsampling (subsample_max_rows) keeps per-round training
    memory bounded while scaling rows up across rounds, and
  - eval_max_rows caps the VAL selection set (the SEALED test stays full),
    so certification can only ever get *more* conservative on big data.

LLM-free / deterministic so the result is reproducible offline.
"""
import json
import resource
import sys
import time

import numpy as np
from sklearn import datasets as skd

from frontier.core.orchestrator import CoreOrchestrator, CoreConfig


def _peak_rss_mb() -> float:
    """Peak RSS of this process + reaped children, in MiB.

    ru_maxrss is KiB on Linux. Children count matters here because every
    proposal is fitted in a sandbox subprocess.
    """
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (self_kb + child_kb) / 1024.0


def make_large(n_samples: int, seed: int = 0):
    X, y = skd.make_classification(
        n_samples=n_samples, n_features=40, n_informative=16, n_redundant=8,
        n_classes=3, class_sep=0.9, flip_y=0.03, random_state=seed)
    return X.astype(np.float32), y


def run(n_samples: int = 150_000, rounds: int = 3):
    X, y = make_large(n_samples)
    goal = "classify a large 3-class tabular dataset"
    events = []
    cfg = CoreConfig(
        rounds=rounds, seed=0, total_seconds=1800.0, wall_seconds=120.0,
        cpu_seconds=90, llm_client=None,
        on_event=lambda e: events.append((e.get("stage"), e)))

    t0 = time.time()
    orch = CoreOrchestrator(cfg)
    res = orch.run(goal=goal, X=X, y=y, theta=0.55, name="large_tabular")
    dt = time.time() - t0

    split_evt = next((e for s, e in events if s == "split"), {})
    notes = []
    for s, e in events:
        for n in e.get("backend_notes", []) or []:
            notes.append(n)
    out = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "rounds": rounds,
        "completed": True,
        "certified": bool(res.certified),
        "winner_val": (round(res.winner_val_score, 4)
                       if res.winner_val_score is not None else None),
        "decline_reason": res.decline_reason or "",
        "n_train": split_evt.get("n_train"),
        "n_val": split_evt.get("n_val"),
        "wall_seconds": round(dt, 1),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "subsample_notes": [n for n in notes if "subsampl" in n or "capped" in n],
    }
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
    print(f"===== large-data scale: {n} rows =====", flush=True)
    try:
        r = run(n_samples=n)
    except Exception as e:  # noqa: BLE001 - report the failure honestly
        r = {"n_samples": n, "completed": False, "error": repr(e),
             "peak_rss_mb": round(_peak_rss_mb(), 1)}
    print(json.dumps(r, indent=2), flush=True)
    with open("scripts/scale_largedata_results.json", "w") as f:
        json.dump(r, f, indent=2)
    print("\nWROTE scripts/scale_largedata_results.json", flush=True)


if __name__ == "__main__":
    main()
