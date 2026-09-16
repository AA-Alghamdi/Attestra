"""SUITE-LEVEL generality, not cherry-picked tasks: run the menu-free regenerative loop across a STANDARD
slice of OpenML-CC18 (study 99) and emit ONE certified summary.

WHY THIS EXISTS
---------------
Every prior code-discovery result was a single hand-chosen dataset. "Use benchmark suites, not cherry-picked
tasks" means the generality claim must be SUITE-LEVEL: the SAME loop (RAW features pinned -> the only lever is
code the system AUTHORS or the gbm head it can reach -> frozen Clopper-Pearson + paired McNemar + BH-FDR ->
never-peeked gold), the SAME frozen certifier (science.py b564fba2 / sealed.py 30ad6245), run across a diverse
slice of a published suite, reporting BOTH the certified wins AND the honest negatives.

The slice deliberately spans the task-shape matrix:
  multiclass (balanced)   : vehicle (4) / segment (7) / mfeat_fourier (10) / optdigits (10) / splice (3)
  near-linearly-separable : optdigits  -- expected HONEST NEGATIVE (a linear head is already near-optimal)
  categorical features     : splice (DNA bases) / credit_g  -- exercise the one-hot encode path
  imbalanced binary        : pima (maj~0.65) / credit_g (maj~0.70) -- a DATA-DRIVEN competence floor
                             (theta = majority-baseline + margin) makes "competent" mean "beats trivial",
                             so raw accuracy is not gameable and the meta-certifier admits the framing.

Each dataset's framing is stress-tested by the meta-certifier BEFORE any sealed peek; a gameable framing
REFUSES (reported honestly, not hidden). Nothing here is cherry-picked: the suite + the per-dataset verdict
(certified lift / honest negative / refused) are all written to docs/CC18_SUITE.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import asdict

from scripts.run_code_discovery import build

# (dataset, shape) -- balanced multiclass uses 'multiclass'; imbalanced binary preserves natural proportions
# via 'imbalanced'. theta_floor='auto' for every run (= max(0.5, majority+margin)): 0.5 on the balanced sets,
# raised to just above the trivial baseline on the imbalanced binary ones.
SUITE = [
    ("vehicle",       "multiclass"),
    ("segment",       "multiclass"),
    ("mfeat_fourier", "multiclass"),
    ("optdigits",     "multiclass"),
    ("splice",        "multiclass"),
    ("pima",          "imbalanced"),
    ("credit_g",      "imbalanced"),
]


def run_one(dataset: str, shape: str, *, use_llm: bool, seed: int, peeks: int):
    roles = ["featurizer", "classifier"]
    arena, gen, researcher, authorer = build(dataset, shape, use_llm=use_llm, seed=seed, peeks=peeks,
                                              roles=roles, split="random", theta_floor="auto")
    t0 = time.time()
    cert = researcher.run()
    dt = time.time() - t0
    best_lift = max((p.mean_lift for p in cert.promotions), default=0.0)
    g = cert.gold_confirmation or {}
    row = {
        "dataset": dataset, "shape": shape, "n_classes": arena.n_classes,
        "n_train": int(len(arena.splits.train_idx)), "theta_floor": round(float(arena.theta_floor), 3),
        "refused": bool(cert.refused), "stop_reason": cert.stop_reason,
        "champion": cert.champion, "peeks_used": cert.peeks_used,
        "menu_free": bool(cert.novelty.get("menu_free")),
        "uses_authored_code": bool(cert.novelty.get("uses_authored_code")),
        "n_promotions": len(cert.promotions), "best_certified_lift": round(float(best_lift), 4),
        "gold_survivors": len(g.get("survivors", [])), "gold_n_tasks": g.get("n_tasks", 0),
        "gold_n": g.get("gold_n", 0), "gold_confirmed": bool(g.get("confirmed", False)),
        "framing_trustworthy": (cert.framing_report or {}).get("trustworthy"),
        "seconds": round(dt, 1),
    }
    return row, cert


def _verdict(r: dict) -> str:
    if r["refused"]:
        return "REFUSED (gameable framing / contamination)"
    if r["n_promotions"] > 0 and r["gold_confirmed"]:
        kind = "authored-code" if r["uses_authored_code"] else "gbm-head"
        return f"CERTIFIED +{r['best_certified_lift']:.3f} ({kind}), gold-confirmed {r['gold_survivors']}/{r['gold_n_tasks']}"
    if r["n_promotions"] > 0:
        return f"certified +{r['best_certified_lift']:.3f} but gold {r['gold_survivors']}/{r['gold_n_tasks']} (not confirmed)"
    return "honest negative (linear seed not beaten under FDR)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="use the open-ended Claude authorer (else template)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peeks", type=int, default=14)
    ap.add_argument("--only", default=None, help="comma-separated dataset subset (default: full suite)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    suite = SUITE
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        suite = [(d, s) for (d, s) in SUITE if d in keep]

    rows = []
    for dataset, shape in suite:
        print(f"\n{'='*78}\n== CC18 suite: {dataset} ({shape}) ==\n{'='*78}", flush=True)
        try:
            row, cert = run_one(dataset, shape, use_llm=args.llm, seed=args.seed, peeks=args.peeks)
        except Exception as exc:  # a single dataset failure must not sink the suite
            print(f"  !! {dataset} FAILED: {exc!r}")
            rows.append({"dataset": dataset, "shape": shape, "error": repr(exc)})
            continue
        row["verdict"] = _verdict(row)
        rows.append(row)
        print(f"  -> {row['verdict']}  [theta={row['theta_floor']} peeks={row['peeks_used']} "
              f"champ={row['champion']} {row['seconds']}s]", flush=True)

    n_cert = sum(1 for r in rows if r.get("n_promotions", 0) > 0 and not r.get("refused"))
    n_conf = sum(1 for r in rows if r.get("gold_confirmed"))
    n_ref = sum(1 for r in rows if r.get("refused"))
    n_neg = sum(1 for r in rows if (not r.get("refused")) and r.get("n_promotions", 0) == 0 and "error" not in r)

    print(f"\n{'='*78}\n== CC18 SUITE SUMMARY ({len(rows)} datasets) ==\n{'='*78}")
    print(f"{'dataset':<14}{'shape':<12}{'cls':>4}{'theta':>7}  verdict")
    for r in rows:
        if "error" in r:
            print(f"{r['dataset']:<14}{r['shape']:<12}{'?':>4}{'?':>7}  ERROR {r['error'][:50]}")
            continue
        print(f"{r['dataset']:<14}{r['shape']:<12}{r['n_classes']:>4}{r['theta_floor']:>7.2f}  {r['verdict']}")
    print(f"\ncertified: {n_cert}   gold-confirmed: {n_conf}   honest-negative: {n_neg}   refused: {n_ref}")

    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "docs", "CC18_SUITE.json")
    payload = {"suite": "OpenML-CC18 (study 99) slice", "authorer": ("LLM" if args.llm else "template"),
               "seed": args.seed, "peeks": args.peeks,
               "totals": {"datasets": len(rows), "certified": n_cert, "gold_confirmed": n_conf,
                          "honest_negative": n_neg, "refused": n_ref},
               "rows": rows}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote {out}")
    return rows


if __name__ == "__main__":
    main()
