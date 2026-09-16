"""#3 -- CERTIFIED DATA-CEILING CURVE: turn "the hardest pairs are a data ceiling" into a bounded, certified
accuracy-vs-label-budget curve, under the SAME frozen discipline as every other phase.

Across the program the hardest confusable pairs (the FGVC 737 variants in vision; the residual 20-Newsgroups
pairs in text) resisted every frozen encoder, and the brain flagged them as a "data ceiling" -- a HYPOTHESIS.
This script falsifies-or-confirms that hypothesis cleanly: for the CHAMPION representation, on those exact
pairs, it sweeps the training label budget while holding the sealed test (and the val-selection set) FIXED,
fits the same strong val-selected head at each budget, and bounds sealed accuracy with the frozen
Clopper-Pearson lower bound. It then CERTIFIES whether more labels move the metric:

  * slope test  -- paired McNemar(champion@FULL_POOL vs champion@MIN_BUDGET) on the FIXED sealed rows, with
    Benjamini-Hochberg across the hard pairs. A surviving, positive slope => the pair is DATA-LIMITED (more
    labels certifiably help); no surviving slope => FLAT within the available labels (a true ceiling that this
    much data does not cross).
  * top-step    -- McNemar(@FULL vs @2nd-largest): is the curve still rising at the top of available data, or
    has it saturated within budget? (reported, characterizing, not FDR-gated).

Discipline (identical to B1->#5): the sealed rows are NEVER used for training or head-selection (select-then-
bound); only the size of the training subsample changes; the val set and sealed set are byte-fixed across all
budgets; the bound is the frozen `science.clopper_pearson_lower` reached via `benchmark_vision_transfer._lb`.
The certified McNemar comparison uses the seed-0 canonical training subsample at each budget (pre-registered);
the plotted curve shows the mean +/- std over `--seeds` independent balanced subsamples to expose noise.

Run:  `python scripts/data_ceiling_curve.py --modality vision`
      `python scripts/data_ceiling_curve.py --modality text`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.benchmark_vision_transfer as B1                              # noqa: E402
from vfplatform.battery import benjamini_hochberg, mcnemar_pvalue           # noqa: E402

FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA = 0.1


def _frozen_hashes():
    return {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:8]
            for f in FROZEN_EXPECTED}


def _budget_grid(max_per_class: int) -> list:
    """A roughly-geometric per-class budget grid from a small floor up to the full training pool."""
    grid = [b for b in (6, 12, 24, 48, 96, 192) if b < max_per_class]
    grid.append(max_per_class)                       # always include the full pool as the top of the curve
    return sorted(set(grid))


def _subsample(tr_ids: np.ndarray, y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """A balanced training subset of `per_class` rows per class, drawn deterministically from the train POOL."""
    rng = np.random.RandomState(1000 + seed)
    chosen = []
    for c in (0, 1):
        ci = tr_ids[y[tr_ids] == c]
        rng.shuffle(ci)
        chosen += list(ci[:per_class])
    return np.array(sorted(int(i) for i in chosen))


def _fit_on_budget(emb, y, sub_ids, val_ids, test_ids):
    """Fit the strong val-selected head on `sub_ids`; return the FIXED-sealed correctness vector + accuracy."""
    est = B1._random_search_best(np.random.RandomState(0), emb[sub_ids], y[sub_ids], emb[val_ids], y[val_ids])
    test_c = B1._correct(est, emb[test_ids], y[test_ids])
    return test_c, float(np.mean(test_c))


def eval_task(emb, y, tr_ids, val_ids, test_ids, grid, seeds):
    """Sweep the per-class training budget over `grid`, holding val + sealed FIXED, and return the curve plus
    the certified slope/top-step statistics. PURE (no arena, no I/O): the sealed rows are never used for
    training or selection, only the size of the balanced training subsample of `tr_ids` changes."""
    y = np.asarray(y)
    tr_ids, val_ids, test_ids = np.asarray(tr_ids), np.asarray(val_ids), np.asarray(test_ids)
    curve, canon = [], {}
    for b in grid:
        accs, lbs = [], []
        for s in range(seeds):
            sub = _subsample(tr_ids, y, b, s)
            assert not (set(sub.tolist()) & set(test_ids.tolist())), "LEAK: train subsample hit sealed rows"
            assert not (set(sub.tolist()) & set(val_ids.tolist())), "LEAK: train subsample hit val rows"
            tc, acc = _fit_on_budget(emb, y, sub, val_ids, test_ids)
            accs.append(acc)
            lbs.append(B1._lb(tc))
            if s == 0:
                canon[b] = tc
        curve.append({"budget_per_class": b, "n_train": 2 * b,
                      "acc_mean": round(float(np.mean(accs)), 4),
                      "acc_std": round(float(np.std(accs)), 4),
                      "acc_lb_mean": round(float(np.mean(lbs)), 4),
                      "acc_lb_canonical": round(B1._lb(canon[b]), 4)})
    lo, hi = grid[0], grid[-1]
    prev = grid[-2] if len(grid) >= 2 else hi
    p_slope = mcnemar_pvalue(list(canon[hi]), list(canon[lo]))
    lift = float(np.mean(canon[hi]) - np.mean(canon[lo]))
    p_top = mcnemar_pvalue(list(canon[hi]), list(canon[prev])) if len(grid) >= 2 else 1.0
    lift_top = float(np.mean(canon[hi]) - np.mean(canon[prev])) if len(grid) >= 2 else 0.0
    return {"grid": grid, "n_test": int(len(test_ids)), "curve": curve,
            "slope_min_to_max": {"from_b": lo, "to_b": hi, "lift": round(lift, 4), "p": round(p_slope, 4)},
            "top_step": {"from_b": prev, "to_b": hi, "lift": round(lift_top, 4), "p": round(p_top, 4)}}


def certify_tasks(per_task: dict, alpha: float = ALPHA) -> list:
    """Apply BH-FDR over the per-task min->max slope p-values and annotate each task with the certified
    data-limited verdict in place. Returns the list of certified-data-limited task names. PURE."""
    tasks = list(per_task.keys())
    slope_p = [per_task[t]["slope_min_to_max"]["p"] for t in tasks]
    slope_lift = [per_task[t]["slope_min_to_max"]["lift"] for t in tasks]
    rej = set(benjamini_hochberg(slope_p, alpha=alpha))
    for i, t in enumerate(tasks):
        certified = (i in rej) and (slope_lift[i] > 0)
        st = per_task[t]
        top_rising = st["top_step"]["p"] < alpha and st["top_step"]["lift"] > 0
        st["data_limited_certified"] = bool(certified)
        st["still_rising_at_top"] = bool(top_rising)
        st["verdict"] = (
            "DATA-LIMITED: more labels certifiably raise sealed accuracy"
            + (" and the curve is still rising at the full pool" if top_rising
               else " but it is approaching the in-budget plateau")
            if certified else
            "FLAT within available labels: this much data does not certifiably move the metric "
            "(consistent with a representation/data ceiling not crossed by the labels on hand)")
    return [tasks[i] for i in range(len(tasks)) if (i in rej and slope_lift[i] > 0)]


def _arena_and_tasks(modality: str):
    """Return (arena, champion_tag, hard_tasks) for the requested modality, defaulting hard_tasks to the
    data-ceiling tasks recorded in that modality's autonomous certificate."""
    if modality == "vision":
        from scripts.repr_arena import FgvcAircraftArena
        arena = FgvcAircraftArena()
        champ = os.environ.get("ATTESTRA_CHAMP", "dinov2_g")
        cert = os.path.join(ROOT, "docs", "REPR_RESEARCHER_CERTIFICATE.json")
    elif modality == "text":
        from scripts.repr_arena_text import TwentyNewsArena
        arena = TwentyNewsArena()
        champ = os.environ.get("ATTESTRA_CHAMP", "mpnet")
        cert = os.path.join(ROOT, "docs", "REPR_RESEARCHER_TEXT_CERTIFICATE.json")
    else:
        raise SystemExit(f"unknown modality {modality!r}")
    hard = json.load(open(cert))["data_ceiling_tasks"] if os.path.exists(cert) else arena.tasks
    hard = [t for t in hard if t in arena.tasks]
    return arena, champ, hard


def run(modality: str, seeds: int):
    hashes = _frozen_hashes()
    assert hashes == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED: {hashes} != {FROZEN_EXPECTED}"
    print(f"frozen certifier verified: {hashes}")

    arena, champ, hard = _arena_and_tasks(modality)
    print(f"modality={modality}  champion={champ}  hard pairs ({len(hard)}): {hard}\n")

    per_task = {}
    for task in hard:
        sp = arena._task_split(task)
        y = np.asarray(sp["y"])
        tr_ids, val_ids, test_ids = np.asarray(sp["tr"]), np.asarray(sp["val"]), np.asarray(sp["test"])
        emb = arena._emb(champ, task)
        grid = _budget_grid(int(min(np.bincount(y[tr_ids]))))   # up to the full per-class training pool
        per_task[task] = eval_task(emb, y, tr_ids, val_ids, test_ids, grid, seeds)
        for c in per_task[task]["curve"]:
            print(f"  {task:30} b={c['budget_per_class']:4d}/class  "
                  f"acc {c['acc_mean']:.3f}+/-{c['acc_std']:.3f}  lb(canon) {c['acc_lb_canonical']:.3f}")

    survivors = certify_tasks(per_task, ALPHA)
    tasks = list(per_task.keys())
    out = {"modality": modality, "champion": champ, "alpha": ALPHA, "seeds": seeds,
           "hard_tasks": tasks, "data_limited_survivors": survivors,
           "n_data_limited": len(survivors), "per_task": per_task,
           "frozen_hashes": hashes}

    print("\n=== CERTIFIED DATA-CEILING VERDICT (BH-FDR over the hard pairs) ===")
    for t in tasks:
        st = per_task[t]
        print(f"  [{'DATA-LIMITED' if st['data_limited_certified'] else 'FLAT       '}] {t:30} "
              f"slope {st['slope_min_to_max']['lift']:+.3f} (p {st['slope_min_to_max']['p']:.4f}), "
              f"top-step {st['top_step']['lift']:+.3f} (p {st['top_step']['p']:.4f})")
    print(f"\n{len(survivors)}/{len(tasks)} hard pairs are CERTIFIED data-limited: {survivors}")

    dst = os.path.join(ROOT, "docs", f"DATA_CEILING_CURVE_{modality.upper()}.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nwrote {dst}")

    _plot(out, modality)
    print(f"frozen certifier (post-run): {_frozen_hashes()}")
    return out


def _plot(out, modality):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for t in out["hard_tasks"]:
        st = out["per_task"][t]
        xs = [c["n_train"] for c in st["curve"]]
        ys = [c["acc_mean"] for c in st["curve"]]
        es = [c["acc_std"] for c in st["curve"]]
        lab = f"{t} ({'data-limited' if st['data_limited_certified'] else 'flat'})"
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=lab)
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("training labels (total, both classes)  [sealed test + val held FIXED]")
    ax.set_ylabel("sealed accuracy (mean +/- std over subsamples)")
    ax.set_title(f"Certified data-ceiling curve -- {modality}, champion {out['champion']}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    dst = os.path.join(ROOT, "docs", f"DATA_CEILING_CURVE_{modality.upper()}.png")
    fig.tight_layout()
    fig.savefig(dst, dpi=120)
    print(f"wrote {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["vision", "text"], default="vision")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    run(args.modality, args.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
