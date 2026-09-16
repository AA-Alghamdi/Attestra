"""PHASE 1 (GPU) ONE-COMMAND LAUNCH -- regenerative recipe discovery vs a human-grade ViT-L fine-tune, on
real WILDS distribution shift, frozen-certified. This is the absolute-SOTA arena the user locked.

    # GPU day (the real run):
    ATTESTRA_DEVICE=cuda python scripts/run_wilds_gpu.py
    # CPU pipeline smoke (tiny, proves the plumbing end-to-end without a GPU):
    ATTESTRA_DEVICE=cpu python scripts/run_wilds_gpu.py --smoke

THE LOOP (unchanged spine): the LLM/generator REGENERATES full recipes -> the GPU runner EXECUTES them
(honouring adaptation/aug/optim/schedule/epochs/aggregation, scripts/wilds_gpu.py) -> the FROZEN certifier
PROMOTES on sealed evidence (vfplatform.verification Tier 3). On top of the discovery certificate, this
script runs the ABSOLUTE-SOTA head-to-head: the discovered champion vs human_baseline_recipe() (a strong
ViT-L full fine-tune) on the IDENTICAL carved sealed shards, paired McNemar + BH-FDR + frozen Clopper-Pearson
lower bounds, then a never-peeked gold confirmation. "Beats a mid-level engineer on absolutes" is earned iff
the champion FDR-beats the human baseline AND its gold-confirmed sealed lower bound clears the human's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.wilds_arena import Camelyon17Splits                       # noqa: E402
from scripts.wilds_gpu import GpuWildsArena, human_baseline_recipe, device  # noqa: E402
from vfplatform.recipe import Recipe                                   # noqa: E402
from vfplatform.recipe_generator import GeneratorConfig, RecipeGenerator  # noqa: E402
from vfplatform.recipe_research import RecipeResearcher                # noqa: E402

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                   "WILDS_GPU_SOTA_CERTIFICATE.json")


def _certify_head_to_head(arena: GpuWildsArena, champion: Recipe, human: Recipe, *, alpha: float = 0.1):
    """The absolute-SOTA referee: score BOTH recipes on the IDENTICAL sealed shards, paired McNemar per
    shard, BH-FDR(alpha), frozen Clopper-Pearson lower bounds, then a never-peeked gold read. The frozen
    certifier is the sole arbiter -- this function only assembles its primitives over identical rows."""
    champ_m = arena.measure(champion)
    human_m = arena.measure(human)
    tasks = arena.tasks
    pvals, lifts, per_task = [], [], []
    for t in tasks:
        c, h = champ_m[t].sealed_correct, human_m[t].sealed_correct
        p = arena.mcnemar(c, h)
        lift = champ_m[t].acc - human_m[t].acc
        pvals.append(p); lifts.append(lift)
        per_task.append({"task": t, "champion_acc": round(champ_m[t].acc, 4),
                         "human_acc": round(human_m[t].acc, 4), "lift": round(lift, 4),
                         "mcnemar_p": round(p, 5)})
    rej = set(arena.bh(pvals, alpha))
    survivors = [tasks[i] for i in range(len(tasks)) if i in rej and lifts[i] > 0]
    champ_pool = [x for t in tasks for x in champ_m[t].sealed_correct]
    human_pool = [x for t in tasks for x in human_m[t].sealed_correct]
    champ_lb, human_lb = arena.lower_bound(champ_pool), arena.lower_bound(human_pool)
    champ_acc = float(np.mean(champ_pool)); human_acc = float(np.mean(human_pool))

    # never-peeked gold confirmation on the disjoint gold set
    gold_c = arena.gold_measure(champion)[tasks[0]]
    gold_h = arena.gold_measure(human)[tasks[0]]
    gold_champ_acc = float(np.mean(gold_c)); gold_human_acc = float(np.mean(gold_h))
    gold_champ_lb = arena.lower_bound(gold_c)
    gold_p = arena.mcnemar(gold_c, gold_h)

    beats_sealed = len(survivors) > 0 and champ_lb > human_acc
    beats_gold = gold_champ_lb > gold_human_acc and gold_champ_acc > gold_human_acc
    return {
        "champion": champion.label(), "human_baseline": human.label(),
        "per_task": per_task, "fdr_survivors": survivors, "n_tasks": len(tasks),
        "champion_sealed_acc": round(champ_acc, 4), "human_sealed_acc": round(human_acc, 4),
        "champion_sealed_lb": round(champ_lb, 4), "human_sealed_lb": round(human_lb, 4),
        "gold_champion_acc": round(gold_champ_acc, 4), "gold_human_acc": round(gold_human_acc, 4),
        "gold_champion_lb": round(gold_champ_lb, 4), "gold_mcnemar_p": round(gold_p, 5),
        # the headline: did the discovered champion beat the human-grade baseline on absolutes?
        "beats_human_sealed": bool(beats_sealed), "beats_human_gold": bool(beats_gold),
        "absolute_sota_earned": bool(beats_sealed and beats_gold),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU pipeline check (no GPU needed)")
    ap.add_argument("--peek-budget", type=int, default=12)
    ap.add_argument("--fanout", type=int, default=4)
    ap.add_argument("--pool-limit", type=int, default=10)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = device()
    if args.smoke:
        splits = Camelyon17Splits(n_train=60, n_val=40, n_test=60, n_gold=40, n_shards=3, seed=args.seed)
        peek_budget, fanout, pool_limit = 3, 2, 4
        # a LIGHT human comparator so the head-to-head plumbing runs on CPU; the real run uses ViT-L.
        human = Recipe(backbone="resnet18", adaptation="full_ft", augmentation="randaug", schedule="cosine",
                       lr=1e-4, weight_decay=5e-2, epochs=1, head="linear", notes="SMOKE human baseline")
        os.environ.setdefault("ATTESTRA_GPU_MAX_TRAIN", "40")
        os.environ.setdefault("ATTESTRA_GPU_EPOCH_CAP", "1")
        os.environ.setdefault("ATTESTRA_GPU_BATCH", "16")
    else:
        splits = Camelyon17Splits(n_train=1200, n_val=600, n_test=900, n_gold=600, n_shards=3, seed=args.seed)
        peek_budget, fanout, pool_limit = args.peek_budget, args.fanout, args.pool_limit
        human = human_baseline_recipe()

    print(f"[wilds-gpu] device={dev} tasks={splits.tasks} train={len(splits.train_idx)} "
          f"val={len(splits.val_idx)} sealed={sum(len(s) for s in splits.shards)} "
          f"gold={len(splits.gold_idx)}", flush=True)

    arena = GpuWildsArena(splits)
    gen = RecipeGenerator(arena.seed_recipes(),
                          config=GeneratorConfig(task_hint="fine_grained", fanout=fanout,
                                                 pool_limit=pool_limit,
                                                 online_discovery=bool(os.environ.get("HF_TOKEN")),
                                                 code_prob=0.0),
                          seed=args.seed, task_shape="binary")
    researcher = RecipeResearcher(arena, gen, alpha=args.alpha, theta_floor=args.theta,
                                  peek_budget=peek_budget, competence_ceiling=0.95, driver="climb",
                                  max_rounds=12,
                                  data_pool_root=os.path.join(os.path.dirname(DOC), "_wilds_gpu_datapool"),
                                  dataset_name="wilds_camelyon17_gpu")
    t0 = time.time()
    cert = researcher.run()
    champion = Recipe.from_genome(cert.champion_recipe) if cert.champion_recipe else arena.seed_recipes()[0]

    print(f"[wilds-gpu] champion = {champion.label()}  (menu_free={cert.novelty['menu_free']}); "
          f"now certifying champion vs human baseline ({human.label()}) on identical sealed rows...",
          flush=True)
    sota = _certify_head_to_head(arena, champion, human, alpha=args.alpha) if not cert.refused else None
    dt = time.time() - t0

    out = {"discovery": asdict(cert), "absolute_sota": sota, "device": dev,
           "human_baseline_recipe": asdict_recipe(human), "wall_seconds": round(dt, 1),
           "dataset": "wilds/camelyon17 (centers train={0,3,4} val=1 test=2)"}
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n==================== WILDS GPU ABSOLUTE-SOTA CERTIFICATE ====================")
    print(f"device           : {dev}")
    print(f"champion         : {champion.label()}")
    print(f"MENU_FREE        : {cert.novelty['menu_free']} (backbone_novel={cert.novelty['backbone_is_novel']})")
    print(f"refused          : {cert.refused}")
    if sota is not None:
        print(f"champion sealed  : acc={sota['champion_sealed_acc']} lb={sota['champion_sealed_lb']}")
        print(f"human   sealed  : acc={sota['human_sealed_acc']} lb={sota['human_sealed_lb']}")
        print(f"FDR survivors    : {sota['fdr_survivors']} / {sota['n_tasks']}")
        print(f"gold champ/human : {sota['gold_champion_acc']} / {sota['gold_human_acc']} "
              f"(champ_lb={sota['gold_champion_lb']})")
        print(f"BEATS HUMAN      : sealed={sota['beats_human_sealed']} gold={sota['beats_human_gold']} "
              f"=> ABSOLUTE_SOTA_EARNED={sota['absolute_sota_earned']}")
    print(f"wall_seconds     : {out['wall_seconds']}")
    print(f"\nwrote {DOC}")


def asdict_recipe(r: Recipe) -> dict:
    from dataclasses import asdict as _ad
    return _ad(r)


if __name__ == "__main__":
    main()
