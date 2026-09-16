"""Run the REGENERATIVE autoresearcher on WILDS Camelyon17 (real distribution shift), CPU Phase 0.

    python scripts/run_wilds_autoresearch.py            # certified run, modest CPU budget
    python scripts/run_wilds_autoresearch.py --smoke    # tiny: resnet18 seed + a couple of proposals
    python scripts/run_wilds_autoresearch.py --driver search

Emits docs/WILDS_AUTORESEARCH_CERTIFICATE.json: the frozen distribution-shift certificate + the ANTI-MENU
proof (DiscoveryLedger.novelty -> menu_free), produced by the OPEN generator (no encoder registry).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.wilds_arena import Camelyon17Splits, WildsCamelyonArena   # noqa: E402
from vfplatform.recipe_generator import GeneratorConfig, RecipeGenerator  # noqa: E402
from vfplatform.recipe_research import RecipeResearcher                 # noqa: E402

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs",
                   "WILDS_AUTORESEARCH_CERTIFICATE.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny budget for a fast pipeline check")
    ap.add_argument("--driver", choices=["climb", "search"], default="climb")
    ap.add_argument("--peek-budget", type=int, default=10)
    ap.add_argument("--fanout", type=int, default=4)
    ap.add_argument("--pool-limit", type=int, default=10)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        splits = Camelyon17Splits(n_train=300, n_val=200, n_test=300, n_gold=200, n_shards=3, seed=args.seed)
        peek_budget, fanout, pool_limit = 4, 3, 6
    else:
        splits = Camelyon17Splits(n_train=1200, n_val=600, n_test=900, n_gold=600, n_shards=3, seed=args.seed)
        peek_budget, fanout, pool_limit = args.peek_budget, args.fanout, args.pool_limit

    print(f"[wilds] tasks={splits.tasks} "
          f"train={len(splits.train_idx)} val={len(splits.val_idx)} "
          f"sealed={sum(len(s) for s in splits.shards)} gold={len(splits.gold_idx)}", flush=True)

    arena = WildsCamelyonArena(splits)
    seed_recipes = arena.seed_recipes()
    # OPEN proposer: discovery is biased toward the task (histopathology -> fine-grained vision), the pool
    # GROWS from the timm/HF zoo at runtime; nothing here enumerates a fixed encoder list.
    cfg = GeneratorConfig(task_hint="fine_grained", fanout=fanout, pool_limit=pool_limit,
                          online_discovery=bool(os.environ.get("HF_TOKEN")), code_prob=0.0)
    gen = RecipeGenerator(seed_recipes, config=cfg, seed=args.seed, task_shape="binary")

    researcher = RecipeResearcher(arena, gen, alpha=0.1, theta_floor=args.theta, peek_budget=peek_budget,
                                  competence_ceiling=0.90, driver=args.driver, max_rounds=12,
                                  data_pool_root=os.path.join(os.path.dirname(DOC), "_wilds_datapool"),
                                  dataset_name="wilds_camelyon17")
    t0 = time.time()
    cert = researcher.run()
    dt = time.time() - t0

    out = asdict(cert)
    out["wall_seconds"] = round(dt, 1)
    out["dataset"] = "wilds/camelyon17 (centers train={0,3,4} val=1 test=2)"
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n==================== WILDS AUTORESEARCH CERTIFICATE ====================")
    print(f"driver           : {cert.driver}")
    print(f"refused          : {cert.refused}  (meta-certifier framing gate / data-hygiene gate)")
    if cert.data_hygiene is not None:
        dh = cert.data_hygiene
        print(f"data_hygiene     : admitted={dh.get('admitted')} "
              f"near_dup_straddle={dh.get('certificate', {}).get('near_dup_straddle')}")
    print(f"champion         : {cert.champion}")
    print(f"champion backbone: {cert.champion_recipe.get('backbone')}")
    print(f"MENU_FREE        : {cert.novelty['menu_free']}  "
          f"(backbone_novel={cert.novelty['backbone_is_novel']}, "
          f"from_retrieval={cert.novelty['from_retrieval']})")
    print(f"promotions       : {[(p.from_recipe.split(chr(183))[0].strip(), '->', p.to_recipe.split(chr(183))[0].strip()) for p in cert.promotions]}")
    print(f"move_class_path  : {cert.move_class_path}")
    print(f"sealed_acc       : {cert.sealed_acc}")
    print(f"sealed_lb        : {cert.sealed_lb}")
    print(f"peeks_used       : {cert.peeks_used}")
    print(f"discovery        : pool={cert.discovery['pool_size']} seeds={cert.discovery['n_seed_backbones']} "
          f"proposed={cert.discovery['n_proposed_backbones']} retrieved={cert.discovery['n_retrieved_backbones']}")
    print(f"qd_coverage      : {cert.qd_coverage}")
    if cert.gold_confirmation:
        print(f"gold (never-peek): confirmed={cert.gold_confirmation['confirmed']} "
              f"mean_lift={cert.gold_confirmation['mean_lift']} n={cert.gold_confirmation['gold_n']}")
    if cert.multiplicity:
        print(f"multiplicity     : M={cert.multiplicity['sealed_comparisons']} "
              f"bonferroni_survivors={len(cert.multiplicity['bonferroni_survivors_session'])} "
              f"robust={cert.multiplicity['robust_to_session_multiplicity']}")
    print(f"wall_seconds     : {out['wall_seconds']}")
    print(f"\nwrote {DOC}")


if __name__ == "__main__":
    main()
