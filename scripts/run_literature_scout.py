"""LITERATURE-GROUNDED DISCOVERY -- live demonstration / smoke.

Runs the FULL regenerative loop with the literature layer engaged:

    LiteratureScout (live arXiv + GitHub + HF Hub [+ Papers-with-Code])  -- reads the research surface
        -> typed backbone ids + technique MOTIFS, each with PROVENANCE
            -> RecipeGenerator (open pool, grafted with the RETRIEVED motifs)
                -> validity cascade -> FROZEN Tier-3 certifier (sole promoter)
                    -> champion that TRACES to a real paper/repo/Hub hit (literature_grounded=True)

The evaluation arena here is a synthetic stand-in (CPU, deterministic) that rewards the strong-transfer
family the literature surfaces for distribution shift -- exactly as tests/test_recipe_research.py does. It
exercises the REAL frozen primitives (McNemar+BH-FDR+Clopper-Pearson), so this proves the PLUMBING end to
end: a backbone the system READ ABOUT IN THE WILD becomes the frozen-certified champion and is traceable to
its source. The absolute-SOTA evaluation on real images is the GPU run (scripts/run_wilds_gpu.py).

NOT a pytest (it hits the network). Usage:
    python scripts/run_literature_scout.py                       # live retrieval
    python scripts/run_literature_scout.py --offline             # bundled corpus only (no network)
    python scripts/run_literature_scout.py --llm                 # let Claude read the findings (needs key)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.literature import LiteratureScout  # noqa: E402
from vfplatform.recipe import Recipe  # noqa: E402
from vfplatform.recipe_generator import GeneratorConfig, RecipeGenerator  # noqa: E402
from vfplatform.recipe_research import RecipeArena, RecipeResearcher  # noqa: E402
from vfplatform.repr_researcher import TaskMeasure  # noqa: E402

FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PER_TASK = 200
TASKS = ["center_a", "center_b", "center_c"]

# The "true accuracy" the literature predicts for distribution shift: a strong self-supervised / large
# transfer backbone (dinov2 / eva02 / siglip / convnext) generalizes across the shift; the resnet seed is
# weak. This is the SAME principle the WILDS result demonstrated -- here it is a fast CPU stand-in so the
# literature->champion->provenance plumbing can be certified without a GPU.
_STRONG = ("dinov2", "eva02", "siglip", "convnext", "beit", "swin", "clip")
_FAMILY_ACC = {"dinov2": 0.90, "eva02": 0.89, "siglip": 0.87, "convnext": 0.85, "beit": 0.84,
               "swin": 0.83, "clip": 0.74, "resnet": 0.60, "other": 0.56}


def _family(backbone: str) -> str:
    b = backbone.lower()
    for fam in _FAMILY_ACC:
        if fam != "other" and fam in b:
            return fam
    return "other"


def _acc(recipe: Recipe) -> float:
    base = _FAMILY_ACC[_family(recipe.backbone)]
    bump = 0.01 if recipe.adaptation in ("lora", "full_ft", "vpt") else 0.0
    return min(0.97, base + bump)


def _correct(acc: float, salt: str) -> list:
    rng = random.Random(hash(salt) & 0xFFFFFFFF)
    return [1 if rng.random() < acc else 0 for _ in range(N_PER_TASK)]


class LitShiftArena(RecipeArena):
    tasks = TASKS
    task_hint = "vision"
    task_shape = "binary"

    def seed_recipes(self):
        return [Recipe(backbone="resnet18", adaptation="linear_probe")]

    def measure(self, recipe: Recipe):
        acc = _acc(recipe)
        return {t: TaskMeasure(sealed_correct=_correct(acc, f"sealed::{t}"),
                               val_correct=_correct(acc, f"val::{t}"),
                               acc=acc) for t in self.tasks}

    def gold_measure(self, recipe: Recipe):
        acc = _acc(recipe)
        return {t: _correct(acc, f"gold::{t}") for t in self.tasks}


def _frozen_hashes():
    return {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:8]
            for f in FROZEN_EXPECTED}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="histopathology tumor classification under hospital "
                    "distribution shift")
    ap.add_argument("--offline", action="store_true", help="bundled corpus only (no network)")
    ap.add_argument("--llm", action="store_true", help="let Claude read the findings (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--with-zoo", action="store_true",
                    help="also union the open timm zoo into discovery (default: literature channel only, "
                         "so the champion's grounding is tested in isolation)")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "LITERATURE_GROUNDED_RESULT.json"))
    args = ap.parse_args()

    hashes = _frozen_hashes()
    assert hashes == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED: {hashes} != {FROZEN_EXPECTED}"
    print(f"frozen certifier verified: {hashes}")

    print(f"\n### LITERATURE SCOUT  (online={not args.offline}, llm={args.llm})")
    print(f"problem: {args.problem!r}")
    scout = LiteratureScout(args.problem, online=not args.offline, use_llm=args.llm, limit=args.limit)
    summ = scout.summary()
    print(f"retrieved {summ['n_findings']} findings {summ['findings_by_source']}")
    print(f"backbones surfaced ({summ['n_backbones']}): {summ['backbones']}")
    print(f"motifs distilled ({summ['n_motifs']}): {summ['motifs']}")
    print(f"used_llm: {summ['used_llm']}")

    arena = LitShiftArena()
    # default: literature is the ONLY non-seed backbone source (zoo disabled), so a certified non-seed
    # champion MUST trace to the literature -- the cleanest isolation of the grounding claim. --with-zoo
    # unions the open zoo too (then the champion is whichever genuinely wins, reported honestly).
    discover_fn = None if args.with_zoo else (lambda: [])
    gen = RecipeGenerator(arena.seed_recipes(),
                          config=GeneratorConfig(task_hint="vision", pool_limit=16,
                                                 online_discovery=False),
                          scout=scout, discover_fn=discover_fn, task_shape="binary", seed=0)
    researcher = RecipeResearcher(arena, gen, alpha=0.1, theta_floor=0.5, peek_budget=24,
                                  competence_ceiling=0.90, max_rounds=24, dataset_name="lit_shift")
    cert = researcher.run()

    nov = cert.novelty
    print(f"\n### CERTIFICATE")
    print(f"champion         : {cert.champion}")
    print(f"menu_free        : {nov['menu_free']}")
    print(f"backbone_is_novel: {nov['backbone_is_novel']}")
    print(f"literature_grounded: {nov.get('literature_grounded')}")
    src = nov.get("literature_source")
    if src:
        print(f"  traces to      : [{src.get('source')}:{src.get('ident')}] {src.get('title')!r}")
        print(f"  url            : {src.get('url')}")
    gc = cert.gold_confirmation
    print(f"gold confirmed   : {None if gc is None else gc.get('confirmed')}")

    post = _frozen_hashes()
    checks = {
        "frozen certifier byte-identical (sole promoter, untouched)": post == FROZEN_EXPECTED,
        "champion is menu-free (reached beyond the seed)": bool(nov["menu_free"]),
        "champion backbone is NOT a seed": bool(nov["backbone_is_novel"]),
        "champion gold-confirmed on a never-peeked set":
            gc is not None and bool(gc.get("confirmed")),
    }
    if not args.with_zoo:
        # literature-only mode: a certified non-seed champion MUST trace to the literature
        checks["champion traces to a LITERATURE source (not a hand-list)"] = bool(nov.get("literature_grounded"))
        checks["literature provenance has a real url"] = bool(src and src.get("url"))
    print("\n### CHECKS")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_pass = all(checks.values())

    out = {
        "problem": args.problem,
        "online": not args.offline,
        "frozen_hashes": post,
        "scout_summary": summ,
        "champion": cert.champion,
        "champion_recipe": cert.champion_recipe,
        "novelty": nov,
        "discovery": cert.discovery,
        "promotions": [{"rung": p.rung, "from": p.from_recipe, "to": p.to_recipe,
                        "survivors": p.survivors, "mean_lift": p.mean_lift} for p in cert.promotions],
        "gold_confirmation": gc,
        "checks": checks,
        "all_pass": all_pass,
        "log": cert.log,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    print(f"frozen certifier (post-run): {post}")
    print(f"\nALL CHECKS {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
