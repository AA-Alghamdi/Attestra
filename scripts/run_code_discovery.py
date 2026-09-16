"""RUN open-ended AUTHORED-CODE discovery under sealed certification, on REAL non-binary task shapes.

The representation is pinned to RAW features, so the certifier is judging the SYSTEM'S OWN CODE: a featurizer
or estimator it authored (LLM or deterministic template), admitted through the frozen three-stage sandbox,
executed on the identical sealed rows, and promoted ONLY if it beats the weak linear champion under paired
McNemar + BH-FDR + frozen Clopper-Pearson, then confirmed on a never-peeked gold set.

  # deterministic, offline (reproducible) -- the default:
  python scripts/run_code_discovery.py --dataset covtype --shape multiclass

  # open-ended (Claude authors the source via the audited ops.llm_propose harness):
  ATTESTRA_CODE_LLM=1 python scripts/run_code_discovery.py --dataset covtype --shape multiclass --llm

The emitted certificate's `novelty.menu_free` is True iff the champion carries authored code
(`uses_authored_code`), i.e. the winning source was never in any menu.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import asdict

from scripts.code_arena import OPENML_CC18, REGRESSION_DATASETS, TabularCodeArena, TabularSplits
from vfplatform.code_authoring import LLMAuthorer, TemplateAuthorer, make_authorer
from vfplatform.recipe_generator import GeneratorConfig, RecipeGenerator
from vfplatform.recipe_research import RecipeResearcher


def _auto_theta_floor(arena, margin: float = 0.03) -> float:
    """A DATA-DRIVEN competence floor = the trivial (majority-class) baseline on TRAIN + a margin, never
    below 0.5. This makes "competent" mean "beats the trivial baseline" -- the honest bar on IMBALANCED
    classification, where raw accuracy 0.5 is meaningless. Computed on TRAIN labels only (never peeked from
    sealed/gold), so it cannot be snooped. On balanced sets the majority rate is small, so it stays at 0.5
    and nothing changes."""
    import numpy as np
    yt = arena.splits.y_train[arena.splits.train_idx]
    _, counts = np.unique(yt, return_counts=True)
    maj = float(counts.max()) / float(len(yt))
    return float(max(0.5, maj + margin))


def build(dataset: str, shape: str, *, use_llm: bool, seed: int, peeks: int, roles, split: str = "random",
          theta_floor=None):
    splits = TabularSplits(dataset=dataset, shape=shape, seed=seed, split=split)
    arena = TabularCodeArena(dataset=dataset, shape=shape, splits=splits)
    if theta_floor == "auto":
        arena.theta_floor = _auto_theta_floor(arena)
    elif theta_floor is not None:
        arena.theta_floor = float(theta_floor)
    if use_llm:
        authorer = LLMAuthorer(n_features=int(arena.splits.X.shape[1]), n_classes=arena.n_classes)
    else:
        authorer = TemplateAuthorer(seed=seed)
    cfg = GeneratorConfig(task_hint=arena.task_hint, fanout=6, code_prob=1.0,
                          code_roles=tuple(roles), graft_prob=0.0,
                          online_discovery=False)
    gen = RecipeGenerator(arena.seed_recipes(), config=cfg, seed=seed, code_authorer=authorer,
                          discover_fn=(lambda: []), task_shape=arena.task_shape)
    researcher = RecipeResearcher(arena, gen, alpha=0.1, theta_floor=arena.theta_floor, peek_budget=peeks,
                                  competence_ceiling=0.999, allow_data_acquisition=False,
                                  data_pool_root=os.path.expanduser("~/wilds_data/_codepool"),
                                  dataset_name=f"{dataset}_{shape}_{split}")
    return arena, gen, researcher, authorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="covtype",
                    choices=["covtype", "digits", "breast_cancer", "wine", "diabetes", "california"]
                    + list(OPENML_CC18))
    ap.add_argument("--shape", default="multiclass",
                    choices=["multiclass", "imbalanced", "noisy_label", "regression"])
    ap.add_argument("--split", default="random", choices=["random", "grouped", "time"],
                    help="random i.i.d. (default); grouped holds out WHOLE groups (covariate-shift / "
                         "extrapolation); time forward-chains train(past)->sealed(future)")
    ap.add_argument("--llm", action="store_true", help="use the open-ended Claude authorer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peeks", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--roles", default=None,
                    help="comma-separated code roles to author (default: featurizer,classifier; "
                         "regressor for --shape regression)")
    args = ap.parse_args()

    is_reg = (args.shape == "regression")
    if is_reg and args.dataset not in REGRESSION_DATASETS:
        ap.error(f"--shape regression needs a regression dataset {REGRESSION_DATASETS}, got {args.dataset!r}")
    if (not is_reg) and args.dataset in REGRESSION_DATASETS:
        ap.error(f"dataset {args.dataset!r} is regression-only; use --shape regression")

    if args.roles is not None:
        roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    elif is_reg:
        roles = ["regressor"]                 # featurizer head-pairing differs for regression; keep it clean
    else:
        roles = ["featurizer", "classifier"]
    # OpenML-CC18 sets get the data-driven competence floor (uncheatable on imbalanced); the curated synthetic
    # sets keep their fixed floor unless overridden.
    tf = "auto" if args.dataset in OPENML_CC18 else None
    arena, gen, researcher, authorer = build(args.dataset, args.shape, use_llm=args.llm, seed=args.seed,
                                             peeks=args.peeks, roles=roles, split=args.split, theta_floor=tf)
    print(f"== code-discovery: dataset={args.dataset} shape={args.shape} split={args.split} "
          f"n_classes={arena.n_classes} authorer={type(authorer).__name__} ==")
    if args.split == "grouped":
        sp = arena.splits
        ng = lambda idx: len(set(sp._groups_full[idx].tolist()))
        print(f"   grouped: held-out groups train={ng(sp.train_idx)} val={ng(sp.val_idx)} "
              f"sealed={[ng(s) for s in sp.shards]} gold={ng(sp.gold_idx)} (NO group straddles a boundary)")
    print(f"   tasks={arena.tasks}")
    print(f"   train={len(arena.splits.train_idx)} val={len(arena.splits.val_idx)} "
          f"sealed={[len(s) for s in arena.splits.shards]} gold={len(arena.splits.gold_idx)} "
          f"noise={arena.splits.noise}")
    if arena.is_regression:
        print(f"   regression: tolerance tau={arena.tau:.4f} (={arena.splits.tau_frac}*train-std) "
              f"theta_floor={arena.theta_floor} -> per-row hit iff |pred-y|<=tau")

    cert = researcher.run()

    print("\n-- climb log --")
    for line in cert.log:
        print("  " + line)

    champ = cert.champion_recipe
    print("\n== CERTIFICATE ==")
    print(f"  champion        : {cert.champion}")
    print(f"  code_role       : {champ.get('code_role')}  has_code={bool(champ.get('uses_code_patch'))}")
    print(f"  stop_reason     : {cert.stop_reason}")
    print(f"  peeks_used      : {cert.peeks_used}")
    print(f"  sealed_acc      : {cert.sealed_acc}")
    print(f"  promotions      : {[ (p.rung, p.to_recipe, round(p.mean_lift,3)) for p in cert.promotions ]}")
    print(f"  novelty         : menu_free={cert.novelty['menu_free']} "
          f"uses_authored_code={cert.novelty['uses_authored_code']} "
          f"backbone_is_novel={cert.novelty['backbone_is_novel']}")
    if cert.gold_confirmation:
        g = cert.gold_confirmation
        print(f"  gold (n={g['gold_n']}) : {len(g['survivors'])}/{g['n_tasks']} FDR survivors "
              f"confirmed={g['confirmed']}")
    if cert.framing_report:
        print(f"  framing         : trustworthy={cert.framing_report['trustworthy']}")
    if cert.refused:
        print("  REFUSED (framing/data-hygiene gate)")
    if isinstance(authorer, LLMAuthorer):
        print(f"  llm_calls       : {authorer.calls}")

    _tag = f"{args.dataset}_{args.shape}" + ("" if args.split == "random" else f"_{args.split}")
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "docs", f"CODE_DISCOVERY_{_tag}.json")
    payload = {k: v for k, v in asdict(cert).items()}
    if isinstance(authorer, LLMAuthorer):
        payload["llm_calls"] = authorer.calls
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote {out}")
    return cert


if __name__ == "__main__":
    main()
