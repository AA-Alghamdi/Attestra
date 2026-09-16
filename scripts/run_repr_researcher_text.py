"""#2 -- RUN THE *SAME* AUTONOMOUS RESEARCHER ON TEXT, FROM A WEAK (LEXICAL) CHAMPION.

This is the generality acceptance for "the autoresearcher is the system, not me, and not vision-specific":
the IDENTICAL vfplatform.repr_researcher.ReprResearcher policy + the IDENTICAL frozen certifier, pointed at a
TEXT arena (20-Newsgroups confusable pairs over frozen sentence encoders). Starting from a weak lexical
bag-of-words champion (TF-IDF+LSA), the policy must AUTONOMOUSLY climb the representation ladder under the
same paired-McNemar + BH-FDR + frozen Clopper-Pearson discipline -- with the frozen certifier as the sole
promoter and no human deciding anything.

The acceptance checks are GENERALITY PROPERTIES (not a pre-baked winner, since the text champion is an
empirical question): the lever must fire (promote off the lexical baseline), the first move must be a
family swap (lexical -> a neural sentence encoder), every promotion must be FDR-certified, and the frozen
certifier must be byte-identical before and after. The concrete reproduced verdict (which encoder wins, what
gets rejected, which pairs are a data ceiling) is recorded in the certificate. Run:
`python scripts/run_repr_researcher_text.py`.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.repr_researcher import ReprResearcher                       # noqa: E402
from scripts.repr_arena_text import BASELINE_TAG, REGISTRY, TwentyNewsArena  # noqa: E402

FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _frozen_hashes():
    return {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:8]
            for f in FROZEN_EXPECTED}


def main():
    hashes = _frozen_hashes()
    assert hashes == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED: {hashes} != {FROZEN_EXPECTED}"
    print(f"frozen certifier verified: {hashes}")

    smoke = int(os.environ.get("ATTESTRA_SMOKE", "0"))
    arena = TwentyNewsArena(smoke=smoke)
    researcher = ReprResearcher(REGISTRY, arena, start_tag=BASELINE_TAG,
                                alpha=0.1, theta_floor=0.5, peek_budget=16, competence_ceiling=0.90)
    print(f"start champion={BASELINE_TAG}  registry={[e.tag for e in REGISTRY]}  tasks={len(arena.tasks)}\n")

    cert = researcher.run()

    print("=== RESEARCH LOG ===")
    for line in cert.log:
        print(" ", line)
    print("\n=== CERTIFICATE ===")
    print(f"champion         : {cert.champion} ({cert.champion_family})")
    print(f"move-class path  : {' -> '.join(cert.move_class_path)}")
    print(f"stop reason      : {cert.stop_reason}")
    print(f"peeks used       : {cert.peeks_used}")
    print("promotions:")
    for p in cert.promotions:
        print(f"  [{p.rung:16}] {p.from_tag} -> {p.to_tag}  "
              f"(survivors {len(p.survivors)}/{p.n}, mean_lift {p.mean_lift:+.3f}) {p.survivors}")
    print("rejections:")
    for r in cert.rejections:
        print(f"  [{r.rung:16}] {r.tag:28} {len(r.survivors)}/{r.n}  {r.reason}")
    print(f"\ndata-ceiling tasks ({len(cert.data_ceiling_tasks)}): {cert.data_ceiling_tasks}")
    print("\n" + cert.pareto_report)

    gc = cert.gold_confirmation
    if gc is not None:
        print("\n=== GOLD CONFIRMATION (disjoint, never-peeked set; read once after climbing) ===")
        print(f"  {gc['champion']} vs {gc['baseline']}  gold_n={gc['gold_n']}  "
              f"FDR survivors {len(gc['survivors'])}/{gc['n_tasks']}  mean_lift {gc['mean_lift']:+.3f}")
        print(f"  champion gold lower bound {gc['champion_gold_lb']:.3f} "
              f"(baseline {gc['baseline_gold_lb']:.3f})  CONFIRMED={gc['confirmed']}")
        for r in gc["per_task"]:
            print(f"    {r['task']:28} champ {r['champ_gold_acc']:.3f}  base {r['base_gold_acc']:.3f}  "
                  f"lift {r['lift']:+.3f}  p {r['p_gt_base']:.4f}")

    mp = cert.multiplicity
    if mp is not None:
        print("\n=== SESSION MULTIPLICITY (honest accounting of sealed-test re-use) ===")
        print(f"  {mp['sealed_comparisons']} sealed certifications spent climbing "
              f"({mp['mcnemar_tests_total']} paired McNemar tests total)")
        print(f"  per-comparison FDR alpha {mp['per_comparison_fdr_alpha']}  ->  "
              f"family-wise Bonferroni alpha/M {mp['session_bonferroni_alpha']:.4f}")
        print(f"  champion vs start on sealed: {len(mp['fdr_survivors_nominal'])} FDR-nominal survivors, "
              f"{len(mp['bonferroni_survivors_session'])}/{mp['n_tasks']} survive the Bonferroni correction")
        print(f"  robust to session multiplicity = {mp['robust_to_session_multiplicity']}  |  "
              f"multiplicity-FREE gold confirmation = {mp['gold_independent_confirmation']}")

    # ---- acceptance: GENERALITY PROPERTIES (the brain climbs the representation axis on a new domain) ----
    promoted_tags = [p.to_tag for p in cert.promotions]
    checks = {
        "representation lever fires (promoted off the lexical baseline)":
            cert.champion != BASELINE_TAG and len(cert.promotions) >= 1,
        "first move is a family swap lexical -> neural sentence encoder":
            len(cert.promotions) >= 1 and cert.promotions[0].rung == "model"
            and cert.promotions[0].from_tag == BASELINE_TAG,
        "every promotion is FDR-certified (>=1 survivor, positive mean lift)":
            all(len(p.survivors) >= 1 and p.mean_lift > 0 for p in cert.promotions),
        "champion is a frozen neural sentence encoder (not lexical)":
            cert.champion in {e.tag for e in REGISTRY} and cert.champion_family != "lexical",
        "champion CONFIRMED over the baseline on a NEVER-PEEKED gold set (>=1 FDR survivor, positive lift)":
            cert.gold_confirmation is not None and cert.gold_confirmation["confirmed"],
        "frozen certifier byte-identical (sole promoter, untouched)":
            _frozen_hashes() == FROZEN_EXPECTED,
    }
    print("\n=== ACCEPTANCE (generality: the same brain climbs the representation axis on TEXT) ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_pass = all(checks.values())

    out = {
        "arena": "20newsgroups-confusable-pairs", "start_champion": BASELINE_TAG, "champion": cert.champion,
        "champion_family": cert.champion_family, "move_class_path": cert.move_class_path,
        "stop_reason": cert.stop_reason, "peeks_used": cert.peeks_used,
        "promoted_tags": promoted_tags,
        "promotions": [vars(p) for p in cert.promotions],
        "rejections": [vars(r) for r in cert.rejections],
        "data_ceiling_tasks": cert.data_ceiling_tasks,
        "sealed_acc": cert.sealed_acc, "sealed_lb": cert.sealed_lb,
        "gold_confirmation": cert.gold_confirmation,
        "multiplicity": cert.multiplicity,
        "pareto_front": cert.pareto_front, "pareto_report": cert.pareto_report,
        "acceptance": checks, "all_pass": all_pass, "frozen_hashes": hashes, "log": cert.log,
    }
    dst = os.path.join(ROOT, "docs", os.environ.get("ATTESTRA_OUT", "REPR_RESEARCHER_TEXT_CERTIFICATE.json"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")
    print(f"frozen certifier (post-run): {_frozen_hashes()}")
    print(f"\nALL ACCEPTANCE CHECKS {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
