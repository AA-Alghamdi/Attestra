"""#1d -- RUN THE AUTONOMOUS RESEARCHER END-TO-END ON THE FGVC ARENA, FROM A WEAK CHAMPION.

The acceptance test for "the autoresearcher is the system, not me": starting from the weak deployed champion
(CLIP ViT-B/32, pooled sealed 0.683), the policy must AUTONOMOUSLY reproduce the hand-run verdict it took me
seven phases to reach -- with the frozen certifier as the sole promoter and no human in the loop:

  1. climb the representation ladder to DINOv2 ViT-g/14 (the strongest self-supervised encoder);
  2. REJECT SigLIP-SO400M and EVA-02-L (a stronger language-aligned and a different SSL encoder) -- "bigger"
     is not the lever, "the winning inductive bias" is;
  3. flag the 737 confusable pairs as a DATA ceiling (representation-exhausted residual).

It also emits the certificate: frozen hashes pinned, the full promotion/rejection log, the move-class path,
and the certified accuracy-vs-cost Pareto front. Run: `python scripts/run_repr_researcher.py`.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.repr_researcher import ReprResearcher                       # noqa: E402
from scripts.repr_arena import FgvcAircraftArena, REGISTRY                  # noqa: E402

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
    arena = FgvcAircraftArena(smoke=smoke)
    researcher = ReprResearcher(REGISTRY, arena, start_tag="clip_vitb32",
                                alpha=0.1, theta_floor=0.5, peek_budget=12, competence_ceiling=0.90)
    print(f"start champion=clip_vitb32  registry={[e.tag for e in REGISTRY]}  tasks={len(arena.tasks)}\n")

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

    # ---- acceptance: did the SYSTEM autonomously reproduce the hand-run verdict? ----
    promoted_tags = [p.to_tag for p in cert.promotions]
    rejected_tags = {r.tag for r in cert.rejections}
    checks = {
        "champion is DINOv2-g": cert.champion == "dinov2_g",
        "climbed via DINOv2-L then g": "dinov2_vitl14" in promoted_tags and "dinov2_g" in promoted_tags,
        "rejected SigLIP-SO400M": "siglip_so" in rejected_tags,
        "rejected EVA-02-L": "eva02_l" in rejected_tags,
        "flagged 737-700/800 as data ceiling": "737-700_vs_737-800" in cert.data_ceiling_tasks,
        "flagged 737-300/400 as data ceiling": "737-300_vs_737-400" in cert.data_ceiling_tasks,
    }
    print("\n=== ACCEPTANCE (system reproduces hand-run verdict) ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_pass = all(checks.values())

    out = {
        "arena": "fgvc-aircraft", "start_champion": "clip_vitb32", "champion": cert.champion,
        "champion_family": cert.champion_family, "move_class_path": cert.move_class_path,
        "stop_reason": cert.stop_reason, "peeks_used": cert.peeks_used,
        "promotions": [vars(p) for p in cert.promotions],
        "rejections": [vars(r) for r in cert.rejections],
        "data_ceiling_tasks": cert.data_ceiling_tasks,
        "sealed_acc": cert.sealed_acc, "sealed_lb": cert.sealed_lb,
        "gold_confirmation": cert.gold_confirmation,
        "multiplicity": cert.multiplicity,
        "pareto_front": cert.pareto_front, "pareto_report": cert.pareto_report,
        "acceptance": checks, "all_pass": all_pass, "frozen_hashes": hashes, "log": cert.log,
    }
    dst = os.path.join(ROOT, "docs", os.environ.get("ATTESTRA_OUT", "REPR_RESEARCHER_CERTIFICATE.json"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")
    print(f"frozen certifier (post-run): {_frozen_hashes()}")
    print(f"\nALL ACCEPTANCE CHECKS {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
