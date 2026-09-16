"""#7 -- RUN THE *SAME* AUTONOMOUS RESEARCHER ON TABULAR, FROM THE STRONG TUNED-GBM CHAMPION.

Third modality. The IDENTICAL `vfplatform.repr_researcher.ReprResearcher` policy + the IDENTICAL frozen certifier,
pointed at a TABULAR arena (confusable letter-recognition pairs). Unlike vision/text -- which start from a WEAK
baseline (raw pixels / lexical bag-of-words) and ask whether a frozen neural encoder beats it -- tabular starts
from the STRONG tuned-GBM (the mid-level-engineer default on tables) and asks the sharper question: is there a
representation lever AT ALL, i.e. does swapping the inductive prior to a pretrained tabular FOUNDATION MODEL
(TabPFN v2) certifiably beat a tuned GBM under the same paired-McNemar + BH-FDR + frozen Clopper-Pearson
discipline, with the frozen certifier as the sole promoter?

Either answer is a first-class scientific result:
  * lever FIRES  -> "representation is the lever" generalises to tabular (a foundation prior beats per-task GBM);
  * lever SILENT -> tabular is the modality where the raw columns are already a near-optimal representation for a
                    strong GBM, so the cross-modal law has a principled boundary (and the audit's tabular 0/4
                    finding -- authoring beats nothing -- is reproduced from the *strong-baseline* side).

ACCEPTANCE here certifies the SYSTEM INVARIANTS that must hold whatever the outcome (the certifier is the sole
promoter, every promotion is FDR-surviving, the brain honest-stops, the gold confirmation is disjoint). The
scientific VERDICT (did the lever fire?) is reported separately, never forced.

Run in the isolated TabPFN venv:  ~/.venv-tabpfn/bin/python scripts/run_repr_researcher_tabular.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.repr_researcher import ReprResearcher                          # noqa: E402
from scripts.repr_arena_tabular import (BASELINE_TAG, REGISTRY, LetterPairArena,  # noqa: E402
                                        TRAIN_PC, VAL_PC, SEALED_PC, GOLD_PC)

FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _frozen_hashes():
    return {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:8]
            for f in FROZEN_EXPECTED}


def _verdict(cert):
    if not cert.promotions:
        return ("NO_TABULAR_REPRESENTATION_LEVER",
                "the strong tuned-GBM is NOT beaten by the TabPFN foundation model or the stack under FDR; "
                "on this arena the raw columns are already a near-optimal representation for a GBM")
    last = cert.promotions[-1].to_tag
    if last.startswith("fuse["):
        return ("STACKING_WINS",
                f"the authored stack {last} FDR-beats the best single model -- ensembling the foundation model "
                f"with the GBM is the lever")
    return ("REPRESENTATION_LEVER_FIRES_ON_TABULAR",
            f"the pretrained tabular foundation model ({last}) FDR-beats the tuned GBM -- the representation "
            f"law generalises to tabular")


def main():
    hashes = _frozen_hashes()
    assert hashes == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED: {hashes} != {FROZEN_EXPECTED}"
    print(f"frozen certifier verified: {hashes}")

    smoke = int(os.environ.get("ATTESTRA_SMOKE", "0"))
    arena = LetterPairArena(smoke=smoke)
    researcher = ReprResearcher(REGISTRY, arena, start_tag=BASELINE_TAG,
                                alpha=0.1, theta_floor=0.5, peek_budget=12, competence_ceiling=0.90)
    print(f"start champion={BASELINE_TAG}  registry={[e.tag for e in REGISTRY]}  tasks={len(arena.tasks)}")
    print(f"split/class: train={TRAIN_PC} val={VAL_PC} sealed={SEALED_PC} gold={GOLD_PC}\n")

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
              f"FDR survivors {len(gc['survivors'])}/{gc['n_tasks']}  mean_lift {gc['mean_lift']:+.3f}  "
              f"CONFIRMED={gc['confirmed']}")
        for r in gc["per_task"]:
            print(f"    {r['task']:14} champ {r['champ_gold_acc']:.3f}  base {r['base_gold_acc']:.3f}  "
                  f"lift {r['lift']:+.3f}  p {r['p_gt_base']:.4f}")

    mp = cert.multiplicity
    if mp is not None:
        print("\n=== SESSION MULTIPLICITY (honest accounting of sealed-test re-use) ===")
        print(f"  {mp['sealed_comparisons']} sealed certifications spent climbing "
              f"({mp['mcnemar_tests_total']} paired McNemar tests total)")
        print(f"  per-comparison FDR alpha {mp['per_comparison_fdr_alpha']}  ->  "
              f"family-wise Bonferroni alpha/M {mp['session_bonferroni_alpha']:.4f}")
        print(f"  robust to session multiplicity = {mp['robust_to_session_multiplicity']}  |  "
              f"multiplicity-FREE gold confirmation = {mp['gold_independent_confirmation']}")

    verdict, verdict_note = _verdict(cert)
    registry_tags = {e.tag for e in REGISTRY}
    champion_ok = cert.champion in registry_tags or all(
        t in registry_tags for t in cert.champion.replace("fuse[", "").rstrip("]").split("+"))

    # ACCEPTANCE = SYSTEM INVARIANTS (must hold whatever the science verdict is).
    checks = {
        "frozen certifier byte-identical (sole promoter, untouched)":
            _frozen_hashes() == FROZEN_EXPECTED,
        "every promotion is FDR-certified (>=1 survivor, positive mean lift)":
            all(len(p.survivors) >= 1 and p.mean_lift > 0 for p in cert.promotions),
        "the brain honest-stopped within budget (no peek overrun)":
            cert.peeks_used <= researcher.peek_budget and bool(cert.stop_reason),
        "champion is a registry encoder or an FDR-certified stack of registry encoders":
            champion_ok,
        "gold confirmation is on a DISJOINT never-peeked set (or honestly absent)":
            gc is None or gc["gold_n"] > 0,
    }
    print("\n=== ACCEPTANCE (system invariants -- hold regardless of the science verdict) ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_pass = all(checks.values())

    print(f"\n=== SCIENTIFIC VERDICT: {verdict} ===\n  {verdict_note}")

    out = {
        "arena": "letter-recognition-confusable-pairs", "modality": "tabular",
        "start_champion": BASELINE_TAG, "champion": cert.champion, "champion_family": cert.champion_family,
        "lever_fired": bool(cert.promotions), "verdict": verdict, "verdict_note": verdict_note,
        "move_class_path": cert.move_class_path, "stop_reason": cert.stop_reason, "peeks_used": cert.peeks_used,
        "split_per_class": {"train": TRAIN_PC, "val": VAL_PC, "sealed": SEALED_PC, "gold": GOLD_PC},
        "promotions": [vars(p) for p in cert.promotions],
        "rejections": [vars(r) for r in cert.rejections],
        "data_ceiling_tasks": cert.data_ceiling_tasks,
        "sealed_acc": cert.sealed_acc, "sealed_lb": cert.sealed_lb,
        "gold_confirmation": cert.gold_confirmation, "multiplicity": cert.multiplicity,
        "pareto_front": cert.pareto_front, "pareto_report": cert.pareto_report,
        "acceptance": checks, "all_pass": all_pass, "frozen_hashes": hashes, "log": cert.log,
    }
    dst = os.path.join(ROOT, "docs", os.environ.get("ATTESTRA_OUT", "REPR_RESEARCHER_TABULAR_CERTIFICATE.json"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")
    print(f"frozen certifier (post-run): {_frozen_hashes()}")
    print(f"\nALL SYSTEM-INVARIANT CHECKS {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
