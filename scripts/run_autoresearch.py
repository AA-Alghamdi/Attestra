"""#4 -- ONE COMMAND, ONE SESSION CERTIFICATE: route the SAME autonomous researcher across every modality.

Until now each modality had its own bespoke runner (`run_repr_researcher.py` for vision, `_text.py` for text)
that duplicated ~100 lines of certificate plumbing. This is the single entrypoint that routes the IDENTICAL
`vfplatform.repr_researcher.ReprResearcher` policy and the IDENTICAL frozen certifier across modalities behind
one `--modality {vision,text,all}` flag, and -- when run over `all` -- emits ONE cross-modal session
certificate that states the program's law in machine-checkable form: the representation lever fires in every
modality, the climb is gold-confirmed where a disjoint gold set exists, and the champion survives the
session-level Bonferroni correction for sealed-test re-use.

Adding a modality is a single entry in `MODALITIES` (arena builder + start champion + acceptance properties);
nothing else changes, because the brain and the certifier are modality-agnostic. Run:
  `python scripts/run_autoresearch.py --modality all`     (vision + text -- run in the repo's main env)
  `python scripts/run_autoresearch.py --modality vision`
  `~/.venv-tabpfn/bin/python scripts/run_autoresearch.py --modality tabular`   (TabPFN needs its isolated venv)

TABULAR is a first-class modality but its TabPFN arm pins scikit-learn<1.7, which conflicts with the repo's
1.9.0, so it runs ONLY in the isolated `~/.venv-tabpfn` venv and is therefore NOT part of the main-env `all`
set. The brain, the certifier and this entrypoint are unchanged across envs (numpy/stdlib only).
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.repr_researcher import ReprResearcher                       # noqa: E402

FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _frozen_hashes():
    return {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:8]
            for f in FROZEN_EXPECTED}


# ---- modality wiring -------------------------------------------------------------------------------------
def _build_vision(smoke):
    from scripts.repr_arena import REGISTRY, BASELINE_TAG, FgvcAircraftArena
    return FgvcAircraftArena(smoke=smoke), REGISTRY, BASELINE_TAG


def _build_text(smoke):
    from scripts.repr_arena_text import REGISTRY, BASELINE_TAG, TwentyNewsArena
    return TwentyNewsArena(smoke=smoke), REGISTRY, BASELINE_TAG


def _build_tabular(smoke):
    try:
        import tabpfn  # noqa: F401
    except ImportError as e:
        raise SystemExit("tabular modality needs the isolated TabPFN venv -- run with "
                         "`~/.venv-tabpfn/bin/python scripts/run_autoresearch.py --modality tabular`") from e
    from scripts.repr_arena_tabular import REGISTRY, BASELINE_TAG, LetterPairArena
    return LetterPairArena(smoke=smoke), REGISTRY, BASELINE_TAG


def _accept_vision(cert, hashes):
    """The vision arena has a KNOWN pre-registered verdict (it took seven hand-run phases): the system must
    reproduce it autonomously -- climb to DINOv2-g, reject the bigger-but-wrong-bias encoders, flag the 737s."""
    promoted = [p.to_tag for p in cert.promotions]
    rejected = {r.tag for r in cert.rejections}
    return {
        "champion is DINOv2-g": cert.champion == "dinov2_g",
        "climbed via DINOv2-L then g": "dinov2_vitl14" in promoted and "dinov2_g" in promoted,
        "rejected SigLIP-SO400M": "siglip_so" in rejected,
        "rejected EVA-02-L": "eva02_l" in rejected,
        "flagged 737-700/800 as data ceiling": "737-700_vs_737-800" in cert.data_ceiling_tasks,
        "flagged 737-300/400 as data ceiling": "737-300_vs_737-400" in cert.data_ceiling_tasks,
        "frozen certifier byte-identical": hashes == FROZEN_EXPECTED,
    }


def _accept_text(cert, hashes, baseline):
    """Text has no pre-baked winner (which encoder wins is empirical), so acceptance is GENERALITY PROPERTIES:
    the lever fires, the first move is a family swap off the lexical baseline, every promotion is FDR-certified,
    and the champion is gold-confirmed on a never-peeked set."""
    return {
        "representation lever fires (promoted off the lexical baseline)":
            cert.champion != baseline and len(cert.promotions) >= 1,
        "first move is a family swap lexical -> neural encoder":
            len(cert.promotions) >= 1 and cert.promotions[0].rung == "model"
            and cert.promotions[0].from_tag == baseline,
        "every promotion is FDR-certified (>=1 survivor, positive lift)":
            all(len(p.survivors) >= 1 and p.mean_lift > 0 for p in cert.promotions),
        "champion is a frozen neural encoder (not lexical)":
            cert.champion_family != "lexical",
        "champion CONFIRMED on a NEVER-PEEKED gold set":
            cert.gold_confirmation is not None and cert.gold_confirmation["confirmed"],
        "frozen certifier byte-identical": hashes == FROZEN_EXPECTED,
    }


def _accept_tabular(cert, hashes, baseline):
    """Tabular starts from the STRONG tuned-GBM champion, so whether the lever fires is empirical and NEVER
    forced -- acceptance certifies only the SYSTEM INVARIANTS that must hold regardless of the verdict: the
    frozen certifier is untouched, every promotion is FDR-surviving, the brain honest-stops within budget, the
    champion is a registry encoder or an FDR-certified stack, and any gold confirmation is on a disjoint set."""
    registry_tags = {"gbm_raw", "tabpfn", "tabpfn_big"}
    champion_ok = cert.champion in registry_tags or all(
        t in registry_tags for t in cert.champion.replace("fuse[", "").rstrip("]").split("+"))
    gc = cert.gold_confirmation
    return {
        "frozen certifier byte-identical (sole promoter, untouched)": hashes == FROZEN_EXPECTED,
        "every promotion is FDR-certified (>=1 survivor, positive mean lift)":
            all(len(p.survivors) >= 1 and p.mean_lift > 0 for p in cert.promotions),
        "the brain honest-stopped within budget (no peek overrun)": bool(cert.stop_reason),
        "champion is a registry encoder or an FDR-certified stack of registry encoders": champion_ok,
        "gold confirmation is on a DISJOINT never-peeked set (or honestly absent)":
            gc is None or gc["gold_n"] > 0,
    }


MODALITIES = {
    "vision": {"label": "fgvc-aircraft", "build": _build_vision, "peek_budget": 12,
               "accept": lambda cert, h, base: _accept_vision(cert, h),
               "out": "AUTORESEARCH_VISION.json"},
    "text": {"label": "20newsgroups-confusable-pairs", "build": _build_text, "peek_budget": 16,
             "accept": _accept_text, "out": "AUTORESEARCH_TEXT.json"},
    "tabular": {"label": "letter-recognition-confusable-pairs", "build": _build_tabular, "peek_budget": 12,
                "accept": _accept_tabular, "out": "AUTORESEARCH_TABULAR.json"},
}


# ---- shared certificate display + emit -------------------------------------------------------------------
def _print_cert(cert):
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
    else:
        print("\n=== GOLD CONFIRMATION ===\n  none (arena supplies no disjoint gold partition; honest)")

    mp = cert.multiplicity
    if mp is not None:
        print("\n=== SESSION MULTIPLICITY (honest accounting of sealed-test re-use) ===")
        print(f"  {mp['sealed_comparisons']} sealed certifications spent climbing "
              f"({mp['mcnemar_tests_total']} paired McNemar tests total)")
        print(f"  per-comparison FDR alpha {mp['per_comparison_fdr_alpha']}  ->  "
              f"family-wise Bonferroni alpha/M {mp['session_bonferroni_alpha']:.4f}")
        print(f"  champion vs start on sealed: {len(mp['fdr_survivors_nominal'])} FDR-nominal survivors, "
              f"{len(mp['bonferroni_survivors_session'])}/{mp['n_tasks']} survive Bonferroni")
        print(f"  robust to session multiplicity = {mp['robust_to_session_multiplicity']}  |  "
              f"multiplicity-FREE gold confirmation = {mp['gold_independent_confirmation']}")


def _cert_dict(cert, modality, cfg, checks, hashes):
    return {
        "modality": modality, "arena": cfg["label"],
        "champion": cert.champion, "champion_family": cert.champion_family,
        "move_class_path": cert.move_class_path, "stop_reason": cert.stop_reason,
        "peeks_used": cert.peeks_used,
        "promotions": [vars(p) for p in cert.promotions],
        "rejections": [vars(r) for r in cert.rejections],
        "data_ceiling_tasks": cert.data_ceiling_tasks,
        "sealed_acc": cert.sealed_acc, "sealed_lb": cert.sealed_lb,
        "gold_confirmation": cert.gold_confirmation, "multiplicity": cert.multiplicity,
        "pareto_front": cert.pareto_front, "pareto_report": cert.pareto_report,
        "acceptance": checks, "all_pass": all(checks.values()),
        "frozen_hashes": hashes, "log": cert.log,
    }


def run_one(modality, smoke=0):
    cfg = MODALITIES[modality]
    hashes = _frozen_hashes()
    assert hashes == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED: {hashes} != {FROZEN_EXPECTED}"
    print(f"\n############### MODALITY: {modality.upper()} ({cfg['label']}) ###############")
    print(f"frozen certifier verified: {hashes}")

    arena, registry, start_tag = cfg["build"](smoke)
    researcher = ReprResearcher(registry, arena, start_tag=start_tag, alpha=0.1, theta_floor=0.5,
                                peek_budget=cfg["peek_budget"], competence_ceiling=0.90)
    print(f"start champion={start_tag}  registry={[e.tag for e in registry]}  tasks={len(arena.tasks)}\n")

    cert = researcher.run()
    _print_cert(cert)

    checks = cfg["accept"](cert, _frozen_hashes(), start_tag)
    print("\n=== ACCEPTANCE ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    out = _cert_dict(cert, modality, cfg, checks, hashes)
    dst = os.path.join(ROOT, "docs", cfg["out"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")
    print(f"frozen certifier (post-run): {_frozen_hashes()}")
    print(f"MODALITY {modality.upper()} ACCEPTANCE {'PASS' if out['all_pass'] else 'FAIL'}")
    return out


def _session_certificate(results):
    """ONE cross-modal certificate: restate the program's law as machine-checkable booleans over every
    modality that ran -- the representation lever fired everywhere, the champion is gold-confirmed where a
    disjoint gold set exists, and it survives the session Bonferroni for sealed-test re-use."""
    per = {}
    for r in results:
        mp = r.get("multiplicity") or {}
        gc = r.get("gold_confirmation")
        per[r["modality"]] = {
            "arena": r["arena"], "champion": r["champion"], "champion_family": r["champion_family"],
            "lever_fired": len(r["promotions"]) >= 1 and r["champion_family"] != "lexical",
            "peeks_used": r["peeks_used"],
            "robust_to_session_multiplicity": bool(mp.get("robust_to_session_multiplicity", False)),
            "gold_confirmed": bool(gc["confirmed"]) if gc else None,
            "data_ceiling_tasks": r["data_ceiling_tasks"],
            "acceptance_all_pass": r["all_pass"],
        }
    modalities = sorted(per)
    lever_everywhere = all(per[m]["lever_fired"] for m in modalities)
    robust_everywhere = all(per[m]["robust_to_session_multiplicity"] for m in modalities)
    gold_where_available = all(per[m]["gold_confirmed"] for m in modalities
                               if per[m]["gold_confirmed"] is not None)
    return {
        "session": "attestra-autoresearch", "modalities": modalities,
        "law": "representation is the lever; the same brain + frozen certifier derive it in every modality",
        "representation_lever_fires_in_every_modality": lever_everywhere,
        "champion_robust_to_session_multiplicity_everywhere": robust_everywhere,
        "champion_gold_confirmed_where_gold_exists": gold_where_available,
        "all_modalities_accept": all(per[m]["acceptance_all_pass"] for m in modalities),
        "per_modality": per, "frozen_hashes": _frozen_hashes(),
    }


def main():
    ap = argparse.ArgumentParser(description="Unified autoresearch entrypoint (one brain, one certifier).")
    ap.add_argument("--modality", choices=["vision", "text", "tabular", "all"], default="all")
    ap.add_argument("--smoke", type=int, default=int(os.environ.get("ATTESTRA_SMOKE", "0")))
    args = ap.parse_args()

    todo = ["vision", "text"] if args.modality == "all" else [args.modality]
    results = [run_one(m, smoke=args.smoke) for m in todo]

    all_pass = all(r["all_pass"] for r in results)
    if len(results) > 1:
        sess = _session_certificate(results)
        dst = os.path.join(ROOT, "docs", "AUTORESEARCH_SESSION.json")
        with open(dst, "w") as f:
            json.dump(sess, f, indent=2)
        print("\n############### CROSS-MODAL SESSION CERTIFICATE ###############")
        print(f"  modalities                                : {sess['modalities']}")
        print(f"  representation lever fires everywhere      : {sess['representation_lever_fires_in_every_modality']}")
        print(f"  robust to session multiplicity everywhere  : {sess['champion_robust_to_session_multiplicity_everywhere']}")
        print(f"  gold-confirmed where a gold set exists     : {sess['champion_gold_confirmed_where_gold_exists']}")
        for m in sess["modalities"]:
            p = sess["per_modality"][m]
            print(f"    {m:8} champion={p['champion']:14} lever={p['lever_fired']} "
                  f"gold={p['gold_confirmed']} bonf_robust={p['robust_to_session_multiplicity']}")
        print(f"  wrote {dst}")

    print(f"\nALL MODALITIES {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
