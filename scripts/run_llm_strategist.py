"""#6 -- IS THE LLM ACTUALLY A BETTER STRATEGIST? A falsifiable head-to-head on sealed-peek efficiency.

Runs the IDENTICAL representation-climbing brain on the IDENTICAL FGVC arena twice -- once with the deterministic
proposal policy, once with the Claude strategist (`LLMStrategist`) -- and compares (champion reached, sealed
certifications spent). The frozen certifier is the sole promoter in BOTH runs, and both share ONE memoized arena
so the only thing that differs is the proposal ORDER/PRUNING. The claim "the LLM is a better strategist" is
accepted iff it reaches the SAME certified champion in STRICTLY fewer sealed peeks; a tie or a regression (or the
LLM pruning the true winner and landing a WORSE champion) is reported as an honest negative.

  python scripts/run_llm_strategist.py            # full arena, real Claude calls (needs ANTHROPIC_API_KEY)
  ATTESTRA_SMOKE=2 python scripts/run_llm_strategist.py   # first 2 pairs, fast wiring check
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.llm_strategist import LLMStrategist, StrategistResearcher, DEFAULT_MODEL   # noqa: E402
from vfplatform.repr_researcher import ReprResearcher                                      # noqa: E402
from scripts.repr_arena import REGISTRY, BASELINE_TAG, FgvcAircraftArena                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}
SMOKE = int(os.environ.get("ATTESTRA_SMOKE", "0"))
PEEK_BUDGET = int(os.environ.get("ATTESTRA_PEEK_BUDGET", "12"))
ALPHA = 0.1
THETA_FLOOR = 0.5
COMPETENCE_CEILING = 0.90


def _frozen_hashes():
    return {f: hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest()[:8]
            for f in FROZEN_EXPECTED}


def _cfg(arena, **extra):
    return dict(registry=REGISTRY, arena=arena, start_tag=BASELINE_TAG, alpha=ALPHA,
                theta_floor=THETA_FLOOR, peek_budget=PEEK_BUDGET,
                competence_ceiling=COMPETENCE_CEILING, **extra)


def _summary(cert):
    return {"champion": cert.champion, "champion_family": cert.champion_family,
            "peeks_used": cert.peeks_used,
            "promotions": [f"{p.from_tag}->{p.to_tag}" for p in cert.promotions],
            "move_class_path": cert.move_class_path, "stop_reason": cert.stop_reason,
            "data_ceiling_tasks": cert.data_ceiling_tasks}


def _verdict(det, llm):
    same = det.champion == llm.champion
    if same and llm.peeks_used < det.peeks_used:
        v = "LLM_MORE_EFFICIENT"
    elif same and llm.peeks_used == det.peeks_used:
        v = "TIE"
    elif same:
        v = "LLM_LESS_EFFICIENT"
    else:
        v = "LLM_REACHED_DIFFERENT_CHAMPION"
    return {"verdict": v, "same_champion": same,
            "peeks_deterministic": det.peeks_used, "peeks_llm": llm.peeks_used,
            "peeks_saved": det.peeks_used - llm.peeks_used,
            "llm_is_better_strategist": v == "LLM_MORE_EFFICIENT"}


def main():
    h0 = _frozen_hashes()
    assert h0 == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED: {h0} != {FROZEN_EXPECTED}"
    arena = FgvcAircraftArena(smoke=SMOKE)          # ONE memoized arena, shared by both runs
    print(f"[#6 strategist] tasks={len(arena.tasks)} start={BASELINE_TAG} budget={PEEK_BUDGET} "
          f"model={DEFAULT_MODEL}")

    print("[A] deterministic policy ...")
    det = ReprResearcher(**_cfg(arena)).run()
    print(f"    champion={det.champion} peeks={det.peeks_used} path={det.move_class_path}")

    print("[B] LLM strategist ...")
    strat = LLMStrategist()
    llm = StrategistResearcher(**_cfg(arena, strategist=strat)).run()
    print(f"    champion={llm.champion} peeks={llm.peeks_used} path={llm.move_class_path}")

    verdict = _verdict(det, llm)
    out = {
        "arena": "fgvc-aircraft", "smoke": SMOKE, "peek_budget": PEEK_BUDGET,
        "model": DEFAULT_MODEL, "start_tag": BASELINE_TAG,
        "deterministic": _summary(det), "llm": _summary(llm),
        "verdict": verdict,
        "llm_calls": strat.calls,
        "note": ("Both runs use the frozen certifier as sole promoter and a shared memoized arena; only the "
                 "proposal order/pruning differs. The strategist proposals are intersected with the policy's "
                 "legal set, so the LLM can only reorder/prune -- never inject an illegal encoder or promote."),
        "frozen_hashes": _frozen_hashes(),
    }
    name = "LLM_STRATEGIST_SMOKE.json" if SMOKE else "LLM_STRATEGIST_RESULT.json"
    dst = os.path.join(ROOT, "docs", name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== VERDICT: {verdict['verdict']}  (deterministic {det.peeks_used} peeks -> "
          f"LLM {llm.peeks_used} peeks; saved {verdict['peeks_saved']}; same champion={verdict['same_champion']})")
    for c in strat.calls:
        tag = c.get("shortlist", c.get("fallback", "?"))
        print(f"  [{c['move_class']}] champ={c['champion']} legal={c['legal']} -> {tag}"
              + (f"  ({c.get('rationale','')})" if c.get("rationale") else ""))
    print(f"wrote {dst}\nfrozen (post-run): {_frozen_hashes()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
