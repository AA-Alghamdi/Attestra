"""RUN the autonomous /goal front door from the command line.

Give it a free-text GOAL and a DATA pointer; it acquires the data, infers the certifiable spec (or honestly
DECLINES an out-of-scope goal), frames a TabularCodeArena, drives the literature-grounded regenerative loop
under the FROZEN Tier-3 certifier, audits the certificate's numbers through the NumericSubstrate, and prints
one GoalCertificate. CPU-only and deterministic given --seed when --online is off (the default).

  # deterministic, offline (reproducible) -- the default:
  python scripts/run_goal.py --goal "classify the wine cultivar" --data wine

  # a regression goal on a bundled set, more sealed budget:
  python scripts/run_goal.py --goal "predict disease progression" --data diabetes --peeks 12

  # your own data (csv: target is the named column or the last column), no literature scout:
  python scripts/run_goal.py --goal "detect fraud" --data /path/to/rows.csv --target is_fraud --no-literature

  # open-ended (Claude refines the spec + authors code + proposes motifs via the audited harnesses):
  python scripts/run_goal.py --goal "..." --data covtype --llm --online

The frozen certifier core is never modified (science.py b564fba2 / sealed.py 30ad6245); see
docs/GOAL_SOLVER.md and docs/NUMERIC_SUBSTRATE.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import goal_solver as G  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="autonomous /goal -> certified model (or honest decline)")
    ap.add_argument("--goal", required=True, help="free-text problem description")
    ap.add_argument("--data", required=True,
                    help="data pointer: a bundled sklearn name (wine/digits/breast_cancer/diabetes/california/"
                         "iris/covtype), an OpenML-CC18 member, openml://<id>, a .npz, or a .csv path")
    ap.add_argument("--target", default=None, help="(csv only) target column name; default = last column")
    ap.add_argument("--peeks", type=int, default=16, help="sealed peek budget for the loop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fanout", type=int, default=6, help="children proposed per expansion")
    ap.add_argument("--online", action="store_true", help="allow live literature retrieval + zoo discovery")
    ap.add_argument("--llm", action="store_true", help="enable the audited, non-binding LLM (spec/code/motifs)")
    ap.add_argument("--no-literature", dest="literature", action="store_false",
                    help="disable the LiteratureScout grounding")
    ap.add_argument("--no-code", dest="code", action="store_false",
                    help="disable authored code patches (the open-ended lever on raw features)")
    ap.add_argument("--out", default=None, help="optional path to write the certificate JSON")
    ap.add_argument("--gpu", action="store_true",
                    help="recruit a torch-MLP head (trains on cuda when present, else CPU)")
    args = ap.parse_args()

    cert = G.solve(args.goal, args.data, peeks=args.peeks, seed=args.seed, use_literature=args.literature,
                   online=args.online, use_llm=args.llm, code=args.code, target=args.target,
                   fanout=args.fanout, gpu=args.gpu)

    spec = cert.spec
    print(f"== /goal: {cert.goal!r} ==")
    print(f"   spec: kind={spec.get('kind')} task={spec.get('task_type')} shape={spec.get('arena_shape')} "
          f"split={spec.get('split')} metric={spec.get('metric')} n_classes={spec.get('n_classes')} "
          f"source={spec.get('source')}")
    if cert.declined:
        print(f"   DECLINED (out of scope): {cert.decline_reason}")
        return _finish(cert, args.out)
    if cert.refused:
        print(f"   REFUSED (framing/data gate): {cert.stop_reason}")
        return _finish(cert, args.out)

    verdict = "SOLVED" if cert.solved else "NOT SOLVED (honest)"
    lift = " + IMPROVED past seed" if cert.improved else ""
    print(f"   {verdict}{lift}: champion = {cert.champion!r}")
    print(f"   theta_floor={cert.theta_floor}  pooled_sealed_lb={cert.pooled_sealed_lb}  "
          f"peeks_used={cert.peeks_used}  stop={cert.stop_reason!r}")
    print(f"   sealed_acc={cert.sealed_acc}")
    if cert.gold_confirmation:
        print(f"   gold_confirmation={cert.gold_confirmation}")
    na = cert.numeric_audit
    print(f"   numeric_audit: clean={na.get('clean')} firewall_held={na.get('firewall_held')} "
          f"single_source_of_truth={na['single_source_of_truth']['agreement']} "
          f"firewall_selftest={na['firewall_selftest']['verdict']}/"
          f"refused={na['firewall_selftest']['refused_by_firewall']}")
    lit = cert.literature or {}
    print(f"   literature: menu_free={lit.get('menu_free')} grounded={lit.get('literature_grounded')} "
          f"uses_authored_code={lit.get('uses_authored_code')}")
    return _finish(cert, args.out)


def _finish(cert, out) -> int:
    if out:
        with open(out, "w") as fh:
            json.dump(cert.to_dict(), fh, indent=2, default=str)
        print(f"   wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
