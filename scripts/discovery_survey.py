"""Honest cross-dataset discovery survey: run the battery over a heterogeneous set of REAL OpenML+HF tasks
(binary/multiclass/regression x tabular/text, easy/hard/near-separable) and report the BH-FDR-controlled
discovery rate -- i.e. on how many tasks the autoresearch search BEATS the no-search baseline with paired
significance. Per-shard theta is set by the battery's pilot (theta = max(registry_floor, measured baseline)),
so a "discovery" means the certified winner genuinely beats the baseline, not that it cleared an arbitrary floor.

This is the deliverable for Tier 1's "battery across many real shards" item. It NEVER promotes a model; the
frozen certifier remains the only promoter. Honest negatives are first-class output.

Run:  PYTHONPATH=. /Users/abdullahalghamdi/jax-env-311/bin/python scripts/discovery_survey.py [registry.json] [alpha]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.battery import run_battery


def main():
    registry = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "survey_registry.json")
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    tasks = json.load(open(registry))
    print(f"=== discovery survey: {len(tasks)} real tasks, alpha={alpha} (BH-FDR over candidate-vs-baseline) ===",
          flush=True)

    def on_task(r):
        st = r.get("status")
        print(f"  [{st:5s}] {r['task_id']:26s} decision={str(r.get('decision')):14s} "
              f"observed={r.get('observed')} lift={r.get('lift_over_baseline')} p={r.get('p_value')}"
              + (f"  ERROR={r.get('error')}" if st == "error" else ""), flush=True)

    rep = run_battery(tasks, alpha=alpha, on_task=on_task)
    ok = [t for t in rep["tasks"] if t.get("status") == "ok"]
    errs = [t for t in rep["tasks"] if t.get("status") == "error"]
    print("\n=== SURVEY REPORT ===")
    print(f"  tasks run ok      : {len(ok)}/{rep['m']}   (errors: {len(errs)})")
    print(f"  certified         : {rep['n_certified']}")
    print(f"  FDR discoveries   : {rep['n_discoveries']}  -> {rep['discoveries']}")
    print(f"  null              : {rep['null']}")
    if errs:
        print(f"  errored shards    : {[t['task_id'] for t in errs]}")
    # honest framing: discovery rate over the tasks that actually ran
    if ok:
        rate = rep["n_discoveries"] / len(ok)
        print(f"  discovery rate    : {rep['n_discoveries']}/{len(ok)} = {rate:.0%} of runnable tasks "
              f"beat their baseline with BH-controlled significance")
    return rep


if __name__ == "__main__":
    main()
