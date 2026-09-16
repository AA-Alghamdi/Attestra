"""Multiclass regression guard: the canonical voice-agent tool-router task (10 intents) must run end to
end and certify. This locks in what was previously listed as 'multiclass untested'.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vectorforge as vf

BANK = Path("/Users/abdullahalghamdi/vectorforge-harnesses/tool-router/data")


def loadj(n):
    return [json.loads(l) for l in (BANK / f"{n}.jsonl").read_text().splitlines() if l.strip()]


def main():
    raw = loadj("train") + loadj("validation") + loadj("test")
    labels = sorted({r["target"] for r in raw})
    assert len(labels) >= 3, "expected a multiclass task"
    g = vf.create_goal(name="mc_tool_router_test", kind="text", labels=labels, raw_rows=raw,
                       metric="accuracy", threshold=0.80, max_latency_ms=50.0, min_heldout_n=200)
    g = vf.run(vf.approve(vf.draft_plan(g)).id)

    checks = []
    checks.append(("status passed", g.status == "passed"))
    checks.append(("certified", bool(g.certificate and g.certificate["decision"] == "certified")))
    checks.append(("lower bound clears threshold", g.certificate["lower_bound"] > g.verification.threshold))
    checks.append(("leakage clean", g.certificate["leakage_passed"]))
    checks.append(("latency within budget", g.certificate["latency_ms_p95"] <= g.verification.max_latency_ms))
    checks.append(("deployed + predicts", vf.predict(g.id, "please end the call now") in labels))
    checks.append(("per-class evidence present", bool((g.research_state or {}).get("decisions"))))

    ok = all(v for _, v in checks)
    for name, v in checks:
        print(f"  [{'PASS' if v else 'FAIL'}] {name}")
    print(f"  certificate: acc={g.certificate['observed']} lb={g.certificate['lower_bound']} "
          f"n={g.certificate['n']} latency={g.certificate['latency_ms_p95']}ms")
    print(f"RESULT: {'PASS' if ok else 'FAIL'}  ({sum(v for _, v in checks)}/{len(checks)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
