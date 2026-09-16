"""End-to-end CORE demo: CoreOrchestrator on a real sklearn dataset, fully OFFLINE (no LLM).

Run:
    /Users/abdullahalghamdi/jax-env-311/bin/python frontier/core/demo_core.py

This exercises the WHOLE core loop with ``llm_client=None`` (the honest offline floor): router
-> harness self-test -> Task -> knowledge warm-start -> [features + neural-template + seed/
mutation] proposals -> LinUCB rank -> budget-admitted ASHA portfolio -> diagnosis feed-forward
-> certify the winner ONCE on the sealed test -> oracles.verify_before_promote ANDed in ->
report. It prints the frozen certificate and the orchestrator's provenance.

It is a DEMO, not a benchmark: theta is the operator's verification standard passed in here, not
a number reverse-engineered from any reference solution (scientific-integrity rule).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier.core.orchestrator import CoreOrchestrator, CoreConfig  # noqa: E402


def main() -> int:
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    X, y = d.data, d.target.astype(str)

    # theta is the operator's promotion bar (the verification standard). 0.90 accuracy is a
    # demonstration threshold supplied by the caller, NOT derived from the answer key.
    cfg = CoreConfig(rounds=3, seed=0, llm_client=None, wall_seconds=45.0, cpu_seconds=40,
                     enable_neural=True, enable_knowledge=True)
    orch = CoreOrchestrator(cfg)
    result = orch.run(goal="classify breast-tumor samples as malignant or benign",
                      X=X, y=y, theta=0.90, name="breast_cancer")

    print("=" * 78)
    print("CORE ORCHESTRATOR RESULT (offline / no LLM)")
    print("=" * 78)
    print(result.summary())
    print("-" * 78)
    print(f"llm_active           : {result.llm_active}")
    print(f"candidates evaluated : {len(result.history)} "
          f"(ok={sum(1 for r in result.history if r.ok)})")
    print(f"sealed peeks         : {result.sealed_peeks}  (must be exactly 1 after the search)")
    if result.certificate is not None:
        c = result.certificate
        print(f"certificate          : observed={c.get('observed')} "
              f"lower_bound={c.get('lower_bound')} theta={c.get('theta')} "
              f"peeks={c.get('peeks')} certified={c.get('certified')}")
    if result.oracle_verdict is not None:
        print(f"oracle promote       : {result.oracle_verdict.get('promote')}")
        for o in result.oracle_verdict.get("oracles", []):
            mark = "ok" if o.get("passed") else "XX"
            print(f"   [{mark}] {o.get('name')}: {o.get('detail')}")
    if result.self_consistency:
        print(f"self-consistency     : {result.self_consistency.get('note')}")
    if result.backend_notes:
        print("backend notes        :")
        for n in result.backend_notes:
            print(f"   - {n}")
    print("=" * 78)
    print("OUTCOME:", "CERTIFIED" if result.certified else f"DECLINED ({result.decline_reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
