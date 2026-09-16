"""Literature-retrieval ablation: does real paper retrieval measurably change the
search prompt the engine builds for each proposal round?

The auditor's concern was that the "literature scout" only emits technique *names*
from LLM recall, with no actual paper retrieval/reading. This ablation runs the
real retrieval path (vfplatform.literature -> arXiv / HuggingFace / Papers-with-Code,
use_llm=False so nothing is hallucinated) and measures, for each goal, the delta
between the proposal context built WITH retrieval enabled vs WITH it disabled:

  - n_findings           : real papers/cards retrieved
  - techniques           : distilled technique motifs injected
  - architectures        : candidate backbones injected
  - key_insights         : real "[source] title: abstract" lines injected
  - llm_guidance_chars   : characters of retrieved context added to the search prompt
  - titles               : a sample of the actual retrieved paper/card titles

Data-driven proposers (feature-eng / ensemble / ASHA-HP) are literature-independent
by design, so the measurable effect of retrieval is precisely this prompt-context
delta -- which is what conditions every LLM/NAS proposal downstream.

Run:  python scripts/ablation_literature.py
Network access is required for live retrieval; offline it honestly reports 0
findings and the script documents that retrieval degraded gracefully.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from frontier.intelligence import (  # noqa: E402
    init_intelligence,
    enrich_round_context,
)
from frontier.task import Task  # noqa: E402


GOALS = [
    "classify imbalanced multiclass tabular data with mixed categorical/numeric features",
    "forecast a noisy univariate time series with regime shifts",
    "image classification on small medical datasets with limited labels",
]


def _toy_task() -> Task:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return Task(X=X, y=y, kind="classification", theta=0.5)


def _measure(goal: str, task: Task, *, enable: bool) -> dict:
    t0 = time.time()
    state = init_intelligence(goal, task, llm_client=None, enable_literature=enable)
    ctx: dict = {}
    enrich_round_context(ctx, state, task, round_idx=0, history=[])
    lit = ctx.get("literature", {}) or {}
    guidance = ctx.get("llm_guidance", "") or ""
    # isolate the literature contribution to the guidance prompt
    lit_chars = 0
    if "[LITERATURE]" in guidance:
        seg = guidance[guidance.index("[LITERATURE]"):]
        # cut at the next bracketed section header if present
        for header in ("\n[FEATURE ENGINEERING]", "\n[KNOWLEDGE BASE", "\n[EVOLUTION"):
            if header in seg:
                seg = seg[: seg.index(header)]
        lit_chars = len(seg)
    titles = []
    src = state.literature
    if src is not None:
        for ins in src.key_insights[:5]:
            titles.append(ins[:120])
    return {
        "n_findings": int(lit.get("n_findings", 0)),
        "techniques": len(lit.get("techniques", [])),
        "architectures": len(lit.get("architectures", [])),
        "key_insights": len(lit.get("key_insights", [])),
        "llm_guidance_chars": lit_chars,
        "titles": titles,
        "secs": round(time.time() - t0, 1),
    }


def main() -> int:
    task = _toy_task()
    report = []
    for goal in GOALS:
        on = _measure(goal, task, enable=True)
        off = _measure(goal, task, enable=False)
        report.append({"goal": goal, "with_retrieval": on, "without_retrieval": off})
        print(f"\n=== {goal}")
        print(f"  WITH retrieval   : findings={on['n_findings']:>3}  techniques={on['techniques']}  "
              f"architectures={on['architectures']}  insights={on['key_insights']}  "
              f"guidance_chars={on['llm_guidance_chars']:>5}  ({on['secs']}s)")
        print(f"  WITHOUT retrieval: findings={off['n_findings']:>3}  techniques={off['techniques']}  "
              f"architectures={off['architectures']}  insights={off['key_insights']}  "
              f"guidance_chars={off['llm_guidance_chars']:>5}")
        for t in on["titles"]:
            print(f"      retrieved: {t}")

    total_on = sum(r["with_retrieval"]["n_findings"] for r in report)
    total_chars = sum(r["with_retrieval"]["llm_guidance_chars"] for r in report)
    print("\n" + "=" * 70)
    print(f"TOTAL real findings injected across {len(GOALS)} goals: {total_on}")
    print(f"TOTAL literature characters added to search prompts:   {total_chars}")
    if total_on == 0:
        print("NOTE: 0 findings -> retrieval is offline/unreachable here; it degraded "
              "gracefully (no fabricated content). Re-run with network access for live counts.")
    out = _ROOT / "scripts" / "ablation_literature_results.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
