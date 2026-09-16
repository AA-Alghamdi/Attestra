"""Live demo: run the spine on real sklearn datasets, end to end, with certification.

Run from the repo root with the sklearn-equipped interpreter:
    /Users/abdullahalghamdi/jax-env-311/bin/python -m frontier.demo

No LLM key is needed: the seed + mutation proposers make it generative offline. If you
wire an llm_client into EngineConfig, the LLMProposer activates automatically.
"""

from __future__ import annotations

import numpy as np

from .engine import ResearchEngine, EngineConfig
from .task import Task


def _classification_task() -> Task:
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    # theta = 0.90: a real bar the sealed LOWER bound must clear, not just the point estimate.
    return Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.90,
                name="breast_cancer")


def _regression_task() -> Task:
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    return Task(X=d.data, y=d.target.astype(float), kind="regression", theta=0.40,
                name="diabetes_r2")


def main():
    for builder in (_classification_task, _regression_task):
        task = builder()
        print("=" * 72)
        print(f"TASK: {task.name}  kind={task.kind}  metric={task.metric}  "
              f"theta={task.theta}  n={len(task.y)}  d={task.n_features}")
        engine = ResearchEngine(EngineConfig(rounds=3, wall_seconds=45, cpu_seconds=40))
        result = engine.run(task)
        print(result.summary())
        print("diagnosis trail:")
        for step in result.diagnosis_trail:
            print(f"  round {step['round']}: proposed {step['n_proposals']}, "
                  f"best {step.get('best_in_label')}({step.get('best_in_score')}) "
                  f"-> {step.get('best_out_label')}({step.get('best_out_score')})")
        print(f"split: {result.split_meta.get('counts')}  "
              f"leakage_dropped={result.split_meta.get('leakage_dropped')}")


if __name__ == "__main__":
    main()
