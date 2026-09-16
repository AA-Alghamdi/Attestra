"""Self-improvement: learn from experiments across runs.

Components:
  - MetaLearner: ranks strategies based on past experience
  - StrategyLearner: records and retrieves strategy outcomes
  - SelfImprovement: prompt evolution, harness authoring, oracle evolution, strategy generation
"""
from .meta_learner import MetaLearner, Experience
from .strategy_learner import StrategyLearner, StrategyOutcome
from .self_improvement import (
    PromptEvolver, HarnessLibrary, OracleEvolver,
    StrategyGenerator, CompoundingMetrics,
)

__all__ = [
    "MetaLearner", "Experience",
    "StrategyLearner", "StrategyOutcome",
    "PromptEvolver", "HarnessLibrary", "OracleEvolver",
    "StrategyGenerator", "CompoundingMetrics",
]
