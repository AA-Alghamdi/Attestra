"""Multi-level routing for the Attestra research pipeline.

Four routing levels:
  1. Problem Router: goal + data → problem_type
  2. Strategy Router: problem_type + data_profile + history → strategy_class
  3. Resource Router: strategy + cost → execution substrate
  4. Verification Router: problem_type + result → verification suite (oracle selection)
"""
from .strategy_router import StrategyRouter, StrategyClass
from .resource_router import ResourceRouter, ExecutionSubstrate
from .verification_router import VerificationRouter, VerificationSuite

__all__ = [
    "StrategyRouter", "StrategyClass",
    "ResourceRouter", "ExecutionSubstrate",
    "VerificationRouter", "VerificationSuite",
]
