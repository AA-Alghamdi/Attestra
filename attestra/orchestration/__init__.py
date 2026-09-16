"""Orchestration: strategy loop, portfolio fan-out, error taxonomy, health monitoring.

Entry points:
  - run_strategy_loop: The outer retry loop (Loop 2) with escalation
  - orchestrate: Single-attempt research run (Loop 1 inner cycle)
  - orchestrate_from_text: Convenience wrapper for text+data
"""
from .orchestrator import orchestrate, orchestrate_from_text, OrchestrateConfig, OrchestrateResult
from .strategy_loop import (
    run_strategy_loop, Strategy, StrategyDiagnosis,
    StrategyLoopResult, StrategyLoopConfig, ESCALATION_LADDER,
    diagnose_strategy_failure, escalate,
)

__all__ = [
    "orchestrate", "orchestrate_from_text", "OrchestrateConfig", "OrchestrateResult",
    "run_strategy_loop", "Strategy", "StrategyDiagnosis",
    "StrategyLoopResult", "StrategyLoopConfig", "ESCALATION_LADDER",
    "diagnose_strategy_failure", "escalate",
]
