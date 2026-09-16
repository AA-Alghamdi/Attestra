"""Structured event emission for experiment monitoring.

Each round emits structured events that a dashboard can consume.
Events are JSON-serializable and append-only (for streaming).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    """Types of experiment events."""
    # Lifecycle
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_END = "experiment_end"
    # Strategy loop
    STRATEGY_ATTEMPT_START = "strategy_attempt_start"
    STRATEGY_ATTEMPT_END = "strategy_attempt_end"
    STRATEGY_ESCALATION = "strategy_escalation"
    # Inner loop
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    # Proposal
    PROPOSAL_GENERATED = "proposal_generated"
    PROPOSAL_EXECUTED = "proposal_executed"
    PROPOSAL_EVALUATED = "proposal_evaluated"
    # Certification
    ORACLE_CHECK = "oracle_check"
    CERTIFICATION_ATTEMPT = "certification_attempt"
    CERTIFICATION_RESULT = "certification_result"
    # Knowledge
    KNOWLEDGE_RETRIEVED = "knowledge_retrieved"
    KNOWLEDGE_RECORDED = "knowledge_recorded"
    # Diagnosis
    DIAGNOSIS_COMPLETE = "diagnosis_complete"
    # Error
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ExperimentEvent:
    """A single structured event from an experiment."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    # Context
    experiment_id: str = ""
    attempt: int = 0
    round_num: int = 0
    # Payload
    data: Dict = field(default_factory=dict)
    # Status snapshot
    best_score: float = 0.0
    budget_remaining_s: float = 0.0
    n_proposals_total: int = 0
    n_successful: int = 0

    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "experiment_id": self.experiment_id,
            "attempt": self.attempt,
            "round_num": self.round_num,
            "data": self.data,
            "best_score": self.best_score,
            "budget_remaining_s": self.budget_remaining_s,
            "n_proposals_total": self.n_proposals_total,
            "n_successful": self.n_successful,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class EventEmitter:
    """Emits structured events for experiment monitoring.

    Supports:
      - File sink (append-only JSONL)
      - Callback sink (for real-time streaming)
      - In-memory buffer (for testing / post-hoc analysis)
    """

    def __init__(
        self,
        experiment_id: str = "",
        file_path: Optional[str] = None,
        callback: Optional[Callable[[ExperimentEvent], None]] = None,
        buffer_size: int = 1000,
    ):
        self._experiment_id = experiment_id
        self._file_path = file_path
        self._callback = callback
        self._buffer: List[ExperimentEvent] = []
        self._buffer_size = buffer_size
        # State
        self._current_attempt = 0
        self._current_round = 0
        self._best_score = 0.0
        self._budget_remaining = 0.0
        self._n_proposals = 0
        self._n_successful = 0

        if self._file_path:
            os.makedirs(os.path.dirname(self._file_path), exist_ok=True)

    def emit(self, event_type: EventType, data: Optional[Dict] = None, **kwargs) -> ExperimentEvent:
        """Emit an event.

        Creates the event with current state context and dispatches
        to all configured sinks.
        """
        event = ExperimentEvent(
            event_type=event_type,
            experiment_id=self._experiment_id,
            attempt=self._current_attempt,
            round_num=self._current_round,
            data=data or {},
            best_score=self._best_score,
            budget_remaining_s=self._budget_remaining,
            n_proposals_total=self._n_proposals,
            n_successful=self._n_successful,
        )

        # Override with kwargs
        for k, v in kwargs.items():
            if hasattr(event, k):
                setattr(event, k, v)

        # Dispatch to sinks
        self._dispatch(event)
        return event

    def _dispatch(self, event: ExperimentEvent) -> None:
        """Send event to all configured sinks."""
        # In-memory buffer
        self._buffer.append(event)
        if len(self._buffer) > self._buffer_size:
            self._buffer = self._buffer[-self._buffer_size:]

        # File sink
        if self._file_path:
            try:
                with open(self._file_path, "a") as f:
                    f.write(event.to_json() + "\n")
            except Exception:
                pass

        # Callback sink
        if self._callback:
            try:
                self._callback(event)
            except Exception:
                pass

    # State updates (called by orchestrator)
    def set_attempt(self, attempt: int) -> None:
        self._current_attempt = attempt

    def set_round(self, round_num: int) -> None:
        self._current_round = round_num

    def update_score(self, score: float) -> None:
        self._best_score = max(self._best_score, score)

    def update_budget(self, remaining_s: float) -> None:
        self._budget_remaining = remaining_s

    def increment_proposals(self, n: int = 1, successful: int = 0) -> None:
        self._n_proposals += n
        self._n_successful += successful

    # Convenience methods
    def experiment_start(self, goal: str, config: Dict) -> ExperimentEvent:
        return self.emit(EventType.EXPERIMENT_START, {"goal": goal, "config": config})

    def experiment_end(self, decision: str, score: float) -> ExperimentEvent:
        return self.emit(EventType.EXPERIMENT_END, {"decision": decision, "final_score": score})

    def round_start(self, round_num: int) -> ExperimentEvent:
        self.set_round(round_num)
        return self.emit(EventType.ROUND_START, {"round": round_num})

    def round_end(self, round_num: int, best_this_round: float) -> ExperimentEvent:
        return self.emit(EventType.ROUND_END, {"round": round_num, "best_this_round": best_this_round})

    def proposal_result(self, label: str, score: Optional[float], success: bool, error: str = "") -> ExperimentEvent:
        self.increment_proposals(1, 1 if success else 0)
        if score and score > self._best_score:
            self.update_score(score)
        return self.emit(EventType.PROPOSAL_EVALUATED, {
            "label": label, "score": score, "success": success, "error": error,
        })

    # Query
    def get_events(self, event_type: Optional[EventType] = None) -> List[ExperimentEvent]:
        """Get buffered events, optionally filtered by type."""
        if event_type is None:
            return list(self._buffer)
        return [e for e in self._buffer if e.event_type == event_type]

    def summary(self) -> Dict:
        """Current experiment status summary."""
        return {
            "experiment_id": self._experiment_id,
            "attempt": self._current_attempt,
            "round": self._current_round,
            "best_score": self._best_score,
            "budget_remaining_s": self._budget_remaining,
            "n_proposals": self._n_proposals,
            "n_successful": self._n_successful,
            "n_events": len(self._buffer),
        }
