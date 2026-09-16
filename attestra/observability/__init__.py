"""Observability — structured events, streaming status, full provenance.

Provides real-time visibility into long-running experiments:
  - EventEmitter: structured event stream for dashboard consumption
  - ProvenanceTracker: full reproducibility record for every experiment
  - ExperimentMonitor: streaming status (round, best score, budget remaining)
"""
from .events import EventEmitter, ExperimentEvent
from .provenance import ProvenanceTracker, ProvenanceRecord

__all__ = [
    "EventEmitter", "ExperimentEvent",
    "ProvenanceTracker", "ProvenanceRecord",
]
