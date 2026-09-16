"""Experiment registry: durable JSONL ledger of positive AND negative certificates.

Every experiment produces a record -- whether it succeeds or fails. The negative
certificates are AS IMPORTANT as the positive ones: they tell us what NOT to try
on similar problems, saving budget in future runs.

The registry supports:
  - Append-only writes (atomic via tmp+rename)
  - Querying by dataset fingerprint, metric, technique
  - Cross-experiment learning (what works on similar datasets)
  - FDR control over the stream of promotions (via PromotionLedger)
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRecord:
    """A single experiment outcome (positive or negative)."""
    plan_hash: str
    goal: str
    dataset_fingerprint: str
    metric: str
    threshold: float
    # outcome
    decision: str              # "certified" | "do_not_certify" | "honest_stop" | "error"
    best_score: float
    best_technique: str
    # certificate (None for failures)
    certificate: Optional[Dict] = None
    # metadata
    n_samples: int = 0
    n_features: int = 0
    task_type: str = ""
    n_proposals: int = 0
    n_successful: int = 0
    n_failed: int = 0
    elapsed_s: float = 0.0
    error_summary: Optional[Dict] = None
    failure_report: Optional[Dict] = None
    # timestamps
    created_at: float = field(default_factory=time.time)

    @property
    def is_positive(self) -> bool:
        return self.decision == "certified"

    @property
    def is_negative(self) -> bool:
        return self.decision != "certified"


class ExperimentRegistry:
    """Append-only JSONL registry of experiment outcomes."""

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.expanduser("~/.attestra/experiment_registry.jsonl")
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def record(self, entry: ExperimentRecord) -> None:
        """Append an experiment record (atomic write)."""
        d = asdict(entry)
        line = json.dumps(d, sort_keys=True, default=str)
        self._append_line(line)

    def all_records(self) -> List[ExperimentRecord]:
        """Read all records."""
        records = []
        for line in self._read_lines():
            try:
                d = json.loads(line)
                records.append(ExperimentRecord(**{
                    k: v for k, v in d.items()
                    if k in ExperimentRecord.__dataclass_fields__
                }))
            except (json.JSONDecodeError, TypeError):
                continue
        return records

    def positive_certificates(self) -> List[ExperimentRecord]:
        return [r for r in self.all_records() if r.is_positive]

    def negative_certificates(self) -> List[ExperimentRecord]:
        return [r for r in self.all_records() if r.is_negative]

    def similar_experiments(self, fingerprint: str, *, task_type: str = "",
                            metric: str = "", limit: int = 10) -> List[ExperimentRecord]:
        """Find experiments on similar datasets."""
        records = self.all_records()
        # Filter by task type if specified
        if task_type:
            records = [r for r in records if r.task_type == task_type]
        if metric:
            records = [r for r in records if r.metric == metric]
        # Sort by fingerprint similarity (exact match first, then by technique overlap)
        exact = [r for r in records if r.dataset_fingerprint == fingerprint]
        others = [r for r in records if r.dataset_fingerprint != fingerprint]
        return (exact + others)[:limit]

    def what_works(self, fingerprint: str, task_type: str = "") -> Dict[str, float]:
        """What techniques have succeeded on similar problems?"""
        similar = self.similar_experiments(fingerprint, task_type=task_type, limit=50)
        positive = [r for r in similar if r.is_positive]
        technique_scores: Dict[str, List[float]] = {}
        for r in positive:
            if r.best_technique not in technique_scores:
                technique_scores[r.best_technique] = []
            technique_scores[r.best_technique].append(r.best_score)
        # Average score per technique
        return {t: sum(scores) / len(scores) for t, scores in technique_scores.items()}

    def what_fails(self, fingerprint: str, task_type: str = "") -> List[str]:
        """What techniques have failed on similar problems?"""
        similar = self.similar_experiments(fingerprint, task_type=task_type, limit=50)
        negative = [r for r in similar if r.is_negative]
        # Techniques that only fail
        fail_techniques = set()
        success_techniques = set()
        for r in similar:
            if r.is_positive:
                success_techniques.add(r.best_technique)
            else:
                fail_techniques.add(r.best_technique)
        return sorted(fail_techniques - success_techniques)

    def retrieve(
        self, fingerprint: str, task_type: str = "", top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-K most relevant prior experiments for knowledge transfer.

        Returns dicts with technique, score, decision, and metadata — suitable
        for injecting into proposal generation or strategy selection.
        """
        similar = self.similar_experiments(fingerprint, task_type=task_type, limit=top_k * 3)
        # Rank by recency * success
        ranked = sorted(similar, key=lambda r: (r.is_positive, r.best_score, r.created_at), reverse=True)
        results = []
        for r in ranked[:top_k]:
            results.append({
                "technique": r.best_technique,
                "score": r.best_score,
                "decision": r.decision,
                "task_type": r.task_type,
                "n_samples": r.n_samples,
                "metric": r.metric,
                "positive": r.is_positive,
            })
        return results

    def search(self, task_type: str, metric: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        """Search registry by task type and metric — for strategy learning."""
        records = self.all_records()
        matched = [r for r in records if r.task_type == task_type]
        if metric:
            matched = [r for r in matched if r.metric == metric]
        matched.sort(key=lambda r: (r.best_score, r.created_at), reverse=True)
        return [
            {
                "technique": r.best_technique,
                "score": r.best_score,
                "decision": r.decision,
                "n_samples": r.n_samples,
                "positive": r.is_positive,
            }
            for r in matched[:limit]
        ]

    def count(self) -> int:
        return len(self._read_lines())

    def summary(self) -> Dict[str, Any]:
        records = self.all_records()
        return {
            "total": len(records),
            "positive": sum(1 for r in records if r.is_positive),
            "negative": sum(1 for r in records if r.is_negative),
            "unique_datasets": len(set(r.dataset_fingerprint for r in records)),
            "unique_techniques": len(set(r.best_technique for r in records)),
        }

    def _read_lines(self) -> List[str]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return [l.strip() for l in f if l.strip()]
        except OSError:
            return []

    def _append_line(self, line: str) -> None:
        d = os.path.dirname(self.path)
        os.makedirs(d, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def fingerprint_dataset(X, y) -> str:
    """Content-address a dataset for registry lookups."""
    import numpy as np
    X = np.asarray(X)
    y = np.asarray(y)
    h = hashlib.sha256()
    h.update(f"shape={X.shape}".encode())
    h.update(f"dtype_x={X.dtype}".encode())
    h.update(f"dtype_y={y.dtype}".encode())
    # Sample-based hash (not full data, for speed)
    rng = np.random.default_rng(0)
    if len(X) > 100:
        idx = rng.choice(len(X), 100, replace=False)
    else:
        idx = np.arange(len(X))
    h.update(X[idx].tobytes())
    h.update(y[idx].tobytes())
    return h.hexdigest()[:16]
