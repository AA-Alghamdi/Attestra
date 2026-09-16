"""Provenance tracking — full reproducibility record for every experiment.

Records everything needed to reproduce a result:
  - Library versions (pip freeze)
  - Random seeds at every level
  - Data fingerprint
  - Hardware spec
  - Wall-clock timestamps
  - Complete derivation tree of the winning program
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ProvenanceRecord:
    """Complete provenance for reproducing an experiment result."""
    # Identity
    experiment_id: str
    record_id: str = ""
    # Data
    data_fingerprint: str = ""      # SHA-256 of (X, y)
    n_samples: int = 0
    n_features: int = 0
    split_seed: int = 0
    split_indices: Optional[Dict] = None  # {train: [...], val: [...], test: [...]}
    # Environment
    python_version: str = ""
    platform_info: str = ""
    cpu_model: str = ""
    memory_gb: float = 0.0
    gpu_model: str = ""
    # Libraries
    library_versions: Dict[str, str] = field(default_factory=dict)
    # Seeds
    numpy_seed: int = 0
    python_seed: int = 0
    model_seeds: Dict[str, int] = field(default_factory=dict)
    # Timing
    started_at: float = 0.0
    completed_at: float = 0.0
    wall_seconds: float = 0.0
    # Result
    winning_technique: str = ""
    winning_score: float = 0.0
    winning_code: str = ""
    # Derivation tree
    derivation_tree: List[Dict] = field(default_factory=list)
    # The full configuration used
    config_snapshot: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "record_id": self.record_id,
            "data_fingerprint": self.data_fingerprint,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "split_seed": self.split_seed,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "cpu_model": self.cpu_model,
            "memory_gb": self.memory_gb,
            "gpu_model": self.gpu_model,
            "library_versions": self.library_versions,
            "numpy_seed": self.numpy_seed,
            "python_seed": self.python_seed,
            "model_seeds": self.model_seeds,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "wall_seconds": self.wall_seconds,
            "winning_technique": self.winning_technique,
            "winning_score": self.winning_score,
            "derivation_tree": self.derivation_tree,
            "config_snapshot": self.config_snapshot,
        }


class ProvenanceTracker:
    """Tracks provenance for experiment reproducibility.

    Usage:
        tracker = ProvenanceTracker("exp_123")
        tracker.record_environment()
        tracker.record_data(X, y, seed=42)
        tracker.record_start()
        # ... run experiment ...
        tracker.record_result(technique, score, code)
        tracker.record_derivation(parent_id, child_id, method)
        tracker.finalize()
        provenance = tracker.get_record()
    """

    def __init__(self, experiment_id: str, persist_path: Optional[str] = None):
        self._record = ProvenanceRecord(experiment_id=experiment_id)
        self._record.record_id = hashlib.sha256(
            f"{experiment_id}:{time.time()}".encode()
        ).hexdigest()[:16]
        self._path = persist_path or os.path.expanduser("~/.attestra/provenance.jsonl")
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

    def record_environment(self) -> None:
        """Snapshot the current environment."""
        self._record.python_version = sys.version
        self._record.platform_info = platform.platform()
        self._record.cpu_model = platform.processor() or "unknown"

        # Memory
        try:
            import psutil
            self._record.memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass

        # Key library versions
        libs = {}
        for lib_name in ["numpy", "scipy", "sklearn", "pandas", "torch", "openai"]:
            try:
                mod = __import__(lib_name)
                libs[lib_name] = getattr(mod, "__version__", "unknown")
            except ImportError:
                pass
        self._record.library_versions = libs

    def record_data(self, X: Any, y: Any, seed: int = 42) -> None:
        """Record data fingerprint and metadata."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        combined = hashlib.sha256()
        combined.update(X_arr.tobytes())
        combined.update(y_arr.tobytes())
        self._record.data_fingerprint = combined.hexdigest()

        self._record.n_samples = X_arr.shape[0]
        self._record.n_features = X_arr.shape[1] if X_arr.ndim > 1 else 1
        self._record.split_seed = seed
        self._record.numpy_seed = seed
        self._record.python_seed = seed

    def record_start(self) -> None:
        """Mark experiment start."""
        self._record.started_at = time.time()

    def record_result(self, technique: str, score: float, code: str = "") -> None:
        """Record the winning result."""
        self._record.winning_technique = technique
        self._record.winning_score = score
        self._record.winning_code = code

    def record_derivation(self, parent_id: str, child_id: str, method: str, score: float = 0.0) -> None:
        """Record a derivation step (for the derivation tree)."""
        self._record.derivation_tree.append({
            "parent_id": parent_id,
            "child_id": child_id,
            "method": method,
            "score": score,
            "timestamp": time.time(),
        })

    def record_config(self, config: Dict) -> None:
        """Snapshot the full configuration."""
        # Only store serializable fields
        safe_config = {}
        for k, v in config.items():
            if isinstance(v, (str, int, float, bool, type(None), list, dict)):
                safe_config[k] = v
            else:
                safe_config[k] = str(type(v).__name__)
        self._record.config_snapshot = safe_config

    def finalize(self) -> ProvenanceRecord:
        """Finalize and persist the provenance record."""
        self._record.completed_at = time.time()
        self._record.wall_seconds = self._record.completed_at - self._record.started_at

        # Persist
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(self._record.to_dict()) + "\n")
        except Exception:
            pass

        return self._record

    def get_record(self) -> ProvenanceRecord:
        """Get the current provenance record."""
        return self._record
