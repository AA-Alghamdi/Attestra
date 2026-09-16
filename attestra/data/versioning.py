"""Data versioning — immutable fingerprints + version tracking.

Every experiment records an immutable data fingerprint so the system can answer:
- "Was this result computed on the same data as that result?"
- "Has the data drifted between runs?"
- "What preprocessing was applied?"

Provides:
  - DataVersion: immutable snapshot of data state
  - DataFingerprint: content-addressed hash of data
  - VersionStore: persistent storage of data versions
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data fingerprint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataFingerprint:
    """Content-addressed fingerprint of a dataset."""
    hash_hex: str              # SHA-256 of (X_bytes + y_bytes + metadata)
    n_samples: int
    n_features: int
    dtype_x: str
    dtype_y: str
    y_cardinality: int         # number of unique y values
    x_checksum: str            # first 8 hex of X hash alone
    y_checksum: str            # first 8 hex of y hash alone

    @staticmethod
    def compute(X: Any, y: Any, metadata: Optional[Dict] = None) -> "DataFingerprint":
        """Compute an immutable fingerprint from data arrays."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        x_bytes = X_arr.tobytes()
        y_bytes = y_arr.tobytes()

        x_hash = hashlib.sha256(x_bytes).hexdigest()
        y_hash = hashlib.sha256(y_bytes).hexdigest()

        # Combined hash includes metadata
        combined = hashlib.sha256()
        combined.update(x_bytes)
        combined.update(y_bytes)
        if metadata:
            combined.update(json.dumps(metadata, sort_keys=True).encode())

        return DataFingerprint(
            hash_hex=combined.hexdigest(),
            n_samples=X_arr.shape[0],
            n_features=X_arr.shape[1] if X_arr.ndim > 1 else 1,
            dtype_x=str(X_arr.dtype),
            dtype_y=str(y_arr.dtype),
            y_cardinality=len(np.unique(y_arr)),
            x_checksum=x_hash[:8],
            y_checksum=y_hash[:8],
        )


# ---------------------------------------------------------------------------
# Data version
# ---------------------------------------------------------------------------

@dataclass
class DataVersion:
    """An immutable snapshot of data state — versioned and traceable."""
    version_id: str            # content-addressed from fingerprint + transforms
    fingerprint: DataFingerprint
    # Provenance
    parent_version_id: Optional[str] = None  # what it was derived from
    transformations: List[Dict] = field(default_factory=list)  # transforms applied
    split_indices: Optional[Dict] = None  # {train: [...], val: [...], test: [...]}
    # Metadata
    created_at: float = 0.0
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @staticmethod
    def create(
        X: Any, y: Any,
        parent: Optional["DataVersion"] = None,
        transformations: Optional[List[Dict]] = None,
        split_indices: Optional[Dict] = None,
        description: str = "",
    ) -> "DataVersion":
        """Create a new DataVersion from arrays."""
        fp = DataFingerprint.compute(X, y)

        # Version ID includes transform history for provenance
        version_content = fp.hash_hex
        if transformations:
            version_content += json.dumps(transformations, sort_keys=True)

        version_id = hashlib.sha256(version_content.encode()).hexdigest()[:16]

        return DataVersion(
            version_id=version_id,
            fingerprint=fp,
            parent_version_id=parent.version_id if parent else None,
            transformations=transformations or [],
            split_indices=split_indices,
            created_at=time.time(),
            description=description,
        )

    def to_dict(self) -> Dict:
        """Serialize to dict for storage."""
        d = {
            "version_id": self.version_id,
            "fingerprint": {
                "hash_hex": self.fingerprint.hash_hex,
                "n_samples": self.fingerprint.n_samples,
                "n_features": self.fingerprint.n_features,
                "dtype_x": self.fingerprint.dtype_x,
                "dtype_y": self.fingerprint.dtype_y,
                "y_cardinality": self.fingerprint.y_cardinality,
                "x_checksum": self.fingerprint.x_checksum,
                "y_checksum": self.fingerprint.y_checksum,
            },
            "parent_version_id": self.parent_version_id,
            "transformations": self.transformations,
            "split_indices": self.split_indices,
            "created_at": self.created_at,
            "description": self.description,
            "tags": self.tags,
        }
        return d

    @staticmethod
    def from_dict(d: Dict) -> "DataVersion":
        """Deserialize from dict."""
        fp_d = d["fingerprint"]
        fp = DataFingerprint(
            hash_hex=fp_d["hash_hex"],
            n_samples=fp_d["n_samples"],
            n_features=fp_d["n_features"],
            dtype_x=fp_d["dtype_x"],
            dtype_y=fp_d["dtype_y"],
            y_cardinality=fp_d["y_cardinality"],
            x_checksum=fp_d["x_checksum"],
            y_checksum=fp_d["y_checksum"],
        )
        return DataVersion(
            version_id=d["version_id"],
            fingerprint=fp,
            parent_version_id=d.get("parent_version_id"),
            transformations=d.get("transformations", []),
            split_indices=d.get("split_indices"),
            created_at=d.get("created_at", 0.0),
            description=d.get("description", ""),
            tags=d.get("tags", []),
        )


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class DataContract:
    """Explicit declaration of what a component expects and produces.

    Components (harness, proposer, evaluator, certifier) declare their
    data contract so the router can validate compatibility.
    """
    # Input requirements
    expected_ndim: int = 2           # 1=flat, 2=matrix, 3=tensor, 4=images
    expected_dtype: str = "numeric"  # "numeric" | "text" | "image" | "audio" | "mixed"
    min_samples: int = 10
    max_features: Optional[int] = None
    requires_labels: bool = True
    label_type: str = "any"          # "binary" | "multiclass" | "continuous" | "any"
    # Output declaration
    produces_predictions: bool = True
    produces_probabilities: bool = False
    produces_embeddings: bool = False
    # Compatibility
    supported_task_types: List[str] = field(default_factory=lambda: ["classification", "regression"])

    def validate(self, X: Any, y: Optional[Any] = None) -> Tuple[bool, str]:
        """Validate that data matches this contract."""
        X_arr = np.asarray(X)

        if X_arr.ndim != self.expected_ndim:
            return False, f"Expected ndim={self.expected_ndim}, got {X_arr.ndim}"

        if X_arr.shape[0] < self.min_samples:
            return False, f"Need at least {self.min_samples} samples, got {X_arr.shape[0]}"

        if self.max_features and X_arr.ndim >= 2 and X_arr.shape[1] > self.max_features:
            return False, f"Max {self.max_features} features, got {X_arr.shape[1]}"

        if self.requires_labels and y is None:
            return False, "Labels required but not provided"

        return True, "ok"


# ---------------------------------------------------------------------------
# Version store (persistent)
# ---------------------------------------------------------------------------

class VersionStore:
    """Persistent store of data versions — append-only JSONL."""

    def __init__(self, path: Optional[str] = None):
        self._path = path or os.path.expanduser("~/.attestra/data_versions.jsonl")
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

    def record(self, version: DataVersion) -> None:
        """Append a version to the store."""
        with open(self._path, "a") as f:
            f.write(json.dumps(version.to_dict()) + "\n")

    def find_by_fingerprint(self, fingerprint: DataFingerprint) -> Optional[DataVersion]:
        """Find a version by its data fingerprint."""
        if not os.path.exists(self._path):
            return None
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("fingerprint", {}).get("hash_hex") == fingerprint.hash_hex:
                    return DataVersion.from_dict(d)
        return None

    def find_by_id(self, version_id: str) -> Optional[DataVersion]:
        """Find a version by its version ID."""
        if not os.path.exists(self._path):
            return None
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("version_id") == version_id:
                    return DataVersion.from_dict(d)
        return None

    def lineage(self, version_id: str) -> List[DataVersion]:
        """Get the full lineage (ancestry chain) of a version."""
        chain = []
        current_id = version_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            version = self.find_by_id(current_id)
            if version is None:
                break
            chain.append(version)
            current_id = version.parent_version_id
        return chain

    def recent(self, n: int = 20) -> List[DataVersion]:
        """Get the N most recent versions."""
        versions = []
        if not os.path.exists(self._path):
            return versions
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                versions.append(DataVersion.from_dict(json.loads(line)))
        return sorted(versions, key=lambda v: v.created_at, reverse=True)[:n]
