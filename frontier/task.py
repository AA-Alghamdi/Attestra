"""Task: the goal + data, in a form the spine can split, run, and certify.

Kept deliberately small for Phase 0 (tabular classification/regression). The harness
fabric (Phase 3) generalizes this to vision/text/timeseries/audio by swapping the
adapter that turns raw data into (rows, metric, theta) while the certify path stays
identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# metric defaults per task kind. theta is the promotion threshold (higher is better,
# matching the science.py convention; regression metrics are higher-is-better too).
_DEFAULT_METRIC = {
    "classification": "accuracy",
    "regression": "r2",
}


@dataclass
class Task:
    X: np.ndarray                      # (n_samples, n_features), float
    y: np.ndarray                      # (n_samples,)
    kind: str                          # "classification" | "regression"
    theta: float                       # promotion threshold the certified lower bound must clear
    metric: str = ""                   # "" -> default for the kind
    name: str = "task"

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y)
        if self.kind not in _DEFAULT_METRIC:
            raise ValueError(f"unknown task kind {self.kind!r}")
        if not self.metric:
            self.metric = _DEFAULT_METRIC[self.kind]
        if len(self.X) != len(self.y):
            raise ValueError("X and y length mismatch")

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def labels(self) -> Sequence:
        if self.kind == "classification":
            return sorted({str(v) for v in self.y})
        return []

    def to_rows(self) -> list:
        """Render to the row-dict shape that science.make_splits / SealedTest expect.

        Each row carries:
          - "target": the label/value (stratification + truth)
          - "features": dict of {f{i}: value} so the certifier's exact-match leakage
            dedup (science.make_splits) works on identical feature vectors
          - "_x": the raw feature vector (what predict_fn reconstructs to score)
        """
        rows = []
        for xi, yi in zip(self.X, self.y):
            feats = {f"f{i}": float(v) for i, v in enumerate(xi)}
            target = str(yi) if self.kind == "classification" else float(yi)
            rows.append({"target": target, "features": feats, "_x": [float(v) for v in xi]})
        return rows

    @staticmethod
    def rows_to_X(rows: list) -> np.ndarray:
        return np.asarray([r["_x"] for r in rows], dtype=float)

    @staticmethod
    def rows_to_y(rows: list, kind: str) -> np.ndarray:
        if kind == "classification":
            return np.asarray([r["target"] for r in rows])
        return np.asarray([float(r["target"]) for r in rows], dtype=float)
