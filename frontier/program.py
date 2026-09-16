"""The proposal unit: a Program.

A Program is the thing the autoresearcher proposes, runs, and (if it wins) certifies.
Crucially it is CODE, not a (family, hyperparameters) selection from a fixed menu.

Contract for the code string
----------------------------
The code must define a callable `build_estimator()` that returns a fresh,
unfitted scikit-learn-compatible estimator (anything with .fit / .predict).
Because it is arbitrary code, the proposal space includes feature engineering,
preprocessing, target transforms, stacking, and (later) torch models. The
catalog families are simply Programs whose build_estimator() returns a known
pipeline; they are SEEDS and BASELINES, not the search space.

The code never reports a metric. It only produces an estimator. All numbers are
computed by the trusted parent (numeric firewall).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Program:
    code: str                          # defines build_estimator() -> sklearn-like estimator
    source: str                        # "seed" | "mutation" | "llm" | "retrieval" | ...
    label: str = ""                    # short human label, e.g. "hist_gbm" or "poly+ridge"
    parent_id: Optional[str] = None    # provenance: which program this was derived from
    provenance: dict = field(default_factory=dict)  # e.g. retrieved arxiv ids, prompt hash

    @property
    def id(self) -> str:
        h = hashlib.sha256(self.code.encode("utf-8")).hexdigest()[:12]
        return f"{self.source}:{self.label or 'prog'}:{h}"

    def __repr__(self) -> str:
        return f"Program({self.id})"


@dataclass
class RunResult:
    """Outcome of executing a Program in the sandbox on (train -> predict on a target split).

    `preds` are the raw predictions for the requested split (val or sealed). On failure,
    `error` carries a short, type-tagged message and `preds` is None. The parent computes
    every metric from `preds`; the sandbox never returns a score.
    """
    program_id: str
    ok: bool
    preds: Optional[list] = None
    error: str = ""
    error_kind: str = ""               # coarse taxonomy: "timeout"|"import"|"fit"|"build"|"oom"|"other"
    wall_seconds: float = 0.0
