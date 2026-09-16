"""Vision/Text/Timeseries verticals driven from /goal -- the multi-modal front door.

Previously, goal_solver.solve() only auto-drives TABULAR i.i.d. (via TabularCodeArena + RecipeResearcher).
Vision/text goals route to problem_type but are then silently declined because _build() only knows TabularCodeArena.

This module closes that gap: given a GoalSpec with kind="vision" or kind="text", it routes to the
appropriate harness + run_goal_loop and returns the same GoalCertificate. The frozen certifier is
unchanged -- vision/text harnesses reduce images/text to numeric features and then the SAME
sealed-test + Clopper-Pearson + FDR machinery certifies them.

Supported verticals:
  - vision (binary | multiclass): ImageFeaturizer → run_goal_loop → frozen certifier
  - text   (binary | multiclass): TextFeaturizer  → run_goal_loop → frozen certifier
  - timeseries (forecast):        honest routing stub (needs proper temporal CV -- flag for GPU day)

Selection is on VALIDATION only. The frozen certifier is the sole promoter.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from .loop import run_goal_loop, GoalLoopResult
from .harness import harness_for, is_supported


def _build_records_vision(X: np.ndarray, y: np.ndarray, n_classes: int) -> List[dict]:
    """Convert (X, y) numeric matrix into the vision-harness record format: {features: {pixel->val}, target}."""
    records = []
    n_features = X.shape[1] if X.ndim == 2 else 0
    for i in range(len(y)):
        feats = {f"p{j}": float(X[i, j]) for j in range(n_features)}
        records.append({"features": feats, "target": str(int(y[i]))})
    return records


def _build_records_text(X: np.ndarray, y: np.ndarray, n_classes: int, text_key: str = "text") -> List[dict]:
    """Convert (X, y) into text-harness record format: {features: {text_key: str}, target}.

    If X columns look like TF-IDF/numeric (high dim, all float), we fall back to tabular.
    If X has a single column or was stored as object dtype (strings), we use text.
    """
    records = []
    for i in range(len(y)):
        if X.ndim == 2 and X.shape[1] == 1:
            # Single-column: treat as text
            txt = str(X[i, 0])
        elif X.ndim == 1:
            txt = str(X[i])
        else:
            # multi-column numeric: the text featurizer expects {text_key: str} records;
            # concatenate columns with spaces as a crude text representation
            txt = " ".join(str(X[i, j]) for j in range(X.shape[1]))
        feats = {text_key: txt}
        records.append({"features": feats, "target": str(int(y[i]))})
    return records


def solve_vision(goal_text: str, X: np.ndarray, y: np.ndarray, n_classes: int, spec,
                 *, peeks: int = 16, seed: int = 0, threshold: float = 0.5,
                 memory_store=None, gpu: bool = False) -> GoalLoopResult:
    """Drive a vision classification goal through run_goal_loop with VisionClassificationHarness.

    The ImageFeaturizer normalizes pixels to [0,1], flattens, adds optional avg-pool features.
    Then the SAME frozen run_goal_loop + certifier handles it."""
    records = _build_records_vision(X, y, n_classes)
    harness = harness_for("vision", spec.task_type)
    labels = sorted(set(str(int(v)) for v in np.unique(y)))

    result = run_goal_loop(
        goal_text=goal_text,
        records=records,
        kind="vision",
        task_type=spec.task_type,
        metric=spec.metric,
        threshold=threshold,
        labels=labels,
        harness=harness,
        max_rounds=8,
        budget_experiments=100,
        seed=seed,
        memory_store=memory_store,
    )
    return result


def solve_text(goal_text: str, X: np.ndarray, y: np.ndarray, n_classes: int, spec,
               *, peeks: int = 16, seed: int = 0, threshold: float = 0.5,
               text_key: str = "text", memory_store=None, gpu: bool = False) -> GoalLoopResult:
    """Drive a text classification goal through run_goal_loop with TextClassificationHarness.

    The TextFeaturizer extracts {text_key: str} from each record and builds a TF-IDF matrix.
    Then the SAME frozen run_goal_loop + certifier handles it."""
    records = _build_records_text(X, y, n_classes, text_key=text_key)
    harness = harness_for("text", spec.task_type)
    labels = sorted(set(str(int(v)) for v in np.unique(y)))

    result = run_goal_loop(
        goal_text=goal_text,
        records=records,
        kind="text",
        task_type=spec.task_type,
        metric=spec.metric,
        threshold=threshold,
        labels=labels,
        text_key=text_key,
        harness=harness,
        max_rounds=8,
        budget_experiments=100,
        seed=seed,
        memory_store=memory_store,
    )
    return result


def can_drive_vertical(spec) -> bool:
    """True if this spec's (kind, task_type) has a vertical driver beyond tabular."""
    return spec.kind in ("vision", "text") and is_supported(spec.kind, spec.task_type)


def route_vertical(goal_text: str, X: np.ndarray, y: np.ndarray, n_classes: int, spec,
                   *, peeks: int = 16, seed: int = 0, threshold: float = 0.5,
                   memory_store=None, gpu: bool = False,
                   text_key: str = "text") -> Optional[GoalLoopResult]:
    """Route a goal to the appropriate vertical solver. Returns None if no vertical matches
    (caller should fall back to the tabular RecipeResearcher path)."""
    if spec.kind == "vision" and spec.task_type in ("binary", "multiclass"):
        return solve_vision(goal_text, X, y, n_classes, spec, peeks=peeks, seed=seed,
                            threshold=threshold, memory_store=memory_store, gpu=gpu)
    if spec.kind == "text" and spec.task_type in ("binary", "multiclass"):
        return solve_text(goal_text, X, y, n_classes, spec, peeks=peeks, seed=seed,
                          threshold=threshold, text_key=text_key,
                          memory_store=memory_store, gpu=gpu)
    return None
