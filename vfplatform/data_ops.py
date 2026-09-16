"""DATA-OPS MOVES + ACTIVE LABELING -- many mid-level wins are data problems, not model problems.

STATUS (2026-06): BUILT + UNIT-TESTED, NOT YET WIRED into the certified researcher. This is the natural
next lever once the representation ceiling is hit (repr_researcher flags residual tasks as a DATA ceiling
-- exactly this module's domain), but it has no certified-arena measurement yet. Wire-or-delete debt --
do not claim value without one.

WHY THIS EXISTS
---------------
On small/commercial data (the n=100 call-ender) the biggest lever is often the DATA, not the model:
deduplicate, reweight imbalanced classes, drop suspected label noise, augment, and -- crucially -- decide
which examples to LABEL NEXT. So data transforms are first-class search Moves alongside model/feature
moves, and active labeling is BOTH an operator and a deliverable ('label these 40 next for the biggest
certified-bound gain').

THE INVARIANT
-------------
Data-ops only transform the TRAIN split (indices/weights/features); they never touch the sealed split and
never certify. Leak-safety is enforced upstream (splits.py) and re-checked by data_cert.py; nothing here
can move a row across the train/sealed boundary. Deterministic given a seed.

CONTRACT: numpy only (sklearn used lazily inside active-labeling helpers, matching the repo's harness).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DataOpResult:
    """The outcome of a data-op: the new train indices (subset/reordering of the originals) and optional
    per-sample weights, plus a human-readable note for the run log."""
    name: str
    train_idx: np.ndarray
    sample_weight: Optional[np.ndarray]
    note: str


def dedupe(X: np.ndarray, train_idx: Sequence[int], *, eps: float = 1e-6) -> DataOpResult:
    """Drop near-duplicate rows WITHIN the train split (keep first occurrence). Reduces leakage-by-repetition
    and over-counting of duplicated points."""
    idx = np.asarray(list(train_idx))
    seen_keys = set()
    keep: List[int] = []
    for i in idx:
        key = tuple(np.round(X[i] / max(eps, 1e-12)).astype(np.int64).tolist())
        if key not in seen_keys:
            seen_keys.add(key)
            keep.append(int(i))
    kept = np.asarray(keep)
    return DataOpResult("dedupe", kept, None, f"kept {len(kept)}/{len(idx)} after near-dup removal")


def reweight_balanced(y: np.ndarray, train_idx: Sequence[int]) -> DataOpResult:
    """Inverse-frequency class weights over the train split, so a rare-but-critical class (e.g. 'escalate')
    is not drowned out. Indices are unchanged; weights are returned aligned to train_idx order."""
    idx = np.asarray(list(train_idx))
    labels = y[idx]
    classes, counts = np.unique(labels, return_counts=True)
    freq = {int(c): int(n) for c, n in zip(classes, counts)}
    n_classes = len(classes)
    total = len(labels)
    w = np.asarray([total / (n_classes * freq[int(l)]) for l in labels], dtype=float)
    return DataOpResult("reweight_balanced", idx, w, f"balanced weights over {n_classes} classes")


def clean_label_noise(X: np.ndarray, y: np.ndarray, train_idx: Sequence[int], *, k: int = 5,
                      drop_threshold: float = 0.75) -> DataOpResult:
    """Drop train rows whose label disagrees with >= drop_threshold of their k nearest TRAIN neighbours
    (a kNN label-noise filter). Conservative: only removes points with strong neighbourhood disagreement."""
    from sklearn.neighbors import NearestNeighbors

    idx = np.asarray(list(train_idx))
    Xt, yt = X[idx], y[idx]
    n = len(idx)
    kk = min(k, max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xt)
    _, nbr = nn.kneighbors(Xt)
    keep: List[int] = []
    for local in range(n):
        neigh = nbr[local][1:]                     # exclude self
        disagree = float(np.mean(yt[neigh] != yt[local])) if len(neigh) else 0.0
        if disagree < drop_threshold:
            keep.append(int(idx[local]))
    kept = np.asarray(keep)
    return DataOpResult("clean_label_noise", kept, None,
                        f"kept {len(kept)}/{n} after kNN noise filter (thr={drop_threshold})")


def augment_jitter(X: np.ndarray, train_idx: Sequence[int], *, factor: int = 1, scale: float = 0.05,
                   seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-jitter augmentation for numeric features: return (X_aug, y_extra_idx_map, new_rows) where
    `new_rows` are appended synthetic rows. Returns the augmented feature matrix and the index list mapping
    each augmented row back to its source train row (so labels can be copied). Used to expand tiny train
    sets; never creates sealed rows."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(list(train_idx))
    base = X[idx]
    col_scale = base.std(axis=0, keepdims=True) * scale
    aug_blocks = [base]
    src_map = [idx]
    for _ in range(factor):
        noise = rng.normal(0.0, 1.0, size=base.shape) * col_scale
        aug_blocks.append(base + noise)
        src_map.append(idx)
    X_aug = np.vstack(aug_blocks)
    src = np.concatenate(src_map)
    return X_aug, src, idx


def _uncertainty_scores(model, X_pool: np.ndarray) -> np.ndarray:
    """Margin uncertainty: 1 - (top1 - top2) probability gap. Higher = more uncertain. Falls back to a
    distance-to-decision proxy if the model lacks predict_proba."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_pool))
        if proba.shape[1] >= 2:
            part = np.sort(proba, axis=1)
            margin = part[:, -1] - part[:, -2]
            return 1.0 - margin
        return 1.0 - np.abs(proba[:, 0] - 0.5) * 2.0
    if hasattr(model, "decision_function"):
        d = np.asarray(model.decision_function(X_pool))
        if d.ndim == 1:
            return 1.0 / (1.0 + np.abs(d))
        part = np.sort(d, axis=1)
        return 1.0 / (1.0 + (part[:, -1] - part[:, -2]))
    raise ValueError("model needs predict_proba or decision_function for uncertainty sampling")


def active_label_query(model, X_pool: np.ndarray, pool_idx: Sequence[int], k: int,
                       *, diversify: bool = True) -> np.ndarray:
    """Select up to k indices from `pool_idx` to label next, by descending model uncertainty. With
    diversify=True, greedily skips picks too close to an already-selected one (uncertainty + coverage).
    Returns the chosen ORIGINAL indices. This is the 'label these next' deliverable."""
    idx = np.asarray(list(pool_idx))
    if len(idx) == 0 or k <= 0:
        return np.asarray([], dtype=int)
    scores = _uncertainty_scores(model, X_pool[idx])
    order = np.argsort(-scores)                    # most uncertain first
    if not diversify:
        return idx[order[:k]]
    chosen_local: List[int] = []
    chosen_vecs: List[np.ndarray] = []
    span = np.linalg.norm(X_pool[idx].std(axis=0)) + 1e-9
    min_sep = 0.25 * span
    for o in order:
        v = X_pool[idx[o]]
        if all(np.linalg.norm(v - cv) > min_sep for cv in chosen_vecs):
            chosen_local.append(int(o))
            chosen_vecs.append(v)
        if len(chosen_local) >= k:
            break
    if len(chosen_local) < k:                       # top up with next-most-uncertain if diversity starved
        for o in order:
            if int(o) not in chosen_local:
                chosen_local.append(int(o))
            if len(chosen_local) >= k:
                break
    return idx[np.asarray(chosen_local[:k])]


__all__ = ["DataOpResult", "dedupe", "reweight_balanced", "clean_label_noise", "augment_jitter",
           "active_label_query"]
