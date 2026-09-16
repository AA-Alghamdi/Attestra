"""DATA CERTIFICATE -- gate a dataset BEFORE it enters the pool or is searched on.

WHY THIS EXISTS
---------------
A data pool that compounds knowledge across runs is only as trustworthy as its dirtiest dataset. One
dataset with train/test near-duplicates poisons every cross-dataset claim built on it. The data
certificate is the entry gate: it checks the HYGIENE properties a dataset must have before the engine is
allowed to certify models on it. It is the data-side analogue of the model certificate.

CHECKS (each MEASURED, none guessed):
  * split leakage      -- exact OR near-duplicate rows that straddle train and sealed (the killer bug).
  * label noise        -- kNN label-disagreement rate (a rough, honest upper-ish estimate; high => the
                          certifier's theta may be unreachable for reasons that are data, not model).
  * class balance      -- min class fraction (a severely imbalanced set needs balanced_accuracy/macro_f1,
                          not accuracy, or the certificate certifies the majority-class baseline).
  * degenerate features-- constant / all-unique-id columns that either do nothing or leak the row id.

A dataset FAILS the certificate if there is split leakage (near-duplicates straddling train/sealed); the
other checks are reported as warnings the policy can act on. Failing is the honest default: we refuse to
build a certified result on a provably leaky split.

CONTRACT: numpy only. No estimator, no model certifier. Returns a report dict + a boolean `passed`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class DataCertReport:
    passed: bool
    n: int
    near_dup_straddle: int          # count of sealed rows near-duplicating a train row (the blocking issue)
    label_noise_est: Optional[float]
    min_class_fraction: Optional[float]
    degenerate_features: List[int]
    warnings: List[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "n": self.n, "near_dup_straddle": self.near_dup_straddle,
                "label_noise_est": self.label_noise_est, "min_class_fraction": self.min_class_fraction,
                "degenerate_features": self.degenerate_features, "warnings": self.warnings,
                "detail": self.detail}


def _to_matrix(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _min_cross_distances(A: np.ndarray, B: np.ndarray, chunk: int = 512) -> np.ndarray:
    """For each row of B, the min Euclidean distance to any row of A. Chunked to bound memory."""
    if len(A) == 0 or len(B) == 0:
        return np.full(len(B), np.inf)
    out = np.empty(len(B), dtype=float)
    a2 = (A * A).sum(axis=1)
    for s in range(0, len(B), chunk):
        b = B[s:s + chunk]
        d2 = a2[None, :] + (b * b).sum(axis=1)[:, None] - 2.0 * b @ A.T
        np.maximum(d2, 0.0, out=d2)
        out[s:s + chunk] = np.sqrt(d2.min(axis=1))
    return out


def detect_near_dup_straddle(X, train_idx: Sequence[int], sealed_idx: Sequence[int], *,
                             eps: float = 1e-6) -> int:
    """Count sealed rows that duplicate a TRAIN row -- EXACT (a bitwise/rounded-identical row) OR NEAR
    (within Euclidean `eps`). A positive count is split leakage: the model scores on sealed rows it
    effectively saw in train. Exact duplicates are caught by a rounded-row hash (robust to the fp error of
    the gemm distance, which can land an identical row at ~1e-7); near duplicates by the distance scan."""
    M = _to_matrix(X)
    A, B = M[list(train_idx)], M[list(sealed_idx)]
    if len(A) == 0 or len(B) == 0:
        return 0
    # exact (rounded) duplicate detection via hashed row tuples -- precise regardless of eps
    train_rows = {tuple(np.round(r, 9)) for r in A}
    exact = np.array([tuple(np.round(r, 9)) in train_rows for r in B])
    # near-duplicate detection via min cross distance
    near = _min_cross_distances(A, B) <= eps
    return int(np.sum(exact | near))


def estimate_label_noise(X, y, *, k: int = 5) -> Optional[float]:
    """kNN label-disagreement: fraction of rows whose label differs from the majority label of its k
    nearest neighbours (excluding itself). A coarse, honest noisiness signal; returns None when n is too
    small to estimate. Classification only (object/int labels)."""
    M = _to_matrix(X)
    yarr = np.asarray(y, dtype=object)
    n = len(M)
    if n <= k + 1:
        return None
    a2 = (M * M).sum(axis=1)
    disagree = 0
    for i in range(n):
        d2 = a2 + a2[i] - 2.0 * M @ M[i]
        d2[i] = np.inf
        nn = np.argpartition(d2, k)[:k]
        vals, counts = np.unique(yarr[nn], return_counts=True)
        maj = vals[int(np.argmax(counts))]
        if maj != yarr[i]:
            disagree += 1
    return disagree / n


def _class_balance(y) -> Optional[float]:
    yarr = np.asarray(y, dtype=object)
    if len(yarr) == 0:
        return None
    _, counts = np.unique(yarr, return_counts=True)
    return float(counts.min() / counts.sum())


def _degenerate_features(X) -> List[int]:
    M = _to_matrix(X)
    bad = []
    n = len(M)
    for j in range(M.shape[1]):
        col = M[:, j]
        if np.all(col == col[0]):                       # constant
            bad.append(j)
        elif n >= 4 and len(np.unique(col)) == n:       # all-unique -> likely a row id (leaks identity)
            bad.append(j)
    return bad


def certify_dataset(X, y, *, train_idx: Sequence[int], sealed_idx: Sequence[int],
                    near_dup_eps: float = 1e-6, max_label_noise: float = 0.5,
                    min_class_fraction: float = 0.0, estimate_noise: bool = True,
                    k: int = 5, is_regression: bool = False) -> DataCertReport:
    """Run the hygiene battery. FAILS (passed=False) iff there is near-duplicate straddle between train and
    sealed (the one non-negotiable leak), OR the label-noise estimate exceeds max_label_noise, OR the min
    class fraction is below min_class_fraction. Other issues are warnings.

    REGRESSION (is_regression=True): the kNN label-DISAGREEMENT estimate and the class-balance check are
    CLASSIFICATION-only (they assume discrete labels -- on a continuous target the kNN-vote disagreement is
    ~1.0 by construction and a raw-feature kNN is scale-sensitive and misleading). They are skipped; the
    non-negotiable near-duplicate-straddle leak check still runs, and the scientific 'is this benchmark
    gameable?' checks (trivial-baseline / label-shuffle under the tolerance metric) are carried by the
    meta-certifier framing gate, not here."""
    n = len(np.asarray(y, dtype=object))
    straddle = detect_near_dup_straddle(X, train_idx, sealed_idx, eps=near_dup_eps)
    noise = estimate_label_noise(X, y, k=k) if (estimate_noise and not is_regression) else None
    bal = None if is_regression else _class_balance(y)
    degen = _degenerate_features(X)
    warnings: List[str] = []
    passed = True
    if straddle > 0:
        passed = False
        warnings.append(f"SPLIT LEAKAGE: {straddle} sealed rows near-duplicate a train row (eps={near_dup_eps})")
    if noise is not None and noise > max_label_noise:
        passed = False
        warnings.append(f"label-noise estimate {round(noise,3)} exceeds max {max_label_noise}")
    if bal is not None and bal < min_class_fraction:
        passed = False
        warnings.append(f"min class fraction {round(bal,3)} below required {min_class_fraction}")
    if degen:
        warnings.append(f"degenerate feature columns (constant or row-id-like): {degen}")
    return DataCertReport(passed=passed, n=n, near_dup_straddle=straddle, label_noise_est=noise,
                          min_class_fraction=bal, degenerate_features=degen, warnings=warnings,
                          detail={"near_dup_eps": near_dup_eps, "k": k, "is_regression": bool(is_regression)})


__all__ = ["DataCertReport", "certify_dataset", "detect_near_dup_straddle", "estimate_label_noise"]
