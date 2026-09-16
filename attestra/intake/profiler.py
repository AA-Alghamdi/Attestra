"""Data profiler: inspect data BEFORE designing experiments.

A real ML researcher starts by understanding the data. This module produces a
structured DataProfile that informs experiment design decisions:
  - Feature types and distributions
  - Missing value patterns
  - Class balance (classification)
  - Target distribution (regression)
  - Correlation structure
  - Potential issues (high cardinality, constant features, outliers)
  - Data quality score

The profile feeds into the experiment design stage (VoI, power analysis, cost)
and informs LLM proposals with concrete data understanding.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FeatureProfile:
    """Profile of a single feature."""
    name: str
    dtype: str               # "numeric" | "categorical" | "text" | "binary"
    n_unique: int
    n_missing: int
    missing_rate: float
    # numeric stats (None for categorical)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skew: Optional[float] = None
    kurtosis: Optional[float] = None
    # categorical stats
    top_values: Optional[List[Tuple[str, int]]] = None
    # flags
    is_constant: bool = False
    is_id_like: bool = False     # high cardinality, likely an ID
    has_outliers: bool = False


@dataclass
class DataProfile:
    """Complete profile of a dataset."""
    n_samples: int
    n_features: int
    n_classes: int                   # 0 for regression
    task_type: str                   # "binary" | "multiclass" | "regression"
    features: List[FeatureProfile] = field(default_factory=list)
    # target analysis
    class_balance: Dict[str, float] = field(default_factory=dict)
    target_stats: Dict[str, float] = field(default_factory=dict)
    # aggregate stats
    missing_rate: float = 0.0       # overall missing rate
    n_constant_features: int = 0
    n_id_like_features: int = 0
    n_numeric_features: int = 0
    n_categorical_features: int = 0
    # correlations
    top_correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    # quality
    quality_score: float = 1.0      # 0-1, lower = more issues
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_llm_context(self) -> str:
        """Format profile for LLM consumption (experiment design prompt)."""
        lines = [
            f"Dataset: {self.n_samples} samples, {self.n_features} features",
            f"Task: {self.task_type}" + (f" ({self.n_classes} classes)" if self.n_classes > 0 else ""),
            f"Numeric features: {self.n_numeric_features}, Categorical: {self.n_categorical_features}",
        ]
        if self.missing_rate > 0:
            lines.append(f"Missing data: {self.missing_rate:.1%} overall")
        if self.n_constant_features > 0:
            lines.append(f"Constant features: {self.n_constant_features} (should be removed)")
        if self.class_balance:
            minority = min(self.class_balance.values())
            majority = max(self.class_balance.values())
            if majority / max(minority, 1e-9) > 3:
                lines.append(f"Class imbalance: minority={minority:.1%}, majority={majority:.1%}")
        if self.target_stats:
            lines.append(f"Target: mean={self.target_stats.get('mean', 0):.3f}, "
                         f"std={self.target_stats.get('std', 0):.3f}")
        if self.top_correlations:
            lines.append("Top correlated feature pairs:")
            for f1, f2, corr in self.top_correlations[:3]:
                lines.append(f"  {f1} <-> {f2}: {corr:.3f}")
        if self.issues:
            lines.append("Issues detected:")
            for issue in self.issues[:5]:
                lines.append(f"  - {issue}")
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  - {rec}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "task_type": self.task_type,
            "missing_rate": self.missing_rate,
            "n_constant_features": self.n_constant_features,
            "n_numeric_features": self.n_numeric_features,
            "n_categorical_features": self.n_categorical_features,
            "quality_score": self.quality_score,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


def profile_data(X: np.ndarray, y: np.ndarray, *, feature_names: Optional[List[str]] = None) -> DataProfile:
    """Profile a dataset (X, y) for experiment design.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector
        feature_names: Optional feature names

    Returns:
        DataProfile with comprehensive data analysis
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape
    names = feature_names or [f"f{i}" for i in range(n_features)]

    # Determine task type
    unique_y = np.unique(y[~_isnan(y)] if _has_nan(y) else y)
    is_regression = _is_regression(y, unique_y)
    n_classes = 0 if is_regression else len(unique_y)
    if n_classes == 2:
        task_type = "binary"
    elif n_classes > 2:
        task_type = "multiclass"
    else:
        task_type = "regression"

    # Profile each feature
    features = []
    total_missing = 0
    n_constant = 0
    n_id_like = 0
    n_numeric = 0
    n_categorical = 0

    for i in range(n_features):
        col = X[:, i]
        fp = _profile_feature(col, names[i], n_samples)
        features.append(fp)
        total_missing += fp.n_missing
        if fp.is_constant:
            n_constant += 1
        if fp.is_id_like:
            n_id_like += 1
        if fp.dtype == "numeric":
            n_numeric += 1
        else:
            n_categorical += 1

    missing_rate = total_missing / max(n_samples * n_features, 1)

    # Target analysis
    class_balance = {}
    target_stats = {}
    if is_regression:
        valid_y = y[~_isnan(y)] if _has_nan(y) else y
        valid_y = valid_y.astype(np.float64)
        target_stats = {
            "mean": float(np.mean(valid_y)),
            "std": float(np.std(valid_y)),
            "min": float(np.min(valid_y)),
            "max": float(np.max(valid_y)),
            "median": float(np.median(valid_y)),
        }
    else:
        counts = Counter(y.tolist())
        total = sum(counts.values())
        class_balance = {str(k): round(v / total, 4) for k, v in counts.items()}

    # Correlations (numeric features only, subsample for speed)
    top_correlations = _top_correlations(X, names, n_numeric, n_samples)

    # Quality assessment
    quality_score, issues, recommendations = _assess_quality(
        n_samples, n_features, missing_rate, n_constant, n_id_like,
        class_balance, is_regression, target_stats, features
    )

    return DataProfile(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        task_type=task_type,
        features=features,
        class_balance=class_balance,
        target_stats=target_stats,
        missing_rate=missing_rate,
        n_constant_features=n_constant,
        n_id_like_features=n_id_like,
        n_numeric_features=n_numeric,
        n_categorical_features=n_categorical,
        top_correlations=top_correlations,
        quality_score=quality_score,
        issues=issues,
        recommendations=recommendations,
    )


def _profile_feature(col: np.ndarray, name: str, n_samples: int) -> FeatureProfile:
    """Profile a single feature column."""
    n_missing = int(np.sum(_isnan(col)))
    missing_rate = n_missing / max(n_samples, 1)
    valid = col[~_isnan(col)] if n_missing > 0 else col

    # Determine type
    is_numeric = _is_numeric_col(valid)
    n_unique = len(np.unique(valid)) if len(valid) > 0 else 0

    if n_unique <= 2 and is_numeric:
        dtype = "binary"
    elif is_numeric:
        dtype = "numeric"
    else:
        dtype = "categorical"

    fp = FeatureProfile(
        name=name,
        dtype=dtype,
        n_unique=n_unique,
        n_missing=n_missing,
        missing_rate=missing_rate,
        is_constant=(n_unique <= 1),
        is_id_like=(n_unique > 0.9 * n_samples and n_samples > 20),
    )

    if is_numeric and len(valid) > 0:
        valid_f = valid.astype(np.float64)
        fp.mean = float(np.mean(valid_f))
        fp.std = float(np.std(valid_f))
        fp.min_val = float(np.min(valid_f))
        fp.max_val = float(np.max(valid_f))
        if len(valid_f) >= 8 and fp.std > 1e-12:
            z = (valid_f - fp.mean) / fp.std
            fp.skew = float(np.mean(z ** 3))
            fp.kurtosis = float(np.mean(z ** 4) - 3.0)
        # Outlier detection (IQR method)
        if len(valid_f) >= 20:
            q1, q3 = np.percentile(valid_f, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                n_outliers = int(np.sum((valid_f < q1 - 3 * iqr) | (valid_f > q3 + 3 * iqr)))
                fp.has_outliers = n_outliers > max(1, n_samples * 0.01)
    elif not is_numeric and len(valid) > 0:
        counts = Counter(valid.tolist())
        fp.top_values = counts.most_common(5)

    return fp


def _top_correlations(X: np.ndarray, names: List[str], n_numeric: int,
                      n_samples: int) -> List[Tuple[str, str, float]]:
    """Find top correlated feature pairs (subsample for speed)."""
    if n_numeric < 2 or n_samples < 10:
        return []

    # Subsample for speed
    n = min(n_samples, 5000)
    idx = np.random.default_rng(0).choice(n_samples, n, replace=False) if n_samples > n else np.arange(n)
    Xs = X[idx]

    # Find numeric columns
    numeric_idx = []
    for i in range(Xs.shape[1]):
        if _is_numeric_col(Xs[:, i]):
            numeric_idx.append(i)
    if len(numeric_idx) < 2:
        return []

    # Limit to top 50 numeric features
    numeric_idx = numeric_idx[:50]
    Xn = Xs[:, numeric_idx].astype(np.float64)

    # Handle NaN for correlation
    for i in range(Xn.shape[1]):
        col = Xn[:, i]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmean(col)

    # Compute correlation matrix
    stds = np.std(Xn, axis=0)
    valid_cols = stds > 1e-12
    if valid_cols.sum() < 2:
        return []

    Xn = Xn[:, valid_cols]
    valid_idx = [numeric_idx[i] for i, v in enumerate(valid_cols) if v]

    corr = np.corrcoef(Xn.T)
    np.fill_diagonal(corr, 0)

    # Top pairs
    pairs = []
    for i in range(len(valid_idx)):
        for j in range(i + 1, len(valid_idx)):
            if math.isfinite(corr[i, j]):
                pairs.append((names[valid_idx[i]], names[valid_idx[j]], abs(float(corr[i, j]))))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:10]


def _assess_quality(n_samples: int, n_features: int, missing_rate: float,
                    n_constant: int, n_id_like: int, class_balance: Dict[str, float],
                    is_regression: bool, target_stats: Dict[str, float],
                    features: List[FeatureProfile]) -> Tuple[float, List[str], List[str]]:
    """Assess data quality and generate recommendations."""
    score = 1.0
    issues = []
    recommendations = []

    # Missing data
    if missing_rate > 0.5:
        score -= 0.3
        issues.append(f"Very high missing rate: {missing_rate:.1%}")
        recommendations.append("Consider imputation or dropping sparse features")
    elif missing_rate > 0.1:
        score -= 0.1
        issues.append(f"Moderate missing rate: {missing_rate:.1%}")
        recommendations.append("Use models tolerant of missing values (HistGBM) or impute")

    # Constant features
    if n_constant > 0:
        score -= 0.05 * min(n_constant, 5)
        issues.append(f"{n_constant} constant feature(s) detected")
        recommendations.append("Remove constant features before training")

    # ID-like features
    if n_id_like > 0:
        score -= 0.1 * min(n_id_like, 3)
        issues.append(f"{n_id_like} ID-like feature(s) with near-unique values")
        recommendations.append("Remove ID columns (they cause overfitting)")

    # Class imbalance
    if class_balance:
        minority = min(class_balance.values())
        majority = max(class_balance.values())
        ratio = majority / max(minority, 1e-9)
        if ratio > 10:
            score -= 0.2
            issues.append(f"Severe class imbalance: {ratio:.1f}:1 ratio")
            recommendations.append("Use class weighting, SMOTE, or stratified sampling")
        elif ratio > 3:
            score -= 0.1
            issues.append(f"Moderate class imbalance: {ratio:.1f}:1 ratio")
            recommendations.append("Consider class_weight='balanced' in model")

    # Sample size
    if n_samples < 50:
        score -= 0.2
        issues.append(f"Very few samples ({n_samples})")
        recommendations.append("Use simple models (logistic regression, small trees)")
    elif n_samples < n_features * 5:
        score -= 0.1
        issues.append(f"Low sample-to-feature ratio ({n_samples}/{n_features} = {n_samples/max(n_features,1):.1f})")
        recommendations.append("Consider feature selection or dimensionality reduction")

    # High dimensionality
    if n_features > 1000:
        score -= 0.05
        issues.append(f"High dimensionality ({n_features} features)")
        recommendations.append("Consider PCA, feature selection, or sparse models")

    # Outliers
    n_outlier_features = sum(1 for f in features if f.has_outliers)
    if n_outlier_features > n_features * 0.3:
        score -= 0.1
        issues.append(f"{n_outlier_features} features have outliers")
        recommendations.append("Use robust preprocessing (RobustScaler) or tree-based models")

    # Target distribution (regression)
    if is_regression and target_stats:
        std = target_stats.get("std", 0)
        mean = target_stats.get("mean", 0)
        if std > 0 and abs(mean) > 0:
            cv = std / abs(mean)
            if cv > 5:
                score -= 0.05
                issues.append(f"High target variance (CV={cv:.1f})")
                recommendations.append("Consider log-transforming the target")

    score = max(0.0, min(1.0, score))
    return score, issues, recommendations


def _isnan(arr: np.ndarray) -> np.ndarray:
    """NaN check that works for both numeric and object arrays."""
    try:
        return np.isnan(arr.astype(np.float64))
    except (ValueError, TypeError):
        return np.array([x is None or (isinstance(x, float) and math.isnan(x)) for x in arr])


def _has_nan(arr: np.ndarray) -> bool:
    try:
        return bool(np.any(np.isnan(arr.astype(np.float64))))
    except (ValueError, TypeError):
        return any(x is None or (isinstance(x, float) and math.isnan(x)) for x in arr)


def _is_numeric_col(col: np.ndarray) -> bool:
    """Check if a column is numeric."""
    if len(col) == 0:
        return True
    try:
        np.asarray(col, dtype=np.float64)
        return True
    except (ValueError, TypeError):
        return False


def _is_regression(y: np.ndarray, unique_y: np.ndarray) -> bool:
    """Heuristic: regression if target is float with many unique values."""
    if len(unique_y) > 20:
        try:
            y.astype(np.float64)
            return True
        except (ValueError, TypeError):
            return False
    return False
