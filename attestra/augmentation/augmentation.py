"""Data augmentation and synthetic data generation.

Strategies:
  1. SMOTE / ADASYN for class imbalance (tabular)
  2. Gaussian noise injection (tabular)
  3. Feature-space mixup (interpolation between samples)
  4. Conditional generation via LLM (for complex domains)
  5. Bootstrap resampling (non-parametric)
  6. Copula-based generation (preserves correlations)
  7. Variational sampling (feature distributions)

Design: augmentation is a PROPOSAL SOURCE — it proposes data transforms
that the research engine evaluates like any other proposal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AugmentationResult:
    """Result of applying an augmentation strategy."""
    strategy: str
    X_aug: np.ndarray
    y_aug: np.ndarray
    n_original: int
    n_generated: int
    metadata: Dict = field(default_factory=dict)


class DataAugmenter:
    """Data augmentation engine.

    Usage:
        aug = DataAugmenter(X_train, y_train, task="classification")
        results = aug.apply_all(strategies=["smote", "noise", "mixup"])
        # Returns list of (X_augmented, y_augmented) for each strategy
    """

    STRATEGIES = [
        "smote", "adasyn", "noise", "mixup", "bootstrap",
        "copula", "variational", "borderline_smote",
    ]

    def __init__(self, X: np.ndarray, y: np.ndarray, task: str = "classification",
                 seed: int = 42):
        self.X = X
        self.y = y
        self.task = task
        self.rng = np.random.default_rng(seed)
        self.n_samples, self.n_features = X.shape

    def apply(self, strategy: str, **kwargs) -> AugmentationResult:
        """Apply a single augmentation strategy."""
        fn = getattr(self, f"_aug_{strategy}", None)
        if fn is None:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {self.STRATEGIES}")
        X_aug, y_aug, meta = fn(**kwargs)
        return AugmentationResult(
            strategy=strategy,
            X_aug=X_aug,
            y_aug=y_aug,
            n_original=self.n_samples,
            n_generated=len(X_aug) - self.n_samples,
            metadata=meta,
        )

    def apply_all(self, strategies: Optional[List[str]] = None,
                  **kwargs) -> List[AugmentationResult]:
        """Apply multiple strategies, return all results."""
        if strategies is None:
            strategies = self._recommend_strategies()
        results = []
        for strat in strategies:
            try:
                results.append(self.apply(strat, **kwargs))
            except Exception:
                continue
        return results

    def recommend_strategies(self) -> List[str]:
        """Recommend augmentation strategies based on data characteristics."""
        return self._recommend_strategies()

    # ========================================================================== strategies

    def _aug_smote(self, k: int = 5, ratio: float = 1.0, **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """SMOTE: Synthetic Minority Over-sampling Technique."""
        if self.task != "classification":
            return self._aug_noise()

        classes, counts = np.unique(self.y, return_counts=True)
        max_count = int(counts.max() * ratio)
        X_new_all = [self.X.copy()]
        y_new_all = [self.y.copy()]

        for cls, count in zip(classes, counts):
            if count >= max_count:
                continue
            n_to_generate = max_count - count
            cls_mask = self.y == cls
            X_cls = self.X[cls_mask]
            n_cls = len(X_cls)
            if n_cls < 2:
                continue

            # Generate synthetic samples
            k_actual = min(k, n_cls - 1)
            synthetic = []
            for _ in range(n_to_generate):
                idx = self.rng.integers(0, n_cls)
                # Find k nearest neighbors
                dists = np.linalg.norm(X_cls - X_cls[idx], axis=1)
                nn_indices = np.argsort(dists)[1:k_actual + 1]
                nn_idx = self.rng.choice(nn_indices)
                # Interpolate
                lam = self.rng.random()
                new_sample = X_cls[idx] + lam * (X_cls[nn_idx] - X_cls[idx])
                synthetic.append(new_sample)

            if synthetic:
                X_new_all.append(np.array(synthetic))
                y_new_all.append(np.full(len(synthetic), cls))

        X_aug = np.vstack(X_new_all)
        y_aug = np.concatenate(y_new_all)
        return X_aug, y_aug, {"method": "smote", "k": k}

    def _aug_adasyn(self, k: int = 5, **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """ADASYN: Adaptive Synthetic Sampling."""
        if self.task != "classification":
            return self._aug_noise()

        classes, counts = np.unique(self.y, return_counts=True)
        max_count = counts.max()
        X_new_all = [self.X.copy()]
        y_new_all = [self.y.copy()]

        for cls, count in zip(classes, counts):
            if count >= max_count * 0.9:
                continue
            n_to_generate = max_count - count
            cls_mask = self.y == cls
            X_cls = self.X[cls_mask]
            n_cls = len(X_cls)
            if n_cls < 2:
                continue

            k_actual = min(k, n_cls - 1)
            # Compute density ratio per sample (ADASYN weighting)
            ratios = np.zeros(n_cls)
            for i in range(n_cls):
                dists = np.linalg.norm(self.X - X_cls[i], axis=1)
                nn_indices = np.argsort(dists)[1:k_actual + 1]
                n_majority = np.sum(self.y[nn_indices] != cls)
                ratios[i] = n_majority / k_actual

            # Normalize ratios
            if ratios.sum() > 0:
                ratios = ratios / ratios.sum()
            else:
                ratios = np.ones(n_cls) / n_cls

            # Generate proportional to difficulty
            synthetic = []
            for i in range(n_cls):
                n_gen = int(np.round(n_to_generate * ratios[i]))
                for _ in range(n_gen):
                    dists = np.linalg.norm(X_cls - X_cls[i], axis=1)
                    nn_idx = self.rng.choice(np.argsort(dists)[1:k_actual + 1])
                    lam = self.rng.random()
                    new_sample = X_cls[i] + lam * (X_cls[nn_idx] - X_cls[i])
                    synthetic.append(new_sample)

            if synthetic:
                X_new_all.append(np.array(synthetic[:n_to_generate]))
                y_new_all.append(np.full(min(len(synthetic), n_to_generate), cls))

        X_aug = np.vstack(X_new_all)
        y_aug = np.concatenate(y_new_all)
        return X_aug, y_aug, {"method": "adasyn", "k": k}

    def _aug_noise(self, scale: float = 0.1, n_copies: int = 1, **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Gaussian noise injection."""
        X_new = [self.X.copy()]
        y_new = [self.y.copy()]

        for _ in range(n_copies):
            # Scale noise relative to feature std
            stds = np.std(self.X, axis=0)
            stds = np.where(stds > 0, stds, 1.0)
            noise = self.rng.normal(0, scale * stds, size=self.X.shape)
            X_new.append(self.X + noise)
            y_new.append(self.y.copy())

        X_aug = np.vstack(X_new)
        y_aug = np.concatenate(y_new)
        return X_aug, y_aug, {"method": "noise", "scale": scale, "copies": n_copies}

    def _aug_mixup(self, alpha: float = 0.2, n_samples: Optional[int] = None, **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Feature-space mixup (Zhang et al., 2018)."""
        if n_samples is None:
            n_samples = self.n_samples

        X_new = [self.X.copy()]
        y_new = [self.y.copy()]

        synthetic_X = []
        synthetic_y = []
        for _ in range(n_samples):
            i, j = self.rng.integers(0, self.n_samples, size=2)
            lam = self.rng.beta(alpha, alpha)
            x_mix = lam * self.X[i] + (1 - lam) * self.X[j]
            synthetic_X.append(x_mix)
            # For classification, use the dominant class
            if self.task == "classification":
                synthetic_y.append(self.y[i] if lam >= 0.5 else self.y[j])
            else:
                synthetic_y.append(lam * self.y[i] + (1 - lam) * self.y[j])

        X_new.append(np.array(synthetic_X))
        y_new.append(np.array(synthetic_y))

        X_aug = np.vstack(X_new)
        y_aug = np.concatenate(y_new)
        return X_aug, y_aug, {"method": "mixup", "alpha": alpha}

    def _aug_bootstrap(self, n_samples: Optional[int] = None, **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Bootstrap resampling (with replacement)."""
        if n_samples is None:
            n_samples = self.n_samples

        indices = self.rng.choice(self.n_samples, size=n_samples, replace=True)
        X_aug = np.vstack([self.X, self.X[indices]])
        y_aug = np.concatenate([self.y, self.y[indices]])
        return X_aug, y_aug, {"method": "bootstrap", "n_resampled": n_samples}

    def _aug_copula(self, n_samples: Optional[int] = None, **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Gaussian copula-based synthetic generation (preserves correlations)."""
        if n_samples is None:
            n_samples = self.n_samples

        # Compute empirical CDF for each feature → uniform marginals
        from scipy import stats

        n_features = self.X.shape[1]
        # Rank-transform to uniform
        U = np.zeros_like(self.X)
        for j in range(n_features):
            ranks = stats.rankdata(self.X[:, j])
            U[:, j] = ranks / (self.n_samples + 1)  # avoid 0/1

        # Transform to normal
        Z = stats.norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))
        # Correlation matrix
        corr = np.corrcoef(Z.T)
        # Handle degenerate cases
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        # Generate from multivariate normal
        try:
            Z_new = self.rng.multivariate_normal(np.zeros(n_features), corr, size=n_samples)
        except np.linalg.LinAlgError:
            # Fallback: add regularization
            corr += np.eye(n_features) * 0.01
            Z_new = self.rng.multivariate_normal(np.zeros(n_features), corr, size=n_samples)

        # Transform back through empirical quantiles
        X_new = np.zeros((n_samples, n_features))
        for j in range(n_features):
            U_new = stats.norm.cdf(Z_new[:, j])
            X_new[:, j] = np.quantile(self.X[:, j], np.clip(U_new, 0, 1))

        # For labels: nearest-neighbor assignment
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if self.task == "classification":
            knn = KNeighborsClassifier(n_neighbors=min(5, self.n_samples))
            knn.fit(self.X, self.y)
            y_new = knn.predict(X_new)
        else:
            knn = KNeighborsRegressor(n_neighbors=min(5, self.n_samples))
            knn.fit(self.X, self.y)
            y_new = knn.predict(X_new)

        X_aug = np.vstack([self.X, X_new])
        y_aug = np.concatenate([self.y, y_new])
        return X_aug, y_aug, {"method": "copula", "n_generated": n_samples}

    def _aug_variational(self, n_samples: Optional[int] = None, **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Variational sampling from per-feature distributions."""
        if n_samples is None:
            n_samples = self.n_samples

        X_new = np.zeros((n_samples, self.n_features))
        for j in range(self.n_features):
            col = self.X[:, j]
            mu = np.mean(col)
            std = np.std(col)
            if std < 1e-10:
                X_new[:, j] = mu
            else:
                X_new[:, j] = self.rng.normal(mu, std, size=n_samples)

        # Label assignment via nearest neighbor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if self.task == "classification":
            knn = KNeighborsClassifier(n_neighbors=min(5, self.n_samples))
            knn.fit(self.X, self.y)
            y_new = knn.predict(X_new)
        else:
            knn = KNeighborsRegressor(n_neighbors=min(5, self.n_samples))
            knn.fit(self.X, self.y)
            y_new = knn.predict(X_new)

        X_aug = np.vstack([self.X, X_new])
        y_aug = np.concatenate([self.y, y_new])
        return X_aug, y_aug, {"method": "variational", "n_generated": n_samples}

    def _aug_borderline_smote(self, k: int = 5, **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Borderline-SMOTE: only oversample near decision boundary."""
        if self.task != "classification":
            return self._aug_noise()

        classes, counts = np.unique(self.y, return_counts=True)
        max_count = counts.max()
        X_new_all = [self.X.copy()]
        y_new_all = [self.y.copy()]

        for cls, count in zip(classes, counts):
            if count >= max_count * 0.9:
                continue
            n_to_generate = max_count - count
            cls_mask = self.y == cls
            X_cls = self.X[cls_mask]
            n_cls = len(X_cls)
            if n_cls < 2:
                continue

            k_actual = min(k, self.n_samples - 1)
            # Find borderline samples
            borderline = []
            for i in range(n_cls):
                dists = np.linalg.norm(self.X - X_cls[i], axis=1)
                nn_indices = np.argsort(dists)[1:k_actual + 1]
                n_other = np.sum(self.y[nn_indices] != cls)
                # Borderline: between k/4 and 3k/4 neighbors from other class
                if k_actual // 4 <= n_other <= 3 * k_actual // 4:
                    borderline.append(i)

            if not borderline:
                borderline = list(range(n_cls))  # fallback to all

            synthetic = []
            k_cls = min(k, n_cls - 1)
            for _ in range(n_to_generate):
                idx = self.rng.choice(borderline)
                dists = np.linalg.norm(X_cls - X_cls[idx], axis=1)
                nn_idx = self.rng.choice(np.argsort(dists)[1:k_cls + 1])
                lam = self.rng.random()
                new_sample = X_cls[idx] + lam * (X_cls[nn_idx] - X_cls[idx])
                synthetic.append(new_sample)

            if synthetic:
                X_new_all.append(np.array(synthetic))
                y_new_all.append(np.full(len(synthetic), cls))

        X_aug = np.vstack(X_new_all)
        y_aug = np.concatenate(y_new_all)
        return X_aug, y_aug, {"method": "borderline_smote", "k": k}

    # ========================================================================== recommendation

    def _recommend_strategies(self) -> List[str]:
        """Recommend strategies based on data characteristics."""
        strategies = []

        if self.task == "classification":
            classes, counts = np.unique(self.y, return_counts=True)
            imbalance_ratio = counts.max() / max(counts.min(), 1)
            if imbalance_ratio > 3:
                strategies.extend(["smote", "borderline_smote", "adasyn"])
            elif imbalance_ratio > 1.5:
                strategies.append("smote")

        if self.n_samples < 200:
            strategies.extend(["noise", "mixup", "bootstrap"])
        elif self.n_samples < 1000:
            strategies.extend(["noise", "mixup"])

        if self.n_features > 10:
            strategies.append("copula")

        if not strategies:
            strategies = ["noise", "mixup"]

        return list(dict.fromkeys(strategies))  # deduplicate preserving order
