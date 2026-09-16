"""Base adapter interface: what every modality adapter must provide.

An adapter bridges a specific ML modality (tabular, vision, text, audio, etc.)
into the unified research cycle. It provides:
  - Data loading and preprocessing
  - Feature extraction / representation
  - Task-appropriate evaluation metrics
  - Modality-specific proposal generation hints
  - Certification hooks (metric -> certifier mapping)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AdapterConfig:
    """Configuration for a modality adapter."""
    modality: str                        # "tabular" | "vision" | "text" | "audio" | "timeseries"
    task_type: str                       # "classification" | "regression" | "generation" | "retrieval"
    metric: str                          # default metric for this modality/task
    threshold: float                     # default threshold
    gpu_required: bool = False
    max_sequence_length: Optional[int] = None
    image_size: Optional[Tuple[int, int]] = None


class ModalityAdapter(ABC):
    """Abstract base for modality adapters.

    Each adapter handles the modality-specific parts of the research cycle:
      - Intake: how to profile this type of data
      - Featurization: how to convert raw data to model-ready features
      - Evaluation: which metrics and certifiers apply
      - Hints: what model families work for this modality
    """

    @abstractmethod
    def modality(self) -> str:
        """Return the modality name."""

    @abstractmethod
    def profile(self, data: Any) -> Dict[str, Any]:
        """Profile the data for this modality."""

    @abstractmethod
    def featurize(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from raw data. Returns (X, y)."""

    @abstractmethod
    def default_metric(self) -> str:
        """Default evaluation metric for this modality/task."""

    @abstractmethod
    def default_threshold(self, n_classes: int = 2) -> float:
        """Default certification threshold."""

    @abstractmethod
    def recommended_families(self) -> List[str]:
        """Model families recommended for this modality."""

    @abstractmethod
    def certifier_hook(self, metric: str):
        """Return the appropriate certifier function for this metric."""

    def gpu_required(self) -> bool:
        """Whether this modality typically requires GPU."""
        return False

    def extra_proposal_hints(self) -> List[str]:
        """Extra hints for LLM proposals specific to this modality."""
        return []


class TabularAdapter(ModalityAdapter):
    """Adapter for tabular (structured) data -- the primary modality."""

    def __init__(self, task: str = "classification"):
        self.task = task

    def modality(self) -> str:
        return "tabular"

    def profile(self, data: Any) -> Dict[str, Any]:
        from ..intake.profiler import profile_data
        if isinstance(data, tuple) and len(data) == 2:
            X, y = data
            p = profile_data(np.asarray(X), np.asarray(y))
            return p.to_dict()
        return {"error": "Expected (X, y) tuple"}

    def featurize(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(data, tuple) and len(data) == 2:
            return np.asarray(data[0]), np.asarray(data[1])
        raise ValueError("Expected (X, y) tuple")

    def default_metric(self) -> str:
        return "accuracy" if self.task == "classification" else "r2"

    def default_threshold(self, n_classes: int = 2) -> float:
        if self.task == "classification":
            chance = 1.0 / max(n_classes, 2)
            return round(chance + (1.0 - chance) * 0.5, 2)
        return 0.3

    def recommended_families(self) -> List[str]:
        return [
            "hist_gradient_boosting", "random_forest", "extra_trees",
            "logistic_regression", "stacking", "voting",
        ]

    def certifier_hook(self, metric: str):
        from attestra.core.science import certify_accuracy, certify_regression
        if metric in ("r2", "neg_rmse", "neg_mae"):
            return certify_regression
        return certify_accuracy


class TextClassificationAdapter(ModalityAdapter):
    """Adapter for text classification tasks."""

    def modality(self) -> str:
        return "text"

    def profile(self, data: Any) -> Dict[str, Any]:
        if isinstance(data, tuple) and len(data) == 2:
            texts, labels = data
            return {
                "n_samples": len(texts),
                "n_classes": len(set(labels)),
                "avg_length": np.mean([len(str(t)) for t in texts]) if texts else 0,
                "modality": "text_classification",
            }
        return {"error": "Expected (texts, labels) tuple"}

    def featurize(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(data, tuple) and len(data) == 2:
            texts, labels = data
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = vectorizer.fit_transform(texts).toarray()
            y = np.asarray(labels)
            return X, y
        raise ValueError("Expected (texts, labels) tuple")

    def default_metric(self) -> str:
        return "macro_f1"

    def default_threshold(self, n_classes: int = 2) -> float:
        chance = 1.0 / max(n_classes, 2)
        return round(chance + (1.0 - chance) * 0.4, 2)

    def recommended_families(self) -> List[str]:
        return [
            "tfidf_lr", "tfidf_svm", "tfidf_hist_gbm",
            "transformer_finetune",
        ]

    def certifier_hook(self, metric: str):
        from attestra.core.science import certify_accuracy
        return certify_accuracy

    def gpu_required(self) -> bool:
        return False  # TF-IDF + sklearn doesn't need GPU

    def extra_proposal_hints(self) -> List[str]:
        return [
            "TF-IDF with subword n-grams often matches transformers on small data",
            "Consider character-level features for morphologically rich languages",
        ]


class VisionAdapter(ModalityAdapter):
    """Adapter for vision (image classification) tasks."""

    def modality(self) -> str:
        return "vision"

    def profile(self, data: Any) -> Dict[str, Any]:
        if isinstance(data, tuple) and len(data) == 2:
            images, labels = data
            images = np.asarray(images)
            return {
                "n_samples": len(images),
                "n_classes": len(set(np.asarray(labels).tolist())),
                "image_shape": images.shape[1:] if images.ndim > 1 else None,
                "modality": "vision",
            }
        return {"error": "Expected (images, labels) tuple"}

    def featurize(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(data, tuple) and len(data) == 2:
            images, labels = data
            images = np.asarray(images)
            if images.ndim > 2:
                X = images.reshape(len(images), -1)
            else:
                X = images
            y = np.asarray(labels)
            return X, y
        raise ValueError("Expected (images, labels) tuple")

    def default_metric(self) -> str:
        return "accuracy"

    def default_threshold(self, n_classes: int = 2) -> float:
        chance = 1.0 / max(n_classes, 2)
        return round(chance + (1.0 - chance) * 0.4, 2)

    def recommended_families(self) -> List[str]:
        return [
            "transfer_resnet", "transfer_vit", "flat_hist_gbm",
            "pca_svm", "random_features_lr",
        ]

    def certifier_hook(self, metric: str):
        from attestra.core.science import certify_accuracy
        return certify_accuracy

    def gpu_required(self) -> bool:
        return True

    def extra_proposal_hints(self) -> List[str]:
        return [
            "Transfer learning from pretrained models dominates for small datasets",
            "Flatten + classical ML is a strong baseline for <1000 images",
        ]


class TimeseriesAdapter(ModalityAdapter):
    """Adapter for timeseries tasks (classification and forecasting)."""

    def modality(self) -> str:
        return "timeseries"

    def profile(self, data: Any) -> Dict[str, Any]:
        if isinstance(data, tuple) and len(data) == 2:
            X, y = np.asarray(data[0]), np.asarray(data[1])
            return {
                "n_samples": X.shape[0],
                "n_features": X.shape[1] if X.ndim > 1 else 1,
                "sequence_length": X.shape[2] if X.ndim > 2 else X.shape[1] if X.ndim > 1 else len(X),
                "modality": "timeseries",
            }
        return {"error": "Expected (X, y) tuple"}

    def featurize(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(data, tuple) and len(data) == 2:
            X, y = np.asarray(data[0]), np.asarray(data[1])
            if X.ndim > 2:
                X = X.reshape(len(X), -1)
            return X, y
        raise ValueError("Expected (X, y) tuple")

    def default_metric(self) -> str:
        return "accuracy"

    def default_threshold(self, n_classes: int = 2) -> float:
        chance = 1.0 / max(n_classes, 2)
        return round(chance + (1.0 - chance) * 0.3, 2)

    def recommended_families(self) -> List[str]:
        return [
            "tsfresh_hist_gbm", "rocket_lr", "flat_rf",
            "statistical_features_svm",
        ]

    def certifier_hook(self, metric: str):
        from attestra.core.science import certify_accuracy
        return certify_accuracy

    def extra_proposal_hints(self) -> List[str]:
        return [
            "ROCKET features + linear model is SOTA-competitive and very fast",
            "Temporal splits required (no future data in training)",
        ]


# ============================================================================== registry

_ADAPTERS: Dict[str, type] = {
    "tabular": TabularAdapter,
    "text": TextClassificationAdapter,
    "vision": VisionAdapter,
    "timeseries": TimeseriesAdapter,
}


def get_adapter(modality: str, **kwargs) -> ModalityAdapter:
    """Get a modality adapter by name."""
    if modality not in _ADAPTERS:
        raise ValueError(f"Unknown modality: {modality}. Available: {list(_ADAPTERS.keys())}")
    return _ADAPTERS[modality](**kwargs)


def register_adapter(modality: str, adapter_class: type) -> None:
    """Register a new modality adapter."""
    _ADAPTERS[modality] = adapter_class


def available_modalities() -> List[str]:
    return list(_ADAPTERS.keys())
