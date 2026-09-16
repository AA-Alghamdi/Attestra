"""Data-as-variable experiments — data is the experimental variable, not just models.

Supports:
  - DataExperiment: test a data transformation hypothesis (augment, clean, resplit, etc.)
  - DataTransform: typed transformations with audit trail
  - A/B comparison between control (original data) and treatment (transformed data)

Use cases:
  - "Does adding synthetic data help?" (augmentation experiment)
  - "Does removing outliers help?" (cleaning experiment)
  - "Does a different split strategy change the conclusion?" (methodology experiment)
  - "Does varying tone dimensionality in training data improve TTS?" (data-centric research)
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .versioning import DataFingerprint, DataVersion


# ---------------------------------------------------------------------------
# Transform types
# ---------------------------------------------------------------------------

class TransformType(str, Enum):
    AUGMENT = "augment"           # add synthetic samples
    CLEAN = "clean"               # remove noise/outliers
    RESPLIT = "resplit"           # different split strategy
    SUBSAMPLE = "subsample"       # reduce dataset size
    RELABEL = "relabel"           # correct/change labels
    FEATURE_SELECT = "feature_select"  # select/drop features
    FEATURE_ENGINEER = "feature_engineer"  # create new features
    NORMALIZE = "normalize"       # scale/transform features
    BALANCE = "balance"           # balance class distribution
    CUSTOM = "custom"             # user-defined transform


@dataclass
class DataTransform:
    """A typed data transformation with parameters and provenance."""
    transform_type: TransformType
    name: str                    # human-readable name
    params: Dict = field(default_factory=dict)
    description: str = ""
    # The transform function (not serialized)
    _fn: Optional[Callable] = field(default=None, repr=False)

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply this transform to data."""
        if self._fn is not None:
            return self._fn(X, y, **self.params)
        # Built-in transforms
        return _apply_builtin_transform(self.transform_type, X, y, self.params)

    def to_dict(self) -> Dict:
        return {
            "transform_type": self.transform_type.value,
            "name": self.name,
            "params": self.params,
            "description": self.description,
        }


def _apply_builtin_transform(
    ttype: TransformType, X: np.ndarray, y: np.ndarray, params: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a built-in transform."""
    if ttype == TransformType.SUBSAMPLE:
        frac = params.get("fraction", 0.5)
        n = max(10, int(len(X) * frac))
        idx = np.random.choice(len(X), size=n, replace=False)
        return X[idx], y[idx]

    elif ttype == TransformType.CLEAN:
        # Remove outliers using IQR
        from scipy import stats
        z = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
        threshold = params.get("z_threshold", 3.0)
        mask = (z < threshold).all(axis=1)
        if mask.sum() < 10:
            return X, y  # don't remove too many
        return X[mask], y[mask]

    elif ttype == TransformType.BALANCE:
        # Oversample minority classes
        from collections import Counter
        counts = Counter(y)
        max_count = max(counts.values())
        X_new, y_new = [X], [y]
        for cls, cnt in counts.items():
            if cnt < max_count:
                deficit = max_count - cnt
                cls_idx = np.where(y == cls)[0]
                extra_idx = np.random.choice(cls_idx, size=deficit, replace=True)
                X_new.append(X[extra_idx])
                y_new.append(y[extra_idx])
        return np.vstack(X_new), np.concatenate(y_new)

    elif ttype == TransformType.FEATURE_SELECT:
        # Keep top-k features by variance
        k = params.get("k", min(20, X.shape[1]))
        variances = np.var(X, axis=0)
        top_k = np.argsort(variances)[-k:]
        return X[:, top_k], y

    elif ttype == TransformType.NORMALIZE:
        method = params.get("method", "standard")
        if method == "standard":
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-8
            return (X - mean) / std, y
        elif method == "minmax":
            xmin = X.min(axis=0)
            xmax = X.max(axis=0)
            denom = (xmax - xmin) + 1e-8
            return (X - xmin) / denom, y
        return X, y

    # Default: return unchanged
    return X, y


# ---------------------------------------------------------------------------
# Data Experiment
# ---------------------------------------------------------------------------

@dataclass
class DataExperiment:
    """An experiment where data is the variable being tested.

    Structure:
      - base_data: the original/control data version
      - transformation: what we're testing
      - hypothesis: why we think this will help
      - decision_rule: how we decide if it worked
    """
    experiment_id: str
    hypothesis: str
    transformation: DataTransform
    decision_rule: str           # "lift > X% on sealed test" or "p_value < 0.05"
    # Data versions
    base_version: Optional[DataVersion] = None
    treatment_version: Optional[DataVersion] = None
    # Results
    control_score: Optional[float] = None
    treatment_score: Optional[float] = None
    lift: Optional[float] = None
    p_value: Optional[float] = None
    decision: Optional[str] = None  # "accept" | "reject" | "inconclusive"
    # Metadata
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    model_used: str = ""         # what model was held constant
    metric: str = "accuracy"
    notes: str = ""

    @staticmethod
    def create(
        hypothesis: str,
        transformation: DataTransform,
        decision_rule: str = "lift > 2% on sealed test",
        metric: str = "accuracy",
    ) -> "DataExperiment":
        """Create a new data experiment."""
        exp_id = hashlib.sha256(
            f"{hypothesis}:{transformation.name}:{time.time()}".encode()
        ).hexdigest()[:12]
        return DataExperiment(
            experiment_id=exp_id,
            hypothesis=hypothesis,
            transformation=transformation,
            decision_rule=decision_rule,
            metric=metric,
        )

    def run(
        self,
        X: np.ndarray, y: np.ndarray,
        model_fn: Callable,
        eval_fn: Callable,
        seed: int = 42,
    ) -> "DataExperiment":
        """Run the data experiment: control vs treatment with same model.

        Parameters
        ----------
        X, y : arrays
            The base data.
        model_fn : callable
            (X_train, y_train) -> fitted_model. Must be deterministic given seed.
        eval_fn : callable
            (model, X_test, y_test) -> score. The evaluation metric.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        self : updated with results.
        """
        rng = np.random.RandomState(seed)

        # Create base version
        self.base_version = DataVersion.create(X, y, description="control/base data")

        # Split for evaluation (held-out test)
        n = len(X)
        idx = rng.permutation(n)
        split_point = int(0.8 * n)
        train_idx, test_idx = idx[:split_point], idx[split_point:]

        X_train_base, y_train_base = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Control: train on base data, evaluate on held-out
        model_control = model_fn(X_train_base, y_train_base)
        self.control_score = eval_fn(model_control, X_test, y_test)

        # Apply transformation to training data only (test stays the same)
        X_train_treat, y_train_treat = self.transformation.apply(X_train_base, y_train_base)
        self.treatment_version = DataVersion.create(
            X_train_treat, y_train_treat,
            parent=self.base_version,
            transformations=[self.transformation.to_dict()],
            description=f"treatment: {self.transformation.name}",
        )

        # Treatment: train on transformed data, evaluate on same held-out
        model_treatment = model_fn(X_train_treat, y_train_treat)
        self.treatment_score = eval_fn(model_treatment, X_test, y_test)

        # Compute lift
        if self.control_score and self.control_score > 0:
            self.lift = (self.treatment_score - self.control_score) / self.control_score
        else:
            self.lift = self.treatment_score - self.control_score if self.treatment_score else 0

        # Simple decision (can be made more sophisticated with bootstrap)
        self.decision = self._evaluate_decision_rule()
        self.completed_at = time.time()

        return self

    def _evaluate_decision_rule(self) -> str:
        """Evaluate the decision rule against results."""
        if self.lift is None:
            return "inconclusive"

        # Parse simple rules
        rule = self.decision_rule.lower()
        if "lift >" in rule:
            # Extract threshold
            try:
                parts = rule.split("lift >")
                threshold_str = parts[1].strip().split("%")[0].strip()
                threshold = float(threshold_str) / 100.0
                return "accept" if self.lift > threshold else "reject"
            except (ValueError, IndexError):
                pass

        # Default: any positive lift with > 1% margin
        if self.lift > 0.01:
            return "accept"
        elif self.lift < -0.01:
            return "reject"
        return "inconclusive"

    def summary(self) -> str:
        """Human-readable summary of the experiment."""
        lines = [
            f"DataExperiment: {self.experiment_id}",
            f"  Hypothesis: {self.hypothesis}",
            f"  Transform: {self.transformation.name} ({self.transformation.transform_type.value})",
            f"  Decision rule: {self.decision_rule}",
        ]
        if self.control_score is not None:
            lines.append(f"  Control score: {self.control_score:.4f}")
            lines.append(f"  Treatment score: {self.treatment_score:.4f}")
            lines.append(f"  Lift: {self.lift:+.4f} ({self.lift*100:+.1f}%)")
            lines.append(f"  Decision: {self.decision}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data Experiment Manager
# ---------------------------------------------------------------------------

class DataExperimentManager:
    """Manages multiple data experiments with tracking."""

    def __init__(self):
        self._experiments: List[DataExperiment] = []
        self._results: List[Dict] = []

    def propose_experiments(
        self,
        X: np.ndarray, y: np.ndarray,
        task_type: str = "classification",
        llm_call: Optional[Callable] = None,
    ) -> List[DataExperiment]:
        """Propose data experiments based on data profile.

        Returns a set of candidate data experiments ranked by expected impact.
        """
        proposals = []

        n_samples, n_features = X.shape if X.ndim > 1 else (len(X), 1)
        n_classes = len(np.unique(y)) if task_type != "regression" else 0

        # Class imbalance → balance experiment
        if n_classes > 0:
            from collections import Counter
            counts = Counter(y)
            ratio = max(counts.values()) / max(min(counts.values()), 1)
            if ratio > 2.0:
                proposals.append(DataExperiment.create(
                    hypothesis=f"Class imbalance (ratio={ratio:.1f}) limits minority recall; "
                              "oversampling will improve macro-F1",
                    transformation=DataTransform(
                        transform_type=TransformType.BALANCE,
                        name="oversample_minority",
                        description="Oversample minority classes to match majority count",
                    ),
                    decision_rule="lift > 1% on sealed test",
                    metric="accuracy",
                ))

        # High dimensionality → feature selection
        if n_features > 50:
            proposals.append(DataExperiment.create(
                hypothesis=f"High dimensionality ({n_features} features) causes overfitting; "
                          "selecting top-20 by variance will improve generalization",
                transformation=DataTransform(
                    transform_type=TransformType.FEATURE_SELECT,
                    name="variance_top_20",
                    params={"k": 20},
                    description="Select top 20 features by variance",
                ),
                decision_rule="lift > 1% on sealed test",
                metric="accuracy",
            ))

        # Outliers → cleaning experiment
        if n_samples > 50:
            proposals.append(DataExperiment.create(
                hypothesis="Outliers distort model training; "
                          "removing z>3 samples will reduce noise and improve accuracy",
                transformation=DataTransform(
                    transform_type=TransformType.CLEAN,
                    name="remove_outliers_z3",
                    params={"z_threshold": 3.0},
                    description="Remove samples with any feature z-score > 3",
                ),
                decision_rule="lift > 1% on sealed test",
                metric="accuracy",
            ))

        # Normalization → always worth trying
        proposals.append(DataExperiment.create(
            hypothesis="Feature scaling differences hurt distance-based models; "
                      "standardization will improve model convergence",
            transformation=DataTransform(
                transform_type=TransformType.NORMALIZE,
                name="standardize",
                params={"method": "standard"},
                description="Z-score normalization (mean=0, std=1)",
            ),
            decision_rule="lift > 0.5% on sealed test",
            metric="accuracy",
        ))

        return proposals

    def run_experiments(
        self,
        experiments: List[DataExperiment],
        X: np.ndarray, y: np.ndarray,
        model_fn: Callable,
        eval_fn: Callable,
        seed: int = 42,
    ) -> List[DataExperiment]:
        """Run a batch of data experiments and return results."""
        results = []
        for exp in experiments:
            try:
                exp.run(X, y, model_fn, eval_fn, seed=seed)
                results.append(exp)
                self._experiments.append(exp)
                self._results.append({
                    "experiment_id": exp.experiment_id,
                    "hypothesis": exp.hypothesis,
                    "transform": exp.transformation.name,
                    "control_score": exp.control_score,
                    "treatment_score": exp.treatment_score,
                    "lift": exp.lift,
                    "decision": exp.decision,
                })
            except Exception as e:
                exp.decision = "error"
                exp.notes = str(e)
                results.append(exp)
        return results

    def best_transforms(self) -> List[DataTransform]:
        """Return transforms that were accepted (produced positive lift)."""
        accepted = [e for e in self._experiments if e.decision == "accept"]
        # Sort by lift descending
        accepted.sort(key=lambda e: e.lift or 0, reverse=True)
        return [e.transformation for e in accepted]

    def summary(self) -> str:
        """Summary of all experiments run."""
        lines = [f"DataExperimentManager: {len(self._experiments)} experiments"]
        for exp in self._experiments:
            status = exp.decision or "pending"
            lift_str = f"{exp.lift*100:+.1f}%" if exp.lift is not None else "n/a"
            lines.append(f"  [{status}] {exp.transformation.name}: lift={lift_str}")
        return "\n".join(lines)
