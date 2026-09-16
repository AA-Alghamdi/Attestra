"""Strategy Router (Level 2) — routes to the right strategy class.

Given:
  - problem_type + data_profile + history of what's been tried
Decides:
  - Which strategy class to use (gradient boosting, neural, ensemble, feature-heavy, etc.)
  - Informed by the meta-learner's past experience
  - Incorporates avoid_list (strategies known to fail on similar tasks)

This is where the meta-learner's rank_strategies() and avoid_list() are CONSUMED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class StrategyClass(str, Enum):
    """High-level strategy classes."""
    GRADIENT_BOOSTING = "gradient_boosting"    # GBM, XGBoost, LightGBM, CatBoost
    LINEAR = "linear"                          # Ridge, Lasso, ElasticNet, LogReg
    TREE_ENSEMBLE = "tree_ensemble"            # Random Forest, Extra Trees
    SVM = "svm"                                # SVM, kernel methods
    NEURAL_TABULAR = "neural_tabular"          # TabNet, FT-Transformer, deep tabular
    NEURAL_VISION = "neural_vision"            # CNN, ViT, EfficientNet
    NEURAL_NLP = "neural_nlp"                  # Transformers, BERT, GPT
    NEURAL_TIMESERIES = "neural_timeseries"    # LSTM, N-BEATS, Temporal Fusion Transformer
    FEATURE_HEAVY = "feature_heavy"            # PCA + model, polynomial features, AutoFeat
    ENSEMBLE_STACKING = "ensemble_stacking"    # Stack top-K models
    DATA_CENTRIC = "data_centric"              # Focus on data quality, not model
    EXPLORATORY = "exploratory"                # LLM-generated novel approaches


@dataclass
class StrategyRecommendation:
    """A recommended strategy with score and reasoning."""
    strategy_class: StrategyClass
    confidence: float           # 0-1, how confident the router is
    reasoning: str             # why this strategy was chosen
    model_families: List[str]  # specific model families to try
    params: Dict = field(default_factory=dict)  # strategy-specific parameters


class StrategyRouter:
    """Routes to the optimal strategy class based on task profile and history.

    Consumes:
      - Meta-learner's rank_strategies() output
      - Meta-learner's avoid_list()
      - Data profile (n_samples, n_features, task_type, etc.)
      - History of what's been tried in this experiment
    """

    def __init__(self, meta_learner: Optional[Any] = None):
        """Initialize with optional meta-learner for history-informed routing."""
        self._meta_learner = meta_learner
        # Default strategy priorities by task type
        self._defaults = {
            "binary": [
                StrategyClass.GRADIENT_BOOSTING,
                StrategyClass.TREE_ENSEMBLE,
                StrategyClass.LINEAR,
                StrategyClass.NEURAL_TABULAR,
            ],
            "multiclass": [
                StrategyClass.GRADIENT_BOOSTING,
                StrategyClass.TREE_ENSEMBLE,
                StrategyClass.LINEAR,
                StrategyClass.NEURAL_TABULAR,
            ],
            "regression": [
                StrategyClass.GRADIENT_BOOSTING,
                StrategyClass.TREE_ENSEMBLE,
                StrategyClass.LINEAR,
                StrategyClass.SVM,
            ],
            "text": [
                StrategyClass.NEURAL_NLP,
                StrategyClass.LINEAR,
                StrategyClass.GRADIENT_BOOSTING,
            ],
            "vision": [
                StrategyClass.NEURAL_VISION,
            ],
            "timeseries": [
                StrategyClass.NEURAL_TIMESERIES,
                StrategyClass.GRADIENT_BOOSTING,
                StrategyClass.LINEAR,
            ],
        }

    def route(
        self,
        task_type: str,
        n_samples: int,
        n_features: int,
        *,
        n_classes: int = 2,
        tried_strategies: Optional[List[str]] = None,
        quality_score: float = 1.0,
        budget_remaining_s: float = 300.0,
    ) -> List[StrategyRecommendation]:
        """Route to recommended strategies, ranked by expected success.

        Parameters
        ----------
        task_type : str
            Problem type ("binary", "multiclass", "regression", "text", "vision", etc.)
        n_samples : int
            Number of training samples.
        n_features : int
            Number of features.
        n_classes : int
            Number of classes (for classification).
        tried_strategies : list, optional
            Strategies already tried (to avoid repeating).
        quality_score : float
            Data quality score (0-1, higher is cleaner data).
        budget_remaining_s : float
            Remaining compute budget in seconds.

        Returns
        -------
        List[StrategyRecommendation]
            Ranked list of strategy recommendations.
        """
        tried = set(tried_strategies or [])
        recommendations = []

        # Get meta-learner recommendations if available
        meta_recs = []
        avoid = []
        if self._meta_learner:
            try:
                meta_recs = self._meta_learner.rank_strategies(task_type, n_samples, n_features)
                avoid = self._meta_learner.avoid_list(task_type)
            except Exception:
                pass

        # Strategy selection logic based on data characteristics
        candidates = self._select_candidates(
            task_type, n_samples, n_features, n_classes,
            quality_score, budget_remaining_s,
        )

        # Merge with meta-learner recommendations
        if meta_recs:
            # Meta-learner has history: use its ranking as primary signal
            for name, score in meta_recs:
                strategy_class = self._name_to_class(name)
                if strategy_class and strategy_class.value not in tried:
                    if name not in avoid:
                        recommendations.append(StrategyRecommendation(
                            strategy_class=strategy_class,
                            confidence=min(score, 1.0),
                            reasoning=f"Meta-learner: score={score:.3f} on similar tasks",
                            model_families=self._class_to_families(strategy_class),
                        ))

        # Add rule-based candidates that meta-learner missed
        for sc, conf, reason in candidates:
            if sc.value not in tried and sc not in [r.strategy_class for r in recommendations]:
                # Check avoid list
                if any(sc.value in a for a in avoid):
                    continue
                recommendations.append(StrategyRecommendation(
                    strategy_class=sc,
                    confidence=conf,
                    reasoning=reason,
                    model_families=self._class_to_families(sc),
                ))

        # Sort by confidence (meta-learner scores + rule-based confidence)
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return recommendations

    def _select_candidates(
        self, task_type: str, n_samples: int, n_features: int,
        n_classes: int, quality_score: float, budget_s: float,
    ) -> List[Tuple[StrategyClass, float, str]]:
        """Rule-based candidate selection based on data characteristics."""
        candidates = []

        # Tabular classification/regression heuristics
        if task_type in ("binary", "multiclass", "regression"):
            # GBM is almost always strong on tabular
            candidates.append((
                StrategyClass.GRADIENT_BOOSTING, 0.85,
                "GBM consistently strong on tabular data",
            ))

            # Tree ensemble for smaller datasets
            if n_samples < 5000:
                candidates.append((
                    StrategyClass.TREE_ENSEMBLE, 0.75,
                    f"Random Forest robust on small datasets (n={n_samples})",
                ))

            # Linear for high-dimensional
            if n_features > n_samples * 0.5:
                candidates.append((
                    StrategyClass.LINEAR, 0.70,
                    f"Linear regularized models handle high p/n ratio (p={n_features}, n={n_samples})",
                ))

            # Feature engineering for moderate dimensionality
            if 10 < n_features < 200 and n_samples > 100:
                candidates.append((
                    StrategyClass.FEATURE_HEAVY, 0.65,
                    f"Feature engineering profitable on moderate dimensionality ({n_features} features)",
                ))

            # Neural for large datasets with budget
            if n_samples > 5000 and budget_s > 120:
                candidates.append((
                    StrategyClass.NEURAL_TABULAR, 0.55,
                    f"Neural models competitive on larger datasets (n={n_samples}) with sufficient budget",
                ))

            # Ensemble stacking when quality is high
            if quality_score > 0.8 and budget_s > 60:
                candidates.append((
                    StrategyClass.ENSEMBLE_STACKING, 0.60,
                    "Stacking top models for final push on clean data",
                ))

            # Data-centric when quality is low
            if quality_score < 0.6:
                candidates.append((
                    StrategyClass.DATA_CENTRIC, 0.70,
                    f"Data quality low ({quality_score:.2f}); focus on data improvement first",
                ))

        elif task_type == "text":
            if n_samples > 1000 and budget_s > 300:
                candidates.append((
                    StrategyClass.NEURAL_NLP, 0.85,
                    "Transformer-based models best for NLP with sufficient data",
                ))
            else:
                candidates.append((
                    StrategyClass.LINEAR, 0.70,
                    "TF-IDF + linear for small text datasets",
                ))

        elif task_type == "vision":
            candidates.append((
                StrategyClass.NEURAL_VISION, 0.90,
                "CNNs/ViT required for vision tasks",
            ))

        elif task_type == "timeseries":
            candidates.append((
                StrategyClass.NEURAL_TIMESERIES, 0.75,
                "Neural temporal models (LSTM, N-BEATS) for time series",
            ))
            candidates.append((
                StrategyClass.GRADIENT_BOOSTING, 0.65,
                "GBM on lag features as strong baseline for time series",
            ))

        # Always add exploratory as last resort
        candidates.append((
            StrategyClass.EXPLORATORY, 0.30,
            "LLM-generated novel approaches for when standard methods plateau",
        ))

        return candidates

    def _name_to_class(self, name: str) -> Optional[StrategyClass]:
        """Map a strategy name to a StrategyClass."""
        name_lower = name.lower()
        mapping = {
            "gradient_boosting": StrategyClass.GRADIENT_BOOSTING,
            "gbm": StrategyClass.GRADIENT_BOOSTING,
            "xgboost": StrategyClass.GRADIENT_BOOSTING,
            "lightgbm": StrategyClass.GRADIENT_BOOSTING,
            "random_forest": StrategyClass.TREE_ENSEMBLE,
            "extra_trees": StrategyClass.TREE_ENSEMBLE,
            "linear": StrategyClass.LINEAR,
            "ridge": StrategyClass.LINEAR,
            "lasso": StrategyClass.LINEAR,
            "logistic": StrategyClass.LINEAR,
            "svm": StrategyClass.SVM,
            "neural": StrategyClass.NEURAL_TABULAR,
            "tabnet": StrategyClass.NEURAL_TABULAR,
            "ensemble": StrategyClass.ENSEMBLE_STACKING,
            "stacking": StrategyClass.ENSEMBLE_STACKING,
            "feature": StrategyClass.FEATURE_HEAVY,
            "pca": StrategyClass.FEATURE_HEAVY,
        }
        for key, cls in mapping.items():
            if key in name_lower:
                return cls
        return None

    def _class_to_families(self, sc: StrategyClass) -> List[str]:
        """Map a StrategyClass to specific model family names."""
        families = {
            StrategyClass.GRADIENT_BOOSTING: [
                "GradientBoostingClassifier", "GradientBoostingRegressor",
                "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
            ],
            StrategyClass.TREE_ENSEMBLE: [
                "RandomForestClassifier", "RandomForestRegressor",
                "ExtraTreesClassifier", "ExtraTreesRegressor",
            ],
            StrategyClass.LINEAR: [
                "LogisticRegression", "Ridge", "Lasso", "ElasticNet",
                "SGDClassifier", "SGDRegressor",
            ],
            StrategyClass.SVM: [
                "SVC", "SVR", "LinearSVC", "LinearSVR",
            ],
            StrategyClass.NEURAL_TABULAR: [
                "MLPClassifier", "MLPRegressor",
            ],
            StrategyClass.FEATURE_HEAVY: [
                "Pipeline_PCA_GBM", "Pipeline_Poly_Linear",
            ],
            StrategyClass.ENSEMBLE_STACKING: [
                "StackingClassifier", "StackingRegressor", "VotingClassifier",
            ],
        }
        return families.get(sc, [])
