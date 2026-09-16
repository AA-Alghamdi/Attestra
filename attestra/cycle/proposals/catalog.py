"""Catalog proposal source: deterministic model families with tuned configurations.

This is the reliable backbone. Every model family has been tested and has known
characteristics. The catalog provides the safety net that always produces valid
baselines, even when LLM proposals fail.

Model families are ordered by expected performance for the task type and data shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ...intake.profiler import DataProfile


@dataclass
class CatalogProposal:
    """A proposal from the deterministic catalog."""
    name: str
    family: str
    build_fn: Callable[[int], Any]    # build_fn(seed) -> sklearn estimator
    priority: float = 1.0             # higher = try first
    estimated_time_s: float = 10.0
    tags: List[str] = field(default_factory=list)


def propose_from_catalog(
    profile: DataProfile,
    *,
    tried_families: Optional[set] = None,
    max_proposals: int = 5,
) -> List[CatalogProposal]:
    """Generate catalog proposals appropriate for the data profile."""
    tried = tried_families or set()
    proposals = []

    if profile.task_type == "regression":
        proposals = _regression_catalog(profile)
    elif profile.task_type == "binary":
        proposals = _classification_catalog(profile, binary=True)
    else:
        proposals = _classification_catalog(profile, binary=False)

    # Filter already-tried families and sort by priority
    proposals = [p for p in proposals if p.name not in tried]
    proposals.sort(key=lambda p: p.priority, reverse=True)
    return proposals[:max_proposals]


def _classification_catalog(profile: DataProfile, binary: bool) -> List[CatalogProposal]:
    """Classification model catalog ordered by expected performance."""
    n = profile.n_samples
    d = profile.n_features
    proposals = []

    # HistGBM: almost always the best first try for tabular
    proposals.append(CatalogProposal(
        name="hist_gbm_tuned",
        family="hist_gradient_boosting",
        build_fn=lambda seed: _build_hist_gbm_clf(seed, n, d),
        priority=10.0,
        estimated_time_s=5 + n * d * 1e-6,
        tags=["tree", "boosting", "handles_missing"],
    ))

    # Random Forest
    proposals.append(CatalogProposal(
        name="random_forest_tuned",
        family="random_forest",
        build_fn=lambda seed: _build_rf_clf(seed, n, d),
        priority=9.0,
        estimated_time_s=5 + n * d * 1e-6,
        tags=["tree", "bagging"],
    ))

    # Extra Trees (often better than RF on noisy data)
    proposals.append(CatalogProposal(
        name="extra_trees_tuned",
        family="extra_trees",
        build_fn=lambda seed: _build_et_clf(seed, n, d),
        priority=8.0,
        estimated_time_s=3 + n * d * 1e-6,
        tags=["tree", "bagging"],
    ))

    # Logistic Regression (strong baseline, fast)
    proposals.append(CatalogProposal(
        name="logistic_regression_tuned",
        family="logistic_regression",
        build_fn=lambda seed: _build_lr_clf(seed, n, d),
        priority=7.0,
        estimated_time_s=2.0,
        tags=["linear", "fast"],
    ))

    # Stacking ensemble (HistGBM + LR + ET)
    if n >= 200:
        proposals.append(CatalogProposal(
            name="stacking_3model",
            family="stacking",
            build_fn=lambda seed: _build_stacking_clf(seed, n, d),
            priority=8.5,
            estimated_time_s=30 + n * d * 3e-6,
            tags=["ensemble", "stacking"],
        ))

    # SVM (good for small/medium datasets with few features)
    if n < 10000 and d < 100:
        proposals.append(CatalogProposal(
            name="svm_rbf_tuned",
            family="svm",
            build_fn=lambda seed: _build_svm_clf(seed, n, d),
            priority=6.0,
            estimated_time_s=5 + n ** 2 * 1e-7,
            tags=["kernel", "svm"],
        ))

    # KNN (good for small datasets)
    if n < 5000:
        proposals.append(CatalogProposal(
            name="knn_tuned",
            family="knn",
            build_fn=lambda seed: _build_knn_clf(seed, n, d),
            priority=4.0,
            estimated_time_s=1.0,
            tags=["instance", "nonparametric"],
        ))

    # Voting ensemble
    if n >= 300:
        proposals.append(CatalogProposal(
            name="voting_3model",
            family="voting",
            build_fn=lambda seed: _build_voting_clf(seed, n, d),
            priority=7.5,
            estimated_time_s=20 + n * d * 2e-6,
            tags=["ensemble", "voting"],
        ))

    # MLP
    if n >= 500:
        proposals.append(CatalogProposal(
            name="mlp_tuned",
            family="mlp",
            build_fn=lambda seed: _build_mlp_clf(seed, n, d),
            priority=5.0,
            estimated_time_s=10 + n * d * 1e-5,
            tags=["neural", "mlp"],
        ))

    return proposals


def _regression_catalog(profile: DataProfile) -> List[CatalogProposal]:
    """Regression model catalog."""
    n = profile.n_samples
    d = profile.n_features
    proposals = []

    proposals.append(CatalogProposal(
        name="hist_gbm_reg_tuned",
        family="hist_gradient_boosting",
        build_fn=lambda seed: _build_hist_gbm_reg(seed, n, d),
        priority=10.0,
        estimated_time_s=5 + n * d * 1e-6,
        tags=["tree", "boosting", "handles_missing"],
    ))

    proposals.append(CatalogProposal(
        name="random_forest_reg_tuned",
        family="random_forest",
        build_fn=lambda seed: _build_rf_reg(seed, n, d),
        priority=9.0,
        estimated_time_s=5 + n * d * 1e-6,
        tags=["tree", "bagging"],
    ))

    proposals.append(CatalogProposal(
        name="ridge_tuned",
        family="ridge",
        build_fn=lambda seed: _build_ridge_reg(seed),
        priority=7.0,
        estimated_time_s=1.0,
        tags=["linear", "fast"],
    ))

    proposals.append(CatalogProposal(
        name="elastic_net_tuned",
        family="elastic_net",
        build_fn=lambda seed: _build_enet_reg(seed),
        priority=6.5,
        estimated_time_s=1.0,
        tags=["linear", "sparse"],
    ))

    if n >= 200:
        proposals.append(CatalogProposal(
            name="stacking_reg_3model",
            family="stacking",
            build_fn=lambda seed: _build_stacking_reg(seed, n, d),
            priority=8.5,
            estimated_time_s=30 + n * d * 3e-6,
            tags=["ensemble", "stacking"],
        ))

    if n < 10000 and d < 100:
        proposals.append(CatalogProposal(
            name="svr_rbf_tuned",
            family="svr",
            build_fn=lambda seed: _build_svr_reg(seed, n, d),
            priority=5.0,
            estimated_time_s=5 + n ** 2 * 1e-7,
            tags=["kernel", "svm"],
        ))

    if n >= 500:
        proposals.append(CatalogProposal(
            name="mlp_reg_tuned",
            family="mlp",
            build_fn=lambda seed: _build_mlp_reg(seed, n, d),
            priority=5.0,
            estimated_time_s=10 + n * d * 1e-5,
            tags=["neural", "mlp"],
        ))

    return proposals


# ============================================================================== builders

def _build_hist_gbm_clf(seed: int, n: int, d: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr = 0.05 if n > 5000 else 0.1
    depth = min(8, max(3, int(np.log2(max(d, 2)))))
    return make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(
            learning_rate=lr, max_depth=depth, max_iter=300,
            min_samples_leaf=max(1, n // 100),
            random_state=seed, early_stopping=True, validation_fraction=0.15,
        )
    )


def _build_rf_clf(seed: int, n: int, d: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200, max_depth=min(20, max(5, int(np.log2(max(d, 2)) * 3))),
            min_samples_leaf=max(1, n // 200), max_features="sqrt",
            random_state=seed, n_jobs=-1,
        )
    )


def _build_et_clf(seed: int, n: int, d: int):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        ExtraTreesClassifier(
            n_estimators=200, max_depth=min(25, max(5, d)),
            min_samples_leaf=max(1, n // 200), max_features="sqrt",
            random_state=seed, n_jobs=-1,
        )
    )


def _build_lr_clf(seed: int, n: int, d: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            multi_class="multinomial", random_state=seed,
        )
    )


def _build_stacking_clf(seed: int, n: int, d: int):
    from sklearn.ensemble import (HistGradientBoostingClassifier, ExtraTreesClassifier,
                                   StackingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    estimators = [
        ("hgb", HistGradientBoostingClassifier(max_iter=100, random_state=seed)),
        ("et", ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500, random_state=seed),
        cv=3, n_jobs=-1,
    )


def _build_svm_clf(seed: int, n: int, d: int):
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed, probability=True),
    )


def _build_knn_clf(seed: int, n: int, d: int):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    k = min(15, max(3, int(np.sqrt(n) / 2)))
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1),
    )


def _build_voting_clf(seed: int, n: int, d: int):
    from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                                   VotingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    estimators = [
        ("hgb", HistGradientBoostingClassifier(max_iter=150, random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)),
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))),
    ]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def _build_mlp_clf(seed: int, n: int, d: int):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    h1 = min(256, max(32, d * 4))
    h2 = min(128, max(16, d * 2))
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(h1, h2), max_iter=500,
            learning_rate="adaptive", early_stopping=True,
            random_state=seed,
        )
    )


# Regression builders

def _build_hist_gbm_reg(seed: int, n: int, d: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr = 0.05 if n > 5000 else 0.1
    return make_pipeline(
        StandardScaler(),
        HistGradientBoostingRegressor(
            learning_rate=lr, max_depth=min(8, max(3, int(np.log2(max(d, 2))))),
            max_iter=300, min_samples_leaf=max(1, n // 100),
            random_state=seed, early_stopping=True, validation_fraction=0.15,
        )
    )


def _build_rf_reg(seed: int, n: int, d: int):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=200, max_depth=min(20, max(5, int(np.log2(max(d, 2)) * 3))),
            min_samples_leaf=max(1, n // 200), random_state=seed, n_jobs=-1,
        )
    )


def _build_ridge_reg(seed: int):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def _build_enet_reg(seed: int):
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000))


def _build_stacking_reg(seed: int, n: int, d: int):
    from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor,
                                   StackingRegressor)
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    estimators = [
        ("hgb", HistGradientBoostingRegressor(max_iter=100, random_state=seed)),
        ("et", ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1)),
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ]
    return StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=3, n_jobs=-1)


def _build_svr_reg(seed: int, n: int, d: int):
    from sklearn.svm import SVR
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, gamma="scale"))


def _build_mlp_reg(seed: int, n: int, d: int):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    h1 = min(256, max(32, d * 4))
    h2 = min(128, max(16, d * 2))
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(h1, h2), max_iter=500,
            learning_rate="adaptive", early_stopping=True,
            random_state=seed,
        )
    )
