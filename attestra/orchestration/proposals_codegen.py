"""Generate variant proposals as executable code strings for parallel evaluation.

Translates catalog-style proposals (model families, hyperparams) into
self-contained Python code strings that the ProposalPool can execute
in separate processes.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


_TEMPLATES: Dict[str, str] = {
    "gradient_boosting_variant": """
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

task = "{task}"
n_estimators = {n_estimators}
max_depth = {max_depth}
learning_rate = {learning_rate}
subsample = {subsample}

if task == "regression":
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample, random_state=42,
    )
else:
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample, random_state=42,
    )
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""",
    "random_forest_variant": """
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

task = "{task}"
n_estimators = {n_estimators}
max_depth = {max_depth}
min_samples_leaf = {min_samples_leaf}

if task == "regression":
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1,
    )
else:
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1,
    )
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""",
    "svm_variant": """
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

task = "{task}"
C = {C}
kernel = "{kernel}"

if task == "regression":
    model = Pipeline([("scaler", StandardScaler()), ("svm", SVR(C=C, kernel=kernel))])
else:
    model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(C=C, kernel=kernel, probability=False))])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""",
    "ridge_variant": """
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np

task = "{task}"
alpha = {alpha}
poly_degree = {poly_degree}

steps = [("scaler", StandardScaler())]
if poly_degree > 1:
    steps.append(("poly", PolynomialFeatures(degree=poly_degree, interaction_only=True)))
if task == "regression":
    steps.append(("model", Ridge(alpha=alpha)))
else:
    steps.append(("model", RidgeClassifier(alpha=alpha)))
model = Pipeline(steps)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""",
}

# Hyperparameter grids for variant generation
_VARIANT_GRIDS: Dict[str, List[Dict]] = {
    "gradient_boosting_variant": [
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.01, "subsample": 0.9},
        {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.7},
    ],
    "random_forest_variant": [
        {"n_estimators": 500, "max_depth": 15, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5},
    ],
    "svm_variant": [
        {"C": 1.0, "kernel": "rbf"},
        {"C": 10.0, "kernel": "rbf"},
        {"C": 0.1, "kernel": "linear"},
    ],
    "ridge_variant": [
        {"alpha": 1.0, "poly_degree": 1},
        {"alpha": 0.1, "poly_degree": 2},
        {"alpha": 10.0, "poly_degree": 1},
    ],
}


def generate_variant_proposals(
    engine: Any,
    cycle_result: Any,
    history: List[Dict],
    prior_successes: Dict[str, float],
    avoid: List[str],
    max_proposals: int = 6,
) -> List[Dict]:
    """Generate code-string proposals suitable for ProposalPool.execute_batch().

    Looks at what worked in the engine's run, creates hyperparameter variants
    of the best-performing families, and returns them as executable code dicts.
    """
    task = "regression" if getattr(engine, "profile", None) and engine.profile.task_type == "regression" else "classification"

    # Determine which families performed well
    scored_families: Dict[str, float] = {}
    for h in history:
        tech = h.get("technique", "")
        score = h.get("score")
        if tech and score is not None and h.get("status") == "success":
            family = tech.split("_")[0].lower() if "_" in tech else tech.lower()
            scored_families[family] = max(scored_families.get(family, 0), score)

    # Add prior successes
    for tech, score in prior_successes.items():
        family = tech.split("_")[0].lower() if "_" in tech else tech.lower()
        scored_families[family] = max(scored_families.get(family, 0), score)

    # Map families to templates
    family_to_template: Dict[str, str] = {
        "gradient": "gradient_boosting_variant",
        "gradientboosting": "gradient_boosting_variant",
        "random": "random_forest_variant",
        "randomforest": "random_forest_variant",
        "svm": "svm_variant",
        "svc": "svm_variant",
        "ridge": "ridge_variant",
        "linear": "ridge_variant",
    }

    # Build proposals for top-performing families
    proposals: List[Dict] = []
    ranked = sorted(scored_families.items(), key=lambda x: -x[1])

    for family, _ in ranked:
        if len(proposals) >= max_proposals:
            break
        if family in avoid:
            continue

        template_key = family_to_template.get(family)
        if template_key is None:
            # Default to gradient boosting variants
            template_key = "gradient_boosting_variant"

        template = _TEMPLATES.get(template_key, "")
        grid = _VARIANT_GRIDS.get(template_key, [])

        for params in grid:
            if len(proposals) >= max_proposals:
                break
            params_with_task = {**params, "task": task}
            code = template.format(**params_with_task)
            pid = f"pool_{family}_{hashlib.sha256(code.encode()).hexdigest()[:8]}"
            proposals.append({
                "id": pid,
                "code": code,
                "label": f"{family}_variant",
            })

    # If no family-specific proposals, add default variants
    if not proposals:
        for template_key in ["gradient_boosting_variant", "random_forest_variant"]:
            grid = _VARIANT_GRIDS[template_key]
            template = _TEMPLATES[template_key]
            for params in grid[:2]:
                if len(proposals) >= max_proposals:
                    break
                params_with_task = {**params, "task": task}
                code = template.format(**params_with_task)
                pid = f"pool_default_{hashlib.sha256(code.encode()).hexdigest()[:8]}"
                proposals.append({
                    "id": pid,
                    "code": code,
                    "label": f"{template_key}_default",
                })

    return proposals
