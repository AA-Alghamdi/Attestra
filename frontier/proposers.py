"""Proposal sources: where candidate Programs come from.

The design decision the owner asked for: the proposal unit is generated CODE, and the
former 12-family catalog is demoted to SEEDS and BASELINES. Feature engineering,
preprocessing, and target transforms are part of the proposal space BY CONSTRUCTION,
because a Program is an arbitrary pipeline, not a (family, params) selection. There is
no n_features >= 60 gate (the live harness.py:391-392 trap that capped California Housing).

Three sources, in increasing power:
  - SeedProposer    : a small library of baseline recipes, some WITH feature engineering
                      and target transforms. Guarantees a floor and seeds the search.
  - MutationProposer: takes the best recipe so far and regenerates variants (add poly,
                      add scaling, log-transform the target, swap the base estimator).
                      This makes the spine GENERATIVE even with NO LLM available.
  - LLMProposer     : pluggable. Given a `client(prompt)->code`, asks a frontier model
                      for full build_estimator() code conditioned on the round's diagnosis.
                      Skipped (returns []) when no client is configured -- and we say so,
                      rather than silently faking it.

"Powered by LLMs, not competing": the seeds/mutations are a FALLBACK and a floor, never
the promotion-bearing decision. When a model is wired in, the LLMProposer outproduces them
and the system improves for free as base models improve.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Protocol

from .program import Program


# --------------------------------------------------------------------------- recipe -> code
# A recipe is a small dict describing a pipeline. make_code() renders it to a Program body.
# The catalog "families" are just base estimators here; the enhancers (scale/poly/target_log)
# are what the old catalog could not propose.

_BASES = {
    ("classification", "hist_gbm"): ("from sklearn.ensemble import HistGradientBoostingClassifier",
                                      "HistGradientBoostingClassifier(random_state=0)"),
    ("classification", "rf"): ("from sklearn.ensemble import RandomForestClassifier",
                               "RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1)"),
    ("classification", "logreg"): ("from sklearn.linear_model import LogisticRegression",
                                   "LogisticRegression(max_iter=2000)"),
    ("classification", "svc_rbf"): ("from sklearn.svm import SVC",
                                    "SVC(C=2.0, gamma='scale')"),
    ("regression", "hist_gbm"): ("from sklearn.ensemble import HistGradientBoostingRegressor",
                                 "HistGradientBoostingRegressor(random_state=0)"),
    ("regression", "ridge"): ("from sklearn.linear_model import Ridge", "Ridge(alpha=1.0)"),
    ("regression", "rf"): ("from sklearn.ensemble import RandomForestRegressor",
                           "RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=1)"),
    ("regression", "gbr"): ("from sklearn.ensemble import GradientBoostingRegressor",
                            "GradientBoostingRegressor(random_state=0)"),
}


def recipe_label(recipe: dict) -> str:
    tags = []
    if recipe.get("target_log"):
        tags.append("tlog")
    if recipe.get("poly"):
        tags.append(f"poly{int(recipe['poly'])}")
    if recipe.get("scale"):
        tags.append("scale")
    tags.append(recipe["base"])
    return "+".join(tags)


def make_code(recipe: dict, kind: str) -> str:
    """Render a recipe dict to a complete module defining build_estimator()."""
    base = recipe["base"]
    if (kind, base) not in _BASES:
        raise KeyError(f"no base {base!r} for {kind}")
    imp, ctor = _BASES[(kind, base)]
    head = [
        "import numpy as np",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import StandardScaler, PolynomialFeatures",
        imp,
    ]
    body = ["", "def build_estimator():", "    steps = []"]
    if recipe.get("scale"):
        body.append("    steps.append(('scaler', StandardScaler()))")
    if recipe.get("poly"):
        body.append(f"    steps.append(('poly', PolynomialFeatures(degree={int(recipe['poly'])}, "
                    f"include_bias=False)))")
    body.append(f"    steps.append(('model', {ctor}))")
    body.append("    pipe = Pipeline(steps)")
    if kind == "regression" and recipe.get("target_log"):
        head.append("from sklearn.compose import TransformedTargetRegressor")
        body.append("    return TransformedTargetRegressor(regressor=pipe, "
                    "func=np.log1p, inverse_func=np.expm1)")
    else:
        body.append("    return pipe")
    return "\n".join(head + body) + "\n"


def _program_from_recipe(recipe: dict, kind: str, source: str,
                         parent_id: Optional[str] = None) -> Program:
    return Program(code=make_code(recipe, kind), source=source,
                   label=recipe_label(recipe), parent_id=parent_id,
                   provenance={"recipe": dict(recipe)})


# --------------------------------------------------------------------------- sources

class ProposalSource(Protocol):
    def propose(self, context: dict) -> List[Program]:
        ...


class SeedProposer:
    """Baseline recipes. Some carry feature engineering / target transforms on purpose,
    so even the floor exercises the axis the old catalog could not reach."""

    def __init__(self):
        self._seeds = {
            "classification": [
                {"base": "hist_gbm"},
                {"base": "rf"},
                {"base": "logreg", "scale": True},
                {"base": "svc_rbf", "scale": True},
            ],
            "regression": [
                {"base": "hist_gbm"},
                {"base": "ridge", "scale": True},
                {"base": "ridge", "scale": True, "poly": 2},   # feature engineering
                {"base": "hist_gbm", "target_log": True},      # target transform
            ],
        }

    def propose(self, context: dict) -> List[Program]:
        kind = context["task_kind"]
        tried = set(context.get("tried_labels", ()))
        out = []
        for r in self._seeds.get(kind, []):
            p = _program_from_recipe(r, kind, "seed")
            if p.label not in tried:
                out.append(p)
        return out


class MutationProposer:
    """Regenerate variants of the best recipe so far. Deterministic, no LLM required.

    This is what makes the spine generative offline: it composes feature engineering,
    scaling, target transforms, and base swaps onto the current champion.
    """

    def propose(self, context: dict) -> List[Program]:
        kind = context["task_kind"]
        best = context.get("best_recipe")
        if not best:
            return []
        tried = set(context.get("tried_labels", ()))
        variants: List[dict] = []

        def add(mut):
            r = dict(best)
            r.update(mut)
            variants.append(r)

        # toggle / add enhancers
        add({"scale": not best.get("scale", False)})
        add({"poly": 2 if not best.get("poly") else 0})
        if kind == "regression":
            add({"target_log": not best.get("target_log", False)})
        # swap base to an untried base of the same kind
        for (k, name) in _BASES:
            if k == kind and name != best["base"]:
                add({"base": name})

        out, seen = [], set()
        for r in variants:
            r = {k: v for k, v in r.items() if v}        # drop falsey flags for a clean label
            r["base"] = r.get("base", best["base"])
            lab = recipe_label(r)
            if lab in tried or lab in seen:
                continue
            seen.add(lab)
            out.append(_program_from_recipe(r, kind, "mutation",
                                            parent_id=context.get("best_id")))
        return out


class LLMProposer:
    """Pluggable LLM authoring. `client` is any callable: prompt(str) -> python code(str).

    Wire vfplatform's resolve_backend / Prime Intellect client here. If no client is
    configured we return [] and the engine reports that the generative LLM path was
    inactive -- we do not fabricate proposals.
    """

    def __init__(self, client: Optional[Callable[[str], str]] = None, n: int = 2):
        self.client = client
        self.n = n

    def _prompt(self, context: dict) -> str:
        kind = context["task_kind"]
        errs = context.get("recent_errors", [])
        err_txt = "\n".join(f"  - {lab}: [{ek}] {msg}" for lab, ek, msg in errs[:5]) or "  (none yet)"
        parts = [
            "You are proposing a scikit-learn pipeline for an ML task.\n"
            f"Task kind: {kind}. Features: {context.get('n_features')}. "
            f"Train size: {context.get('n_train')}.\n"
            f"Best so far: {context.get('best_label')} (val {context.get('best_score')}).\n"
            "Recent failed attempts (avoid repeating these errors):\n"
            f"{err_txt}",
        ]
        # Inject deterministic diagnosis guidance
        diag_guidance = context.get("diag_guidance", "")
        if diag_guidance:
            parts.append(f"\nDIAGNOSIS:\n{diag_guidance}")
        # Inject LLM diagnosis guidance (deeper analysis)
        llm_guidance = context.get("llm_guidance", "")
        if llm_guidance and llm_guidance != diag_guidance:
            parts.append(f"\nLLM ANALYSIS:\n{llm_guidance}")
        # Inject confusion matrix info (classification)
        confusion = context.get("confusion")
        if confusion:
            weak = confusion.get("weakest_class")
            pairs = confusion.get("confused_pairs", [])
            if weak is not None:
                parts.append(f"\nWEAKEST CLASS: '{weak}' — focus improvement here.")
            if pairs:
                pair_txt = "; ".join(
                    f"'{p['true']}'->'{p['pred']}' ({p['count']}x)"
                    for p in pairs[:3]
                )
                parts.append(f"MOST CONFUSED: {pair_txt}")
        parts.append(
            "\n\nReturn ONLY Python code defining build_estimator() that returns an unfitted "
            "sklearn-compatible estimator. You MAY use feature engineering, preprocessing, "
            "target transforms, and stacking. Do not fit; do not print; no markdown fences."
        )
        return "\n".join(parts)

    def propose(self, context: dict) -> List[Program]:
        if self.client is None:
            return []
        prompt = self._prompt(context)
        out = []
        for i in range(self.n):
            try:
                code = self.client(prompt)
            except Exception:
                break
            if not code or "build_estimator" not in code:
                continue
            out.append(Program(code=code, source="llm", label=f"llm{context.get('round', 0)}_{i}",
                               provenance={"prompt_chars": len(prompt)}))
        return out
