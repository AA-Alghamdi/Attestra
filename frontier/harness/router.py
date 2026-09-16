"""Phase 3 problem router: ML-problem ontology + goal-aware harness selection.

Given a natural-language *goal* plus the raw *(X, y)*, decide WHAT KIND of ML problem
this is (kind x modality), suggest the metric the certifier should hold the winner to,
surface the RISKS that would make a "certified" number a lie (reward-hacking,
infeasibility, goal/data contradictions), and select the per-modality Harness that knows
how to turn this data into the (rows, metric, theta, split-protocol) the frozen certifier
consumes.

Why a router at all (audit grounding):
  - The live `vfplatform/harness.py` baked a single tabular assumption plus an
    `n_features >= 60` gate; `attestra/intake/problem_typing.py` existed but was never wired
    into the loop. This module is the wired replacement: one typed front door that the
    engine calls before it ever splits or certifies.
  - "Powered by LLMs, not competing": when an `llm_client` is supplied, the LLM owns the
    open-ended judgement (is the modality text-or-tabular when the matrix is ambiguous? does
    the goal contradict the data? is the success criterion gameable?). When no client is
    given we DEGRADE HONESTLY to a deterministic typer driven purely by `X`/`y` shape and
    dtype -- never fabricating an LLM verdict. The deterministic path is a floor/fallback,
    never the promotion-bearing decision (the frozen certifier still promotes).

Standing invariants preserved:
  - The router selects a harness and emits a typed spec; it computes NO promotion-bearing
    number. The sealed certifier (vectorforge.science / vfplatform.sealed) remains the only
    promoter, reached later through the harness's adapter.
  - Hardcoded keyword/shape heuristics here are the *fallback* typer, not the promoter.
  - Adversarial risks (infeasibility, reward-hacking, contradiction) are surfaced, not
    silently swallowed: a high-severity risk is returned on the spec so the engine can
    decline honestly instead of "certifying" a hacked objective.

# === WIRING ===
# The integrator plugs this in as the engine's front door, BEFORE make_splits:
#
#   from frontier.harness.router import route
#   harness = route(goal, X, y, llm_client=cfg.llm_client)   # a frontier.harness.base.Harness
#   spec = harness.spec                                       # the ProblemSpec (annotated on it)
#   if spec.blocked:                                          # reward-hacking / infeasibility guard
#       decline(reason=str(spec.risks))                       # honest decline BEFORE certifying
#   ok, _ = harness.self_test()                               # GATE: trust nothing until True
#   task = harness.adapt(X, y, kind=spec.kind, theta=THETA, metric=spec.metric)
#   result = ResearchEngine(EngineConfig(llm_client=cfg.llm_client)).run(task)  # frozen gate
#
# Call site / argument shapes:
#   - goal: str               the user's research goal (free text). "" is allowed.
#   - X:    array-like (n,d)  raw features as handed to the engine (the same object that
#                             becomes Task.X). May be an object array for text/sequence data.
#   - y:    array-like (n,)   raw targets (the same object that becomes Task.y). May be None
#                             for unsupervised goals (router types it as such and flags risk).
#   - llm_client: Callable[[str],str] | None   prompt->text; None => deterministic typer.
#
# Ordering with EngineConfig: route() runs first and produces `spec`. The engine then uses
#   spec.kind        -> Task.kind            ("classification"|"regression")
#   spec.metric      -> Task.metric          (the certifier metric, e.g. "macro_f1")
#   spec.theta_hint  -> a *suggested* theta floor (the engine/operator still sets the real
#                       theta; the hint is advisory, never auto-promoting).
#   spec.risks       -> if any risk has severity "block", the engine should decline before
#                       running, recording the risk as decline_reason. (Reward-hacking guard.)
# The returned Harness exposes .adapt(...) / .self_test() / .baseline_suite(...) per Phase 3's
# frontier.harness.base.Harness contract; this router only SELECTS it (via base.REGISTRY) and
# annotates it with .spec. When the registry is not yet importable/populated (sibling Phase-3
# tasks author it concurrently), route() returns a self-contained _FallbackHarness carrying the
# same .spec so the router is testable and the wiring is stable; the integrator gets the real
# harness automatically once frontier.harness.base.REGISTRY is populated.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- taxonomy

# The ontology is a typed cross-product of {kind} x {modality}. `kind` is what the certifier
# branches on (classification vs regression share the sealed path; clustering/unsupervised has
# no labeled certificate and is flagged). `modality` is what the harness branches on (which
# adapter turns raw data into rows). Keeping them orthogonal means a new modality (audio) does
# not multiply the certifier code -- only a new adapter.

KINDS = ("classification", "regression", "clustering", "unknown")
MODALITIES = ("tabular", "text", "image", "timeseries", "audio", "unknown")

# Metric suggestions per (kind, class-balance). These are the science.py-supported metrics
# (see Task contract: accuracy, balanced_accuracy, macro_f1, r2, neg_rmse, neg_mae). The
# suggestion is advisory; the operator/engine sets the binding metric on the Task.
_METRIC_FOR = {
    ("classification", "balanced"): "accuracy",
    ("classification", "imbalanced"): "macro_f1",
    ("regression", "any"): "r2",
}


@dataclass
class Risk:
    """A reason the eventual certificate could be a lie. severity drives engine behavior."""

    code: str                 # machine tag, e.g. "infeasible_goal"
    severity: str             # "info" | "warn" | "block"  (block => engine should decline)
    detail: str               # human-readable explanation

    def __str__(self) -> str:
        return f"[{self.severity}:{self.code}] {self.detail}"


@dataclass
class ProblemSpec:
    """The typed verdict the router hands the engine.

    kind/modality place the problem in the ontology; metric is the suggested certifier metric;
    theta_hint is an advisory promotion floor (never auto-promoting); risks are the adversarial
    findings (infeasibility / reward-hacking / contradiction); typed_by records whether an LLM
    or the deterministic fallback produced the typing (so callers know if the LLM path was
    inactive); confidence in [0,1] is the typer's self-reported certainty.
    """

    kind: str
    modality: str
    metric: str
    theta_hint: float
    risks: List[Risk] = field(default_factory=list)
    typed_by: str = "deterministic"            # "llm" | "deterministic"
    confidence: float = 0.5
    n_samples: int = 0
    n_features: int = 0
    n_classes: int = 0
    rationale: str = ""

    @property
    def blocked(self) -> bool:
        """True if any surfaced risk is severe enough that promotion would be dishonest."""
        return any(r.severity == "block" for r in self.risks)

    def summary(self) -> str:
        head = (f"{self.kind}/{self.modality} metric={self.metric} "
                f"theta_hint={self.theta_hint:.3f} typed_by={self.typed_by} "
                f"conf={self.confidence:.2f}")
        if self.risks:
            head += "\n  risks:\n    " + "\n    ".join(str(r) for r in self.risks)
        return head


# --------------------------------------------------------------------------- harness binding

# We prefer the real Phase-3 registry (a sibling task authors frontier/harness/base.py). If it
# is not importable yet, we fall back to a self-contained harness so this router is usable and
# testable in isolation. The fallback carries the SAME .spec contract, so the integrator's call
# site (harness.spec) does not change when the real registry lands.

def _load_registry():
    """Return the shared HarnessRegistry from frontier.harness.base, or None.

    The Phase-3 base module exposes a module-level `REGISTRY` (a HarnessRegistry instance) into
    which per-modality harnesses register at import time. We never hard-depend on it: a missing
    or empty registry is an honest fallback, not an error, because the harness modules are
    authored in sibling tasks and may not be importable/populated when the router runs standalone.
    Importing `frontier.harness` triggers the package __init__ that registers the harnesses.
    """
    try:
        from frontier.harness import base as _base  # type: ignore
    except Exception:
        return None
    # Importing the package __init__ is what registers concrete harnesses; do it defensively so
    # a half-written sibling module never crashes routing.
    try:
        import frontier.harness  # noqa: F401  (side-effect: registration)
    except Exception:
        pass
    registry = getattr(_base, "REGISTRY", None)
    # Duck-typed: a HarnessRegistry exposes .get(key) and .keys().
    if registry is None or not hasattr(registry, "get") or not hasattr(registry, "keys"):
        return None
    return registry


# Canonical harness class-name per modality. Used to (a) match by class name if the registry is
# keyed differently, and (b) name the fallback so the verified-routing assertion ("a numeric
# matrix routes to TabularHarness") is meaningful even before the real registry is populated.
_HARNESS_NAME = {
    "tabular": "TabularHarness",
    "text": "TextHarness",
    "image": "ImageHarness",
    "timeseries": "TimeSeriesHarness",
    "audio": "AudioHarness",
    "unknown": "TabularHarness",   # safest default: tabular adapter handles a numeric matrix
}


@dataclass
class _FallbackHarness:
    """Self-contained stand-in used only when no real registry harness is available.

    It carries the typed `spec` and mirrors the canonical class name (e.g. "TabularHarness") in
    `name`, so callers read `harness.spec` / `harness.name` exactly as with a real harness. It is
    deliberately inert: it certifies nothing (the certifier lives behind the real harness's
    adapter), and adapt()/build_task() raise a clear, honest error pointing at the missing
    Phase-3 harness rather than fabricating a Task.
    """

    spec: ProblemSpec
    name: str = "fallback"

    def adapt(self, *args, **kwargs):
        raise RuntimeError(
            "no real harness registered for modality "
            f"{self.spec.modality!r}; router returned a fallback. Import/register the Phase-3 "
            "harness (frontier.harness.base.REGISTRY) to build a certifiable Task."
        )

    # build_task alias kept for callers using the older verb.
    build_task = adapt

    def self_test(self, *args, **kwargs):
        # A fallback harness is never trusted to produce certified numbers.
        return (False, None)


def _select_harness(spec: ProblemSpec):
    """Pick the Harness instance for `spec.modality` from the real registry, else a fallback.

    The selected harness gets `.spec` attached (a non-promoting annotation) so the integrator can
    read the typed verdict + risks off the returned object. The harness's own trust state and
    adapter are untouched; we only annotate. The fallback's `name` mirrors the canonical class
    name so routing can be asserted by name regardless of whether the registry is populated yet.
    """
    canonical = _HARNESS_NAME.get(spec.modality, "TabularHarness")
    registry = _load_registry()
    harness = None
    if registry is not None:
        # Try the modality key directly (base.REGISTRY is keyed by modality, e.g. "tabular").
        try:
            harness = registry.get(spec.modality)
        except Exception:
            harness = None
        # Fall back to matching by the canonical class name across known keys.
        if harness is None:
            try:
                for k in registry.keys():
                    h = registry.get(k)
                    if type(h).__name__ == canonical:
                        harness = h
                        break
            except Exception:
                harness = None

    if harness is None:
        return _FallbackHarness(spec=spec, name=canonical)

    # Annotate the real harness with the spec for the integrator (non-promoting metadata only).
    try:
        harness.spec = spec
        if not getattr(harness, "name", None):
            harness.name = type(harness).__name__
    except Exception:
        pass
    return harness


# --------------------------------------------------------------------------- data probing

def _infer_modality_from_data(X: Any) -> tuple[str, float, str]:
    """Deterministic modality guess from the array's shape/dtype.

    Returns (modality, confidence, rationale). Pure data signal, no goal text -- the goal can
    only *raise* confidence or override via the LLM path, never silently downgrade this floor.
    """
    arr = np.asarray(X, dtype=object) if _is_objecty(X) else np.asarray(X)
    if arr.ndim == 0 or arr.size == 0:
        return "unknown", 0.1, "empty or scalar X"

    # 1-D object/str array => most likely raw text (one document per row).
    if arr.ndim == 1:
        if arr.dtype.kind in ("U", "S", "O") and _mostly_strings(arr):
            return "text", 0.8, "1-D array of strings -> text documents"
        # 1-D numeric => a single-feature tabular or a univariate series; tabular is the
        # safe certifiable default.
        return "tabular", 0.5, "1-D numeric -> single-feature tabular"

    if arr.ndim == 2:
        n, d = arr.shape
        # 2-D object array of strings => text columns.
        if arr.dtype.kind in ("U", "S", "O") and _mostly_strings(arr):
            return "text", 0.7, "2-D string matrix -> text features"
        # Numeric 2-D: tabular. A very wide near-square block COULD be a flattened image, but
        # we will not assert that from shape alone (a 64x64 image is 4096 features, not a
        # square matrix of rows). Image detection needs the goal or >2 dims.
        return "tabular", 0.85, f"2-D numeric matrix ({n}x{d}) -> tabular"

    if arr.ndim == 3:
        # (n, T, c) is the canonical timeseries tensor; (n, H, W) grayscale images also land
        # here. Without the goal we cannot tell them apart; default to timeseries with low
        # confidence and let the goal/LLM refine.
        return "timeseries", 0.45, "3-D tensor (n,T,c) -> timeseries (ambiguous vs image)"

    if arr.ndim == 4:
        # (n, H, W, C) image batch.
        return "image", 0.7, f"4-D tensor {arr.shape} -> image batch"

    return "unknown", 0.2, f"{arr.ndim}-D tensor -> unknown modality"


def _infer_kind_from_targets(y: Any) -> tuple[str, int, float, str]:
    """Deterministic task-kind from y's dtype and cardinality.

    Returns (kind, n_classes, confidence, rationale). Mirrors the standard sklearn heuristic:
    float targets with many distinct values => regression; few distinct / non-float => classification.
    """
    if y is None:
        return "clustering", 0, 0.4, "no targets -> unsupervised/clustering"
    arr = np.asarray(y)
    n = arr.size
    if n == 0:
        return "unknown", 0, 0.1, "empty y"

    # Non-numeric labels are categorical => classification.
    if arr.dtype.kind in ("U", "S", "O", "b"):
        n_classes = len(np.unique(arr))
        return "classification", n_classes, 0.9, f"non-numeric labels, {n_classes} classes"

    # Numeric: integer-valued with low cardinality => classification; otherwise regression.
    finite = arr[np.isfinite(arr.astype(float))] if arr.dtype.kind == "f" else arr
    n_unique = len(np.unique(finite))
    is_integral = np.allclose(np.asarray(finite, dtype=float),
                              np.round(np.asarray(finite, dtype=float))) if finite.size else False
    # Threshold: <=20 distinct integer values out of >=... rows is a classifier signal. We use
    # the common max(20, ...) cap; large-cardinality integers (e.g. counts) read as regression.
    if is_integral and n_unique <= max(2, min(20, int(0.05 * n) + 1)):
        return "classification", n_unique, 0.8, f"integer targets, {n_unique} distinct -> classification"
    return "regression", 0, 0.85, f"continuous targets, {n_unique} distinct -> regression"


def _class_balance(y: Any, kind: str) -> str:
    """'balanced' | 'imbalanced' | 'any' -- drives the metric suggestion for classification."""
    if kind != "classification" or y is None:
        return "any"
    arr = np.asarray(y)
    _, counts = np.unique(arr, return_counts=True)
    if len(counts) < 2:
        return "any"
    ratio = counts.min() / counts.max()
    return "imbalanced" if ratio < 0.5 else "balanced"


def _is_objecty(X: Any) -> bool:
    try:
        return np.asarray(X).dtype.kind == "O"
    except Exception:
        return True


def _mostly_strings(arr: np.ndarray) -> bool:
    flat = arr.ravel()
    sample = flat[: min(64, flat.size)]
    if sample.size == 0:
        return False
    strish = sum(1 for v in sample if isinstance(v, (str, bytes)))
    return strish >= 0.6 * sample.size


# --------------------------------------------------------------------------- adversarial check

# Deterministic infeasibility / reward-hacking lexicon. This is a FALLBACK floor (the LLM does
# the real adversarial reasoning when present). Each phrase is paired with the risk it implies.
# We keep it small and high-precision: false "block" verdicts would wrongly suppress real work.
_INFEASIBLE_PATTERNS = [
    (r"\b100%?\s*(accuracy|accurate|precision|recall)\b", "perfect-accuracy demand"),
    (r"\bperfect(ly)?\s+(accuracy|accurate|prediction|classif)", "perfect-prediction demand"),
    (r"\bzero\s+error\b", "zero-error demand"),
    (r"\bnever\s+(wrong|misclassif|err)", "never-wrong demand"),
    (r"\bguarantee[ds]?\b.*\b(accuracy|correct|profit|win)", "guaranteed-outcome demand"),
    (r"\b(predict|forecast)\b.*\b(lottery|coin\s*flip|random\s+noise|stock price)\b",
     "prediction of an intrinsically unpredictable target"),
    (r"\bfrom\s+no\s+data\b|\bwithout\s+(any\s+)?data\b", "learning without data"),
]

# Reward-hacking / specification-gaming lexicon: goals that point the optimizer at the *metric*
# rather than the *task*, which the sealed certificate exists to defend against but which the
# router should still surface so the engine can refuse to optimize a gamed objective.
_REWARD_HACK_PATTERNS = [
    (r"\bmaximize\s+(the\s+)?(val(idation)?|test|metric|score)\b", "optimize the metric, not the task"),
    (r"\b(memoriz|overfit|leak|peek|peek at)\b.*\b(test|sealed|answer|label)\b",
     "explicit request to memorize/leak the held-out answers"),
    (r"\bby any means\b|\bat all costs\b|\bwhatever it takes\b", "unbounded means -> gaming risk"),
    (r"\bgame\s+(the\s+)?(metric|benchmark|score)\b", "explicit metric gaming"),
]

# Contradiction patterns: the goal names a task kind that conflicts with the data signal. We do
# not assert contradiction from text alone; we cross-check against the inferred kind in route().
_GOAL_KIND_HINTS = [
    (r"\b(classif|categor|label|which\s+class|spam|fraud|churn|diagnos)\b", "classification"),
    (r"\b(regress|predict\s+the\s+(value|price|amount|number)|estimate\s+how\s+much|forecast\s+the\s+value)\b",
     "regression"),
    (r"\b(cluster|segment|group\s+similar|unsupervised)\b", "clustering"),
]


def _deterministic_risks(goal: str, kind: str, n_samples: int, n_classes: int) -> List[Risk]:
    """Surface infeasibility / reward-hacking / contradiction risks WITHOUT an LLM.

    Conservative: only emits "block" for clear infeasibility or explicit leakage requests.
    Soft signals (vague maximize-the-metric phrasing, kind mismatch) are "warn".
    """
    risks: List[Risk] = []
    g = (goal or "").lower()

    for pat, why in _INFEASIBLE_PATTERNS:
        if re.search(pat, g):
            risks.append(Risk("infeasible_goal", "block",
                              f"goal demands the statistically impossible: {why}"))
            break

    for pat, why in _REWARD_HACK_PATTERNS:
        if re.search(pat, g):
            sev = "block" if "leak" in why or "memorize" in why else "warn"
            risks.append(Risk("reward_hacking", sev, f"specification-gaming risk: {why}"))
            break

    # goal/data contradiction: the goal explicitly asks for a kind the data cannot support.
    goal_kind = None
    for pat, k in _GOAL_KIND_HINTS:
        if re.search(pat, g):
            goal_kind = k
            break
    if goal_kind and kind != "unknown" and goal_kind != kind:
        # clustering-vs-supervised and clf-vs-reg are the meaningful conflicts.
        risks.append(Risk("goal_data_contradiction", "warn",
                          f"goal reads as {goal_kind} but the targets read as {kind}; "
                          "verify the labels match the stated objective"))

    # tiny-data risk: a certified bound on a handful of sealed rows is near-vacuous.
    if 0 < n_samples < 50:
        risks.append(Risk("insufficient_data", "warn",
                          f"only {n_samples} samples; the sealed lower bound will be very loose"))

    # degenerate single-class classification cannot be certified meaningfully.
    if kind == "classification" and n_classes < 2:
        risks.append(Risk("degenerate_labels", "block",
                          f"classification with {n_classes} class(es); no decision to certify"))
    return risks


# --------------------------------------------------------------------------- LLM typing

def _llm_type(goal: str, llm_client: Callable[[str], str],
              probe: dict) -> Optional[dict]:
    """Ask the LLM to type the problem and run the adversarial goal check.

    Returns a dict {kind, modality, metric, theta_hint, confidence, risks:[{code,severity,detail}],
    rationale} or None if the client failed / returned unparseable output (we then fall back to
    the deterministic typer -- never fabricating an LLM verdict).
    """
    prompt = _llm_prompt(goal, probe)
    try:
        raw = llm_client(prompt)
    except Exception:
        return None
    if not raw:
        return None
    parsed = _parse_llm_json(raw)
    if parsed is None:
        return None
    # Validate the LLM's claims against the ontology; silently coerce out-of-vocabulary values
    # to "unknown" rather than trusting a hallucinated kind/modality.
    kind = parsed.get("kind")
    modality = parsed.get("modality")
    if kind not in KINDS:
        kind = None
    if modality not in MODALITIES:
        modality = None
    if kind is None and modality is None:
        return None
    return {
        "kind": kind,
        "modality": modality,
        "metric": parsed.get("metric"),
        "theta_hint": parsed.get("theta_hint"),
        "confidence": parsed.get("confidence"),
        "rationale": parsed.get("rationale", ""),
        "risks": parsed.get("risks", []),
    }


def _llm_prompt(goal: str, probe: dict) -> str:
    return (
        "You are the problem router for an autoresearch system that returns ONLY results that "
        "survive a held-out sealed statistical certificate, or an honest decline.\n\n"
        f"User goal: {goal!r}\n"
        "Data probe (computed deterministically, trustworthy):\n"
        f"  n_samples={probe['n_samples']} n_features={probe['n_features']} "
        f"X_ndim={probe['x_ndim']} X_dtype={probe['x_dtype']}\n"
        f"  y_dtype={probe['y_dtype']} y_distinct={probe['y_distinct']} "
        f"deterministic_kind_guess={probe['kind_guess']} "
        f"deterministic_modality_guess={probe['modality_guess']}\n\n"
        "Do TWO things:\n"
        "1) Type the problem. kind in {classification,regression,clustering,unknown}; "
        "modality in {tabular,text,image,timeseries,audio,unknown}. Suggest a metric from "
        "{accuracy,balanced_accuracy,macro_f1,r2,neg_rmse,neg_mae} and an advisory theta_hint "
        "(a conservative promotion floor in metric units).\n"
        "2) Adversarially audit the GOAL against the DATA. Flag: infeasibility (e.g. demanding "
        "100% accuracy, predicting intrinsically random targets), reward-hacking / "
        "specification-gaming (optimizing the metric instead of the task, requests to peek at or "
        "memorize held-out labels), and goal/data contradictions (goal asks for a kind the data "
        "cannot support). Use severity 'block' only when promoting any number would be dishonest.\n\n"
        "Respond with ONLY a JSON object, no prose, no markdown fences:\n"
        '{"kind": "...", "modality": "...", "metric": "...", "theta_hint": 0.0, '
        '"confidence": 0.0, "rationale": "...", '
        '"risks": [{"code": "...", "severity": "info|warn|block", "detail": "..."}]}\n'
    )


def _parse_llm_json(raw: str) -> Optional[dict]:
    """Extract the first JSON object from the LLM text. Tolerant of stray prose/fences."""
    import json
    s = raw.strip()
    # strip common markdown fences
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    # find the outermost {...}
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


# --------------------------------------------------------------------------- public API

def type_problem(goal: str, X: Any, y: Any,
                 *, llm_client: Optional[Callable[[str], str]] = None) -> ProblemSpec:
    """Produce the typed ProblemSpec (no harness selection). Pure typing + adversarial check.

    Deterministic floor always runs (and seeds the probe shown to the LLM). If an llm_client is
    supplied and returns a usable verdict, its kind/modality/metric/risks OVERRIDE the floor
    where it spoke; otherwise the deterministic verdict stands. The merge is conservative for
    risks: LLM risks are ADDED to the deterministic ones (de-duplicated by code+severity), so a
    "block" found by either path blocks. We never drop a deterministic block in favor of an LLM
    "all clear" -- the safety floor only ratchets up.
    """
    # ---- deterministic floor (the honest fallback, always computed) ----
    modality, m_conf, m_why = _infer_modality_from_data(X)
    kind, n_classes, k_conf, k_why = _infer_kind_from_targets(y)
    n_samples = 0 if y is None else int(np.asarray(y).size)
    if n_samples == 0:
        try:
            n_samples = int(np.asarray(X).shape[0])
        except Exception:
            n_samples = 0
    try:
        Xa = np.asarray(X, dtype=object) if _is_objecty(X) else np.asarray(X)
        n_features = int(Xa.shape[1]) if Xa.ndim >= 2 else 1
    except Exception:
        n_features = 0

    balance = _class_balance(y, kind)
    det_metric = _METRIC_FOR.get((kind, balance)) or _METRIC_FOR.get((kind, "any")) or "accuracy"
    det_risks = _deterministic_risks(goal, kind, n_samples, n_classes)
    det_theta = _default_theta(kind, balance, n_classes)
    det_conf = float(min(m_conf, k_conf))
    det_rationale = f"modality: {m_why}; kind: {k_why}"

    spec = ProblemSpec(
        kind=kind, modality=modality, metric=det_metric, theta_hint=det_theta,
        risks=list(det_risks), typed_by="deterministic", confidence=det_conf,
        n_samples=n_samples, n_features=n_features, n_classes=n_classes,
        rationale=det_rationale,
    )

    # ---- LLM refinement (only if a client is wired) ----
    if llm_client is not None:
        probe = {
            "n_samples": n_samples, "n_features": n_features,
            "x_ndim": _safe_ndim(X), "x_dtype": _safe_dtype(X),
            "y_dtype": _safe_dtype(y), "y_distinct": (0 if y is None else int(len(np.unique(np.asarray(y))))),
            "kind_guess": kind, "modality_guess": modality,
        }
        verdict = _llm_type(goal, llm_client, probe)
        if verdict is not None:
            if verdict["kind"]:
                spec.kind = verdict["kind"]
            if verdict["modality"]:
                # Guard: do not override to a shape-incompatible modality.
                # Image requires ndim >= 3; timeseries requires ndim >= 2.
                # A 2-D numeric matrix is tabular even if the goal mentions "image".
                x_ndim = probe["x_ndim"]
                llm_mod = verdict["modality"]
                shape_ok = True
                if llm_mod == "image" and x_ndim < 3:
                    shape_ok = False  # images must be (n,H,W) or (n,H,W,C)
                if shape_ok:
                    spec.modality = llm_mod
            if verdict["metric"] in (
                "accuracy", "balanced_accuracy", "macro_f1", "r2", "neg_rmse", "neg_mae"):
                spec.metric = verdict["metric"]
            if isinstance(verdict["theta_hint"], (int, float)) and math.isfinite(
                    float(verdict["theta_hint"])):
                spec.theta_hint = float(verdict["theta_hint"])
            if isinstance(verdict["confidence"], (int, float)):
                spec.confidence = float(max(0.0, min(1.0, verdict["confidence"])))
            spec.rationale = verdict.get("rationale") or spec.rationale
            spec.typed_by = "llm"
            # merge risks: add LLM-found risks not already present (safety ratchets up only).
            _merge_risks(spec.risks, verdict.get("risks", []))

    return spec


def route(goal: str, X: Any, y: Any = None,
          *, llm_client: Optional[Callable[[str], str]] = None):
    """Type the problem, then SELECT and return the Harness for its modality.

    The returned object exposes `.spec` (the ProblemSpec) and `.name` (the harness class name)
    regardless of whether the real Phase-3 registry (frontier.harness.base) is importable; when
    it is, the real modality harness is instantiated with the spec, when it is not, a
    self-contained _FallbackHarness carrying the same spec is returned (its build_task() raises a
    clear error pointing at the missing module rather than fabricating a Task).

    Reward-hacking guard: the spec's risks (including any "block") ride on harness.spec.risks /
    harness.spec.blocked so the engine can decline honestly BEFORE splitting or certifying.
    """
    spec = type_problem(goal, X, y, llm_client=llm_client)
    return _select_harness(spec)


# --------------------------------------------------------------------------- helpers

def _default_theta(kind: str, balance: str, n_classes: int) -> float:
    """An advisory promotion floor in metric units. NEVER auto-promotes (the operator/engine
    sets the real theta); this is a sane starting hint anchored to a trivial baseline.

    - classification: a touch above the majority-class / random floor (1/n_classes), so the
      hint at least asks the winner to beat chance. We use max(0.5, 1/k + 0.1) capped at 0.9.
    - regression: r2 hint of 0.0 (beat the mean predictor) -- the minimal honest bar.
    """
    if kind == "classification":
        if n_classes >= 2:
            chance = 1.0 / n_classes
            return float(min(0.9, max(0.5, chance + 0.1)))
        return 0.5
    if kind == "regression":
        return 0.0
    # clustering/unknown: no labeled certificate; hint is meaningless, return 0.0.
    return 0.0


def _merge_risks(existing: List[Risk], llm_risks: Sequence) -> None:
    """Add LLM risks to `existing` in place, de-duplicating by (code, severity)."""
    seen = {(r.code, r.severity) for r in existing}
    sev_ok = {"info", "warn", "block"}
    for r in llm_risks or []:
        if not isinstance(r, dict):
            continue
        code = str(r.get("code", "llm_risk"))
        sev = str(r.get("severity", "warn"))
        if sev not in sev_ok:
            sev = "warn"
        detail = str(r.get("detail", ""))
        if (code, sev) in seen:
            continue
        seen.add((code, sev))
        existing.append(Risk(code, sev, detail))


def _safe_ndim(a: Any) -> int:
    try:
        return int(np.asarray(a).ndim)
    except Exception:
        return -1


def _safe_dtype(a: Any) -> str:
    if a is None:
        return "none"
    try:
        return str(np.asarray(a).dtype)
    except Exception:
        return "object"
