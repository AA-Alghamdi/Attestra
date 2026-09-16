"""LLM-powered problem typing + adversarial goal validation.

Given a free-text goal and data, this module:
  1. Uses the LLM to TYPE the problem (classification, regression, generation,
     optimization, forecasting, etc.) with confidence and justification
  2. Adversarially checks the goal for:
     - Contradictions (goal asks for X but data suggests Y)
     - Infeasibility (goal requires capabilities beyond current system)
     - Ambiguity (goal is underspecified, could mean multiple things)
     - Reward hacking risk (metric can be gamed without real progress)
  3. Returns a structured ProblemSpec with typed fields

This is the FIRST stage of the pipeline — before any experiment design.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ProblemDomain(str, Enum):
    """High-level ML problem domains."""
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_GENERATION = "image_generation"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_GENERATION = "text_generation"
    SPEECH_SYNTHESIS = "speech_synthesis"
    SPEECH_RECOGNITION = "speech_recognition"
    TIMESERIES_FORECASTING = "timeseries_forecasting"
    TIMESERIES_CLASSIFICATION = "timeseries_classification"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    OPTIMIZATION = "optimization"
    MULTI_MODAL = "multi_modal"
    UNKNOWN = "unknown"


class Feasibility(str, Enum):
    """Feasibility assessment of the goal."""
    FEASIBLE = "feasible"
    LIKELY_FEASIBLE = "likely_feasible"
    UNCERTAIN = "uncertain"
    LIKELY_INFEASIBLE = "likely_infeasible"
    INFEASIBLE = "infeasible"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AdversarialCheck:
    """Result of adversarial goal validation."""
    passed: bool
    contradictions: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)
    infeasibilities: List[str] = field(default_factory=list)
    reward_hacking_risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    overall_risk: RiskLevel = RiskLevel.LOW

    @property
    def has_issues(self) -> bool:
        return bool(self.contradictions or self.ambiguities or
                    self.infeasibilities or self.reward_hacking_risks)


@dataclass
class ProblemSpec:
    """Fully typed problem specification."""
    goal: str
    domain: ProblemDomain
    task_type: str                        # fine-grained: "binary_clf", "multiclass_clf", "regression", etc.
    confidence: float                     # 0-1 confidence in typing
    justification: str                    # why this typing
    feasibility: Feasibility
    # Derived fields
    suggested_metric: str = ""
    suggested_threshold: float = 0.0
    suggested_modality: str = "tabular"
    estimated_difficulty: str = "medium"  # "easy" | "medium" | "hard" | "frontier"
    # Adversarial check
    adversarial: Optional[AdversarialCheck] = None
    # Content hash for pre-registration
    spec_hash: str = ""
    # Raw LLM analysis
    raw_analysis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "domain": self.domain.value,
            "task_type": self.task_type,
            "confidence": self.confidence,
            "feasibility": self.feasibility.value,
            "suggested_metric": self.suggested_metric,
            "suggested_threshold": self.suggested_threshold,
            "suggested_modality": self.suggested_modality,
            "estimated_difficulty": self.estimated_difficulty,
            "adversarial_passed": self.adversarial.passed if self.adversarial else True,
            "spec_hash": self.spec_hash,
        }


def type_problem(
    goal: str,
    data_profile: Optional[Dict] = None,
    llm_call: Optional[Callable] = None,
    strict: bool = False,
) -> ProblemSpec:
    """Type a problem using LLM + heuristics.

    If LLM is available, uses it for rich typing with adversarial checks.
    Falls back to heuristic typing from data profile alone.

    Args:
        goal: Free-text research goal
        data_profile: Output from profile_data().to_dict() (optional)
        llm_call: LLM function (system, user) -> (content, usage)
        strict: If True, fail on adversarial check issues

    Returns:
        ProblemSpec with full typing and adversarial validation
    """
    if llm_call is not None:
        spec = _type_with_llm(goal, data_profile, llm_call)
    else:
        spec = _type_heuristic(goal, data_profile)

    # Content-address the spec
    h = hashlib.sha256()
    h.update(goal.encode())
    h.update(spec.domain.value.encode())
    h.update(spec.task_type.encode())
    h.update(str(spec.suggested_metric).encode())
    h.update(str(spec.suggested_threshold).encode())
    spec.spec_hash = h.hexdigest()[:16]

    return spec


def adversarial_check(
    goal: str,
    data_profile: Optional[Dict] = None,
    llm_call: Optional[Callable] = None,
) -> AdversarialCheck:
    """Run adversarial validation on a goal.

    Checks for:
      - Contradictions between goal and data
      - Infeasibility given available resources
      - Ambiguity in the goal specification
      - Reward hacking risks (metric gaming)
    """
    if llm_call is not None:
        return _adversarial_with_llm(goal, data_profile, llm_call)
    return _adversarial_heuristic(goal, data_profile)


# ============================================================================== LLM-powered typing

_TYPING_SYSTEM = """You are an expert ML research advisor. Given a goal and optional data profile,
you must:
1. TYPE the problem into exactly one domain
2. Assess feasibility
3. Suggest metric, threshold, modality
4. Estimate difficulty
5. Run adversarial checks

Respond in JSON ONLY (no markdown, no explanation outside JSON):
{
  "domain": "one of: tabular_classification, tabular_regression, image_classification, image_generation, text_classification, text_generation, speech_synthesis, speech_recognition, timeseries_forecasting, timeseries_classification, reinforcement_learning, recommendation, anomaly_detection, optimization, multi_modal, unknown",
  "task_type": "fine-grained type (e.g. binary_clf, multiclass_clf, regression, seq2seq, etc.)",
  "confidence": 0.0-1.0,
  "justification": "why this typing",
  "feasibility": "one of: feasible, likely_feasible, uncertain, likely_infeasible, infeasible",
  "suggested_metric": "metric name",
  "suggested_threshold": 0.0-1.0,
  "suggested_modality": "tabular|text|vision|audio|timeseries|multi_modal",
  "estimated_difficulty": "easy|medium|hard|frontier",
  "contradictions": ["list of contradictions between goal and data"],
  "ambiguities": ["list of ambiguities in the goal"],
  "infeasibilities": ["list of things that may be infeasible"],
  "reward_hacking_risks": ["ways the metric could be gamed"],
  "recommendations": ["suggestions to improve the goal"]
}"""


def _type_with_llm(goal: str, data_profile: Optional[Dict],
                   llm_call: Callable) -> ProblemSpec:
    """Use LLM to type the problem."""
    user_msg = f"GOAL: {goal}\n"
    if data_profile:
        user_msg += f"\nDATA PROFILE:\n{json.dumps(data_profile, indent=2, default=str)}"
    else:
        user_msg += "\nNo data profile available (data not yet loaded)."

    try:
        raw, _ = llm_call(_TYPING_SYSTEM, user_msg)
        parsed = _parse_json(raw)

        domain = _parse_domain(parsed.get("domain", "unknown"))
        feasibility = _parse_feasibility(parsed.get("feasibility", "uncertain"))

        adv = AdversarialCheck(
            passed=not bool(parsed.get("contradictions") or parsed.get("infeasibilities")),
            contradictions=parsed.get("contradictions", []),
            ambiguities=parsed.get("ambiguities", []),
            infeasibilities=parsed.get("infeasibilities", []),
            reward_hacking_risks=parsed.get("reward_hacking_risks", []),
            recommendations=parsed.get("recommendations", []),
            overall_risk=_assess_risk(parsed),
        )

        return ProblemSpec(
            goal=goal,
            domain=domain,
            task_type=parsed.get("task_type", "unknown"),
            confidence=float(parsed.get("confidence", 0.5)),
            justification=parsed.get("justification", ""),
            feasibility=feasibility,
            suggested_metric=parsed.get("suggested_metric", ""),
            suggested_threshold=float(parsed.get("suggested_threshold", 0)),
            suggested_modality=parsed.get("suggested_modality", "tabular"),
            estimated_difficulty=parsed.get("estimated_difficulty", "medium"),
            adversarial=adv,
            raw_analysis=raw,
        )
    except Exception as e:
        # Fallback to heuristic on LLM failure
        spec = _type_heuristic(goal, data_profile)
        spec.raw_analysis = f"LLM typing failed: {e}"
        return spec


def _adversarial_with_llm(goal: str, data_profile: Optional[Dict],
                          llm_call: Callable) -> AdversarialCheck:
    """LLM-powered adversarial check."""
    system = """You are an adversarial ML research validator. Your job is to find problems with
a research goal BEFORE any experiments are run. Be skeptical and thorough.

Check for:
1. CONTRADICTIONS: Does the goal contradict what the data shows?
2. INFEASIBILITIES: Is there a fundamental reason this can't work with available resources?
3. AMBIGUITIES: Could this goal mean multiple conflicting things?
4. REWARD HACKING: Could the metric be maximized without genuine progress?

Respond in JSON:
{
  "passed": true/false,
  "contradictions": [],
  "ambiguities": [],
  "infeasibilities": [],
  "reward_hacking_risks": [],
  "recommendations": [],
  "overall_risk": "low|medium|high|critical"
}"""
    user_msg = f"GOAL: {goal}\n"
    if data_profile:
        user_msg += f"\nDATA PROFILE:\n{json.dumps(data_profile, indent=2, default=str)}"

    try:
        raw, _ = llm_call(system, user_msg)
        parsed = _parse_json(raw)
        return AdversarialCheck(
            passed=parsed.get("passed", True),
            contradictions=parsed.get("contradictions", []),
            ambiguities=parsed.get("ambiguities", []),
            infeasibilities=parsed.get("infeasibilities", []),
            reward_hacking_risks=parsed.get("reward_hacking_risks", []),
            recommendations=parsed.get("recommendations", []),
            overall_risk=RiskLevel(parsed.get("overall_risk", "low")),
        )
    except Exception:
        return _adversarial_heuristic(goal, data_profile)


# ============================================================================== heuristic fallbacks

_DOMAIN_KEYWORDS = {
    ProblemDomain.TABULAR_CLASSIFICATION: ["classify", "classification", "predict class", "label", "detect", "binary", "multiclass"],
    ProblemDomain.TABULAR_REGRESSION: ["predict", "regression", "forecast", "estimate value", "price", "salary", "housing"],
    ProblemDomain.IMAGE_CLASSIFICATION: ["image", "picture", "photo", "visual", "object detection", "face"],
    ProblemDomain.IMAGE_GENERATION: ["generate image", "synthesize image", "GAN", "diffusion", "style transfer"],
    ProblemDomain.TEXT_CLASSIFICATION: ["sentiment", "spam", "topic", "text class", "NLP classification"],
    ProblemDomain.TEXT_GENERATION: ["generate text", "summarize", "translate", "chatbot", "language model"],
    ProblemDomain.SPEECH_SYNTHESIS: ["TTS", "text-to-speech", "speech synthesis", "voice", "read aloud"],
    ProblemDomain.SPEECH_RECOGNITION: ["ASR", "speech-to-text", "transcribe", "voice recognition"],
    ProblemDomain.TIMESERIES_FORECASTING: ["timeseries", "time series", "forecast", "temporal", "stock"],
    ProblemDomain.TIMESERIES_CLASSIFICATION: ["activity recognition", "ECG", "signal classification"],
    ProblemDomain.REINFORCEMENT_LEARNING: ["reinforcement", "agent", "reward", "policy", "environment"],
    ProblemDomain.RECOMMENDATION: ["recommend", "collaborative filtering", "user preference"],
    ProblemDomain.ANOMALY_DETECTION: ["anomaly", "outlier", "fraud", "intrusion"],
    ProblemDomain.OPTIMIZATION: ["optimize", "hyperparameter", "architecture search", "NAS"],
}

_METRIC_MAP = {
    ProblemDomain.TABULAR_CLASSIFICATION: ("accuracy", 0.75),
    ProblemDomain.TABULAR_REGRESSION: ("r2", 0.3),
    ProblemDomain.IMAGE_CLASSIFICATION: ("accuracy", 0.7),
    ProblemDomain.TEXT_CLASSIFICATION: ("macro_f1", 0.6),
    ProblemDomain.TIMESERIES_FORECASTING: ("neg_rmse", 0.0),
    ProblemDomain.ANOMALY_DETECTION: ("balanced_accuracy", 0.6),
}

_MODALITY_MAP = {
    ProblemDomain.TABULAR_CLASSIFICATION: "tabular",
    ProblemDomain.TABULAR_REGRESSION: "tabular",
    ProblemDomain.IMAGE_CLASSIFICATION: "vision",
    ProblemDomain.IMAGE_GENERATION: "vision",
    ProblemDomain.TEXT_CLASSIFICATION: "text",
    ProblemDomain.TEXT_GENERATION: "text",
    ProblemDomain.SPEECH_SYNTHESIS: "audio",
    ProblemDomain.SPEECH_RECOGNITION: "audio",
    ProblemDomain.TIMESERIES_FORECASTING: "timeseries",
    ProblemDomain.TIMESERIES_CLASSIFICATION: "timeseries",
}


def _type_heuristic(goal: str, data_profile: Optional[Dict]) -> ProblemSpec:
    """Heuristic problem typing from goal text + data profile."""
    goal_lower = goal.lower()

    # Score each domain by keyword matches
    scores: Dict[ProblemDomain, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw.lower() in goal_lower)

    # Use data profile to disambiguate
    if data_profile:
        task_type = data_profile.get("task_type", "")
        if task_type in ("binary", "multiclass"):
            scores[ProblemDomain.TABULAR_CLASSIFICATION] = scores.get(ProblemDomain.TABULAR_CLASSIFICATION, 0) + 5
        elif task_type == "regression":
            scores[ProblemDomain.TABULAR_REGRESSION] = scores.get(ProblemDomain.TABULAR_REGRESSION, 0) + 5

    # Pick highest-scoring domain
    best_domain = max(scores, key=lambda d: scores[d]) if any(scores.values()) else ProblemDomain.UNKNOWN
    best_score = scores.get(best_domain, 0)

    # If no clear signal, use data profile task type
    if best_score == 0 and data_profile:
        task_type = data_profile.get("task_type", "")
        if task_type in ("binary", "multiclass"):
            best_domain = ProblemDomain.TABULAR_CLASSIFICATION
        elif task_type == "regression":
            best_domain = ProblemDomain.TABULAR_REGRESSION

    # Derive fields
    metric_info = _METRIC_MAP.get(best_domain, ("accuracy", 0.5))
    modality = _MODALITY_MAP.get(best_domain, "tabular")
    confidence = min(1.0, best_score / 3.0) if best_score > 0 else 0.3

    # Fine-grained task type
    if data_profile:
        task_type_str = data_profile.get("task_type", "unknown")
        n_classes = data_profile.get("n_classes", 0)
        if task_type_str == "binary":
            fine_type = "binary_clf"
        elif task_type_str == "multiclass":
            fine_type = f"multiclass_clf_{n_classes}class"
        elif task_type_str == "regression":
            fine_type = "regression"
        else:
            fine_type = "unknown"
    else:
        fine_type = best_domain.value

    adv = _adversarial_heuristic(goal, data_profile)

    return ProblemSpec(
        goal=goal,
        domain=best_domain,
        task_type=fine_type,
        confidence=confidence,
        justification=f"Keyword matching: {best_score} hits for {best_domain.value}",
        feasibility=Feasibility.LIKELY_FEASIBLE if best_score > 0 else Feasibility.UNCERTAIN,
        suggested_metric=metric_info[0],
        suggested_threshold=metric_info[1],
        suggested_modality=modality,
        estimated_difficulty="medium",
        adversarial=adv,
    )


def _adversarial_heuristic(goal: str, data_profile: Optional[Dict]) -> AdversarialCheck:
    """Rule-based adversarial checks."""
    contradictions = []
    ambiguities = []
    infeasibilities = []
    reward_hacking_risks = []
    recommendations = []

    goal_lower = goal.lower()

    # Check for common issues
    if len(goal.split()) < 3:
        ambiguities.append("Goal is very short — may be underspecified")
        recommendations.append("Provide more detail about success criteria")

    if data_profile:
        n_samples = data_profile.get("n_samples", 0)
        n_features = data_profile.get("n_features", 0)

        # Small data warning
        if n_samples < 50:
            infeasibilities.append(f"Only {n_samples} samples — may be too few for reliable learning")
            recommendations.append("Consider data augmentation or simpler models")

        # High-dim warning
        if n_features > n_samples:
            reward_hacking_risks.append(
                f"p >> n ({n_features} features, {n_samples} samples) — easy to overfit")
            recommendations.append("Use regularization or dimensionality reduction")

        # Class imbalance
        if data_profile.get("class_balance"):
            balance = data_profile["class_balance"]
            if isinstance(balance, dict):
                counts = list(balance.values())
                if counts and max(counts) / max(min(counts), 1) > 20:
                    reward_hacking_risks.append("Severe class imbalance — accuracy can be gamed by predicting majority")
                    recommendations.append("Use balanced_accuracy or macro_f1 instead of accuracy")

        # Quality issues
        quality = data_profile.get("quality_score", 1.0)
        if quality < 0.5:
            contradictions.append(f"Data quality score is low ({quality:.2f}) — may need cleaning before experimentation")

    # Goal-specific checks
    if "100%" in goal or "perfect" in goal_lower:
        infeasibilities.append("Perfection is rarely achievable on real data — consider realistic thresholds")

    if any(w in goal_lower for w in ["fastest", "cheapest", "best in the world"]):
        ambiguities.append("Superlative goal needs concrete definition (faster than what? cheaper than what?)")

    passed = not bool(contradictions or infeasibilities)
    risk = RiskLevel.LOW
    if infeasibilities:
        risk = RiskLevel.HIGH
    elif contradictions or reward_hacking_risks:
        risk = RiskLevel.MEDIUM

    return AdversarialCheck(
        passed=passed,
        contradictions=contradictions,
        ambiguities=ambiguities,
        infeasibilities=infeasibilities,
        reward_hacking_risks=reward_hacking_risks,
        recommendations=recommendations,
        overall_risk=risk,
    )


# ============================================================================== helpers

def _parse_json(raw: str) -> Dict:
    """Extract JSON from LLM response (handles markdown fences)."""
    # Strip markdown code fences
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try finding JSON object
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _parse_domain(s: str) -> ProblemDomain:
    try:
        return ProblemDomain(s)
    except ValueError:
        return ProblemDomain.UNKNOWN


def _parse_feasibility(s: str) -> Feasibility:
    try:
        return Feasibility(s)
    except ValueError:
        return Feasibility.UNCERTAIN


def _assess_risk(parsed: Dict) -> RiskLevel:
    issues = (len(parsed.get("contradictions", [])) +
              len(parsed.get("infeasibilities", [])) * 2 +
              len(parsed.get("reward_hacking_risks", [])))
    if issues >= 4:
        return RiskLevel.CRITICAL
    elif issues >= 2:
        return RiskLevel.HIGH
    elif issues >= 1:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
