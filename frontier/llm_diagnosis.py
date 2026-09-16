"""LLM-powered diagnosis: deep failure analysis + targeted improvement proposals.

Extends the deterministic ``frontier.diagnosis`` module with LLM-driven analysis.
When an LLM client is available, this module:

  1. Analyzes the structured Diagnosis (per-family stats, axis lifts, error patterns,
     residual structure) and generates *targeted* improvement recommendations that go
     beyond the deterministic heuristics in ``diagnosis._build_directives``.

  2. For classification tasks, performs confusion-matrix analysis: identifies the most
     confused class pairs and generates class-specific remediation advice.

  3. For regression tasks, analyzes residual patterns (heteroscedasticity, skew,
     feature-specific error clusters) and proposes architecture/feature changes.

  4. Generates executable code proposals that target the diagnosed weaknesses, fed
     back into the next round's proposal context.

# === WIRING ===
The integrator calls ``llm_diagnose`` after ``diagnosis.diagnose`` in the per-round loop.
It enriches the context dict with LLM-derived guidance under ``context["llm_diagnosis"]``,
which the ``CoreAuthoringProposer`` and ``LLMProposer`` consume in their prompts.

    from frontier.llm_diagnosis import llm_diagnose, enrich_context_with_llm_diagnosis
    ...
    diag = diagnosis.diagnose(history, trail, task)
    llm_diag = llm_diagnose(diag, task, history, llm_client=cfg.llm_client,
                             val_truth=val_truth, val_preds=val_preds)
    enrich_context_with_llm_diagnosis(ctx, llm_diag)

Honest degradation: with ``llm_client=None``, returns an empty LLMDiagnosis (no
fabricated insights). The deterministic diagnosis.py directives remain the floor.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .diagnosis import Diagnosis
from .task import Task


@dataclass
class ConfusionAnalysis:
    """Per-class error analysis from the champion's validation predictions."""
    n_classes: int
    total_errors: int
    total_correct: int
    error_rate: float
    confused_pairs: List[Dict[str, Any]]    # [{true, pred, count, share}, ...]
    per_class_accuracy: Dict[str, float]    # class_label -> accuracy
    weakest_class: Optional[str]            # class with lowest accuracy
    strongest_class: Optional[str]          # class with highest accuracy


@dataclass
class LLMDiagnosis:
    """LLM-enhanced diagnosis result. Extends the deterministic Diagnosis with deeper insights."""
    available: bool                          # True iff an LLM produced the analysis
    targeted_guidance: str                   # LLM-generated improvement strategy
    code_suggestions: List[str]             # suggested code snippets for improvement
    confusion_analysis: Optional[ConfusionAnalysis]  # classification only
    residual_analysis: str                   # LLM analysis of residual patterns (regression)
    root_causes: List[str]                   # identified root causes of poor performance
    recommended_architectures: List[str]     # specific architecture suggestions
    repair_strategies: List[str]             # targeted repair strategies for failing proposals

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "available": self.available,
            "targeted_guidance": self.targeted_guidance,
            "code_suggestions": self.code_suggestions,
            "residual_analysis": self.residual_analysis,
            "root_causes": self.root_causes,
            "recommended_architectures": self.recommended_architectures,
            "repair_strategies": self.repair_strategies,
        }
        if self.confusion_analysis is not None:
            ca = self.confusion_analysis
            d["confusion_analysis"] = {
                "n_classes": ca.n_classes,
                "total_errors": ca.total_errors,
                "error_rate": round(ca.error_rate, 4),
                "confused_pairs": ca.confused_pairs[:5],
                "weakest_class": ca.weakest_class,
                "strongest_class": ca.strongest_class,
            }
        return d


def _empty_diagnosis() -> LLMDiagnosis:
    return LLMDiagnosis(
        available=False, targeted_guidance="", code_suggestions=[],
        confusion_analysis=None, residual_analysis="",
        root_causes=[], recommended_architectures=[], repair_strategies=[],
    )


# --------------------------------------------------------------------------- confusion matrix

def _build_confusion_analysis(
    val_truth: Sequence, val_preds: Sequence,
) -> Optional[ConfusionAnalysis]:
    """Build a confusion analysis from validation truth/predictions (classification only)."""
    if val_truth is None or val_preds is None:
        return None
    if len(val_truth) == 0 or len(val_truth) != len(val_preds):
        return None

    yt = [str(v) for v in val_truth]
    yp = [str(v) for v in val_preds]
    classes = sorted(set(yt))
    n_classes = len(classes)

    # Per-class accuracy
    class_correct: Counter = Counter()
    class_total: Counter = Counter()
    confusion: Counter = Counter()

    for t, p in zip(yt, yp):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1
        else:
            confusion[(t, p)] += 1

    total_correct = sum(class_correct.values())
    total_errors = len(yt) - total_correct
    error_rate = total_errors / max(1, len(yt))

    per_class_acc = {}
    for c in classes:
        if class_total[c] > 0:
            per_class_acc[c] = class_correct[c] / class_total[c]
        else:
            per_class_acc[c] = 0.0

    # Most confused pairs
    confused_pairs = []
    for (true_c, pred_c), count in confusion.most_common(10):
        confused_pairs.append({
            "true": true_c, "pred": pred_c, "count": count,
            "share": round(count / max(1, total_errors), 3),
        })

    weakest = min(per_class_acc, key=per_class_acc.get) if per_class_acc else None
    strongest = max(per_class_acc, key=per_class_acc.get) if per_class_acc else None

    return ConfusionAnalysis(
        n_classes=n_classes, total_errors=total_errors, total_correct=total_correct,
        error_rate=error_rate, confused_pairs=confused_pairs,
        per_class_accuracy=per_class_acc, weakest_class=weakest,
        strongest_class=strongest,
    )


# --------------------------------------------------------------------------- LLM prompts

def _build_diagnosis_prompt(
    diag: Diagnosis, task: Task,
    confusion: Optional[ConfusionAnalysis],
    history: Sequence,
) -> str:
    """Build a structured prompt for the LLM to analyze failure patterns."""
    parts = [
        "You are analyzing an ML experiment to diagnose why performance is stuck and propose "
        "targeted improvements.\n",
    ]

    # Task context
    parts.append(
        f"TASK: {task.kind} with {task.n_features} features, "
        f"metric={task.metric}, theta={task.theta}.\n"
    )

    # Current state
    parts.append(
        f"CURRENT STATE: round={diag.round_index}, {diag.n_ok} succeeded / "
        f"{diag.n_fail} failed out of {diag.n_candidates} candidates. "
        f"Best val score: {diag.best_score} (label: {diag.best_label}).\n"
    )

    # Plateau info
    if diag.plateau:
        parts.append(
            f"WARNING: search has PLATEAUED for {diag.plateau_span} rounds. "
            f"Score not improving.\n"
        )

    # Error pattern
    if diag.dominant_error_kind:
        parts.append(
            f"DOMINANT ERROR: '{diag.dominant_error_kind}' accounts for "
            f"{diag.dominant_error_share:.0%} of failures.\n"
        )

    # Family performance
    if diag.family_rank:
        fam_lines = []
        for f in diag.family_rank[:6]:
            status = f"best={f.best:.4f}" if f.best is not None else "never succeeded"
            fam_lines.append(f"  {f.family}: {f.n_ok}ok/{f.n_fail}fail, {status}")
        parts.append("FAMILY PERFORMANCE:\n" + "\n".join(fam_lines) + "\n")

    # Axis lift
    active_lifts = {k: v for k, v in diag.axis_lift.items() if v is not None}
    if active_lifts:
        lift_lines = [f"  {k}: {v:+.4f}" for k, v in active_lifts.items()]
        parts.append("AXIS LIFT (positive=helps, negative=hurts):\n" + "\n".join(lift_lines) + "\n")

    # Residual structure
    if diag.residual_summary:
        parts.append(f"RESIDUAL STRUCTURE: {diag.residual_summary}\n")

    # Confusion matrix (classification)
    if confusion is not None:
        parts.append(
            f"CONFUSION ANALYSIS: {confusion.n_classes} classes, "
            f"error_rate={confusion.error_rate:.3f}.\n"
        )
        if confusion.confused_pairs:
            pair_lines = [
                f"  class '{p['true']}' misclassified as '{p['pred']}': "
                f"{p['count']} times ({p['share']:.0%} of errors)"
                for p in confusion.confused_pairs[:5]
            ]
            parts.append("MOST CONFUSED PAIRS:\n" + "\n".join(pair_lines) + "\n")
        if confusion.weakest_class is not None:
            acc = confusion.per_class_accuracy.get(confusion.weakest_class, 0)
            parts.append(
                f"WEAKEST CLASS: '{confusion.weakest_class}' "
                f"(accuracy={acc:.3f})\n"
            )

    # Recent errors
    err_samples = []
    for r in history:
        ek = getattr(r, 'error_kind', '') or (r.get('error_kind', '') if isinstance(r, dict) else '')
        err = getattr(r, 'error', '') or (r.get('error', '') if isinstance(r, dict) else '')
        ok = getattr(r, 'ok', True) if not isinstance(r, dict) else r.get('ok', True)
        if not ok and ek:
            err_samples.append(f"  [{ek}] {err[:120]}")
    if err_samples:
        parts.append("RECENT ERROR SAMPLES:\n" + "\n".join(err_samples[:5]) + "\n")

    parts.append(
        "\nBased on this analysis, provide:\n"
        "1. ROOT CAUSES: 2-3 specific reasons why performance is limited.\n"
        "2. TARGETED IMPROVEMENTS: 2-3 specific, actionable changes (architecture, "
        "features, preprocessing, hyperparameters).\n"
        "3. ARCHITECTURE RECOMMENDATIONS: 1-2 specific model architectures to try "
        "(with sklearn/pytorch class names).\n"
        "4. REPAIR STRATEGIES: For the failing proposals, 1-2 specific code fixes.\n"
        "\nBe specific and technical. Reference sklearn/torch class names. "
        "Keep each section to 2-3 sentences."
    )
    return "\n".join(parts)


def _build_repair_prompt(
    code: str, error_kind: str, error: str,
    diag: Diagnosis, task: Task,
) -> str:
    """Build a targeted repair prompt using diagnosis context."""
    parts = [
        "You are repairing a failing ML pipeline. Use the diagnosis context below to "
        "understand the broader experiment state and propose a TARGETED fix.\n",
        f"TASK: {task.kind}, {task.n_features} features.\n",
        f"EXPERIMENT STATE: best_score={diag.best_score}, "
        f"dominant_error='{diag.dominant_error_kind}', plateau={diag.plateau}.\n",
        f"THIS CANDIDATE'S ERROR: [{error_kind}] {error}\n",
        f"CODE:\n{code}\n",
        "Return ONLY the corrected Python code defining build_estimator(). "
        "No markdown fences, no explanation.",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- parse response

def _parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse the LLM's structured analysis into sections."""
    sections: Dict[str, Any] = {
        "root_causes": [],
        "targeted_guidance": "",
        "recommended_architectures": [],
        "repair_strategies": [],
        "code_suggestions": [],
    }

    if not response:
        return sections

    # Extract sections by headers
    current_section = "targeted_guidance"
    current_lines: List[str] = []

    for line in response.split("\n"):
        stripped = line.strip()
        # Strip leading numbering (e.g., "1.", "2.") and markdown bold markers
        clean = stripped.lstrip("0123456789.)#*_ ").strip()
        lower = clean.lower()

        new_section = None
        if "root cause" in lower:
            new_section = "root_causes"
        elif "targeted" in lower and ("improvement" in lower or "guidance" in lower):
            new_section = "targeted_guidance"
        elif "architecture" in lower and ("recommend" in lower or "suggestion" in lower):
            new_section = "recommended_architectures"
        elif "repair" in lower and "strateg" in lower:
            new_section = "repair_strategies"

        if new_section is not None:
            _flush(sections, current_section, current_lines)
            current_section = new_section
            current_lines = []
            # Capture content after the colon on the header line itself
            after = clean.split(":", 1)[1].strip() if ":" in clean else ""
            if after:
                current_lines.append(after)
        elif stripped:
            current_lines.append(stripped)

    _flush(sections, current_section, current_lines)
    return sections


def _flush(sections: Dict[str, Any], section: str, lines: List[str]) -> None:
    """Flush accumulated lines into the appropriate section."""
    if not lines:
        return
    text = "\n".join(lines)
    if section == "targeted_guidance":
        sections["targeted_guidance"] = text
    elif section in ("root_causes", "recommended_architectures", "repair_strategies",
                     "code_suggestions"):
        # Extract bullet points
        items = []
        for line in lines:
            # Strip bullet markers
            cleaned = line.lstrip("- *0123456789.)").strip()
            if cleaned:
                items.append(cleaned)
        sections[section] = items


# --------------------------------------------------------------------------- public API

def llm_diagnose(
    diag: Diagnosis,
    task: Task,
    history: Sequence,
    *,
    llm_client: Optional[Callable[[str], str]] = None,
    val_truth: Optional[Sequence] = None,
    val_preds: Optional[Sequence] = None,
) -> LLMDiagnosis:
    """Run LLM-powered diagnosis on the experiment state.

    Parameters
    ----------
    diag : the deterministic Diagnosis from frontier.diagnosis.diagnose
    task : the Task being solved
    history : per-candidate _Record log
    llm_client : (prompt: str) -> str, or None for honest degradation
    val_truth, val_preds : champion's validation truth/predictions for confusion analysis

    Returns
    -------
    LLMDiagnosis with targeted improvement strategies. Empty when no LLM available.
    """
    if llm_client is None:
        result = _empty_diagnosis()
        # Still compute confusion analysis even without LLM
        if task.kind == "classification" and val_truth is not None:
            result.confusion_analysis = _build_confusion_analysis(val_truth, val_preds)
        return result

    # Build confusion analysis (classification only)
    confusion = None
    if task.kind == "classification" and val_truth is not None:
        confusion = _build_confusion_analysis(val_truth, val_preds)

    # Build prompt and call LLM
    prompt = _build_diagnosis_prompt(diag, task, confusion, history)

    try:
        response = llm_client(prompt)
    except Exception:
        result = _empty_diagnosis()
        result.confusion_analysis = confusion
        return result

    if not response:
        result = _empty_diagnosis()
        result.confusion_analysis = confusion
        return result

    # Parse the LLM response
    parsed = _parse_llm_response(response)

    return LLMDiagnosis(
        available=True,
        targeted_guidance=parsed.get("targeted_guidance", ""),
        code_suggestions=parsed.get("code_suggestions", []),
        confusion_analysis=confusion,
        residual_analysis=diag.residual_summary,
        root_causes=parsed.get("root_causes", []),
        recommended_architectures=parsed.get("recommended_architectures", []),
        repair_strategies=parsed.get("repair_strategies", []),
    )


def enrich_context_with_llm_diagnosis(
    context: Dict[str, Any], llm_diag: LLMDiagnosis,
) -> Dict[str, Any]:
    """Inject LLM diagnosis into the proposer context (in place, additive).

    Adds:
      - context["llm_diagnosis"]     : full LLM diagnosis dict
      - context["llm_guidance"]      : targeted guidance string for LLM proposers
      - context["confusion"]         : confusion analysis dict (classification only)
      - context["repair_strategies"] : targeted repair strategies

    Never removes or overwrites Phase-0 keys or deterministic diagnosis keys.
    """
    context["llm_diagnosis"] = llm_diag.as_dict()
    if llm_diag.available and llm_diag.targeted_guidance:
        existing = context.get("diag_guidance", "")
        context["llm_guidance"] = (
            existing + "\n\n[LLM DIAGNOSIS]\n" + llm_diag.targeted_guidance
            if existing else llm_diag.targeted_guidance
        )
    else:
        context["llm_guidance"] = context.get("diag_guidance", "")
    if llm_diag.confusion_analysis is not None:
        ca = llm_diag.confusion_analysis
        context["confusion"] = {
            "error_rate": ca.error_rate,
            "weakest_class": ca.weakest_class,
            "confused_pairs": ca.confused_pairs[:3],
        }
    if llm_diag.repair_strategies:
        context["repair_strategies"] = llm_diag.repair_strategies
    return context


__all__ = [
    "LLMDiagnosis", "ConfusionAnalysis",
    "llm_diagnose", "enrich_context_with_llm_diagnosis",
]
