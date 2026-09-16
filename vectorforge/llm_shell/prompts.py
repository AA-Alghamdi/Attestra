"""The versioned prompt template for the A1 goal proposer.

Bumping the template MUST bump PROMPT_VERSION in schema.py (the cache key and the golden-set CI both
key on it), so a wording change is an observable, re-reviewable event -- never a silent drift.
"""
import json

from .profiler import ProfileView

SYSTEM = (
    "You are the goal-intake proposer for VectorForge, an autonomous ML system whose certificates are "
    "produced by a FROZEN deterministic verifier you never touch. Your job is narrow and bounded: read a "
    "natural-language goal and a dataset's COLUMN SCHEMA (names + statistics only -- never row values), "
    "and emit a single structured GoalProposal describing how the goal maps to a task specification.\n\n"
    "Hard rules:\n"
    "1. `target` and every entry of `forbidden_fields` MUST be an exact column name from the provided "
    "schema. Never invent a column. If the goal implies no clear target, set target to null.\n"
    "2. You do NOT set a threshold or a pass/fail bar of any kind. That number is measured by a separate "
    "frozen component. Express only the evaluation INTENT in `metric_intent` (plain words are fine).\n"
    "3. `forbidden_fields` is purely subtractive -- columns the user said not to use (policy/fairness/"
    "privacy exclusions like 'don't use zipcode'). Never put the target here.\n"
    "4. `unmapped_phrases` is REQUIRED. List every phrase in the goal you could NOT honor or map to a "
    "field (an unsupported ask, an ambiguous reference, a constraint with no column). When in doubt, "
    "surface it here rather than guessing. An honest 'I could not map this' is worth more than a guess.\n"
    "5. Prefer null/empty over fabrication. A downstream frozen resolver will reject anything that does "
    "not resolve to a real column or a valid metric, so guessing only wastes a round-trip.\n"
    "Call the emit_goal_proposal tool exactly once with your proposal."
)


def build_user_prompt(goal: str, view: ProfileView) -> str:
    """Render the goal + the goal-free structural profile (schema/stats only) into the user turn."""
    cols = []
    for name, stats in view.columns.items():
        s = stats if isinstance(stats, dict) else {}
        cols.append({
            "name": name,
            "numeric": s.get("numeric"),
            "n_unique": s.get("n_unique"),
            "n_missing": s.get("n_missing"),
            "median_tokens": s.get("median_tokens"),
            "distinct_frac": s.get("distinct_frac"),
        })
    payload = {
        "goal": goal,
        "n_rows": view.n_rows,
        "modality": view.modality,
        "columns": cols,
        "structural_inference": {
            "candidate_target": view.candidate_target,
            "candidate_target_rule": view.target_rule,
            "candidate_task_type": view.candidate_task_type,
            "valid_metrics_for_candidate_task_type": list(view.valid_metrics),
            "default_metric": view.default_metric,
        },
        "note": ("The structural_inference above is a deterministic baseline. Agree with it unless the "
                 "goal clearly implies otherwise; if you override task_type, the metric_intent must be "
                 "appropriate for the new type. Row values are intentionally omitted."),
    }
    return ("Dataset profile and goal (JSON below). Emit one GoalProposal.\n\n"
            + json.dumps(payload, indent=2, default=str))
