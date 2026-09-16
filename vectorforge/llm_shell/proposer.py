"""Step 2 -- the A1 LLM proposer, now a thin adapter over the A9 `llm_propose` primitive.

A1 supplies the SEMANTIC bound (GoalProposal shape validation) and the deterministic fallback; ops.py
owns caching, quarantine, schema-call mechanics, replay records, and graceful degradation. The goal
string is passed as QUARANTINED untrusted input; the column profile is trusted context. The proposal
is NON_BINDING and is consumed only by the frozen resolver.
"""
from dataclasses import dataclass
from typing import Optional

from . import ops
from .schema import GOAL_PROPOSAL_JSON_SCHEMA, PROMPT_VERSION, GoalProposal, Constraints
from .prompts import SYSTEM
from .profiler import ProfileView

DEFAULT_MODEL = ops.DEFAULT_MODEL
SURFACE = "a1.goal_proposal"
_TOOL_NAME = "emit_goal_proposal"


@dataclass
class ProposeResult:
    proposal: GoalProposal
    used_llm: bool
    cache_hit: bool
    model: Optional[str]
    cache_key: str
    error: Optional[str] = None
    raw: Optional[dict] = None
    usage: Optional[dict] = None


def _profile_context(view: ProfileView) -> dict:
    """The TRUSTED instruction-side context: column schema/stats + structural inference. No raw rows."""
    cols = []
    for name, stats in view.columns.items():
        s = stats if isinstance(stats, dict) else {}
        cols.append({"name": name, "numeric": s.get("numeric"), "n_unique": s.get("n_unique"),
                     "n_missing": s.get("n_missing"), "median_tokens": s.get("median_tokens"),
                     "distinct_frac": s.get("distinct_frac")})
    return {"n_rows": view.n_rows, "modality": view.modality, "columns": cols,
            "structural_inference": {
                "candidate_target": view.candidate_target,
                "candidate_target_rule": view.target_rule,
                "candidate_task_type": view.candidate_task_type,
                "valid_metrics_for_candidate_task_type": list(view.valid_metrics),
                "default_metric": view.default_metric}}


def _build_request(goal: str, view: ProfileView, model: str) -> ops.LLMRequest:
    return ops.LLMRequest(
        surface=SURFACE, system=SYSTEM,
        trusted_context=_profile_context(view),
        untrusted_inputs={"goal": goal},        # quarantined: the goal is user-supplied
        model=model, schema=GOAL_PROPOSAL_JSON_SCHEMA, tool_name=_TOOL_NAME,
        prompt_version=PROMPT_VERSION)


def cache_key(goal: str, view: ProfileView, model: str = DEFAULT_MODEL, tenant_id: str = "default") -> str:
    return ops.request_digest(_build_request(goal, view, model), tenant_id)


def _deterministic_fallback(view: ProfileView, *, reason: str) -> GoalProposal:
    """No-LLM path: the structural inference IS the proposal (exactly today's behavior)."""
    return GoalProposal(target=view.candidate_target, task_type=view.candidate_task_type,
                        metric_intent=None, forbidden_fields=[], constraints=Constraints(),
                        unmapped_phrases=[],
                        rationale=f"deterministic fallback ({reason}): structural inference only, no LLM",
                        source="fallback")


def propose(goal: str, view: ProfileView, *, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
            use_llm: bool = True, cache_path: Optional[str] = None, tenant_id: str = "default",
            max_tokens: int = 40000, timeout: float = 120.0) -> ProposeResult:
    """Produce a GoalProposal via the A9 primitive. Never raises; falls back deterministically."""
    req = _build_request(goal, view, model)
    req.max_tokens = max_tokens

    def _verify(raw):
        try:
            return True, GoalProposal.validate(raw, source="llm").as_dict(), None
        except Exception as ex:  # noqa: BLE001
            return False, None, str(ex)[:160]

    def _fallback(reason):
        return _deterministic_fallback(view, reason=reason).as_dict()

    p = ops.llm_propose(req, kind="NON_BINDING", verify=_verify, fallback=_fallback,
                        tenant_id=tenant_id, api_key=api_key, use_llm=use_llm,
                        cache_path=cache_path, timeout=timeout)
    value = p.value or {}
    proposal = GoalProposal.validate(value, source=value.get("source",
                                                             "llm" if p.used_llm else "fallback"))
    return ProposeResult(proposal=proposal, used_llm=p.used_llm, cache_hit=p.cache_hit,
                         model=(model if p.used_llm else None), cache_key=p.cache_key,
                         error=p.error, raw=value, usage=p.usage)
