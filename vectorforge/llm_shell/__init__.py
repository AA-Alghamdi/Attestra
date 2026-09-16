"""VectorForge LLM-shell -- A1 goal intake & spec inference behind a frozen harness.

An LLM proposes a draft TaskSpec from a natural-language goal; a frozen resolver narrows or rejects it
against the real columns and the ontology; the threshold and the certificate stay entirely with the
frozen deterministic core (ar.spec.propose_spec / science). The LLM is never on the promotion path.

Public surface:
    from llm_shell import intake, approve
    r = intake(records_or_path, "build a fair churn predictor, don't use zipcode")
    print(r.summary())                       # reviewable contract + decisions + human_review
    r = approve(r, approved_by="alice")       # human freeze -> content-addressed digest

Set use_llm=False (or run with no ANTHROPIC_API_KEY) to get the deterministic structural behavior --
the product remains runnable with no agent in the loop.
"""
# The natural-language INTAKE path (intake/proposer/resolver) depends on the external autoresearch
# ontology package `ar`. It is OPTIONAL: if `ar` is absent (e.g. a slim deployment container), the package
# still imports and the deterministic/served paths work -- only free-text goal inference is unavailable
# (callers pass an explicit kind/task_type instead). This is the scheme's "delete the LLM -> deterministic".
try:
    from .intake import intake, approve, IntakeResult
    from .proposer import propose, ProposeResult
    from .resolver import resolve, ResolvedSpec
    LLM_INTAKE_AVAILABLE = True
except ImportError:                                  # `ar` (or another intake dep) not installed
    intake = approve = IntakeResult = None
    propose = ProposeResult = resolve = ResolvedSpec = None
    LLM_INTAKE_AVAILABLE = False
from .profiler import profile, ProfileView
from .schema import GoalProposal, Constraints, GOAL_PROPOSAL_JSON_SCHEMA, PROMPT_VERSION
from .ops import llm_propose, LLMRequest, Proposal, ReplayCache, request_digest, PROPOSAL_KINDS

__all__ = [
    "intake", "approve", "IntakeResult",
    "profile", "ProfileView", "propose", "ProposeResult", "resolve", "ResolvedSpec",
    "GoalProposal", "Constraints", "GOAL_PROPOSAL_JSON_SCHEMA", "PROMPT_VERSION",
    "llm_propose", "LLMRequest", "Proposal", "ReplayCache", "request_digest", "PROPOSAL_KINDS",
]
