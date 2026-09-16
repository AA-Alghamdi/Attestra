"""The A1 goal-intake pipeline: NL goal + dataset -> a frozen, content-addressed contract.

Orchestrates the six steps. The LLM is step 2 ONLY; every other step is frozen code.
  1. profile        (frozen)  structural pass, no goal, no rows leave
  2. propose        (LLM)     sandboxed, schema-forced, cached, deterministic fallback
  3. resolve        (frozen)  narrow-or-reject against real columns + ontology
  4. threshold      (frozen)  left to ar.spec.propose_spec downstream; the LLM never writes it
  5. gates          (frozen)  science.audit (downstream) backstops forbidden fields independently
  6. freeze         (frozen)  content-address the resolved contract via science.digest

`intake(...)` returns an IntakeResult. It NEVER promotes anything and never sets a threshold; it
produces a reviewable, hashable contract that the product feeds into the existing compile/build/
certify path. Removing the LLM (use_llm=False) returns the exact structural behavior -- the product
stays runnable with no agent in the loop.
"""
from dataclasses import dataclass, field
from typing import Optional

from . import _frozen
from .profiler import profile, ProfileView
from .proposer import propose, ProposeResult, DEFAULT_MODEL
from .resolver import resolve, ResolvedSpec


@dataclass
class IntakeResult:
    goal: str
    view: ProfileView
    propose_result: ProposeResult
    resolved: ResolvedSpec
    contract: dict                 # the frozen-candidate contract (pre-approval)
    contract_digest: str           # content address of `contract`
    approved: bool = False
    approved_by: Optional[str] = None
    frozen_digest: Optional[str] = None   # set on approve(); equals contract_digest unless edited

    def summary(self):
        r = self.resolved
        return {
            "goal": self.goal,
            "used_llm": self.propose_result.used_llm,
            "proposal_source": r.proposal_source,
            "target": r.target, "task_type": r.task_type, "metric": r.metric,
            "drop_cols": r.drop_cols,
            "constraints": r.constraints,
            "unenforceable_constraints": r.unenforceable_constraints,
            "needs_human": r.needs_human,
            "human_review": r.human_review,
            "contract_digest": self.contract_digest,
            "approved": self.approved,
            "supported": r.supported,
            "error": self.propose_result.error,
        }


def _build_contract(goal: str, resolved: ResolvedSpec) -> dict:
    """The reviewable contract. Deterministic function of the RESOLVED spec (post-frozen-narrowing),
    NOT of the raw LLM output -- so the same resolved spec always hashes identically (MI-5)."""
    return {
        "goal": goal,
        "target": resolved.target,
        "task_type": resolved.task_type,
        "metric": resolved.metric,
        "drop_cols": list(resolved.drop_cols),
        "constraints": resolved.constraints,
        "unenforceable_constraints": list(resolved.unenforceable_constraints),
        "needs_human": resolved.needs_human,
        "supported": resolved.supported,
        "threshold": "DEFERRED_TO_FROZEN_propose_spec",   # explicit: the LLM never sets this
    }


def intake(file_or_records, goal: str, *, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
           use_llm: bool = True, cache_path: Optional[str] = None, seed: int = 0,
           min_test_n: int = 200, max_rows: Optional[int] = None) -> IntakeResult:
    # 1. profile (frozen, goal-free)
    view = profile(file_or_records, seed=seed, min_test_n=min_test_n, max_rows=max_rows)

    # 2. propose (LLM, sandboxed, cached, fallback)
    pr = propose(goal, view, model=model, api_key=api_key, use_llm=use_llm, cache_path=cache_path)

    # 3. resolve (frozen narrow-or-reject)
    resolved = resolve(pr.proposal, view)

    # 4/5 handled downstream by frozen code (propose_spec threshold; science.audit backstop).
    # 6. freeze candidate: content-address the resolved contract
    contract = _build_contract(goal, resolved)
    contract_digest = _frozen.digest(contract)

    return IntakeResult(goal=goal, view=view, propose_result=pr, resolved=resolved,
                        contract=contract, contract_digest=contract_digest)


def approve(result: IntakeResult, *, approved_by: str, edits: Optional[dict] = None) -> IntakeResult:
    """Human approval + freeze. Optional `edits` apply human corrections to the contract BEFORE the
    freeze hash is taken (so an edited contract gets a NEW, different digest -- drift is observable).
    Edits may touch only review-surface fields; they can never introduce a threshold (still DEFERRED)."""
    if edits:
        forbidden = {"threshold"}
        bad = forbidden & set(edits)
        if bad:
            raise ValueError(f"edits may not set {sorted(bad)} -- the threshold is frozen-derived only")
        result.contract = dict(result.contract)
        result.contract.update({k: v for k, v in edits.items() if k != "threshold"})
        result.contract_digest = _frozen.digest(result.contract)
    result.approved = True
    result.approved_by = approved_by
    result.frozen_digest = result.contract_digest
    return result
