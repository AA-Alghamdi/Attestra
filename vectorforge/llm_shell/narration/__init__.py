"""A6 -- conversational, audience-adaptive narration over a FROZEN evidence bundle.

The evidence is frozen by the deterministic core BEFORE the LLM sees it. The LLM writes prose; a
deterministic groundedness gate (type-b) rejects any narrated number or verdict not entailed by the
bundle, falling back to a deterministic template. The LLM output is READ-ONLY over immutable evidence
(shape (c)) and can never overclaim a certificate.
"""
from .evidence import EvidenceBundle
from .groundedness import check_groundedness, GroundednessVerdict
from .narrator import narrate, NarrationResult, AUDIENCES

__all__ = ["EvidenceBundle", "check_groundedness", "GroundednessVerdict",
           "narrate", "NarrationResult", "AUDIENCES"]
