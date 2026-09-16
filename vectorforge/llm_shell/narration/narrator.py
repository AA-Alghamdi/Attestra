"""The A6 narrator, now a thin adapter over the A9 `llm_propose` primitive (text mode).

The frozen evidence bundle is TRUSTED context (system-produced, immutable). A6 supplies the semantic
bound (the groundedness gate) and the deterministic template fallback; ops.py owns the call mechanics,
caching, replay record, and graceful degradation. The narration is READ_ONLY over immutable evidence.
"""
from dataclasses import dataclass
from typing import Optional

from .. import ops
from .evidence import EvidenceBundle
from .groundedness import check_groundedness, GroundednessVerdict

DEFAULT_MODEL = ops.DEFAULT_MODEL
SURFACE = "a6.narration"
PROMPT_VERSION = "a6-narration/v1"

AUDIENCES = {
    "simple": "a non-technical user. Plain language, one short paragraph, no jargon. State plainly "
              "whether the result can be trusted and why, in everyday terms.",
    "developer": "a software developer integrating this. Be precise and concrete; reference the "
                 "metric, the lower bound, the threshold, and the held-out size; 2-4 sentences.",
    "researcher": "an ML researcher. Be exact about the statistical claim (lower-bound certification "
                  "vs point estimate), the threshold, n, and the auditor verdict; terse and rigorous.",
}

_SYSTEM = (
    "You are the results narrator for VectorForge. You are given a FROZEN evidence bundle produced by a "
    "deterministic certifier. Write a faithful narration for the specified audience. ABSOLUTE rules: "
    "(1) Mention ONLY numbers that appear in the bundle; never compute, round to new values, or invent a "
    "number. (2) Say 'certified'/'verified'/'guaranteed' ONLY if the bundle's `certified` is true. "
    "(3) Claim the data is leakage-clean ONLY if `audit_passed` is true. (4) If not certified, say so "
    "plainly and do not imply success. A downstream deterministic gate will REJECT your text if it "
    "contains any ungrounded number or unsupported claim, so stay strictly within the bundle.")


@dataclass
class NarrationResult:
    text: str
    used_llm: bool
    grounded: bool
    verdict: GroundednessVerdict
    audience: str
    error: Optional[str] = None
    usage: Optional[dict] = None

    def as_dict(self):
        return {"text": self.text, "used_llm": self.used_llm, "grounded": self.grounded,
                "audience": self.audience, "violations": self.verdict.violations, "error": self.error}


def narrate(bundle: EvidenceBundle, *, audience: str = "simple", model: str = DEFAULT_MODEL,
            api_key: Optional[str] = None, use_llm: bool = True, cache_path: Optional[str] = None,
            tenant_id: str = "default", max_tokens: int = 40000, timeout: float = 60.0) -> NarrationResult:
    """Produce a grounded narration via the A9 primitive. Never raises; the delivered text is ALWAYS
    grounded (LLM text that passes the gate, or the deterministic template)."""
    if audience not in AUDIENCES:
        audience = "simple"
    fallback_text = bundle.deterministic_narrative()

    req = ops.LLMRequest(
        surface=SURFACE, system=_SYSTEM,
        trusted_context={"audience_instruction": AUDIENCES[audience], "evidence": bundle.facts()},
        untrusted_inputs={}, model=model, schema=None, prompt_version=PROMPT_VERSION,
        max_tokens=max_tokens)

    def _verify(text):
        if not text:
            return False, None, "empty narration"
        v = check_groundedness(text, bundle)
        return v.ok, text, ("; ".join(v.violations) if not v.ok else None)

    def _fallback(reason):
        return fallback_text

    p = ops.llm_propose(req, kind="READ_ONLY", verify=_verify, fallback=_fallback,
                        tenant_id=tenant_id, api_key=api_key, use_llm=use_llm,
                        cache_path=cache_path, timeout=timeout)
    text = p.value if isinstance(p.value, str) and p.value else fallback_text
    verdict = check_groundedness(text, bundle)
    return NarrationResult(text=text, used_llm=p.used_llm, grounded=verdict.ok, verdict=verdict,
                           audience=audience, error=p.error, usage=p.usage)
