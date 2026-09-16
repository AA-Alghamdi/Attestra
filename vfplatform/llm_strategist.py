"""#6 -- an LLM STRATEGIST that proposes which representation to peek next, WITHOUT ever being able to promote one.

The audit's sharpest gap: across all seven hand-run phases the *strategist* was me -- I read each FDR table and
decided "swap to the dinov2 family, skip siglip/eva02, then scale to giant." The deterministic policy
(`ReprResearcher._propose`) reproduces that ladder by brute force: it proposes EVERY legal candidate for a rung
and spends one sealed certification on each competent one. A good strategist should reach the same certified
champion in FEWER sealed peeks by proposing a shortlist -- the right family first, skipping the encoders it can
predict will lose. This module is that strategist.

THE SAFETY CONTRACT (why this can never weaken a certificate):
  * The strategist only ever REORDERS or PRUNES the deterministic legal proposal set. `StrategistResearcher`
    recomputes the legal set with the unmodified `ReprResearcher._propose` and INTERSECTS the strategist's
    output with it, so an LLM hallucination can at worst (a) propose an illegal encoder -> dropped, or
    (b) prune everything -> falls back to the full legal set. It can never inject a candidate the policy would
    not allow, and it never touches the gate.
  * The frozen Tier-3 certifier remains the SOLE promoter. The strategist sees only public metadata (encoder
    tags/families/sizes, what's been tried, the champion's sealed lower bound) -- never a sealed label. A wrong
    call wastes or saves a peek; it can never mint a false promotion.

THE FALSIFIABLE CLAIM (measured by scripts/run_llm_strategist.py): run the SAME arena + SAME brain twice, once
with the deterministic proposer and once with this strategist, and compare (champion reached, sealed peeks
spent). "The LLM is a better strategist" is true iff it reaches the SAME certified champion in STRICTLY fewer
sealed peeks. A tie or a regression is an honest negative.
"""
import json
import os
from typing import Dict, List, Optional, Tuple

from .repr_researcher import Encoder, ReprResearcher

Proposal = Tuple[str, Optional[str]]            # (candidate_tag, fuse_partner_or_None)

DEFAULT_MODEL = os.environ.get("ATTESTRA_STRATEGIST_MODEL", "claude-sonnet-4-5")


def _proposal_id(p: Proposal) -> str:
    tag, partner = p
    return f"fuse[{tag}+{partner}]" if partner is not None else tag


class Strategist:
    """Orders/prunes the legal proposals for a rung. Base = the identity (deterministic) strategist."""

    name = "deterministic"

    def rank(self, move_class: str, champion: Encoder, tried: set,
             registry: Dict[str, Encoder], legal: List[Proposal]) -> List[Proposal]:
        return list(legal)


class LLMStrategist(Strategist):
    """Asks Claude for a ranked shortlist of the legal candidates, to minimise wasted sealed certifications.

    Returns a subset/reordering of `legal`; on ANY failure (no key, API error, unparseable reply, nothing legal
    left) it returns `legal` unchanged, so the climb is never blocked and never less safe than the deterministic
    policy. Records a per-call log (`self.calls`) of what it proposed and why, for the run certificate."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None, max_tokens: int = 1200):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = int(max_tokens)
        self.name = f"llm[{model}]"
        self.calls: List[dict] = []

    # -- prompt ------------------------------------------------------------------------------------------
    @staticmethod
    def _rung_doc(move_class: str) -> str:
        return {
            "model": "swap to a DIFFERENT encoder family than the champion (e.g. self-supervised vs "
                     "language-aligned vs supervised-CNN). This is where the big representation wins live.",
            "features": "fuse (concatenate) the champion with another encoder -- an authored featurizer. "
                        "Historically this dilutes on hard tasks and has never certified.",
            "capacity": "scale UP within the champion's OWN (winning) family to a bigger model.",
        }.get(move_class, move_class)

    def _prompt(self, move_class: str, champion: Encoder, tried: set,
                registry: Dict[str, Encoder], legal: List[Proposal]) -> str:
        def enc_row(e: Encoder) -> dict:
            return {"tag": e.tag, "family": e.family, "scale_rank": e.scale_rank,
                    "params_m": e.params_m}
        cands = []
        for tag, partner in legal:
            row = {"id": _proposal_id((tag, partner)), "encoder": enc_row(registry[tag])}
            if partner is not None and partner in registry:
                row["fuse_partner"] = enc_row(registry[partner])
            cands.append(row)
        state = {
            "move_class": move_class,
            "move_class_meaning": self._rung_doc(move_class),
            "champion": enc_row(champion),
            "already_tried": sorted(t for t in tried),
            "legal_candidates": cands,
        }
        return (
            "You are the STRATEGIST of an autonomous representation researcher. A frozen statistical certifier "
            "(paired McNemar + Benjamini-Hochberg FDR + Clopper-Pearson bound on a held-out sealed test) is the "
            "ONLY thing that can promote a candidate; you cannot promote anything. Your single job: order and "
            "PRUNE the legal candidates for this rung so the researcher reaches the true best representation "
            "while spending as FEW sealed certifications as possible (each candidate you keep costs exactly one "
            "sealed peek). Drop candidates you are confident will NOT beat the current champion; keep and rank "
            "first the ones most likely to certify a win.\n\n"
            "Evidence base from prior runs you should use: changing the representation FAMILY is the only lever "
            "that has ever moved the metric; self-supervised encoders (DINOv2) beat language-aligned (CLIP) and "
            "supervised CNNs on fine-grained vision; within the winning family, scaling up pays; fusion has "
            "never certified; bigger-but-different families (SigLIP, EVA-02) tend to lose on fine-grained tasks.\n\n"
            f"STATE:\n{json.dumps(state, indent=2)}\n\n"
            "Reply with ONLY a JSON object (no prose) of the form:\n"
            '{\"shortlist\": [\"<candidate id>\", ...], \"rationale\": \"<one sentence>\"}\n'
            "where shortlist is a subset of the legal candidate ids, ordered best-first. Keep it short; omit "
            "candidates you would not spend a sealed peek on."
        )

    # -- call --------------------------------------------------------------------------------------------
    def rank(self, move_class: str, champion: Encoder, tried: set,
             registry: Dict[str, Encoder], legal: List[Proposal]) -> List[Proposal]:
        if not legal:
            return []
        by_id = {_proposal_id(p): p for p in legal}
        rec = {"move_class": move_class, "champion": champion.tag,
               "legal": list(by_id.keys()), "model": self.model}
        try:
            import anthropic
            if not self.api_key:
                raise RuntimeError("no ANTHROPIC_API_KEY")
            client = anthropic.Anthropic(api_key=self.api_key)
            msg = client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "user",
                           "content": self._prompt(move_class, champion, tried, registry, legal)}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
            if text.startswith("```"):
                text = text.strip("`").split("\n", 1)[-1]
            obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
            raw = list(obj.get("shortlist", []))
            shortlist = [s for s in raw if s in by_id]
            rec.update({"shortlist": shortlist, "rationale": obj.get("rationale", ""), "ok": True})
            if raw and not shortlist:            # named only illegal ids -> malfunction, degrade to safe set
                rec["fallback"] = "all_ids_illegal -> full legal set"
                self.calls.append(rec)
                return list(legal)
            # raw empty == a DELIBERATE "skip this rung" (the strategist's most valuable move: don't spend a
            # sealed peek where it predicts no win). The frozen certifier still gates every promotion, and the
            # deterministic baseline run is the ground truth that catches a wrongly-skipped winner.
            self.calls.append(rec)
            return [by_id[s] for s in shortlist]
        except Exception as e:                   # noqa: BLE001 -- any failure must degrade to deterministic
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}", "fallback": "full legal set"})
            self.calls.append(rec)
            return list(legal)


class StrategistResearcher(ReprResearcher):
    """ReprResearcher whose PROPOSAL ordering is supplied by a Strategist. The gate, the certifier, the climb
    and escalation logic are inherited UNCHANGED -- only `_propose` is intercepted, and its result is always
    intersected with the unmodified parent's legal proposals, so the strategist can only reorder/prune."""

    def __init__(self, *args, strategist: Optional[Strategist] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategist = strategist or Strategist()

    def _propose(self, move_class: str, champion: Encoder, tried: set) -> List[Proposal]:
        legal = super()._propose(move_class, champion, tried)
        if not legal:
            return legal
        legal_set = {_proposal_id(p) for p in legal}
        chosen = self.strategist.rank(move_class, champion, set(tried), self.registry, list(legal))
        if not chosen:
            return []                            # a DELIBERATE empty prune -> skip this rung (escalate)
        filtered = [p for p in chosen if _proposal_id(p) in legal_set]   # airtight: drop any illegal candidate
        return filtered or list(legal)           # non-empty intent but all illegal -> safe deterministic set


__all__ = ["Strategist", "LLMStrategist", "StrategistResearcher", "Proposal", "DEFAULT_MODEL"]
