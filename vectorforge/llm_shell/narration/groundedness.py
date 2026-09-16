"""The groundedness gate (type-b): a deterministic check that a narrative is entailed by the frozen
evidence. It rejects (1) any decimal/percentage number not present in the bundle's allowed set, and
(2) any success/certification claim the bundle does not support. On any violation the caller must use
the deterministic fallback narrative -- the LLM is never allowed to overclaim.

Scope (documented): the number check covers DECIMALS (e.g. 0.873) and PERCENTAGES (e.g. 87.3%) -- the
form every metric/bound/threshold takes. Bare integers in prose ("the 2 classes", "3 steps") are not
gated as claims; the bundle's load-bearing quantities are all decimals/percentages and ARE gated.
"""
import re
from dataclasses import dataclass, field

# decimals like 0.87 / 12.5, and percentages like 87% / 87.3%
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DEC = re.compile(r"(?<![\d.])(\d+\.\d+)(?![\d.])")

# success / certification language that may appear ONLY if the bundle certifies. Checked with
# negation-awareness so an honest "not certified" / "does not clear the threshold" is NOT flagged.
_SUCCESS_TERMS = ("certified", "verified", "guarantee", "guaranteed", "proven", "proves",
                  "passes the bar", "clears the threshold", "meets the threshold", "succeeded")
# negation tokens that, present ANYWHERE in the same sentence as a success term, make it a non-claim.
# (Sentence-scoped, not 3-word-scoped: honest text like "should not rely on this as a certified result"
#  places the negation far from the term.) "yet"/"nor"/"neither" are excluded here to avoid over-skipping.
_NEG_TOKENS = {"not", "no", "never", "without", "cannot", "can't", "isn't", "wasn't", "aren't",
               "won't", "fails", "fail", "n't", "couldn't", "doesn't", "didn't", "below", "fell"}
# benign collocations where a success WORD is terminology, not a verdict on the result. Stripped before
# the claim scan so "certified lower bound" / "certification criteria" do not read as success claims.
_BENIGN = ("certified lower bound", "certified upper bound", "certified bound",
           "certification criteria", "certification threshold", "certification bar",
           "certification standard", "certification process", "lower-bound certification",
           "lower bound certification")
_SENT_SPLIT = re.compile(r"[.!?;:\n]+")
# leakage-clean language that may appear ONLY if the auditor passed.
_CLEAN_TERMS = ("no leakage", "leakage passed", "no contamination", "audit passed",
                "clean of leakage", "passed the leakage")
_TOL = 0.005   # absolute tolerance on the proportion scale


@dataclass
class GroundednessVerdict:
    ok: bool
    violations: list = field(default_factory=list)

    def as_dict(self):
        return {"ok": self.ok, "violations": list(self.violations)}


def numbers_in(text):
    """Extract the gated numeric values from text on the proportion scale: every decimal as-is, and
    every percentage divided by 100. Shared by A4's checker. Returns a list of (literal, value)."""
    out = []
    for m in _PCT.finditer(text or ""):
        out.append((m.group(0), float(m.group(1)) / 100.0))
    stripped = _PCT.sub(" ", text or "")
    for m in _DEC.finditer(stripped):
        out.append((m.group(1), float(m.group(1))))
    return out


def _allowed_renderings(allowed_numbers):
    """Build the set of proportion-scale values a narrative number may match: each allowed value as-is
    plus its percentage equivalent (so 0.87 grounds both '0.87' and '87%')."""
    vals = []
    for a in allowed_numbers:
        vals.append(float(a))
    return vals


def _matches(value_proportion, allowed_props):
    return any(abs(value_proportion - a) <= _TOL for a in allowed_props)


def _sentence_negated(sentence_low):
    words = set(re.findall(r"[a-z']+", sentence_low))
    return bool(words & _NEG_TOKENS)


def _affirmative_success(terms, low_text):
    """Return the first SUCCESS term used as an affirmative verdict on the result, else None.
    A term is a non-claim if its sentence contains a negation, or it is part of a benign collocation
    (terminology like 'certified lower bound'). Sentence-scoped to match real prose; the residual
    blind spot (an in-sentence double negative) is accepted and documented."""
    for sent in _SENT_SPLIT.split(low_text):
        s = sent.strip()
        if not s or _sentence_negated(s):
            continue
        for b in _BENIGN:               # strip terminology so it does not read as a claim
            s = s.replace(b, " ")
        for term in terms:
            if term in s:
                return term
    return None


def check_groundedness(narrative: str, bundle) -> GroundednessVerdict:
    """Return ok=True iff every gated number is grounded AND no unsupported claim is made."""
    violations = []
    text = narrative or ""
    low = text.lower()
    allowed = _allowed_renderings(bundle.allowed_numbers())
    n_test = float(bundle.n_test)

    # ---- percentages: compare on the proportion scale (87% -> 0.87) -----------------------------
    for m in _PCT.finditer(text):
        raw = float(m.group(1))
        prop = raw / 100.0
        # a percentage may also legitimately equal an allowed count rendered oddly; accept either the
        # proportion match OR an exact match to an allowed raw value.
        if not (_matches(prop, allowed) or _matches(raw, allowed)):
            violations.append(f"ungrounded percentage: {m.group(0)!r}")

    # ---- decimals: must match an allowed value (proportion scale) -------------------------------
    # Strip percentage substrings first so the digits inside "87.3%" aren't double-counted as a decimal.
    decimal_scan = _PCT.sub(" ", text)
    for m in _DEC.finditer(decimal_scan):
        val = float(m.group(1))
        if not _matches(val, allowed):
            violations.append(f"ungrounded decimal: {m.group(1)!r}")

    # ---- claim checks (negation-aware for success terms) ----------------------------------------
    if not bundle.certified:
        hit = _affirmative_success(_SUCCESS_TERMS, low)
        if hit:
            violations.append(f"unsupported success claim {hit!r} (bundle is NOT certified)")
    if not bundle.audit_passed:
        for term in _CLEAN_TERMS:
            if term in low:
                violations.append(f"unsupported leakage-clean claim {term!r} (audit did NOT pass)")
                break

    return GroundednessVerdict(ok=(len(violations) == 0), violations=violations)
