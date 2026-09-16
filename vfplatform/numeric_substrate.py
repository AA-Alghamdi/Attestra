"""THE NUMERICAL SUBSTRATE -- the typed LLM<->numbers firewall (a single source of numerical truth).

WHY THIS EXISTS
---------------
An LLM is a LANGUAGE model: it pattern-matches token sequences and is unreliable at the exact arithmetic,
statistics, and bookkeeping that ML rigor depends on -- effect sizes, confidence bounds, multiplicity
correction, power, calibration. Across this system the LLM is therefore confined to the one thing it is good
at -- proposing STRUCTURE (which backbone, which motif, which authored code, which axis to escalate) -- and
is structurally forbidden from producing any NUMBER that affects a decision.

This module makes that firewall explicit and FALSIFIABLE:

  * Every decision-bearing number (a lift, a p-value, a confidence bound, a threshold, a required sample
    size, a calibration error) is produced HERE, by a deterministic substrate that DELEGATES to the frozen
    science / battery / power core. There is exactly one place numbers come from, and it is audited.

  * A number an LLM (or any untrusted source) emits is treated as an untrusted HINT: a `NumericClaim` with
    `source="llm"`. It is INADMISSIBLE as a decision input until `recompute()` re-derives it from raw data
    and finds it `verified` (agrees within tolerance). If the recomputation disagrees, the substrate's value
    wins and the discrepancy is recorded as `contradicted`. A `contradicted`/`unverifiable` LLM number can
    NEVER reach a decision: `decide()` refuses it.

  * `audit()` returns the full ledger, so a certificate can carry proof that no LLM-emitted number leaked
    into a decision.

WHAT THIS IS NOT
----------------
This is NOT a new statistical core. It owns no formulas: every operation forwards to the FROZEN primitives
(vectorforge.science -- Clopper-Pearson, calibration, metrics; vfplatform.battery -- McNemar, BH-FDR;
vfplatform.power -- power / required-n). The frozen Tier-3 certifier remains the SOLE promoter; this module
is the disciplined accountant that sits BETWEEN the LLM's structural proposals and those frozen primitives,
guaranteeing the LLM never hands a number to the gate. Lock tests assert byte-identical agreement with the
frozen primitives, so the substrate can never silently diverge into a second, weaker source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from vectorforge import science as S
from . import battery as B
from . import power as P

# the trusted origins of a number. "computed"/"frozen" come from the substrate itself; "llm"/"user" are
# untrusted hints that must be recomputed before they can decide anything.
TRUSTED_SOURCES = ("computed", "frozen")
UNTRUSTED_SOURCES = ("llm", "user")

# verdict of an untrusted claim after recompute().
VERIFIED = "verified"            # recomputed value agrees within tolerance -> admissible
CONTRADICTED = "contradicted"    # recomputed value disagrees -> substrate value wins, claim refused
UNVERIFIABLE = "unverifiable"    # no recompute was possible (no data / no recompute fn) -> refused
PENDING = "pending"              # untrusted, not yet recomputed -> refused


class NumberLeak(Exception):
    """Raised when an untrusted (LLM/user) number is asked to drive a decision without a verified recompute.
    This is the firewall tripping: the one error that proves an LLM number tried to reach the gate."""


@dataclass
class NumericClaim:
    """A single quantitative assertion flowing through the system, with its provenance and (if untrusted) the
    substrate's recomputation of it. The type makes 'where did this number come from' un-skippable."""
    name: str
    value: float
    source: str = "computed"                 # one of TRUSTED_SOURCES | UNTRUSTED_SOURCES
    tol: float = 1e-9                         # absolute tolerance for a recompute to count as agreement
    recomputed: Optional[float] = None        # the substrate's own value (filled by recompute())
    verdict: Optional[str] = None             # VERIFIED | CONTRADICTED | UNVERIFIABLE | PENDING | None
    note: str = ""

    def __post_init__(self):
        if self.source not in TRUSTED_SOURCES + UNTRUSTED_SOURCES:
            raise ValueError(f"unknown source {self.source!r}; expected one of "
                             f"{TRUSTED_SOURCES + UNTRUSTED_SOURCES}")
        self.value = float(self.value)
        if self.source in UNTRUSTED_SOURCES and self.verdict is None:
            self.verdict = PENDING

    @property
    def trusted(self) -> bool:
        """True iff this number may drive a decision: either it was produced by the substrate, or it was an
        untrusted hint that the substrate RECOMPUTED and found to agree."""
        if self.source in TRUSTED_SOURCES:
            return True
        return self.verdict == VERIFIED

    @property
    def decided_value(self) -> float:
        """The value a decision must use: ALWAYS the substrate's own recomputation when present (never the
        untrusted hint), else the trusted computed value."""
        return float(self.recomputed) if self.recomputed is not None else float(self.value)

    def to_dict(self) -> dict:
        return {"name": self.name, "value": round(self.value, 6), "source": self.source,
                "recomputed": (round(self.recomputed, 6) if self.recomputed is not None else None),
                "verdict": self.verdict, "trusted": self.trusted, "note": self.note}


class NumericSubstrate:
    """The single source of numerical truth + the recompute/audit firewall.

    Two responsibilities:
      1. PRODUCE numbers. Every method below returns a trusted `NumericClaim` (source="frozen") computed by a
         frozen primitive. Use these instead of letting any number be asserted by prose / an LLM.
      2. POLICE numbers. `claim_llm()` wraps an LLM-emitted number; `recompute()` re-derives it from raw data
         and sets a verdict; `decide()` refuses to return any number that is not trusted. `audit()` exposes
         the full ledger.

    Every claim it touches (produced or policed) is appended to `self.ledger`, so a run certificate can carry
    a complete, machine-checkable record that no untrusted number reached a decision."""

    def __init__(self):
        self.ledger: List[NumericClaim] = []

    # -- record-keeping ---------------------------------------------------------------------------------
    def _record(self, claim: NumericClaim) -> NumericClaim:
        self.ledger.append(claim)
        return claim

    # =============================================================== PRODUCE (frozen-backed, trusted)
    def accuracy_lower_bound(self, correct: Sequence[int], alpha: float = 0.05, *,
                             name: str = "accuracy_lower_bound") -> NumericClaim:
        """Clopper-Pearson lower confidence bound on accuracy from a 0/1 correctness vector (frozen
        vectorforge.science.clopper_pearson_lower). The bound the certifier uses to clear theta."""
        k = int(sum(int(c) for c in correct))
        n = int(len(correct))
        val = S.clopper_pearson_lower(k, n, float(alpha))
        return self._record(NumericClaim(name, val, source="frozen",
                                         note=f"clopper_pearson_lower(k={k}, n={n}, alpha={alpha})"))

    def paired_lift(self, cand_correct: Sequence[int], base_correct: Sequence[int], *,
                    name: str = "paired_lift") -> NumericClaim:
        """Mean paired accuracy lift (candidate - baseline) on the SAME rows. The effect size an LLM is most
        tempted to assert and most likely to get wrong; here it is computed from the row vectors."""
        if len(cand_correct) != len(base_correct) or not cand_correct:
            return self._record(NumericClaim(name, 0.0, source="computed", note="empty/mismatched vectors"))
        n = len(cand_correct)
        lift = sum(int(a) - int(b) for a, b in zip(cand_correct, base_correct)) / n
        return self._record(NumericClaim(name, float(lift), source="computed",
                                         note=f"mean(cand-base) over n={n} paired rows"))

    def paired_pvalue(self, cand_correct: Sequence[int], base_correct: Sequence[int], *,
                      name: str = "paired_pvalue") -> NumericClaim:
        """One-sided exact McNemar p-value, P(candidate does NOT beat baseline) (frozen
        vfplatform.battery.mcnemar_pvalue)."""
        val = B.mcnemar_pvalue(list(cand_correct), list(base_correct))
        return self._record(NumericClaim(name, float(val), source="frozen", note="mcnemar one-sided"))

    def regression_pvalue(self, cand_pred: Sequence[float], base_pred: Sequence[float],
                          y_true: Sequence[float], *, error: str = "abs",
                          name: str = "regression_pvalue") -> NumericClaim:
        """One-sided paired significance that the candidate's per-row error is lower (frozen
        vfplatform.battery.regression_paired_pvalue)."""
        val = B.regression_paired_pvalue(list(cand_pred), list(base_pred), list(y_true), error=error)
        return self._record(NumericClaim(name, float(val), source="frozen", note=f"regression paired ({error})"))

    def fdr_survivors(self, pvalues: Sequence[float], alpha: float = 0.1) -> List[int]:
        """Indices that survive Benjamini-Hochberg at FDR<=alpha (frozen vfplatform.battery.benjamini_hochberg).
        Returns the index set directly (this is a set decision, not a single scalar claim)."""
        return sorted(B.benjamini_hochberg(list(pvalues), alpha=float(alpha)))

    def required_n(self, theta: float, p_assumed: float, *, alpha: float = 0.05, checks: int = 1,
                   target_power: float = 0.8, name: str = "required_n") -> NumericClaim:
        """Smallest sealed-n at which projected power to certify >= target_power, assuming true accuracy
        p_assumed (frozen vfplatform.power.min_n_for_power). value=-1 when p_assumed<=theta (uncertifiable at
        any n) -- the honest 'this cannot be powered' signal."""
        n = P.min_n_for_power(float(theta), float(p_assumed), alpha=alpha, checks=checks,
                              target_power=target_power)
        val = float(n) if n is not None else -1.0
        return self._record(NumericClaim(name, val, source="frozen",
                                         note=f"min_n_for_power(theta={theta}, p={p_assumed}, "
                                              f"power>={target_power})"))

    def power_at_n(self, n: int, theta: float, p_assumed: float, *, alpha: float = 0.05, checks: int = 1,
                   name: str = "power_at_n") -> NumericClaim:
        """Probability the frozen certifier certifies at sealed size n if true accuracy==p_assumed (frozen
        vfplatform.power.power_at_n)."""
        val = P.power_at_n(int(n), float(theta), float(p_assumed), alpha=alpha, checks=checks)
        return self._record(NumericClaim(name, float(val), source="frozen",
                                         note=f"power_at_n(n={n}, theta={theta}, p={p_assumed})"))

    def calibration_error(self, confidences: Sequence[float], correct: Sequence[int], *, bins: int = 10,
                          name: str = "calibration_error") -> NumericClaim:
        """Expected Calibration Error (frozen vectorforge.science.expected_calibration_error)."""
        val = S.expected_calibration_error(list(confidences), list(correct), bins=bins)
        return self._record(NumericClaim(name, float(val), source="frozen", note=f"ECE({bins} bins)"))

    def metric(self, metric_name: str, y_true, y_pred, labels=None, *, name: Optional[str] = None) -> NumericClaim:
        """A certifiable metric value via the FROZEN scorer (classification -> science.score_metric;
        regression -> science.score_regression_metric). Single source for 'what is the score'."""
        if metric_name in ("r2", "neg_rmse", "neg_mae"):
            val = S.score_regression_metric(metric_name, list(y_true), list(y_pred))
        else:
            if labels is None:
                labels = sorted(set(int(v) for v in y_true))
            val = S.score_metric(metric_name, list(y_true), list(y_pred), list(labels))
        return self._record(NumericClaim(name or f"metric[{metric_name}]", float(val), source="frozen",
                                         note=f"frozen score_metric({metric_name})"))

    # =============================================================== POLICE (untrusted hints)
    def claim_llm(self, name: str, value: float, *, tol: float = 1e-6, note: str = "") -> NumericClaim:
        """Wrap a number an LLM (or any untrusted source) emitted. It enters the ledger as PENDING and is
        INADMISSIBLE as a decision input until recompute() verifies it. Use `tol` to set the agreement band
        (looser for noisy quantities like a sampled lift; tight for an exact statistic)."""
        return self._record(NumericClaim(name, float(value), source="llm", tol=float(tol),
                                         note=note or "LLM-emitted hint (untrusted until recomputed)"))

    def recompute(self, claim: NumericClaim, recompute_fn: Optional[Callable[[], float]]) -> NumericClaim:
        """Re-derive an untrusted claim from raw data via `recompute_fn` (which MUST itself use the substrate's
        frozen-backed producers) and set the verdict:
          VERIFIED      iff |recomputed - hint| <= tol,
          CONTRADICTED  iff it disagrees,
          UNVERIFIABLE  iff no recompute_fn was given or it raised.
        The substrate's recomputed value is always retained as the value any decision must use -- the hint is
        never trusted in its place. Trusted claims pass through unchanged (nothing to verify)."""
        if claim.source in TRUSTED_SOURCES:
            return claim
        if recompute_fn is None:
            claim.verdict = UNVERIFIABLE
            claim.note = (claim.note + " | no recompute_fn -> unverifiable").strip(" |")
            return claim
        try:
            rv = float(recompute_fn())
        except Exception as ex:  # noqa: BLE001 -- any failure to recompute is honestly 'unverifiable'
            claim.verdict = UNVERIFIABLE
            claim.note = (claim.note + f" | recompute raised {type(ex).__name__}").strip(" |")
            return claim
        claim.recomputed = rv
        claim.verdict = VERIFIED if abs(rv - claim.value) <= claim.tol else CONTRADICTED
        return claim

    def decide(self, claim: NumericClaim) -> float:
        """Return the number a decision is allowed to use -- or trip the firewall. A trusted claim returns its
        value; a verified untrusted hint returns the SUBSTRATE'S recomputed value (never the hint); anything
        else raises NumberLeak. This is the chokepoint that makes 'no LLM number decides' enforceable, not
        merely documented."""
        if not claim.trusted:
            raise NumberLeak(
                f"refused to use untrusted number {claim.name!r} (source={claim.source}, "
                f"verdict={claim.verdict}): an LLM/user number cannot drive a decision unless the substrate "
                f"recomputed it and it agreed. hint={claim.value}, recomputed={claim.recomputed}")
        return claim.decided_value

    # =============================================================== AUDIT
    def audit(self) -> dict:
        """A machine-checkable record of every number the substrate touched. `clean` is the headline
        guarantee: True iff NO untrusted number was left in a state where it could decide (every llm/user
        claim is either verified-by-recompute or has been refused). Embed this in a certificate."""
        untrusted = [c for c in self.ledger if c.source in UNTRUSTED_SOURCES]
        contradicted = [c for c in untrusted if c.verdict == CONTRADICTED]
        unverifiable = [c for c in untrusted if c.verdict in (UNVERIFIABLE, PENDING)]
        return {
            "n_claims": len(self.ledger),
            "n_untrusted": len(untrusted),
            "n_contradicted": len(contradicted),
            "n_unverifiable": len(unverifiable),
            "n_verified": len([c for c in untrusted if c.verdict == VERIFIED]),
            # the firewall held iff no untrusted number is currently admissible-by-default: contradicted and
            # unverifiable hints are recorded but refused by decide(); none was silently trusted.
            "clean": all(c.verdict == VERIFIED for c in untrusted),
            "claims": [c.to_dict() for c in self.ledger],
        }


# --------------------------------------------------------------------------- recipe-number guard
# the numeric genes a recipe is ALLOWED to carry, with their legal bounds (mirrors recipe_space in
# vfplatform/recipe.py). The LLM/generator proposes STRUCTURE; these are the only free-floating numbers a
# proposal may set, and they are clamped to the audited space -- never a threshold, alpha, lift, or p-value.
_NUMERIC_GENE_BOUNDS = {"lr": (1e-5, 1e-1), "weight_decay": (1e-6, 1e-2), "epochs": (3, 40)}


def guard_recipe_numbers(recipe):
    """Assert a proposed recipe carries ONLY typed, space-clamped numeric genes (lr / weight_decay / epochs)
    and clamp any out-of-range value back into the legal interval. A recipe genome can never carry a
    decision number (a theta, an alpha, a claimed lift); those live solely in the frozen certifier and the
    substrate. Returns (clamped_recipe, info) where info={clamped: {gene: (old,new)}, ok: bool}; the recipe
    is returned UNCHANGED when already in range (`Recipe` is frozen, so an out-of-range one is replaced).

    This is the structural half of the firewall: even the numbers a generator IS allowed to set are bounded
    by the audited search space, so an LLM-authored recipe cannot smuggle an extreme hyperparameter past the
    space, let alone a decision threshold."""
    clamped, fixes = {}, {}
    for gene, (lo, hi) in _NUMERIC_GENE_BOUNDS.items():
        if not hasattr(recipe, gene):
            continue
        cur = getattr(recipe, gene)
        new = min(max(cur, lo), hi)
        if gene == "epochs":
            new = int(round(new))
        if new != cur:
            clamped[gene] = (cur, new)
            fixes[gene] = new
    out = recipe.with_(**fixes) if fixes else recipe
    return out, {"clamped": clamped, "ok": True}


__all__ = [
    "NumericClaim", "NumericSubstrate", "NumberLeak", "guard_recipe_numbers",
    "TRUSTED_SOURCES", "UNTRUSTED_SOURCES", "VERIFIED", "CONTRADICTED", "UNVERIFIABLE", "PENDING",
]
