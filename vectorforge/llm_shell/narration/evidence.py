"""The FROZEN evidence bundle the narrator reads. Built from the deterministic core's outputs
(certificate, audit verdict, spec) -- never from anything the LLM produced. Immutable once built.

It exposes:
  * `facts`            : the canonical dict shown to the LLM (and the only thing it may describe),
  * `allowed_numbers()`: every numeric value the narrative is permitted to mention,
  * `certified` / `audit_passed`: the boolean claims the gate enforces.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvidenceBundle:
    metric: str
    observed: float                 # the point estimate measured on the locked test
    lower_bound: float              # the certifier's Clopper-Pearson / bootstrap LOWER bound
    threshold: float                # the bar (theta) -- from frozen propose_spec / human
    n_test: int                     # locked-test size
    certified: bool                 # the certifier's verdict (lower_bound > threshold etc.)
    audit_passed: bool              # the leakage auditor's verdict
    verdict_label: str              # e.g. "VERIFIED RUNNING" / "measured-not-certified" / "BLOCKED"
    task_type: str = ""
    target: str = ""
    drop_cols: list = field(default_factory=list)
    extra_numbers: list = field(default_factory=list)   # any other bundle numbers (alpha, checks, ...)

    def facts(self) -> dict:
        """The canonical, frozen fact set the LLM is given. The LLM may describe ONLY these."""
        return {
            "target": self.target,
            "task_type": self.task_type,
            "metric": self.metric,
            "observed": self.observed,
            "lower_bound": self.lower_bound,
            "threshold": self.threshold,
            "n_test": self.n_test,
            "certified": self.certified,
            "audit_passed": self.audit_passed,
            "verdict_label": self.verdict_label,
            "excluded_features": list(self.drop_cols),
        }

    def allowed_numbers(self):
        """Every numeric value the narrative may mention (proportion scale). The gate matches both
        decimal and percentage renderings of these."""
        nums = [self.observed, self.lower_bound, self.threshold, float(self.n_test)]
        nums.extend(float(x) for x in self.extra_numbers if isinstance(x, (int, float)))
        return [n for n in nums if isinstance(n, (int, float))]

    def deterministic_narrative(self) -> str:
        """The fallback narration: a pure template over the frozen facts. Always grounded by
        construction. Used when the LLM narrative fails the groundedness gate or no LLM is available."""
        if self.verdict_label and self.certified:
            head = (f"Certified. On {self.n_test} held-out examples the model reached "
                    f"{self.metric}={self.observed:.3f}; the certifier's lower bound "
                    f"{self.lower_bound:.3f} clears the threshold {self.threshold:.3f}.")
        elif not self.audit_passed:
            head = ("Blocked before certification: the leakage auditor flagged the data, so no "
                    "certificate was issued.")
        else:
            head = (f"Not certified. Measured {self.metric}={self.observed:.3f} on {self.n_test} "
                    f"held-out examples, but the lower bound {self.lower_bound:.3f} does not clear "
                    f"the threshold {self.threshold:.3f}.")
        tail = (f" Excluded features: {', '.join(self.drop_cols)}." if self.drop_cols else "")
        return head + tail
