"""CROSS-DATASET REPLICATION + SCOPE -- a dataset-specific win is not a general one.

WHY THIS EXISTS
---------------
A certificate on one dataset D answers 'does this work on D?'. The frontier question is 'does this
GENERALIZE?'. So a winning recipe is re-certified on a REPLICATION SET (other hygiene-certified datasets of
the same task type from datapool.py) and the result is labeled with an honest SCOPE (D7):
  * in_distribution -- certified on the primary dataset only (no replication attempted/available).
  * shift_robust    -- certified on the primary AND replicates on >= a required fraction of the set.
  * scoped          -- certified on the primary but FAILS to replicate -> the win is dataset-specific.

Not overclaiming generalization is itself a way the system beats sloppy humans/papers, who routinely report
a single-dataset number as if it were general.

THE INVARIANT
-------------
Each per-dataset outcome must come from the SAME frozen Tier-3 certify path (passed in as certify_fn).
This module aggregates and labels; it never certifies, never relaxes a bound, and a missing replication set
yields the WEAKER claim (in_distribution), never the stronger one.

CONTRACT: stdlib only. certify_fn is supplied by the caller (the loop's frozen certify path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

IN_DISTRIBUTION = "in_distribution"
SHIFT_ROBUST = "shift_robust"
SCOPED = "scoped"
VALID_SCOPES = (IN_DISTRIBUTION, SHIFT_ROBUST, SCOPED)


@dataclass
class DatasetCertOutcome:
    """The frozen certify result on ONE dataset: did it certify, and the certified lower bound."""
    dataset: str
    certified: bool
    lower_bound: float


@dataclass
class ReplicationReport:
    primary: DatasetCertOutcome
    replicas: List[DatasetCertOutcome]
    scope: str
    pass_rate: float
    reason: str
    worst_case_bound: float = field(default=0.0)

    def is_general(self) -> bool:
        return self.scope == SHIFT_ROBUST

    def summary(self) -> str:
        rep = ", ".join(f"{r.dataset}={'Y' if r.certified else 'n'}({r.lower_bound:.3f})" for r in self.replicas)
        return (f"scope={self.scope} primary={self.primary.dataset}({self.primary.lower_bound:.3f}) "
                f"pass_rate={self.pass_rate:.2f} worst={self.worst_case_bound:.3f} replicas=[{rep}]")


def classify_scope(primary: DatasetCertOutcome, replicas: Sequence[DatasetCertOutcome],
                   *, min_pass_rate: float = 0.8) -> ReplicationReport:
    """Assign an honest scope. Primary must certify for any positive claim; otherwise the result is scoped
    (it did not even certify on its own dataset, so it cannot be presented as general)."""
    reps = list(replicas)
    if not primary.certified:
        return ReplicationReport(primary, reps, SCOPED, 0.0,
                                 "primary did not certify", primary.lower_bound)
    if not reps:
        return ReplicationReport(primary, reps, IN_DISTRIBUTION, 0.0,
                                 "no replication set available -> in-distribution claim only",
                                 primary.lower_bound)
    n_pass = sum(1 for r in reps if r.certified)
    pass_rate = n_pass / len(reps)
    bounds = [primary.lower_bound] + [r.lower_bound for r in reps if r.certified]
    worst = min(bounds) if bounds else primary.lower_bound
    if pass_rate >= min_pass_rate:
        return ReplicationReport(primary, reps, SHIFT_ROBUST, pass_rate,
                                 f"replicated on {n_pass}/{len(reps)} datasets (>= {min_pass_rate:.0%})", worst)
    return ReplicationReport(primary, reps, SCOPED, pass_rate,
                             f"replicated on only {n_pass}/{len(reps)} datasets (< {min_pass_rate:.0%})",
                             primary.lower_bound)


def replicate(certify_fn: Callable[[str], DatasetCertOutcome], primary: str,
              replication_set: Sequence[str], *, min_pass_rate: float = 0.8) -> ReplicationReport:
    """Re-certify a winning recipe across the replication set using the frozen certify_fn and label scope.
    certify_fn(dataset_name) -> DatasetCertOutcome (the same frozen Tier-3 path used for the primary)."""
    primary_outcome = certify_fn(primary)
    replica_outcomes = [certify_fn(name) for name in replication_set if name != primary]
    return classify_scope(primary_outcome, replica_outcomes, min_pass_rate=min_pass_rate)


__all__ = ["IN_DISTRIBUTION", "SHIFT_ROBUST", "SCOPED", "VALID_SCOPES", "DatasetCertOutcome",
           "ReplicationReport", "classify_scope", "replicate"]
