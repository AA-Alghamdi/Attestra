"""Cross-experiment False Discovery Rate (FDR) control.

When running many experiments, the probability of at least one false positive
grows with the number of experiments. This module implements:

1. Benjamini-Hochberg (BH) procedure — controls FDR at level q
2. Benjamini-Yekutieli (BY) — controls FDR under arbitrary dependence
3. Holm-Bonferroni — family-wise error rate (FWER) control (stricter)
4. Alpha-spending for sequential experiments (Lan-DeMets)

The FDR controller is a STREAM processor: each new experiment result is fed in,
and the controller decides whether to accept/reject while controlling FDR
across ALL experiments seen so far.

INVARIANT: The frozen certifier produces per-experiment p-values (from
Clopper-Pearson or bootstrap). This module operates ABOVE the certifier,
never modifying its internals.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ExperimentCertificate:
    """A certificate produced by the frozen certifier for one experiment."""
    experiment_id: str
    metric: str
    threshold: float
    observed_lower_bound: float
    p_value: float                        # p-value for H0: metric <= threshold
    certified_locally: bool               # did it pass per-experiment test?
    dataset_fingerprint: str = ""
    technique: str = ""
    n_test: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class FDRDecision:
    """FDR-controlled decision for a single experiment."""
    experiment_id: str
    accepted: bool                        # accepted after FDR correction
    adjusted_p_value: float               # adjusted p-value (BH/BY/Holm)
    rank: int                             # rank in sorted p-values
    alpha_threshold: float                # the threshold it was compared against
    method: str = "benjamini_hochberg"


class FDRController:
    """Stream-based FDR controller for cross-experiment certification.

    Usage:
        fdr = FDRController(alpha=0.05, method="benjamini_hochberg")
        
        # As experiments complete:
        cert = ExperimentCertificate(...)
        decision = fdr.submit(cert)
        
        # decision.accepted tells you if this experiment passes FDR control
        
        # View all decisions:
        fdr.summary()
    """

    METHODS = ("benjamini_hochberg", "benjamini_yekutieli", "holm_bonferroni")

    def __init__(self, alpha: float = 0.05, method: str = "benjamini_hochberg",
                 persist_path: Optional[str] = None):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.alpha = alpha
        self.method = method
        self.certificates: List[ExperimentCertificate] = []
        self.decisions: List[FDRDecision] = []
        self._persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self._load()

    def submit(self, cert: ExperimentCertificate) -> FDRDecision:
        """Submit a new experiment certificate and get FDR-controlled decision.

        This re-runs the FDR procedure over ALL certificates (streaming BH).
        """
        self.certificates.append(cert)
        self._recompute_all()
        self._persist()
        # Return decision for the latest certificate
        return self.decisions[-1] if self.decisions else FDRDecision(
            experiment_id=cert.experiment_id, accepted=False,
            adjusted_p_value=1.0, rank=0, alpha_threshold=self.alpha,
        )

    def get_decision(self, experiment_id: str) -> Optional[FDRDecision]:
        """Get the FDR decision for a specific experiment."""
        for d in self.decisions:
            if d.experiment_id == experiment_id:
                return d
        return None

    def acceptance_rate(self) -> float:
        """Fraction of experiments accepted under FDR control."""
        if not self.decisions:
            return 0.0
        return sum(1 for d in self.decisions if d.accepted) / len(self.decisions)

    def estimated_fdr(self) -> float:
        """Estimate the actual FDR (fraction of false discoveries)."""
        if not self.decisions:
            return 0.0
        accepted = [d for d in self.decisions if d.accepted]
        if not accepted:
            return 0.0
        # Use the highest rank's threshold as an estimate
        return min(self.alpha, max(d.adjusted_p_value for d in accepted))

    def summary(self) -> Dict:
        n_total = len(self.certificates)
        n_accepted = sum(1 for d in self.decisions if d.accepted)
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n_experiments": n_total,
            "n_accepted": n_accepted,
            "n_rejected": n_total - n_accepted,
            "acceptance_rate": self.acceptance_rate(),
            "estimated_fdr": self.estimated_fdr(),
        }

    def _recompute_all(self) -> None:
        """Recompute FDR decisions over all certificates."""
        if self.method == "benjamini_hochberg":
            self.decisions = self._bh_procedure()
        elif self.method == "benjamini_yekutieli":
            self.decisions = self._by_procedure()
        elif self.method == "holm_bonferroni":
            self.decisions = self._holm_procedure()

    def _bh_procedure(self) -> List[FDRDecision]:
        """Benjamini-Hochberg step-up procedure."""
        n = len(self.certificates)
        if n == 0:
            return []

        # Sort by p-value
        indexed = [(i, c.p_value) for i, c in enumerate(self.certificates)]
        indexed.sort(key=lambda x: x[1])

        decisions = [None] * n
        # Step up from smallest p-value
        max_rank_accepted = 0
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            threshold = (rank / n) * self.alpha
            if p <= threshold:
                max_rank_accepted = rank

        # Accept all with rank <= max_rank_accepted
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            cert = self.certificates[orig_idx]
            adjusted_p = min(1.0, p * n / rank)
            decisions[orig_idx] = FDRDecision(
                experiment_id=cert.experiment_id,
                accepted=rank <= max_rank_accepted,
                adjusted_p_value=adjusted_p,
                rank=rank,
                alpha_threshold=(rank / n) * self.alpha,
                method="benjamini_hochberg",
            )
        return decisions

    def _by_procedure(self) -> List[FDRDecision]:
        """Benjamini-Yekutieli (works under arbitrary dependence)."""
        n = len(self.certificates)
        if n == 0:
            return []

        # Harmonic correction factor
        c_n = sum(1.0 / k for k in range(1, n + 1))

        indexed = [(i, c.p_value) for i, c in enumerate(self.certificates)]
        indexed.sort(key=lambda x: x[1])

        decisions = [None] * n
        max_rank_accepted = 0
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            threshold = (rank / (n * c_n)) * self.alpha
            if p <= threshold:
                max_rank_accepted = rank

        for rank, (orig_idx, p) in enumerate(indexed, 1):
            cert = self.certificates[orig_idx]
            adjusted_p = min(1.0, p * n * c_n / rank)
            decisions[orig_idx] = FDRDecision(
                experiment_id=cert.experiment_id,
                accepted=rank <= max_rank_accepted,
                adjusted_p_value=adjusted_p,
                rank=rank,
                alpha_threshold=(rank / (n * c_n)) * self.alpha,
                method="benjamini_yekutieli",
            )
        return decisions

    def _holm_procedure(self) -> List[FDRDecision]:
        """Holm-Bonferroni step-down for FWER control."""
        n = len(self.certificates)
        if n == 0:
            return []

        indexed = [(i, c.p_value) for i, c in enumerate(self.certificates)]
        indexed.sort(key=lambda x: x[1])

        decisions = [None] * n
        # Step down: reject while p <= alpha / (n - rank + 1)
        rejected = True
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            threshold = self.alpha / (n - rank + 1)
            if not rejected or p > threshold:
                rejected = False
            cert = self.certificates[orig_idx]
            adjusted_p = min(1.0, p * (n - rank + 1))
            decisions[orig_idx] = FDRDecision(
                experiment_id=cert.experiment_id,
                accepted=rejected if rank <= len(indexed) else False,
                adjusted_p_value=adjusted_p,
                rank=rank,
                alpha_threshold=threshold,
                method="holm_bonferroni",
            )
        return decisions

    def _persist(self) -> None:
        if not self._persist_path:
            return
        os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
        data = {
            "alpha": self.alpha,
            "method": self.method,
            "certificates": [asdict(c) for c in self.certificates],
        }
        with open(self._persist_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self) -> None:
        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)
            self.alpha = data.get("alpha", self.alpha)
            self.method = data.get("method", self.method)
            for c_dict in data.get("certificates", []):
                self.certificates.append(ExperimentCertificate(**{
                    k: v for k, v in c_dict.items()
                    if k in ExperimentCertificate.__dataclass_fields__
                }))
            self._recompute_all()
        except (json.JSONDecodeError, IOError):
            pass


# ============================================================================== Alpha-spending

class AlphaSpending:
    """Lan-DeMets alpha-spending for sequential monitoring.

    Used when experiments are run sequentially and you want to control
    type-I error while allowing early stopping.

    Spending functions:
      - "obrien_fleming": Conservative early, aggressive late
      - "pocock": Uniform spending
      - "linear": Linear spending

    Usage:
        spender = AlphaSpending(alpha=0.05, max_looks=10)
        for i in range(max_looks):
            alpha_i = spender.spend(fraction=(i+1)/10)
            # Use alpha_i as the significance level for this peek
    """

    def __init__(self, alpha: float = 0.05, max_looks: int = 10,
                 spending_fn: str = "obrien_fleming"):
        self.alpha = alpha
        self.max_looks = max_looks
        self.spending_fn = spending_fn
        self._spent: List[float] = []
        self._cumulative_spent = 0.0

    def spend(self, fraction: float) -> float:
        """Get the alpha to spend at this information fraction.

        Args:
            fraction: Information fraction (0 to 1), e.g. n_current / n_total

        Returns:
            Alpha level to use for this peek (incremental spend)
        """
        fraction = np.clip(fraction, 0.0, 1.0)
        target_cumulative = self._spending_function(fraction)
        incremental = max(0, target_cumulative - self._cumulative_spent)
        self._cumulative_spent = target_cumulative
        self._spent.append(incremental)
        return incremental

    def remaining_alpha(self) -> float:
        """How much alpha is left to spend."""
        return max(0, self.alpha - self._cumulative_spent)

    def _spending_function(self, t: float) -> float:
        """Cumulative alpha spent at information fraction t."""
        if self.spending_fn == "obrien_fleming":
            # O'Brien-Fleming: 2 * (1 - Phi(z_{alpha/2} / sqrt(t)))
            from scipy.stats import norm
            z = norm.ppf(1 - self.alpha / 2)
            return float(2 * (1 - norm.cdf(z / np.sqrt(max(t, 1e-10)))))
        elif self.spending_fn == "pocock":
            # Pocock: alpha * log(1 + (e-1)*t)
            return float(self.alpha * np.log(1 + (np.e - 1) * t))
        elif self.spending_fn == "linear":
            return float(self.alpha * t)
        else:
            return float(self.alpha * t)
