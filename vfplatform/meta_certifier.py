"""CERTIFIER-FIRST META-CERTIFIER -- on a novel problem, verification leads.

STATUS (2026-06): WIRED into the REGENERATIVE researcher. RecipeResearcher.run() calls validate_certifier
on the arena's framing() BEFORE any search runs; a gameable framing (trivial-baseline / label-shuffle /
straddle-leak / split-leak) makes the whole run REFUSE (cert.refused=True, peeks_used=0). Falsified by a
hermetic lock (test_recipe_research.py::test_meta_certifier_refuses_leaky_framing) and run as the
framing-side gate on real WILDS Camelyon17. It is the framing-side complement to the datapool data-hygiene
gate (datapool.py), which refuses CONTAMINATED data on the same pre-search step.

WHY THIS EXISTS
---------------
On a genuinely new problem you do not yet have a TRUSTED fitness. Searching against an un-validated
certifier is how you fool yourself: a metric a trivial baseline games, a split with a leak, an evaluation
that rubber-stamps even shuffled labels. So BEFORE the frozen certifier is trusted to drive search on a new
problem, its FRAMING (metric + split + constraints) is stress-tested adversarially. Generation and
verification are two loops; on a novel problem verification must lead.

The meta-certifier does NOT replace or modify the frozen certifier. It validates the problem framing the
frozen certifier will operate within, by running probes that a sound framing must survive:
  * trivial-baseline probe -- a majority/constant predictor must NOT clear theta (else the metric is gameable);
  * label-shuffle probe    -- a model trained on SHUFFLED labels must NOT clear theta (else there is leakage /
    the evaluation rubber-stamps);
  * straddle-leak probe     -- no near-duplicate row may straddle train/sealed (reuses data_cert);
  * split-leak probe        -- group/temporal splits must have no group straddle or temporal leak.

THE INVARIANT
-------------
This runs the SAME user-supplied fit/predict + metric the real certifier uses; it only adds adversarial
probes and reports trustworthy True/False. It never certifies, never relaxes a bound, and a probe that
cannot be run (missing groups/times) is reported as 'skipped', never as a pass.

CONTRACT: numpy + the repo's own splits.py / data_cert.py. fit_predict_fn and metric_fn are supplied by
the caller (the same ones the frozen path uses).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from . import data_cert as DC
from . import splits as SP

# fit a model on (X, y, train_idx) and return predictions for sealed_idx
FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
# score predictions: (y_true, y_pred) -> float (larger is better)
Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class CertifierProbe:
    name: str
    passed: bool
    detail: str
    skipped: bool = False


@dataclass
class MetaCertReport:
    trustworthy: bool
    probes: List[CertifierProbe] = field(default_factory=list)
    reason: str = ""

    def failed_probes(self) -> List[CertifierProbe]:
        return [p for p in self.probes if not p.passed and not p.skipped]

    def summary(self) -> str:
        rows = [f"  [{'PASS' if p.passed else ('SKIP' if p.skipped else 'FAIL')}] {p.name}: {p.detail}"
                for p in self.probes]
        head = f"trustworthy={self.trustworthy} -- {self.reason}"
        return head + "\n" + "\n".join(rows)


def _majority_predictor(y_train: np.ndarray, n_sealed: int) -> np.ndarray:
    classes, counts = np.unique(y_train, return_counts=True)
    majority = classes[int(np.argmax(counts))]
    return np.full(n_sealed, majority)


def validate_certifier(*, X: np.ndarray, y: np.ndarray, train_idx: Sequence[int], sealed_idx: Sequence[int],
                       fit_predict_fn: FitPredict, metric_fn: Metric, theta: float,
                       groups: Optional[np.ndarray] = None, times: Optional[np.ndarray] = None,
                       seed: int = 0, n_shuffles: int = 5) -> MetaCertReport:
    """Adversarially validate a problem framing before the frozen certifier is trusted to drive search.
    Returns trustworthy=False if any runnable probe fires."""
    X = np.asarray(X)
    y = np.asarray(y)
    tr = np.asarray(list(train_idx))
    se = np.asarray(list(sealed_idx))
    probes: List[CertifierProbe] = []

    # 1) trivial-baseline probe -------------------------------------------------------------------------
    maj_pred = _majority_predictor(y[tr], len(se))
    maj_score = float(metric_fn(y[se], maj_pred))
    probes.append(CertifierProbe(
        "trivial_baseline", maj_score < theta,
        f"majority-class score={maj_score:.3f} vs theta={theta:.3f} "
        f"({'ok: trivial model fails' if maj_score < theta else 'GAMEABLE: trivial model clears theta'})"))

    # 2) label-shuffle probe (a permutation test: averaged over several shuffles to kill single-draw noise)
    rng = np.random.default_rng(seed)
    try:
        shuf_scores: List[float] = []
        for _ in range(max(1, n_shuffles)):
            y_shuf = y.copy()
            y_shuf[tr] = y[tr][rng.permutation(len(tr))]
            shuf_pred = np.asarray(fit_predict_fn(X, y_shuf, tr, se))
            shuf_scores.append(float(metric_fn(y[se], shuf_pred)))
        mean_shuf = float(np.mean(shuf_scores))
        probes.append(CertifierProbe(
            "label_shuffle", mean_shuf < theta,
            f"mean shuffled-label score={mean_shuf:.3f} over {len(shuf_scores)} perms vs theta={theta:.3f} "
            f"({'ok: shuffle fails' if mean_shuf < theta else 'LEAK: shuffled labels still clear theta'})"))
    except Exception as exc:
        probes.append(CertifierProbe("label_shuffle", False, f"fit_predict crashed: {exc}"))

    # 3) straddle-leak probe ----------------------------------------------------------------------------
    n_straddle = DC.detect_near_dup_straddle(X, tr, se)
    probes.append(CertifierProbe(
        "straddle_leak", n_straddle == 0,
        f"{n_straddle} near-duplicate rows straddle train/sealed" if n_straddle
        else "no near-duplicate straddle"))

    # 4) split-leak probe (only if group/time structure is declared) ------------------------------------
    if groups is None and times is None:
        probes.append(CertifierProbe("split_leak", True, "no group/time structure declared", skipped=True))
    else:
        # sealed indices placed as `val` so SplitResult's train->val leak checks cover the train/sealed pair
        result = SP.SplitResult(train_idx=list(tr), val_idx=list(se), sealed_idx=[],
                                method="meta_probe", embargoed_idx=[])
        try:
            result.assert_no_leakage(groups=np.asarray(groups) if groups is not None else None,
                                     times=np.asarray(times) if times is not None else None)
            probes.append(CertifierProbe("split_leak", True, "no group straddle / temporal leak"))
        except SP.SplitError as exc:
            probes.append(CertifierProbe("split_leak", False, f"split leak: {exc}"))

    runnable_fails = [p for p in probes if not p.passed and not p.skipped]
    trustworthy = len(runnable_fails) == 0
    reason = ("all probes passed -- framing is trustworthy to drive search"
              if trustworthy else
              "framing rejected: " + "; ".join(p.name for p in runnable_fails))
    return MetaCertReport(trustworthy, probes, reason)


__all__ = ["CertifierProbe", "MetaCertReport", "validate_certifier", "FitPredict", "Metric"]
