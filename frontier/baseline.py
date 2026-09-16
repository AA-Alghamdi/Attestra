"""Baseline-first / landscape-anchoring: a CERTIFIED floor a real win must beat.

WHY THIS EXISTS
---------------
A frontier autoresearcher that reports "certified at theta" without first establishing a
strong, well-known floor is reporting a number with no landscape anchor. A candidate can
clear an operator threshold theta and still be *no better than a tuned reference model* (or,
worse, no better than majority/mean). The audit line for this whole build is: "honest gating
-- never report something as a win unless it is one." This module supplies the missing anchor.

It does three things, all on the SOUND frozen certifier (imported, never re-implemented):

  1. Fit a small set of strong, well-known baselines for the task kind:
       - the TRIVIAL floor    : majority class (clf) / mean (reg)  -- the absolute floor;
       - a REFERENCE model    : the standard go-to (HistGBM) + a linear reference
                                 (logreg/ridge). These are the models a competent practitioner
                                 reaches for first; if the autoresearcher cannot beat them, it
                                 has not done research.
     Each baseline is a Program (build_estimator() defined) run through the SAME predictions-only
     sandbox firewall as any candidate -- no special trust, no in-process exec.

  2. CERTIFY the best baseline on a sealed test as a "floor certificate" (the floor's certified
     lower bound). Selection among baselines is done on a VAL split (trusted parent); only the
     single best baseline is certified on the sealed test -- exactly the spine's one-winner /
     one-peek discipline, applied to the floor.

  3. Expose a GATE the orchestrator uses for its FINAL decision: a candidate win is reported as
     an *improvement over the baseline* only if the winner's certified LOWER bound strictly beats
     the floor's certified LOWER bound. Otherwise the run is reported honestly as
     "did not beat baseline" -- certified-against-theta is necessary but NOT sufficient to claim
     an improvement.

SEALED-PEEK ACCOUNTING (the load-bearing integrity point)
---------------------------------------------------------
The orchestrator owns ONE sealed peek: the winner is evaluated exactly once on
``splits.sealed_test`` (the main sealed partition). The floor must NOT contaminate that peek --
i.e. it must not add an uncounted/uncharged evaluation to the *same* sealed labels, which would
silently inflate the multiplicity the winner's certificate paid for.

Discipline used here: the floor builds its OWN independent three-way split via
``certify.make_splits`` with a DIFFERENT seed (``seed + FLOOR_SEED_OFFSET``). That split's sealed
partition is a different held-out subset of the data, with its OWN ``SealedTest`` instance and its
OWN one-peek budget. The floor selects among baselines on the floor split's VAL, then certifies
the single best baseline ONCE on the floor split's sealed test (``floor_sealed_peeks == 1``).
Consequently:

  * the winner's sealed test is peeked exactly once (orchestrator, unchanged);
  * the floor's sealed test is peeked exactly once (here);
  * the two are DISJOINT sealed partitions, so there is no shared-label multiplicity to correct
    across them -- each certificate is an independent, honestly-counted one-peek bound.

The cost is statistical (the floor is certified on a different held-out fold than the winner) but
the alternative -- two peeks of the same sealed labels -- would require Bonferroni-charging the
winner's certificate for the floor's peek, i.e. *weakening the winner's already-issued bound after
the fact*. We choose independent folds so neither certificate's discipline is retroactively
altered. ``floor_split_seed`` and ``floor_sealed_peeks`` are reported for audit.

WIRING
------
``CoreOrchestrator.run`` builds the floor (``compute_floor``) right after its own split and BEFORE
the round loop, then ANDs ``beats_floor(winner_cert, floor_cert)`` into the final ``certified``
decision and writes the verdict into the report. A winner that certifies against theta but does
NOT beat the floor's certified lower bound is reported as ``certified=False`` with decline reason
"did not beat baseline floor ..." -- an honest non-win, not a relabeled val score.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

# Repo root on sys.path (mirror certify.py / orchestrator.py so imports are identical).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier import certify, sandbox          # noqa: E402  (frozen spine)
from frontier.program import Program           # noqa: E402
from frontier.task import Task                 # noqa: E402


# The floor uses a DIFFERENT split seed than the winner so its sealed partition is disjoint
# from the winner's; see module docstring "SEALED-PEEK ACCOUNTING".
FLOOR_SEED_OFFSET = 9973   # a fixed, arbitrary large prime; NOT tuned to any benchmark.


# ============================================================================================
# Baseline program library (strong, well-known references -- not reverse-engineered)
# ============================================================================================

def _trivial_clf_code() -> str:
    # Majority-class predictor: the absolute classification floor.
    return (
        "from sklearn.dummy import DummyClassifier\n"
        "def build_estimator():\n"
        "    return DummyClassifier(strategy='most_frequent')\n"
    )


def _trivial_reg_code() -> str:
    # Mean predictor: the absolute regression floor.
    return (
        "from sklearn.dummy import DummyRegressor\n"
        "def build_estimator():\n"
        "    return DummyRegressor(strategy='mean')\n"
    )


def _ref_codes(kind: str) -> List[Tuple[str, str]]:
    """(label, code) for the well-known REFERENCE baselines a competent practitioner reaches for.

    These are deliberately standard and untuned-to-this-task: a gradient-boosted-trees default
    (the modern tabular workhorse) and a regularized linear model on standardized features (the
    classical strong-linear reference). They are the landscape anchor, nothing more.
    """
    if kind == "classification":
        return [
            ("baseline_hgb", (
                "from sklearn.ensemble import HistGradientBoostingClassifier\n"
                "def build_estimator():\n"
                "    return HistGradientBoostingClassifier(random_state=0)\n")),
            ("baseline_logreg", (
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "def build_estimator():\n"
                "    return Pipeline([('s', StandardScaler()),"
                " ('m', LogisticRegression(max_iter=2000))])\n")),
        ]
    return [
        ("baseline_hgb", (
            "from sklearn.ensemble import HistGradientBoostingRegressor\n"
            "def build_estimator():\n"
            "    return HistGradientBoostingRegressor(random_state=0)\n")),
        ("baseline_ridge", (
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.linear_model import Ridge\n"
            "def build_estimator():\n"
            "    return Pipeline([('s', StandardScaler()), ('m', Ridge(alpha=1.0))])\n")),
    ]


def baseline_programs(kind: str) -> List[Program]:
    """The floor's candidate set: trivial floor + strong reference models. All real Programs."""
    if kind == "classification":
        progs = [Program(code=_trivial_clf_code(), source="baseline", label="baseline_majority",
                         provenance={"role": "trivial_floor"})]
    elif kind == "regression":
        progs = [Program(code=_trivial_reg_code(), source="baseline", label="baseline_mean",
                         provenance={"role": "trivial_floor"})]
    else:
        raise ValueError(f"unknown task kind {kind!r}")
    for label, code in _ref_codes(kind):
        progs.append(Program(code=code, source="baseline", label=label,
                             provenance={"role": "reference_model"}))
    return progs


# ============================================================================================
# Floor result
# ============================================================================================

@dataclass
class FloorCertificate:
    """The certified baseline floor + the audit trail of how its sealed peek was accounted."""
    certified: bool                          # did the floor itself clear theta on its sealed fold?
    certificate: Optional[dict]              # the frozen-certifier cert dict for the best baseline
    best_label: str                          # which baseline became the floor
    best_val_score: Optional[float]          # floor's selection (VAL) score
    lower_bound: Optional[float]             # floor's certified sealed LOWER bound (the anchor)
    observed: Optional[float]                # floor's sealed point estimate
    floor_split_seed: int                    # the INDEPENDENT seed used for the floor's split
    floor_sealed_peeks: int                  # must be exactly 1 (the floor's own one-peek budget)
    candidates: List[dict] = field(default_factory=list)  # per-baseline VAL scores (audit)
    note: str = ""                           # honest degradation note (e.g. all baselines failed)

    def to_dict(self) -> dict:
        return {
            "certified": self.certified,
            "best_label": self.best_label,
            "best_val_score": self.best_val_score,
            "lower_bound": self.lower_bound,
            "observed": self.observed,
            "floor_split_seed": self.floor_split_seed,
            "floor_sealed_peeks": self.floor_sealed_peeks,
            "candidates": list(self.candidates),
            "note": self.note,
        }


# ============================================================================================
# Compute the floor
# ============================================================================================

def compute_floor(task: Task, *, seed: int = 0, test_frac: float = 0.30, val_frac: float = 0.20,
                  wall_seconds: float = 60.0, cpu_seconds: int = 55,
                  run_fn: Optional[Callable[[Program, np.ndarray, np.ndarray, np.ndarray], Optional[list]]] = None
                  ) -> FloorCertificate:
    """Fit baselines on an INDEPENDENT split, select on VAL, certify the best on sealed (one peek).

    Parameters mirror the orchestrator's split protocol so the floor sees a comparable task; the
    seed is offset (``seed + FLOOR_SEED_OFFSET``) so the floor's sealed partition is DISJOINT from
    the winner's (see module docstring). ``run_fn`` lets a caller inject a firewall executor; the
    default routes every baseline through the Phase-0 ``sandbox`` (predictions only).

    Returns a ``FloorCertificate`` carrying the certified floor lower bound (the anchor the gate
    compares against). Degrades honestly: if every baseline fails to run, the floor is reported as
    uncertified with a note, and the gate then admits the winner (it cannot be held to a floor that
    could not be computed) -- documented behaviour, never a fabricated bound.
    """
    floor_seed = int(seed) + FLOOR_SEED_OFFSET
    splits = certify.make_splits(task, seed=floor_seed, test_frac=test_frac, val_frac=val_frac)
    X_train = Task.rows_to_X(splits.train_rows)
    y_train = Task.rows_to_y(splits.train_rows, task.kind)
    X_val = Task.rows_to_X(splits.val_rows)

    if run_fn is None:
        def run_fn(prog, Xtr, ytr, Xev):  # firewall: preds-only via the Phase-0 sandbox
            rr = sandbox.run_program(prog, Xtr, ytr, Xev, kind=task.kind,
                                     wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)
            return rr.preds if rr.ok else None

    progs = baseline_programs(task.kind)

    # ---- selection: VAL only (the floor's selection touches no sealed labels) -----------------
    best_prog: Optional[Program] = None
    best_score: Optional[float] = None
    candidates: List[dict] = []
    for p in progs:
        preds = run_fn(p, X_train, y_train, X_val)
        if preds is None or len(preds) != len(splits.val_rows):
            candidates.append({"label": p.label, "val_score": None,
                               "role": p.provenance.get("role")})
            continue
        score = certify.score_val(task, splits.val_rows, preds)
        candidates.append({"label": p.label, "val_score": round(float(score), 4),
                           "role": p.provenance.get("role")})
        if best_score is None or score > best_score:
            best_score, best_prog = float(score), p

    if best_prog is None:
        # honest degradation: no baseline ran -> no floor. The gate treats this as "no floor".
        return FloorCertificate(
            certified=False, certificate=None, best_label="", best_val_score=None,
            lower_bound=None, observed=None, floor_split_seed=floor_seed,
            floor_sealed_peeks=0, candidates=candidates,
            note="no baseline executed successfully; floor unavailable (gate admits winner)")

    # ---- certify the single best baseline ONCE on the floor's OWN sealed test -----------------
    X_sealed = Task.rows_to_X(splits.sealed_rows)
    sealed_preds = run_fn(best_prog, X_train, y_train, X_sealed)
    if sealed_preds is None or len(sealed_preds) != len(splits.sealed_rows):
        return FloorCertificate(
            certified=False, certificate=None, best_label=best_prog.label,
            best_val_score=round(float(best_score), 4), lower_bound=None, observed=None,
            floor_split_seed=floor_seed, floor_sealed_peeks=0, candidates=candidates,
            note="best baseline failed on its sealed re-fit; floor unavailable (gate admits winner)")

    cert = certify.certify_on_sealed(task, splits, sealed_preds)  # the floor's ONE counted peek
    return FloorCertificate(
        certified=bool(cert.get("certified")),
        certificate=cert,
        best_label=best_prog.label,
        best_val_score=round(float(best_score), 4),
        lower_bound=cert.get("lower_bound"),
        observed=cert.get("observed"),
        floor_split_seed=floor_seed,
        floor_sealed_peeks=int(cert.get("peeks", 1)),
        candidates=candidates,
        note="",
    )


# ============================================================================================
# The gate
# ============================================================================================

@dataclass
class FloorVerdict:
    """Result of gating a winner's certificate against the baseline floor."""
    beats_floor: bool                 # winner's certified LB strictly beats the floor's LB?
    floor_available: bool             # was a floor certificate computed at all?
    winner_lower_bound: Optional[float]
    floor_lower_bound: Optional[float]
    floor_label: str
    margin: Optional[float]           # winner_lb - floor_lb (None if either missing)
    reason: str

    def to_dict(self) -> dict:
        return {
            "beats_floor": self.beats_floor,
            "floor_available": self.floor_available,
            "winner_lower_bound": self.winner_lower_bound,
            "floor_lower_bound": self.floor_lower_bound,
            "floor_label": self.floor_label,
            "margin": self.margin,
            "reason": self.reason,
        }


def beats_floor(winner_cert: Optional[dict], floor: Optional[FloorCertificate],
                *, tol: float = 1e-9) -> FloorVerdict:
    """Honest gate: a winner is an IMPROVEMENT only if its certified LOWER bound beats the floor's.

    Comparison is LOWER bound vs LOWER bound -- the certified quantities, not point estimates --
    so the verdict inherits the same statistical discipline as promotion. ``tol`` makes a numeric
    TIE (winner_lb <= floor_lb + tol) count as "did NOT beat the floor": a tie is not an
    improvement. If no floor could be computed, the gate admits the winner (floor_available=False)
    and says so -- it never invents a floor to block a result.
    """
    wlb = None if winner_cert is None else winner_cert.get("lower_bound")
    if floor is None or floor.lower_bound is None:
        flb = None if floor is None else floor.lower_bound
        return FloorVerdict(
            beats_floor=True, floor_available=False, winner_lower_bound=wlb,
            floor_lower_bound=flb, floor_label=(floor.best_label if floor else ""),
            margin=None,
            reason="no baseline floor available; winner admitted (floor could not be certified)")

    flb = float(floor.lower_bound)
    if wlb is None:
        return FloorVerdict(
            beats_floor=False, floor_available=True, winner_lower_bound=None,
            floor_lower_bound=flb, floor_label=floor.best_label, margin=None,
            reason="winner has no certified lower bound to compare against the floor")

    wlb = float(wlb)
    margin = wlb - flb
    if wlb > flb + tol:
        reason = (f"winner sealed lower bound {round(wlb, 4)} beats baseline floor "
                  f"'{floor.best_label}' lower bound {round(flb, 4)} (margin {round(margin, 4)})")
        return FloorVerdict(True, True, round(wlb, 4), round(flb, 4), floor.best_label,
                            round(margin, 4), reason)
    reason = (f"winner sealed lower bound {round(wlb, 4)} does NOT beat baseline floor "
              f"'{floor.best_label}' lower bound {round(flb, 4)} (margin {round(margin, 4)}); "
              f"reported as no improvement over baseline")
    return FloorVerdict(False, True, round(wlb, 4), round(flb, 4), floor.best_label,
                        round(margin, 4), reason)


__all__ = [
    "FLOOR_SEED_OFFSET", "baseline_programs", "FloorCertificate", "compute_floor",
    "FloorVerdict", "beats_floor",
]
