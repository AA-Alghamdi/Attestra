"""Phase 5 (data-centric): DATA interventions as certifiable experiments.

The owner's framing: a data change (oversample the minority class, mix up examples,
synthesize rows from a fitted distribution, drop duplicates) is a HYPOTHESIS, and the
question is not "which model scored highest" but "did this data change CAUSE a certified
lift?". The TTS tone-dimensionality example generalizes: vary one factor (the training
DATA), hold everything else fixed (the same proposer set, the same splits, the same frozen
certifier), and read off a go/no-go from the sealed certificate.

The load-bearing integrity rule for data-centric work
------------------------------------------------------
A data intervention is allowed to touch the TRAIN side ONLY. Synthetic / resampled rows
must never enter the validation or sealed splits: if they did, the model would be selected
and certified against data drawn from the same generator as its training data, and the
"lift" would be a leakage artifact, not a real generalization gain. So the protocol is:

    1. split the BASE task once (train / val / sealed) via the sound certify.make_splits;
    2. apply the intervention to TRAIN ROWS ONLY -> new train rows;
    3. run the SAME proposer/sandbox loop and certify the winner on the SAME, untouched
       sealed split via the frozen certifier.

Base and intervention arms share the identical val and sealed rows. That is what makes the
A/B a controlled experiment: the only thing that varies between arms is the training data.

This module reuses the sound pieces by import and NEVER recomputes a promotion-bearing
number itself: `frontier.certify.make_splits` / `score_val` / `certify_on_sealed` (which wrap
`vectorforge.science` + `vfplatform.sealed`) own every decision number; the interventions
here only manufacture train rows, and the data-quality audit below is purely diagnostic
(it informs which intervention to PROPOSE, it never PROMOTES anything).

# === WIRING ===
# The integrator plugs this in two ways.
#
# (A) As a standalone data-centric A/B (the headline capability), with NO change to the
#     Phase-0 spine:
#
#         from frontier.data_ops import (DataAudit, audit_task, SMOTEOversample,
#                                        certify_data_change)
#         report = certify_data_change(base_task, SMOTEOversample(k=5, target="balance"),
#                                      config=EngineConfig(rounds=2), seed=0)
#         # report.decision in {"go","no-go"}; report.caused_lift is the certified verdict.
#
#     certify_data_change runs BOTH arms through the EXACT engine discipline (split once,
#     select on val, one sealed peek for the winner) using a shared split so the arms differ
#     only in training data. The base arm's split is the canonical one; the intervention arm
#     reuses base.val_rows and base.sealed_rows verbatim (asserted disjoint from synthetic).
#
# (B) As a proposer-side capability inside ResearchEngine: an intervention is selected by the
#     audit (audit_task -> recommend_interventions) and the engine is run on the intervened
#     Task variant. To keep the val/sealed-clean invariant when going through ResearchEngine
#     directly, pass build_intervened_task(...) ONLY when you are willing to let the engine
#     re-split; for a clean controlled A/B always prefer certify_data_change, which shares the
#     split. Both paths certify through the same frozen gate; neither lets a heuristic promote.
#
# Argument shapes: interventions implement DataIntervention.apply(train_rows, kind, rng) ->
# new train_rows (list[dict] in the Task.to_rows() schema: {"target","features","_x"}). They
# operate on rows, not on the Task, precisely so they can be applied AFTER the split.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

# Repo root on sys.path so the sound certifier imports resolve (same pattern as certify.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from . import certify                       # sound split / score / sealed-certify wrappers
from . import sandbox                        # real out-of-process executor (firewall)
from .engine import EngineConfig
from .program import Program
from .proposers import LLMProposer, MutationProposer, ProposalSource, SeedProposer
from .task import Task


# =========================================================================== row helpers
# Interventions speak the row schema (Task.to_rows): {"target", "features": {f{i}:..}, "_x"}.
# Keeping a single canonical constructor here means every synthetic row is byte-shaped like a
# real one, so downstream Task.rows_to_X / rows_to_y / the certifier dedup all work unchanged.

def _row(x: Sequence[float], target, kind: str) -> dict:
    """Build one row in the canonical schema from a feature vector and a target."""
    xv = [float(v) for v in x]
    tgt = str(target) if kind == "classification" else float(target)
    return {"target": tgt, "features": {f"f{i}": float(v) for i, v in enumerate(xv)}, "_x": xv}


def _rows_to_xy(rows: Sequence[dict], kind: str) -> Tuple[np.ndarray, np.ndarray]:
    X = Task.rows_to_X(rows)
    y = Task.rows_to_y(rows, kind)
    return X, y


def _row_key(row: dict) -> Tuple[float, ...]:
    """Exact feature-vector key, used for duplicate detection and val/sealed disjointness."""
    return tuple(float(v) for v in row["_x"])


# =========================================================================== data audit

@dataclass
class DataAudit:
    """Diagnostic summary of a training set. PURELY informational: it tells the system which
    data intervention to PROPOSE, it never certifies or promotes anything.

    Fields:
      n, n_features                 : size of the (train) data audited.
      kind                          : "classification" | "regression".
      duplicate_fraction            : fraction of rows that are exact-feature duplicates of an
                                      earlier row (high -> dedup or the model overweights copies).
      class_counts / imbalance_ratio: per-class counts and max/min ratio (clf only; 1.0 = balanced,
                                      inf if a class is empty). Drives oversampling proposals.
      minority_classes              : classes below the mean count (clf only).
      label_noise_estimate          : crude kNN disagreement rate -- the fraction of points whose
                                      label disagrees with the majority of their k nearest neighbors.
                                      A NOISE PROXY, not ground truth; flagged as a heuristic.
      drift_scores / max_drift      : per-feature train-vs-sealed standardized mean shift (when a
                                      sealed reference is supplied). High drift warns that a
                                      train-only synthetic intervention may not transfer.
      notes                         : human-readable caveats.
    """
    n: int
    n_features: int
    kind: str
    duplicate_fraction: float
    class_counts: Dict[str, int] = field(default_factory=dict)
    imbalance_ratio: float = 1.0
    minority_classes: List[str] = field(default_factory=list)
    label_noise_estimate: float = 0.0
    drift_scores: Dict[str, float] = field(default_factory=dict)
    max_drift: float = 0.0
    notes: List[str] = field(default_factory=list)

    def recommend_interventions(self) -> List["DataIntervention"]:
        """Heuristic shortlist of interventions worth A/B-testing. HEURISTIC SEED ONLY: the
        recommendation never decides anything; certify_data_change must still prove the lift.
        An LLM could replace this ranking; it is a fallback, not the promoter."""
        recs: List[DataIntervention] = []
        if self.kind == "classification" and self.imbalance_ratio > 1.5:
            recs.append(SMOTEOversample(k=5, target="balance"))
            recs.append(GaussianClassSynthesize(target="balance"))
        if self.duplicate_fraction > 0.02:
            recs.append(Deduplicate())
        # Mixup is a general regularizer; propose it when data is plentiful enough to interpolate.
        if self.n >= 50:
            recs.append(Mixup(alpha=0.2, n_synth_frac=0.5))
        return recs


def audit_task(task: Task, *, seed: int = 0, test_frac: float = 0.30, val_frac: float = 0.20,
               k_noise: int = 5) -> Tuple[DataAudit, "certify.Splits"]:
    """Split the task once (the canonical split) and audit the TRAIN side, with the SEALED
    side used only as an untouched drift reference. Returns (audit, splits) so callers can
    reuse the exact same split for a controlled A/B (no second, divergent split)."""
    splits = certify.make_splits(task, seed=seed, test_frac=test_frac, val_frac=val_frac)
    audit = audit_rows(splits.train_rows, task.kind, sealed_rows=splits.sealed_rows, k_noise=k_noise)
    return audit, splits


def audit_rows(train_rows: Sequence[dict], kind: str, *, sealed_rows: Optional[Sequence[dict]] = None,
               k_noise: int = 5) -> DataAudit:
    """Compute the data-quality audit over a set of train rows (post-split)."""
    X, y = _rows_to_xy(train_rows, kind)
    n, d = (X.shape[0], X.shape[1]) if X.ndim == 2 else (len(train_rows), 0)
    notes: List[str] = []

    # --- duplication: exact feature-vector repeats (first occurrence not counted as a dup).
    seen, dups = set(), 0
    for r in train_rows:
        key = _row_key(r)
        if key in seen:
            dups += 1
        else:
            seen.add(key)
    dup_frac = dups / n if n else 0.0

    audit = DataAudit(n=n, n_features=d, kind=kind, duplicate_fraction=round(dup_frac, 4), notes=notes)

    if kind == "classification":
        counts = Counter(str(v) for v in y)
        audit.class_counts = dict(sorted(counts.items()))
        cvals = list(counts.values())
        mn, mx = (min(cvals), max(cvals)) if cvals else (0, 0)
        audit.imbalance_ratio = float("inf") if mn == 0 else round(mx / mn, 3)
        mean_ct = float(np.mean(cvals)) if cvals else 0.0
        audit.minority_classes = sorted([c for c, ct in counts.items() if ct < mean_ct])
        audit.label_noise_estimate = round(_knn_label_noise(X, np.asarray([str(v) for v in y]),
                                                            k=k_noise), 4)
        if audit.imbalance_ratio > 3:
            notes.append(f"strong class imbalance (ratio {audit.imbalance_ratio})")
    else:
        notes.append("regression: class-imbalance / label-noise metrics not applicable")

    # --- drift: standardized mean shift train-vs-sealed per feature (sealed is reference,
    # never modified). This warns that a train-only synthetic change may not transfer.
    if sealed_rows is not None and d > 0:
        Xs, _ = _rows_to_xy(sealed_rows, kind)
        if Xs.ndim == 2 and Xs.shape[0] > 1:
            mu_t, mu_s = X.mean(axis=0), Xs.mean(axis=0)
            sd = X.std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            drift = np.abs(mu_t - mu_s) / sd
            audit.drift_scores = {f"f{i}": round(float(v), 4) for i, v in enumerate(drift)}
            audit.max_drift = round(float(drift.max()), 4)
            if audit.max_drift > 1.0:
                notes.append(f"large train/sealed drift (max {audit.max_drift} sd)")
    return audit


def _knn_label_noise(X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Crude label-noise proxy: fraction of points whose label disagrees with the majority
    vote of their k nearest (excluding-self) neighbors. HEURISTIC -- it conflates genuine
    label noise with class overlap, so it is a flag for further inspection, not a measurement.
    Uses sklearn's exact KNN; falls back to 0.0 on degenerate inputs."""
    n = X.shape[0]
    if n <= k + 1 or X.ndim != 2:
        return 0.0
    try:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
        _, idx = nn.kneighbors(Xs)
        idx = idx[:, 1:]  # drop self
        disagree = 0
        for i in range(n):
            neigh = y[idx[i]]
            vals, cts = np.unique(neigh, return_counts=True)
            maj = vals[int(np.argmax(cts))]
            if maj != y[i]:
                disagree += 1
        return disagree / n
    except Exception:
        return 0.0


# =========================================================================== interventions

class DataIntervention(Protocol):
    """A train-side data transformation. Implementations MUST be pure (return new rows; do not
    mutate the input) and MUST operate on rows so they can be applied AFTER the split."""

    name: str

    def apply(self, train_rows: List[dict], kind: str, rng: np.random.Generator) -> List[dict]:
        ...


@dataclass
class Deduplicate:
    """Drop exact feature-vector duplicates, keeping the first occurrence. Train-only.

    Removing duplicate training rows prevents the model from overweighting copied points and
    can sharpen generalization; it is a data-quality fix the audit flags via duplicate_fraction."""
    name: str = "dedup"

    def apply(self, train_rows: List[dict], kind: str, rng: np.random.Generator) -> List[dict]:
        seen, out = set(), []
        for r in train_rows:
            key = _row_key(r)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(r))
        return out


@dataclass
class SMOTEOversample:
    """SMOTE-like minority oversampling (pure numpy, no imbalanced-learn).

    Classic SMOTE (Chawla et al., 2002): for a minority sample x_i, pick one of its k nearest
    same-class neighbors x_j and emit x_i + u*(x_j - x_i), u ~ U(0,1). This interpolates within
    the minority manifold rather than duplicating, which is why it beats naive replication.

    target:
      - "balance": oversample every minority class up to the majority count.
      - {class: count}: explicit per-class target counts (must be >= current count).
    Train-only by construction (operates on train_rows). Degrades to within-class jitter when a
    minority class has fewer than 2 members (no neighbor to interpolate toward) -- flagged honest,
    never fabricated as SMOTE."""
    k: int = 5
    target = "balance"
    name: str = "smote"

    def __init__(self, k: int = 5, target="balance"):
        self.k = int(k)
        self.target = target
        self.name = "smote"

    def apply(self, train_rows: List[dict], kind: str, rng: np.random.Generator) -> List[dict]:
        if kind != "classification":
            # Oversampling by class is undefined for regression; return unchanged (honest no-op).
            return [dict(r) for r in train_rows]
        X, y = _rows_to_xy(train_rows, kind)
        y = np.asarray([str(v) for v in y])
        counts = Counter(y.tolist())
        if not counts:
            return [dict(r) for r in train_rows]
        if self.target == "balance":
            goal = max(counts.values())
            targets = {c: goal for c in counts}
        else:
            targets = {c: int(self.target.get(c, counts[c])) for c in counts}

        out = [dict(r) for r in train_rows]
        from sklearn.neighbors import NearestNeighbors
        for cls, want in targets.items():
            have = counts[cls]
            need = want - have
            if need <= 0:
                continue
            cls_X = X[y == cls]
            if len(cls_X) == 1:
                # Single exemplar: cannot interpolate. Honest fallback = small gaussian jitter
                # around the lone point (labelled as a fallback, not as SMOTE interpolation).
                scale = 1e-6 + 0.01 * np.abs(cls_X[0])
                for _ in range(need):
                    synth = cls_X[0] + rng.normal(0.0, 1.0, size=cls_X.shape[1]) * scale
                    out.append(_row(synth, cls, kind))
                continue
            kk = min(self.k, len(cls_X) - 1)
            nn = NearestNeighbors(n_neighbors=kk + 1).fit(cls_X)
            _, idx = nn.kneighbors(cls_X)
            idx = idx[:, 1:]  # drop self
            for _ in range(need):
                i = int(rng.integers(len(cls_X)))
                j = int(idx[i, int(rng.integers(kk))])
                u = float(rng.random())
                synth = cls_X[i] + u * (cls_X[j] - cls_X[i])
                out.append(_row(synth, cls, kind))
        return out


@dataclass
class Mixup:
    """Mixup augmentation (Zhang et al., 2018), train-only.

    Emit convex combinations of pairs: x~ = lam*x_a + (1-lam)*x_b, lam ~ Beta(alpha, alpha).
    For classification we keep the label of the dominant parent (lam>=0.5) so targets stay
    valid discrete classes (soft labels are not certifiable through the accuracy bound). For
    regression we blend the targets too. Adds n_synth_frac * n new rows.

    Mixup is a regularizer: it smooths the decision function between examples. It is proposed
    as a hypothesis; whether it CAUSES a certified lift is decided by certify_data_change."""
    alpha: float = 0.2
    n_synth_frac: float = 0.5
    name: str = "mixup"

    def apply(self, train_rows: List[dict], kind: str, rng: np.random.Generator) -> List[dict]:
        X, y = _rows_to_xy(train_rows, kind)
        n = len(train_rows)
        out = [dict(r) for r in train_rows]
        if n < 2:
            return out
        n_synth = int(round(self.n_synth_frac * n))
        a = max(self.alpha, 1e-3)
        for _ in range(n_synth):
            ia, ib = rng.integers(n), rng.integers(n)
            lam = float(rng.beta(a, a))
            xa, xb = X[ia], X[ib]
            xm = lam * xa + (1.0 - lam) * xb
            if kind == "classification":
                tgt = y[ia] if lam >= 0.5 else y[ib]  # dominant parent's hard label
            else:
                tgt = lam * float(y[ia]) + (1.0 - lam) * float(y[ib])
            out.append(_row(xm, tgt, kind))
        return out


@dataclass
class GaussianClassSynthesize:
    """Gaussian (per-class multivariate-normal) synthetic sampling, train-only.

    Fit a Gaussian to each class's features (mean + covariance, shrinkage-regularized) and
    sample new rows from it -- a lightweight gaussian-copula-style synthesizer that respects
    each class's first/second moments. Used to top up minority classes when SMOTE's local
    interpolation is too conservative. Train-only.

    target as in SMOTEOversample: "balance" or {class: count}."""
    target = "balance"
    shrinkage: float = 0.1
    name: str = "gauss_synth"

    def __init__(self, target="balance", shrinkage: float = 0.1):
        self.target = target
        self.shrinkage = float(shrinkage)
        self.name = "gauss_synth"

    def apply(self, train_rows: List[dict], kind: str, rng: np.random.Generator) -> List[dict]:
        if kind != "classification":
            return [dict(r) for r in train_rows]
        X, y = _rows_to_xy(train_rows, kind)
        y = np.asarray([str(v) for v in y])
        counts = Counter(y.tolist())
        if not counts:
            return [dict(r) for r in train_rows]
        if self.target == "balance":
            goal = max(counts.values())
            targets = {c: goal for c in counts}
        else:
            targets = {c: int(self.target.get(c, counts[c])) for c in counts}

        out = [dict(r) for r in train_rows]
        d = X.shape[1]
        for cls, want in targets.items():
            need = want - counts[cls]
            if need <= 0:
                continue
            cls_X = X[y == cls]
            mu = cls_X.mean(axis=0)
            if len(cls_X) < 2:
                cov = np.eye(d) * (1e-6 + 0.01 * np.abs(mu).mean())
            else:
                cov = np.cov(cls_X, rowvar=False)
                cov = np.atleast_2d(cov)
                # Ledoit-Wolf-style shrinkage toward a diagonal target keeps cov PSD/invertible.
                diag = np.diag(np.diag(cov))
                cov = (1.0 - self.shrinkage) * cov + self.shrinkage * diag
                cov += np.eye(d) * 1e-9
            synth = rng.multivariate_normal(mu, cov, size=need)
            for s in synth:
                out.append(_row(s, cls, kind))
        return out


# =========================================================================== certified A/B

@dataclass
class ArmResult:
    """One arm of the data A/B (base or intervention): the winner + its sealed certificate."""
    label: str
    certified: bool
    certificate: Optional[dict]
    winner_label: Optional[str]
    winner_source: Optional[str]
    winner_val_score: Optional[float]
    n_train: int
    decline_reason: str = ""


@dataclass
class DataChangeReport:
    """Verdict of certify_data_change: did the DATA intervention CAUSE a certified lift?

    decision: "go" if caused_lift else "no-go".
    caused_lift: the certified verdict -- True iff the intervention arm's sealed LOWER BOUND
      strictly exceeds the base arm's sealed lower bound AND the intervention arm itself
      certifies (clears theta). Both numbers come from the frozen certifier; we only compare.
    The decision rule is stated up front (this is the experiment's pre-registered rule), and
    both arms' raw certificates are returned so the comparison is fully auditable.
    """
    hypothesis: str
    decision: str
    caused_lift: bool
    base: ArmResult
    intervention: ArmResult
    lower_bound_delta: Optional[float]
    decision_rule: str
    invariants_ok: bool
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lb_b = self.base.certificate.get("lower_bound") if self.base.certificate else None
        lb_i = self.intervention.certificate.get("lower_bound") if self.intervention.certificate else None
        return (f"[{self.decision.upper()}] {self.hypothesis}\n"
                f"  base: certified={self.base.certified} sealed_lb={lb_b} "
                f"(winner {self.base.winner_label}, n_train={self.base.n_train})\n"
                f"  intervention: certified={self.intervention.certified} sealed_lb={lb_i} "
                f"(winner {self.intervention.winner_label}, n_train={self.intervention.n_train})\n"
                f"  lower_bound_delta={self.lower_bound_delta}  caused_lift={self.caused_lift}\n"
                f"  rule: {self.decision_rule}  invariants_ok={self.invariants_ok}")


def _default_proposers(cfg: EngineConfig) -> List[ProposalSource]:
    """The SAME proposer set the engine builds by default. Holding this fixed across arms is
    what makes the A/B controlled: only the training data varies between base and intervention."""
    return [SeedProposer(), MutationProposer(), LLMProposer(cfg.llm_client)]


def _assert_train_only(intervened_train: Sequence[dict], val_rows: Sequence[dict],
                       sealed_rows: Sequence[dict]) -> None:
    """Hard invariant guard: no synthesized/changed train row may collide with a val or sealed
    feature vector. The originals already shared keys is impossible (split is disjoint); what we
    must rule out is a SYNTHETIC row landing on a held-out point. Raises on violation so a buggy
    intervention can never silently contaminate the certification set."""
    held = {_row_key(r) for r in val_rows} | {_row_key(r) for r in sealed_rows}
    for r in intervened_train:
        if _row_key(r) in held:
            raise AssertionError("data intervention produced a row colliding with val/sealed; "
                                 "synthetic data must never enter the held-out sets")


def _run_arm_on_split(task: Task, train_rows: Sequence[dict], splits: "certify.Splits",
                      cfg: EngineConfig, label: str) -> ArmResult:
    """Run the EXACT engine discipline on a fixed split with a (possibly intervened) train set:
    propose -> sandbox-fit on train_rows -> score on the SHARED val -> select -> certify the
    single winner on the SHARED, untouched sealed split (one counted peek via the frozen path).

    This mirrors frontier.engine.ResearchEngine.run but takes the train rows as an input so the
    base and intervention arms can share val/sealed verbatim. Every number is computed by
    certify.score_val / certify.certify_on_sealed (the sound wrappers); nothing here scores."""
    X_train, y_train = _rows_to_xy(train_rows, task.kind)
    X_val = Task.rows_to_X(splits.val_rows)

    proposers = _default_proposers(cfg)
    tried_labels: set = set()
    recent_errors: List[tuple] = []
    best_score: Optional[float] = None
    best_prog: Optional[Program] = None

    for r in range(cfg.rounds):
        context = {
            "task_kind": task.kind,
            "n_features": task.n_features,
            "n_train": len(train_rows),
            "round": r,
            "tried_labels": set(tried_labels),
            "best_label": best_prog.label if best_prog else None,
            "best_score": round(best_score, 4) if best_score is not None else None,
            "best_id": best_prog.id if best_prog else None,
            "best_recipe": (best_prog.provenance.get("recipe") if best_prog else None),
            "recent_errors": list(recent_errors),
        }
        proposals: List[Program] = []
        seen_ids = set()
        for src in proposers:
            for p in src.propose(context):
                if p.label in tried_labels or p.id in seen_ids:
                    continue
                seen_ids.add(p.id)
                proposals.append(p)
        if not proposals:
            break
        for p in proposals:
            res = sandbox.run_program(p, X_train, y_train, X_val, kind=task.kind,
                                      wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)
            tried_labels.add(p.label)
            if res.ok:
                score = certify.score_val(task, splits.val_rows, res.preds)
                if best_score is None or score > best_score:
                    best_score, best_prog = score, p
            else:
                recent_errors.append((p.label, res.error_kind, res.error))

    if best_prog is None:
        return ArmResult(label, False, None, None, None, None, len(train_rows),
                         decline_reason="no candidate executed successfully")

    # Certify the single winner on the SHARED sealed split (the only sealed peek for this arm).
    X_sealed = Task.rows_to_X(splits.sealed_rows)
    final = sandbox.run_program(best_prog, X_train, y_train, X_sealed, kind=task.kind,
                                wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)
    if not final.ok:
        return ArmResult(label, False, None, best_prog.label, best_prog.source,
                         round(best_score, 4), len(train_rows),
                         decline_reason=f"winner failed on sealed re-fit: [{final.error_kind}] {final.error}")

    cert = certify.certify_on_sealed(task, splits, final.preds)
    return ArmResult(label, bool(cert.get("certified")), cert, best_prog.label, best_prog.source,
                     round(best_score, 4), len(train_rows),
                     decline_reason=("" if cert.get("certified") else "sealed lower bound below theta"))


def certify_data_change(base_task: Task, intervention: DataIntervention, *,
                        config: Optional[EngineConfig] = None, seed: int = 0,
                        test_frac: float = 0.30, val_frac: float = 0.20) -> DataChangeReport:
    """Run a controlled A/B that answers: did `intervention` (a TRAIN-side data change) CAUSE a
    certified lift over the base data?

    Protocol (controlled experiment):
      1. split base_task ONCE -> the canonical train/val/sealed split;
      2. base arm  = engine discipline on the original train rows;
      3. intervention arm = engine discipline on intervention.apply(train_rows) -- val and sealed
         are the SAME rows as the base arm (shared, never synthesized into);
      4. compare the two arms' SEALED LOWER BOUNDS (both from the frozen certifier).

    Decision rule (pre-registered, stated in the report): GO iff the intervention arm certifies
    (clears theta on its sealed lower bound) AND its sealed lower bound strictly exceeds the base
    arm's sealed lower bound. Otherwise NO-GO. We never relax theta and never relabel a val score.

    Honest degradation: if a separate LLM proposer is desired, pass config.llm_client; with None
    the LLM arm is inactive and the offline seed/mutation proposers run (reported, not faked).
    """
    cfg = config or EngineConfig(seed=seed, test_frac=test_frac, val_frac=val_frac)
    rng = np.random.default_rng(seed)

    # 1. one canonical split, shared by both arms (this is what makes it controlled).
    splits = certify.make_splits(base_task, seed=cfg.seed, test_frac=test_frac, val_frac=val_frac)
    base_train = [dict(r) for r in splits.train_rows]

    # 2. apply the intervention to TRAIN ONLY, then HARD-ASSERT it never touched val/sealed.
    intervened_train = intervention.apply([dict(r) for r in splits.train_rows], base_task.kind, rng)
    notes: List[str] = []
    invariants_ok = True
    try:
        _assert_train_only(intervened_train, splits.val_rows, splits.sealed_rows)
    except AssertionError as e:
        invariants_ok = False
        notes.append(f"INVARIANT VIOLATION: {e}")
    # The val/sealed rows handed to both arms are literally the same objects -> provably unchanged.

    base_arm = _run_arm_on_split(base_task, base_train, splits, cfg, label="base")
    int_arm = _run_arm_on_split(base_task, intervened_train, splits, cfg,
                                label=f"intervention:{getattr(intervention, 'name', 'data')}")

    # 4. compare sealed lower bounds (both produced by the frozen certifier).
    lb_b = base_arm.certificate.get("lower_bound") if base_arm.certificate else None
    lb_i = int_arm.certificate.get("lower_bound") if int_arm.certificate else None
    delta = (float(lb_i) - float(lb_b)) if (lb_b is not None and lb_i is not None) else None

    rule = ("GO iff intervention arm certifies (sealed lower bound clears theta) AND its sealed "
            "lower bound strictly exceeds the base arm's sealed lower bound")
    caused_lift = bool(invariants_ok and int_arm.certified and delta is not None and delta > 0.0)

    return DataChangeReport(
        hypothesis=f"train-side '{getattr(intervention, 'name', 'data')}' improves sealed generalization",
        decision="go" if caused_lift else "no-go",
        caused_lift=caused_lift,
        base=base_arm,
        intervention=int_arm,
        lower_bound_delta=(round(delta, 6) if delta is not None else None),
        decision_rule=rule,
        invariants_ok=invariants_ok,
        notes=notes,
    )


def build_intervened_task(base_task: Task, intervention: DataIntervention, *, seed: int = 0,
                          test_frac: float = 0.30, val_frac: float = 0.20) -> Tuple[Task, "certify.Splits"]:
    """Materialize a NEW Task whose X/y is (intervened train rows) + (the ORIGINAL val + sealed
    rows). Returned alongside the split so a caller can hand the variant straight to a certifier
    that respects this exact split. WARNING: a plain ResearchEngine.run(task) RE-SPLITS the task,
    which would mix synthetic rows into val/sealed -- so for a clean A/B use certify_data_change,
    which shares the split. This helper exists for inspection and for harnesses that accept a
    pre-built split."""
    splits = certify.make_splits(base_task, seed=seed, test_frac=test_frac, val_frac=val_frac)
    rng = np.random.default_rng(seed)
    intervened_train = intervention.apply([dict(r) for r in splits.train_rows], base_task.kind, rng)
    _assert_train_only(intervened_train, splits.val_rows, splits.sealed_rows)

    new_rows = list(intervened_train) + list(splits.val_rows) + list(splits.sealed_rows)
    X = Task.rows_to_X(new_rows)
    y = Task.rows_to_y(new_rows, base_task.kind)
    variant = Task(X=X, y=y, kind=base_task.kind, theta=base_task.theta, metric=base_task.metric,
                   name=f"{base_task.name}+{getattr(intervention, 'name', 'data')}")
    return variant, splits
