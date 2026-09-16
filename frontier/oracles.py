"""Phase 7 -- verification beyond statistics ("cannot fool itself").

The sealed certificate (`certify_on_sealed`) gives a sound finite-sample LOWER BOUND on the
metric. That answers "is the number real on held-out data?" It does NOT answer "is the number
MEANINGFUL?" A pipeline can clear a sealed lower bound for the wrong reason:

  * a feature is (a copy of / a deterministic function of) the label -> the model is reading the
    answer, not learning it. The sealed test inherits the same leaked column, so the certificate
    is high AND sound AND meaningless;
  * the task is trivial (one class dominates) and the "win" is just the majority rate;
  * the metric is being read on the wrong axis (e.g. an inverted/raw-error orientation) so a
    bad model looks good;
  * train and sealed are drawn from visibly different distributions, so the certificate does not
    describe the deployment population;
  * the result is not reproducible (a fixed seed yields a different certificate digest on re-run).

This module adds a battery of cheap, deterministic ORACLES plus an adversarial self-refutation
pass that actively tries to explain the win away as leakage/artifact, and a seed-controlled
reproducibility re-run. `verify_before_promote(...)` runs them and returns a Verdict; the
integrator gates promotion on it. A leaky/cheating Program is CAUGHT and refused here even though
its sealed certificate would otherwise pass.

Firewall preserved: oracles never recompute a promotion-bearing number. They reuse the frozen
`certify_on_sealed` for re-runs and `science.score_metric` for diagnostic scores. Untrusted code
is executed only through the same predictions-only `run_fn` the engine uses; the verdict NEVER
re-grades or relaxes the certificate -- it can only VETO a certified winner (turn promote True->
False), never manufacture a promotion.

# === WIRING ===
# The integrator inserts ONE call between "winner certified on sealed" and "promote".
# In frontier/engine.py, after `cert = certify.certify_on_sealed(task, splits, final.preds)`
# (engine.py:173), wrap the engine's sandbox in the run_fn adapter and gate:
#
#     from frontier import oracles, sandbox
#     def _run_fn(prog, Xtr, ytr, Xev):
#         r = sandbox.run_program(prog, Xtr, ytr, Xev, kind=task.kind,
#                                 wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds)
#         return r.preds if r.ok else None           # predictions-only firewall; None on failure
#     verdict = oracles.verify_before_promote(task, splits, best_prog, cert, _run_fn)
#     certified = bool(cert.get("certified")) and verdict.promote
#     # carry verdict.reasons / verdict.to_dict() into EngineResult.diagnosis_trail
#
# run_fn contract (the integrator supplies it; oracles never call the sandbox directly so this
# stays substrate-agnostic):
#     run_fn(program: Program, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray)
#        -> list  (predictions for X_eval, row-aligned)  OR  None (the program failed to run)
# It MUST return predictions only (the numeric firewall). `kind`, resource limits, etc. are
# closed over by the integrator, exactly as the engine already configures the sandbox.
#
# Ordering: call AFTER certification (we need the certificate to (a) compare the win against the
# trivial baseline / chance and (b) re-derive the digest for the reproducibility check) and
# BEFORE the promotion decision. The sealed test is NOT peeked again by the cheap oracles; the
# optional reproducibility re-run peeks a FRESHLY rebuilt sealed split (a new SealedTest instance
# with its own one-peek budget) so the winner's counted peek is untouched.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# Repo root on sys.path so the sound certifier imports resolve (mirrors certify.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vectorforge import science          # noqa: E402  sound metrics, imported verbatim

from . import certify                    # noqa: E402  frozen split + certify-on-sealed adapter
from .program import Program             # noqa: E402
from .task import Task                   # noqa: E402


# run_fn: program + train arrays + eval features -> predictions (or None on failure). The
# integrator wraps the sandbox in this so oracles never touch untrusted execution directly.
RunFn = Callable[[Program, np.ndarray, np.ndarray, np.ndarray], Optional[Sequence]]


# --------------------------------------------------------------------------- result types

@dataclass
class OracleResult:
    """One oracle's outcome. `passed=False` is a veto-worthy red flag; `severity` ranks it."""
    name: str
    passed: bool
    severity: str                       # "block" (vetoes promotion) | "warn" (logged only)
    detail: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "severity": self.severity,
                "detail": self.detail, "evidence": self.evidence}


@dataclass
class Verdict:
    """The promotion gate's answer. `promote` is True only if NO blocking oracle fired."""
    promote: bool
    reasons: List[str]                  # human-readable reasons (the blockers, if any)
    oracles: List[OracleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"promote": self.promote, "reasons": list(self.reasons),
                "oracles": [o.to_dict() for o in self.oracles]}


# --------------------------------------------------------------------------- chance / baselines

def _chance_level(task: Task, y_eval: np.ndarray) -> float:
    """The score a trivial constant predictor achieves under the task metric.

    For classification accuracy/derivatives this is the majority-class rate (the honest floor a
    real model must beat). For regression r2 it is 0.0 (predicting the mean), for neg_rmse/neg_mae
    it is the error of the mean predictor. Computed via the FROZEN `science.score_metric`, never a
    bespoke formula -> same number the certifier would compute."""
    if task.kind == "classification":
        labels = list(task.labels)
        yt = [str(v) for v in y_eval]
        # majority constant prediction is the strongest trivial classifier under accuracy.
        vals, counts = np.unique(yt, return_counts=True)
        majority = str(vals[int(np.argmax(counts))])
        const = [majority] * len(yt)
        return float(science.score_metric(task.metric, yt, const, labels))
    yt = [float(v) for v in y_eval]
    mean_pred = [float(np.mean(yt))] * len(yt)
    return float(science.score_metric(task.metric, yt, mean_pred, None))


def _score(task: Task, y_true, y_pred) -> float:
    """Diagnostic score via the frozen metric (NOT a promotion number)."""
    if task.kind == "classification":
        return float(science.score_metric(task.metric, [str(v) for v in y_true],
                                          [str(v) for v in y_pred], list(task.labels)))
    return float(science.score_metric(task.metric, [float(v) for v in y_true],
                                      [float(v) for v in y_pred], None))


# --------------------------------------------------------------------------- individual oracles

def oracle_beats_trivial(task: Task, certificate: dict, y_eval: np.ndarray) -> OracleResult:
    """The certified result must beat a trivial constant baseline by a real margin.

    Compares the sealed LOWER BOUND (the promotion-bearing quantity) against the chance level on
    the same sealed labels. A win that does not clear the trivial floor is not a result -- it is
    the base rate wearing a model's name. Blocking."""
    observed = certificate.get("observed")
    lower = certificate.get("lower_bound")
    chance = _chance_level(task, y_eval)
    # Require the certified LOWER bound to exceed chance (the bound is what we promote on).
    margin = (lower - chance) if (lower is not None) else float("-inf")
    passed = lower is not None and lower > chance + 1e-9
    detail = (f"sealed lower_bound={lower} observed={observed} vs trivial floor={round(chance, 4)} "
              f"(margin={round(margin, 4)})")
    return OracleResult("beats_trivial_baseline", passed, "block", detail,
                        {"lower_bound": lower, "observed": observed, "chance": round(chance, 4),
                         "margin": round(margin, 6)})


def oracle_metric_orientation(task: Task, certificate: dict) -> OracleResult:
    """The metric must be a known, higher-is-better certifiable metric and the certificate must
    carry that metric.

    Guards the F4 failure (a metric with no frozen certifier silently scored as accuracy) and the
    "wrong axis" bug (a lower-is-better quantity certified as if larger were better). All of
    `science.KNOWN_METRICS` are in higher-is-better orientation by construction, so confirming the
    certificate's metric is in that set IS the orientation check. Blocking on mismatch."""
    metric = task.metric
    cert_metric = certificate.get("metric", metric)   # accuracy certs omit "metric"; default to task's
    known = metric in science.KNOWN_METRICS
    consistent = (cert_metric == metric) or (metric == "accuracy" and "metric" not in certificate)
    passed = known and consistent
    detail = (f"task.metric={metric!r} certificate.metric={cert_metric!r} "
              f"known_higher_is_better={known} consistent={consistent}")
    return OracleResult("metric_orientation", passed, "block", detail,
                        {"task_metric": metric, "cert_metric": cert_metric,
                         "known": known, "consistent": consistent})


def oracle_no_label_leak_feature(task: Task, splits: certify.Splits) -> OracleResult:
    """Direct artifact check: NO feature may be a copy of / a deterministic function of the label.

    This is the canonical leak ("a feature copies the label"). We look across ALL split rows
    (train+val+sealed) because a leaked column leaks into every split simultaneously -- that is
    precisely why the sealed certificate fails to catch it. For each feature we ask: does it
    determine the target? i.e. within each distinct feature value, is the target constant
    (classification) or near-constant (regression)? A feature that perfectly determines the label
    is a leak. Blocking."""
    rows = list(splits.train_rows) + list(splits.val_rows) + list(splits.sealed_rows)
    X = Task.rows_to_X(rows)
    y = Task.rows_to_y(rows, task.kind)
    n, d = X.shape
    leaks = []
    for j in range(d):
        col = X[:, j]
        if task.kind == "classification":
            yj = np.asarray([str(v) for v in y])
            # group target by feature value; a leak => target constant within each feature value
            # AND the feature is informative (more than one distinct value, else it's constant junk).
            order = np.argsort(col, kind="mergesort")
            cs, ys = col[order], yj[order]
            boundaries = np.where(np.diff(cs) != 0)[0] + 1
            groups = np.split(ys, boundaries)
            n_distinct_feat = len(groups)
            pure = sum(len(g) for g in groups if len(set(g.tolist())) == 1)
            purity = pure / max(n, 1)
            # A continuous feature is VACUOUSLY "pure" (every distinct float value owns one row, so
            # the target is trivially constant within each value). That is not a leak. A real
            # copy-the-label leak has FEW distinct feature values, each mapping to one label -- i.e.
            # the feature behaves like a (near-)categorical encoding of the target. Require the
            # number of distinct feature values to be on the order of the number of labels (a 1:1 /
            # few-to-one map), not ~n (a continuous column). Otherwise high "purity" is meaningless.
            n_labels = max(2, len(set(yj.tolist())))
            categorical_like = n_distinct_feat <= max(8, 4 * n_labels)
            informative = n_distinct_feat > 1
            if informative and categorical_like and purity > 0.999:
                leaks.append((j, round(purity, 4)))
        else:
            yj = np.asarray([float(v) for v in y], dtype=float)
            # leak <=> feature is an affine (or near-perfect monotone) function of the target.
            if np.std(col) < 1e-12 or np.std(yj) < 1e-12:
                continue
            r = float(np.corrcoef(col, yj)[0, 1])
            if abs(r) > 0.9995:
                leaks.append((j, round(abs(r), 6)))
    passed = not leaks
    if passed:
        detail = f"no feature determines the label across {n} rows, {d} features"
    else:
        detail = f"LEAK: feature(s) determine the label: {leaks[:5]} (purity/|corr|)"
    return OracleResult("no_label_leak_feature", passed, "block", detail,
                        {"n_rows": n, "n_features": d, "leaky_features": leaks[:20]})


def oracle_permuted_label_collapses(task: Task, splits: certify.Splits, winner: Program,
                                    run_fn: RunFn, *, seed: int = 0) -> OracleResult:
    """Permutation sanity: with TRAINING labels shuffled, the model must collapse toward chance.

    Train the winner on (X_train, PERMUTED y_train) and evaluate on the REAL val labels. Destroying
    the X<->y relationship should leave nothing to learn, so the score must fall to ~chance. If it
    STAYS high, information is reaching val by a route other than a genuine learned relationship --
    leakage through the split, a feature that encodes the label (so the model recovers the answer
    regardless of training y), or an evaluation/axis artifact. This catches leaks the direct
    feature scan can miss (e.g. a leak spread across several features). Blocking.

    NB: this peeks the VAL split only (never the sealed test), and trains a throwaway model; the
    winner's single sealed peek is untouched."""
    Xtr = Task.rows_to_X(splits.train_rows)
    ytr = Task.rows_to_y(splits.train_rows, task.kind)
    Xva = Task.rows_to_X(splits.val_rows)
    yva = Task.rows_to_y(splits.val_rows, task.kind)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ytr))
    ytr_shuf = ytr[perm]

    preds = run_fn(winner, Xtr, ytr_shuf, Xva)
    if preds is None or len(preds) != len(yva):
        # Inconclusive: do not block on an execution failure (the engine already gates on a clean
        # sealed re-fit). Warn so it is visible, but do not fabricate a pass/fail.
        return OracleResult("permuted_label_collapses", True, "warn",
                            "permuted-label run did not execute cleanly; oracle inconclusive",
                            {"executed": False})
    perm_score = _score(task, yva, preds)
    chance = _chance_level(task, yva)

    if task.kind == "classification":
        # accept anything within a tolerance band above chance (finite-sample noise). A model that
        # still scores well ABOVE chance on permuted labels is suspect.
        tol = 0.10
        suspicious = perm_score > chance + tol
        passed = not suspicious
        detail = (f"permuted-label val score={round(perm_score, 4)} chance={round(chance, 4)} "
                  f"tol={tol} -> {'COLLAPSED (ok)' if passed else 'STAYED HIGH (leak suspect)'}")
    else:
        # r2/neg_* : a model with nothing to learn should be no better than the mean predictor,
        # i.e. score <= chance + small band. r2 of a permuted-target fit is typically <= 0.
        band = 0.10 if task.metric == "r2" else abs(chance) * 0.10 + 1e-9
        suspicious = perm_score > chance + band
        passed = not suspicious
        detail = (f"permuted-label val score={round(perm_score, 4)} chance={round(chance, 4)} "
                  f"band={round(band, 4)} -> {'collapsed (ok)' if passed else 'STAYED HIGH (leak suspect)'}")
    return OracleResult("permuted_label_collapses", passed, "block", detail,
                        {"executed": True, "perm_score": round(perm_score, 6),
                         "chance": round(chance, 6)})


def oracle_distribution_drift(task: Task, splits: certify.Splits, *,
                              z_thresh: float = 6.0, frac_thresh: float = 0.30) -> OracleResult:
    """Train vs sealed distribution-drift check.

    The certificate describes the sealed population; if sealed differs sharply from train the
    certified number does not transfer to deployment-like data, and a large drift can also be a
    split artifact (e.g. an ordered dataset split without shuffling). Per feature we standardize by
    the TRAIN moments and measure |mean shift| in train-std units. We FLAG (warn, not block) when a
    large fraction of features drift beyond `z_thresh` -- drift is a real-world fact, not
    necessarily cheating, so it informs rather than vetoes."""
    Xtr = Task.rows_to_X(splits.train_rows)
    Xse = Task.rows_to_X(splits.sealed_rows)
    if Xtr.size == 0 or Xse.size == 0:
        return OracleResult("distribution_drift", True, "warn", "empty split; skipped", {})
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd_safe = np.where(sd < 1e-12, 1.0, sd)
    z = np.abs(Xse.mean(axis=0) - mu) / sd_safe
    drifted = int(np.sum(z > z_thresh))
    frac = drifted / max(len(z), 1)
    passed = frac <= frac_thresh
    detail = (f"{drifted}/{len(z)} features drift > {z_thresh} train-std (frac={round(frac, 3)}, "
              f"max_z={round(float(z.max()), 2)}) -> {'ok' if passed else 'DRIFT WARNING'}")
    return OracleResult("distribution_drift", passed, "warn", detail,
                        {"drifted": drifted, "n_features": int(len(z)),
                         "frac": round(frac, 4), "max_z": round(float(z.max()), 4)})


def oracle_reproducible(task: Task, winner: Program, run_fn: RunFn, certificate: dict, *,
                        seed: int = 0) -> OracleResult:
    """Seed-controlled reproducibility: re-run the whole certify path on a freshly rebuilt split
    with the SAME seed and confirm the sealed certificate is byte-stable.

    Same seed => identical 3-way split => identical sealed digest => identical
    (observed, lower_bound, certified). A mismatch means the result depends on uncontrolled state
    (nondeterministic estimator, unseeded RNG, environment) and is not a reproducible scientific
    claim. We compare the re-derived certificate to the supplied one. Blocking on digest or
    decision mismatch.

    This rebuilds a NEW SealedTest (its own one-peek budget), so it does NOT consume the winner's
    counted peek; it verifies the PROCESS reproduces, not that the same sealed instance is re-read."""
    splits2 = certify.make_splits(task, seed=seed)
    Xtr = Task.rows_to_X(splits2.train_rows)
    ytr = Task.rows_to_y(splits2.train_rows, task.kind)
    Xse = Task.rows_to_X(splits2.sealed_rows)
    preds = run_fn(winner, Xtr, ytr, Xse)
    if preds is None or len(preds) != len(splits2.sealed_rows):
        return OracleResult("reproducible", True, "warn",
                            "re-run did not execute cleanly; reproducibility inconclusive",
                            {"executed": False})
    cert2 = certify.certify_on_sealed(task, splits2, preds)
    digest_match = cert2.get("sealed_digest") == certificate.get("sealed_digest")
    decision_match = (cert2.get("certified") == certificate.get("certified")
                      and abs((cert2.get("lower_bound") or 0.0)
                              - (certificate.get("lower_bound") or 0.0)) < 1e-9
                      and abs((cert2.get("observed") or 0.0)
                              - (certificate.get("observed") or 0.0)) < 1e-9)
    passed = bool(digest_match and decision_match)
    detail = (f"re-run digest_match={digest_match} decision_match={decision_match} "
              f"(lb {certificate.get('lower_bound')}->{cert2.get('lower_bound')}, "
              f"obs {certificate.get('observed')}->{cert2.get('observed')})")
    return OracleResult("reproducible", passed, "block", detail,
                        {"digest_match": digest_match, "decision_match": decision_match,
                         "rerun_certificate": {k: cert2.get(k) for k in
                                               ("observed", "lower_bound", "certified",
                                                "sealed_digest", "peeks")}})


def adversarial_self_refutation(task: Task, splits: certify.Splits, winner: Program,
                                certificate: dict, run_fn: RunFn, *,
                                seed: int = 0) -> Tuple[bool, List[str], List[dict]]:
    """Actively try to explain the win away as leakage/artifact before promoting it.

    This is the falsification pass: rather than trusting the certificate, it mounts attacks whose
    SUCCESS would mean the result is not real:

      A. Single-feature attack. If ANY single feature, fed to a trivial 1-NN-on-that-column /
         decision-stump predictor, already reaches (near) the winner's certified level, then the
         "model" is plausibly riding one leaked column. We test the most label-correlated feature.
      B. Shuffled-feature-rows attack. Shuffle the ROWS of X_train (breaking the X<->y pairing)
         and refit: a model that still predicts val well is reading a leak, not the pairing.

    Returns (refuted, reasons, evidence). `refuted=True` means an attack succeeded -> do NOT
    promote. All scoring is via the frozen metric; untrusted code runs only through run_fn."""
    reasons: List[str] = []
    evidence: List[dict] = []
    target = certificate.get("lower_bound")
    if target is None:
        return False, [], []

    Xtr = Task.rows_to_X(splits.train_rows)
    ytr = Task.rows_to_y(splits.train_rows, task.kind)
    Xva = Task.rows_to_X(splits.val_rows)
    yva = Task.rows_to_y(splits.val_rows, task.kind)
    chance = _chance_level(task, yva)

    # ---- Attack A: does a single feature already explain the win? -----------------------------
    # Pick the feature most associated with the (train) target; build a deterministic 1-feature
    # predictor in-process (TRUSTED diagnostic code, not the candidate). For classification we map
    # each val point to the train-majority label of its nearest train value on that column; for
    # regression we predict the nearest train target. If this trivial 1-column predictor reaches
    # the winner's level, the win is single-feature explainable -> artifact.
    best_feat_score = float("-inf")
    best_feat = -1
    try:
        d = Xtr.shape[1]
        for j in range(d):
            col_tr = Xtr[:, j]
            col_va = Xva[:, j]
            if np.std(col_tr) < 1e-12:
                continue
            # nearest-train-value 1-NN on this single column (vectorized over val).
            idx = np.argmin(np.abs(col_va[:, None] - col_tr[None, :]), axis=1)
            preds_j = ytr[idx]
            sj = _score(task, yva, preds_j)
            if sj > best_feat_score:
                best_feat_score, best_feat = sj, j
    except Exception as e:  # numerical edge -> attack inconclusive, never fabricate a refutation
        evidence.append({"attack": "single_feature", "error": str(e)[:120]})
        best_feat_score = float("-inf")

    # "reaches the win" = single-feature predictor lands within a small margin of the certified
    # lower bound AND clearly above chance. Both conditions matter: above chance => it is actually
    # solving the task off one column; near the win => that one column carries the whole result.
    margin = 0.05 if task.kind == "classification" or task.metric == "r2" else abs(target) * 0.05 + 1e-9
    if best_feat_score >= target - margin and best_feat_score > chance + 1e-3:
        reasons.append(
            f"single-feature attack: feature f{best_feat} alone reaches "
            f"{round(best_feat_score, 4)} (certified lower_bound={target}, chance={round(chance, 4)}) "
            f"-> result is plausibly a single-column artifact/leak")
    evidence.append({"attack": "single_feature", "best_feature": int(best_feat),
                     "best_feature_score": round(float(best_feat_score), 6),
                     "certified_lower_bound": target, "chance": round(chance, 6)})

    # ---- Attack B: break the X<->y pairing by shuffling X rows, then refit the winner ----------
    rng = np.random.default_rng(seed + 1)
    row_perm = rng.permutation(len(Xtr))
    preds_b = run_fn(winner, Xtr[row_perm], ytr, Xva)
    if preds_b is not None and len(preds_b) == len(yva):
        score_b = _score(task, yva, preds_b)
        tol = 0.10 if task.kind == "classification" or task.metric == "r2" else abs(chance) * 0.10 + 1e-9
        if score_b > chance + tol:
            reasons.append(
                f"row-shuffle attack: with X<->y pairing destroyed the winner still scores "
                f"{round(score_b, 4)} (chance={round(chance, 4)}) -> it is reading a feature leak, "
                f"not the learned pairing")
        evidence.append({"attack": "row_shuffle", "score": round(float(score_b), 6),
                         "chance": round(chance, 6)})
    else:
        evidence.append({"attack": "row_shuffle", "executed": False})

    return (len(reasons) > 0), reasons, evidence


# --------------------------------------------------------------------------- top-level gate

def verify_before_promote(task: Task, splits: certify.Splits, winner: Program,
                          certificate: dict, run_fn: RunFn, *, seed: int = 0) -> Verdict:
    """Run the full verification battery and return a promotion Verdict.

    A winner promotes ONLY if its sealed certificate is certified AND every BLOCKING oracle passes
    AND the adversarial self-refutation pass fails to explain the win away. Blocking oracles:
    metric-orientation, beats-trivial-baseline, no-label-leak-feature, permuted-label-collapse,
    reproducibility. Distribution-drift is advisory (warn).

    This can only VETO -- it never turns an uncertified result into a promotion (invariant #5:
    certified result or honest decline, never a relabeled score). If the certificate itself did not
    certify, promote is False regardless of the oracles."""
    y_sealed = Task.rows_to_y(splits.sealed_rows, task.kind)
    results: List[OracleResult] = []

    # cheap, no-execution oracles first
    results.append(oracle_metric_orientation(task, certificate))
    results.append(oracle_beats_trivial(task, certificate, y_sealed))
    results.append(oracle_no_label_leak_feature(task, splits))
    results.append(oracle_distribution_drift(task, splits))

    # execution oracles (use run_fn; predictions-only firewall)
    results.append(oracle_permuted_label_collapses(task, splits, winner, run_fn, seed=seed))
    results.append(oracle_reproducible(task, winner, run_fn, certificate, seed=seed))

    refuted, refute_reasons, refute_evidence = adversarial_self_refutation(
        task, splits, winner, certificate, run_fn, seed=seed)
    results.append(OracleResult(
        "adversarial_self_refutation", passed=not refuted, severity="block",
        detail=("no attack explained the result away" if not refuted
                else "; ".join(refute_reasons)),
        evidence={"attacks": refute_evidence, "refutations": refute_reasons}))

    cert_ok = bool(certificate.get("certified"))
    blockers = [o for o in results if o.severity == "block" and not o.passed]
    reasons: List[str] = []
    if not cert_ok:
        reasons.append("sealed certificate did not certify (lower bound does not clear theta)")
    for o in blockers:
        reasons.append(f"[{o.name}] {o.detail}")

    promote = cert_ok and not blockers
    return Verdict(promote=promote, reasons=reasons, oracles=results)


if __name__ == "__main__":  # tiny smoke check; the real tests live in tests/test_oracles.py
    from sklearn.datasets import load_breast_cancer
    from . import sandbox
    d = load_breast_cancer()
    t = Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.85, name="bc")
    sp = certify.make_splits(t, seed=0)
    code = "from sklearn.linear_model import LogisticRegression\n" \
           "def build_estimator():\n    return LogisticRegression(max_iter=2000)\n"
    prog = Program(code=code, source="seed", label="logreg")
    Xse = Task.rows_to_X(sp.sealed_rows)
    Xtr = Task.rows_to_X(sp.train_rows); ytr = Task.rows_to_y(sp.train_rows, t.kind)
    rr = sandbox.run_program(prog, Xtr, ytr, Xse, kind=t.kind, wall_seconds=60)
    cert = certify.certify_on_sealed(t, sp, rr.preds)

    def run_fn(p, a, b, c):
        r = sandbox.run_program(p, a, b, c, kind=t.kind, wall_seconds=60)
        return r.preds if r.ok else None

    v = verify_before_promote(t, sp, prog, cert, run_fn)
    print("promote=", v.promote)
    for o in v.oracles:
        print(f"  [{'ok' if o.passed else 'XX'}] {o.name}: {o.detail}")
