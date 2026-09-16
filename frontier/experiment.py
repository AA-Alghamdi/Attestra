"""Phase 5: scientific experiment design (the lead -> junior "is this idea a go?" pattern).

The Phase-0 spine answers "which Program scored highest, and does it certify?". That is a
*ranking* question. A scientist asks a different, sharper question: "does THIS change
(this factor) actually cause a lift, holding everything else fixed?" -- an ABLATION, with a
hypothesis, a single varied factor, controls held fixed, and a decision rule stated *before*
the data is touched. The owner's example (does TTS tone-dimensionality help?) is exactly this
shape: it is not a leaderboard row, it is a controlled A/B with a go/no-go verdict.

This module builds that object and runs it honestly through the frozen certifier.

What "honest" means here, concretely
------------------------------------
A controlled comparison needs the sealed test scored for BOTH arms (treatment + control).
That is TWO peeks of the SAME locked test. The audited SealedTest counts every peek and the
frozen certifier pays Bonferroni multiplicity for the realized peek count: the first arm
certifies at checks=1, the second at checks=2 (the cumulative count of the shared instance).
We do NOT launder this by certifying each arm against a fresh checks=1 test -- that would be
the exact moat-laundering bug `vfplatform/sealed.py` was written to stop. The cost of asking
two questions of one locked test is paid in full, and both certificates report their `checks`.

The verdict is a contrast of the two *certified lower bounds* (the conservative, defensible
quantity), not of point estimates. A factor is a "go" only if the treatment's sealed lower
bound exceeds the control's sealed lower bound by a pre-registered margin. A beneficial factor
(e.g. standardization for an RBF-SVM) clears it; a no-op factor (e.g. a redundant duplicate
preprocessing step, or scaling for a scale-invariant tree) does not -- and the no-go is just
as much a result as the go (negative results are first-class, per the project rules).

# === WIRING ===
# The integrator wires this ALONGSIDE ResearchEngine, not inside its promote path (the sealed
# test is touched once per QUESTION; an experiment asks two questions, so it owns its own
# SealedTest instance with max_peeks raised to the number of arms -- a deliberate, documented
# spec choice, NOT a relaxation of the per-arm correction, which is still paid via `checks`).
#
# Typical call site (e.g. from engine.py's caller, or a Phase-6 portfolio):
#
#   from frontier.experiment import (Experiment, run_experiment, run_ablation,
#                                     factor_program, scale_factor)
#   from frontier import certify
#
#   splits = certify.make_splits(task, seed=cfg.seed,
#                                test_frac=cfg.test_frac, val_frac=cfg.val_frac)
#   exp = Experiment(
#       hypothesis="Standardizing features helps the RBF-SVM on this task.",
#       factor_varied="standardize_features",
#       controls_held_fixed={"base": "svc_rbf", "estimator_hyperparams": "fixed",
#                            "splits": "shared", "seed": cfg.seed},
#       decision_rule="go iff treatment sealed lower bound > control sealed lower bound + 0.0",
#   )
#   treatment, control = scale_factor(task.kind, base="svc_rbf")   # differ ONLY in scaling
#   verdict = run_experiment(exp, task, splits, treatment, control,
#                            sandbox_run=None,        # defaults to frontier.sandbox.run_program
#                            wall_seconds=cfg.wall_seconds, cpu_seconds=cfg.cpu_seconds,
#                            margin=0.0)
#   print(verdict.report())
#
# An ablation over levels of a factor (each level is a treatment vs the SAME control):
#
#   levels = {"none": control_prog, "scale": scaled_prog, "scale+poly2": scaled_poly_prog}
#   abl = run_ablation(exp, task, splits, baseline=control_prog, levels=levels, ...)
#
# Arguments / shapes the integrator must respect:
#   - run_experiment(exp, task, splits, treatment, control, ...) -> Verdict
#       task    : frontier.task.Task          (the goal + data)
#       splits  : frontier.certify.Splits     (SAME splits for both arms -- the control)
#       treatment, control : frontier.program.Program (differ ONLY in `factor_varied`)
#   - sandbox_run : Callable matching frontier.sandbox.run_program (firewall: returns
#                   predictions only). Pass None to use the real out-of-process sandbox.
#   - The module NEVER recomputes a promotion-bearing number; every sealed number comes from
#     vfplatform.sealed.certify_on_sealed (the frozen, audited path).
#   - LLM hook: design_experiment(goal, factor, client) lets a frontier model AUTHOR the
#     hypothesis/decision-rule text from a goal string; with client=None it degrades honestly
#     to a deterministic template (no fabricated reasoning).
# === END WIRING ===
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Repo root on sys.path so the frozen certifier imports resolve when this module is imported
# directly (mirrors certify.py's bootstrap).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vfplatform import sealed              # noqa: E402  (frozen, audited; the only promoter)

from . import certify                      # noqa: E402
from . import sandbox as _sandbox          # noqa: E402
from .program import Program               # noqa: E402
from .proposers import make_code           # noqa: E402
from .task import Task                      # noqa: E402


# --------------------------------------------------------------------------- the experiment

@dataclass
class Experiment:
    """A pre-registered controlled comparison.

    Fields mirror how a careful scientist scopes an ablation BEFORE looking at the data:

      hypothesis          : the directional claim under test, in plain language.
      factor_varied       : the single thing that differs between treatment and control.
      controls_held_fixed : everything that is the SAME across arms (base estimator,
                            hyperparameters, splits, seed). This is what makes the
                            comparison clean -- any lift is attributable to the factor.
      decision_rule       : the go/no-go rule, stated up front so the verdict cannot be
                            rationalized after seeing the numbers.

    Stating these before execution is the anti-p-hacking discipline: the decision rule and the
    multiplicity budget are fixed before the sealed test is touched.
    """

    hypothesis: str
    factor_varied: str
    controls_held_fixed: Dict[str, object]
    decision_rule: str
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.hypothesis:
            raise ValueError("an experiment needs a hypothesis")
        if not self.factor_varied:
            raise ValueError("an experiment needs exactly one varied factor")


@dataclass
class ArmResult:
    """One arm's outcome: its program, whether it ran, and its sealed certificate."""

    name: str                       # "treatment" | "control" | a factor-level label
    program_id: str
    label: str
    ok: bool
    certificate: Optional[dict] = None      # the frozen sealed certificate (or None on failure)
    sealed_preds: Optional[list] = None     # predictions on the sealed rows (parent scored them)
    error_kind: str = ""
    error: str = ""

    @property
    def lower_bound(self) -> Optional[float]:
        if self.certificate is None:
            return None
        return self.certificate.get("lower_bound")

    @property
    def observed(self) -> Optional[float]:
        if self.certificate is None:
            return None
        return self.certificate.get("observed")


@dataclass
class Verdict:
    """The go/no-go answer with its evidence."""

    go: bool
    experiment: Experiment
    treatment: ArmResult
    control: ArmResult
    margin: float
    lift_lower_bound: Optional[float]       # treatment.lower_bound - control.lower_bound
    lift_observed: Optional[float]          # treatment.observed   - control.observed
    paired_lift_lower: Optional[float]      # paired-bootstrap lower bound on the per-sample lift
    reasoning: str
    checks: int                             # realized sealed peek count (multiplicity paid)
    sealed_digest: str = ""

    def report(self) -> str:
        verdict = "GO" if self.go else "NO-GO"
        lines = [
            f"[{verdict}] {self.experiment.hypothesis}",
            f"  factor varied      : {self.experiment.factor_varied}",
            f"  controls held fixed: {self.experiment.controls_held_fixed}",
            f"  decision rule      : {self.experiment.decision_rule}",
            f"  treatment ({self.treatment.label}): "
            f"obs={_fmt(self.treatment.observed)} lb={_fmt(self.treatment.lower_bound)}"
            + ("" if self.treatment.ok else f"  [FAILED: {self.treatment.error_kind}]"),
            f"  control   ({self.control.label}): "
            f"obs={_fmt(self.control.observed)} lb={_fmt(self.control.lower_bound)}"
            + ("" if self.control.ok else f"  [FAILED: {self.control.error_kind}]"),
            f"  lift (observed)    : {_fmt(self.lift_observed)}",
            f"  lift (lower-bound) : {_fmt(self.lift_lower_bound)}  (margin={self.margin})",
            f"  paired-lift lower  : {_fmt(self.paired_lift_lower)}",
            f"  sealed peeks paid  : checks={self.checks} (Bonferroni-corrected)",
            f"  reasoning          : {self.reasoning}",
        ]
        return "\n".join(lines)


def _fmt(v) -> str:
    return "n/a" if v is None else f"{float(v):.4f}"


# --------------------------------------------------------------------------- factor builders
# Helpers that produce a (treatment, control) PAIR differing in EXACTLY ONE factor, with all
# other recipe knobs identical. These are the disciplined way to construct an experiment's two
# arms so the "controls held fixed" claim is literally true at the code level. They are seeds /
# conveniences; an LLM (or the integrator) may supply arbitrary Program pairs instead.

def factor_program(kind: str, recipe: dict, source: str = "experiment",
                   label_suffix: str = "") -> Program:
    """Render one arm's Program from a recipe dict (reusing the frozen make_code path)."""
    from .proposers import recipe_label
    code = make_code(recipe, kind)
    lab = recipe_label(recipe) + (f":{label_suffix}" if label_suffix else "")
    return Program(code=code, source=source, label=lab, provenance={"recipe": dict(recipe)})


def scale_factor(kind: str, base: str = "svc_rbf") -> Tuple[Program, Program]:
    """Treatment = base WITH standardization; control = base WITHOUT. Everything else fixed.

    Beneficial for distance/kernel models (SVC-RBF, logreg); a no-op for scale-invariant trees
    (hist_gbm, rf) -- which is exactly why it is a good positive/negative control pair.
    """
    treatment = factor_program(kind, {"base": base, "scale": True}, label_suffix="treat")
    control = factor_program(kind, {"base": base}, label_suffix="ctrl")
    return treatment, control


def poly_factor(kind: str, base: str, degree: int = 2) -> Tuple[Program, Program]:
    """Treatment adds polynomial interaction features; control does not. Base fixed."""
    treatment = factor_program(kind, {"base": base, "scale": True, "poly": degree},
                               label_suffix="treat")
    control = factor_program(kind, {"base": base, "scale": True}, label_suffix="ctrl")
    return treatment, control


def noop_factor(kind: str, base: str = "rf") -> Tuple[Program, Program]:
    """A deliberately INERT factor: scaling a scale-invariant tree.

    StandardScaler is a monotone affine per-feature transform; tree splits are invariant to it,
    so treatment and control are statistically identical up to noise. The decision rule must
    return NO-GO. This is the negative control the spec asks for.
    """
    treatment = factor_program(kind, {"base": base, "scale": True}, label_suffix="treat")
    control = factor_program(kind, {"base": base}, label_suffix="ctrl")
    return treatment, control


# --------------------------------------------------------------------------- the runner

def _paired_bootstrap_lift_lower(task: Task, splits: certify.Splits,
                                 treat_preds: Sequence, ctrl_preds: Sequence,
                                 *, checks: int, alpha: float = 0.05,
                                 B: int = 2000, seed: int = 0) -> Optional[float]:
    """A conservative lower bound on the PAIRED per-sample lift (treatment minus control).

    Comparing two independent lower bounds (treatment vs control) ignores that both arms were
    scored on the SAME sealed rows -- a paired design with positively correlated errors. The
    paired contrast recovers that power: resample the sealed rows with replacement, recompute
    metric(treatment) - metric(control) per draw, and take the alpha/checks percentile (with the
    same alpha*0.5 tail-shrink discipline the frozen regression bootstrap uses). Multiplicity is
    paid via `checks` exactly as the frozen certifier does.

    Returned as SUPPLEMENTARY evidence in the verdict; the primary, most-defensible decision is
    still the contrast of the two frozen certified lower bounds. Returns None if metric is not a
    per-sample-poolable one (we only pair accuracy and the regression error metrics, whose draw
    statistic is an honest resample of the metric).
    """
    y_true = Task.rows_to_y(splits.sealed_rows, task.kind)
    n = len(y_true)
    if n == 0:
        return None
    metric = task.metric

    if task.kind == "classification" and metric == "accuracy":
        yt = np.asarray([str(v) for v in y_true])
        tp = np.asarray([str(v) for v in treat_preds])
        cp = np.asarray([str(v) for v in ctrl_preds])
        # per-sample correctness; the paired statistic is mean(correct_treat - correct_ctrl)
        d = (tp == yt).astype(float) - (cp == yt).astype(float)
    elif task.kind == "regression" and metric in ("neg_rmse", "neg_mae"):
        yt = np.asarray([float(v) for v in y_true])
        tp = np.asarray([float(v) for v in treat_preds])
        cp = np.asarray([float(v) for v in ctrl_preds])
        # higher-is-better: negative error. We bootstrap the metric difference directly below.
    else:
        # r2 / balanced_accuracy / macro_f1 are not clean per-sample means; defer to the
        # lower-bound contrast (do not fabricate a paired statistic that is not valid).
        return None

    checks = max(1, int(checks))
    a = (alpha / checks) * 0.5            # one-sided, with the frozen path's tail-shrink
    rng = np.random.default_rng(seed)
    draws = np.empty(B, dtype=float)
    idx_all = np.arange(n)
    for b in range(B):
        idx = rng.choice(idx_all, size=n, replace=True)
        if task.kind == "classification":
            draws[b] = float(d[idx].mean())
        else:
            et = -np.sqrt(np.mean((yt[idx] - tp[idx]) ** 2)) if metric == "neg_rmse" \
                else -np.mean(np.abs(yt[idx] - tp[idx]))
            ec = -np.sqrt(np.mean((yt[idx] - cp[idx]) ** 2)) if metric == "neg_rmse" \
                else -np.mean(np.abs(yt[idx] - cp[idx]))
            draws[b] = float(et - ec)
    return float(np.quantile(draws, a))


def run_experiment(exp: Experiment, task: Task, splits: certify.Splits,
                   treatment: Program, control: Program, *,
                   sandbox_run: Optional[Callable] = None,
                   wall_seconds: float = 60.0, cpu_seconds: int = 55,
                   margin: float = 0.0, require_treatment_certified: bool = False,
                   alpha: float = 0.05) -> Verdict:
    """Run a controlled treatment-vs-control comparison and return a go/no-go Verdict.

    Discipline enforced here:
      1. Both arms fit on the SAME train split and are scored on the SAME sealed rows
         (controls held fixed; the only difference is `factor_varied`).
      2. The sealed test is a SINGLE shared SealedTest with max_peeks = 2 (one per arm). Each
         arm is certified through the frozen `sealed.certify_on_sealed`, so the realized peek
         count escalates 1 -> 2 and Bonferroni multiplicity is paid in full. We never mint a
         fresh checks=1 test per arm (that would launder the correction).
      3. The verdict compares the two frozen CERTIFIED LOWER BOUNDS (conservative), with a
         supplementary paired-bootstrap lower bound on the per-sample lift.
      4. The sandbox is the numeric firewall: it returns predictions only; every number comes
         from the frozen certifier.

    Returns a Verdict even when an arm fails to run (go=False, with the failure recorded) --
    an honest decline, never a relabeled score.
    """
    run = sandbox_run or _sandbox.run_program

    X_train = Task.rows_to_X(splits.train_rows)
    y_train = Task.rows_to_y(splits.train_rows, task.kind)
    X_sealed = Task.rows_to_X(splits.sealed_rows)

    # One shared sealed instance for BOTH arms. max_peeks = 2 is a DELIBERATE, documented spec
    # choice for a 2-arm experiment: it does not relax the per-arm correction (still paid via
    # `checks`), it only declares up front that this locked test answers two questions.
    shared_sealed = sealed.SealedTest(splits.sealed_rows, target_key="target", max_peeks=2)
    labels = list(task.labels) if task.kind == "classification" else None

    def _certify_arm(name: str, prog: Program) -> ArmResult:
        res = run(prog, X_train, y_train, X_sealed, kind=task.kind,
                  wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)
        if not res.ok:
            # Still must NOT leave the shared sealed under-peeked relative to the other arm in a
            # way that mis-states multiplicity. We simply do not peek for a failed arm; the
            # surviving arm pays only for the peeks actually taken. Honest.
            return ArmResult(name, prog.id, prog.label, ok=False,
                             error_kind=res.error_kind, error=res.error)
        preds = list(res.preds)
        if task.kind == "classification":
            preds_for_cert = [str(p) for p in preds]
        else:
            preds_for_cert = [float(p) for p in preds]

        def predict_fn(_rows):
            return preds_for_cert

        cert = sealed.certify_on_sealed(shared_sealed, predict_fn, task.theta,
                                        metric=task.metric, labels=labels, alpha=alpha,
                                        who=f"experiment:{name}")
        return ArmResult(name, prog.id, prog.label, ok=True, certificate=cert,
                         sealed_preds=preds)

    # Order matters for multiplicity transparency: treatment peeks first (checks=1), control
    # second (checks=2). Both certificates carry their realized `checks`.
    t_arm = _certify_arm("treatment", treatment)
    c_arm = _certify_arm("control", control)

    checks = shared_sealed.peek_count()
    digest = shared_sealed.digest

    # ---- decision ----------------------------------------------------------------------
    if not (t_arm.ok and c_arm.ok):
        which = "treatment" if not t_arm.ok else "control"
        reasoning = (f"NO-GO by failure: the {which} arm did not execute "
                     f"([{getattr(t_arm if which=='treatment' else c_arm, 'error_kind')}]); "
                     f"a controlled comparison requires both arms to run. Honest decline.")
        return Verdict(False, exp, t_arm, c_arm, margin, None, None, None, reasoning,
                       checks, digest)

    lift_lb = float(t_arm.lower_bound) - float(c_arm.lower_bound)
    lift_obs = float(t_arm.observed) - float(c_arm.observed)
    paired_lb = _paired_bootstrap_lift_lower(task, splits, t_arm.sealed_preds, c_arm.sealed_preds,
                                             checks=checks, alpha=alpha)

    bound_clears = lift_lb > margin
    treat_ok = (not require_treatment_certified) or bool(t_arm.certificate.get("certified"))
    go = bool(bound_clears and treat_ok)

    if go:
        reasoning = (
            f"GO: treatment sealed lower bound ({t_arm.lower_bound:.4f}) exceeds control "
            f"({c_arm.lower_bound:.4f}) by {lift_lb:.4f} > margin {margin}, after paying "
            f"Bonferroni for checks={checks}. The lift is attributable to the factor "
            f"'{exp.factor_varied}' alone (controls held fixed)."
        )
        if paired_lb is not None:
            reasoning += f" Paired-bootstrap lift lower bound = {paired_lb:.4f}."
        if require_treatment_certified and t_arm.certificate.get("certified"):
            reasoning += " Treatment also independently clears theta."
    else:
        why = []
        if not bound_clears:
            why.append(f"the certified-lower-bound lift {lift_lb:.4f} does not exceed the "
                       f"pre-registered margin {margin}")
        if require_treatment_certified and not t_arm.certificate.get("certified"):
            why.append("the treatment does not independently clear theta")
        reasoning = ("NO-GO: " + "; ".join(why) + ". The factor "
                     f"'{exp.factor_varied}' is not supported as beneficial (a negative result "
                     "is a result; we do not relax the rule to manufacture a go).")
        if paired_lb is not None:
            reasoning += f" Paired-bootstrap lift lower bound = {paired_lb:.4f} (also <= margin)."

    return Verdict(go, exp, t_arm, c_arm, margin, round(lift_lb, 4), round(lift_obs, 4),
                   (round(paired_lb, 4) if paired_lb is not None else None),
                   reasoning, checks, digest)


# --------------------------------------------------------------------------- ablation runner

@dataclass
class AblationResult:
    """Result of an ablation: each factor level contrasted against a shared baseline."""

    experiment: Experiment
    baseline: ArmResult
    levels: List[ArmResult]                 # one ArmResult per non-baseline level
    margin: float
    checks: int
    best_level: Optional[str]               # level name with the highest certified lower bound
    best_lift_lower: Optional[float]        # its lift over the baseline (lower-bound contrast)
    go_levels: List[str]                    # level names that cleared the decision rule

    def report(self) -> str:
        lines = [f"ABLATION: {self.experiment.hypothesis}",
                 f"  factor: {self.experiment.factor_varied}  margin={self.margin}  "
                 f"sealed checks={self.checks}",
                 f"  baseline ({self.baseline.label}): lb={_fmt(self.baseline.lower_bound)}"]
        base_lb = self.baseline.lower_bound
        for lv in self.levels:
            if lv.ok and base_lb is not None and lv.lower_bound is not None:
                lift = lv.lower_bound - base_lb
                tag = "GO " if lv.name in self.go_levels else "no "
                lines.append(f"  [{tag}] {lv.name} ({lv.label}): lb={_fmt(lv.lower_bound)} "
                             f"lift={lift:+.4f}")
            else:
                lines.append(f"  [---] {lv.name} ({lv.label}): "
                             f"FAILED [{lv.error_kind}]" if not lv.ok else
                             f"  [---] {lv.name}: no baseline to contrast")
        lines.append(f"  best level: {self.best_level} "
                     f"(lift_lb={_fmt(self.best_lift_lower)})")
        return "\n".join(lines)


def run_ablation(exp: Experiment, task: Task, splits: certify.Splits,
                 baseline: Program, levels: Dict[str, Program], *,
                 sandbox_run: Optional[Callable] = None,
                 wall_seconds: float = 60.0, cpu_seconds: int = 55,
                 margin: float = 0.0, alpha: float = 0.05) -> AblationResult:
    """Run an ablation: contrast each factor LEVEL against a shared baseline on one sealed test.

    The baseline plus K levels => K+1 peeks of the same locked test. max_peeks is set to that
    count, and the frozen certifier pays Bonferroni for the realized peek total (the LAST arm
    pays checks=K+1). This is the multi-arm generalization of run_experiment: any level whose
    certified lower bound beats the baseline's by `margin` is a "go". The strongest correction
    is therefore applied uniformly, which is the conservative, defensible choice for a family of
    contrasts against one locked test.
    """
    run = sandbox_run or _sandbox.run_program
    X_train = Task.rows_to_X(splits.train_rows)
    y_train = Task.rows_to_y(splits.train_rows, task.kind)
    X_sealed = Task.rows_to_X(splits.sealed_rows)

    n_arms = 1 + len(levels)
    shared_sealed = sealed.SealedTest(splits.sealed_rows, target_key="target", max_peeks=n_arms)
    clf_labels = list(task.labels) if task.kind == "classification" else None

    def _certify(name: str, prog: Program) -> ArmResult:
        res = run(prog, X_train, y_train, X_sealed, kind=task.kind,
                  wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)
        if not res.ok:
            return ArmResult(name, prog.id, prog.label, ok=False,
                             error_kind=res.error_kind, error=res.error)
        preds = list(res.preds)
        pf = ([str(p) for p in preds] if task.kind == "classification"
              else [float(p) for p in preds])
        cert = sealed.certify_on_sealed(shared_sealed, lambda _r: pf, task.theta,
                                        metric=task.metric, labels=clf_labels, alpha=alpha,
                                        who=f"ablation:{name}")
        return ArmResult(name, prog.id, prog.label, ok=True, certificate=cert, sealed_preds=preds)

    base_arm = _certify("baseline", baseline)
    level_arms = [_certify(name, prog) for name, prog in levels.items()]

    checks = shared_sealed.peek_count()
    base_lb = base_arm.lower_bound

    go_levels: List[str] = []
    best_level: Optional[str] = None
    best_lift: Optional[float] = None
    best_abs_lb: Optional[float] = None
    for arm in level_arms:
        if not arm.ok or base_lb is None or arm.lower_bound is None:
            continue
        lift = arm.lower_bound - base_lb
        if lift > margin:
            go_levels.append(arm.name)
        if best_abs_lb is None or arm.lower_bound > best_abs_lb:
            best_abs_lb = arm.lower_bound
            best_level = arm.name
            best_lift = lift

    return AblationResult(exp, base_arm, level_arms, margin, checks, best_level,
                          (round(best_lift, 4) if best_lift is not None else None), go_levels)


# --------------------------------------------------------------------------- LLM hook

def design_experiment(goal: str, factor: str,
                      client: Optional[Callable[[str], str]] = None,
                      *, controls: Optional[Dict[str, object]] = None,
                      margin: float = 0.0) -> Experiment:
    """Author an Experiment object from a goal + a candidate factor.

    With a `client` (a frontier model: prompt -> text), the LLM writes the hypothesis and the
    decision-rule prose conditioned on the goal. With `client=None` it DEGRADES HONESTLY to a
    deterministic template -- it does not fabricate a model's reasoning. Either way the returned
    object is a fully-formed, pre-registerable Experiment; the numbers later come only from the
    frozen certifier.
    """
    fixed = controls or {"base": "fixed", "estimator_hyperparams": "fixed",
                         "splits": "shared", "seed": "fixed"}
    decision_rule = (f"go iff treatment sealed lower bound > control sealed lower bound + "
                     f"{margin} (Bonferroni-paid for both peeks)")
    if client is None:
        hypothesis = (f"Varying '{factor}' improves the certified outcome for the goal: "
                      f"{goal}.")
        return Experiment(hypothesis=hypothesis, factor_varied=factor,
                          controls_held_fixed=fixed, decision_rule=decision_rule,
                          metadata={"authored_by": "deterministic_template"})

    prompt = (
        "You are a senior ML scientist scoping a controlled ablation.\n"
        f"Goal: {goal}\n"
        f"Single factor to vary: {factor}\n"
        "Write ONE sentence: a falsifiable, directional hypothesis about whether varying this "
        "factor improves the certified result. No preamble, no markdown."
    )
    try:
        text = client(prompt)
    except Exception:
        text = ""
    hypothesis = (text.strip().splitlines()[0].strip() if text and text.strip()
                  else f"Varying '{factor}' improves the certified outcome for: {goal}.")
    return Experiment(hypothesis=hypothesis, factor_varied=factor,
                      controls_held_fixed=fixed, decision_rule=decision_rule,
                      metadata={"authored_by": "llm"})
