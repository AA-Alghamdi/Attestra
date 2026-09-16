"""Phase 2 -- diagnosis feeds forward.

Round N's result conditions round N+1's proposals. The audit found this channel dead in two
places: `vfplatform/loop.py:781` computed a `_relevant` diagnosis and then discarded it, and PR19
`attestra/cycle/engine.py:395` consumed only the scalar `gap` (target - best) and threw away the
typed-failure / per-family structure. This module makes the channel load-bearing.

`diagnose(history, diagnosis_trail, task, ...)` reads what the spine already records -- the per
candidate `_Record` log (label, source, ok, val_score, error_kind, error) plus the per-round trail
-- and (optionally) the winner's validation residuals, and returns a `Diagnosis` with:

  - plateau detection           : has best-on-val stopped improving across rounds?
  - dominant error_kind         : which failure mode (timeout/import/fit/build/oom/...) dominates
                                  the *failed* candidates, with its share.
  - per-family / per-recipe rank : mean+best val score per base estimator family and per recipe
                                  axis (scale / poly / target_log), so we know what to exploit.
  - residual / error structure  : for regression, skew + heteroscedastic-vs-magnitude signal in
                                  the champion's residuals (motivates target_log / robust models);
                                  for classification, the failure mix. Degrades honestly to "" when
                                  residuals are not supplied.
  - ACTIONABLE directives       : `fire_sources` (which proposers to run), `llm_guidance` (a string
                                  appended to the LLM prompt), `avoid_axes` / `exploit_axes`
                                  (recipe axes), `avoid_bases` / `prefer_bases`, and a `reason`.

`enrich_context(context, diagnosis)` injects the directives (and a compact human-readable summary)
into the proposer context dict in place, under the `"diagnosis"` key plus a few flat convenience
keys, WITHOUT clobbering the Phase-0 keys the existing proposers read.

`DiagnosisDrivenProposer` is a thin wrapper: it gates a wrapped `ProposalSource` -- it only fires
when the diagnosis says its source kind should fire, and it filters the wrapped source's output to
drop proposals on `avoid_bases` / `avoid_axes`. The wrapped source is never modified.

Integrity (CONTRACT.md standing invariants this preserves):
  - It NEVER promotes and never recomputes a promotion-bearing number. It only re-ranks/gates what
    may be PROPOSED (invariant 4: generalization expands proposals, never what may promote).
  - Directives are derived from the run's own typed history, not from any target answer.
  - The LLM is the open-ended driver: `llm_guidance` is advice fed to the model, and when no LLM
    client is wired the deterministic `fire_sources` / axis directives still steer seeds+mutations.
    Honest degradation, never fabrication.

# === WIRING ===
# The integrator threads diagnosis between rounds inside `ResearchEngine.run` (frontier/engine.py).
# Phase 0 already builds a `context` dict each round and appends a `round_log` to `trail`. Phase 2
# inserts ONE diagnose call + ONE enrich call right after the context is built, and wraps each
# proposer once at construction. Concretely (engine.py.run, no Phase-0 line deleted):
#
#   from .diagnosis import diagnose, enrich_context, DiagnosisDrivenProposer
#   ...
#   # (a) at __init__, gate each proposer by diagnosis (sources self-identify; see _source_kind):
#   self.proposers = [DiagnosisDrivenProposer(p) for p in self.proposers]
#
#   # (b) inside the round loop, after `context = {...}` is assembled and BEFORE proposers run:
#   diag = diagnose(history, trail, task)            # history/trail are what the loop already keeps
#   enrich_context(context, diag)                    # injects context["diagnosis"] = diag.directives
#   trail[-1] if trail else None                     # (optional) attach diag.as_dict() to the round log
#
#   # (c) DiagnosisDrivenProposer.propose(context) reads context["diagnosis"]; when absent (round 0
#   #     or Phase-0 callers that never enrich) it falls through to the wrapped source unchanged, so
#   #     wrapping is backward compatible with the existing test_spine.py loop.
#
# Argument shapes: `history` is List[_Record] (engine._Record); `trail` is List[dict] (round logs
# each carrying round/best_out_score); `task` is frontier.task.Task. `context` is the exact dict
# Phase 0 builds (keys: task_kind, n_features, n_train, round, tried_labels, best_label, best_score,
# best_id, best_recipe, recent_errors). `diagnose` also accepts optional val_truth/val_preds for the
# champion to enable residual structure; the spine can pass them after it scores the winner on val.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .proposers import ProposalSource
from .program import Program
from .task import Task


# The recipe axes the seed/mutation proposers understand (proposers.py recipe_label / make_code).
_AXES = ("scale", "poly", "target_log")

# Known source kinds, matched against ProposalSource class names so we can gate by kind without
# importing concrete classes (avoids a hard dependency + lets later phases add sources).
_SOURCE_KINDS = ("seed", "mutation", "llm", "retrieval")


# --------------------------------------------------------------------------- result types

@dataclass
class FamilyStat:
    """Aggregate val performance for one base-estimator family (e.g. 'ridge', 'hist_gbm')."""
    family: str
    n_ok: int
    n_fail: int
    best: Optional[float]      # best val score seen for this family (None if never ran ok)
    mean: Optional[float]      # mean val score over its ok runs


@dataclass
class Diagnosis:
    """Structured read of the run so far, plus actionable directives for the next round.

    `directives` is the machine-actionable payload `enrich_context` injects and
    `DiagnosisDrivenProposer` gates on. Everything else is explanatory / for the round log.
    """
    round_index: int
    n_candidates: int
    n_ok: int
    n_fail: int
    best_score: Optional[float]
    best_label: Optional[str]

    plateau: bool                       # best-on-val has not improved by > eps across recent rounds
    plateau_span: int                   # how many consecutive rounds without improvement
    dominant_error_kind: Optional[str]  # most common error_kind among failures (None if no failures)
    dominant_error_share: float         # its fraction of all failures (0..1)

    family_rank: List[FamilyStat]       # families sorted best-first (ok families before failed-only)
    axis_lift: Dict[str, Optional[float]]   # axis -> (mean val WITH axis) - (mean val WITHOUT axis)
    residual_summary: str               # "" when residuals unavailable / not applicable

    directives: Dict[str, Any]          # the actionable payload (see _build_directives)
    reason: str                         # one-line human explanation of why these directives

    def as_dict(self) -> Dict[str, Any]:
        """JSON-friendly view for attaching to the round log / diagnosis_trail."""
        return {
            "round": self.round_index,
            "n_candidates": self.n_candidates,
            "n_ok": self.n_ok,
            "n_fail": self.n_fail,
            "best_score": self.best_score,
            "best_label": self.best_label,
            "plateau": self.plateau,
            "plateau_span": self.plateau_span,
            "dominant_error_kind": self.dominant_error_kind,
            "dominant_error_share": round(self.dominant_error_share, 3),
            "family_rank": [
                {"family": f.family, "n_ok": f.n_ok, "n_fail": f.n_fail,
                 "best": f.best, "mean": f.mean}
                for f in self.family_rank
            ],
            "axis_lift": {k: (round(v, 4) if v is not None else None)
                          for k, v in self.axis_lift.items()},
            "residual_summary": self.residual_summary,
            "directives": self.directives,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- label parsing

def _parse_recipe_from_label(label: str) -> Dict[str, Any]:
    """Recover a recipe dict from a proposer label like 'tlog+poly2+scale+ridge'.

    proposers.recipe_label joins tags with '+'; the LAST token is always the base estimator,
    the rest are axis tags. This lets diagnosis read per-axis structure from history alone --
    history `_Record`s carry the label, not the provenance recipe. Unknown formats (e.g. raw
    LLM labels like 'llm0_1') return {'base': '<label>'} so they still rank as their own family.
    """
    if not label:
        return {"base": ""}
    parts = label.split("+")
    base = parts[-1]
    recipe: Dict[str, Any] = {"base": base}
    for tag in parts[:-1]:
        if tag == "tlog":
            recipe["target_log"] = True
        elif tag == "scale":
            recipe["scale"] = True
        elif tag.startswith("poly"):
            try:
                recipe["poly"] = int(tag[4:])
            except ValueError:
                recipe["poly"] = 2
    return recipe


def _axis_present(recipe: Dict[str, Any], axis: str) -> bool:
    val = recipe.get(axis)
    if axis == "poly":
        return bool(val) and int(val) > 1
    return bool(val)


# --------------------------------------------------------------------------- core diagnose

def _records(history: Sequence) -> List[Any]:
    """history is List[engine._Record]; tolerate dict rows too (for tests / serialized logs)."""
    out = []
    for r in history:
        if isinstance(r, dict):
            out.append(_RecordView(r))
        else:
            out.append(r)
    return out


class _RecordView:
    """Read-only adapter so diagnose() accepts either _Record objects or plain dicts."""
    def __init__(self, d: Dict[str, Any]):
        self.label = d.get("label", "")
        self.source = d.get("source", "")
        self.ok = bool(d.get("ok", False))
        self.val_score = d.get("val_score")
        self.error_kind = d.get("error_kind", "")
        self.error = d.get("error", "")


def _detect_plateau(trail: Sequence, current_best: Optional[float],
                    eps: float = 1e-4) -> Tuple[bool, int]:
    """Has the best-on-val score stalled across recent rounds?

    Reads `best_out_score` from each completed round log (engine writes it). A plateau is >= 2
    consecutive completed rounds whose best did not improve by more than `eps`. Returns
    (is_plateau, span) where span counts the trailing no-improvement rounds.
    """
    scores = [rl.get("best_out_score") for rl in trail
              if isinstance(rl, dict) and rl.get("best_out_score") is not None]
    if len(scores) < 2:
        return False, 0
    span = 0
    for i in range(len(scores) - 1, 0, -1):
        if scores[i] - scores[i - 1] <= eps:
            span += 1
        else:
            break
    return (span >= 2), span


def _family_stats(recs: Sequence) -> List[FamilyStat]:
    """Aggregate val score per base family. Families with any ok run rank before failed-only ones."""
    by_fam_scores: Dict[str, List[float]] = defaultdict(list)
    by_fam_ok: Counter = Counter()
    by_fam_fail: Counter = Counter()
    for r in recs:
        fam = _parse_recipe_from_label(r.label).get("base", "") or "?"
        if r.ok and r.val_score is not None:
            by_fam_scores[fam].append(float(r.val_score))
            by_fam_ok[fam] += 1
        else:
            by_fam_fail[fam] += 1
    fams = set(by_fam_ok) | set(by_fam_fail)
    stats = []
    for fam in fams:
        s = by_fam_scores.get(fam, [])
        stats.append(FamilyStat(
            family=fam, n_ok=by_fam_ok[fam], n_fail=by_fam_fail[fam],
            best=(max(s) if s else None),
            mean=(sum(s) / len(s) if s else None),
        ))
    # rank: ok-families by best desc, then failed-only families by fewest failures
    stats.sort(key=lambda f: (f.best is None, -(f.best if f.best is not None else 0.0), f.n_fail))
    return stats


def _axis_lift(recs: Sequence) -> Dict[str, Optional[float]]:
    """For each recipe axis, mean(val | axis present) - mean(val | axis absent).

    Positive => the axis helps and should be EXPLOITED; clearly negative => AVOID. None when there
    is no contrast (no ok run on one side of the axis). This is the per-recipe-axis signal that the
    old `gap`-only path threw away.
    """
    lift: Dict[str, Optional[float]] = {}
    for axis in _AXES:
        with_a, without_a = [], []
        for r in recs:
            if not (r.ok and r.val_score is not None):
                continue
            recipe = _parse_recipe_from_label(r.label)
            (with_a if _axis_present(recipe, axis) else without_a).append(float(r.val_score))
        if with_a and without_a:
            lift[axis] = (sum(with_a) / len(with_a)) - (sum(without_a) / len(without_a))
        else:
            lift[axis] = None
    return lift


def _residual_summary(task: Task,
                      val_truth: Optional[Sequence],
                      val_preds: Optional[Sequence]) -> str:
    """Summarize the champion's error structure on validation, when available.

    Regression: report residual skew and whether |residual| grows with |y| (heteroscedasticity),
    which motivates a target transform (target_log) or a more robust/flexible base. Classification:
    report the error rate (a hook later phases extend to per-class confusion). Returns "" when no
    residuals are supplied -- honest degradation, never a fabricated structure claim.
    """
    if val_truth is None or val_preds is None or len(val_truth) == 0:
        return ""
    if len(val_truth) != len(val_preds):
        return ""
    if task.kind == "regression":
        import numpy as np
        yt = np.asarray([float(v) for v in val_truth], dtype=float)
        yp = np.asarray([float(v) for v in val_preds], dtype=float)
        resid = yt - yp
        if resid.std() < 1e-12:
            return "residuals near-zero (champion fits val almost exactly)"
        # skew (third standardized moment) -- large |skew| suggests a target transform may help
        z = (resid - resid.mean()) / (resid.std() + 1e-12)
        skew = float((z ** 3).mean())
        # heteroscedasticity: corr(|resid|, |y|). Positive => bigger errors at bigger targets.
        ay = np.abs(yt - yt.mean())
        ar = np.abs(resid)
        if ay.std() > 1e-12 and ar.std() > 1e-12:
            het = float(np.corrcoef(ay, ar)[0, 1])
        else:
            het = 0.0
        parts = [f"resid skew={skew:+.2f}", f"hetero(|r|~|y|) corr={het:+.2f}"]
        return "; ".join(parts)
    else:
        import numpy as np
        yt = [str(v) for v in val_truth]
        yp = [str(v) for v in val_preds]
        err = sum(1 for a, b in zip(yt, yp) if a != b) / max(1, len(yt))
        return f"val error rate={err:.3f} over {len(yt)} examples"


# --------------------------------------------------------------------------- directives

def _build_directives(*, plateau: bool, dominant_error_kind: Optional[str],
                      dominant_error_share: float, family_rank: List[FamilyStat],
                      axis_lift: Dict[str, Optional[float]], residual_summary: str,
                      task_kind: str, n_ok: int,
                      axis_eps: float = 5e-3) -> Tuple[Dict[str, Any], str]:
    """Turn the structured read into a machine-actionable directive payload + a one-line reason.

    Decision logic (each rule explained by WHY, all derived from the run's own history):
      - On a plateau: stop re-firing seeds (they are exhausted -- a fixed library) and escalate to
        the generative sources (mutation + llm). This is the whole point of Phase 2: a stalled run
        should pivot, not keep proposing the same baselines.
      - Dominant systemic error (e.g. mostly 'import'/'oom'/'timeout' with a high share): tell the
        LLM explicitly to avoid that failure mode; keep seeds firing as a safe floor.
      - Axis lift drives exploit/avoid: axes that helped go to `exploit_axes`, axes that clearly
        hurt go to `avoid_axes` (the mutation proposer + LLM read these).
      - Family rank drives prefer/avoid bases: the top ok family is preferred; bases that only ever
        failed are avoided so we stop wasting the budget on them.
    """
    fire = set(_SOURCE_KINDS)            # default: let every wired source fire
    avoid_axes: List[str] = []
    exploit_axes: List[str] = []
    avoid_bases: List[str] = []
    prefer_bases: List[str] = []
    llm_bits: List[str] = []
    reasons: List[str] = []

    # --- axis exploit / avoid from measured lift ----------------------------------------------
    for axis, lift in axis_lift.items():
        if lift is None:
            continue
        if lift > axis_eps:
            exploit_axes.append(axis)
        elif lift < -axis_eps:
            avoid_axes.append(axis)
    if exploit_axes:
        llm_bits.append("Exploit these preprocessing axes that helped on val: "
                        + ", ".join(exploit_axes) + ".")
    if avoid_axes:
        llm_bits.append("Avoid these axes that hurt on val: " + ", ".join(avoid_axes) + ".")

    # --- family prefer / avoid -----------------------------------------------------------------
    ok_fams = [f for f in family_rank if f.best is not None]
    failed_only = [f for f in family_rank if f.best is None and f.n_fail > 0]
    if ok_fams:
        prefer_bases.append(ok_fams[0].family)
        llm_bits.append(f"Best base family so far: {ok_fams[0].family} "
                        f"(best val {ok_fams[0].best:.4f}).")
    for f in failed_only:
        # only-ever-failed bases: stop spending budget on them
        if f.family and f.family != "?":
            avoid_bases.append(f.family)
    if avoid_bases:
        llm_bits.append("These bases only ever failed; do not reuse: " + ", ".join(avoid_bases) + ".")

    # --- plateau pivot -------------------------------------------------------------------------
    if plateau:
        fire.discard("seed")             # the seed library is a fixed set; re-firing it is wasted
        reasons.append("plateau -> drop seeds, escalate to mutation+llm")
        llm_bits.append("Search has plateaued on val. Propose a structurally DIFFERENT pipeline "
                        "(e.g. stacking, feature selection, a different model class), not a tweak "
                        "of the current champion.")
    else:
        reasons.append("improving -> keep all sources")

    # --- dominant systemic error ---------------------------------------------------------------
    if dominant_error_kind and dominant_error_share >= 0.5 and n_ok >= 0:
        reasons.append(f"dominant error '{dominant_error_kind}' ({dominant_error_share:.0%})")
        if dominant_error_kind in ("import",):
            llm_bits.append("Several candidates failed on IMPORT. Use only scikit-learn / numpy; "
                            "no external packages.")
        elif dominant_error_kind in ("timeout", "cpu"):
            llm_bits.append("Several candidates TIMED OUT. Prefer cheaper models / fewer estimators "
                            "/ no high-degree polynomial expansion on many features.")
        elif dominant_error_kind in ("oom",):
            llm_bits.append("Several candidates ran OUT OF MEMORY. Avoid dense feature blow-ups "
                            "(high-degree PolynomialFeatures, huge ensembles).")
        elif dominant_error_kind in ("build", "fit"):
            llm_bits.append("Several candidates failed to BUILD/FIT. Return a valid unfitted "
                            "sklearn estimator from build_estimator(); check pipeline step names.")

    # --- residual-structure -> targeted advice (regression) -----------------------------------
    if residual_summary and task_kind == "regression":
        # parse the cheap signals we emitted; only ADD advice, never override measured axis_lift
        if "skew=" in residual_summary:
            try:
                skew = float(residual_summary.split("skew=")[1].split(";")[0])
            except (IndexError, ValueError):
                skew = 0.0
            if abs(skew) > 0.75 and "target_log" not in exploit_axes and "target_log" not in avoid_axes:
                llm_bits.append("Champion residuals are skewed; try a target transform "
                                "(e.g. log1p) or a quantile-robust loss.")
        if "corr=" in residual_summary:
            try:
                het = float(residual_summary.split("corr=")[1].rstrip("."))
            except (IndexError, ValueError):
                het = 0.0
            if het > 0.3:
                llm_bits.append("Errors grow with target magnitude (heteroscedastic); a target "
                                "transform or a tree ensemble may help.")

    directives: Dict[str, Any] = {
        "fire_sources": sorted(fire),
        "avoid_axes": sorted(set(avoid_axes)),
        "exploit_axes": sorted(set(exploit_axes)),
        "avoid_bases": sorted(set(avoid_bases)),
        "prefer_bases": sorted(set(prefer_bases)),
        "llm_guidance": " ".join(llm_bits),
    }
    reason = "; ".join(reasons) if reasons else "no actionable signal yet"
    return directives, reason


def diagnose(history: Sequence, diagnosis_trail: Sequence, task: Task, *,
             val_truth: Optional[Sequence] = None,
             val_preds: Optional[Sequence] = None,
             plateau_eps: float = 1e-4) -> Diagnosis:
    """Diagnose the run so far and emit actionable directives for the next round.

    Parameters
    ----------
    history : list of engine._Record (or dicts with the same keys). The per-candidate log the
        spine already keeps: label, source, ok, val_score, error_kind, error.
    diagnosis_trail : list of per-round log dicts (each carries best_out_score). Used for plateau.
    task : frontier.task.Task -- supplies kind (classification/regression) and metric.
    val_truth, val_preds : OPTIONAL champion validation truth + predictions, for residual structure.
        When omitted, residual_summary is "" (honest degradation) and everything else still works.

    Returns
    -------
    Diagnosis with structured findings + a `directives` payload for enrich_context /
    DiagnosisDrivenProposer.
    """
    recs = _records(history)
    n_ok = sum(1 for r in recs if r.ok)
    n_fail = len(recs) - n_ok

    # best on val + its label
    best_score: Optional[float] = None
    best_label: Optional[str] = None
    for r in recs:
        if r.ok and r.val_score is not None:
            if best_score is None or float(r.val_score) > best_score:
                best_score, best_label = float(r.val_score), r.label

    # dominant error kind among failures
    err_counter = Counter(r.error_kind or "other" for r in recs if not r.ok)
    if err_counter:
        dom_kind, dom_n = err_counter.most_common(1)[0]
        dom_share = dom_n / sum(err_counter.values())
    else:
        dom_kind, dom_share = None, 0.0

    plateau, span = _detect_plateau(diagnosis_trail, best_score, eps=plateau_eps)
    fam_rank = _family_stats(recs)
    lift = _axis_lift(recs)
    resid = _residual_summary(task, val_truth, val_preds)

    round_index = sum(1 for rl in diagnosis_trail if isinstance(rl, dict) and "round" in rl)

    directives, reason = _build_directives(
        plateau=plateau, dominant_error_kind=dom_kind, dominant_error_share=dom_share,
        family_rank=fam_rank, axis_lift=lift, residual_summary=resid,
        task_kind=task.kind, n_ok=n_ok,
    )

    return Diagnosis(
        round_index=round_index, n_candidates=len(recs), n_ok=n_ok, n_fail=n_fail,
        best_score=best_score, best_label=best_label,
        plateau=plateau, plateau_span=span,
        dominant_error_kind=dom_kind, dominant_error_share=dom_share,
        family_rank=fam_rank, axis_lift=lift, residual_summary=resid,
        directives=directives, reason=reason,
    )


# --------------------------------------------------------------------------- context injection

def enrich_context(context: Dict[str, Any], diagnosis: Diagnosis) -> Dict[str, Any]:
    """Inject the diagnosis directives into the proposer context dict (in place) and return it.

    Adds:
      - context["diagnosis"]      : the full directive payload (what DiagnosisDrivenProposer gates on)
      - context["diag_guidance"]  : the LLM guidance string (LLMProposer can append it to its prompt)
      - context["diag_summary"]   : a compact human-readable line for logs

    Does NOT remove or rewrite any Phase-0 key (task_kind, best_recipe, recent_errors, ...), so the
    unmodified Phase-0 proposers keep working; new behavior is purely additive.
    """
    d = diagnosis.directives
    context["diagnosis"] = d
    context["diag_guidance"] = d.get("llm_guidance", "")
    context["diag_summary"] = (
        f"round={diagnosis.round_index} best={diagnosis.best_score} "
        f"plateau={diagnosis.plateau} dom_err={diagnosis.dominant_error_kind} "
        f"fire={d.get('fire_sources')} exploit={d.get('exploit_axes')} "
        f"avoid_axes={d.get('avoid_axes')} avoid_bases={d.get('avoid_bases')}"
    )
    return context


# --------------------------------------------------------------------------- gating proposer

def _source_kind(src: ProposalSource) -> str:
    """Best-effort source-kind tag from the wrapped source's class name.

    SeedProposer->'seed', MutationProposer->'mutation', LLMProposer->'llm', RetrievalProposer->
    'retrieval'. Unknown sources default to 'other' and are always allowed to fire (we never gate
    out a source we cannot classify -- fail open, since gating is an optimization not a safety gate).
    """
    name = type(src).__name__.lower()
    for k in _SOURCE_KINDS:
        if k in name:
            return k
    return "other"


class DiagnosisDrivenProposer:
    """Wrap a ProposalSource and gate it by the diagnosis directives in `context["diagnosis"]`.

    Two gates, both pure proposal-side (never promotion-side, per CONTRACT invariant 4):
      1. fire gate : if the directives list `fire_sources` and this source's kind is NOT in it,
                     propose nothing this round. (E.g. a plateau drops 'seed' so the fixed seed
                     library stops re-firing and the budget shifts to generative sources.)
      2. filter    : drop any proposal whose recipe uses an `avoid_base` or an `avoid_axis`. The
                     wrapped source is untouched; we only thin its output.

    Backward compatible: when `context` has no "diagnosis" key (round 0, or a Phase-0 caller that
    never enriches), it falls through to the wrapped source's output unchanged -- so wrapping every
    proposer is safe even for the existing test_spine.py loop.
    """

    def __init__(self, source: ProposalSource):
        self.source = source
        self.kind = _source_kind(source)

    def propose(self, context: Dict[str, Any]) -> List[Program]:
        directives = context.get("diagnosis")
        if not directives:
            return list(self.source.propose(context))   # no diagnosis yet -> unchanged behavior

        fire = directives.get("fire_sources")
        if fire is not None and self.kind not in fire and self.kind != "other":
            return []                                   # gated off this round

        proposals = list(self.source.propose(context))
        avoid_bases = set(directives.get("avoid_bases", ()))
        avoid_axes = set(directives.get("avoid_axes", ()))
        if not avoid_bases and not avoid_axes:
            return proposals

        kept = []
        for p in proposals:
            recipe = p.provenance.get("recipe") if isinstance(p.provenance, dict) else None
            if recipe is None:
                recipe = _parse_recipe_from_label(p.label)
            if recipe.get("base") in avoid_bases:
                continue
            if any(_axis_present(recipe, ax) for ax in avoid_axes):
                continue
            kept.append(p)
        return kept
