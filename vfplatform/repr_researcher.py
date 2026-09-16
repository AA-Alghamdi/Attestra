"""AUTONOMOUS REPRESENTATION RESEARCHER -- the system runs B1->#5 by itself.

WHY THIS EXISTS
---------------
Across seven phases the *researcher* was the agent: I wrote each benchmark, picked the next encoder, read
the FDR table, and decided the next move ("swap family", "fusion is dead", "scale the winning family",
"the residual is a data ceiling"). The product thesis is an autonomous frontier researcher, so the SYSTEM
must close that loop. This module is the policy that does it: it proposes a representation, has it certified
on a sealed test by the frozen certifier, promotes it ONLY on an FDR-surviving win over the current
champion, escalates the KIND of move when a class saturates, and honest-stops -- emitting a replayable
certificate. The agent no longer decides anything; it only supplies the arena.

THE LADDER (escalate.py, used verbatim) maps exactly onto the hand-run phases:
    model            -- swap to a different frozen encoder FAMILY      (B2-repr: CLIP/DINOv2 > resnet18)
    features         -- engineer features on the rep = FUSE encoders    (#4: concat/PLS, proved 0/10 dead)
    capacity         -- scale to a bigger member of the WINNING family  (#5: DINOv2-L -> DINOv2-g, 3/10)
    data_acquisition -- acquire more labels (the residual DATA ceiling) (#5: the 737 pairs stay stuck)
    stop             -- honest-stop (the sealed peek is preserved)

INVARIANTS (inherited, not re-implemented here)
-----------------------------------------------
  * The frozen certifier is the SOLE promoter. Every champion-relative decision goes through
    verification.VerificationCascade.certify_tier3, whose Tier-3 delegates to a caller-supplied frozen
    certify_fn (the paired-McNemar + BH-FDR suite comparison vs the current champion) and whose
    `assert_only_tier3_promotes` guard rejects any promotion not minted at Tier 3. The cheap tiers
    (e-process race + frozen val Clopper-Pearson bound) only check COMPETENCE against a fixed floor and
    ALLOCATE the scarce sealed peeks (select-then-bound); they NEVER make the champion-relative call and
    NEVER promote. When several challengers each certify over the champion in a round, the best is chosen by
    its certified sealed lower bound -- an argmax over already-certified winners, not a fresh hypothesis test.
  * The policy never relaxes a threshold, never touches the sealed labels, and is deterministic given the
    arena. A wrong decision can at worst waste a peek; it can never mint a false certificate.

CONTRACT: this module is the BRAIN only -- it imports the policy primitives (verification cascade, escalate
ladder, pareto front) and an Arena that supplies measurements. It does not import torch/sklearn or the
frozen science core directly; the arena's certify path reaches the frozen bound. That keeps the policy
unit-testable with a synthetic arena (tests/test_repr_researcher.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import escalate as ESC
from .pareto import MAXIMIZE, MINIMIZE, ParetoCandidate, ParetoFront
from .verification import Candidate, VerificationCascade

# Each escalation rung -> the representation move it expands into. "stop"/"continue" come from the escalate
# policy itself; these name the rungs whose proposals this researcher knows how to generate.
RUNG_FAMILY = "model"             # swap to a different encoder family
RUNG_FUSE = "features"            # fuse the current rep with another (authored featurizer)
RUNG_SCALE = "capacity"           # scale up within the winning family
RUNG_DATA = "data_acquisition"    # residual is a data ceiling (no encoder move)


@dataclass(frozen=True)
class Encoder:
    """A frozen representation in the registry. `family` groups encoders sharing an inductive bias
    (dinov2 / clip / siglip / eva02 / resnet); `scale_rank` orders members within a family (bigger last);
    `params_m` is the parameter count in millions (a cost proxy for the certified Pareto front)."""
    tag: str
    name: str
    family: str
    scale_rank: int
    params_m: float


@dataclass
class TaskMeasure:
    """A challenger's measurement on ONE task, on the IDENTICAL sealed rows as the champion.
    `sealed_correct` / `val_correct` are 0/1 per-example correctness vectors; `acc` is sealed accuracy."""
    sealed_correct: Sequence[int]
    val_correct: Sequence[int]
    acc: float


class Arena:
    """The domain the researcher operates on. The script supplies a concrete arena backed by cached
    embeddings + the tuned-GBM head + the frozen Clopper-Pearson bound; tests supply a synthetic one.

    Required surface:
      tasks                       -> list of task names (the suite McNemar+BH-FDR runs across these).
      measure(encoder_tag)        -> {task_name: TaskMeasure} on the identical sealed rows.
      fuse_measure(tag_a, tag_b)  -> same, for the concat-fused representation (the 'features' rung).
      mcnemar(c_chal, c_base)     -> one-sided exact McNemar p-value for (challenger > baseline) on a task.
      bh(pvalues, alpha)          -> indices surviving Benjamini-Hochberg(alpha).
      lower_bound(correct)        -> frozen Clopper-Pearson lower bound for a correctness vector.
    Subclasses override measure/fuse_measure/mcnemar/bh/lower_bound. This base only fixes the contract."""

    tasks: List[str] = []

    def measure(self, encoder_tag: str) -> Dict[str, TaskMeasure]:        # pragma: no cover - interface
        raise NotImplementedError

    def fuse_measure(self, tag_a: str, tag_b: str) -> Dict[str, TaskMeasure]:  # pragma: no cover
        raise NotImplementedError

    def mcnemar(self, chal_correct: Sequence[int], base_correct: Sequence[int]) -> float:  # pragma: no cover
        raise NotImplementedError

    def bh(self, pvalues: Sequence[float], alpha: float) -> List[int]:    # pragma: no cover - interface
        raise NotImplementedError

    def lower_bound(self, correct: Sequence[int]) -> float:              # pragma: no cover - interface
        raise NotImplementedError

    def gold_measure(self, name: str) -> Optional[Dict[str, Sequence[int]]]:
        """OPTIONAL never-peeked GOLD confirmation. Return {task: 0/1 correctness vector} for the
        encoder/fusion `name` on a held-out GOLD partition the climbing loop NEVER queries -- a DISJOINT
        set of fresh examples, not the working sealed rows. The head is trained the standard way (train
        rows, val-selected) and scored on this gold set, which was selected before any climbing and read
        EXACTLY ONCE, for the final champion and the start baseline only. Return None if the domain has no
        such set (then no gold confirmation is emitted). Default: not available."""
        return None


@dataclass
class Comparison:
    """The certified suite-level comparison of a challenger measurement against the current champion.
    `sealed_lb` is the challenger's pooled (all-tasks) frozen Clopper-Pearson lower bound -- used ONLY to
    pick among challengers that have ALREADY each certified over the champion (argmax, never a fresh test)."""
    survivors: List[str]
    n: int
    mean_lift: float
    sealed_lb: float
    per_task: List[dict]
    certified: bool

    def as_cert(self) -> dict:
        # The dict the frozen Tier-3 reads: 'certified' is the only field certify_tier3 checks.
        return {"certified": self.certified, "survivors": self.survivors, "n": self.n,
                "mean_lift": round(self.mean_lift, 4), "sealed_lb": round(self.sealed_lb, 4),
                "per_task": self.per_task}


@dataclass
class Promotion:
    rung: str
    from_tag: str
    to_tag: str
    survivors: List[str]
    n: int
    mean_lift: float


@dataclass
class Rejection:
    rung: str
    tag: str
    survivors: List[str]
    n: int
    mean_lift: float
    reason: str


@dataclass
class ResearchCertificate:
    champion: str
    champion_family: str
    promotions: List[Promotion]
    rejections: List[Rejection]
    move_class_path: List[str]
    stop_reason: str
    sealed_lb: Dict[str, float]            # champion's frozen CP lower bound per task
    sealed_acc: Dict[str, float]           # champion's sealed accuracy per task
    pareto_front: List[str]                # certified, non-dominated encoder tags (accuracy vs cost)
    pareto_report: str
    data_ceiling_tasks: List[str]          # tasks the best representation still can't certify (data ceiling)
    peeks_used: int
    log: List[str] = field(default_factory=list)
    # The champion's confirmation on a NEVER-PEEKED gold set (queried once, after climbing). None when the
    # arena supplies no disjoint gold partition (e.g. data-limited domains). Climbing NEVER reads gold, so a
    # gold-confirmed champion's win is immune to the sealed-test re-use multiplicity accrued during the climb.
    gold_confirmation: Optional[dict] = None
    # Honest session-level multiplicity accounting: how many certified looks the climb spent on the ONE
    # working sealed set, the family-wise Bonferroni threshold that implies, and a conservative sensitivity
    # re-test of the final champion vs the start baseline at that threshold. Always present.
    multiplicity: Optional[dict] = None


class ReprResearcher:
    """The champion-climbing controller. Wires verification.VerificationCascade (sole promoter),
    escalate.escalation_decision (move-class ladder), and pareto.ParetoFront (the certified deliverable).

    Policy per round:
      1. PROPOSE candidates for the current move class (different family / fusion / bigger-in-family).
      2. For each candidate, build a verification.Candidate whose val_outcomes drive the cheap COMPETENCE
         screen (vs a fixed floor) and whose certify_fn runs the frozen suite-level FDR comparison vs the
         CURRENT champion.
      3. cheap_screen prunes the incompetent; among survivors, spend Tier-3 peeks best-validation-first
         (select-then-bound). certify_tier3 is the ONLY thing that can certify a win over the champion.
      4. PROMOTE the best certified challenger (argmax certified sealed lower bound) to champion; reset and
         keep climbing. Else mark the class saturated and ask escalate.escalation_decision for the next rung;
         honest-stop at the top of the ladder.
    """

    def __init__(self, registry: Sequence[Encoder], arena: Arena, *, start_tag: str,
                 alpha: float = 0.1, theta_floor: float = 0.5, peek_budget: int = 12,
                 competence_ceiling: float = 0.90, allow_data_acquisition: bool = False):
        self.registry: Dict[str, Encoder] = {e.tag: e for e in registry}
        if start_tag not in self.registry:
            raise ValueError(f"start_tag {start_tag!r} not in registry")
        self.arena = arena
        self.start_tag = start_tag
        self.alpha = float(alpha)
        self.theta_floor = float(theta_floor)
        self.peek_budget = int(peek_budget)
        self.competence_ceiling = float(competence_ceiling)
        self.allow_data_acquisition = bool(allow_data_acquisition)

    # -- proposal policy: which encoders this move class proposes, given champion + what's been tried ----
    def _propose(self, move_class: str, champion: Encoder, tried: set) -> List[Tuple[str, Optional[str]]]:
        """Return [(candidate_tag, fuse_partner_or_None)] for the move class. Deterministic ordering.
        The model rung proposes the SMALLEST untried member of each other family (cheapest-first; the
        capacity rung is what scales the winning family afterward), preserving the family-swap-then-scale
        structure of the hand run."""
        if move_class == RUNG_FAMILY:
            by_family: Dict[str, List[Encoder]] = {}
            for e in self.registry.values():
                if e.family != champion.family and e.tag not in tried:
                    by_family.setdefault(e.family, []).append(e)
            out: List[Tuple[str, Optional[str]]] = []
            for fam in sorted(by_family):
                best = sorted(by_family[fam], key=lambda e: (e.scale_rank, e.params_m))[0]
                out.append((best.tag, None))
            return out
        if move_class == RUNG_FUSE:
            # authored featurizer: fuse the champion with EACH other-family encoder tried so far (the most
            # thorough "is any fusion complementary?" sweep). Deterministic order: biggest partner first.
            partners = [self.registry[t] for t in tried
                        if t in self.registry and self.registry[t].family != champion.family]
            partners.sort(key=lambda e: (-e.params_m, e.tag))
            return [(champion.tag, p.tag) for p in partners]
        if move_class == RUNG_SCALE:
            # scale up within the WINNING family: untried members with a higher scale_rank than champion
            bigger = [e for e in self.registry.values()
                      if e.family == champion.family and e.scale_rank > champion.scale_rank
                      and e.tag not in tried]
            return [(e.tag, None) for e in sorted(bigger, key=lambda e: e.scale_rank)]
        return []   # data_acquisition / unknown: no encoder move

    # -- the certified suite-level comparison of a measurement vs the current champion ------------------
    def _compare(self, meas: Dict[str, TaskMeasure],
                 champ_correct: Dict[str, Sequence[int]]) -> Comparison:
        per_task, pvals, lifts = [], [], []
        for t in self.arena.tasks:
            m = meas[t]
            p = self.arena.mcnemar(m.sealed_correct, champ_correct[t])
            base_acc = sum(champ_correct[t]) / len(champ_correct[t])
            lift = m.acc - base_acc
            per_task.append({"task": t, "acc": round(m.acc, 4), "base_acc": round(base_acc, 4),
                             "lift": round(lift, 4), "p_gt_champ": round(p, 4)})
            pvals.append(p)
            lifts.append(lift)
        rej = set(self.arena.bh(pvals, self.alpha))
        survivors = [self.arena.tasks[i] for i in range(len(self.arena.tasks))
                     if i in rej and lifts[i] > 0]
        mean_lift = sum(lifts) / len(lifts) if lifts else 0.0
        sealed_lb = self.arena.lower_bound([x for t in self.arena.tasks for x in meas[t].sealed_correct])
        certified = len(survivors) > 0 and mean_lift > 0
        return Comparison(survivors, len(self.arena.tasks), mean_lift, sealed_lb, per_task, certified)

    def _measure_state(self, encoder_tag: str):
        m = self.arena.measure(encoder_tag)
        correct = {t: m[t].sealed_correct for t in self.arena.tasks}
        lb = {t: self.arena.lower_bound(m[t].sealed_correct) for t in self.arena.tasks}
        return correct, lb

    def _pooled_lb(self, sealed: Dict[str, Sequence[int]]) -> float:
        return self.arena.lower_bound([x for t in self.arena.tasks for x in sealed[t]])

    def _gold_compare(self, champion_name: str, baseline_name: str,
                      champ_gold: Dict[str, Sequence[int]],
                      base_gold: Dict[str, Sequence[int]]) -> dict:
        """Confirm the final champion over the start baseline on the NEVER-PEEKED gold set, using the SAME
        frozen primitives as the climb (paired McNemar + BH-FDR + Clopper-Pearson). This is a single,
        post-hoc confirmation -- not part of the promotion loop -- so it carries no multiplicity from the
        sealed-test re-use the climb accrued."""
        per_task, pvals, lifts = [], [], []
        for t in self.arena.tasks:
            cg, bg = list(champ_gold[t]), list(base_gold[t])
            p = self.arena.mcnemar(cg, bg)
            ca = sum(cg) / len(cg) if cg else 0.0
            ba = sum(bg) / len(bg) if bg else 0.0
            per_task.append({"task": t, "champ_gold_acc": round(ca, 4), "base_gold_acc": round(ba, 4),
                             "lift": round(ca - ba, 4), "p_gt_base": round(p, 4), "n": len(cg)})
            pvals.append(p)
            lifts.append(ca - ba)
        rej = set(self.arena.bh(pvals, self.alpha))
        survivors = [self.arena.tasks[i] for i in range(len(self.arena.tasks))
                     if i in rej and lifts[i] > 0]
        mean_lift = sum(lifts) / len(lifts) if lifts else 0.0
        champ_lb = self.arena.lower_bound([x for t in self.arena.tasks for x in champ_gold[t]])
        base_lb = self.arena.lower_bound([x for t in self.arena.tasks for x in base_gold[t]])
        return {"champion": champion_name, "baseline": baseline_name,
                "n_tasks": len(self.arena.tasks),
                "gold_n": sum(len(champ_gold[t]) for t in self.arena.tasks),
                "survivors": survivors, "mean_lift": round(mean_lift, 4),
                "champion_gold_lb": round(champ_lb, 4), "baseline_gold_lb": round(base_lb, 4),
                "per_task": per_task,
                "confirmed": len(survivors) > 0 and mean_lift > 0,
                "note": ("champion FDR-beats the start baseline on a NEVER-PEEKED gold set, queried exactly "
                         "once after all climbing -- so the confirmation carries none of the sealed-test "
                         "re-use multiplicity accrued during the climb")}

    def _multiplicity_report(self, champion_name: str, start_correct: Dict[str, Sequence[int]],
                             champ_correct: Dict[str, Sequence[int]], sealed_comparisons: int,
                             gold_confirmation: Optional[dict]) -> dict:
        """Honest session-level multiplicity accounting. The climb spends M=`sealed_comparisons` certified
        looks at the ONE working sealed set; within each look BH-FDR(alpha) controls false discoveries, but
        the final champion is the argmax OVER those M looks, so its sealed lower bound is selection-biased
        upward. This report discloses M, the family-wise Bonferroni threshold alpha/M, and -- as a
        conservative sensitivity analysis -- re-states the FINAL champion vs the START baseline on the sealed
        set at BOTH the nominal alpha (BH-FDR) and the session-corrected alpha/M (Bonferroni over all looks),
        using only already-acquired correctness (no new encoder is measured). The multiplicity-FREE verdict
        is the gold confirmation, which never queried the sealed set at all."""
        n_tasks = len(self.arena.tasks)
        m = max(int(sealed_comparisons), 0)
        bonf_alpha = self.alpha / m if m > 0 else self.alpha
        per_task, pvals, lifts = [], [], []
        for t in self.arena.tasks:
            cc, sc = list(champ_correct[t]), list(start_correct[t])
            p = self.arena.mcnemar(cc, sc)
            ca = sum(cc) / len(cc) if cc else 0.0
            sa = sum(sc) / len(sc) if sc else 0.0
            per_task.append({"task": t, "champ_acc": round(ca, 4), "start_acc": round(sa, 4),
                             "lift": round(ca - sa, 4), "p_gt_start": round(p, 4)})
            pvals.append(p)
            lifts.append(ca - sa)
        fdr_rej = set(self.arena.bh(pvals, self.alpha))
        fdr_survivors = [self.arena.tasks[i] for i in range(n_tasks) if i in fdr_rej and lifts[i] > 0]
        bonf_survivors = [self.arena.tasks[i] for i in range(n_tasks)
                          if pvals[i] < bonf_alpha and lifts[i] > 0]
        champ_lb = self.arena.lower_bound([x for t in self.arena.tasks for x in champ_correct[t]])
        gold_ok = bool(gold_confirmation and gold_confirmation.get("confirmed"))
        return {"champion": champion_name, "baseline": self.start_tag, "n_tasks": n_tasks,
                "sealed_comparisons": m, "mcnemar_tests_total": m * n_tasks,
                "per_comparison_fdr_alpha": self.alpha, "session_bonferroni_alpha": round(bonf_alpha, 6),
                "champion_vs_start_sealed": per_task,
                "fdr_survivors_nominal": fdr_survivors,
                "bonferroni_survivors_session": bonf_survivors,
                "champion_sealed_lb": round(champ_lb, 4),
                "robust_to_session_multiplicity": len(bonf_survivors) > 0,
                "gold_independent_confirmation": gold_ok,
                "note": ("M sealed certifications were spent climbing; the champion is the argmax over them, "
                         "so its sealed lower bound is selection-biased. bonferroni_survivors_session "
                         "re-tests champion vs start at alpha/M as a conservative family-wise check; the "
                         "multiplicity-FREE verdict is the gold confirmation, which never read the sealed "
                         "set.")}

    @staticmethod
    def _val_pool(meas: Dict[str, TaskMeasure], tasks: Sequence[str]) -> List[float]:
        return [float(x) for t in tasks for x in meas[t].val_correct]

    def run(self) -> ResearchCertificate:
        # The cheap tiers only check COMPETENCE against a fixed floor; the frozen Tier-3 FDR comparison is the
        # sole champion-relative decider. theta is pinned to theta_floor for the whole run, never the champion.
        cascade = VerificationCascade(theta=self.theta_floor, alpha=0.05)
        champion = self.registry[self.start_tag]      # the registry Encoder used for family/scale logic
        champion_name = champion.tag                   # the current champion's display name (may be a fusion)
        champ_correct, champ_lb = self._measure_state(champion.tag)
        start_correct = {t: list(champ_correct[t]) for t in self.arena.tasks}   # frozen START baseline rows
        tried = {champion.tag}
        promotions: List[Promotion] = []
        rejections: List[Rejection] = []
        log: List[str] = []
        peeks_used = 0
        val_lb_hist: List[Optional[float]] = [self._pooled_lb(champ_correct)]
        move_class = RUNG_FAMILY
        move_class_path = [move_class]
        stop_reason = "loop exhausted"
        log.append(f"start champion={champion.tag} ({champion.family}) "
                   f"sealed_lb={val_lb_hist[0]:.3f} theta_floor={self.theta_floor}")

        while True:
            proposals = self._propose(move_class, champion, tried)

            # (1)+(2)+(3a): build each candidate, COMPETENCE-screen it, collect Tier-3-eligible survivors.
            eligible: List[Tuple[float, str, Optional[str], Comparison, Candidate,
                                 Dict[str, TaskMeasure]]] = []
            for tag, partner in proposals:
                cand_name = f"fuse[{tag}+{partner}]" if partner is not None else tag
                if cand_name in tried:
                    continue   # never re-spend a sealed peek on a candidate already measured this run
                if partner is not None:
                    meas = self.arena.fuse_measure(tag, partner)
                else:
                    meas = self.arena.measure(tag)
                tried.add(cand_name)
                tried.add(tag)
                cmp = self._compare(meas, champ_correct)
                val_pool = self._val_pool(meas, self.arena.tasks)
                val_mean = sum(val_pool) / len(val_pool) if val_pool else 0.0
                cand = Candidate(name=cand_name, sanity_ok=True, val_outcomes=val_pool,
                                 surrogate_score=None, certify_fn=(lambda c=cmp: c.as_cert()))
                screen = cascade.cheap_screen(cand)
                if screen.survived:
                    eligible.append((val_mean, cand_name, partner, cmp, cand, meas))
                    log.append(f"[{move_class}] {cand_name}: competence-screen SURVIVED "
                               f"(val_acc {val_mean:.3f}) -> Tier-3 eligible")
                else:
                    rejections.append(Rejection(move_class, cand_name, cmp.survivors, cmp.n, cmp.mean_lift,
                                                 f"pruned at {screen.killed_at} "
                                                 f"(val competence < floor {self.theta_floor})"))
                    log.append(f"[{move_class}] {cand_name}: competence-screen KILLED at {screen.killed_at}")

            # (3b): spend Tier-3 peeks on eligible candidates best-validation-first (select-then-bound).
            # certify_tier3 is THE ONLY champion-relative decision; record every certified winner.
            eligible.sort(key=lambda e: (-e[0], e[1]))
            certified: List[Tuple[float, str, Optional[str], Comparison, Dict[str, TaskMeasure]]] = []
            for val_mean, cand_name, partner, cmp, cand, meas in eligible:
                if peeks_used >= self.peek_budget:
                    log.append("peek budget exhausted -> stop spending sealed certifications")
                    break
                res = cascade.certify_tier3(cand)   # SOLE PROMOTER (assert_only_tier3_promotes guards)
                peeks_used += 1
                if res.promoted:
                    certified.append((cmp.sealed_lb, cand_name, partner, cmp, meas))
                    log.append(f"[{move_class}] {cand_name}: frozen Tier-3 CERTIFIED over {champion_name} "
                               f"(survivors {len(cmp.survivors)}/{cmp.n}, mean_lift {cmp.mean_lift:+.3f}, "
                               f"sealed_lb {cmp.sealed_lb:.3f})")
                else:
                    rejections.append(Rejection(move_class, cand_name, cmp.survivors, cmp.n, cmp.mean_lift,
                                                 f"frozen Tier-3 did not certify over champion "
                                                 f"{champion_name} ({len(cmp.survivors)} FDR survivors, "
                                                 f"mean_lift {cmp.mean_lift:+.3f})"))
                    log.append(f"[{move_class}] reject {cand_name} (not certified over {champion_name}: "
                               f"{len(cmp.survivors)}/{cmp.n}, mean_lift {cmp.mean_lift:+.3f})")

            # (4): promote the best certified challenger by its certified sealed lower bound (argmax over
            # already-certified winners; NOT a fresh test). Others that certified are superseded.
            if certified:
                certified.sort(key=lambda e: (-e[0], e[1]))
                best_lb, best_name, best_partner, best_cmp, best_meas = certified[0]
                from_tag = champion_name
                promotions.append(Promotion(move_class, from_tag, best_name, best_cmp.survivors,
                                            best_cmp.n, best_cmp.mean_lift))
                for lb2, name2, _p2, cmp2, _m2 in certified[1:]:
                    rejections.append(Rejection(move_class, name2, cmp2.survivors, cmp2.n, cmp2.mean_lift,
                                                 f"certified over prior champion {from_tag} but superseded "
                                                 f"by {best_name} (sealed lower bound {best_lb:.3f} > "
                                                 f"{lb2:.3f})"))
                    log.append(f"[{move_class}] supersede {name2} (sealed_lb {lb2:.3f}) "
                               f"by {best_name} (sealed_lb {best_lb:.3f})")
                # update champion bookkeeping. A fusion deliverable updates the correctness the next round
                # compares against (and the display name), but the registry Encoder used for scaling stays
                # the best single encoder (you cannot "scale" a fused representation within one family).
                champion_name = best_name
                if best_partner is None and best_name in self.registry:
                    champion = self.registry[best_name]
                champ_correct = {t: best_meas[t].sealed_correct for t in self.arena.tasks}
                champ_lb = {t: self.arena.lower_bound(best_meas[t].sealed_correct) for t in self.arena.tasks}
                log.append(f"[{move_class}] PROMOTE {from_tag} -> {best_name} "
                           f"(survivors {len(best_cmp.survivors)}/{best_cmp.n}, "
                           f"mean_lift {best_cmp.mean_lift:+.3f}, sealed_lb {best_lb:.3f})")
                val_lb_hist.append(self._pooled_lb(champ_correct))
                continue   # keep climbing from the new champion on the same move class

            # class saturated (no certified challenger) -> ask the escalate ladder for the next rung. A
            # saturated full sweep IS a plateau, so force the plateau signal; the ladder stops at the top.
            val_lb_hist.append(None)
            decision = ESC.escalation_decision(val_lb_hist, move_class, ESC.DEFAULT_K,
                                                budget_left=self.peek_budget - peeks_used,
                                                allow_data_acquisition=self.allow_data_acquisition)
            log.append(f"escalate: {decision}")
            if decision.decision == ESC.STOP or decision.to_class is None:
                stop_reason = decision.reason
                break
            move_class = decision.to_class
            move_class_path.append(move_class)

        # the residual DATA ceiling: tasks where even the FINAL (best) representation we could climb to still
        # cannot reach competence. This is an ABSOLUTE, baseline-independent criterion (final champion sealed
        # accuracy < the competence floor) -- it does not matter which intermediate move last touched a task;
        # what matters for the deliverable is that the best representation available is still below competence,
        # so the residual gap is a data/label-budget ceiling, not a representation one.
        champ_acc = {t: sum(champ_correct[t]) / len(champ_correct[t]) for t in self.arena.tasks}
        data_ceiling = sorted(t for t in self.arena.tasks if champ_acc[t] < self.competence_ceiling)

        # certified Pareto front over the start + promoted champions (accuracy lower bound vs encoder cost)
        pareto_cands, seen = [], set()
        chain = [self.start_tag] + [p.to_tag for p in promotions]
        for tag in chain:
            if tag in seen or tag not in self.registry:
                continue
            seen.add(tag)
            correct, lb = self._measure_state(tag)
            enc = self.registry[tag]
            # every champion in the climb chain has a real frozen CP lower bound on the sealed test, so each
            # is an honest certified deliverable (accuracy lower bound vs parameter cost).
            pareto_cands.append(ParetoCandidate(name=tag, metric_lb=min(lb[t] for t in self.arena.tasks),
                                                 latency_ms=enc.params_m, cost_usd=enc.params_m / 1000.0,
                                                 ece=0.0, certified=True))
        front = ParetoFront(pareto_cands, axes=[("metric_lb", MAXIMIZE), ("cost_usd", MINIMIZE)])
        front_tags = [c.name for c in front.nondominated()]

        # GOLD CONFIRMATION (read EXACTLY ONCE, after all climbing): if the arena supplies a disjoint
        # never-peeked gold partition, confirm the FINAL champion (display name -- may be a fusion) over the
        # START baseline on it. The climb never queried gold, so this single post-hoc comparison certifies the
        # final lift free of the sealed-test re-use multiplicity that the climb accrued. No gold set -> None.
        gold_confirmation = None
        champ_gold = self.arena.gold_measure(champion_name)
        if champ_gold is not None:
            base_gold = self.arena.gold_measure(self.start_tag)
            gold_confirmation = self._gold_compare(champion_name, self.start_tag, champ_gold, base_gold)
            log.append(
                f"gold-confirmation on a NEVER-PEEKED set (n={gold_confirmation['gold_n']}): "
                f"{champion_name} vs {self.start_tag} -> "
                f"{len(gold_confirmation['survivors'])}/{gold_confirmation['n_tasks']} FDR survivors, "
                f"champion_gold_lb {gold_confirmation['champion_gold_lb']:.3f} "
                f"(baseline {gold_confirmation['baseline_gold_lb']:.3f}), "
                f"confirmed={gold_confirmation['confirmed']}")

        # SESSION MULTIPLICITY: disclose how many sealed certifications the climb spent and how robust the
        # final champion-vs-start sealed win is to a family-wise Bonferroni correction over those looks.
        multiplicity = self._multiplicity_report(champion_name, start_correct, champ_correct,
                                                  peeks_used, gold_confirmation)
        log.append(
            f"session-multiplicity: {multiplicity['sealed_comparisons']} sealed certifications "
            f"({multiplicity['mcnemar_tests_total']} McNemar tests); Bonferroni alpha/M="
            f"{multiplicity['session_bonferroni_alpha']:.4f} -> "
            f"{len(multiplicity['bonferroni_survivors_session'])}/{multiplicity['n_tasks']} champion-vs-start "
            f"survivors (FDR-nominal {len(multiplicity['fdr_survivors_nominal'])}); "
            f"robust={multiplicity['robust_to_session_multiplicity']}, "
            f"gold-independent={multiplicity['gold_independent_confirmation']}")

        return ResearchCertificate(
            champion=champion.tag, champion_family=champion.family, promotions=promotions,
            rejections=rejections, move_class_path=move_class_path, stop_reason=stop_reason,
            sealed_lb={t: round(champ_lb[t], 4) for t in self.arena.tasks},
            sealed_acc={t: round(champ_acc[t], 4) for t in self.arena.tasks},
            pareto_front=front_tags, pareto_report=front.report(),
            data_ceiling_tasks=data_ceiling, peeks_used=peeks_used, log=log,
            gold_confirmation=gold_confirmation, multiplicity=multiplicity)


__all__ = ["Encoder", "TaskMeasure", "Arena", "Comparison", "Promotion", "Rejection",
           "ResearchCertificate", "ReprResearcher",
           "RUNG_FAMILY", "RUNG_FUSE", "RUNG_SCALE", "RUNG_DATA"]
