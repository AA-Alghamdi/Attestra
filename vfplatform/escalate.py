"""PLATEAU DETECTION + STRATEGY ESCALATION -- a PURE proposal-policy module (build-order gap-fill).

The recursive /goal cycle today sweeps a FIXED catalog of (family, hyperparameter) moves every round.
When the validation lower bound stalls, the loop's only stagnation handling is to keep re-proposing from
the SAME model-capacity grid (rounds 4-5 today re-sweep the zoo for no bound gain), then -- once any time
budget is exhausted -- honest-stop. That wastes the budget on a saturated axis: if MORE CAPACITY is not
lifting the bound, the bottleneck is almost always the *representation* (features) or the *data*, not yet
another tree-depth value.

This module makes the policy decision the loop is missing: WHEN the val lower bound has plateaued for K
rounds, switch the KIND of move (the "move class") instead of re-sweeping the same class. The escalation
ladder follows the standard practitioner ordering of cheapest-first interventions:

    model  ->  features  ->  capacity  ->  data_acquisition  ->  stop

  * model            -- the current class: sweep families/hyperparameters within the catalog.
  * features         -- feature engineering (interactions, scaling, encodings, text n-gram range). CHEAP,
                        no new labels, no new compute tier; the first thing to try when capacity sweeps
                        saturate. (The KNOWN GAP: the catalog has no feature moves yet; this names the axis
                        the loop should add, and the policy emits the decision so the loop can route to it.)
  * capacity         -- jump to a higher-capacity family/compute tier (deep_model / wider hyperparameters)
                        BEYOND the default sweep. More expensive than features (compute), cheaper than labels.
  * data_acquisition -- acquire more labeled rows (the loop's acquire_fn path). The most expensive lever
                        (labels cost money/time), so it is last before stopping, and only when budget allows.
  * stop             -- no cheaper lever left and/or no budget: honest-stop (the loop preserves the sealed peek).

CONTRACT / WHAT THIS MODULE IS NOT:
  * PURE: no import of loop.py, harness.py, sealed.py, or science.py. It takes a val_lb history + the
    current move class + bookkeeping scalars and returns ONE string decision. No I/O, no global state.
  * It NEVER touches a certificate, the sealed peek, select-then-bound, or any threshold. It only influences
    which KIND of move the PROPOSE node should make next. A wrong decision here can at worst waste budget;
    it can never produce a false certificate (the frozen sealed gate is downstream and unaffected).
  * Deterministic. Same inputs -> same decision. No randomness, no clock, no network.

The plateau test is on the VALIDATION LOWER BOUND history (the loop's true objective is to raise the bound,
not the val point -- see loop._running_val_lb), so a val point that wiggles while the bound is flat still
counts as a plateau and correctly escalates.
"""
from __future__ import annotations

from dataclasses import dataclass

# The escalation ladder, cheapest-intervention-first. Index in this tuple == rung; escalate_from() walks it.
MOVE_CLASSES = ("model", "features", "capacity", "data_acquisition")

# Decision strings the policy may return (the loop routes on these). "continue" == stay on the current class.
CONTINUE = "continue"
ESCALATE_TO_FEATURES = "escalate_to_features"
ESCALATE_TO_CAPACITY = "escalate_to_capacity"
ESCALATE_TO_DATA = "escalate_to_data_acquisition"
STOP = "stop"

# class name -> the decision string that escalates INTO that class.
_ESCALATE_DECISION = {
    "features": ESCALATE_TO_FEATURES,
    "capacity": ESCALATE_TO_CAPACITY,
    "data_acquisition": ESCALATE_TO_DATA,
}

# Default plateau window: how many consecutive rounds of no val-LB improvement before we escalate the class.
# Mirrors loop.K_NO_IMPROVE (3) so the policy fires at the SAME stagnation point the loop already tolerates,
# rather than introducing a second, inconsistent stagnation clock.
DEFAULT_K = 3
# Improvement smaller than EPS (in metric units) is "no improvement" -- guards against float jitter in the
# bound passing as progress. 1e-9 matches loop._running_val_lb's improvement test (after_lb > before + 1e-9).
DEFAULT_EPS = 1e-9


@dataclass(frozen=True)
class EscalationDecision:
    """The full policy output. `decision` is the one-of-five string the caller routes on; the rest is the
    auditable WHY (so a round log / reviewer can see the plateau evidence behind a class switch)."""
    decision: str
    plateaued: bool
    rounds_since_improve: int
    from_class: str
    to_class: str | None         # the move class to switch to (None when continuing or stopping)
    reason: str

    def __str__(self):  # compact, log-friendly
        return f"{self.decision} (from={self.from_class} since_improve={self.rounds_since_improve}: {self.reason})"


def is_plateau(val_lb_history, k=DEFAULT_K, eps=DEFAULT_EPS):
    """True iff the validation LOWER BOUND has not improved by more than `eps` for the last `k` rounds.

    val_lb_history: chronological list of the winner's val lower bound per round (most recent LAST). Entries
    may be None (a round where no bound could be computed -- e.g. no finished candidate); a None is treated
    as 'no improvement' for that step (it cannot be progress).

    Definition of "no improvement over the last k rounds": across the last (k+1) recorded bounds, no step
    rose by more than eps above the running max seen up to that point. Equivalently: the running max of the
    bound has been flat for k consecutive steps. Needs at least k+1 entries to assert a k-round plateau; with
    fewer, returns False (not enough evidence -- never escalate prematurely)."""
    hist = [h for h in val_lb_history]                 # shallow copy; keep Nones in place for indexing
    if k < 1 or len(hist) < k + 1:
        return False
    # Walk the running max; count trailing consecutive steps with no >eps rise above the prior running max.
    running = None
    no_improve_run = 0
    for v in hist:
        if v is None:
            # a missing bound is not progress: it extends the stagnation run, running max unchanged
            if running is not None:
                no_improve_run += 1
            continue
        if running is None:
            running = float(v)
            no_improve_run = 0                          # first real bound: establishes the baseline
            continue
        if float(v) > running + eps:
            running = float(v)
            no_improve_run = 0                          # genuine improvement resets the clock
        else:
            no_improve_run += 1
    return no_improve_run >= k


def next_class(current_move_class):
    """The next rung UP the escalation ladder from `current_move_class`, or None if already at the top
    (data_acquisition) or the class is unknown. Pure lookup over MOVE_CLASSES."""
    cls = current_move_class if current_move_class in MOVE_CLASSES else "model"
    i = MOVE_CLASSES.index(cls)
    return MOVE_CLASSES[i + 1] if i + 1 < len(MOVE_CLASSES) else None


def escalation_decision(val_lb_history, current_move_class, rounds_since_improve, budget_left,
                        *, k=DEFAULT_K, eps=DEFAULT_EPS, allow_data_acquisition=True):
    """THE policy entry point. Decide what KIND of move the cycle should make next.

    val_lb_history       chronological list of the winner's validation LOWER BOUND per round (most recent
                         last). May contain None for rounds with no computable bound. This is the plateau
                         signal -- the loop's true objective (raise the bound), not the val point.
    current_move_class   one of MOVE_CLASSES ("model" | "features" | "capacity" | "data_acquisition").
                         Unknown values are treated as "model" (the default starting class).
    rounds_since_improve the loop's own no-improve counter (loop.no_improve). Used as a fallback plateau
                         signal: a plateau fires if EITHER the history test trips OR this counter >= k. This
                         keeps the policy in agreement with the loop's existing stagnation clock even when a
                         caller passes a short/empty history.
    budget_left          remaining experiment/label budget (int or float). <= 0 means no budget: the only
                         honest move left is to stop. For data_acquisition specifically, budget_left also
                         gates whether acquiring labels is affordable.
    k, eps               plateau window / improvement epsilon (default to match loop.K_NO_IMPROVE).
    allow_data_acquisition  False when the caller has no acquisition source wired (loop's acquire_fn is None):
                         the ladder then ends at capacity, and a plateau at/after capacity -> stop.

    Returns an EscalationDecision (str(decision) is one of CONTINUE / ESCALATE_TO_* / STOP). The loop reads
    .decision; the rest is the auditable rationale. NEVER raises on ordinary inputs.

    Decision rule (cheapest-lever-first):
      1. No budget left            -> stop (regardless of plateau: nothing can be run).
      2. Not plateaued             -> continue (the current class is still making progress; don't churn).
      3. Plateaued, rung available -> escalate to the next rung UP the ladder (model->features->capacity->
                                      data_acquisition), skipping data_acquisition if it is disallowed.
      4. Plateaued, top of ladder  -> stop (every cheaper lever has been tried; honest-stop, the loop keeps
                                      the sealed peek).
    """
    cur = current_move_class if current_move_class in MOVE_CLASSES else "model"
    # robust scalar coercions (a misbehaving caller never breaks the policy)
    try:
        rsi = int(rounds_since_improve)
    except (TypeError, ValueError):
        rsi = 0
    try:
        budget = float(budget_left)
    except (TypeError, ValueError):
        budget = 0.0
    hist = list(val_lb_history or [])

    # (1) Out of budget: the only honest action is to stop. No move of any class can be executed.
    if budget <= 0:
        return EscalationDecision(STOP, plateaued=True, rounds_since_improve=rsi, from_class=cur,
                                  to_class=None,
                                  reason="no budget left; cannot run any further move -> honest stop")

    # plateau iff the history test trips OR the loop's own counter already hit the window (belt-and-braces:
    # agree with the loop's existing stagnation clock even when history is short/empty).
    plateaued = is_plateau(hist, k=k, eps=eps) or (rsi >= k)
    if not plateaued:
        return EscalationDecision(CONTINUE, plateaued=False, rounds_since_improve=rsi, from_class=cur,
                                  to_class=None,
                                  reason=f"val_lb still improving within the last {k} rounds; keep sweeping "
                                         f"the '{cur}' class")

    # (3)/(4) plateaued with budget: climb to the next cheaper-first rung, skipping data_acquisition if the
    # caller has no acquisition source. If no rung remains, honest-stop.
    nxt = next_class(cur)
    while nxt == "data_acquisition" and not allow_data_acquisition:
        nxt = next_class(nxt)        # ladder ends at capacity when acquisition is unavailable -> nxt = None
    if nxt is None:
        return EscalationDecision(STOP, plateaued=True, rounds_since_improve=rsi, from_class=cur,
                                  to_class=None,
                                  reason=f"plateaued at the top of the escalation ladder ('{cur}'); every "
                                         f"cheaper lever exhausted -> honest stop")
    return EscalationDecision(_ESCALATE_DECISION[nxt], plateaued=True, rounds_since_improve=rsi,
                              from_class=cur, to_class=nxt,
                              reason=f"val_lb plateaued for >= {k} rounds on the '{cur}' class; the same-class "
                                     f"sweep is saturated -> switch KIND of move to '{nxt}'")


# --------------------------------------------------------------------------------------------------------
# Self-test / runnable example (mirrors the module self-test convention: `python vfplatform/escalate.py`).
# --------------------------------------------------------------------------------------------------------
def _demo():
    print("escalate.py -- plateau detection + strategy escalation (PURE policy)\n")
    cases = [
        ("climbing val_lb, model class",
         [0.60, 0.64, 0.68, 0.72], "model", 0, 50),
        ("flat val_lb at K rounds, model class",
         [0.70, 0.70, 0.70, 0.70], "model", 3, 50),
        ("flat after features class -> capacity",
         [0.70, 0.70, 0.70, 0.70], "features", 3, 50),
        ("flat after capacity -> data acquisition",
         [0.70, 0.70, 0.70, 0.70], "capacity", 3, 50),
        ("flat at top of ladder -> stop",
         [0.70, 0.70, 0.70, 0.70], "data_acquisition", 3, 50),
        ("flat but NO budget -> stop",
         [0.70, 0.70, 0.70, 0.70], "model", 3, 0),
    ]
    for label, hist, cls, rsi, budget in cases:
        d = escalation_decision(hist, cls, rsi, budget)
        print(f"  {label}")
        print(f"      hist={hist} class={cls} budget={budget}")
        print(f"      -> {d.decision}\n")


if __name__ == "__main__":
    _demo()
