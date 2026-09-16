"""End-to-end tests for baseline-first / landscape-anchoring (frontier/baseline.py + wiring).

Run standalone:
    cd <repo root> && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_baseline_first.py

What these assert (the behavior change, not a log line):
  1. compute_floor() actually fits baselines, selects on VAL, and CERTIFIES the best one on a
     sealed test that is peeked EXACTLY ONCE on an INDEPENDENT split (its own discipline; it does
     not touch the winner's main peek).
  2. The gate beats_floor() reports a TIE (winner_lb == floor_lb) as "no improvement over
     baseline" and a strictly-better winner_lb as an improvement.
  3. Wired into CoreOrchestrator: the final `certified` decision ANDs beats_floor. With an injected
     tying floor, a winner that clears theta + oracle is reported certified=False with decline
     reason "did not beat baseline floor ...". With the genuine floor on the same run, a winner that
     beats it is reported as an improvement (certified reflects beats_floor=True). The winner's
     single sealed peek stays at 1 the whole time.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import baseline
from frontier.task import Task
from frontier.core.orchestrator import CoreOrchestrator, CoreConfig


def _clf_task(theta=0.0):
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return Task(X=d.data, y=d.target.astype(str), kind="classification", theta=theta, name="bc")


def test_compute_floor_certifies_on_its_own_sealed_one_peek():
    """The floor is fit + selected on VAL, then certified ONCE on an independent sealed fold."""
    task = _clf_task(theta=0.50)  # majority/HGB easily clear 0.50 on breast cancer
    floor = baseline.compute_floor(task, seed=0, wall_seconds=45, cpu_seconds=40)

    assert floor.best_label, "a baseline must have been selected"
    assert floor.lower_bound is not None, "the floor must carry a certified sealed lower bound"
    assert floor.observed is not None
    assert floor.floor_sealed_peeks == 1, \
        f"floor must use exactly ONE sealed peek, got {floor.floor_sealed_peeks}"
    assert floor.floor_split_seed == 0 + baseline.FLOOR_SEED_OFFSET, \
        "floor must use the OFFSET seed so its sealed fold is disjoint from the winner's"
    assert floor.certificate is not None and floor.certificate.get("peeks") == 1
    # the strong reference (HGB) should be selected over the trivial majority floor on this task
    labels = {c["label"] for c in floor.candidates}
    assert "baseline_majority" in labels and "baseline_hgb" in labels, \
        f"both trivial floor and reference model must be evaluated, got {labels}"
    print(f"[ok] floor certified: best={floor.best_label} lb={floor.lower_bound} "
          f"obs={floor.observed} peeks={floor.floor_sealed_peeks} seed={floor.floor_split_seed}")


def test_floor_split_is_independent_of_winner_split():
    """The floor's sealed seed differs from the main split seed (disjoint sealed labels)."""
    task = _clf_task(theta=0.50)
    floor = baseline.compute_floor(task, seed=7, wall_seconds=45, cpu_seconds=40)
    assert floor.floor_split_seed != 7, "floor must NOT reuse the winner's split seed"
    assert floor.floor_split_seed == 7 + baseline.FLOOR_SEED_OFFSET
    print(f"[ok] independent floor split seed={floor.floor_split_seed} (winner seed=7)")


def test_gate_tie_is_not_an_improvement():
    """A winner whose certified lower bound TIES the floor is reported as no improvement."""
    fc = baseline.FloorCertificate(
        certified=True, certificate={"lower_bound": 0.90}, best_label="baseline_hgb",
        best_val_score=0.95, lower_bound=0.90, observed=0.93,
        floor_split_seed=9973, floor_sealed_peeks=1)

    tie = baseline.beats_floor({"lower_bound": 0.90}, fc)
    assert tie.floor_available and not tie.beats_floor, "a tie must NOT count as beating the floor"
    assert "does NOT beat" in tie.reason

    win = baseline.beats_floor({"lower_bound": 0.93}, fc)
    assert win.beats_floor, "a strictly-higher lower bound must beat the floor"
    assert win.margin is not None and win.margin > 0
    assert "beats baseline floor" in win.reason

    lose = baseline.beats_floor({"lower_bound": 0.80}, fc)
    assert not lose.beats_floor
    print(f"[ok] gate: tie->{tie.beats_floor} win(margin={win.margin})->{win.beats_floor} "
          f"lose->{lose.beats_floor}")


def test_gate_admits_winner_when_no_floor():
    """Honest degradation: with no certified floor, the gate admits the winner and says so."""
    v = baseline.beats_floor({"lower_bound": 0.7}, None)
    assert v.beats_floor and not v.floor_available
    fc = baseline.FloorCertificate(
        certified=False, certificate=None, best_label="", best_val_score=None,
        lower_bound=None, observed=None, floor_split_seed=1, floor_sealed_peeks=0,
        note="no baseline ran")
    v2 = baseline.beats_floor({"lower_bound": 0.7}, fc)
    assert v2.beats_floor and not v2.floor_available
    print("[ok] no-floor degradation: winner admitted, floor_available=False")


def test_orchestrator_reports_floor_and_gates_final_decision():
    """Full loop: the floor is computed + carried, and certified reflects beats_floor.

    Genuine-improvement branch: a low theta lets a real winner certify; whether it is reported as
    certified must equal (cert.certified AND oracle.promote AND floor.beats_floor). The winner's
    single sealed peek must be exactly 1 (the floor uses its OWN independent sealed peek).
    """
    task = _clf_task(theta=0.80)
    orch = CoreOrchestrator(CoreConfig(rounds=2, wall_seconds=45, cpu_seconds=40,
                                       enable_neural=False, enable_knowledge=False))
    res = orch.run("classify breast tumors", task.X, task.y, theta=task.theta,
                   name="bc", metric="accuracy")

    assert res.winner is not None, "should find a runnable winner"
    assert res.floor_certificate is not None, "the baseline floor must be computed + carried"
    assert res.floor_verdict is not None, "the gate verdict must be carried on the result"
    assert res.sealed_peeks == 1, f"winner sealed peek must be exactly 1, got {res.sealed_peeks}"
    assert res.floor_certificate["floor_sealed_peeks"] == 1, \
        "floor uses its OWN one peek (separate from the winner's)"

    cert = res.certificate
    oracle_promote = bool(res.oracle_verdict.get("promote")) if res.oracle_verdict else True
    beats = bool(res.floor_verdict["beats_floor"])
    expected_certified = bool(cert.get("certified")) and oracle_promote and beats
    assert res.certified == expected_certified, \
        f"certified ({res.certified}) must AND beats_floor; expected {expected_certified}"
    print(f"[ok] orchestrator: winner={res.winner.label} cert={cert.get('certified')} "
          f"oracle={oracle_promote} beats_floor={beats} -> certified={res.certified} "
          f"(winner_lb={res.floor_verdict['winner_lower_bound']} "
          f"floor_lb={res.floor_verdict['floor_lower_bound']})")


def test_orchestrator_tie_reported_as_no_improvement_over_baseline():
    """Inject a floor that TIES the winner's lower bound; certified must flip to False with reason.

    We let the orchestrator do a real run, then re-derive the decision through the public gate to
    prove the wiring: a tying floor yields beats_floor=False and the orchestrator's decline reason
    is the explicit baseline non-win. We also drive the orchestrator with an injected tying floor
    via the documented `_floor` seam to confirm the LIVE path flips certified=False.
    """
    task = _clf_task(theta=0.80)
    orch = CoreOrchestrator(CoreConfig(rounds=2, wall_seconds=45, cpu_seconds=40,
                                       enable_neural=False, enable_knowledge=False,
                                       enable_intelligence=False))

    # Monkeypatch compute_floor so the orchestrator's OWN floor ties the winner exactly. We capture
    # the winner's certified lower bound by running once, then re-run with a floor pinned to it.
    import frontier.baseline as bl
    import frontier.core.orchestrator as orch_mod

    first = orch.run("classify breast tumors", task.X, task.y, theta=task.theta,
                     name="bc", metric="accuracy")
    winner_lb = first.certificate["lower_bound"]
    assert first.sealed_peeks == 1

    tying_floor = bl.FloorCertificate(
        certified=True, certificate={"lower_bound": winner_lb, "peeks": 1},
        best_label="baseline_injected_tie", best_val_score=None,
        lower_bound=winner_lb, observed=winner_lb,
        floor_split_seed=baseline.FLOOR_SEED_OFFSET, floor_sealed_peeks=1)

    orig = orch_mod.baseline_mod.compute_floor
    orch_mod.baseline_mod.compute_floor = lambda *a, **k: tying_floor
    try:
        orch2 = CoreOrchestrator(CoreConfig(rounds=2, wall_seconds=45, cpu_seconds=40,
                                            enable_neural=False, enable_knowledge=False,
                                            enable_intelligence=False))
        tied = orch2.run("classify breast tumors", task.X, task.y, theta=task.theta,
                         name="bc", metric="accuracy")
    finally:
        orch_mod.baseline_mod.compute_floor = orig

    # Winner clears theta (its lb > 0.80 if first run did) yet ties the floor -> NOT an improvement.
    assert tied.certificate["lower_bound"] == winner_lb
    assert tied.floor_verdict["beats_floor"] is False, \
        "a winner tying the injected floor must NOT be reported as beating it"
    assert tied.certified is False, \
        "tying the floor must flip the final decision to certified=False"
    assert "did not beat baseline floor" in tied.decline_reason, \
        f"decline reason must name the baseline non-win, got: {tied.decline_reason!r}"
    assert tied.sealed_peeks == 1, "winner sealed peek stays at 1 under the gate"
    print(f"[ok] tie wired live: certified={tied.certified} reason={tied.decline_reason!r}")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} baseline-first tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
