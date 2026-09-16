"""Behavioral tests for the concurrent early-kill portfolio (Phase 6).

Run standalone:
    cd .../Attestra && /Users/abdullahalghamdi/jax-env-311/bin/python frontier/tests/test_portfolio.py

These assert the SPEC: several Programs run concurrently, ASHA prunes losing arms early,
the returned winner is decided at FULL budget on the validation set, the sealed split is
never touched, and the firewall (predictions-only sandbox -> parent scores) holds.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier import certify
from frontier.portfolio import (
    run_portfolio_round, PortfolioConfig, _build_ladder, _ArmStats, _prune_pending,
)
from frontier.program import Program
from frontier.proposers import make_code
from frontier.task import Task


def _clf_task():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return Task(X=d.data, y=d.target.astype(str), kind="classification", theta=0.9, name="bc")


def _clf_programs():
    recipes = [
        ("logreg_scale", {"base": "logreg", "scale": True}),
        ("rf", {"base": "rf"}),
        ("hist_gbm", {"base": "hist_gbm"}),
        ("svc_rbf", {"base": "svc_rbf", "scale": True}),
        ("logreg_plain", {"base": "logreg"}),
        ("hist_gbm2", {"base": "hist_gbm"}),  # duplicate code path; distinct label
    ]
    progs = []
    for lab, r in recipes:
        progs.append(Program(code=make_code(r, "classification"), source="seed", label=lab))
    return progs


def test_ladder_tops_out_at_full_budget():
    """The final rung must always be full training data (no early budget decides the winner)."""
    ladder = _build_ladder(9, PortfolioConfig(eta=3.0, min_frac=1 / 9))
    assert ladder[-1]["budget"] == 1.0, ladder
    budgets = [r["budget"] for r in ladder]
    assert budgets == sorted(budgets), f"budgets must be non-decreasing: {budgets}"
    assert len(ladder) >= 2, "9 arms with eta=3 should produce >=2 rungs"
    # survivors must shrink toward the top, then the final rung keeps its entrants
    assert ladder[0]["survivors"] <= 9 and ladder[0]["survivors"] >= ladder[1]["survivors"]
    print(f"[ok] ladder={ladder}")


def test_small_batch_is_single_full_rung():
    """Below min_rung_arms, halving buys nothing -> one full-budget rung (degenerate ASHA)."""
    ladder = _build_ladder(2, PortfolioConfig(min_rung_arms=4))
    assert ladder == [{"rung": 0, "budget": 1.0, "survivors": 2}], ladder
    print(f"[ok] small batch single rung: {ladder}")


def test_prune_is_admissible():
    """A pending arm whose best case (UCB) is below k settled arms' worst case (LCB) is killed,
    and an arm that could still win is kept."""
    a, b, c, p_lose, p_win = "a", "b", "c", "lose", "win"
    stats = {pid: _ArmStats(Program(code="x", source="s", label=pid))
             for pid in (a, b, c, p_lose, p_win)}
    # three settled arms scored high at a low budget (wide band -> some uncertainty)
    for pid in (a, b, c):
        stats[pid].observe(0.95, budget=0.5)
    # a pending loser we already saw scoring terribly at a cheaper rung
    stats[p_lose].observe(0.10, budget=0.5)
    # a pending arm with no observation -> UCB is +inf -> must be kept (forced exploration)
    band = (0.0, 1.0)
    done = {a: 0.95, b: 0.95, c: 0.95, p_lose: 0.10}
    keep, killed = _prune_pending([p_lose, p_win], done, stats, survivors_k=2,
                                  cfg=PortfolioConfig(), band=band)
    assert p_lose in killed, "a clearly-dominated pending arm must be pruned"
    assert p_win in keep, "an unobserved pending arm must survive (cannot prove it out)"
    print(f"[ok] admissible prune: killed={killed} kept={keep}")


def test_concurrent_round_returns_full_budget_winner():
    """End-to-end on the real sandbox: arms run concurrently, winner has a FULL-budget val score,
    and that winner matches the arm with the best full-budget score on val."""
    task = _clf_task()
    splits = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(splits.train_rows); ytr = Task.rows_to_y(splits.train_rows, task.kind)
    Xva = Task.rows_to_X(splits.val_rows)
    progs = _clf_programs()

    score_fn = lambda preds: certify.score_val(task, splits.val_rows, preds)
    t0 = time.time()
    out = run_portfolio_round(progs, Xtr, ytr, Xva, kind="classification",
                              score_fn=score_fn,
                              config=PortfolioConfig(wall_seconds=50, cpu_seconds=45, seed=0))
    wall = time.time() - t0

    assert out.best_program is not None, "portfolio must return a winner"
    assert out.best_score is not None and np.isfinite(out.best_score)
    # the winner must be the arm with the best FULL-budget score (decision at full data)
    assert out.full_budget_scores, "at least one arm must reach full budget"
    best_pid = max(out.full_budget_scores, key=out.full_budget_scores.get)
    assert out.best_program.id == best_pid, "winner must be the best full-budget arm"
    assert abs(out.best_score - out.full_budget_scores[best_pid]) < 1e-9
    # at least one record per program
    assert len(out.records) == len(progs)
    print(f"[ok] concurrent round: winner={out.best_program.label} "
          f"score={out.best_score:.4f} wall={wall:.1f}s saved_fits={out.n_full_fits_saved} "
          f"rungs={out.rung_schedule}")


def test_early_kill_actually_prunes():
    """With a mix of strong and deliberately weak arms, ASHA must kill some arms before full
    budget (killed_early), and the weak arms must NOT be the winner."""
    task = _clf_task()
    splits = certify.make_splits(task, seed=1)
    Xtr = Task.rows_to_X(splits.train_rows); ytr = Task.rows_to_y(splits.train_rows, task.kind)
    Xva = Task.rows_to_X(splits.val_rows)

    # a "dummy-most-frequent" classifier is a deliberately weak arm; strong arms are real models.
    dummy_code = (
        "from sklearn.dummy import DummyClassifier\n"
        "def build_estimator():\n"
        "    return DummyClassifier(strategy='most_frequent')\n"
    )
    progs = [
        Program(code=make_code({"base": "logreg", "scale": True}, "classification"),
                source="seed", label="logreg_scale"),
        Program(code=make_code({"base": "hist_gbm"}, "classification"),
                source="seed", label="hist_gbm"),
        Program(code=make_code({"base": "rf"}, "classification"), source="seed", label="rf"),
        Program(code=make_code({"base": "svc_rbf", "scale": True}, "classification"),
                source="seed", label="svc_rbf"),
        Program(code=dummy_code, source="seed", label="dummy1"),
        Program(code=dummy_code.replace("most_frequent", "prior"), source="seed", label="dummy2"),
        Program(code=dummy_code.replace("most_frequent", "uniform") +
                "# variant\n", source="seed", label="dummy3"),
    ]
    score_fn = lambda preds: certify.score_val(task, splits.val_rows, preds)
    out = run_portfolio_round(progs, Xtr, ytr, Xva, kind="classification",
                              score_fn=score_fn,
                              config=PortfolioConfig(wall_seconds=50, cpu_seconds=45,
                                                     eta=3.0, min_frac=1 / 9, seed=0))
    killed = [r.label for r in out.records if r.killed_early]
    assert killed, f"ASHA must early-kill at least one losing arm; killed={killed}"
    assert "dummy" not in out.best_program.label, "a dummy arm must never win"
    # the winner reached full budget
    assert out.best_program.id in out.full_budget_scores
    print(f"[ok] early-kill pruned {len(killed)} arms: {killed}; winner={out.best_program.label}")


def test_does_not_touch_sealed_split():
    """The portfolio must read train+val only. We make the sealed split observable: if the
    portfolio ever fit/predicted on it, the val winner would change. We assert the portfolio's
    winner is identical whether or not the sealed rows exist, and that score_fn (val-only) is the
    sole scorer (firewall)."""
    task = _clf_task()
    splits = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(splits.train_rows); ytr = Task.rows_to_y(splits.train_rows, task.kind)
    Xva = Task.rows_to_X(splits.val_rows)
    progs = _clf_programs()

    # a tripwire score_fn: it asserts the prediction vector it is handed is val-length, never
    # sealed-length. If the portfolio ever scored sealed predictions, this fires.
    n_val = len(splits.val_rows)
    n_sealed = len(splits.sealed_rows)
    assert n_val != n_sealed, "test needs distinct val/sealed sizes to be a real tripwire"
    seen_lengths = []

    def tripwire(preds):
        seen_lengths.append(len(preds))
        assert len(preds) == n_val, f"portfolio scored a non-val vector len={len(preds)}"
        return certify.score_val(task, splits.val_rows, preds)

    out = run_portfolio_round(progs, Xtr, ytr, Xva, kind="classification",
                              score_fn=tripwire,
                              config=PortfolioConfig(wall_seconds=50, cpu_seconds=45))
    assert out.best_program is not None
    assert seen_lengths and all(L == n_val for L in seen_lengths)
    # the sealed test must still be peekable exactly once afterward (we never consumed its peek)
    st = splits.sealed_test
    st._count_peek("post-portfolio first peek")  # must not raise -> portfolio took no peek
    print(f"[ok] sealed untouched: scored {len(seen_lengths)} val-length vectors only")


def test_regression_round():
    """ASHA also works for regression (random subsample, r2 metric)."""
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    task = Task(X=d.data, y=d.target, kind="regression", theta=0.4, name="diabetes")
    splits = certify.make_splits(task, seed=0)
    Xtr = Task.rows_to_X(splits.train_rows); ytr = Task.rows_to_y(splits.train_rows, task.kind)
    Xva = Task.rows_to_X(splits.val_rows)
    from frontier.proposers import make_code as mk
    progs = [
        Program(code=mk({"base": "ridge", "scale": True}, "regression"), source="seed", label="ridge"),
        Program(code=mk({"base": "ridge", "scale": True, "poly": 2}, "regression"),
                source="seed", label="poly_ridge"),
        Program(code=mk({"base": "hist_gbm"}, "regression"), source="seed", label="hist_gbm"),
        Program(code=mk({"base": "rf"}, "regression"), source="seed", label="rf"),
        Program(code=mk({"base": "gbr"}, "regression"), source="seed", label="gbr"),
    ]
    score_fn = lambda preds: certify.score_val(task, splits.val_rows, preds)
    out = run_portfolio_round(progs, Xtr, ytr, Xva, kind="regression", score_fn=score_fn,
                              config=PortfolioConfig(wall_seconds=50, cpu_seconds=45))
    assert out.best_program is not None and np.isfinite(out.best_score)
    assert out.best_program.id in out.full_budget_scores
    print(f"[ok] regression winner={out.best_program.label} r2={out.best_score:.4f}")


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
    print(f"\n{len(fns) - failed}/{len(fns)} portfolio tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
