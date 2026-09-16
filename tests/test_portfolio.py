"""Acceptance tests for vfplatform.portfolio (the portfolio scheduler over many approaches).

The load-bearing INTEGRITY claim: the portfolio NEVER certifies. It runs each approach through the EXISTING
frozen run_goal_loop and only SELECTS among already-certified results by their frozen lower bounds. Deleting
the portfolio layer must leave every individual certificate's certification content byte-identical.

These tests use the REAL frozen loop on the REAL breast_cancer dataset (numpy + sklearn + stdlib only),
with llm_propose=False for determinism and speed.

Run:  /Users/abdullahalghamdi/jax-env-311/bin/python tests/test_portfolio.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import load_breast_cancer

from vfplatform.portfolio import run_portfolio
from vfplatform.loop import run_goal_loop          # imported read-only, for the byte-identical check
from vfplatform import cross_experiment as cx


# certification-bearing certificate fields: the scientific content that the frozen certifier produces. The
# excluded fields (winner_run store-id, leaderboard_digest derived from store-ids, latency_ms, cost_usd) are
# per-run bookkeeping/timing, NOT certification -- they differ between two identical standalone runs too.
_CERT_CORE = ("observed", "lower_bound", "certified", "n", "k", "theta", "alpha_per_check", "checks",
              "p_value", "sealed_digest", "reason", "objective", "winner_family", "winner_content_id",
              "latency_ok", "cost_ok", "ece_ok", "val_ece", "peeks")


def _cert_core(cert):
    return {k: cert.get(k) for k in _CERT_CORE} if cert else None


def _records(n=400):
    d = load_breast_cancer()
    recs = [{"features": {f: float(v) for f, v in zip(d.feature_names, row)}, "target": int(t)}
            for row, t in zip(d.data, d.target)]
    return recs[:n]


def _base():
    return dict(records=_records(), kind="tabular", task_type="binary", target_key="target",
                metric="accuracy", llm_propose=False, seeds=(0,), max_rounds=3)


# ----------------------------------------------------------------- selection
def test_portfolio_selects_the_certified_approach():
    out = run_portfolio([
        dict(goal_text="easy bar", threshold=0.80, **_base()),       # certifies
        dict(goal_text="impossible bar", threshold=0.999, **_base()),# does not certify
    ])
    assert out["decision"] == "portfolio_certified", out["decision"]
    assert out["winner_index"] == 0, out["winner_index"]
    assert out["n_certified"] == 1, out["n_certified"]
    assert out["best_certificate"]["certified"] is True
    # the winner's reported lower bound is the FROZEN certificate's lower bound (not recomputed)
    assert out["best_certificate"]["lower_bound"] == out["per_approach"][0]["lower_bound"]


def test_portfolio_picks_highest_lower_bound_among_certified():
    # two certified approaches at different (achievable) thetas -> winner is the higher frozen lower bound,
    # ties broken by index. Both certify on breast_cancer; selection must be by lower_bound, deterministically.
    out = run_portfolio([
        dict(goal_text="bar 0.80", threshold=0.80, **_base()),
        dict(goal_text="bar 0.85", threshold=0.85, **_base()),
    ])
    assert out["n_certified"] == 2, out["n_certified"]
    lbs = [out["per_approach"][i]["lower_bound"] for i in range(2)]
    expected = max(range(2), key=lambda i: (lbs[i], -i))   # highest lb, ties -> lowest index
    assert out["winner_index"] == expected, (out["winner_index"], lbs)
    assert out["best_certificate"]["lower_bound"] == max(lbs)


# ----------------------------------------------------------------- no-certify
def test_portfolio_no_certify_when_none_certify():
    out = run_portfolio([
        dict(goal_text="impossible A", threshold=0.999, **_base()),
        dict(goal_text="impossible B", threshold=0.9999, **_base()),
    ])
    assert out["decision"] == "portfolio_no_certify", out["decision"]
    assert out["winner_index"] is None
    assert out["best_certificate"] is None
    assert out["n_certified"] == 0
    # honest-stop: the closest approach is flagged and carries its own failure_report (actionable)
    flagged = [s for s in out["per_approach"] if s.get("closest")]
    assert len(flagged) == 1, "exactly one closest approach should be flagged"
    assert flagged[0]["failure_report"] is not None


# ----------------------------------------------------------------- INTEGRITY: certificate unchanged
def test_portfolio_does_not_alter_certification():
    """The portfolio's per-approach certificate must match a STANDALONE run_goal_loop on the same kwargs --
    on every certification-bearing field. This proves the portfolio did NOT re-certify or change theta:
    deleting the portfolio layer leaves the certificate's certification content byte-identical."""
    kwargs = dict(goal_text="easy bar", threshold=0.80, **_base())

    standalone = run_goal_loop(**kwargs)
    assert standalone.decision == "certified"

    out = run_portfolio([dict(kwargs)])   # same kwargs, through the portfolio
    pa = out["per_approach"][0]
    assert pa["decision"] == "certified"

    core_standalone = _cert_core(standalone.certificate)
    core_portfolio = _cert_core(pa["certificate"])
    # byte-identical on the certification core
    a = json.dumps(core_standalone, sort_keys=True, default=str)
    b = json.dumps(core_portfolio, sort_keys=True, default=str)
    assert a == b, f"portfolio altered certification:\n  standalone={a}\n  portfolio ={b}"
    # and the headline numbers in particular
    assert core_portfolio["lower_bound"] == core_standalone["lower_bound"]
    assert core_portfolio["observed"] == core_standalone["observed"]
    assert core_portfolio["theta"] == 0.80 == core_standalone["theta"]   # theta never relaxed
    assert core_portfolio["certified"] is True


# ----------------------------------------------------------------- reporting layers (additive)
def test_portfolio_ledger_rows_are_reporting_only():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "portfolio_promo.jsonl")
        out = run_portfolio([
            dict(goal_text="easy bar", threshold=0.80, ts=1.0, **_base()),
            dict(goal_text="impossible bar", threshold=0.999, ts=2.0, **_base()),
        ], ledger_path=path)
        led = cx.PromotionLedger(path)
        rows = led.all()
        assert led.count() == 2, led.count()
        # ts is exactly what the caller passed (clock-free) and certified flag matches the decisions
        by_idx = {r["approach_index"]: r for r in rows}
        assert by_idx[0]["ts"] == 1.0 and by_idx[1]["ts"] == 2.0
        assert by_idx[0]["certified"] is True and by_idx[1]["certified"] is False
        # ledger is reporting-only: the selection still picks approach 0
        assert out["winner_index"] == 0


def test_portfolio_cross_experiment_view_when_computable():
    out = run_portfolio([
        dict(goal_text="bar 0.80", threshold=0.80, **_base()),
        dict(goal_text="bar 0.85", threshold=0.85, **_base()),
    ])
    # accuracy + certified -> binomial p-values available -> a LordFDR summary is returned
    xe = out["cross_experiment"]
    if xe is not None:   # requires scipy; skip the assertion if the p-value channel is unavailable
        assert xe["n_tests"] == out["n_certified"]
        assert 0.0 <= xe["alpha_spent"] <= xe["n_tests"] * xe["alpha"] + 1e-9


def test_portfolio_emits_progress_events():
    events = []
    run_portfolio([
        dict(goal_text="easy bar", threshold=0.80, **_base()),
        dict(goal_text="impossible bar", threshold=0.999, **_base()),
    ], on_event=events.append)
    assert all(e.get("stage") == "portfolio" for e in events)
    # one per approach + one terminal selection event
    per_approach = [e for e in events if e.get("approach_index") in (0, 1)]
    assert len(per_approach) >= 2
    terminal = events[-1]
    assert terminal["decision"] in ("portfolio_certified", "portfolio_no_certify")


# ----------------------------------------------------------------- input guards
def test_portfolio_rejects_malformed_approach():
    for bad in (["not a dict"], [dict(goal_text="missing records")], [dict(records=[])]):
        try:
            run_portfolio(bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


def test_portfolio_empty_is_no_certify():
    out = run_portfolio([])
    assert out["decision"] == "portfolio_no_certify"
    assert out["winner_index"] is None and out["n_certified"] == 0
    assert out["per_approach"] == []


TESTS = [
    test_portfolio_selects_the_certified_approach,
    test_portfolio_picks_highest_lower_bound_among_certified,
    test_portfolio_no_certify_when_none_certify,
    test_portfolio_does_not_alter_certification,
    test_portfolio_ledger_rows_are_reporting_only,
    test_portfolio_cross_experiment_view_when_computable,
    test_portfolio_emits_progress_events,
    test_portfolio_rejects_malformed_approach,
    test_portfolio_empty_is_no_certify,
]


def run(tests):
    p = f = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); p += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}"); f += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {e}"); f += 1
    print(f"  ---- {p} passed, {f} failed ----")
    return p, f


def main():
    _, fails = run(TESTS)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
