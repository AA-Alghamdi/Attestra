"""PORTFOLIO scheduler over MANY approaches toward ONE objective.

The platform's promotion gate is a single frozen certifier reached through ``vfplatform.loop.run_goal_loop``
(the val-bound gate + ``certify_on_sealed``, both in the frozen core). A real autoresearcher rarely has one
approach; it has several (different families, feature sets, time budgets, ...). This layer runs each approach
through the EXISTING ``run_goal_loop`` unchanged and then SELECTS among the results.

INTEGRITY CONTRACT (the whole point of this file):
  * The portfolio NEVER certifies, re-certifies, relaxes theta, or invents a promotion path. The ONLY thing
    that produces a certificate is each approach's own ``run_goal_loop`` call, whose frozen certifier alone
    certifies. This module reads ``GoalLoopResult.decision`` / ``GoalLoopResult.certificate`` and nothing
    else load-bearing.
  * Selection is over results that are ALREADY frozen-certified (``decision == "certified"``), ranked by
    their frozen ``certificate["lower_bound"]``. Ties break deterministically by approach index (lowest
    index wins).
  * If NO approach certifies, it returns an honest portfolio-level no-certify summary pointing at the closest
    approach + that approach's own ``failure_report`` -- it does NOT promote anything.
  * Deleting this layer leaves every individual certificate byte-identical: this module mutates nothing in a
    ``GoalLoopResult`` and passes no argument that changes how ``run_goal_loop`` certifies. The cross-experiment
    FDR view is REPORTING ONLY (it does not gate selection).

It is deterministic given the approaches: no clock, no RNG. Any timestamp is passed through by the caller
(``ts`` per approach), exactly like ``cross_experiment.PromotionLedger`` expects.

stdlib + numpy + the existing read-only imports only. No torch/skimage.
"""
from __future__ import annotations

import math

from .loop import run_goal_loop, GoalLoopResult
from .cross_experiment import PromotionLedger, LordFDR


def _is_certified(result: GoalLoopResult) -> bool:
    """A result counts as certified ONLY when the frozen loop said so: decision == 'certified' AND the
    frozen certificate's own certified flag is True. We never re-derive certification from raw numbers."""
    if result is None or result.decision != "certified":
        return False
    cert = result.certificate or {}
    return bool(cert.get("certified"))


def _certified_lower_bound(result: GoalLoopResult):
    """The frozen certified lower bound to RANK certified results by. Pulled verbatim from the certificate;
    never recomputed. Returns None if (defensively) absent or non-finite."""
    cert = result.certificate or {}
    lb = cert.get("lower_bound")
    try:
        lb = float(lb)
    except (TypeError, ValueError):
        return None
    return lb if math.isfinite(lb) else None


def _binom_pvalue(observed, n, theta):
    """One-sided p-value for H0: true accuracy <= theta on n sealed examples (binomial). Returns None when
    inputs are missing or scipy is unavailable. This MIRRORS loop._binom_pvalue (the same frozen-cert fields
    'observed'/'n'); it is used ONLY to feed the reporting-only LordFDR view, never to certify."""
    if observed is None or n is None:
        return None
    try:
        from scipy.stats import binom
        k = int(round(float(observed) * int(n)))
        return float(binom.sf(k - 1, int(n), float(theta)))
    except Exception:  # noqa: BLE001
        return None


def _approach_summary(index, approach, result: GoalLoopResult) -> dict:
    """A compact, JSON-friendly per-approach summary. Carries the FROZEN certificate verbatim (when the
    approach certified) so a caller can verify it byte-for-byte against a standalone run."""
    cert = result.certificate
    return {
        "approach_index": index,
        "goal_text": approach.get("goal_text"),
        "experiment": result.experiment,
        "decision": result.decision,
        "objective": result.objective,
        "certified": _is_certified(result),
        "lower_bound": _certified_lower_bound(result) if _is_certified(result) else (
            None if not cert else cert.get("lower_bound")),
        "observed": None if not cert else cert.get("observed"),
        "threshold": approach.get("threshold"),
        "n_test": result.n_test,
        "provider": result.provider,
        "certificate": cert,                       # verbatim frozen certificate (None if no peek)
        "failure_report": result.failure_report,   # honest next-step on any non-certified outcome
    }


def run_portfolio(approaches, *, ledger_path=None, on_event=None) -> dict:
    """Run a PORTFOLIO of approaches toward one objective and SELECT the winner from already-certified results.

    Parameters
    ----------
    approaches : list[dict]
        Each dict is a kwargs payload for ``run_goal_loop`` (must contain at least ``records`` and
        ``goal_text``; any other run_goal_loop kwarg is forwarded as-is: kind, task_type, target_key, labels,
        metric, threshold, objective, time_budget_s, llm_propose, ...). An optional ``ts`` key (popped, not
        forwarded) is the caller-supplied timestamp recorded on the ledger row (this module never reads a clock).
    ledger_path : str | None
        If given, one ``cross_experiment.PromotionLedger`` row is appended per approach (REPORTING ONLY).
    on_event : callable | None
        Optional portfolio-level progress sink. Emitted events: {stage:"portfolio", approach_index, decision}
        per approach, and a terminal {stage:"portfolio", approach_index: winner_index, decision: <decision>}.

    Returns
    -------
    dict with keys:
        winner_index       -- index of the selected certified approach, or None if none certified.
        decision           -- "portfolio_certified" if a winner was selected, else "portfolio_no_certify".
        best_certificate   -- the winner's FROZEN certificate verbatim (None if none certified).
        per_approach        -- [per-approach summary dict] in input order.
        n_certified        -- number of approaches whose frozen result certified.
        cross_experiment    -- LordFDR summary over the certified-approach p-values (None if not computable).

    The portfolio NEVER certifies. It only ranks results that the frozen certifier already certified.
    """
    if approaches is None:
        approaches = []
    approaches = list(approaches)

    def _emit(**detail):
        if on_event is None:
            return
        try:
            on_event({"stage": "portfolio", **detail})
        except Exception:  # noqa: BLE001  a misbehaving observer never breaks the portfolio
            pass

    ledger = PromotionLedger(ledger_path) if ledger_path else None

    summaries = []
    results = []
    for i, approach in enumerate(approaches):
        if not isinstance(approach, dict):
            raise ValueError(f"approach {i} must be a dict of run_goal_loop kwargs, got {type(approach)!r}")
        kwargs = dict(approach)
        ts = kwargs.pop("ts", None)              # caller-supplied timestamp; NOT a run_goal_loop kwarg
        if "records" not in kwargs or "goal_text" not in kwargs:
            raise ValueError(f"approach {i} must supply at least 'records' and 'goal_text'")

        # Run THIS approach through the EXISTING frozen loop, unchanged. We pass nothing that alters how it
        # certifies; the certificate it returns is exactly what a standalone run_goal_loop(**kwargs) returns.
        result = run_goal_loop(**kwargs)
        results.append(result)
        summary = _approach_summary(i, approach, result)
        summaries.append(summary)

        # REPORTING-ONLY ledger row (when a path is given). This records the terminal outcome; it does not
        # feed selection and cannot change any certificate. ts is whatever the caller passed (clock-free).
        if ledger is not None:
            cert = result.certificate or {}
            metric = kwargs.get("metric")
            theta = kwargs.get("threshold")
            pv = (_binom_pvalue(cert.get("observed"), cert.get("n"), theta)
                  if (metric == "accuracy" and cert and theta is not None) else cert.get("p_value"))
            try:
                ledger.record({
                    "source": "portfolio", "approach_index": i, "plan_hash": cert.get("plan_hash"),
                    "decision": result.decision, "metric": metric, "theta": theta,
                    "observed": cert.get("observed"), "lower_bound": cert.get("lower_bound"),
                    "p_value": pv, "certified": _is_certified(result),
                    "ts": ts if ts is not None else (ledger.count() + 1),
                })
            except Exception:  # noqa: BLE001  the reporting ledger must never break the portfolio
                pass

        _emit(approach_index=i, decision=result.decision)

    # SELECT the winner: among the ALREADY-frozen-certified results, the highest certified lower bound wins;
    # ties break by lowest approach index (deterministic). NO re-certification, NO theta relaxation.
    certified_idx = [i for i, r in enumerate(results) if _is_certified(r)]
    n_certified = len(certified_idx)

    winner_index = None
    best_certificate = None
    if certified_idx:
        # sort key: (-lower_bound, index). A None lower bound (defensive) sorts last via -inf.
        def _key(i):
            lb = _certified_lower_bound(results[i])
            return (-(lb if lb is not None else -math.inf), i)
        winner_index = sorted(certified_idx, key=_key)[0]
        best_certificate = results[winner_index].certificate   # verbatim frozen certificate
        decision = "portfolio_certified"
    else:
        # No approach certified. Honest portfolio no-certify: point at the CLOSEST approach (the one whose
        # measured lower bound is nearest its threshold, i.e. smallest positive gap; falls back to first).
        decision = "portfolio_no_certify"
        if summaries:
            def _gap(i):
                s = summaries[i]
                lb = s.get("lower_bound")
                thr = s.get("threshold")
                if lb is None or thr is None:
                    return math.inf
                try:
                    return float(thr) - float(lb)   # smaller (or more negative) == closer to / over the bar
                except (TypeError, ValueError):
                    return math.inf
            closest = sorted(range(len(summaries)), key=lambda i: (_gap(i), i))[0]
            # surface the closest approach's failure_report at the top level for actionability
            summaries[closest] = dict(summaries[closest], closest=True)

    # CROSS-EXPERIMENT view: replay LordFDR over the certified approaches' p-values (REPORTING ONLY -- it does
    # not gate selection). Computable only when every certified approach yields a finite p-value.
    cross = None
    if certified_idx:
        pvals = []
        for i in certified_idx:
            cert = results[i].certificate or {}
            metric = approaches[i].get("metric")
            theta = approaches[i].get("threshold")
            pv = cert.get("p_value")
            if pv is None and metric == "accuracy" and theta is not None:
                pv = _binom_pvalue(cert.get("observed"), cert.get("n"), theta)
            if isinstance(pv, (int, float)) and math.isfinite(float(pv)):
                pvals.append(float(pv))
        if pvals and len(pvals) == len(certified_idx):
            alpha = approaches[certified_idx[0]].get("alpha", 0.05)
            try:
                fdr = LordFDR(alpha=float(alpha))
                for p in pvals:
                    fdr.test(p)
                cross = fdr.summary()
            except Exception:  # noqa: BLE001  reporting layer; never break selection
                cross = None

    _emit(approach_index=winner_index, decision=decision)

    return {
        "winner_index": winner_index,
        "decision": decision,
        "best_certificate": best_certificate,
        "per_approach": summaries,
        "n_certified": n_certified,
        "cross_experiment": cross,
    }


# --------------------------------------------------------------------------- self-test
def _selftest():
    """Tiny smoke test on the REAL frozen loop: breast_cancer at an easy theta certifies and is selected."""
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    recs = [{"features": {f: float(v) for f, v in zip(d.feature_names, row)}, "target": int(t)}
            for row, t in zip(d.data, d.target)][:400]
    base = dict(records=recs, kind="tabular", task_type="binary", target_key="target",
                metric="accuracy", llm_propose=False, seeds=(0,), max_rounds=3)
    out = run_portfolio([
        dict(goal_text="easy bar", threshold=0.80, **base),
        dict(goal_text="impossible bar", threshold=0.999, **base),
    ])
    assert out["decision"] == "portfolio_certified", out["decision"]
    assert out["winner_index"] == 0, out["winner_index"]
    assert out["n_certified"] == 1, out["n_certified"]
    assert out["best_certificate"]["certified"] is True
    print("portfolio self-test OK", {"winner": out["winner_index"], "n_certified": out["n_certified"]})


if __name__ == "__main__":
    _selftest()
