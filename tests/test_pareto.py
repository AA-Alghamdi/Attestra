"""Tests for the certified Pareto front.

Acceptance (Phase 9): only certified candidates appear on the front; the named picks are genuinely
different points ('fastest certified' != 'most accurate' != 'most accurate under a latency budget'),
proving the deliverable is a real multi-objective frontier and not one dominating model. Uncertified
candidates are never returned even if they look best on an axis."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import pareto as PF


def _bench():
    return [
        # name,         metric_lb, latency, cost,   ece,  certified
        PF.ParetoCandidate("tiny_fast",   0.82,  8.0,  0.0001, 0.04, True),
        PF.ParetoCandidate("balanced",    0.88, 25.0,  0.0005, 0.03, True),
        PF.ParetoCandidate("big_accurate",0.93, 90.0,  0.0040, 0.06, True),
        PF.ParetoCandidate("well_cal",    0.86, 40.0,  0.0008, 0.01, True),
        PF.ParetoCandidate("uncert_best", 0.97,  5.0,  0.0001, 0.005, False),  # best on every axis but NOT certified
        PF.ParetoCandidate("dominated",   0.80, 60.0,  0.0050, 0.09, True),    # dominated by balanced/big
    ]


def test_uncertified_never_on_front_or_picks():
    front = PF.ParetoFront(_bench(), axes=PF.DEFAULT_AXES)
    names = {c.name for c in front.nondominated()}
    assert "uncert_best" not in names
    for pick in front.picks(latency_budget_ms=50.0).values():
        assert pick is None or pick.name != "uncert_best"


def test_named_picks_are_distinct_points():
    front = PF.ParetoFront(_bench())
    fastest = front.fastest_certified()
    accurate = front.most_accurate()
    under_budget = front.most_accurate_under_latency(50.0)
    best_cal = front.best_calibrated()
    assert fastest.name == "tiny_fast"
    assert accurate.name == "big_accurate"
    assert under_budget.name == "balanced"            # big_accurate excluded (90ms > 50ms); balanced is most accurate <=50ms
    assert best_cal.name == "well_cal"
    # fastest and most-accurate are different models -> a real tradeoff frontier
    assert fastest.name != accurate.name


def test_dominated_candidate_excluded():
    front = PF.ParetoFront(_bench())
    names = {c.name for c in front.nondominated()}
    assert "dominated" not in names
    assert {"tiny_fast", "big_accurate"} <= names      # clear corners of the frontier survive


def test_empty_when_nothing_certified():
    cands = [PF.ParetoCandidate("a", 0.9, 10, 0.001, 0.02, False)]
    front = PF.ParetoFront(cands)
    assert front.nondominated() == []
    assert front.most_accurate() is None


def test_report_renders():
    rep = PF.ParetoFront(_bench()).report(latency_budget_ms=50.0)
    assert "Certified Pareto front" in rep and "tiny_fast" in rep


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
