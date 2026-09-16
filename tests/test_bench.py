"""Tests for the benchmark harness scaffold.

Acceptance (Phase 10): the harness runs a suite through the frozen certify path and produces a leaderboard
+ summary; a crashing/uncertified task scores zero honestly (it is not hidden or counted as a solve)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import bench as B
from vfplatform.replication import DatasetCertOutcome


def test_runs_suite_and_summarizes():
    tasks = [
        B.BenchTask("easy", lambda: DatasetCertOutcome("easy", True, 0.95)),
        B.BenchTask("medium", lambda: DatasetCertOutcome("medium", True, 0.83)),
        B.BenchTask("hard", lambda: DatasetCertOutcome("hard", False, 0.0)),
    ]
    results = B.run_benchmark(tasks)
    summ = B.summarize(results)
    assert summ.n_tasks == 3 and summ.n_certified == 2
    assert summ.solve_rate == pytest.approx(2 / 3)
    assert summ.mean_certified_bound == pytest.approx((0.95 + 0.83) / 2)


def test_failing_task_scores_zero_not_hidden():
    def boom():
        raise RuntimeError("dataset download failed")

    results = B.run_benchmark([B.BenchTask("flaky", boom),
                               B.BenchTask("ok", lambda: DatasetCertOutcome("ok", True, 0.9))])
    by_name = {r.task: r for r in results}
    assert by_name["flaky"].certified is False and by_name["flaky"].lower_bound == 0.0
    assert "RuntimeError" in by_name["flaky"].error           # failure is recorded, not swallowed silently
    assert by_name["ok"].certified is True
    assert B.summarize(results).n_certified == 1


def test_leaderboard_renders():
    results = B.run_benchmark([B.BenchTask("t1", lambda: DatasetCertOutcome("t1", True, 0.9))])
    lb = B.leaderboard(results)
    assert "t1" in lb and "solved 1/1" in lb


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
