"""BENCHMARK HARNESS SCAFFOLD -- measure 'where we stand' instead of arguing it.

STATUS (2026-06): SCAFFOLD, NOT YET WIRED. The concrete certified arenas that actually run are the
scripts/benchmark_*.py suite and scripts/repr_arena.py (FgvcAircraftArena). This generic harness is
unfalsified -- wire-or-delete debt; do not claim value without a certified-arena measurement.

WHY THIS EXISTS
---------------
The OpenML toy sets are solved, so live results are honest zeros; you can only answer 'beats a chat LLM /
average researcher?' by running on arenas with headroom (the internal hygiene-certified pool, and -- via
the extension point below -- MLE-bench / RE-bench / Kaggle). This is the harness that runs a suite of tasks
through the SAME frozen certify path and produces a leaderboard of certified outcomes + wall-clock.

EXTENDING TO MLE-bench / RE-bench
---------------------------------
A real external benchmark is just a BenchTask whose `runner` shells out to / calls run_goal_loop on the
downloaded dataset and returns the certified outcome. The harness is dataset-source-agnostic on purpose;
no network or heavyweight deps are imported here so it runs on a slim CPU box.

THE INVARIANT
-------------
The harness only ORCHESTRATES and records. Each task's runner must return a certified outcome from the
frozen Tier-3 path; the harness never certifies, never adjusts a bound, and reports certified==False
honestly (a failed/uncertified task scores zero, it is not hidden).

CONTRACT: stdlib only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Sequence

from .replication import DatasetCertOutcome


@dataclass
class BenchTask:
    """One benchmark task. `runner()` runs the full loop on the task and returns its certified outcome."""
    name: str
    runner: Callable[[], DatasetCertOutcome]


@dataclass
class BenchResult:
    task: str
    certified: bool
    lower_bound: float
    seconds: float
    error: str = ""


def run_benchmark(tasks: Sequence[BenchTask]) -> List[BenchResult]:
    """Run each task, timing it and capturing failures honestly (a crashed/uncertified task scores zero)."""
    results: List[BenchResult] = []
    for task in tasks:
        t0 = time.perf_counter()
        try:
            outcome = task.runner()
            dt = time.perf_counter() - t0
            results.append(BenchResult(task.name, bool(outcome.certified),
                                       float(outcome.lower_bound), dt))
        except Exception as exc:                       # a task failure is a zero, not a hidden gap
            dt = time.perf_counter() - t0
            results.append(BenchResult(task.name, False, 0.0, dt, error=f"{type(exc).__name__}: {exc}"))
    return results


@dataclass
class BenchSummary:
    n_tasks: int
    n_certified: int
    solve_rate: float
    mean_certified_bound: float

    def __str__(self) -> str:
        return (f"solved {self.n_certified}/{self.n_tasks} ({self.solve_rate:.0%}); "
                f"mean certified lower bound = {self.mean_certified_bound:.3f}")


def summarize(results: Sequence[BenchResult]) -> BenchSummary:
    n = len(results)
    certified = [r for r in results if r.certified]
    mean_bound = sum(r.lower_bound for r in certified) / len(certified) if certified else 0.0
    return BenchSummary(n, len(certified), (len(certified) / n if n else 0.0), mean_bound)


def leaderboard(results: Sequence[BenchResult]) -> str:
    lines = ["task                          certified   lower_bound   seconds"]
    for r in sorted(results, key=lambda x: (-x.certified, -x.lower_bound)):
        flag = "Y" if r.certified else "n"
        extra = f"   ! {r.error}" if r.error else ""
        lines.append(f"{r.task:<28}  {flag:^9}   {r.lower_bound:>10.3f}   {r.seconds:>7.2f}{extra}")
    lines.append(str(summarize(results)))
    return "\n".join(lines)


__all__ = ["BenchTask", "BenchResult", "BenchSummary", "run_benchmark", "summarize", "leaderboard"]
