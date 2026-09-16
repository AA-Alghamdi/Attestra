"""Health monitoring for running experiments.

Detects stalling, divergence, numerical instability, and resource exhaustion
DURING execution so the cycle can intervene early rather than waste budget
on a doomed approach.

Probes run periodically and emit health signals that the orchestrator uses
to decide whether to continue, pivot, or abort an experiment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    STALLED = "stalled"


@dataclass
class HealthSignal:
    """A health observation at a point in time."""
    status: HealthStatus
    probe: str               # which probe emitted this
    message: str
    ts: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Aggregated health over a window of observations."""
    overall: HealthStatus
    signals: List[HealthSignal]
    recommendation: str      # "continue" | "pivot" | "abort"

    @property
    def should_continue(self) -> bool:
        return self.recommendation == "continue"

    @property
    def should_pivot(self) -> bool:
        return self.recommendation == "pivot"

    @property
    def should_abort(self) -> bool:
        return self.recommendation == "abort"


class HealthMonitor:
    """Monitors experiment health through periodic probes.

    Usage:
        monitor = HealthMonitor(time_budget_s=300)
        for round_num in range(max_rounds):
            monitor.observe_round(round_num, val_score, elapsed_s)
            report = monitor.check()
            if report.should_abort:
                break
            if report.should_pivot:
                # switch strategy
    """

    def __init__(
        self,
        *,
        time_budget_s: float = 300.0,
        stall_patience: int = 3,
        min_improvement: float = 1e-4,
        divergence_threshold: float = 0.1,
        max_error_rate: float = 0.8,
    ):
        self.time_budget_s = time_budget_s
        self.stall_patience = stall_patience
        self.min_improvement = min_improvement
        self.divergence_threshold = divergence_threshold
        self.max_error_rate = max_error_rate

        self._start_time = time.time()
        self._scores: List[float] = []
        self._round_times: List[float] = []
        self._errors: int = 0
        self._proposals: int = 0
        self._signals: List[HealthSignal] = []

    def observe_round(self, round_num: int, val_score: Optional[float],
                      elapsed_s: float, *, error: bool = False) -> None:
        """Record an observation after a round completes."""
        if val_score is not None:
            self._scores.append(val_score)
        self._round_times.append(elapsed_s)
        self._proposals += 1
        if error:
            self._errors += 1

    def check(self) -> HealthReport:
        """Run all health probes and return an aggregated report."""
        signals = []
        signals.append(self._probe_time_budget())
        signals.append(self._probe_stalling())
        signals.append(self._probe_divergence())
        signals.append(self._probe_error_rate())
        signals.append(self._probe_throughput())
        self._signals.extend(signals)

        # Determine overall status and recommendation
        statuses = [s.status for s in signals]
        if HealthStatus.CRITICAL in statuses:
            overall = HealthStatus.CRITICAL
            recommendation = "abort"
        elif statuses.count(HealthStatus.WARNING) >= 2:
            overall = HealthStatus.WARNING
            recommendation = "pivot"
        elif HealthStatus.STALLED in statuses:
            overall = HealthStatus.STALLED
            recommendation = "pivot"
        else:
            overall = HealthStatus.HEALTHY
            recommendation = "continue"

        return HealthReport(overall=overall, signals=signals, recommendation=recommendation)

    def _probe_time_budget(self) -> HealthSignal:
        """Check remaining time budget."""
        elapsed = time.time() - self._start_time
        remaining = self.time_budget_s - elapsed
        fraction_used = elapsed / max(self.time_budget_s, 1)

        if remaining <= 0:
            return HealthSignal(
                status=HealthStatus.CRITICAL,
                probe="time_budget",
                message=f"Time budget exhausted ({elapsed:.0f}s / {self.time_budget_s:.0f}s)",
                metrics={"elapsed_s": elapsed, "budget_s": self.time_budget_s, "fraction_used": fraction_used},
            )
        if fraction_used > 0.9:
            return HealthSignal(
                status=HealthStatus.WARNING,
                probe="time_budget",
                message=f"90%+ budget used ({elapsed:.0f}s / {self.time_budget_s:.0f}s)",
                metrics={"elapsed_s": elapsed, "budget_s": self.time_budget_s, "fraction_used": fraction_used},
            )
        return HealthSignal(
            status=HealthStatus.HEALTHY,
            probe="time_budget",
            message=f"{remaining:.0f}s remaining",
            metrics={"elapsed_s": elapsed, "budget_s": self.time_budget_s, "fraction_used": fraction_used},
        )

    def _probe_stalling(self) -> HealthSignal:
        """Detect when validation score stops improving."""
        if len(self._scores) < self.stall_patience + 1:
            return HealthSignal(
                status=HealthStatus.HEALTHY,
                probe="stalling",
                message="Too few observations to assess stalling",
                metrics={"n_scores": len(self._scores)},
            )

        recent = self._scores[-self.stall_patience:]
        best_recent = max(recent)
        best_before = max(self._scores[:-self.stall_patience]) if len(self._scores) > self.stall_patience else 0.0
        improvement = best_recent - best_before

        if improvement < self.min_improvement:
            return HealthSignal(
                status=HealthStatus.STALLED,
                probe="stalling",
                message=f"No improvement in {self.stall_patience} rounds "
                        f"(best: {best_recent:.4f}, prev best: {best_before:.4f})",
                metrics={"improvement": improvement, "best_recent": best_recent,
                         "best_overall": max(self._scores), "stall_rounds": self.stall_patience},
            )
        return HealthSignal(
            status=HealthStatus.HEALTHY,
            probe="stalling",
            message=f"Improving: +{improvement:.4f} over last {self.stall_patience} rounds",
            metrics={"improvement": improvement, "best": max(self._scores)},
        )

    def _probe_divergence(self) -> HealthSignal:
        """Detect when scores are getting WORSE (divergence)."""
        if len(self._scores) < 3:
            return HealthSignal(
                status=HealthStatus.HEALTHY,
                probe="divergence",
                message="Too few observations",
            )

        recent_3 = self._scores[-3:]
        if len(self._scores) > 3:
            prev_3 = self._scores[-6:-3] if len(self._scores) >= 6 else self._scores[:3]
            avg_recent = sum(recent_3) / len(recent_3)
            avg_prev = sum(prev_3) / len(prev_3)
            drop = avg_prev - avg_recent

            if drop > self.divergence_threshold:
                return HealthSignal(
                    status=HealthStatus.CRITICAL,
                    probe="divergence",
                    message=f"Scores diverging: avg dropped by {drop:.4f}",
                    metrics={"avg_recent": avg_recent, "avg_prev": avg_prev, "drop": drop},
                )

        return HealthSignal(
            status=HealthStatus.HEALTHY,
            probe="divergence",
            message="No divergence detected",
        )

    def _probe_error_rate(self) -> HealthSignal:
        """Check the fraction of proposals that fail."""
        if self._proposals < 3:
            return HealthSignal(
                status=HealthStatus.HEALTHY,
                probe="error_rate",
                message="Too few proposals to assess error rate",
            )

        rate = self._errors / max(self._proposals, 1)
        if rate > self.max_error_rate:
            return HealthSignal(
                status=HealthStatus.CRITICAL,
                probe="error_rate",
                message=f"Error rate {rate:.0%} exceeds threshold {self.max_error_rate:.0%} "
                        f"({self._errors}/{self._proposals})",
                metrics={"error_rate": rate, "errors": self._errors, "proposals": self._proposals},
            )
        if rate > self.max_error_rate * 0.7:
            return HealthSignal(
                status=HealthStatus.WARNING,
                probe="error_rate",
                message=f"High error rate: {rate:.0%} ({self._errors}/{self._proposals})",
                metrics={"error_rate": rate, "errors": self._errors, "proposals": self._proposals},
            )
        return HealthSignal(
            status=HealthStatus.HEALTHY,
            probe="error_rate",
            message=f"Error rate: {rate:.0%}",
            metrics={"error_rate": rate},
        )

    def _probe_throughput(self) -> HealthSignal:
        """Check if rounds are taking too long (possible infinite loops)."""
        if len(self._round_times) < 2:
            return HealthSignal(
                status=HealthStatus.HEALTHY,
                probe="throughput",
                message="Too few rounds to assess throughput",
            )

        avg_round = sum(self._round_times) / len(self._round_times)
        last_round = self._round_times[-1]

        # If last round took > 5x the average, something is wrong
        if avg_round > 0 and last_round > avg_round * 5:
            return HealthSignal(
                status=HealthStatus.WARNING,
                probe="throughput",
                message=f"Last round took {last_round:.1f}s vs avg {avg_round:.1f}s",
                metrics={"last_round_s": last_round, "avg_round_s": avg_round},
            )
        return HealthSignal(
            status=HealthStatus.HEALTHY,
            probe="throughput",
            message=f"Avg round: {avg_round:.1f}s",
            metrics={"avg_round_s": avg_round, "total_rounds": len(self._round_times)},
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "elapsed_s": time.time() - self._start_time,
            "n_rounds": len(self._round_times),
            "n_scores": len(self._scores),
            "best_score": max(self._scores) if self._scores else None,
            "error_rate": self._errors / max(self._proposals, 1),
            "avg_round_s": sum(self._round_times) / max(len(self._round_times), 1),
        }
