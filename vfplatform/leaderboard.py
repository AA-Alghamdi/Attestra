"""The rich leaderboard: one Run per (family x params x seed), ranked by the VALIDATION metric.

The leaderboard is the "best models with details and the different runs" surface the demo returns. It is
VALIDATION-only; the sealed test is never consulted to build it (only the single winner is certified once).
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Run:
    run_id: str
    family: str
    params: dict
    seed: int
    val_score: Optional[float]
    latency_ms: Optional[float]
    provider: str
    status: str = "FINISHED"           # FINISHED | FAILED | GATED
    error: Optional[str] = None
    artifact_ref: Optional[str] = None
    cost_usd: float = 0.0
    move: Optional[str] = None         # which proposed move produced this run
    round: Optional[int] = None        # which loop round

    def as_dict(self):
        return {"run_id": self.run_id, "family": self.family, "params": self.params, "seed": self.seed,
                "val_score": self.val_score, "latency_ms": self.latency_ms, "provider": self.provider,
                "status": self.status, "error": self.error, "artifact_ref": self.artifact_ref,
                "cost_usd": self.cost_usd, "move": self.move, "round": self.round}


@dataclass
class Leaderboard:
    metric: str
    runs: list = field(default_factory=list)

    def finished(self):
        return [r for r in self.runs if r.status == "FINISHED" and r.val_score is not None]

    def ranked(self):
        # higher is better for the metrics we use (accuracy / balanced_accuracy / macro_f1 / r2)
        return sorted(self.finished(), key=lambda r: r.val_score, reverse=True)

    def best(self) -> Optional[Run]:
        r = self.ranked()
        return r[0] if r else None

    def table(self, top=None):
        rows = self.ranked()
        if top:
            rows = rows[:top]
        out = [f"  {'rank':>4} {'family':22} {'move':15} {'rnd':>3} {'seed':>4} "
               f"{'val_'+self.metric:>13} {'lat_ms':>8} {'$':>6}"]
        for i, r in enumerate(rows, 1):
            out.append(f"  {i:>4} {r.family[:22]:22} {(r.move or '-')[:15]:15} {(r.round or 0):>3} "
                       f"{r.seed:>4} {r.val_score:>13.4f} {(r.latency_ms or 0):>8.1f} {r.cost_usd:>6.3f}")
        failed = [r for r in self.runs if r.status != "FINISHED"]
        if failed:
            out.append(f"  ({len(failed)} run(s) failed/gated: "
                       f"{', '.join(sorted({r.error or r.status for r in failed}))[:120]})")
        return "\n".join(out)

    def as_dict(self, top=None):
        rows = self.ranked()
        if top:
            rows = rows[:top]
        return {"metric": self.metric, "n_runs": len(self.runs), "n_finished": len(self.finished()),
                "ranked": [r.as_dict() for r in rows]}
