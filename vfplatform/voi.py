"""Value-of-Information move ranking (build-order step 4).

Ranks the closed menu of proposable moves by expected validation gain per unit cost. Uses the move's
prior until a CaseBase of REALIZED gains (this run + persisted across runs) replaces the prior. Selection
is on VALIDATION only -- VoI never reads the sealed test.
"""
import json
import os
import uuid
from dataclasses import dataclass, field


class CaseBase:
    """Persisted realized val-gains per move name, warm-starting VoI across runs."""

    def __init__(self, path=None):
        self.path = path
        self.gains = {}
        self.costs = {}                                # realized cost (gpu-seconds / wall-seconds) per move
        if path and os.path.exists(path):
            try:
                blob = json.load(open(path))
                self.gains = blob.get("gains", blob if isinstance(blob, dict) else {})
                self.costs = blob.get("costs", {})
            except Exception:  # noqa: BLE001
                self.gains, self.costs = {}, {}

    def _flush(self):
        if self.path:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            # atomic write: tmp keyed on pid+uuid (NOT pid alone -- threads in one process share a pid and
            # would collide on the staging file when writing the same path). os.replace is atomic.
            tmp = f"{self.path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
            with open(tmp, "w") as fh:
                json.dump({"gains": self.gains, "costs": self.costs}, fh)
            os.replace(tmp, self.path)

    def record(self, move_name, realized_gain):
        self.gains.setdefault(move_name, []).append(round(float(realized_gain), 5))
        self._flush()

    def record_cost(self, move_name, realized_cost):
        self.costs.setdefault(move_name, []).append(round(float(realized_cost), 5))
        self._flush()

    def expected_gain(self, move):
        hist = self.gains.get(move.name, [])
        if hist:
            return max(0.0, sum(hist) / len(hist))     # realized average (clamped >= 0)
        return move.prior_gain

    def expected_cost(self, move):
        hist = self.costs.get(move.name, [])
        if hist:
            return max(1e-6, sum(hist) / len(hist))    # MEASURED cost replaces the prior once observed
        return max(move.prior_cost, 1e-6)


@dataclass
class RankedMove:
    name: str
    voi: float
    expected_gain: float
    cost: float


def voi_rank(moves, casebase: CaseBase):
    """Return moves ordered by VoI = expected_gain / cost, descending. Deterministic."""
    ranked = []
    for m in moves:
        g = casebase.expected_gain(m)
        cost = casebase.expected_cost(m)            # MEASURED cost once observed, else the prior
        ranked.append(RankedMove(m.name, round(g / cost, 5), round(g, 5), round(cost, 5)))
    ranked.sort(key=lambda r: (r.voi, r.expected_gain), reverse=True)
    return ranked
