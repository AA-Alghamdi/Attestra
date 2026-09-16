"""Experiment tracking + monitoring (MLflow / Weights-&-Biases parity, local-first).

A run store: experiment -> run -> {params, metrics (step-wise), artifacts, system, tags, status}. Each run
carries a content digest (the same `science.digest` the evidence ledger uses) so tracking and certification
share ONE root of trust. Tracking is OBSERVATIONAL: it records what happened and CANNOT alter a certificate
(the certificate is produced by the frozen core and stored separately/immutably). Runs are mutable while
RUNNING and finalized (read-only) at end_run.
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import sys as _sys
_VF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VF not in _sys.path:
    _sys.path.insert(0, _VF)
from vectorforge import science                       # the frozen digest = shared root of trust


@dataclass
class RunRecord:
    run_id: str
    experiment: str
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)        # key -> [{step, value}]
    artifacts: dict = field(default_factory=dict)      # name -> ref
    system: dict = field(default_factory=dict)         # provider, device, latency, etc.
    tags: dict = field(default_factory=dict)
    status: str = "RUNNING"                            # RUNNING | FINISHED | FAILED | GATED
    started: float = 0.0
    ended: Optional[float] = None
    digest: Optional[str] = None                       # content address of the run spec (ledger link)

    def final_metric(self, key):
        steps = self.metrics.get(key, [])
        return steps[-1]["value"] if steps else None


class ExperimentStore:
    """JSON-per-run store under `root/<experiment>/<run_id>.json`, plus a per-experiment index."""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _exp_dir(self, experiment):
        d = os.path.join(self.root, experiment)
        os.makedirs(d, exist_ok=True)
        return d

    def _path(self, experiment, run_id):
        return os.path.join(self._exp_dir(experiment), f"{run_id}.json")

    def _read(self, experiment, run_id) -> RunRecord:
        with open(self._path(experiment, run_id)) as fh:
            return RunRecord(**json.load(fh))

    def _write(self, rec: RunRecord):
        with open(self._path(rec.experiment, rec.run_id), "w") as fh:
            json.dump(asdict(rec), fh, indent=2, default=str)

    def start_run(self, experiment, *, params=None, tags=None, system=None, run_id=None) -> str:
        params = params or {}
        digest = science.digest({"experiment": experiment, "params": params})
        run_id = run_id or ("run-" + science.digest({"e": experiment, "p": params, "t": time.time()})
                            .split(":")[-1][:10])
        rec = RunRecord(run_id=run_id, experiment=experiment, params=params, tags=tags or {},
                        system=system or {}, status="RUNNING", started=time.time(), digest=digest)
        self._write(rec)
        return run_id

    def log_metric(self, experiment, run_id, key, value, step=0):
        rec = self._read(experiment, run_id)
        rec.metrics.setdefault(key, []).append({"step": int(step), "value": float(value)})
        self._write(rec)

    def log_params(self, experiment, run_id, params):
        rec = self._read(experiment, run_id)
        rec.params.update(params)
        self._write(rec)

    def log_system(self, experiment, run_id, **kw):
        rec = self._read(experiment, run_id)
        rec.system.update(kw)
        self._write(rec)

    def log_artifact(self, experiment, run_id, name, ref):
        rec = self._read(experiment, run_id)
        rec.artifacts[name] = ref
        self._write(rec)

    def end_run(self, experiment, run_id, status="FINISHED"):
        rec = self._read(experiment, run_id)
        rec.status = status
        rec.ended = time.time()
        self._write(rec)

    def get(self, experiment, run_id) -> RunRecord:
        return self._read(experiment, run_id)

    def query(self, experiment):
        d = self._exp_dir(experiment)
        recs = []
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                recs.append(self._read(experiment, fn[:-5]))
        return recs

    def dashboard(self, experiment, *, sort_metric=None, ascending=False):
        """A UI-ready JSON roll-up of an experiment (the contract a dashboard would consume)."""
        runs = self.query(experiment)
        rows = []
        for r in runs:
            row = {"run_id": r.run_id, "status": r.status, "params": r.params, "system": r.system,
                   "metrics": {k: (v[-1]["value"] if v else None) for k, v in r.metrics.items()},
                   "duration_s": (round((r.ended or r.started) - r.started, 3))}
            rows.append(row)
        if sort_metric:
            rows.sort(key=lambda x: (x["metrics"].get(sort_metric) is None,
                                     x["metrics"].get(sort_metric, 0.0)), reverse=not ascending)
        return {"experiment": experiment, "n_runs": len(rows), "runs": rows}
