"""Parallel execution — process pool for proposals + async GPU dispatch.

Enables:
  - Within-run parallelism: run N proposals simultaneously
  - ASHA successive halving with parallel arms
  - Async GPU job submission and polling
  - Budget-aware early termination of parallel workers

Architecture:
  - ProposalPool: manages a pool of workers executing proposals in parallel
  - AsyncExecutor: submit-and-poll for long-running GPU jobs
  - ParallelASHA: parallel successive halving (start many, kill losers early)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Proposal execution result
# ---------------------------------------------------------------------------

@dataclass
class ProposalResult:
    """Result of executing a single proposal."""
    proposal_id: str
    success: bool
    score: Optional[float] = None
    predictions: Optional[np.ndarray] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0
    worker_id: int = 0
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parallel proposal pool
# ---------------------------------------------------------------------------

def _execute_proposal_worker(
    proposal_code: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric_fn_name: str,
    cpu_seconds: int,
    mem_mb: int,
    proposal_id: str,
) -> Dict:
    """Worker function for parallel proposal execution.

    Runs in a separate process with restricted builtins and AST checking.
    """
    import resource
    import signal

    t0 = time.time()

    # Set resource limits
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
    except (ValueError, resource.error):
        pass

    try:
        mem_bytes = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, resource.error):
        pass

    # AST static check before execution
    try:
        from .sandbox import static_check, _build_safe_builtins
        report = static_check(proposal_code, entrypoint="")
        if not report.ok:
            return {
                "proposal_id": proposal_id,
                "success": False,
                "error": f"AST gate rejected: {'; '.join(report.violations[:3])}",
                "elapsed_s": time.time() - t0,
            }
        safe_builtins = _build_safe_builtins()
    except ImportError:
        safe_builtins = __builtins__

    # Execute with restricted builtins
    exec_globals = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "np": np,
        "__builtins__": safe_builtins,
    }

    try:
        exec(proposal_code, exec_globals)
        predictions = exec_globals.get("predictions")

        if predictions is None:
            return {
                "proposal_id": proposal_id,
                "success": False,
                "error": "No 'predictions' variable produced",
                "elapsed_s": time.time() - t0,
            }

        predictions = np.asarray(predictions)

        # Evaluate
        score = None
        if y_test is not None and metric_fn_name:
            from sklearn import metrics as sklearn_metrics
            metric_fn = getattr(sklearn_metrics, metric_fn_name, None)
            if metric_fn:
                score = metric_fn(y_test, predictions)

        return {
            "proposal_id": proposal_id,
            "success": True,
            "score": score,
            "predictions": predictions.tolist(),
            "elapsed_s": time.time() - t0,
        }

    except Exception as e:
        return {
            "proposal_id": proposal_id,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "elapsed_s": time.time() - t0,
        }


class ProposalPool:
    """Manages parallel execution of proposals.

    Runs N proposals simultaneously using a process pool,
    with budget-aware early termination.
    """

    def __init__(
        self,
        max_workers: int = 4,
        cpu_seconds_per_proposal: int = 60,
        mem_mb_per_proposal: int = 2048,
    ):
        self._max_workers = max(1, min(max_workers, mp.cpu_count() - 1, 8))
        self._cpu_seconds = cpu_seconds_per_proposal
        self._mem_mb = mem_mb_per_proposal

    def execute_batch(
        self,
        proposals: List[Dict],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "accuracy_score",
        wall_timeout_s: float = 300.0,
    ) -> List[ProposalResult]:
        """Execute a batch of proposals in parallel.

        Parameters
        ----------
        proposals : list of dict
            Each dict has: {"id": str, "code": str, "label": str}
        X_train, y_train : arrays
            Training data passed to each proposal.
        X_val, y_val : arrays
            Validation data for evaluation.
        metric : str
            sklearn metric function name (e.g., "accuracy_score").
        wall_timeout_s : float
            Maximum wall-clock time for the entire batch.

        Returns
        -------
        List[ProposalResult]
            Results for each proposal (in order of completion).
        """
        results: List[ProposalResult] = []
        t0 = time.time()

        if not proposals:
            return results

        # For small batches or single CPU, run sequentially
        if len(proposals) <= 1 or self._max_workers <= 1:
            for p in proposals:
                if time.time() - t0 > wall_timeout_s:
                    break
                result = self._execute_single(
                    p, X_train, y_train, X_val, y_val, metric,
                )
                results.append(result)
            return results

        # Parallel execution
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            futures: Dict[Future, str] = {}

            for p in proposals:
                future = executor.submit(
                    _execute_proposal_worker,
                    p["code"],
                    X_train, y_train, X_val, y_val,
                    metric,
                    self._cpu_seconds,
                    self._mem_mb,
                    p["id"],
                )
                futures[future] = p["id"]

            # Collect results with timeout
            remaining_time = wall_timeout_s - (time.time() - t0)
            for future in as_completed(futures, timeout=max(remaining_time, 1)):
                try:
                    raw = future.result(timeout=10)
                    results.append(ProposalResult(
                        proposal_id=raw["proposal_id"],
                        success=raw["success"],
                        score=raw.get("score"),
                        predictions=np.array(raw["predictions"]) if raw.get("predictions") else None,
                        error=raw.get("error"),
                        elapsed_s=raw.get("elapsed_s", 0.0),
                    ))
                except Exception as e:
                    pid = futures[future]
                    results.append(ProposalResult(
                        proposal_id=pid,
                        success=False,
                        error=f"Worker error: {type(e).__name__}: {str(e)[:200]}",
                        elapsed_s=time.time() - t0,
                    ))

                # Budget check
                if time.time() - t0 > wall_timeout_s:
                    # Cancel remaining futures
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break

        return results

    def _execute_single(
        self, proposal: Dict,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        metric: str,
    ) -> ProposalResult:
        """Execute a single proposal (sequential fallback)."""
        t0 = time.time()
        try:
            raw = _execute_proposal_worker(
                proposal["code"],
                X_train, y_train, X_val, y_val,
                metric,
                self._cpu_seconds,
                self._mem_mb,
                proposal["id"],
            )
            return ProposalResult(
                proposal_id=raw["proposal_id"],
                success=raw["success"],
                score=raw.get("score"),
                predictions=np.array(raw["predictions"]) if raw.get("predictions") else None,
                error=raw.get("error"),
                elapsed_s=raw.get("elapsed_s", 0.0),
            )
        except Exception as e:
            return ProposalResult(
                proposal_id=proposal["id"],
                success=False,
                error=str(e),
                elapsed_s=time.time() - t0,
            )


# ---------------------------------------------------------------------------
# Async executor for GPU / long-running jobs
# ---------------------------------------------------------------------------

@dataclass
class AsyncJob:
    """A submitted async job."""
    job_id: str
    status: str = "pending"     # "pending" | "running" | "completed" | "failed"
    submitted_at: float = 0.0
    completed_at: Optional[float] = None
    result: Optional[Dict] = None
    error: Optional[str] = None


class AsyncExecutor:
    """Submit-and-poll executor for long-running jobs (GPU training, etc.).

    Supports:
      - Job submission (returns immediately with job_id)
      - Status polling (check if job is done)
      - Result retrieval
      - Timeout enforcement
    """

    def __init__(self, poll_interval_s: float = 10.0, max_wait_s: float = 3600.0):
        self._poll_interval = poll_interval_s
        self._max_wait = max_wait_s
        self._jobs: Dict[str, AsyncJob] = {}

    def submit(self, fn: Callable, *args, **kwargs) -> str:
        """Submit a job for async execution.

        Returns a job_id for polling.
        """
        import uuid
        job_id = str(uuid.uuid4())[:8]
        job = AsyncJob(job_id=job_id, status="pending", submitted_at=time.time())
        self._jobs[job_id] = job

        # Start in a background process
        def _runner():
            try:
                result = fn(*args, **kwargs)
                job.status = "completed"
                job.result = result
                job.completed_at = time.time()
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                job.completed_at = time.time()

        import threading
        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        job.status = "running"

        return job_id

    def poll(self, job_id: str) -> AsyncJob:
        """Check the status of a submitted job."""
        return self._jobs.get(job_id, AsyncJob(job_id=job_id, status="not_found"))

    def wait(self, job_id: str, timeout_s: Optional[float] = None) -> AsyncJob:
        """Block until a job completes or times out."""
        timeout = timeout_s or self._max_wait
        t0 = time.time()
        while time.time() - t0 < timeout:
            job = self.poll(job_id)
            if job.status in ("completed", "failed", "not_found"):
                return job
            time.sleep(self._poll_interval)
        # Timeout
        job = self.poll(job_id)
        if job.status not in ("completed", "failed"):
            job.status = "failed"
            job.error = f"Timeout after {timeout:.0f}s"
        return job

    def submit_and_wait(self, fn: Callable, *args, timeout_s: Optional[float] = None, **kwargs) -> AsyncJob:
        """Submit and block until completion."""
        job_id = self.submit(fn, *args, **kwargs)
        return self.wait(job_id, timeout_s=timeout_s)


# ---------------------------------------------------------------------------
# Parallel ASHA (successive halving)
# ---------------------------------------------------------------------------

class ParallelASHA:
    """Parallel Asynchronous Successive Halving Algorithm.

    Starts many proposals, evaluates cheaply, kills the bottom half,
    evaluates survivors with more budget, repeat.

    Rungs:
      rung 0: all proposals, 1/4 budget each
      rung 1: top 50%, 1/2 budget each
      rung 2: top 25%, full budget each
    """

    def __init__(
        self,
        n_rungs: int = 3,
        reduction_factor: int = 2,
        max_workers: int = 4,
    ):
        self._n_rungs = n_rungs
        self._reduction = reduction_factor
        self._pool = ProposalPool(max_workers=max_workers)

    def run(
        self,
        proposals: List[Dict],
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        metric: str = "accuracy_score",
        total_budget_s: float = 300.0,
    ) -> List[ProposalResult]:
        """Run ASHA: parallel execution with successive halving.

        Returns results sorted by score (best first).
        """
        n_proposals = len(proposals)
        if n_proposals == 0:
            return []

        # Calculate per-rung budget
        rung_budget = total_budget_s / self._n_rungs
        surviving = list(proposals)
        all_results: Dict[str, ProposalResult] = {}

        for rung in range(self._n_rungs):
            if not surviving:
                break

            # Budget per proposal in this rung
            n_alive = len(surviving)
            per_proposal_budget = rung_budget / max(n_alive, 1)

            # Execute all surviving proposals
            self._pool._cpu_seconds = max(10, int(per_proposal_budget * 0.9))
            results = self._pool.execute_batch(
                surviving, X_train, y_train, X_val, y_val,
                metric=metric,
                wall_timeout_s=rung_budget,
            )

            # Update results
            for r in results:
                all_results[r.proposal_id] = r

            # Rank by score and keep top half
            scored = [(r.proposal_id, r.score or 0.0) for r in results if r.success]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Keep top 1/reduction_factor
            n_keep = max(1, len(scored) // self._reduction)
            kept_ids = {pid for pid, _ in scored[:n_keep]}

            surviving = [p for p in surviving if p["id"] in kept_ids]

        # Return all results sorted by score
        final = sorted(all_results.values(), key=lambda r: r.score or 0.0, reverse=True)
        return final
