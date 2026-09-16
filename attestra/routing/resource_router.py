"""Resource Router (Level 3) — routes strategy to execution substrate.

Given:
  - strategy_class + cost estimate + available resources
Decides:
  - Where to execute: local CPU, local GPU, serverless GPU, parallel pool
  - How many resources to allocate
  - Timeout and memory limits
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ExecutionSubstrate(str, Enum):
    """Where code runs."""
    LOCAL_CPU = "local_cpu"                # In-process or subprocess, CPU only
    LOCAL_GPU = "local_gpu"                # Local GPU (if available)
    SERVERLESS_GPU = "serverless_gpu"      # Remote GPU (Prime Intellect, RunPod, etc.)
    PARALLEL_CPU = "parallel_cpu"          # Process pool on local CPUs
    PARALLEL_GPU = "parallel_gpu"          # Multiple GPUs
    ASYNC_REMOTE = "async_remote"          # Submit and poll (for long-running)


@dataclass
class ResourceAllocation:
    """Concrete resource allocation for an execution."""
    substrate: ExecutionSubstrate
    n_workers: int = 1
    cpu_seconds: int = 60
    memory_mb: int = 4096
    wall_seconds: float = 120.0
    gpu_type: Optional[str] = None    # "A100", "T4", etc.
    gpu_memory_gb: Optional[int] = None
    # Async config
    poll_interval_s: float = 10.0
    max_poll_time_s: float = 3600.0
    # Cost estimate
    estimated_cost_usd: float = 0.0
    reasoning: str = ""


class ResourceRouter:
    """Routes strategy decisions to execution substrates.

    Considers:
      - Strategy requirements (neural → GPU, sklearn → CPU)
      - Dataset size (large → more memory, more time)
      - Budget constraints (expensive strategies only if budget allows)
      - Available hardware (GPU present? Multi-core?)
    """

    def __init__(
        self,
        gpu_available: bool = False,
        n_cpus: int = 4,
        memory_gb: float = 16.0,
        serverless_enabled: bool = False,
    ):
        self._gpu_available = gpu_available
        self._n_cpus = n_cpus
        self._memory_gb = memory_gb
        self._serverless = serverless_enabled

    def route(
        self,
        strategy_class: str,
        n_samples: int,
        n_features: int,
        *,
        budget_s: float = 300.0,
        model_family: str = "",
        parallel_proposals: int = 1,
    ) -> ResourceAllocation:
        """Route a strategy to an execution substrate.

        Parameters
        ----------
        strategy_class : str
            The strategy class (from StrategyClass enum value).
        n_samples : int
            Training set size.
        n_features : int
            Feature count.
        budget_s : float
            Time budget for this execution.
        model_family : str
            Specific model family name.
        parallel_proposals : int
            How many proposals to run (for parallelism).

        Returns
        -------
        ResourceAllocation
            Where and how to execute.
        """
        # Determine if GPU is needed
        needs_gpu = self._needs_gpu(strategy_class, model_family)

        # Estimate memory needs
        mem_mb = self._estimate_memory(n_samples, n_features, strategy_class)

        # Determine substrate
        if needs_gpu:
            if self._gpu_available:
                substrate = ExecutionSubstrate.LOCAL_GPU
                reasoning = "Neural model requires GPU; local GPU available"
            elif self._serverless:
                substrate = ExecutionSubstrate.SERVERLESS_GPU
                reasoning = "Neural model requires GPU; using serverless"
            else:
                # Fallback to CPU for small neural models
                substrate = ExecutionSubstrate.LOCAL_CPU
                reasoning = "Neural model requested but no GPU available; falling back to CPU"
                # Reduce ambition
                mem_mb = min(mem_mb, int(self._memory_gb * 1024 * 0.7))
        elif parallel_proposals > 1 and self._n_cpus > 1:
            substrate = ExecutionSubstrate.PARALLEL_CPU
            reasoning = f"Parallel execution of {parallel_proposals} proposals on {self._n_cpus} CPUs"
        else:
            substrate = ExecutionSubstrate.LOCAL_CPU
            reasoning = "Standard CPU execution"

        # Calculate workers
        n_workers = 1
        if substrate == ExecutionSubstrate.PARALLEL_CPU:
            n_workers = min(parallel_proposals, self._n_cpus - 1, 8)

        # Calculate timeouts
        cpu_seconds = int(min(budget_s * 0.9, 600))  # 90% of budget, max 10 min
        wall_seconds = min(budget_s, 1800.0)  # max 30 min

        # Large datasets need more time
        if n_samples > 50000:
            cpu_seconds = int(cpu_seconds * 1.5)
            wall_seconds *= 1.5

        return ResourceAllocation(
            substrate=substrate,
            n_workers=n_workers,
            cpu_seconds=cpu_seconds,
            memory_mb=mem_mb,
            wall_seconds=wall_seconds,
            gpu_type="T4" if substrate in (ExecutionSubstrate.LOCAL_GPU, ExecutionSubstrate.SERVERLESS_GPU) else None,
            reasoning=reasoning,
        )

    def _needs_gpu(self, strategy_class: str, model_family: str) -> bool:
        """Determine if a strategy requires GPU."""
        gpu_strategies = {
            "neural_tabular", "neural_vision", "neural_nlp", "neural_timeseries",
        }
        if strategy_class in gpu_strategies:
            return True

        gpu_families = {
            "pytorch", "tensorflow", "tabnet", "ft_transformer",
            "resnet", "efficientnet", "vit", "bert", "gpt",
            "lstm", "nbeats", "temporal_fusion",
        }
        if any(f in model_family.lower() for f in gpu_families):
            return True

        return False

    def _estimate_memory(self, n_samples: int, n_features: int, strategy_class: str) -> int:
        """Estimate memory needs in MB."""
        # Base: data itself
        data_mb = (n_samples * n_features * 8) / (1024 * 1024)  # float64

        # Overhead multiplier by strategy
        multipliers = {
            "gradient_boosting": 3.0,
            "tree_ensemble": 4.0,
            "linear": 2.0,
            "svm": 5.0,  # SVM can use lots of memory
            "neural_tabular": 4.0,
            "neural_vision": 8.0,
            "neural_nlp": 8.0,
            "ensemble_stacking": 6.0,
            "feature_heavy": 4.0,
        }
        mult = multipliers.get(strategy_class, 3.0)

        # Minimum 512 MB, maximum 80% of available
        mem_mb = max(512, int(data_mb * mult + 256))
        max_mb = int(self._memory_gb * 1024 * 0.8)
        return min(mem_mb, max_mb)
