"""frontier: the corrected generative autoresearch spine.

This package is ADDITIVE and SEPARATE. It does not edit or delete anything in
vectorforge/, vfplatform/, or attestra/. It REUSES the audited-sound certifier
(vectorforge.science + vfplatform.sealed) by importing it.

What it demonstrates (the integrity properties the audit found broken elsewhere):

  1. The proposal unit is a PROGRAM (arbitrary pipeline code), not a (family, params)
     pick from a fixed catalog. The catalog is demoted to seed/baseline priors.
     -> kills "model selection from a list"; feature engineering / target transforms
        are in-scope by construction (no n_features >= 60 gate).
  2. Candidate code runs in a REAL subprocess sandbox (rlimits + wall-clock timeout),
     never in-process exec with full builtins.
  3. The numeric firewall holds: untrusted code returns PREDICTIONS only; every
     decision-bearing number is recomputed by the trusted parent via science.py.
  4. The selected winner is certified on a TRUE held-out sealed test (one counted
     peek), never on the validation set used for selection.
  5. Round N's diagnosis (best-so-far + failures) conditions round N+1 proposals.

See ROADMAP.md for the full phased plan; this package is Phase 0 (the spine).
"""

from .program import Program, RunResult
from .task import Task
from .engine import ResearchEngine, EngineConfig, EngineResult

__all__ = [
    "Program",
    "RunResult",
    "Task",
    "ResearchEngine",
    "EngineConfig",
    "EngineResult",
]
