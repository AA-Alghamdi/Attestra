"""VectorForge: autonomous ML execution (LEGACY — use attestra instead).

This package is maintained for backward compatibility only.
All active development is in the `attestra` package (primary) and `frontier` (engine).
"""
import warnings as _warnings

_warnings.warn(
    "vectorforge is deprecated. Use `attestra` instead: "
    "from attestra import orchestrate",
    DeprecationWarning,
    stacklevel=2,
)

from .service import create_goal, draft_plan, approve, run, predict  # noqa: F401
from . import store, science, ml, runner, domain  # noqa: F401

__version__ = "0.1.0"
