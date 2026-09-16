"""frontier.harness: the self-certifying harness fabric (Phase 3 core).

A harness adapts raw, modality-specific data into a Phase-0 `Task` (rows/metric/theta), exposes
a baseline suite + split protocol + right-axis metric, and SELF-CERTIFIES on a known-good
built-in dataset through the same frozen sealed gate before any of its numbers are trusted. A
HarnessRegistry routes by task-type key.

# === WIRING ===
# At import time this package AUTO-REGISTERS the full built-in modality fabric into the shared
# REGISTRY, so `import frontier.harness; REGISTRY.keys()` lists every modality and the Phase-3
# router (frontier/harness/router.py) resolves each modality to a REAL harness (not a declining
# fallback):
#
#   TabularHarness     -> "tabular", "classification", "regression"  (dense numeric tables)
#   TextHarness        -> "text", "text_classification"              (bag-of-words tfidf)
#   VisionHarness      -> "image", "vision"                          (pure-sklearn image features)
#   TimeSeriesHarness  -> "timeseries", "forecasting"                (lag/rolling + temporal split)
#
# The router keys on the inferred MODALITY ("image"/"timeseries"/"text"/"tabular"), so registering
# the vision harness under "image" and the time-series harness under "timeseries" is what makes a
# 4-D image tensor route to VisionHarness and a series route to TimeSeriesHarness end to end. The
# integrator/orchestrator uses:
#
#     from frontier.harness import lookup, REGISTRY
#     h = lookup("image")                 # also "vision"; timeseries via "timeseries"/"forecasting"
#     ok, cert = h.self_test()            # must be True before trusting h.adapt() downstream
#     task = h.adapt(X, y, kind="classification", theta=0.85)
#     # then ResearchEngine(EngineConfig(...)).run(task)  (frozen certify path)
#
# To author a NEW modality harness later: subclass Harness, implement
# adapt/baseline_suite/metric_for/_self_test_case, register() it here. Its numbers are trusted only
# after its self_test() certifies against its own known-good dataset.
"""

from .base import (
    Harness,
    HarnessCertificate,
    HarnessRegistry,
    REGISTRY,
    register,
    lookup,
)
from .tabular import TabularHarness
from .text import TextHarness, _TEXT          # noqa: F401  (import = self-registration side-effect)
from .vision import VisionHarness, _VISION    # noqa: F401  (import = self-registration side-effect)
from .timeseries import TimeSeriesHarness, _TIMESERIES  # noqa: F401  (self-registration side-effect)

# Register the built-in tabular harness under modality + kind keys (aliasing supported). The text,
# vision, and timeseries harnesses self-register on import (above); we register tabular here so the
# registration policy for the always-present default lives in one obvious place. Each register()
# is idempotent under re-import because the modality modules guard their own register() in try/except.
if "tabular" not in REGISTRY:
    _TABULAR = TabularHarness()
    register(_TABULAR, "tabular", "classification", "regression")

__all__ = [
    "Harness",
    "HarnessCertificate",
    "HarnessRegistry",
    "REGISTRY",
    "register",
    "lookup",
    "TabularHarness",
    "TextHarness",
    "VisionHarness",
    "TimeSeriesHarness",
]
