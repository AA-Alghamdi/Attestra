"""Read-only bootstrap of the FROZEN trust core.

This module is the ONLY place the LLM-shell reaches into the existing codebase, and it does so
strictly read-only: it puts the autoresearch root (for `ar.*`) and the product `vectorforge`
directory (for `science`) on sys.path and re-exports the frozen primitives the shell CALLS but
never edits. Nothing here mutates, monkey-patches, or shadows the originals.

Frozen primitives re-exported:
  - ar.compiler.compile      : zero-per-dataset STRUCTURAL inference (the step-1 profile)
  - ar.ontology.get          : task_type -> valid_metrics / default_metric / supported
  - ar.spec.propose_spec     : MEASURED threshold deriver (baseline + 0.60*headroom, NEEDS_HUMAN)
  - science.digest           : content-addressing primitive (sha256 of repr, 16 hex)
  - science.audit            : leakage auditor (referenced for the forbidden-field backstop)

The shell adds an LLM PROPOSER in front of these; it adds no statistic and touches no promotion
path. If a path is wrong on another machine, set VF_AUTORESEARCH_ROOT / VF_SCIENCE_DIR env vars.
"""
import os
import sys

# Defaults match this workspace; override via env for portability.
_AUTORESEARCH_ROOT = os.environ.get(
    "VF_AUTORESEARCH_ROOT", "/Users/abdullahalghamdi/vectorforge-autoresearch")
_SCIENCE_DIR = os.environ.get(
    "VF_SCIENCE_DIR", "/Users/abdullahalghamdi/vectorforge-product/vectorforge")

for _p in (_AUTORESEARCH_ROOT, _SCIENCE_DIR):
    if _p and _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

try:
    import science as _science           # noqa: E402  (when the autoresearch path puts it on sys.path)
except ModuleNotFoundError:
    from vectorforge import science as _science   # slim deploy (container): proper package import

# The autoresearch `ar` ontology package is OPTIONAL. Guarding its import lets _frozen (and thus the whole
# llm_shell package -- profiler/ops/intake -- and narration) IMPORT in a slim deployment WITHOUT `ar`; only
# free-text goal INTAKE actually needs it. If absent, the ar-backed handles raise a clear error WHEN CALLED
# (the UI/served paths pass an explicit kind and never call them). digest/audit come from local science.
try:
    import ar.compiler as _compiler      # noqa: E402
    import ar.ontology as _ontology      # noqa: E402
    import ar.spec as _spec              # noqa: E402
    compile = _compiler.compile
    ontology_get = _ontology.get
    propose_spec = _spec.propose_spec
    AR_AVAILABLE = True
except ImportError:
    AR_AVAILABLE = False

    def _ar_missing(*_a, **_k):
        raise RuntimeError("the autoresearch 'ar' ontology package is not installed; free-text goal intake "
                           "is unavailable -- pass an explicit kind/task_type/metric instead.")
    compile = ontology_get = propose_spec = _ar_missing

digest = _science.digest
audit = _science.audit

# The frozen task types the ontology knows (used by the resolver to reject hallucinated types).
KNOWN_TASK_TYPES = ("binary", "multiclass", "regression", "multilabel", "ranking")


def valid_metrics_for(task_type):
    """Return the ontology's valid metric tuple for a task_type, or () if unknown/unsupported."""
    try:
        spec = ontology_get(task_type)
    except Exception:  # noqa: BLE001  unknown task type
        return ()
    return tuple(spec.valid_metrics)


def default_metric_for(task_type):
    try:
        return ontology_get(task_type).default_metric
    except Exception:  # noqa: BLE001
        return None


def task_type_supported(task_type):
    try:
        return bool(ontology_get(task_type).supported)
    except Exception:  # noqa: BLE001
        return False


def valid_actions_for(task_type):
    """The ontology's CLOSED action vocabulary for a task_type (e.g. acquire_uncertainty,
    stronger_model, expand_representation, decompose). A4's proposed_next_action must be a member."""
    try:
        return tuple(ontology_get(task_type).valid_actions)
    except Exception:  # noqa: BLE001
        return ()
