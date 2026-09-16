"""PHASE 0 SAFETY NET -- every autoresearcher module is importable + reachable, and the FROZEN core is intact.

This is the net that makes every later 'wired' claim testable. It asserts (1) the frozen certifier core is
byte-identical to its pinned baseline (no PR in this series may change it), and (2) each new module imports
cleanly and exposes the public surface the loop integrates against. A regression here means a module went
dark or the trust invariant was touched.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (module, [required public attributes]) -- the integration surface each phase exposes to run_goal_loop
WIRED_MODULES = [
    ("vfplatform.envelope", ["Envelope", "Objective", "Constraints", "DataRegime", "from_goal"]),
    ("vfplatform.gate", ["certified_under_envelope", "Measurements", "CertDecision"]),
    ("vfplatform.verification", ["VerificationCascade", "Candidate", "CascadeResult"]),
    ("vfplatform.splits", ["make_leaksafe_splits", "SplitResult", "stratified_split", "temporal_split"]),
    ("vfplatform.data_cert", ["certify_dataset", "detect_near_dup_straddle"]),
    ("vfplatform.datapool", ["DataPool", "PoolDataset"]),
    ("vfplatform.surrogate", ["Surrogate", "CalibrationTracker", "OutcomeRecord"]),
    ("vfplatform.search", ["BudgetedSearch", "SearchProblem", "SearchResult"]),
    ("vfplatform.regeneration", ["regenerate", "QDArchive", "mutate", "recombine"]),
    ("vfplatform.featurizers", ["ImageFeaturizer", "TextFeaturizer", "AudioFeaturizer"]),
    ("vfplatform.pareto", ["ParetoFront", "ParetoCandidate"]),
    ("vfplatform.data_ops", ["dedupe", "reweight_balanced", "clean_label_noise", "active_label_query"]),
    ("vfplatform.replication", ["replicate", "classify_scope", "ReplicationReport"]),
    ("vfplatform.bench", ["run_benchmark", "leaderboard", "summarize"]),
    ("vfplatform.meta_certifier", ["validate_certifier", "MetaCertReport"]),
    ("vfplatform.task_variants", ["get_variant", "variant_for_regime", "REGISTRY"]),
    ("vfplatform.authoring_bridge", ["admit_method", "evaluate_admitted", "to_candidate"]),
]


@pytest.mark.parametrize("modname,attrs", WIRED_MODULES, ids=[m for m, _ in WIRED_MODULES])
def test_module_imports_and_exposes_surface(modname, attrs):
    mod = importlib.import_module(modname)
    missing = [a for a in attrs if not hasattr(mod, a)]
    assert not missing, f"{modname} is missing public symbols: {missing}"


def test_frozen_core_intact():
    from vfplatform import frozen_integrity as fi
    ok, detail = fi.verify_frozen_core(raise_on_fail=False)
    assert ok, f"FROZEN CORE MODIFIED -- the trust invariant is broken: {detail}"


def test_frozen_hashes_match_pinned_constants():
    from vfplatform import frozen_integrity as fi
    live = fi.compute_live()
    assert live["vectorforge/science.py"] == fi.FROZEN_SHA256["vectorforge/science.py"]
    assert live["vfplatform/sealed.py"] == fi.FROZEN_SHA256["vfplatform/sealed.py"]


def test_envelope_gate_chain_reachable():
    # a representative end-to-end import chain: envelope describes the problem, gate wraps the frozen cert
    from vfplatform.envelope import from_goal
    from vfplatform.gate import certified_under_envelope, Measurements, CertDecision
    env = from_goal(metric="accuracy", theta=0.8, n=500, modality="tabular", task_type="binary")
    assert env.objective.theta == 0.8
    assert callable(certified_under_envelope)
    assert Measurements is not None and CertDecision is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
