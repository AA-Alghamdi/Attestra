"""Tests for cross-dataset replication + scope labeling.

Acceptance (Phase 10): the same recipe gets honestly different scopes -- shift_robust when it replicates
across the set, scoped when it only works on its own dataset, in_distribution when there is no replication
set. A primary that fails to certify is never labeled general."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import replication as RP


def _mk(name, certified, lb):
    return RP.DatasetCertOutcome(name, certified, lb)


def test_shift_robust_when_replicates():
    primary = _mk("D0", True, 0.88)
    replicas = [_mk("D1", True, 0.86), _mk("D2", True, 0.84), _mk("D3", True, 0.87), _mk("D4", True, 0.85)]
    rep = RP.classify_scope(primary, replicas, min_pass_rate=0.8)
    assert rep.scope == RP.SHIFT_ROBUST and rep.is_general()
    assert rep.worst_case_bound == pytest.approx(0.84)        # general claim is the worst-case bound


def test_scoped_when_fails_to_replicate():
    primary = _mk("D0", True, 0.92)
    replicas = [_mk("D1", False, 0.60), _mk("D2", False, 0.58), _mk("D3", True, 0.81), _mk("D4", False, 0.55)]
    rep = RP.classify_scope(primary, replicas, min_pass_rate=0.8)
    assert rep.scope == RP.SCOPED and not rep.is_general()
    assert rep.pass_rate == pytest.approx(0.25)


def test_in_distribution_when_no_replicas():
    rep = RP.classify_scope(_mk("D0", True, 0.9), [], min_pass_rate=0.8)
    assert rep.scope == RP.IN_DISTRIBUTION


def test_primary_failure_never_general():
    primary = _mk("D0", False, 0.40)
    replicas = [_mk("D1", True, 0.9), _mk("D2", True, 0.9)]
    rep = RP.classify_scope(primary, replicas, min_pass_rate=0.5)
    assert rep.scope == RP.SCOPED and not rep.is_general()


def test_replicate_drives_certify_fn_over_set():
    table = {"D0": _mk("D0", True, 0.88), "D1": _mk("D1", True, 0.85),
             "D2": _mk("D2", True, 0.86), "D3": _mk("D3", False, 0.6)}
    calls = []

    def certify_fn(name):
        calls.append(name)
        return table[name]

    rep = RP.replicate(certify_fn, "D0", ["D0", "D1", "D2", "D3"], min_pass_rate=0.6)
    assert "D0" in calls and calls.count("D0") == 1            # primary not double-certified
    assert rep.scope == RP.SHIFT_ROBUST                        # 2/3 replicas pass >= 0.6
    assert rep.pass_rate == pytest.approx(2 / 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
