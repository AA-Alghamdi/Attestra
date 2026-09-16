"""Tests for the problem Envelope spec."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import envelope as E


def test_basic_envelope_builds_and_digests():
    env = E.from_goal(metric="macro_f1", theta=0.85, n=100, modality="audio", task_type="multiclass",
                      n_classes=6, max_latency_ms=50.0,
                      per_class_recall_floor={"escalate": 0.95}, group_key="agent_id", shift="grouped")
    assert env.objective.metric == "macro_f1"
    assert env.is_regression() is False
    assert env.is_small_data() is True
    assert env.wants_transfer() is True
    d = env.digest()
    assert d.startswith("env:")
    # round-trips through dict
    assert E.from_dict(env.as_dict()).digest() == d


def test_uncertifiable_metric_is_refused():
    with pytest.raises(E.EnvelopeError):
        E.Objective(metric="auroc", theta=0.8)   # not in the frozen certifier's list


def test_regression_objective_requires_regression_task():
    with pytest.raises(E.EnvelopeError):
        E.from_goal(metric="r2", theta=0.5, n=500, task_type="binary")
    # the consistent version builds
    env = E.from_goal(metric="r2", theta=0.5, n=500, task_type="regression")
    assert env.is_regression() is True


def test_per_class_floor_meaningless_for_regression():
    with pytest.raises(E.EnvelopeError):
        E.Envelope(objective=E.Objective("r2", 0.5),
                   data_regime=E.DataRegime(n=500, task_type="regression"),
                   constraints=E.Constraints(per_class_recall_floor={"x": 0.9}))


def test_grouped_shift_requires_group_key():
    with pytest.raises(E.EnvelopeError):
        E.DataRegime(n=100, shift="grouped")
    with pytest.raises(E.EnvelopeError):
        E.DataRegime(n=100, shift="temporal")


def test_constraint_bounds_validated():
    with pytest.raises(E.EnvelopeError):
        E.Constraints(per_class_recall_floor={"a": 1.5})
    with pytest.raises(E.EnvelopeError):
        E.Constraints(subgroup_parity={"attr": "skin_tone"})   # missing max_gap


def test_assert_objective_certifiable_matches_frozen_core():
    env = E.from_goal(metric="accuracy", theta=0.8, n=200)
    E.assert_objective_certifiable(env)   # must not raise; cross-checks the frozen science list


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
