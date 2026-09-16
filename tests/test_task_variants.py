"""Tests for the task-type variant registry.

Acceptance (Phase 11): each task type maps to the correct leak-safe discipline, and the regime-aware chooser
upgrades to the STRICTER variant when group/temporal structure is declared (timeseries -> temporal+embargo+
block bootstrap; a declared group_key -> whole-group split). A variant never relaxes below the task default."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vfplatform import task_variants as TV
from vfplatform.envelope import DataRegime


def test_registry_disciplines():
    assert TV.get_variant("binary").split_kind == TV.SPLIT_STRATIFIED
    assert TV.get_variant("binary").bootstrap_kind == TV.BOOTSTRAP_IID
    ts = TV.get_variant("timeseries")
    assert ts.split_kind == TV.SPLIT_TEMPORAL and ts.bootstrap_kind == TV.BOOTSTRAP_BLOCK and ts.requires_embargo
    assert TV.get_variant("ranking").split_kind == TV.SPLIT_GROUP


def test_unknown_task_type_raises():
    with pytest.raises(KeyError):
        TV.get_variant("clustering")


def test_regime_upgrades_to_temporal():
    regime = DataRegime(n=500, modality="tabular", task_type="binary", time_key="ts", shift="temporal")
    v = TV.variant_for_regime(regime)
    assert v.split_kind == TV.SPLIT_TEMPORAL and v.requires_embargo and v.bootstrap_kind == TV.BOOTSTRAP_BLOCK


def test_regime_upgrades_to_group():
    regime = DataRegime(n=500, modality="vision", task_type="multiclass", group_key="patient", shift="grouped")
    v = TV.variant_for_regime(regime)
    assert v.split_kind == TV.SPLIT_GROUP and v.bootstrap_kind == TV.BOOTSTRAP_BLOCK


def test_plain_regime_uses_default():
    regime = DataRegime(n=1000, modality="tabular", task_type="binary")
    v = TV.variant_for_regime(regime)
    assert v.split_kind == TV.SPLIT_STRATIFIED and not v.requires_embargo


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
