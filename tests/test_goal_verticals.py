"""Tests for vfplatform.goal_verticals — vision/text vertical routing from /goal.

Verifies:
1. can_drive_vertical correctly identifies vision/text specs
2. _build_records_vision produces valid harness-compatible records
3. _build_records_text produces valid harness-compatible records
4. route_vertical returns None for non-matching specs
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from vfplatform.goal_verticals import (
    can_drive_vertical, route_vertical,
    _build_records_vision, _build_records_text,
)


class FakeSpec:
    def __init__(self, kind, task_type, metric="accuracy"):
        self.kind = kind
        self.task_type = task_type
        self.metric = metric


class TestCanDriveVertical:
    def test_vision_binary(self):
        assert can_drive_vertical(FakeSpec("vision", "binary")) is True

    def test_vision_multiclass(self):
        assert can_drive_vertical(FakeSpec("vision", "multiclass")) is True

    def test_text_binary(self):
        assert can_drive_vertical(FakeSpec("text", "binary")) is True

    def test_text_multiclass(self):
        assert can_drive_vertical(FakeSpec("text", "multiclass")) is True

    def test_tabular_not_vertical(self):
        assert can_drive_vertical(FakeSpec("tabular", "binary")) is False

    def test_vision_regression_not_supported(self):
        assert can_drive_vertical(FakeSpec("vision", "regression")) is False


class TestBuildRecords:
    def test_vision_records(self):
        X = np.random.rand(10, 28 * 28)  # simulated MNIST-like
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        records = _build_records_vision(X, y, n_classes=2)
        assert len(records) == 10
        assert "features" in records[0]
        assert "target" in records[0]
        assert isinstance(records[0]["features"], dict)
        assert len(records[0]["features"]) == 28 * 28

    def test_text_records_1d(self):
        X = np.array(["good movie", "bad movie", "ok film", "terrible"])
        y = np.array([1, 0, 1, 0])
        records = _build_records_text(X, y, n_classes=2, text_key="text")
        assert len(records) == 4
        assert records[0]["features"]["text"] == "good movie"
        assert records[0]["target"] == "1"

    def test_text_records_2d_single_column(self):
        X = np.array([["hello"], ["world"], ["foo"]])
        y = np.array([0, 1, 0])
        records = _build_records_text(X, y, n_classes=2)
        assert records[0]["features"]["text"] == "hello"


class TestRouteVertical:
    def test_tabular_returns_none(self):
        """route_vertical returns None for tabular specs (caller falls back)."""
        X = np.random.rand(50, 10)
        y = np.array([0, 1] * 25)
        spec = FakeSpec("tabular", "binary")
        result = route_vertical("classify", X, y, 2, spec)
        assert result is None

    def test_vision_regression_returns_none(self):
        """Unsupported vertical combination returns None."""
        X = np.random.rand(50, 784)
        y = np.random.rand(50)
        spec = FakeSpec("vision", "regression")
        result = route_vertical("predict", X, y, 0, spec)
        assert result is None
