"""Tests for vfplatform.deploy — certificate → servable artifact + scoring entrypoint.

Verifies:
1. package_artifact writes model.joblib, predict.py, manifest.json, Dockerfile
2. Manifest contains correct metadata (family, sealed_lb, threshold, metric)
3. predict.py is syntactically valid Python
4. Round-trip: serialize → load → predict
5. Handles missing estimator (writes REFIT_REQUIRED.txt)
"""
import json
import os
import tempfile
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from vfplatform.deploy import package_artifact, summary, DeployArtifact


@pytest.fixture
def trained_model():
    """A simple trained sklearn model for testing."""
    X = np.random.rand(100, 5)
    y = (X[:, 0] > 0.5).astype(int)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def mock_certificate():
    """A mock GoalCertificate as a dict."""
    return {
        "champion": "logistic_regression",
        "champion_recipe": {"C": 1.0, "solver": "lbfgs"},
        "pooled_sealed_lb": 0.85,
        "theta_floor": 0.52,
        "spec": {"metric": "accuracy", "task_type": "binary", "n_rows": 100},
        "sealed_acc": {"task0": 0.9},
        "literature": {"source": "baseline"},
    }


class TestPackageArtifact:
    def test_basic_packaging(self, trained_model, mock_certificate):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = package_artifact(
                mock_certificate, trained_model,
                output_dir=tmpdir, docker=True, n_features=5, n_classes=2,
            )
            assert isinstance(artifact, DeployArtifact)
            assert os.path.exists(artifact.model_path)
            assert os.path.exists(artifact.predict_script)
            assert os.path.exists(artifact.manifest_path)
            assert os.path.exists(artifact.dockerfile_path)
            assert artifact.family == "logistic_regression"
            assert artifact.sealed_lb == 0.85
            assert artifact.estimator_available is True

    def test_manifest_content(self, trained_model, mock_certificate):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = package_artifact(
                mock_certificate, trained_model,
                output_dir=tmpdir, n_features=5, n_classes=2,
            )
            with open(artifact.manifest_path) as f:
                m = json.load(f)
            assert m["family"] == "logistic_regression"
            assert m["metric"] == "accuracy"
            assert m["threshold"] == 0.52
            assert m["sealed_lower_bound"] == 0.85
            assert m["n_features"] == 5
            assert m["n_classes"] == 2
            assert m["estimator_available"] is True
            assert m["model_file"] == "model.joblib"

    def test_predict_script_valid_python(self, trained_model, mock_certificate):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = package_artifact(
                mock_certificate, trained_model,
                output_dir=tmpdir, n_features=5, n_classes=2,
            )
            with open(artifact.predict_script) as f:
                code = f.read()
            # should be valid Python
            compile(code, artifact.predict_script, "exec")

    def test_round_trip(self, trained_model, mock_certificate):
        """Serialize → load → predict should produce valid predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = package_artifact(
                mock_certificate, trained_model,
                output_dir=tmpdir, n_features=5, n_classes=2,
            )
            import joblib
            loaded = joblib.load(artifact.model_path)
            X_new = np.random.rand(5, 5)
            preds = loaded.predict(X_new)
            assert len(preds) == 5
            assert all(p in (0, 1) for p in preds)

    def test_no_estimator(self, mock_certificate):
        """When estimator is None, writes REFIT_REQUIRED.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = package_artifact(
                mock_certificate, None,
                output_dir=tmpdir, n_features=5, n_classes=2,
            )
            assert artifact.estimator_available is False
            refit_path = os.path.join(tmpdir, "REFIT_REQUIRED.txt")
            assert os.path.exists(refit_path)
            with open(refit_path) as f:
                content = f.read()
            assert "logistic_regression" in content

    def test_no_docker(self, trained_model, mock_certificate):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = package_artifact(
                mock_certificate, trained_model,
                output_dir=tmpdir, docker=False, n_features=5, n_classes=2,
            )
            assert artifact.dockerfile_path is None


class TestSummary:
    def test_summary_output(self, trained_model, mock_certificate):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = package_artifact(
                mock_certificate, trained_model,
                output_dir=tmpdir, n_features=5, n_classes=2,
            )
            s = summary(artifact)
            assert s["family"] == "logistic_regression"
            assert "model.joblib" in s["files"]
            assert "predict.py" in s["files"]
            assert "manifest.json" in s["files"]
