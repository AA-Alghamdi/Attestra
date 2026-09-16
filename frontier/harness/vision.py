"""VisionHarness: REAL image classification through the same frozen sealed gate.

What this proves (the Phase-3 breadth goal)
-------------------------------------------
The Phase-0 spine (propose -> sandbox -> score-on-val -> certify-on-sealed) is *modality
agnostic*: only the data->Task adapter changes; the certify path is byte for byte the audited
sound one (vectorforge.science + vfplatform.sealed). This harness is that adapter for image
classification using a PURE-SKLEARN featurizer (no torch, no GPU, no network), and it
SELF-CERTIFIES on a known-good built-in dataset (sklearn `load_digits`, treated as 8x8 images)
before any of its numbers are trusted -- exactly the trust gate `frontier.harness.base.Harness`
defines.

The featurizer (all sklearn / numpy, deterministic, UNSUPERVISED)
----------------------------------------------------------------
An image batch is a tensor `(n, H, W)` (grayscale) or `(n, H, W, C)`. adapt() turns it into a
dense float feature matrix the Phase-0 sandbox can ship and a plain sklearn classifier can fit:

  1. flatten the raw pixels (intensity features), AND
  2. compute simple HOG-like *gradient* features: per-pixel finite-difference gradients
     (Sobel-ish 1-step differences) in x and y, whose magnitude is pooled into a small grid of
     orientation-free cells. Gradients are the canonical hand-crafted image feature (HOG/SIFT
     lineage): they make edges/strokes linearly separable where raw intensities are not.
  3. standardize (StandardScaler) and, when the flattened dimension is large, project with PCA to
     a bounded number of components (whiten=False). StandardScaler+PCA is fit on the WHOLE batch;
     both are UNSUPERVISED (they never see labels), so fitting them before the spine splits leaks
     no target information -- the sealed certificate is still computed on held-out LABELS via the
     one-peek gate. (A production deployment refits the transform inside the train fold; that is a
     substrate refinement, not a contract change, and identical to the TextHarness tfidf rationale.)

Everything that *decides* (which classifier wins, whether it promotes) is the frozen certifier;
the harness only ASSEMBLES the feature matrix + chooses baselines/split/metric.

# === WIRING ===
# This is the concrete Harness the Phase-3 router resolves for the "image"/"vision" task-type
# keys. It is registered into the shared REGISTRY by frontier/harness/__init__.py at import time,
# so the integrator/orchestrator does exactly what it does for the tabular/text harness:
#
#     from frontier.harness import lookup
#     h = lookup("image")                               # also reachable via "vision"
#     ok, cert = h.self_test()                          # GATE: trust nothing until ok is True
#     assert ok, f"vision harness failed self-test: {cert.detail}"
#     task = h.adapt(images, labels, kind="classification", theta=0.85, name="my_images")
#     #   adapt() featurizes the image tensor and returns a Task whose .X is a dense float matrix.
#     result = ResearchEngine(EngineConfig(rounds=2)).run(task)   # SAME Phase-0 certify path
#
# The CoreOrchestrator reaches this harness automatically: router.route() types a 3-D/4-D image
# tensor as modality "image" and resolves REGISTRY.get("image") to this instance, so a run on
# load_digits-as-images certifies end to end with a VISION baseline as the winner -- NOT a generic
# tabular fallback. (See frontier/tests/test_harness_modalities.py for the wired end-to-end test.)
#
# Ordering / argument-shape contract the integrator MUST preserve (same as base.Harness):
#   1. self_test() returns True BEFORE any adapt() Task is trusted (runs the harness's
#      adapt->split->baseline path through the REAL Phase-0 ResearchEngine, seeded with this
#      harness's baseline_suite()). The frozen sealed certifier -- not this harness -- promotes.
#   2. adapt(X, y, *, kind, theta, name, metric): X is an image tensor (n,H,W) or (n,H,W,C); y is
#      the parallel label sequence. theta is the caller's standard; the harness never invents a
#      promotion-bearing number.
#   3. metric_for("classification") -> "accuracy" (the default vision clf axis; balanced_accuracy
#      / macro_f1 also available for imbalanced sets). split_protocol -> (0.30, 0.20).
#   4. baseline_suite("classification") -> [Program] (SVC-rbf / logreg / kNN); seeds/fallbacks,
#      never promoters.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Repo root on sys.path so sibling frontier modules + the sound certifier resolve from anywhere
# (mirrors base.py / certify.py / text.py; this file lives under frontier/harness/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier.program import Program            # noqa: E402
from frontier.task import Task                  # noqa: E402

from .base import Harness, register             # noqa: E402


# Metrics the frozen certifier (vectorforge.science.score_metric) supports for classification. We
# expose only right-axis choices so an image task cannot be routed to a regression metric.
_VALID_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
_DEFAULT_METRIC = "accuracy"


# --------------------------------------------------------------------------- baseline suite
# Each baseline is a Program whose build_estimator() operates on the FLOAT feature matrix adapt()
# produced. These are the canonical strong baselines for dense, standardized + PCA-reduced image
# features: an RBF-kernel SVM (the textbook strong digits baseline), a linear logistic regression,
# and a k-NN (instance-based, strong on featurized digits). SEEDS/FALLBACKS, never the promoter.
_VISION_BASELINES: List[Tuple[str, str]] = [
    ("vision_svc_rbf",
     "from sklearn.svm import SVC\n"
     "def build_estimator():\n"
     "    return SVC(C=5.0, gamma='scale', kernel='rbf')\n"),
    ("vision_logreg",
     "from sklearn.linear_model import LogisticRegression\n"
     "def build_estimator():\n"
     "    return LogisticRegression(C=2.0, max_iter=3000)\n"),
    ("vision_knn",
     "from sklearn.neighbors import KNeighborsClassifier\n"
     "def build_estimator():\n"
     "    return KNeighborsClassifier(n_neighbors=5, weights='distance')\n"),
]


def vision_baseline_programs() -> List[Program]:
    """The vision baseline suite as seed Programs (SVC-rbf / logreg / kNN). Fresh list per call."""
    return [
        Program(code=code, source="seed", label=name,
                provenance={"suite": "vision_baselines", "harness": "image"})
        for name, code in _VISION_BASELINES
    ]


# --------------------------------------------------------------------------- featurizer

def _to_grayscale_batch(X) -> np.ndarray:
    """Coerce an image batch to a float `(n, H, W)` grayscale tensor.

    Accepts (n,H,W) grayscale or (n,H,W,C) color (averaged over channels). Raises on anything
    that is not a plausible image batch rather than emit a silently-corrupt feature matrix.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 4:
        # (n, H, W, C) -> average channels to grayscale (luminance-agnostic, sufficient for the
        # hand-crafted gradient/intensity features; color-specific harnesses can subclass).
        arr = arr.mean(axis=3)
    if arr.ndim != 3:
        raise ValueError(
            f"image X must be (n,H,W) or (n,H,W,C); got ndim={arr.ndim} shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"image X has {n_bad} non-finite pixels; clean before adapt")
    return arr


def _gradient_cell_features(batch: np.ndarray, *, grid: int = 2) -> np.ndarray:
    """HOG-like gradient features: pool finite-difference gradient magnitude into a grid of cells.

    For each image we compute 1-step horizontal/vertical gradients (np.diff, edge-padded to keep
    HxW), take their magnitude, and average-pool over a `grid`x`grid` partition of the image. This
    yields `grid*grid` orientation-free edge-energy features per image -- a compact, deterministic,
    pure-numpy stand-in for HOG that makes strokes/edges linearly informative. No sklearn, no torch.
    """
    n, H, W = batch.shape
    # Horizontal gradient (along W) and vertical gradient (along H), edge-padded back to HxW.
    gx = np.zeros_like(batch)
    gy = np.zeros_like(batch)
    gx[:, :, :-1] = np.diff(batch, axis=2)
    gy[:, :-1, :] = np.diff(batch, axis=1)
    mag = np.sqrt(gx * gx + gy * gy)
    # Average-pool the magnitude into a grid of cells. Use linspace edges so non-divisible sizes
    # still partition cleanly (each cell gets a contiguous block of rows/cols).
    r_edges = np.linspace(0, H, grid + 1).astype(int)
    c_edges = np.linspace(0, W, grid + 1).astype(int)
    feats = np.empty((n, grid * grid), dtype=float)
    k = 0
    for i in range(grid):
        for j in range(grid):
            r0, r1 = r_edges[i], max(r_edges[i] + 1, r_edges[i + 1])
            c0, c1 = c_edges[j], max(c_edges[j] + 1, c_edges[j + 1])
            cell = mag[:, r0:r1, c0:c1]
            feats[:, k] = cell.reshape(n, -1).mean(axis=1)
            k += 1
    return feats


class VisionHarness(Harness):
    """Harness for image classification via a pure-sklearn featurizer + the frozen sealed gate.

    `adapt()` takes an image tensor (n,H,W) or (n,H,W,C) (NOT a pre-flattened matrix), builds
    flatten+gradient features, standardizes (StandardScaler) and optionally projects (PCA), and
    returns a standard `frontier.task.Task` with a dense float `X` the Phase-0 engine certifies
    unchanged. `self_test()` (inherited from base.Harness) certifies the harness on sklearn digits
    treated as 8x8 images before its numbers are trusted.

    Constructor knobs control the featurizer:
      - pca_components : cap on PCA components when the flattened+gradient dim exceeds it. None or
                         a value >= the feature dim disables PCA (StandardScaler only). PCA is
                         UNSUPERVISED and fit on the whole batch (see module docstring).
      - grid           : HOG-like gradient pooling grid (grid*grid edge-energy features added).
      - use_gradients  : include the gradient-cell features (True) or intensities only (False).
    """

    key = "image"
    kinds = ("classification",)

    def __init__(self, *, pca_components: Optional[int] = 40, grid: int = 2,
                 use_gradients: bool = True):
        super().__init__()
        self.pca_components = pca_components
        self.grid = grid
        self.use_gradients = use_gradients

    # ------------------------------------------------------------------ adapter API
    def metric_for(self, kind: str) -> str:
        """Right-axis default metric for image classification (accuracy)."""
        if kind != "classification":
            raise ValueError(f"VisionHarness handles 'classification', not {kind!r}")
        return _DEFAULT_METRIC

    def _featurize(self, batch: np.ndarray) -> np.ndarray:
        """Flatten + (optional) gradient features -> StandardScaler -> (optional) PCA.

        All transforms are sklearn/numpy and UNSUPERVISED. Returns a dense float matrix. PCA only
        engages when the raw feature dimension exceeds `pca_components` (and there are enough
        samples to estimate the components); otherwise StandardScaler alone is used.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        n = batch.shape[0]
        flat = batch.reshape(n, -1).astype(float)              # intensity features
        if self.use_gradients:
            grad = _gradient_cell_features(batch, grid=self.grid)
            raw = np.concatenate([flat, grad], axis=1)
        else:
            raw = flat

        Xs = StandardScaler().fit_transform(raw)
        d = Xs.shape[1]
        if self.pca_components is not None and d > self.pca_components:
            # n_components must be <= min(n_samples, n_features).
            k = min(self.pca_components, d, n)
            if k >= 1:
                Xs = PCA(n_components=k, random_state=0).fit_transform(Xs)
        return np.asarray(Xs, dtype=float)

    def adapt(self, X, y, *, kind: str, theta: float, name: str = "task",
              metric: str = "") -> Task:
        """Featurize an image tensor into a Phase-0 Task the engine certifies unchanged.

        `X` is an image tensor (n,H,W) or (n,H,W,C); `y` is the parallel label sequence. Validates
        early rather than emit a silently-corrupt Task:
          - kind must be classification;
          - X is a 3-D or 4-D finite image tensor;
          - labels length matches the batch and there are >= 2 distinct labels;
          - the metric, if supplied, is on the right axis.
        theta is the caller's verification standard; the harness does not invent it.
        """
        if kind != "classification":
            raise ValueError(f"VisionHarness handles 'classification', not {kind!r}")
        batch = _to_grayscale_batch(X)
        labels = [str(v) for v in np.asarray(y).ravel()]
        if len(labels) != batch.shape[0]:
            raise ValueError(f"images ({batch.shape[0]}) and labels ({len(labels)}) mismatch")
        if len(set(labels)) < 2:
            raise ValueError("image classification needs >= 2 distinct labels")
        if metric:
            if metric not in _VALID_METRICS:
                raise ValueError(f"metric {metric!r} not valid for classification; "
                                 f"choose one of {_VALID_METRICS}")
        else:
            metric = self.metric_for(kind)

        Xmat = self._featurize(batch)
        if Xmat.shape[1] == 0:
            raise ValueError("featurizer produced 0 features (degenerate image batch)")
        return Task(X=Xmat, y=np.asarray(labels), kind="classification",
                    theta=float(theta), metric=metric, name=name)

    def baseline_suite(self, kind: str) -> List[Program]:
        """Vision floor recipes (SVC-rbf / logreg / kNN), as seed Programs (never promoters)."""
        if kind != "classification":
            raise ValueError(f"VisionHarness handles 'classification', not {kind!r}")
        return vision_baseline_programs()

    def split_protocol(self, kind: str) -> Tuple[float, float]:
        """Vision uses the Phase-0 default 30/20 split (train 50% / val 20% / sealed 30%)."""
        return (0.30, 0.20)

    # ------------------------------------------------------------------ self-test case
    def _self_test_case(self) -> Tuple[np.ndarray, np.ndarray, str, str, float]:
        """Known-good case: sklearn `load_digits` as 8x8 grayscale images, NO network.

        load_digits is 1797 8x8 images of handwritten digits, 10 classes. A featurized SVC/kNN
        baseline reaches ~0.97+ accuracy; the sealed LOWER bound at this n comfortably clears 0.85.
        We pick 0.85 (well below the achievable score) on purpose: the self-test verifies the
        adapter/featurizer/metric/split PLUMBING with margin to spare, not the modeling difficulty.
        The X returned is the (n,8,8) image tensor -- adapt() featurizes it exactly as a real task.
        """
        from sklearn.datasets import load_digits
        d = load_digits()
        images = d.images.astype(float)                     # (1797, 8, 8)
        return images, d.target.astype(str), "classification", "sklearn_digits_8x8_images", 0.85


# --------------------------------------------------------------------------- registration

# Register a default instance into the shared HarnessRegistry under the image task-type keys, so a
# router resolving modality "image" (and the human alias "vision") gets this harness. A caller
# wanting different featurizer knobs constructs its own VisionHarness and registers a fresh key.
_VISION = VisionHarness(pca_components=40, grid=2, use_gradients=True)
try:
    register(_VISION, "image", "vision")
except KeyError:
    # Idempotent under re-import (e.g. pytest reimport): keep the already-registered instance.
    pass
