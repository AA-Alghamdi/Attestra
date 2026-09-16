"""The FIRST new modality: image classification, reusing the FROZEN classification path (NEW module).

Generalization without weakening rigor: vision adds a new FEATURIZER (images -> a fixed numeric matrix) and
a new Harness subclass, but it REUSES everything that touches correctness:
  * the existing classification model zoo  -> catalog_for("tabular", task_type)  (harness.py),
  * the base Harness target encoding        -> Harness.encode_targets,
  * the EXISTING frozen certifier            -> run_goal_loop -> certify_on_sealed (sealed.py / science.py).
There is NO new promotion path, NO new certificate, NO new sealed-peek logic. An image goal is certified by
exactly the same select-on-validation -> one-counted-sealed-peek machinery as a tabular classification goal;
vision only changes how pixels become a feature vector.

Image records follow the platform's nested shape (like connectors.load_sklearn("digits")):
    {"features": {pixel_col: value, ...}, "target": <label>}      # a flat pixel dict (digits: 64 keys)
The featurizer also accepts features that are a flat pixel LIST or a nested 2D/3D array; all are normalized
to [0,1], flattened to a fixed-length vector, and (when the per-image pixel count is a perfect square) given
a light translation-robust 2x2 average-pool block appended as extra features. Pure numpy -- no torch, no
scikit-image; it runs on a slim CPU container.
"""
import numpy as np

from .harness import Harness, catalog_for, move_from_proposal, _grid_configs, Move


def _as_pixel_vector(feats):
    """Coerce one record's `features` into a flat 1-D float pixel vector, supporting:
       * a flat dict {pixel_col: value, ...} (the digits/MNIST flattened-image convention) -- ORDER by the
         sorted column key so the vector is stable across rows (the featurizer pins the column order anyway),
       * a flat list/tuple of pixel values,
       * a nested 2-D (HxW) or 3-D (HxWxC) array (list-of-lists or ndarray).
    Returns a 1-D float ndarray. Non-numeric / missing entries become 0.0."""
    if isinstance(feats, dict):
        # ordered by sorted key for determinism; the Featurizer below pins this exact order at fit time.
        vals = [feats[k] for k in sorted(feats.keys())]
    else:
        vals = feats
    arr = np.asarray(vals, dtype=object)
    flat = arr.reshape(-1)
    out = np.zeros(flat.shape[0], dtype=float)
    for i, v in enumerate(flat):
        try:
            fv = float(v)
            out[i] = fv if np.isfinite(fv) else 0.0
        except (TypeError, ValueError):
            out[i] = 0.0
    return out


def _avg_pool_2x2(vec):
    """Light translation-robust feature: if the pixel vector is a perfect square (HxH image), 2x2
    average-pool it (stride 2) and return the pooled values flattened. Otherwise return an empty array
    (no pooling feature for non-square images). Pure numpy; no scikit-image."""
    n = vec.shape[0]
    side = int(round(n ** 0.5))
    if side * side != n or side < 2:
        return np.zeros(0, dtype=float)
    img = vec.reshape(side, side)
    h = side - (side % 2)                       # crop to an even side so 2x2 blocks tile exactly
    if h < 2:
        return np.zeros(0, dtype=float)
    img = img[:h, :h]
    pooled = img.reshape(h // 2, 2, h // 2, 2).mean(axis=(1, 3))
    return pooled.reshape(-1)


class ImageFeaturizer:
    """Mirror of TabularFeaturizer's interface EXACTLY so the loop is unchanged:
        .fit(rows_for_schema, all_rows) -> self     # pins the pixel length + normalization scale
        .transform(rows) -> np.ndarray (n_rows x n_features, dtype float)
    Normalizes pixels to [0,1] by dividing by the max absolute pixel value seen at fit time (a single global
    scale, so a constant offset/scale of the input does not change the relative geometry), flattens to the
    pinned length, and appends a 2x2 average-pool block for translation robustness when the image is square.
    Rows whose pixel vector is shorter/longer than the pinned length are padded/truncated (so a stray
    malformed row never changes the matrix width and breaks the fit)."""

    def __init__(self):
        self.length = None          # pinned flat pixel length
        self.scale = 1.0            # global normalization divisor (max abs pixel at fit)
        self.dict_keys = None       # if features are dicts, the pinned sorted key order
        self.pool_len = 0           # width of the appended pooling block (0 if non-square)

    @staticmethod
    def _raw_feats(r):
        f = r.get("features")
        return f if f is not None else {k: v for k, v in r.items() if k != "target"}

    def fit(self, rows_for_schema, all_rows):
        # pin the pixel length + (for dict features) the key order from the FIRST schema row.
        first = self._raw_feats(rows_for_schema[0])
        if isinstance(first, dict):
            self.dict_keys = sorted(first.keys())
        vec0 = _as_pixel_vector(first)
        self.length = int(vec0.shape[0])
        # global normalization scale from the FULL split union (already content-addressed + audited
        # upstream, so this adds no new leakage path beyond the featurizer vocab, like TabularFeaturizer).
        mx = 0.0
        for r in all_rows:
            v = _as_pixel_vector(self._raw_feats(r))
            if v.shape[0]:
                m = float(np.max(np.abs(v)))
                if m > mx:
                    mx = m
        self.scale = mx if mx > 1e-9 else 1.0
        self.pool_len = int(_avg_pool_2x2(np.zeros(self.length, dtype=float)).shape[0])
        return self

    def _vector_for(self, r):
        feats = self._raw_feats(r)
        # honor the pinned dict-key order so a dict whose keys arrive in a different order is read the same.
        if isinstance(feats, dict) and self.dict_keys is not None:
            vec = _as_pixel_vector({k: feats.get(k, 0.0) for k in self.dict_keys})
        else:
            vec = _as_pixel_vector(feats)
        # pad/truncate to the pinned length so the matrix width is constant.
        if vec.shape[0] < self.length:
            vec = np.concatenate([vec, np.zeros(self.length - vec.shape[0], dtype=float)])
        elif vec.shape[0] > self.length:
            vec = vec[:self.length]
        vec = vec / self.scale                                   # normalize to [0,1] (or [-1,1] if signed)
        pool = _avg_pool_2x2(vec)                                # pooled from the ALREADY-normalized pixels
        if pool.shape[0] != self.pool_len:                       # keep the block width fixed
            fixed = np.zeros(self.pool_len, dtype=float)
            fixed[:min(self.pool_len, pool.shape[0])] = pool[:self.pool_len]
            pool = fixed
        return np.concatenate([vec, pool])

    def transform(self, rows):
        if self.length is None:
            raise RuntimeError("ImageFeaturizer.transform called before fit")
        return np.asarray([self._vector_for(r) for r in rows], dtype=float)


class VisionClassificationHarness(Harness):
    """Image classification harness. modality="vision"; task_type in {binary, multiclass}; metric "accuracy".
    Reuses the EXISTING classification catalog (catalog_for("tabular", task_type)) and the base Harness
    target encoding + sealed certifier. No new certificate, no new promotion path.

    On `kind`: the FROZEN pipeline identity this harness presents to run_goal_loop is "tabular" -- and that is
    not a coercion, it is the truth. The ImageFeaturizer turns each image into a fixed numeric feature matrix;
    to the frozen leakage auditor (vectorforge.science.audit) and the frozen certifier, certified pixels are
    exactly the legitimate numeric features that connectors.load_sklearn("digits") already certifies today.
    The frozen auditor's policy is "non-tabular kinds carry no raw feature columns" (it gates feature columns
    behind allow_features, which the loop derives as `harness.kind == "tabular"`). Image pixels DO need to flow
    as feature columns through that exact frozen path, so the harness's pipeline `kind` is "tabular" and the
    image-specific identity lives in `modality="vision"`. The catalog still draws from catalog_for("tabular",
    task_type) (the existing classification zoo) -- there is no vision-specific estimator and no new cert path.
    `vision_demo()` advertises the connector-level kind "vision" (modality metadata, never the audit lever)."""
    kind = "tabular"            # FROZEN-pipeline identity (audit + catalog routing); see the class docstring
    modality = "vision"         # image-classification modality identity (public-facing; not the audit lever)
    default_metric = "accuracy"

    def __init__(self, task_type="multiclass"):
        if task_type not in ("binary", "multiclass"):
            raise ValueError(f"VisionClassificationHarness supports binary|multiclass, got {task_type!r}")
        self.task_type = task_type

    def featurizer(self):
        return ImageFeaturizer()

    def catalog(self):
        # REUSE the existing classification model zoo (one place; harness.py). Vision is featurized to a
        # numeric matrix, so the tabular CLASSIFICATION catalog applies unchanged.
        return catalog_for("tabular", self.task_type)

    def moves(self, seeds, provider=None):
        """A diagnosis-gated move menu drawn from the SHARED classification catalog (same family names the
        diagnose->propose loop expects: baseline/stronger_model/more_capacity/regularize). Built from the
        catalog grid so every family/param is the frozen, clamped one -- no vision-specific estimator."""
        cat = self.catalog()

        def _move(move_name, family, prior_gain, prior_cost, pick=0):
            entry = cat.get(family)
            if entry is None:
                return None
            fam, params = _grid_configs(entry)[min(pick, len(_grid_configs(entry)) - 1)]
            m = move_from_proposal(cat, fam, params)
            if m is None:
                return None
            return Move(move_name, m.families, prior_gain=prior_gain, prior_cost=prior_cost)

        candidates = [
            _move("baseline", "logistic", 0.0, 0.3, pick=3),
            _move("stronger_model", "hist_gbm", 0.10, 1.6),
            _move("more_capacity", "random_forest", 0.05, 2.0, pick=1),
            _move("regularize", "logistic", 0.02, 0.5, pick=0),
        ]
        return [m for m in candidates if m is not None]


def vision_demo():
    """A runnable vision example OUT OF THE BOX: sklearn load_digits as 8x8 image records (features = the 64
    pixel values 0..16, target = the digit 0..9). Same record shape as connectors.load_sklearn so it flows
    straight through run_goal_loop with the VisionClassificationHarness. No download, CPU-only."""
    from sklearn.datasets import load_digits
    d = load_digits()
    rows = [{"features": {f"p{j}": float(x[j]) for j in range(d.data.shape[1])}, "target": str(int(y))}
            for x, y in zip(d.data, d.target)]
    labels = sorted({r["target"] for r in rows})
    return {"records": rows, "kind": "vision", "task_type": "multiclass", "target_key": "target",
            "labels": labels, "metric": "accuracy", "source": "sklearn:digits(vision)", "n": len(rows)}


# --------------------------------------------------------------------------- self-test
def _selftest():
    p = f = 0

    def check(name, cond):
        nonlocal p, f
        if cond:
            print(f"  PASS  {name}"); p += 1
        else:
            print(f"  FAIL  {name}"); f += 1

    demo = vision_demo()
    rows = demo["records"]
    check("vision_demo shape", demo["kind"] == "vision" and demo["task_type"] == "multiclass"
          and demo["metric"] == "accuracy")

    h = VisionClassificationHarness(task_type="multiclass")
    feat = h.featurizer().fit(rows[:50], rows)
    X = feat.transform(rows[:10])
    # 64 pixels + 16 pooled (8x8 -> 4x4) = 80 features
    check("featurizer fit/transform shape", X.shape == (10, 80))
    check("pixels normalized to [0,1]", float(X[:, :64].max()) <= 1.0 + 1e-9 and float(X.min()) >= -1e-9)
    # determinism + ctor reuse from the shared catalog
    X2 = feat.transform(rows[:10])
    check("featurizer deterministic", np.allclose(X, X2))
    check("reuses classification catalog", set(h.catalog().keys()) == set(catalog_for("tabular", "multiclass").keys()))

    # nested 2D image features also work
    img2d = [{"features": [[float((i + a + b) % 16) for b in range(8)] for a in range(8)], "target": str(i % 2)}
             for i in range(10)]
    f2 = ImageFeaturizer().fit(img2d, img2d)
    X3 = f2.transform(img2d)
    check("nested 2D image featurizes", X3.shape == (10, 80))

    print(f"  ---- {p} passed, {f} failed ----")
    return f == 0


if __name__ == "__main__":
    import sys
    print("== vision self-test ==")
    sys.exit(0 if _selftest() else 1)
