"""WILDS Camelyon17 DISTRIBUTION-SHIFT arena for the regenerative autoresearcher (Phase 0, CPU).

This is the generality target the user locked: REAL distribution shift, not cherry-picked binary pairs.
Camelyon17 (WILDS) is binary tumor classification on histopathology patches where train and test come from
DIFFERENT HOSPITALS (centers) -- the model must generalize across an acquisition-domain shift:

    train      = centers {0, 3, 4}     (in-distribution hospitals)
    OOD val     = center 1              (model/head selection only -- select-then-bound)
    OOD test    = center 2              (the SEALED rows; split into K disjoint shards -> the FDR suite)
    OOD gold    = center 2 (disjoint)   (never-peeked confirmation; read once at the end)

PHASE-0 (CPU) CONTRACT -- honest about what runs without a GPU
--------------------------------------------------------------
On CPU we cannot fine-tune, so this arena measures the FROZEN-FEATURE probe of a recipe's `backbone`:
features are extracted once per backbone (timm, cached to disk), then the recipe's `head` (gbm|linear) and
`aggregation` (single|logit_ensemble) are honoured cheaply on top. The `adaptation`/`optimizer`/`epochs`
axes are RECORDED faithfully in the recipe but are realised as a frozen probe here; the GPU arena
(scripts/run_gpu_ceiling.py, Phase 1) honours the full recipe. This is deliberate: Phase 0 proves the
MACHINERY produces a frozen distribution-shift certificate on real data; Phase 1 chases the absolute number.
The decisive lever -- the backbone, drawn from the OPEN zoo -- is fully exercised here.

The select-then-bound discipline, the frozen Clopper-Pearson bound, paired McNemar + BH-FDR, the
meta-certifier framing probe (which here is a genuine train/test CONTAMINATION check across centers) and the
never-peeked gold read are IDENTICAL to the prior arenas; only the data + the OPEN proposer changed.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from vfplatform.recipe import Recipe
from vfplatform.recipe_research import RecipeArena
from vfplatform.repr_researcher import TaskMeasure

DATA_ROOT = os.environ.get("WILDS_DATA_ROOT", "/home/ubuntu/wilds_data")
CACHE_DIR = os.environ.get("WILDS_FEATURE_CACHE", "/home/ubuntu/wilds_data/_features")
TRAIN_CENTERS = (0, 3, 4)
VAL_CENTER = 1
TEST_CENTER = 2


def _seeded_indices(mask: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Pick up to n indices where mask is True, deterministically."""
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    if len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    return np.sort(idx)


def _balanced(idx: np.ndarray, y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Down-sample `idx` to ~n rows with a 50/50 class balance (so the trivial baseline cannot clear theta)."""
    rng = np.random.default_rng(seed)
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    k = min(n // 2, len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
    return np.sort(sel)


class Camelyon17Splits:
    """Loads the Camelyon17 metadata and carves deterministic, class-balanced subsets for the arena. The
    image tensors are loaded lazily only when a backbone needs featurizing (and then cached to disk)."""

    def __init__(self, *, n_train: int = 1200, n_val: int = 600, n_test: int = 900, n_gold: int = 600,
                 n_shards: int = 3, seed: int = 0, data_root: str = DATA_ROOT):
        from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
        self.ds = Camelyon17Dataset(root_dir=data_root, download=False)
        meta = self.ds.metadata_array.numpy()
        # metadata columns: [hospital(center), slide, y]
        self.center = meta[:, 0].astype(int)
        self.y = self.ds.y_array.numpy().astype(int)
        self.n_shards = int(n_shards)

        train_mask = np.isin(self.center, TRAIN_CENTERS)
        val_mask = self.center == VAL_CENTER
        test_mask = self.center == TEST_CENTER

        self.train_idx = _balanced(np.flatnonzero(train_mask), self.y, n_train, seed + 1)
        self.val_idx = _balanced(np.flatnonzero(val_mask), self.y, n_val, seed + 2)
        test_all = _balanced(np.flatnonzero(test_mask), self.y, n_test + n_gold, seed + 3)
        rng = np.random.default_rng(seed + 4)
        rng.shuffle(test_all)
        self.gold_idx = np.sort(test_all[:n_gold])
        sealed_pool = np.sort(test_all[n_gold:])
        # split the sealed OOD rows into K disjoint, balanced shards -> the BH-FDR suite
        self.shards: List[np.ndarray] = []
        per = len(sealed_pool) // self.n_shards
        for s in range(self.n_shards):
            chunk = sealed_pool[s * per:(s + 1) * per] if s < self.n_shards - 1 else sealed_pool[s * per:]
            self.shards.append(np.sort(chunk))
        self.tasks = [f"ood_center{TEST_CENTER}_shard{i}" for i in range(self.n_shards)]

    def all_used_idx(self) -> np.ndarray:
        return np.unique(np.concatenate([self.train_idx, self.val_idx, self.gold_idx] + self.shards))


def _model_input_size(backbone: str) -> int:
    b = backbone.lower()
    if "dinov2" in b or "patch14" in b:
        return 98          # dinov2 needs a multiple of 14; 98 = 14*7 keeps 96px patches cheap on CPU
    return 96 if "resnet" in b else 224


def extract_features(backbone: str, splits: Camelyon17Splits, idx: np.ndarray, *,
                     cache_dir: str = CACHE_DIR, batch: int = 64) -> np.ndarray:
    """Frozen features for `idx` from `backbone` (timm), cached to disk keyed by (backbone, idx-hash).
    CPU-only; never fine-tunes. Returns (len(idx), D) float32."""
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha256((backbone + "::" + hashlib.sha256(idx.tobytes()).hexdigest()).encode()).hexdigest()[:24]
    path = os.path.join(cache_dir, f"{key}.npy")
    if os.path.exists(path):
        return np.load(path)

    import timm
    import torch
    from PIL import Image
    import torchvision.transforms as T

    size = _model_input_size(backbone)
    try:
        model = timm.create_model(backbone, pretrained=True, num_classes=0, dynamic_img_size=True)
    except TypeError:
        model = timm.create_model(backbone, pretrained=True, num_classes=0)
    model.eval()
    cfg = model.pretrained_cfg if hasattr(model, "pretrained_cfg") else {}
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])

    feats: List[np.ndarray] = []
    with torch.no_grad():
        buf: List[torch.Tensor] = []

        def flush():
            if not buf:
                return
            x = torch.stack(buf)
            feats.append(model(x).cpu().numpy().astype(np.float32))
            buf.clear()

        for i in idx:
            img = splits.ds.get_input(int(i)).convert("RGB") if hasattr(splits.ds, "get_input") \
                else Image.open(os.path.join(splits.ds.data_dir, splits.ds._input_array[int(i)])).convert("RGB")
            buf.append(tf(img))
            if len(buf) >= batch:
                flush()
        flush()
    out = np.concatenate(feats, axis=0).astype(np.float32)
    np.save(path, out)
    return out


def _fit_head(recipe: Recipe, Xtr, ytr, Xeval):
    """Fit the recipe's head on TRAIN features, predict EVAL features. Honours head (gbm|linear) and
    aggregation (logit_ensemble = bag of seeds). Sealed-blind: only features cross."""
    def one(seed: int):
        if recipe.head == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)
        else:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xtr, ytr)
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(Xeval)[:, 1]
        return clf.decision_function(Xeval)

    n_members = 3 if recipe.aggregation in ("logit_ensemble", "model_soup") else 1
    probs = np.mean([one(s) for s in range(n_members)], axis=0)
    return (probs >= 0.5).astype(int)


class WildsCamelyonArena(RecipeArena):
    """RecipeArena over Camelyon17 OOD shards. measure() featurizes the recipe's backbone (cached), fits the
    head on train, selects nothing extra (val is used by the framing probe + as the cheap screen stream),
    and scores the OOD sealed shards. Distribution shift is intrinsic: train centers != test center."""

    task_hint = "histopathology"
    task_shape = "binary"

    def __init__(self, splits: Optional[Camelyon17Splits] = None, *, competence_floor: float = 0.5):
        self.splits = splits if splits is not None else Camelyon17Splits()
        self.tasks = list(self.splits.tasks)
        self._floor = competence_floor

    def seed_recipes(self) -> List[Recipe]:
        # a deliberately weak champion: a small ImageNet CNN, frozen linear probe
        return [Recipe(backbone="resnet18", adaptation="linear_probe", head="linear")]

    def _feats(self, backbone: str, idx: np.ndarray) -> np.ndarray:
        return extract_features(backbone, self.splits, idx)

    def measure(self, recipe: Recipe) -> Dict[str, TaskMeasure]:
        sp = self.splits
        Xtr = self._feats(recipe.backbone, sp.train_idx)
        ytr = sp.y[sp.train_idx]
        Xval = self._feats(recipe.backbone, sp.val_idx)
        yval = sp.y[sp.val_idx]
        val_pred = _fit_head(recipe, Xtr, ytr, Xval)
        val_correct = (val_pred == yval).astype(int).tolist()

        out: Dict[str, TaskMeasure] = {}
        for t, shard in zip(sp.tasks, sp.shards):
            Xte = self._feats(recipe.backbone, shard)
            yte = sp.y[shard]
            pred = _fit_head(recipe, Xtr, ytr, Xte)
            correct = (pred == yte).astype(int).tolist()
            out[t] = TaskMeasure(sealed_correct=correct, val_correct=val_correct,
                                 acc=float(np.mean(correct)))
        return out

    def gold_measure(self, recipe: Recipe) -> Dict[str, Sequence[int]]:
        sp = self.splits
        Xtr = self._feats(recipe.backbone, sp.train_idx)
        ytr = sp.y[sp.train_idx]
        Xg = self._feats(recipe.backbone, sp.gold_idx)
        yg = sp.y[sp.gold_idx]
        pred = _fit_head(recipe, Xtr, ytr, Xg)
        correct = (pred == yg).astype(int).tolist()
        # one gold vector per task (the gold set is shared; the suite-level read pools it)
        return {t: correct for t in sp.tasks}

    def framing(self) -> dict:
        """The meta-certifier framing probe = a genuine cross-center CONTAMINATION + gameability check on the
        weak baseline's features: trivial-baseline (balanced -> ~0.5 < theta), label-shuffle, and near-dup
        STRADDLE between train (centers 0/3/4) and sealed (center 2). A leak here would REFUSE the run."""
        sp = self.splits
        base = self.seed_recipes()[0].backbone
        sealed_concat = np.concatenate(sp.shards)
        idx = np.concatenate([sp.train_idx, sealed_concat])
        Xtr = self._feats(base, sp.train_idx)
        Xse = self._feats(base, sealed_concat)
        X = np.concatenate([Xtr, Xse], axis=0)
        y = np.concatenate([sp.y[sp.train_idx], sp.y[sealed_concat]])
        n_tr = len(sp.train_idx)
        train_idx = np.arange(n_tr)
        sealed_idx = np.arange(n_tr, n_tr + len(sealed_concat))

        def fit_predict_fn(Xa, ya, tr, se):
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=500).fit(Xa[tr], ya[tr])
            return clf.predict(Xa[se])

        def metric_fn(yt, yp):
            return float((np.asarray(yt) == np.asarray(yp)).mean())

        return {"X": X, "y": y, "train_idx": train_idx, "sealed_idx": sealed_idx,
                "fit_predict_fn": fit_predict_fn, "metric_fn": metric_fn}


__all__ = ["Camelyon17Splits", "WildsCamelyonArena", "extract_features",
           "TRAIN_CENTERS", "VAL_CENTER", "TEST_CENTER"]
