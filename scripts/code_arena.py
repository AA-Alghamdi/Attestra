"""RAW-FEATURE CODE-DISCOVERY arena -- the decisive test of OPEN-ENDED AUTHORED CODE under sealed certification.

WHY THIS ARENA EXISTS (two gaps at once)
----------------------------------------
1. OPEN-ENDED CODE (the anti-menu capstone). Every other arena's lever is the BACKBONE (drawn from the open
   zoo). Here the representation is deliberately FIXED to RAW tabular features, so the ONLY thing that can move
   the metric is CODE the system AUTHORS itself -- a novel featurizer or estimator emitted by the LLM/template
   authorer, admitted through the frozen three-stage sandbox, EXECUTED here, and judged by the frozen
   certifier. A certified champion that carries a `code_patch` is menu-free at the CODE axis: the winning
   source was never handed to the system. This is the honest counterpart to result B2 (authoring on a fixed
   PRETRAINED representation added nothing): on RAW features there IS headroom for a better model, and the
   question is whether the system can DISCOVER + CERTIFY it without being told the answer.

2. TASK-SHAPE GENERALITY (kills "binary-pair-only"). The shapes here are real and non-binary:
     multiclass   -- covtype (7 forest cover types) / digits (10 classes), class-balanced
     imbalanced   -- covtype with the NATURAL long-tailed class proportions (trivial baseline is high)
     noisy_label  -- multiclass with a fraction of TRAIN labels flipped (sealed/gold stay clean)

DISCIPLINE (identical to every other arena; only the data + the CODE lever changed)
-----------------------------------------------------------------------------------
Select-then-bound: features are computed once; train fits the recipe; the OOD-style sealed rows are carved
ONCE into K disjoint shards (the BH-FDR suite) plus a disjoint never-peeked GOLD set read exactly once at the
end. A code-bearing recipe is admitted through the sandbox BEFORE it can spend a peek and then EXECUTED on the
identical sealed rows (sealed-blind: only features cross to predict()). The frozen Clopper-Pearson bound,
paired McNemar + BH-FDR, the meta-certifier framing/contamination probe, and the gold read are unchanged.

SECURITY NOTE: admission runs in a spawned, rlimited child; the admitted method's fit/predict then runs
in-process here (the repo's documented local-only path for authored estimators, worker_safe=False). The
deterministic TemplateAuthorer is trusted source; the LLM authorer's output has already cleared the static
AST gate (no os/sys/file/network/numpy-IO/dunder) before it reaches this in-process execution.
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

from vfplatform import authoring_bridge as AB
from vfplatform.recipe import Recipe
from vfplatform.recipe_research import RecipeArena
from vfplatform.repr_researcher import TaskMeasure

warnings.filterwarnings("ignore")


# ============================================================================ real datasets (offline-first)
REGRESSION_DATASETS = ("diabetes", "california")

# OpenML-CC18 (study 99) members -- a STANDARD curated benchmark suite, NOT cherry-picked tasks. Each entry is
# (openml_data_id, task_hint). These are fetched from openml.org ONCE, encoded (median-impute numeric +
# one-hot categorical), then cached to disk so every later run is fully offline + deterministic. The suite
# deliberately spans the task-shape matrix: 3..10-class multiclass (vehicle/segment/splice/mfeat/optdigits),
# near-linearly-separable digits (optdigits -> expected honest negative), categorical features (splice/
# credit_g), and IMBALANCED binary (pima/credit_g, handled by a data-driven competence floor in the runner).
OPENML_CC18 = {
    "vehicle":       (54,    "vehicle_silhouette"),   # 4-class, 18 numeric  -- clear nonlinear headroom
    "segment":       (40984, "image_segmentation"),   # 7-class, 18 numeric
    "mfeat_fourier": (14,    "fourier_digit"),         # 10-class, 76 numeric
    "optdigits":     (28,    "optical_digit"),         # 10-class, 64 numeric -- near-linear (honest negative)
    "splice":        (46,    "dna_splice"),            # 3-class, 60 CATEGORICAL (DNA bases)
    "pima":          (37,    "pima_diabetes"),         # binary, imbalanced (maj~0.65)
    "credit_g":      (31,    "german_credit"),         # binary, imbalanced (maj~0.70), mixed cat+numeric
}


def _load_openml(data_id: int, task_hint: str):
    """Fetch an OpenML dataset, encode it to a dense float matrix, and cache to disk. Numeric columns are
    median-imputed; categorical columns are most-frequent-imputed then one-hot encoded (capped cardinality);
    the target is label-encoded to 0..K-1. Returns (X, y, n_classes, task_hint). Cached as .npz so repeat
    runs (and the whole suite re-run) are fully offline + byte-stable."""
    cache = os.path.expanduser(f"~/wilds_data/_openml_cache/{int(data_id)}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return z["X"].astype(np.float64), z["y"].astype(int), int(z["n_classes"]), task_hint
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    d = fetch_openml(data_id=int(data_id), as_frame=True, parser="auto")
    Xdf = d.data
    y = LabelEncoder().fit_transform(d.target.astype(str)).astype(int)
    cat_cols = [c for c in Xdf.columns if str(Xdf[c].dtype) in ("category", "object")]
    num_cols = [c for c in Xdf.columns if c not in cat_cols]
    blocks = []
    if num_cols:
        num = Xdf[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        blocks.append(SimpleImputer(strategy="median").fit_transform(num))
    if cat_cols:
        raw = SimpleImputer(strategy="most_frequent").fit_transform(Xdf[cat_cols].astype("object"))
        oh = OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False)
        blocks.append(oh.fit_transform(raw).astype(np.float64))
    X = np.concatenate(blocks, axis=1).astype(np.float64)
    # DEDUPLICATE exact-duplicate rows (several CC18 sets -- e.g. segment -- contain many repeated feature
    # vectors). Left in, an identical row can land in BOTH train and sealed, which the meta-certifier's
    # near-duplicate straddle probe (correctly) treats as split leakage and REFUSES the whole run. Dropping
    # all-but-one representative of each identical row is the honest hygiene fix: it makes the split provably
    # leak-clean without touching the certifier. Order-preserving (keep first occurrence) for determinism.
    _, keep = np.unique(np.round(X, 9), axis=0, return_index=True)
    keep = np.sort(keep)
    X, y = X[keep], y[keep]
    n_classes = int(len(np.unique(y)))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, X=X, y=y, n_classes=np.int64(n_classes))
    return X, y, n_classes, task_hint


def _load_dataset(name: str):
    """Return (X, y, n_classes, task_hint) for a REAL dataset. n_classes==0 marks a REGRESSION target (y is
    continuous). covtype / california download once then cache; digits / breast_cancer / wine / diabetes ship
    with sklearn (fully offline); OpenML-CC18 members (OPENML_CC18) fetch + cache once."""
    if name in OPENML_CC18:
        return _load_openml(*OPENML_CC18[name])
    from sklearn import datasets as D
    if name == "covtype":
        d = D.fetch_covtype()
        return d.data.astype(np.float64), d.target.astype(int) - 1, 7, "forest_cartographic"
    if name == "digits":
        d = D.load_digits()
        return d.data.astype(np.float64), d.target.astype(int), 10, "pixel_tabular"
    if name == "breast_cancer":
        d = D.load_breast_cancer()
        return d.data.astype(np.float64), d.target.astype(int), 2, "clinical_tabular"
    if name == "wine":
        d = D.load_wine()
        return d.data.astype(np.float64), d.target.astype(int), 3, "spectro_tabular"
    if name == "diabetes":
        d = D.load_diabetes()
        return d.data.astype(np.float64), d.target.astype(np.float64), 0, "clinical_progression_reg"
    if name == "california":
        d = D.fetch_california_housing()
        return d.data.astype(np.float64), d.target.astype(np.float64), 0, "housing_value_reg"
    raise ValueError(f"unknown dataset {name!r}")


def _balanced_take(y: np.ndarray, idx_pool: np.ndarray, per_class: int, rng) -> np.ndarray:
    out = []
    for c in np.unique(y[idx_pool]):
        cc = idx_pool[y[idx_pool] == c]
        k = min(per_class, len(cc))
        out.append(rng.choice(cc, size=k, replace=False))
    return np.sort(np.concatenate(out))


def _stratified_take(y: np.ndarray, idx_pool: np.ndarray, n: int, rng) -> np.ndarray:
    """Take ~n rows preserving the NATURAL class proportions (for the imbalanced shape)."""
    n = min(n, len(idx_pool))
    sel = rng.choice(idx_pool, size=n, replace=False)
    return np.sort(sel)


def _derive_groups(dataset: str, X: np.ndarray) -> np.ndarray:
    """A REAL, domain-meaningful GROUP id per row, so a grouped split holds out WHOLE groups (a covariate-
    shift / extrapolation test, not a random row split). california -> a coarse spatial grid on (Latitude,
    Longitude): held-out cells are unseen REGIONS (the canonical spatial-CV protocol for housing value).
    covtype -> Elevation bands: held-out bands are unseen elevation regimes. Any other dataset must pass
    `groups=` explicitly (fabricating a group axis where none exists would be dishonest)."""
    if dataset == "california":                          # fetch_california_housing feature order: [..,6]=Lat [7]=Lon
        lat, lon = X[:, 6], X[:, 7]
        lat_b = np.digitize(lat, np.quantile(lat, np.linspace(0, 1, 6)[1:-1]))
        lon_b = np.digitize(lon, np.quantile(lon, np.linspace(0, 1, 6)[1:-1]))
        return (lat_b * 10 + lon_b).astype(int)
    if dataset == "covtype":
        elev = X[:, 0]                                    # Elevation (meters)
        return np.digitize(elev, np.quantile(elev, np.linspace(0, 1, 13)[1:-1])).astype(int)
    raise ValueError(f"no built-in group structure for dataset {dataset!r}; pass groups= explicitly")


class TabularSplits:
    """Carve a real dataset into train / val / K sealed shards / gold with a fixed seed. `shape` controls the
    sampling: 'multiclass' (class-balanced), 'imbalanced' (natural proportions), 'noisy_label' (balanced, with
    a fraction of TRAIN labels flipped -- sealed/gold stay clean), 'regression' (continuous target).

    `split` controls HOW the pool is partitioned -- this is the GENERALITY-of-task-shape axis:
      'random'  -- i.i.d. row split (the default; what every prior arena used).
      'grouped' -- whole GROUPS are held out (no group straddles train/val/sealed/gold). The sealed + gold
                   rows belong to groups the model NEVER trained on -> a real covariate-shift / extrapolation
                   test (e.g. california spatial regions, covtype elevation bands), not an i.i.d. read.
      'time'    -- forward-chaining: train is the EARLIEST block, the sealed shards + gold are strictly LATER
                   in time -> a train-on-past / certify-on-future test (no future row leaks into training).
    The group/time partitions reuse the audited leak-safe logic in `vfplatform.splits` (assert_no_leakage),
    and `framing()` re-exposes the groups/times so the meta-certifier's split-leak probe REFUSES a leaky split
    before any sealed peek is spent."""

    def __init__(self, dataset: str = "covtype", shape: str = "multiclass", *, per_class: int = 1200,
                 n_total: int = 8400, n_shards: int = 3, noise: float = 0.2, tau_frac: float = 0.5,
                 seed: int = 0, data: Optional[tuple] = None, split: str = "random",
                 groups: Optional[np.ndarray] = None, times: Optional[np.ndarray] = None):
        self.is_regression = (shape == "regression")
        if data is not None:                       # injected (X, y, n_classes, task_hint) -- used by tests
            self.X, self.y, self.n_classes, self.task_hint = data
        else:
            self.X, self.y, self.n_classes, self.task_hint = _load_dataset(dataset)
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=(np.float64 if self.is_regression else int))
        self.dataset = dataset
        self.shape = shape
        self.split = split
        self.n_shards = int(n_shards)
        rng = np.random.default_rng(seed)
        all_idx = np.arange(len(self.y))
        rng.shuffle(all_idx)

        if self.is_regression:                     # continuous target: random subsample, no class structure
            pool = np.sort(all_idx[:min(n_total, len(all_idx))])
        elif shape == "imbalanced":
            pool = _stratified_take(self.y, all_idx, n_total, rng)
        else:
            pc = min(per_class, n_total // max(1, self.n_classes))
            pool = _balanced_take(self.y, all_idx, pc, rng)

        # the GROUP / TIME structure, aligned to GLOBAL rows (so framing() can re-expose it for the leak probe)
        self._groups_full: Optional[np.ndarray] = None
        self._times_full: Optional[np.ndarray] = None
        if split == "grouped":
            g = groups if groups is not None else _derive_groups(dataset, self.X)
            self._groups_full = np.asarray(g)
            if len(self._groups_full) != len(self.y):
                raise ValueError("groups must be one id per row")
        elif split == "time":
            if times is None:
                raise ValueError("split='time' needs a `times` array (these datasets carry no honest "
                                 "timestamp; inject a real time index)")
            self._times_full = np.asarray(times, dtype=float)
            if len(self._times_full) != len(self.y):
                raise ValueError("times must be one timestamp per row")

        # carve the pool -> train / val / K sealed shards / gold by the chosen split discipline
        if split == "random":
            self._carve_random(pool, rng)
        elif split == "grouped":
            self._carve_grouped(pool, rng)
        elif split == "time":
            self._carve_time(pool)
        else:
            raise ValueError(f"unknown split {split!r}")
        self.tasks = [f"{dataset}_{shape}_shard{i}" for i in range(self.n_shards)]

        # the TRAIN labels actually used (noisy_label flips a fraction; everything else is clean)
        self.y_train = self.y.copy()
        if shape == "noisy_label":
            k = int(noise * len(self.train_idx))
            flip = rng.choice(self.train_idx, size=k, replace=False)
            for i in flip:
                choices = [c for c in range(self.n_classes) if c != self.y[i]]
                self.y_train[i] = rng.choice(choices)
        self.noise = noise if shape == "noisy_label" else 0.0

        # REGRESSION: a PRE-REGISTERED tolerance turns the continuous target into a per-row Bernoulli
        # ("hit" iff |pred - y| <= tau), so the IDENTICAL frozen primitives (Clopper-Pearson lower bound,
        # paired McNemar, BH-FDR) certify regression BYTE-IDENTICALLY -- no new statistical primitive, no
        # change to science.py/sealed.py. tau is a fixed fraction of the TRAIN target's spread (computed on
        # TRAIN only, never on sealed/gold), so it is not snooped from the evaluation rows.
        self.tau_frac = float(tau_frac)
        self.tau = (self.tau_frac * float(np.std(self.y[self.train_idx]))) if self.is_regression else None

    # -- carve helpers (one per split discipline) -------------------------------------------------------
    def _shard(self, sealed_pool: np.ndarray) -> None:
        per = len(sealed_pool) // self.n_shards
        self.shards: List[np.ndarray] = []
        for s in range(self.n_shards):
            chunk = sealed_pool[s * per:(s + 1) * per] if s < self.n_shards - 1 else sealed_pool[s * per:]
            self.shards.append(np.sort(chunk))

    def _carve_random(self, pool: np.ndarray, rng) -> None:
        """i.i.d. row split (byte-identical to the original behavior)."""
        pool = pool.copy()
        rng.shuffle(pool)
        n = len(pool)
        n_tr, n_val, n_gold = int(0.42 * n), int(0.17 * n), int(0.17 * n)
        self.train_idx = np.sort(pool[:n_tr])
        self.val_idx = np.sort(pool[n_tr:n_tr + n_val])
        self.gold_idx = np.sort(pool[n - n_gold:])
        self._shard(pool[n_tr + n_val:n - n_gold])

    def _carve_grouped(self, pool: np.ndarray, rng) -> None:
        """Assign WHOLE groups to train / val / gold / K sealed shards so no group straddles any boundary.
        Greedy deficit-packing toward the same 0.42 / 0.17 / 0.17 / 0.24 row targets, seeding one group per
        bucket first so none starves. The sealed + gold rows are groups the model never trained on."""
        gp = self._groups_full[pool]
        uniq = list(dict.fromkeys(gp.tolist()))
        rng.shuffle(uniq)
        K = self.n_shards
        nb = 3 + K                                       # 0=train 1=val 2=gold 3..=shards
        if len(uniq) < nb:
            raise ValueError(f"grouped split needs >= {nb} distinct groups in the pool, found {len(uniq)}")
        n = len(pool)
        targets = [0.42 * n, 0.17 * n, 0.17 * n] + [(0.24 / K) * n] * K
        counts = {u: int(np.sum(gp == u)) for u in uniq}
        buckets: List[List] = [[] for _ in range(nb)]
        filled = [0.0] * nb
        for i, u in enumerate(uniq):
            j = i if i < nb else int(np.argmax([targets[k] - filled[k] for k in range(nb)]))
            buckets[j].append(u)
            filled[j] += counts[u]
        sel = lambda us: np.sort(pool[np.isin(gp, us)]) if us else np.array([], dtype=pool.dtype)
        self.train_idx, self.val_idx, self.gold_idx = sel(buckets[0]), sel(buckets[1]), sel(buckets[2])
        self.shards = [sel(buckets[3 + s]) for s in range(K)]

    def _carve_time(self, pool: np.ndarray) -> None:
        """Forward-chaining: order the pool by time, take the EARLIEST block as train, then val, then the K
        sealed shards as strictly-later contiguous blocks, and the FURTHEST-future block as gold. No future
        row can leak into training (the temporal-leak probe verifies max(train_time) <= min(sealed_time))."""
        t = self._times_full[pool]
        order = pool[np.argsort(t, kind="stable")]
        n = len(order)
        n_tr, n_val, n_gold = int(0.42 * n), int(0.17 * n), int(0.17 * n)
        self.train_idx = np.sort(order[:n_tr])
        self.val_idx = np.sort(order[n_tr:n_tr + n_val])
        self.gold_idx = np.sort(order[n - n_gold:])
        self._shard(order[n_tr + n_val:n - n_gold])


class TabularCodeArena(RecipeArena):
    """RecipeArena over a real dataset where the representation is FIXED to raw features, so AUTHORED CODE is
    the lever. measure() executes a recipe's `code_patch` (admitted featurizer/classifier) on the identical
    sealed rows, else falls back to the recipe's head (the weak linear/gbm baseline)."""

    def __init__(self, dataset: str = "covtype", shape: str = "multiclass", *, splits: Optional[TabularSplits] = None,
                 seed: int = 0):
        self.splits = splits if splits is not None else TabularSplits(dataset=dataset, shape=shape, seed=seed)
        self.dataset = self.splits.dataset
        self.shape = self.splits.shape
        self.split = self.splits.split
        self.tasks = list(self.splits.tasks)
        self.task_hint = self.splits.task_hint
        self.is_regression = self.splits.is_regression
        self.task_shape = ("regression" if self.is_regression else
                           ("multiclass" if self.splits.n_classes > 2 else "binary"))
        self.n_classes = self.splits.n_classes
        self.tau = self.splits.tau
        # GPU/neural head gate: when True, a recipe with head='torch_mlp' is fit with the SAME torch model
        # code the GPU worker runs (worker/torch_models.py), device-swapping cuda<->cpu. Default OFF, so the
        # recipe path is byte-identical to the linear/gbm baseline unless a caller explicitly enables it.
        self.allow_torch_head = False
        # the competence floor + certification bar. theta_floor=0.5 means: a model must place >50% of rows
        # within tau (regression) or be right >50% of the time (classification). On regression a CONSTANT
        # (median/mode) predictor lands well below 0.5 at tau=0.5*train-std, so the trivial-baseline probe
        # correctly fails -- the tolerance metric is not gameable by predicting the center.
        self.theta_floor = 0.5

    # -- the FIXED representation (raw features); a backbone axis would be the lever, here it is pinned ----
    def _feats(self, backbone: str) -> np.ndarray:
        return self.splits.X

    def seed_recipes(self) -> List[Recipe]:
        # the deliberately weak champion: a plain LINEAR head on raw features (a mid-level default), no code.
        return [Recipe(backbone="raw", adaptation="linear_probe", head="linear")]

    # -- the per-row Bernoulli outcome (classification: exact match; regression: within tolerance) --------
    def _hit(self, pred: np.ndarray, ytrue: np.ndarray) -> np.ndarray:
        if self.is_regression:
            return (np.abs(np.asarray(pred, dtype=float) - np.asarray(ytrue, dtype=float)) <= self.tau).astype(int)
        return (np.asarray(pred) == np.asarray(ytrue)).astype(int)

    # -- heads (the non-code baseline path) -------------------------------------------------------------
    def _fit_torch_head(self, Xtr, ytr, Xeval):
        """Fit the GPU/neural head with the SAME torch model code the worker runs (single source of truth);
        on a GPU box it trains on cuda, else on CPU. Returns predictions, or None on any failure so the
        caller can fall back to a sklearn head (the loop still gets a measurement)."""
        import os as _os
        import sys as _sys
        wdir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "worker")
        if wdir not in _sys.path:
            _sys.path.insert(0, wdir)
        try:
            from torch_models import TorchMLPClassifier, TorchMLPRegressor  # type: ignore
        except Exception:  # noqa: BLE001  torch not installed -> head unavailable, fall back
            return None
        try:
            if self.is_regression:
                est = TorchMLPRegressor(hidden=(128, 128), epochs=120, lr=1e-3, seed=0)
                est.fit(np.asarray(Xtr, dtype=float), np.asarray(ytr, dtype=float))
            else:
                est = TorchMLPClassifier(n_classes=int(self.n_classes), hidden=(128, 128), epochs=120,
                                         lr=1e-3, seed=0)
                est.fit(np.asarray(Xtr, dtype=float), np.asarray(ytr))
            return np.asarray(est.predict(np.asarray(Xeval, dtype=float)))
        except Exception:  # noqa: BLE001  any torch failure -> fall back to a sklearn head
            return None

    def _fit_head(self, recipe: Recipe, Xtr, ytr, Xeval):
        if recipe.head == "torch_mlp" and self.allow_torch_head:
            pred = self._fit_torch_head(Xtr, ytr, Xeval)
            if pred is not None:
                return pred                     # else: graceful fall-through to the sklearn baseline below
        if self.is_regression:
            if recipe.head == "gbm":
                from sklearn.ensemble import HistGradientBoostingRegressor
                est = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=0)
            else:
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import make_pipeline
                est = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            est.fit(Xtr, ytr)
            return np.asarray(est.predict(Xeval))
        if recipe.head == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Xtr, ytr)
        return np.asarray(clf.predict(Xeval))

    def _predict(self, recipe: Recipe, Xtr, ytr, Xeval) -> np.ndarray:
        """Predict EVAL with the recipe. If the recipe carries an authored code_patch, ADMIT + EXECUTE it
        (the open-ended lever); a rejected/erroring patch falls back to the head so the loop still gets a
        measurement (it simply won't beat the champion)."""
        if recipe.code_patch:
            role = recipe.code_role or ("regressor" if self.is_regression else "classifier")
            ntr = len(Xtr)
            X = np.concatenate([Xtr, Xeval], axis=0)
            yfull = np.concatenate([ytr, np.zeros(len(Xeval), dtype=ytr.dtype)])  # eval labels are placeholders
            tr_idx = np.arange(ntr)
            ev_idx = np.arange(ntr, ntr + len(Xeval))
            try:
                pred, report = AB.run_authored_predict(recipe.code_patch, role, X, yfull, tr_idx, ev_idx,
                                                        n_classes=self.n_classes, family="raw", seed=0)
                if pred is not None:
                    return np.asarray(pred)
            except Exception:
                pass
        return self._fit_head(recipe, Xtr, ytr, Xeval)

    def measure(self, recipe: Recipe) -> Dict[str, TaskMeasure]:
        sp = self.splits
        X = self._feats(recipe.backbone)
        Xtr, ytr = X[sp.train_idx], sp.y_train[sp.train_idx]
        val_pred = self._predict(recipe, Xtr, ytr, X[sp.val_idx])
        val_correct = self._hit(val_pred, sp.y[sp.val_idx]).tolist()
        out: Dict[str, TaskMeasure] = {}
        for t, shard in zip(sp.tasks, sp.shards):
            pred = self._predict(recipe, Xtr, ytr, X[shard])
            correct = self._hit(pred, sp.y[shard]).tolist()
            out[t] = TaskMeasure(sealed_correct=correct, val_correct=val_correct, acc=float(np.mean(correct)))
        return out

    def gold_measure(self, recipe: Recipe) -> Dict[str, Sequence[int]]:
        sp = self.splits
        X = self._feats(recipe.backbone)
        Xtr, ytr = X[sp.train_idx], sp.y_train[sp.train_idx]
        pred = self._predict(recipe, Xtr, ytr, X[sp.gold_idx])
        correct = self._hit(pred, sp.y[sp.gold_idx]).tolist()
        return {t: correct for t in sp.tasks}

    def framing(self) -> dict:
        """meta-certifier / data-hygiene probe on the FIXED raw features: trivial-baseline (< theta on the
        balanced shapes), label-shuffle, and near-dup train/sealed straddle. A leak REFUSES the run."""
        sp = self.splits
        sealed_concat = np.concatenate(sp.shards)
        X = np.concatenate([sp.X[sp.train_idx], sp.X[sealed_concat]], axis=0)
        y = np.concatenate([sp.y[sp.train_idx], sp.y[sealed_concat]])
        n_tr = len(sp.train_idx)
        train_idx = np.arange(n_tr)
        sealed_idx = np.arange(n_tr, n_tr + len(sealed_concat))

        if self.is_regression:
            tau = float(self.tau)

            def fit_predict_fn(Xa, ya, tr, se):
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import make_pipeline
                est = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xa[tr], ya[tr])
                return est.predict(Xa[se])

            def metric_fn(yt, yp):                       # within-tolerance hit-rate (matches _hit)
                return float((np.abs(np.asarray(yt, dtype=float) - np.asarray(yp, dtype=float)) <= tau).mean())
        else:
            def fit_predict_fn(Xa, ya, tr, se):
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import make_pipeline
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xa[tr], ya[tr])
                return clf.predict(Xa[se])

            def metric_fn(yt, yp):
                return float((np.asarray(yt) == np.asarray(yp)).mean())

        out = {"X": X, "y": y, "train_idx": train_idx, "sealed_idx": sealed_idx,
               "fit_predict_fn": fit_predict_fn, "metric_fn": metric_fn}
        # re-expose the declared group/time structure (aligned to the concatenated [train, sealed] order)
        # so the meta-certifier's split-leak probe can REFUSE a grouped/temporal leak before any peek.
        if sp.split == "grouped" and sp._groups_full is not None:
            out["groups"] = np.concatenate([sp._groups_full[sp.train_idx], sp._groups_full[sealed_concat]])
        elif sp.split == "time" and sp._times_full is not None:
            out["times"] = np.concatenate([sp._times_full[sp.train_idx], sp._times_full[sealed_concat]])
        return out


__all__ = ["TabularSplits", "TabularCodeArena", "_load_dataset", "_load_openml", "_derive_groups",
           "OPENML_CC18", "REGRESSION_DATASETS"]
