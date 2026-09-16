"""CONCRETE ARENA #3 -- TABULAR: does the cross-modal "representation is the lever" law extend to tabular data?

On vision (7 phases) and text (#2), the certified law is: given a representation, authoring on top adds nothing
(0/20 FDR), and the only lever that moves the metric is *changing the representation* (a frozen encoder). On
tabular there is no embedding to swap -- the columns ARE the native representation, and a TUNED GRADIENT-BOOSTING
MACHINE on the raw columns is the strong baseline a competent engineer reaches for. The honest tabular analog of
"change the representation" is therefore to swap the whole inductive prior: from a per-dataset-fit GBM to a
PRETRAINED TABULAR FOUNDATION MODEL (TabPFN v2), whose transformer carries an in-context Bayesian prior learned
over millions of synthetic tabular tasks. So this arena asks, under the IDENTICAL frozen certifier + the SAME
`ReprResearcher` brain as vision/text:

    start champion = the strong tuned-GBM on raw features (NOT a weak baseline -- the mid-level-engineer default)
    model rung     = swap family GBM -> TabPFN          (the representation lever, if it exists on tabular)
    features rung  = STACK champion + the other family   (authored featurizer analog = soft-vote of the two)
    capacity rung  = scale the winning family            (TabPFN 8- -> 32-member ensemble)
    -> promote ONLY on an FDR-surviving win over the current champion; the frozen certifier is the sole promoter.

DOMAIN: UCI letter-recognition (openml `letter`, 16 numeric features, 26 classes, ~750 rows/class), reduced to a
pre-registered suite of SHAPE-CONFUSABLE binary letter pairs -- the tabular analog of confusable aircraft variants
(#3) and confusable newsgroup pairs (#2). Each task has its OWN identical sealed rows (GBM and TabPFN are scored on
byte-identical sealed examples, so McNemar pairing is exact), and a DISJOINT never-peeked GOLD tail for the
multiplicity-free confirmation. Training is deliberately small (the regime TabPFN is designed for and where a GBM
is not yet saturated), so the family-swap question is a FAIR test rather than a foregone GBM win.

ENVIRONMENT: TabPFN v2 pins scikit-learn<1.7, which conflicts with the repo's 1.9.0. To keep the validated
vision/text environment byte-identical, this arena runs in an ISOLATED venv (`~/.venv-tabpfn`, sklearn 1.6.1 +
tabpfn 2.2.1, system torch). The frozen science core (`vectorforge/science.py` b564fba2, `vfplatform/sealed.py`
30ad6245) is numpy-only and verified byte-identical there; only the GBM head's sklearn minor version differs, and
that head is the *baseline being beaten*, never the certifier.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import fetch_openml                                       # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,  # noqa: E402
                              ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsClassifier                             # noqa: E402
from sklearn.linear_model import LogisticRegression                            # noqa: E402
from sklearn.pipeline import make_pipeline                                     # noqa: E402
from sklearn.preprocessing import StandardScaler                              # noqa: E402

from vectorforge import science                                               # noqa: E402
from vfplatform.battery import benjamini_hochberg, mcnemar_pvalue              # noqa: E402
from vfplatform.repr_researcher import Arena, Encoder, TaskMeasure             # noqa: E402

ALPHA = 0.05    # the frozen Clopper-Pearson bound's confidence level (identical to vision/text `_lb`)
SEED = 0

# Pre-registered SHAPE-CONFUSABLE letter pairs (fixed before any run; not chosen by their hardness). These are the
# classically hard-to-separate letter shapes -- the tabular analog of confusable aircraft variants.
SUITE: List[Tuple[str, str]] = [
    ("O", "Q"), ("E", "F"), ("M", "N"), ("U", "V"), ("B", "D"),
    ("I", "J"), ("K", "X"), ("P", "R"), ("C", "G"), ("V", "Y"),
]

# Deliberately SMALL training budget (TabPFN's design regime; a GBM is not yet saturated), with a well-powered
# sealed test and a DISJOINT never-peeked gold tail. Per class, drawn from one fixed shuffle.
TRAIN_PC = int(os.environ.get("ATTESTRA_TAB_TRAIN_PC", "40"))
VAL_PC = int(os.environ.get("ATTESTRA_TAB_VAL_PC", "25"))
SEALED_PC = int(os.environ.get("ATTESTRA_TAB_SEALED_PC", "90"))
GOLD_PC = int(os.environ.get("ATTESTRA_TAB_GOLD_PC", "90"))

BASELINE_TAG = "gbm_raw"   # START champion = the strong tuned-GBM (the competent-engineer default)
TABPFN_TAGS = {"tabpfn", "tabpfn_big"}
_TABPFN_ENSEMBLE = {"tabpfn": 8, "tabpfn_big": 32}

# The registry the system climbs on this arena. `family` groups a shared inductive prior; `scale_rank` orders
# members within a family (bigger last); `params_m` is a cost proxy for the certified Pareto front.
REGISTRY: List[Encoder] = [
    Encoder("gbm_raw",    "Tuned GBM/RF/ET/logreg/knn search on raw columns (strong baseline)", "gbm",    0,   0.5),
    Encoder("tabpfn",     "TabPFN v2 foundation model (8-member ensemble)",                     "tabpfn", 0,  25.0),
    Encoder("tabpfn_big", "TabPFN v2 foundation model (32-member ensemble)",                    "tabpfn", 1,  25.0),
]


def _strong_baseline(rng, Xtr, ytr, Xva, yva, k=15):
    """The IDENTICAL strong tuned baseline recipe used by the vision/text arenas (`benchmark_vision_transfer.
    _random_search_best`): a random search over GBM/RF/ET/logreg/knn selected on val, plus a tuned-GBM default so
    the baseline is never weaker than one strong model. Replicated here so the tabular arena imports no vision
    (torch) stack; the search space is byte-for-byte the same."""
    best, best_v = None, -1.0
    for _ in range(k):
        fam = rng.choice(["gbm", "rf", "et", "logreg", "knn"])
        if fam == "gbm":
            est = HistGradientBoostingClassifier(learning_rate=float(rng.choice([0.05, 0.1, 0.2])),
                                                 max_iter=int(rng.choice([100, 300])),
                                                 max_depth=rng.choice([None, 6, 12]), random_state=0)
        elif fam == "rf":
            est = RandomForestClassifier(n_estimators=int(rng.choice([200, 400])),
                                         max_depth=rng.choice([None, 12]),
                                         max_features=rng.choice(["sqrt", 0.5]), random_state=0, n_jobs=1)
        elif fam == "et":
            est = ExtraTreesClassifier(n_estimators=int(rng.choice([200, 400])),
                                       max_features=rng.choice(["sqrt", 0.5]), random_state=0, n_jobs=1)
        elif fam == "logreg":
            est = make_pipeline(StandardScaler(),
                                LogisticRegression(C=float(rng.choice([0.1, 1.0, 10.0])), max_iter=2000))
        else:
            est = make_pipeline(StandardScaler(),
                                KNeighborsClassifier(n_neighbors=int(rng.choice([5, 11, 21])), n_jobs=1))
        try:
            est.fit(Xtr, ytr)
            v = est.score(Xva, yva)
        except Exception:  # noqa: BLE001
            continue
        if v > best_v:
            best_v, best = v, est
    gbm = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    if gbm.score(Xva, yva) > best_v or best is None:
        best = gbm
    return best


def _lb(correct: Sequence[int]) -> float:
    k, n = int(sum(correct)), len(correct)
    return round(science.clopper_pearson_lower(k, n, ALPHA), 4)


class LetterPairArena(Arena):
    """Confusable letter-pair tabular suite. Predictors are END-TO-END (a GBM search or a TabPFN forward pass);
    each is trained on the train rows, GBM is val-selected, and every arm is scored on the IDENTICAL sealed rows
    of the task. fuse_measure soft-votes two predictors (the authored-featurizer analog for tabular)."""

    def __init__(self, smoke: int = 0):
        suite = SUITE[:smoke] if smoke else SUITE
        self.tasks: List[str] = [f"{a}_vs_{b}" for a, b in suite]
        self._pairs = {f"{a}_vs_{b}": (a, b) for a, b in suite}
        self._raw: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._split: Dict[str, dict] = {}
        self._meas: Dict[str, Dict[str, TaskMeasure]] = {}

    # -- data plumbing ----------------------------------------------------------------------------------
    def _load(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._raw is None:
            d = fetch_openml("letter", version=1, as_frame=False, cache=True)
            self._raw = (d.data.astype(np.float32), np.asarray(d.target))
        return self._raw

    def _task_split(self, task: str) -> dict:
        """Deterministic class-balanced split into train/val/sealed (the working pool) + a DISJOINT gold tail,
        all drawn from ONE fixed per-class shuffle so gold shares no row with the working pool."""
        if task not in self._split:
            a, b = self._pairs[task]
            X, y = self._load()
            need = TRAIN_PC + VAL_PC + SEALED_PC + GOLD_PC
            cols = {"Xtr": [], "ytr": [], "Xva": [], "yva": [], "Xte": [], "yte": [],
                    "Xgo": [], "ygo": []}
            for lbl, letter in ((0, a), (1, b)):
                idx = np.where(y == letter)[0]
                rng = np.random.RandomState(SEED + 17 + lbl)
                rng.shuffle(idx)
                if len(idx) < need:
                    raise RuntimeError(f"class {letter!r} has only {len(idx)} rows (< {need})")
                tr = idx[:TRAIN_PC]
                va = idx[TRAIN_PC:TRAIN_PC + VAL_PC]
                te = idx[TRAIN_PC + VAL_PC:TRAIN_PC + VAL_PC + SEALED_PC]
                go = idx[TRAIN_PC + VAL_PC + SEALED_PC:need]
                cols["Xtr"].append(X[tr]); cols["ytr"] += [lbl] * len(tr)
                cols["Xva"].append(X[va]); cols["yva"] += [lbl] * len(va)
                cols["Xte"].append(X[te]); cols["yte"] += [lbl] * len(te)
                cols["Xgo"].append(X[go]); cols["ygo"] += [lbl] * len(go)
            self._split[task] = {
                "Xtr": np.concatenate(cols["Xtr"]), "ytr": np.asarray(cols["ytr"]),
                "Xva": np.concatenate(cols["Xva"]), "yva": np.asarray(cols["yva"]),
                "Xte": np.concatenate(cols["Xte"]), "yte": np.asarray(cols["yte"]),
                "Xgo": np.concatenate(cols["Xgo"]), "ygo": np.asarray(cols["ygo"]),
            }
        return self._split[task]

    # -- predictors -------------------------------------------------------------------------------------
    @staticmethod
    def _fit(tag: str, sp: dict):
        """Fit the tag's predictor on the train rows (GBM is val-selected; TabPFN has no per-task tuning)."""
        if tag in TABPFN_TAGS:
            from tabpfn import TabPFNClassifier
            clf = TabPFNClassifier(device="cpu", n_estimators=_TABPFN_ENSEMBLE[tag], random_state=0)
            clf.fit(sp["Xtr"], sp["ytr"])
            return clf
        return _strong_baseline(np.random.RandomState(0), sp["Xtr"], sp["ytr"], sp["Xva"], sp["yva"])

    @staticmethod
    def _proba(est, X: np.ndarray) -> np.ndarray:
        """predict_proba aligned to class order [0, 1] (robust to either estimator's internal class ordering)."""
        p = est.predict_proba(X)
        classes = list(est.classes_)
        return np.column_stack([p[:, classes.index(0)], p[:, classes.index(1)]])

    # -- Arena interface --------------------------------------------------------------------------------
    def measure(self, encoder_tag: str) -> Dict[str, TaskMeasure]:
        if encoder_tag not in self._meas:
            out: Dict[str, TaskMeasure] = {}
            for task in self.tasks:
                sp = self._task_split(task)
                est = self._fit(encoder_tag, sp)
                vc = (est.predict(sp["Xva"]) == sp["yva"]).astype(int).tolist()
                tc = (est.predict(sp["Xte"]) == sp["yte"]).astype(int).tolist()
                out[task] = TaskMeasure(sealed_correct=tc, val_correct=vc, acc=float(np.mean(tc)))
            self._meas[encoder_tag] = out
        return self._meas[encoder_tag]

    def fuse_measure(self, tag_a: str, tag_b: str) -> Dict[str, TaskMeasure]:
        """Authored-featurizer analog for tabular: soft-vote (average predicted class probabilities) of the two
        predictors -- does STACKING the foundation model with the GBM beat the best single model?"""
        key = f"fuse[{tag_a}+{tag_b}]"
        if key not in self._meas:
            out: Dict[str, TaskMeasure] = {}
            for task in self.tasks:
                sp = self._task_split(task)
                ea, eb = self._fit(tag_a, sp), self._fit(tag_b, sp)
                pv = (self._proba(ea, sp["Xva"]) + self._proba(eb, sp["Xva"])) / 2.0
                pt = (self._proba(ea, sp["Xte"]) + self._proba(eb, sp["Xte"])) / 2.0
                vc = (pv.argmax(1) == sp["yva"]).astype(int).tolist()
                tc = (pt.argmax(1) == sp["yte"]).astype(int).tolist()
                out[task] = TaskMeasure(sealed_correct=tc, val_correct=vc, acc=float(np.mean(tc)))
            self._meas[key] = out
        return self._meas[key]

    def mcnemar(self, chal_correct: Sequence[int], base_correct: Sequence[int]) -> float:
        return mcnemar_pvalue(list(chal_correct), list(base_correct))

    def bh(self, pvalues: Sequence[float], alpha: float) -> List[int]:
        return list(benjamini_hochberg(list(pvalues), alpha=alpha))

    def lower_bound(self, correct: Sequence[int]) -> float:
        return _lb(list(correct))

    def gold_measure(self, name: str) -> Optional[Dict[str, Sequence[int]]]:
        """Score `name` (a single tag or a `fuse[a+b]` key) on the DISJOINT never-peeked gold rows. The head is
        trained the standard way (train rows, GBM val-selected); gold is read EXACTLY ONCE by the brain (final
        champion + start baseline) after all climbing, so the gold confirmation carries no climb multiplicity."""
        out: Dict[str, Sequence[int]] = {}
        if name.startswith("fuse[") and name.endswith("]"):
            a, b = name[5:-1].split("+", 1)
            for task in self.tasks:
                sp = self._task_split(task)
                ea, eb = self._fit(a, sp), self._fit(b, sp)
                pg = (self._proba(ea, sp["Xgo"]) + self._proba(eb, sp["Xgo"])) / 2.0
                out[task] = (pg.argmax(1) == sp["ygo"]).astype(int).tolist()
        else:
            for task in self.tasks:
                sp = self._task_split(task)
                est = self._fit(name, sp)
                out[task] = (est.predict(sp["Xgo"]) == sp["ygo"]).astype(int).tolist()
        return out


__all__ = ["LetterPairArena", "REGISTRY", "BASELINE_TAG", "SUITE",
           "TRAIN_PC", "VAL_PC", "SEALED_PC", "GOLD_PC"]
