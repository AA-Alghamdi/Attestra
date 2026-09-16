"""CONCRETE ARENA for the autonomous representation researcher, backed by the EXACT phase-#3/#4/#5 machinery.

This is the domain the policy (vfplatform.repr_researcher.ReprResearcher) operates on for the FGVC-Aircraft
fine-grained arena. It reuses, verbatim, the helpers that produced every hand-run number so the system's
autonomous verdict is measured on byte-identical sealed rows with the same tuned-GBM head and the same frozen
Clopper-Pearson bound:

  * the sealed split is derived ONCE from the CLIP-B baseline embeddings (`benchmark_backbones._sealed_split`)
    and reused for every encoder -> all McNemar pairing is on identical sealed rows;
  * each encoder's head is the SAME strong tuned-GBM + random search (`benchmark_vision_transfer._random_search_best`)
    selected on val and scored on the sealed rows -> ONLY the representation differs;
  * quality is the frozen `science.clopper_pearson_lower` (read-only), reached via `benchmark_vision_transfer._lb`.

Embeddings are loaded from the phase-#5 disk cache (data/_emb_cache_phase5), so the whole autonomous run is
seconds of CPU. measurements are memoized so the researcher's repeated measure() calls (e.g. for the Pareto
front) are free. This module supplies measurements only; it never decides a promotion.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.battery import benjamini_hochberg, mcnemar_pvalue          # noqa: E402
from vfplatform.repr_researcher import Arena, Encoder, TaskMeasure          # noqa: E402
import scripts.benchmark_aircraft as A                                      # noqa: E402
import scripts.benchmark_backbones as BB                                    # noqa: E402
import scripts.benchmark_vision_transfer as B1                              # noqa: E402

PER_CLASS = int(os.environ.get("ATTESTRA_PER_CLASS", "100"))
CACHE_DIR = os.environ.get("ATTESTRA_EMB_CACHE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "_emb_cache_phase5"))
BASELINE_TAG = "clip_vitb32"   # the split is derived from this encoder's embeddings (as in #3/#4/#5)

# The registry the system climbs on this arena: the five frozen encoders with cached embeddings, each tagged
# with its inductive-bias FAMILY, within-family SCALE rank (bigger last), and parameter count (a cost proxy).
REGISTRY: List[Encoder] = [
    Encoder("clip_vitb32",   "CLIP ViT-B/32 (language-aligned)", "clip",   0,   88.0),
    Encoder("dinov2_vitl14", "DINOv2 ViT-L/14 (self-supervised)", "dinov2", 1,  300.0),
    Encoder("dinov2_g",      "DINOv2 ViT-g/14 (self-supervised)", "dinov2", 2, 1100.0),
    Encoder("eva02_l",       "EVA-02 ViT-L/14 (masked-image SSL)", "eva02", 1,  304.0),
    Encoder("siglip_so",     "SigLIP SO400M/14 (language-aligned)", "siglip", 1, 428.0),
]


class FgvcAircraftArena(Arena):
    """FGVC-Aircraft confusable-variant binary suite, backed by the cached phase-#5 embeddings."""

    def __init__(self, per_class: int = PER_CLASS, cache_dir: str = CACHE_DIR, smoke: int = 0):
        self.per_class = per_class
        self.cache_dir = cache_dir
        suite = A.SUITE[:smoke] if smoke else A.SUITE
        self.tasks: List[str] = [f"{a}_vs_{b}" for a, b in suite]
        self._pairs = {f"{a}_vs_{b}": (a, b) for a, b in suite}
        self._split: Dict[str, dict] = {}        # task -> {y, tr, val, test}
        self._meas: Dict[str, Dict[str, TaskMeasure]] = {}   # encoder_tag / fuse-key -> measurement

    # -- embedding + split plumbing (identical to the hand-run) -----------------------------------------
    def _emb(self, tag: str, task: str) -> np.ndarray:
        a, b = self._pairs[task]
        paths, _ = A._task_paths(a, b, self.per_class, 0)
        fp = os.path.join(self.cache_dir, f"{tag}__{task}__pc{self.per_class}_n{len(paths)}.npy")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"cached embedding missing: {fp}")
        return np.load(fp).astype(np.float32)

    def _task_split(self, task: str) -> dict:
        if task not in self._split:
            a, b = self._pairs[task]
            _, y = A._task_paths(a, b, self.per_class, 0)
            emb_base = self._emb(BASELINE_TAG, task)
            tr, val, test = BB._sealed_split(emb_base, y, 0)
            self._split[task] = {"y": y, "tr": tr, "val": val, "test": test}
        return self._split[task]

    @staticmethod
    def _fit_correct(emb: np.ndarray, y: np.ndarray, sp: dict):
        """Fit the strong tuned-GBM head (val-selected) and return (val_correct, test_correct, test_acc)."""
        Xtr, ytr = emb[sp["tr"]], y[sp["tr"]]
        Xva, yva = emb[sp["val"]], y[sp["val"]]
        Xte, yte = emb[sp["test"]], y[sp["test"]]
        est = B1._random_search_best(np.random.RandomState(0), Xtr, ytr, Xva, yva)
        val_c = B1._correct(est, Xva, yva)
        test_c = B1._correct(est, Xte, yte)
        return val_c, test_c, float(np.mean(test_c))

    # -- Arena interface --------------------------------------------------------------------------------
    def measure(self, encoder_tag: str) -> Dict[str, TaskMeasure]:
        if encoder_tag not in self._meas:
            out: Dict[str, TaskMeasure] = {}
            for task in self.tasks:
                sp = self._task_split(task)
                val_c, test_c, acc = self._fit_correct(self._emb(encoder_tag, task), sp["y"], sp)
                out[task] = TaskMeasure(sealed_correct=test_c, val_correct=val_c, acc=acc)
            self._meas[encoder_tag] = out
        return self._meas[encoder_tag]

    def fuse_measure(self, tag_a: str, tag_b: str) -> Dict[str, TaskMeasure]:
        key = f"fuse[{tag_a}+{tag_b}]"
        if key not in self._meas:
            out: Dict[str, TaskMeasure] = {}
            for task in self.tasks:
                sp = self._task_split(task)
                emb = np.concatenate([self._emb(tag_a, task), self._emb(tag_b, task)], axis=1)
                val_c, test_c, acc = self._fit_correct(emb, sp["y"], sp)
                out[task] = TaskMeasure(sealed_correct=test_c, val_correct=val_c, acc=acc)
            self._meas[key] = out
        return self._meas[key]

    def mcnemar(self, chal_correct: Sequence[int], base_correct: Sequence[int]) -> float:
        return mcnemar_pvalue(list(chal_correct), list(base_correct))

    def bh(self, pvalues: Sequence[float], alpha: float) -> List[int]:
        return list(benjamini_hochberg(list(pvalues), alpha=alpha))

    def lower_bound(self, correct: Sequence[int]) -> float:
        return B1._lb(list(correct))

    def gold_measure(self, name: str):
        """No disjoint gold partition exists for this arena, so it honestly returns None (the brain then emits
        no gold confirmation rather than a fabricated one). FGVC-Aircraft pools trainval+test to exactly 100
        images per variant; at the arena's per_class=100 the train/val/sealed rows already consume EVERY image
        of each variant, so any "gold" set would overlap the working rows -- a leak. The gold tier is therefore
        demonstrated on the text arena, which has real corpus headroom (~600-990 docs/class vs 240 used). A
        vision arena with image headroom (an ImageNet-scale / CIFAR source) would carve gold exactly as the
        text arena does: disjoint images, encoded through the frozen backbone, head trained on train rows only,
        scored once on the never-peeked gold rows."""
        return None


__all__ = ["FgvcAircraftArena", "REGISTRY", "BASELINE_TAG", "PER_CLASS", "CACHE_DIR"]
