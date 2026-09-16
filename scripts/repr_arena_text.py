"""TEXT arena for the autonomous representation researcher -- the #2 GENERALITY test.

This plugs the IDENTICAL brain (vfplatform.repr_researcher.ReprResearcher) and the IDENTICAL frozen certifier
(verification.VerificationCascade -> science.clopper_pearson_lower) into a TEXT domain. Nothing about the
policy or the statistics changes; only the domain (images -> text) and the registry (vision backbones ->
frozen sentence encoders). If "the representation is the lever, and authoring on a fixed representation adds
nothing" is a real law rather than a vision artifact, the same controller must reproduce it here, autonomously.

The discipline is byte-identical to B1->#5:
  * the sealed split is derived ONCE from the WEAK baseline (tfidf+LSA) representation via the exact same
    `benchmark_backbones._sealed_split`, then reused for every encoder -> all McNemar pairing is on identical
    sealed rows (only the representation differs across arms);
  * every encoder's head is the SAME strong tuned model-search (`benchmark_vision_transfer._random_search_best`,
    a random search over GBM/RF/ET/logreg/knn + a tuned-GBM default), selected on val and scored on sealed;
  * quality is the frozen `science.clopper_pearson_lower` (read-only), reached via `benchmark_vision_transfer._lb`.

EVERY encoder -- including the lexical baseline -- is an UNSUPERVISED, frozen feature map computed on the pair's
documents (TF-IDF+LSA uses no labels; the sentence transformers are frozen). Only the HEAD sees labels, and only
on train rows. So "encoder" and "head" are cleanly separated exactly as in the vision arena.

Domain: 20 Newsgroups, six genuinely CONFUSABLE binary pairs (high lexical overlap, distinct semantics) so a
lexical bag-of-words has real headroom for a semantic representation to climb into.

Embeddings are cached to data/_emb_cache_text so the autonomous run (and re-runs for the Pareto front) are fast.
This module supplies measurements only; it never decides a promotion.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform.battery import benjamini_hochberg, mcnemar_pvalue          # noqa: E402
from vfplatform.repr_researcher import Arena, Encoder, TaskMeasure          # noqa: E402
import scripts.benchmark_backbones as BB                                    # noqa: E402
import scripts.benchmark_vision_transfer as B1                              # noqa: E402

PER_CLASS = int(os.environ.get("ATTESTRA_TEXT_PER_CLASS", "240"))
# A DISJOINT, never-peeked GOLD set per class -- drawn from documents AFTER the first PER_CLASS in the same
# deterministic shuffle, so it shares no rows with train/val/sealed and the existing sealed results stay
# byte-identical (gold lives in separate cache files). The smallest 20NG category has ~597 usable docs, so
# 240 + 200 = 440 fits with margin. The brain reads gold EXACTLY ONCE (champion + baseline) after climbing.
GOLD_PER_CLASS = int(os.environ.get("ATTESTRA_TEXT_GOLD_PER_CLASS", "200"))
CACHE_DIR = os.environ.get("ATTESTRA_TEXT_EMB_CACHE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "_emb_cache_text"))
BASELINE_TAG = "tfidf_lsa"   # the split is derived from the WEAK lexical baseline (the start champion)
SEED = 0

# Six genuinely confusable 20-Newsgroups binary pairs. Each (a, b) becomes a task "a_vs_b". The first three
# share heavy surface vocabulary (PC/Mac hardware, the two windowing systems, religion-vs-atheism debate) so a
# lexical model has real headroom; the rest span easier topic splits so the suite is not cherry-picked.
SUITE: List[Tuple[str, str]] = [
    ("comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"),
    ("comp.os.ms-windows.misc", "comp.windows.x"),
    ("alt.atheism", "talk.religion.misc"),
    ("rec.sport.baseball", "rec.sport.hockey"),
    ("rec.autos", "rec.motorcycles"),
    ("sci.electronics", "sci.space"),
]


def _short(cat: str) -> str:
    """A filesystem-safe short tag for a newsgroup category (last two dotted segments)."""
    return ".".join(cat.split(".")[-2:])


def task_name(a: str, b: str) -> str:
    return f"{_short(a)}_vs_{_short(b)}"


# The registry the system climbs on this arena. `family` groups a shared pretraining bias; `scale_rank` orders
# members within a family (bigger last) -- the e5 family gives a clean small->base->large scale ladder, the
# direct text analog of DINOv2 S->L->g. `params_m` is a cost proxy for the certified Pareto front.
REGISTRY: List[Encoder] = [
    Encoder("tfidf_lsa", "TF-IDF + LSA-300 (lexical bag-of-words)",       "lexical", 0,   0.0),
    Encoder("minilm",    "all-MiniLM-L6-v2 (distilled SBERT)",            "minilm",  0,  22.0),
    Encoder("mpnet",     "all-mpnet-base-v2 (MPNet SBERT)",               "mpnet",   0, 110.0),
    Encoder("e5_small",  "intfloat/e5-small-v2 (contrastive retrieval)",  "e5",      0,  33.0),
    Encoder("e5_base",   "intfloat/e5-base-v2 (contrastive retrieval)",   "e5",      1, 110.0),
    Encoder("e5_large",  "intfloat/e5-large-v2 (contrastive retrieval)",  "e5",      2, 335.0),
]

# tag -> (sentence-transformers model id, e5-style instruction prefix). tfidf_lsa is handled separately.
_ST_MODELS: Dict[str, Tuple[str, str]] = {
    "minilm":   ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "mpnet":    ("sentence-transformers/all-mpnet-base-v2", ""),
    "e5_small": ("intfloat/e5-small-v2", "query: "),
    "e5_base":  ("intfloat/e5-base-v2",  "query: "),
    "e5_large": ("intfloat/e5-large-v2", "query: "),
}

_MAX_WORDS = 256   # cap each document so CPU encoding of the larger models stays tractable


# ---------------------------------------------------------------------------------------------------------
# Deterministic per-pair document set (cached to disk so n is stable across encoders and runs).
# ---------------------------------------------------------------------------------------------------------
def _pair_docs(a: str, b: str, per_class: int) -> Tuple[List[str], np.ndarray]:
    """Return (texts, y) for a confusable pair: `per_class` non-empty documents from each category, drawn
    deterministically from the full 20NG corpus with headers/footers/quotes stripped (so the signal is the
    body text, not metadata). Truncated to _MAX_WORDS words. This uses NO labels beyond the category folder."""
    from sklearn.datasets import fetch_20newsgroups
    texts: List[str] = []
    y: List[int] = []
    for cls, cat in enumerate((a, b)):
        ds = fetch_20newsgroups(subset="all", categories=[cat],
                                remove=("headers", "footers", "quotes"), random_state=SEED)
        docs = [d.strip() for d in ds.data if len(d.split()) >= 5]   # drop empties / near-empty stubs
        rng = np.random.RandomState(SEED + cls + 13)
        idx = np.arange(len(docs)); rng.shuffle(idx)
        if len(docs) < per_class:
            raise RuntimeError(f"category {cat!r} has only {len(docs)} usable docs (< {per_class})")
        for i in idx[:per_class]:
            texts.append(" ".join(docs[i].split()[:_MAX_WORDS]))
            y.append(cls)
    return texts, np.asarray(y, dtype=int)


def _gold_pair_docs(a: str, b: str, per_class: int, gold_per_class: int) -> Tuple[List[str], np.ndarray]:
    """The DISJOINT gold documents for a pair: the `gold_per_class` documents drawn AFTER the first
    `per_class` in the IDENTICAL deterministic shuffle used by `_pair_docs`. Because the shuffle (seed, rng)
    is byte-identical, `idx[:per_class]` (train/val/sealed pool) and `idx[per_class:per_class+gold]` (gold)
    are guaranteed non-overlapping -- gold is genuinely fresh, never-seen text. Capped at what each category
    actually has beyond `per_class` (honest degradation; the realized n is reported)."""
    from sklearn.datasets import fetch_20newsgroups
    texts: List[str] = []
    y: List[int] = []
    for cls, cat in enumerate((a, b)):
        ds = fetch_20newsgroups(subset="all", categories=[cat],
                                remove=("headers", "footers", "quotes"), random_state=SEED)
        docs = [d.strip() for d in ds.data if len(d.split()) >= 5]
        rng = np.random.RandomState(SEED + cls + 13)           # SAME rng as _pair_docs -> same shuffle
        idx = np.arange(len(docs)); rng.shuffle(idx)
        avail = len(docs) - per_class
        g = max(0, min(gold_per_class, avail))
        for i in idx[per_class:per_class + g]:
            texts.append(" ".join(docs[i].split()[:_MAX_WORDS]))
            y.append(cls)
    return texts, np.asarray(y, dtype=int)


def _tfidf_lsa_pipeline(ref_texts: List[str]):
    """Build + fit the EXACT tfidf_lsa feature map of `_encode` on `ref_texts`, returned as a fitted
    transformer so the SAME lexical space can be applied to fresh gold text. Mirrors `_encode` verbatim
    (sublinear tf, min_df=2, max_df=0.5, english stopwords, LSA-300, L2 normalize)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, stop_words="english")
    X = vec.fit_transform(ref_texts)
    k = min(300, X.shape[1] - 1)
    svd = make_pipeline(TruncatedSVD(n_components=k, random_state=SEED), Normalizer(copy=False))
    svd.fit(X)

    def transform(texts: List[str]) -> np.ndarray:
        return svd.transform(vec.transform(texts)).astype(np.float32)

    return transform


_ST_CACHE: Dict[str, object] = {}   # loaded SentenceTransformer models, memoized within a process


def _encode(tag: str, texts: List[str]) -> np.ndarray:
    """Compute the FROZEN embedding matrix for `texts` under encoder `tag`. Unsupervised for every tag."""
    if tag == "tfidf_lsa":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        # Fit unsupervised on the pair's documents (no labels) -> a fixed lexical feature map, exactly the role
        # a frozen neural encoder plays. LSA to 300 dims gives a dense representation for the same tuned head.
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, stop_words="english")
        X = vec.fit_transform(texts)
        k = min(300, X.shape[1] - 1)
        svd = make_pipeline(TruncatedSVD(n_components=k, random_state=SEED), Normalizer(copy=False))
        return svd.fit_transform(X).astype(np.float32)
    model_id, prefix = _ST_MODELS[tag]
    if tag not in _ST_CACHE:
        from sentence_transformers import SentenceTransformer
        _ST_CACHE[tag] = SentenceTransformer(model_id, device="cpu")
    model = _ST_CACHE[tag]
    inp = [prefix + t for t in texts] if prefix else texts
    emb = model.encode(inp, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


class TwentyNewsArena(Arena):
    """20-Newsgroups confusable-pair binary suite over frozen text representations, cached to disk."""

    def __init__(self, per_class: int = PER_CLASS, cache_dir: str = CACHE_DIR, smoke: int = 0,
                 gold_per_class: int = GOLD_PER_CLASS):
        self.per_class = per_class
        self.gold_per_class = gold_per_class
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        suite = SUITE[:smoke] if smoke else SUITE
        self._pairs = {task_name(a, b): (a, b) for a, b in suite}
        self.tasks: List[str] = list(self._pairs.keys())
        self._docs: Dict[str, Tuple[List[str], np.ndarray]] = {}
        self._gold_docs: Dict[str, Tuple[List[str], np.ndarray]] = {}
        self._split: Dict[str, dict] = {}
        self._meas: Dict[str, Dict[str, TaskMeasure]] = {}
        self._gold_meas: Dict[str, Optional[Dict[str, List[int]]]] = {}

    # -- embedding + split plumbing (lazy, disk-cached) -------------------------------------------------
    def _pair(self, task: str) -> Tuple[List[str], np.ndarray]:
        if task not in self._docs:
            a, b = self._pairs[task]
            self._docs[task] = _pair_docs(a, b, self.per_class)
        return self._docs[task]

    def _emb(self, tag: str, task: str) -> np.ndarray:
        texts, _ = self._pair(task)
        fp = os.path.join(self.cache_dir, f"{tag}__{task}__pc{self.per_class}_n{len(texts)}.npy")
        if os.path.exists(fp):
            return np.load(fp).astype(np.float32)
        emb = _encode(tag, texts)
        np.save(fp, emb)
        return emb

    def _task_split(self, task: str) -> dict:
        if task not in self._split:
            _, y = self._pair(task)
            emb_base = self._emb(BASELINE_TAG, task)
            tr, val, test = BB._sealed_split(emb_base, y, SEED)
            self._split[task] = {"y": y, "tr": tr, "val": val, "test": test}
        return self._split[task]

    @staticmethod
    def _fit_correct(emb: np.ndarray, y: np.ndarray, sp: dict):
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

    # -- GOLD (never-peeked) confirmation set -----------------------------------------------------------
    def _golddocs(self, task: str) -> Tuple[List[str], np.ndarray]:
        if task not in self._gold_docs:
            a, b = self._pairs[task]
            self._gold_docs[task] = _gold_pair_docs(a, b, self.per_class, self.gold_per_class)
        return self._gold_docs[task]

    def _emb_gold(self, tag: str, task: str) -> np.ndarray:
        """Frozen embedding of the gold documents under `tag` (cached separately from the sealed cache)."""
        gold_texts, _ = self._golddocs(task)
        fp = os.path.join(self.cache_dir,
                          f"{tag}__{task}__pc{self.per_class}_gold{self.gold_per_class}_n{len(gold_texts)}.npy")
        if os.path.exists(fp):
            return np.load(fp).astype(np.float32)
        emb = _encode(tag, gold_texts)
        np.save(fp, emb)
        return emb

    def _emb_pair(self, tag: str, task: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (original-row embeddings, gold-row embeddings) for `tag` in ONE consistent feature space.
        Stateless sentence encoders reuse the cached original embedding and freshly encode gold (same model,
        same normalization). The STATEFUL tfidf_lsa map is refit once on the original texts (deterministic)
        and applied to both, so the gold rows live in the identical lexical space as the head's training."""
        if tag == "tfidf_lsa":
            texts, _ = self._pair(task)
            gold_texts, _ = self._golddocs(task)
            transform = _tfidf_lsa_pipeline(texts)
            return transform(texts), transform(gold_texts)
        return self._emb(tag, task), self._emb_gold(tag, task)

    @staticmethod
    def _parse_name(name: str) -> List[str]:
        if name.startswith("fuse[") and name.endswith("]"):
            return name[5:-1].split("+")
        return [name]

    @staticmethod
    def _fit_gold_correct(X_full: np.ndarray, X_gold: np.ndarray, y: np.ndarray,
                          y_gold: np.ndarray, sp: dict) -> List[int]:
        """Train the IDENTICAL val-selected head on the original train rows and score it on the gold rows."""
        Xtr, ytr = X_full[sp["tr"]], y[sp["tr"]]
        Xva, yva = X_full[sp["val"]], y[sp["val"]]
        est = B1._random_search_best(np.random.RandomState(0), Xtr, ytr, Xva, yva)
        return B1._correct(est, X_gold, y_gold)

    def gold_measure(self, name: str) -> Optional[Dict[str, List[int]]]:
        """Per-task 0/1 correctness of `name` (an encoder tag or fuse[a+b]) on the DISJOINT gold rows. The
        brain calls this exactly once each for the final champion and the start baseline, after climbing."""
        tags = self._parse_name(name)
        known = {e.tag for e in REGISTRY}
        if any(t not in known for t in tags):
            return None
        if name not in self._gold_meas:
            out: Dict[str, List[int]] = {}
            for task in self.tasks:
                sp = self._task_split(task)
                _, y = self._pair(task)
                _, y_gold = self._golddocs(task)
                fulls, golds = [], []
                for t in tags:
                    Xf, Xg = self._emb_pair(t, task)
                    fulls.append(Xf)
                    golds.append(Xg)
                X_full = np.concatenate(fulls, axis=1) if len(fulls) > 1 else fulls[0]
                X_gold = np.concatenate(golds, axis=1) if len(golds) > 1 else golds[0]
                out[task] = self._fit_gold_correct(X_full, X_gold, y, y_gold, sp)
            self._gold_meas[name] = out
        return self._gold_meas[name]

    def mcnemar(self, chal_correct: Sequence[int], base_correct: Sequence[int]) -> float:
        return mcnemar_pvalue(list(chal_correct), list(base_correct))

    def bh(self, pvalues: Sequence[float], alpha: float) -> List[int]:
        return list(benjamini_hochberg(list(pvalues), alpha=alpha))

    def lower_bound(self, correct: Sequence[int]) -> float:
        return B1._lb(list(correct))


def build_cache(smoke: int = 0, gold: bool = True) -> None:
    """Pre-compute and cache every encoder's embeddings for the whole suite (one model loaded at a time).
    With `gold`, also pre-cache the disjoint gold embeddings (tfidf_lsa gold is computed on demand, cheaply)."""
    arena = TwentyNewsArena(smoke=smoke)
    for enc in REGISTRY:
        for task in arena.tasks:
            arena._emb(enc.tag, task)
            if gold and enc.tag != "tfidf_lsa":
                arena._emb_gold(enc.tag, task)
            print(f"cached {enc.tag:10s} {task}", flush=True)
        _ST_CACHE.pop(enc.tag, None)   # free the model before loading the next (CPU RAM is tight)


__all__ = ["TwentyNewsArena", "REGISTRY", "BASELINE_TAG", "PER_CLASS", "GOLD_PER_CLASS", "CACHE_DIR",
           "SUITE", "task_name", "build_cache"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="pre-build the embedding cache")
    ap.add_argument("--smoke", type=int, default=0, help="only the first N pairs")
    args = ap.parse_args()
    if args.build:
        build_cache(smoke=args.smoke)
