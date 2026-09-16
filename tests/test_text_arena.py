"""HERMETIC tests for the TEXT arena's honesty-critical plumbing (no network, no 20NG download, no
sentence-transformers weights). The full arena needs HuggingFace models + the 20-Newsgroups corpus, so it is
not a CI test; but the properties that make its measurements a valid, leak-free input to the frozen certifier
are pure and MUST be locked:

  (1) the sealed split is derived ONCE from the lexical BASELINE representation and reused VERBATIM for every
      encoder -> all McNemar pairing is on byte-identical sealed rows (only the representation differs);
  (2) measure() is strictly SELECT-THEN-BOUND: flipping ONLY the sealed-test labels inverts the sealed
      correctness vector elementwise while leaving the val-selected head's validation correctness identical --
      i.e. the tuned-GBM head is fit on train rows and selected on val, and sealed labels never leak in;
  (3) the arena's suite statistics ARE the real frozen functions (mcnemar_pvalue, benjamini_hochberg,
      science.clopper_pearson_lower), so the genuine certifier path decides every promotion.

A synthetic embedding map (monkeypatched in place of the sentence encoders) supplies fully-controlled,
deterministic vectors so the test is fast and offline. The autonomous-promotion invariant itself is already
locked, domain-agnostically, by tests/test_repr_researcher.py (same brain, same frozen stats)."""
import hashlib
import importlib.util
import os

import numpy as np

from vfplatform.battery import benjamini_hochberg, mcnemar_pvalue
from vectorforge.science import clopper_pearson_lower

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}

# Load the arena by file path; its own top-of-module sys.path.insert bootstraps the `scripts.*` namespace imports.
_PATH = os.path.join(_ROOT, "scripts", "repr_arena_text.py")
_spec = importlib.util.spec_from_file_location("repr_arena_text", _PATH)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

N_PER = 40   # 40 docs/class -> 80 rows, enough for a stable stratified split + a deterministic GBM head


def _make_arena(monkeypatch, strong=1.6):
    """A TwentyNewsArena whose encoder is a synthetic, deterministic, separable embedding map (no network).
    The baseline ('tfidf_lsa') is weakly separable; every neural tag is strongly separable -- so the split is
    well-defined and the head is learnable, while staying fully offline."""
    arena = T.TwentyNewsArena.__new__(T.TwentyNewsArena)
    arena.per_class = N_PER
    arena.cache_dir = "/tmp/_text_arena_test_cache_does_not_persist"
    arena._pairs = {"toy": ("cat_a", "cat_b")}
    arena.tasks = ["toy"]
    arena._docs = {"toy": (["doc"] * (2 * N_PER), np.array([0] * N_PER + [1] * N_PER))}
    arena._split = {}
    arena._meas = {}

    def fake_emb(tag, task):
        y = arena._docs[task][1]
        rng = np.random.RandomState(abs(hash(("emb", tag))) % (2 ** 31))
        X = rng.randn(len(y), 8).astype(np.float32)
        strength = 0.5 if tag == T.BASELINE_TAG else strong   # baseline weak, neural strong
        X[:, 0] += (y * 2 - 1) * strength
        return X

    monkeypatch.setattr(arena, "_emb", fake_emb)
    return arena


def test_frozen_certifier_unchanged():
    got = {rel: hashlib.sha256(open(os.path.join(_ROOT, rel), "rb").read()).hexdigest()[:8]
           for rel in FROZEN_EXPECTED}
    assert got == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED: {got} != {FROZEN_EXPECTED}"


def test_split_derived_from_baseline_and_reused_across_encoders(monkeypatch):
    """The sealed split is a function of the BASELINE representation alone, is disjoint+exhaustive, and is the
    SAME object for every encoder measured -> McNemar pairs challenger vs baseline on identical sealed rows."""
    arena = _make_arena(monkeypatch)
    sp = arena._task_split("toy")
    tr, val, test = set(sp["tr"]), set(sp["val"]), set(sp["test"])
    assert tr and val and test
    assert tr.isdisjoint(val) and tr.isdisjoint(test) and val.isdisjoint(test)
    assert tr | val | test == set(range(2 * N_PER))           # exhaustive partition of all rows

    # measuring different encoders must not re-derive or perturb the split: identical sealed rows for pairing.
    mb = arena.measure(T.BASELINE_TAG)["toy"]
    mn = arena.measure("minilm")["toy"]
    assert len(mb.sealed_correct) == len(test) == len(mn.sealed_correct)
    assert arena._task_split("toy")["test"] is sp["test"]      # cached, never recomputed per encoder

    # the split is reproducible from the baseline alone (a second fresh arena yields identical sealed ids).
    arena2 = _make_arena(monkeypatch)
    assert list(arena2._task_split("toy")["test"]) == list(sp["test"])


def test_measure_is_select_then_bound_sealed_labels_never_leak(monkeypatch):
    """Flip ONLY the sealed-test labels: the sealed correctness vector must invert elementwise while the
    val-selected head's validation correctness stays identical -> the head is fit on train + selected on val,
    and the sealed test is bound exactly once at the end (no leakage into training or model selection)."""
    arena = _make_arena(monkeypatch)
    sp = arena._task_split("toy")
    emb = arena._emb("minilm", "toy")
    y = sp["y"]

    vc1, tc1, acc1 = arena._fit_correct(emb, y, sp)
    y_flip = y.copy()
    y_flip[sp["test"]] = 1 - y_flip[sp["test"]]               # corrupt ONLY the sealed-test labels
    vc2, tc2, acc2 = arena._fit_correct(emb, y_flip, sp)

    assert len(tc1) == len(sp["test"]) and set(tc1) <= {0, 1}
    assert list(vc1) == list(vc2)                             # validation correctness unaffected by test labels
    assert list(tc2) == [1 - x for x in tc1]                  # sealed correctness inverts -> bound-only
    assert abs(acc1 - (1.0 - acc2)) < 1e-9


def test_arena_uses_real_frozen_statistics(monkeypatch):
    """The arena's mcnemar/bh/lower_bound delegate to the REAL frozen functions, byte-for-byte."""
    arena = _make_arena(monkeypatch)
    a = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    b = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    assert arena.mcnemar(a, b) == mcnemar_pvalue(a, b)
    pvals = [0.001, 0.2, 0.04, 0.5, 0.009]
    assert arena.bh(pvals, 0.1) == list(benjamini_hochberg(pvals, alpha=0.1))
    correct = [1] * 47 + [0] * 13
    # the arena's lower_bound delegates to benchmark_vision_transfer._lb (frozen CP bound, its ALPHA).
    assert arena.lower_bound(correct) == round(clopper_pearson_lower(47, 60, T.B1.ALPHA), 4)


# ---------------------------------------------------------------------------------------------------------
# GOLD (never-peeked) confirmation set: the disjointness + select-then-bound properties that make a single
# post-hoc gold read a VALID, leak-free confirmation must be locked exactly like the sealed plumbing above.
# ---------------------------------------------------------------------------------------------------------
class _FakeNG:
    def __init__(self, docs):
        self.data = docs


def _patch_corpus(monkeypatch, n_docs=400):
    """Patch fetch_20newsgroups with a deterministic offline corpus of UNIQUE, >=5-word docs per category, so
    the pure document-selection logic (_pair_docs vs _gold_pair_docs) is testable without any network."""
    import sklearn.datasets as skd

    def fake_fetch(subset, categories, remove, random_state):
        cat = categories[0]
        docs = [f"{cat} document number {i} alpha beta gamma delta" for i in range(n_docs)]
        return _FakeNG(docs)

    monkeypatch.setattr(skd, "fetch_20newsgroups", fake_fetch)


def test_gold_docs_are_disjoint_from_working_docs_and_are_the_same_shuffle_tail(monkeypatch):
    """The gold documents are the slice of the IDENTICAL deterministic shuffle that comes AFTER the first
    per_class -> they share zero rows with the train/val/sealed pool (no leakage) yet come from the same
    distribution. This is the property that makes a post-hoc gold read genuinely never-peeked."""
    _patch_corpus(monkeypatch, n_docs=400)
    per, gold = 40, 25
    tw, yw = T._pair_docs("cat_a", "cat_b", per)
    tg, yg = T._gold_pair_docs("cat_a", "cat_b", per, gold)

    assert len(tw) == 2 * per and len(tg) == 2 * gold
    assert set(tw).isdisjoint(set(tg))                       # gold never overlaps the working rows
    assert len(set(tw)) == 2 * per and len(set(tg)) == 2 * gold
    # working + gold are exactly the first (per+gold) of each class's shared shuffle (a clean contiguous tail).
    assert len(set(tw) | set(tg)) == 2 * (per + gold)
    assert list(yg) == [0] * gold + [1] * gold


def test_gold_headroom_is_capped_honestly(monkeypatch):
    """If a category has fewer spare docs than requested, the gold set is capped at what actually exists
    (never reused working rows), so the realized gold n degrades honestly rather than leaking or crashing."""
    _patch_corpus(monkeypatch, n_docs=50)                    # only 50 docs/class
    tw, _ = T._pair_docs("cat_a", "cat_b", 40)               # working takes 40/class
    tg, _ = T._gold_pair_docs("cat_a", "cat_b", 40, 25)      # asks 25 but only 10 remain/class
    assert len(tg) == 2 * 10                                 # capped at the 10 spare per class
    assert set(tw).isdisjoint(set(tg))


def _make_gold_arena(monkeypatch, strong=1.6, n_gold=24):
    """Extend the synthetic arena with fully-controlled, separable GOLD embeddings (offline)."""
    arena = _make_arena(monkeypatch, strong)
    arena.gold_per_class = n_gold
    arena._gold_docs = {}
    arena._gold_meas = {}
    y_gold = np.array([0] * n_gold + [1] * n_gold)
    monkeypatch.setattr(arena, "_golddocs", lambda task: (["g"] * (2 * n_gold), y_gold))

    def fake_emb_pair(tag, task):
        Xf = arena._emb(tag, task)                           # working-row embeddings (separable)
        rng = np.random.RandomState(abs(hash(("gold", tag))) % (2 ** 31))
        Xg = rng.randn(2 * n_gold, 8).astype(np.float32)
        strength = 0.5 if tag == T.BASELINE_TAG else strong
        Xg[:, 0] += (y_gold * 2 - 1) * strength
        return Xf, Xg

    monkeypatch.setattr(arena, "_emb_pair", fake_emb_pair)
    return arena, y_gold


def test_gold_measure_is_select_then_bound_gold_labels_never_leak(monkeypatch):
    """The gold head is trained on the WORKING train rows and scored on the gold rows: flipping ONLY the gold
    labels inverts the gold correctness vector elementwise (the fit is unchanged), proving the gold set is
    bound exactly once at scoring and never enters training or selection -- a leak-free post-hoc confirmation."""
    arena, y_gold = _make_gold_arena(monkeypatch)
    sp = arena._task_split("toy")
    _, y = arena._pair("toy")
    Xf, Xg = arena._emb_pair("minilm", "toy")

    gc1 = arena._fit_gold_correct(Xf, Xg, y, y_gold, sp)
    gc2 = arena._fit_gold_correct(Xf, Xg, y, 1 - y_gold, sp)
    assert len(gc1) == len(y_gold) and set(gc1) <= {0, 1}
    assert list(gc2) == [1 - x for x in gc1]                 # gold correctness inverts -> bound-only at scoring


def test_gold_measure_returns_per_task_vectors_and_none_for_unknown(monkeypatch):
    """gold_measure yields a 0/1 correctness vector per task over the gold rows for a known encoder (or
    fuse[a+b]), and None for any tag outside the registry (so the brain emits no fabricated confirmation)."""
    arena, y_gold = _make_gold_arena(monkeypatch)
    g = arena.gold_measure("minilm")
    assert set(g.keys()) == {"toy"}
    assert len(g["toy"]) == len(y_gold) and set(g["toy"]) <= {0, 1}
    assert sum(g["toy"]) / len(g["toy"]) > 0.6               # the separable gold rows are mostly correct
    assert arena.gold_measure("not_a_real_encoder") is None
    assert T.TwentyNewsArena._parse_name("fuse[mpnet+minilm]") == ["mpnet", "minilm"]
    assert T.TwentyNewsArena._parse_name("e5_base") == ["e5_base"]
