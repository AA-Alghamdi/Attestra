"""HERMETIC tests for the TABULAR arena's honesty-critical plumbing (no network, no openml `letter` download,
no TabPFN -- the synthetic data + the GBM baseline tag exercise the real fit path entirely offline, so this runs
in the repo's MAIN env without the isolated `~/.venv-tabpfn` venv). The full arena needs the openml corpus and
TabPFN weights and is therefore not a CI test; but the properties that make its measurements a valid, leak-free
input to the frozen certifier are pure and MUST be locked:

  (1) the per-task split is a DETERMINISTIC class-balanced partition into train/val/sealed plus a DISJOINT
      never-peeked GOLD tail -- gold shares zero rows with the working pool (the property that makes a single
      post-hoc gold read genuinely multiplicity-free), and is reproducible across fresh arenas;
  (2) measure() is strictly SELECT-THEN-BOUND: the predictor is fit on train rows (GBM val-selected) and the
      sealed/gold rows are bound EXACTLY ONCE at scoring -- flipping ONLY the sealed (or gold) labels inverts
      that correctness vector elementwise while the fit is unchanged, so sealed/gold labels never leak in;
  (3) the arena's suite statistics ARE the real frozen functions (mcnemar_pvalue, benjamini_hochberg,
      science.clopper_pearson_lower), so the genuine certifier path decides every promotion.

The autonomous-promotion invariant itself (promotion rides on sealed labels alone) is already locked,
domain-agnostically, by tests/test_repr_researcher.py -- the SAME brain + frozen stats drive every modality."""
import hashlib
import importlib.util
import os

import numpy as np

from vfplatform.battery import benjamini_hochberg, mcnemar_pvalue
from vectorforge.science import clopper_pearson_lower

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}

# Load the arena by file path; its own top-of-module sys.path.insert bootstraps the `scripts.*`/`vfplatform.*`
# imports. Importing it pulls in sklearn (present in the main env); TabPFN is imported lazily inside _fit ONLY
# for the tabpfn tags, so a GBM-only test never touches it.
_PATH = os.path.join(_ROOT, "scripts", "repr_arena_tabular.py")
_spec = importlib.util.spec_from_file_location("repr_arena_tabular", _PATH)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

# Enough rows/class for the real split (train+val+sealed+gold) with margin, kept small so the GBM fits fast.
_PER_CLASS = T.TRAIN_PC + T.VAL_PC + T.SEALED_PC + T.GOLD_PC + 20


def _make_arena(sep=3.0, dim=16):
    """A LetterPairArena over a synthetic, deterministic, linearly-separable 2-class table (no network). Two
    'letters' A and B are drawn from offset Gaussians so the GBM head is learnable while the split machinery,
    gold disjointness and select-then-bound contract are fully controlled."""
    arena = T.LetterPairArena.__new__(T.LetterPairArena)
    arena.tasks = ["A_vs_B"]
    arena._pairs = {"A_vs_B": ("A", "B")}
    rng = np.random.RandomState(0)
    XA = rng.randn(_PER_CLASS, dim).astype(np.float32); XA[:, 0] += sep
    XB = rng.randn(_PER_CLASS, dim).astype(np.float32); XB[:, 0] -= sep
    X = np.concatenate([XA, XB])
    y = np.array(["A"] * _PER_CLASS + ["B"] * _PER_CLASS)
    arena._raw = (X, y)              # _load() returns this; no openml fetch
    arena._split = {}
    arena._meas = {}
    return arena


def test_frozen_certifier_unchanged():
    got = {rel: hashlib.sha256(open(os.path.join(_ROOT, rel), "rb").read()).hexdigest()[:8]
           for rel in FROZEN_EXPECTED}
    assert got == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED: {got} != {FROZEN_EXPECTED}"


def test_split_is_deterministic_disjoint_partition_with_disjoint_gold_tail():
    """train/val/sealed are sized exactly and disjoint, and the GOLD rows share zero examples with the whole
    working pool (train+val+sealed) -> a post-hoc gold read carries none of the climb's sealed multiplicity.
    The split is reproducible from the fixed shuffle alone (a second fresh arena yields byte-identical gold)."""
    arena = _make_arena()
    sp = arena._task_split("A_vs_B")
    assert len(sp["ytr"]) == 2 * T.TRAIN_PC
    assert len(sp["yva"]) == 2 * T.VAL_PC
    assert len(sp["yte"]) == 2 * T.SEALED_PC
    assert len(sp["ygo"]) == 2 * T.GOLD_PC

    work = np.concatenate([sp["Xtr"], sp["Xva"], sp["Xte"]])
    gold = sp["Xgo"]
    wb = {r.tobytes() for r in work}
    gb = {r.tobytes() for r in gold}
    assert len(wb) == len(work) and len(gb) == len(gold)      # every row distinct (no accidental reuse)
    assert wb.isdisjoint(gb)                                  # gold never overlaps the working pool

    arena2 = _make_arena()
    assert np.array_equal(arena2._task_split("A_vs_B")["Xgo"], gold)   # reproducible never-peeked tail


def test_measure_is_select_then_bound_sealed_labels_never_leak():
    """Fit the real GBM head on train rows; the fit consumes ONLY train/val (the _fit signature never receives
    the sealed labels). Flipping the sealed labels therefore inverts the sealed correctness vector elementwise
    while the validation correctness is unchanged -> the sealed test is bound exactly once, no leakage."""
    arena = _make_arena()
    sp = arena._task_split("A_vs_B")
    est = T.LetterPairArena._fit("gbm_raw", sp)               # fit on train, val-selected; sealed not passed in

    vc = (est.predict(sp["Xva"]) == sp["yva"]).astype(int)
    tc1 = (est.predict(sp["Xte"]) == sp["yte"]).astype(int)
    tc2 = (est.predict(sp["Xte"]) == (1 - sp["yte"])).astype(int)   # corrupt ONLY the sealed-test labels
    vc_after = (est.predict(sp["Xva"]) == sp["yva"]).astype(int)

    assert len(tc1) == 2 * T.SEALED_PC and set(tc1.tolist()) <= {0, 1}
    assert list(vc) == list(vc_after)                         # validation correctness unaffected by test labels
    assert list(tc2) == [1 - x for x in tc1]                  # sealed correctness inverts -> bound-only
    assert tc1.mean() > 0.8                                   # the separable head is genuinely accurate


def test_gold_measure_is_select_then_bound_gold_labels_never_leak():
    """The gold head is trained on the working train rows and scored on the disjoint gold rows: flipping ONLY
    the gold labels inverts the gold correctness vector elementwise (the fit is unchanged), proving the gold
    set is bound exactly once at scoring and never enters training or selection."""
    arena = _make_arena()
    sp = arena._task_split("A_vs_B")
    est = T.LetterPairArena._fit("gbm_raw", sp)
    gc1 = (est.predict(sp["Xgo"]) == sp["ygo"]).astype(int)
    gc2 = (est.predict(sp["Xgo"]) == (1 - sp["ygo"])).astype(int)
    assert len(gc1) == 2 * T.GOLD_PC and set(gc1.tolist()) <= {0, 1}
    assert list(gc2) == [1 - x for x in gc1]

    # gold_measure() returns a per-task 0/1 vector for a registry tag and for a fuse[a+b] key (offline GBM-only).
    g = arena.gold_measure("gbm_raw")
    assert set(g.keys()) == {"A_vs_B"}
    assert len(g["A_vs_B"]) == 2 * T.GOLD_PC and np.mean(g["A_vs_B"]) > 0.8
    gf = arena.gold_measure("fuse[gbm_raw+gbm_raw]")
    assert len(gf["A_vs_B"]) == 2 * T.GOLD_PC


def test_measure_and_fuse_return_aligned_sealed_vectors():
    """measure()/fuse_measure() yield a TaskMeasure per task whose sealed vector matches the sealed-row count and
    whose accuracy is the mean of that vector -- the exact object the brain pairs with McNemar."""
    arena = _make_arena()
    m = arena.measure("gbm_raw")["A_vs_B"]
    assert len(m.sealed_correct) == 2 * T.SEALED_PC
    assert abs(m.acc - float(np.mean(m.sealed_correct))) < 1e-9 and m.acc > 0.8
    f = arena.fuse_measure("gbm_raw", "gbm_raw")["A_vs_B"]      # soft-vote of a head with itself == itself
    assert len(f.sealed_correct) == 2 * T.SEALED_PC


def test_arena_uses_real_frozen_statistics():
    """mcnemar/bh/lower_bound delegate to the REAL frozen functions, byte-for-byte -- the genuine certifier path
    decides every promotion on this arena exactly as on vision/text."""
    arena = _make_arena()
    a = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    b = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    assert arena.mcnemar(a, b) == mcnemar_pvalue(a, b)
    pvals = [0.001, 0.2, 0.04, 0.5, 0.009]
    assert arena.bh(pvals, 0.1) == list(benjamini_hochberg(pvals, alpha=0.1))
    correct = [1] * 150 + [0] * 30
    assert arena.lower_bound(correct) == round(clopper_pearson_lower(150, 180, T.ALPHA), 4)
