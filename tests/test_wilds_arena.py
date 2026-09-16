"""Hermetic locks for the WILDS Camelyon17 distribution-shift arena (scripts/wilds_arena.py).

These prove the arena enforces the same governance the synthetic locks prove for the orchestrator, but on
the REAL distribution-shift carving:
  1. CARVING: train comes only from centers {0,3,4}; val only from center 1; sealed + gold only from
     center 2; sealed shards are disjoint; gold is disjoint from every sealed shard (never-peeked). Skips
     cleanly when the WILDS Camelyon17 download is absent (so the suite is portable).
  2. SELECT-THEN-BOUND: every recipe is scored on the IDENTICAL sealed rows (the arena's fixed shards); the
     head is fit on train features only and never sees a sealed label. Exercised with a monkeypatched
     feature extractor so it is fast and dataset-free.
  3. FROZEN HEAD HONESTY: a separable backbone yields a high sealed accuracy; a noise backbone collapses to
     ~chance -- the arena reports honest per-shard correctness, it does not leak the answer.
  4. CONTAMINATION FRAMING: the framing() probe carries the exact (X, y, train_idx, sealed_idx) the
     data-hygiene gate consumes; a forced near-duplicate straddle is detected by data_cert.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.wilds_arena as wa  # noqa: E402
from vfplatform import data_cert  # noqa: E402
from vfplatform.recipe import Recipe  # noqa: E402

WILDS_PRESENT = os.path.isdir(os.path.join(wa.DATA_ROOT, "camelyon17_v1.0"))


# ----------------------------------------------------------------------------- dataset-free fake splits
class _FakeSplits:
    """A stand-in for Camelyon17Splits exposing exactly the attributes the arena reads, so measure() /
    gold_measure() / framing() can be exercised without the 11GB download or any featurization."""

    def __init__(self, n_shards: int = 3, per: int = 40, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_shards = n_shards
        # global pool of row ids with a balanced label per id
        n = 200 + n_shards * per
        self.y = (np.arange(n) % 2).astype(int)
        ids = np.arange(n)
        self.train_idx = np.sort(ids[:120])
        self.val_idx = np.sort(ids[120:200])
        rest = ids[200:]
        self.shards = [np.sort(rest[s * per:(s + 1) * per]) for s in range(n_shards)]
        gold_start = 200 + n_shards * per
        self.gold_idx = np.sort(np.arange(gold_start, gold_start + 60)) if gold_start < n else \
            np.sort(rng.choice(self.train_idx, 0, replace=False))
        # ensure gold exists & is disjoint: extend y to cover gold ids
        if gold_start >= n:
            extra = np.arange(n, n + 60)
            self.y = np.concatenate([self.y, (extra % 2).astype(int)])
            self.gold_idx = np.sort(extra)
        self.tasks = [f"ood_shardv{i}" for i in range(n_shards)]
        self.ds = None  # never touched: extract_features is monkeypatched


def _install_fake_features(monkeypatch, *, separable_backbones=("dinov2",)):
    """Monkeypatch extract_features: a row id -> a deterministic feature vector. For a 'separable' backbone
    the feature encodes the label (so a linear head recovers it); otherwise it is label-independent noise."""
    captured = {"sealed_calls": []}

    def fake(backbone, splits, idx, **kw):
        idx = np.asarray(idx)
        # record any call that scores a sealed shard, to prove select-then-bound row identity
        for s, shard in enumerate(splits.shards):
            if idx.shape == shard.shape and np.array_equal(idx, shard):
                captured["sealed_calls"].append((backbone, s, tuple(idx.tolist())))
        # features are deterministic PER ROW ID (like the real disk cache, keyed by idx) so distinct rows
        # get distinct vectors and train != sealed -> no spurious straddle.
        sep = any(tag in backbone.lower() for tag in separable_backbones)
        bseed = abs(hash(backbone)) % (2**16)
        feats = []
        for row in idx:
            r = np.random.default_rng(bseed * 1_000_003 + int(row))
            v = r.normal(size=8).astype(np.float32)
            if sep:
                v[0] = float(splits.y[int(row)]) * 4.0 - 2.0 + float(r.normal(scale=0.1))
            feats.append(v)
        return np.stack(feats).astype(np.float32)

    monkeypatch.setattr(wa, "extract_features", fake)
    return captured


# ----------------------------------------------------------------------------- 1. real-data carving
@pytest.mark.skipif(not WILDS_PRESENT, reason="WILDS Camelyon17 not downloaded on this machine")
def test_carving_respects_centers_and_disjointness():
    sp = wa.Camelyon17Splits(n_train=120, n_val=80, n_test=120, n_gold=80, n_shards=3, seed=0)
    assert set(sp.center[sp.train_idx].tolist()) <= set(wa.TRAIN_CENTERS)
    assert set(sp.center[sp.val_idx].tolist()) == {wa.VAL_CENTER}
    sealed_all = np.concatenate(sp.shards)
    assert set(sp.center[sealed_all].tolist()) == {wa.TEST_CENTER}
    assert set(sp.center[sp.gold_idx].tolist()) == {wa.TEST_CENTER}
    # disjointness: train / val / sealed / gold never share a row
    pools = [set(sp.train_idx.tolist()), set(sp.val_idx.tolist()),
             set(sealed_all.tolist()), set(sp.gold_idx.tolist())]
    for i in range(len(pools)):
        for j in range(i + 1, len(pools)):
            assert pools[i].isdisjoint(pools[j])
    # shards are mutually disjoint
    for i in range(len(sp.shards)):
        for j in range(i + 1, len(sp.shards)):
            assert set(sp.shards[i].tolist()).isdisjoint(sp.shards[j].tolist())
    # balanced (both classes present in train so the trivial baseline cannot clear theta)
    assert set(sp.y[sp.train_idx].tolist()) == {0, 1}


# ----------------------------------------------------------------------------- 2. select-then-bound
def test_measure_scores_identical_sealed_rows_across_recipes(monkeypatch):
    cap = _install_fake_features(monkeypatch)
    sp = _FakeSplits()
    arena = wa.WildsCamelyonArena(sp)
    r_weak = Recipe(backbone="resnet18", adaptation="linear_probe", head="linear")
    r_strong = Recipe(backbone="vit_base_patch14_dinov2", adaptation="linear_probe", head="linear")

    m_weak = arena.measure(r_weak)
    m_strong = arena.measure(r_strong)

    # every task maps to a shard of the SAME length, and both recipes were scored on the identical rows
    for s, t in enumerate(sp.tasks):
        assert len(m_weak[t].sealed_correct) == len(sp.shards[s])
        assert len(m_strong[t].sealed_correct) == len(sp.shards[s])
    seen = {(s, rows) for (_, s, rows) in cap["sealed_calls"]}
    for s, shard in enumerate(sp.shards):
        assert (s, tuple(shard.tolist())) in seen, "a sealed shard was not scored on its fixed rows"
    # both recipes saw the same set of shard-rows (select-then-bound: rows are carved once, reused)
    weak_rows = {(s, rows) for (b, s, rows) in cap["sealed_calls"] if "resnet" in b}
    strong_rows = {(s, rows) for (b, s, rows) in cap["sealed_calls"] if "dinov2" in b}
    assert weak_rows == strong_rows


def test_separable_backbone_beats_noise_backbone(monkeypatch):
    _install_fake_features(monkeypatch)
    sp = _FakeSplits()
    arena = wa.WildsCamelyonArena(sp)
    strong = arena.measure(Recipe(backbone="vit_base_patch14_dinov2", head="linear"))
    weak = arena.measure(Recipe(backbone="resnet18", head="linear"))
    strong_acc = np.mean([strong[t].acc for t in sp.tasks])
    weak_acc = np.mean([weak[t].acc for t in sp.tasks])
    assert strong_acc > 0.9, strong_acc       # label-encoding backbone is recovered by the frozen head
    assert weak_acc < 0.75, weak_acc           # noise backbone cannot exceed chance materially
    # gold read uses a DISJOINT never-peeked set and returns one vector per task
    gold = arena.gold_measure(Recipe(backbone="vit_base_patch14_dinov2", head="linear"))
    assert set(gold) == set(sp.tasks)
    assert np.mean(gold[sp.tasks[0]]) > 0.9


# ----------------------------------------------------------------------------- 4. contamination framing
def test_framing_payload_is_clean_then_detects_forced_straddle(monkeypatch):
    _install_fake_features(monkeypatch)
    sp = _FakeSplits()
    arena = wa.WildsCamelyonArena(sp)
    fr = arena.framing()
    assert {"X", "y", "train_idx", "sealed_idx", "fit_predict_fn", "metric_fn"} <= set(fr)
    clean = data_cert.certify_dataset(fr["X"], fr["y"], train_idx=fr["train_idx"],
                                      sealed_idx=fr["sealed_idx"])
    assert clean.near_dup_straddle == 0, "clean cross-center features must not straddle"
    # force a leak: copy train rows verbatim into the sealed block -> data_cert must catch it
    X = fr["X"].copy()
    X[fr["sealed_idx"][:5]] = X[fr["train_idx"][:5]]
    leaked = data_cert.certify_dataset(X, fr["y"], train_idx=fr["train_idx"], sealed_idx=fr["sealed_idx"])
    assert leaked.near_dup_straddle >= 5
    assert leaked.passed is False
