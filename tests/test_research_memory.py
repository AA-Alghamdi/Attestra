"""Tests for vfplatform.research_memory -- the durable learning substrate adapter."""
import json
import os

import pytest

from vfplatform import casebase_store
from vfplatform.research_memory import ResearchMemory, family_of


# --- helpers ---
class _FakeMove:
    def __init__(self, name, prior_gain=0.01, prior_cost=1.0):
        self.name = name
        self.prior_gain = prior_gain
        self.prior_cost = prior_cost


def _store(tmp_path):
    return casebase_store.CaseBaseStore(
        outcome_path=str(tmp_path / "outcomes.jsonl"),
        negative_path=str(tmp_path / "negative.jsonl"))


def _fp(kind="tabular", task_type="binary", n=1000, d=10, nc=2):
    return casebase_store.fingerprint(
        {"n_rows": n, "n_features": d, "n_classes": nc, "kind": kind, "task_type": task_type},
        kind=kind, task_type=task_type)


# --- tests ---

def test_family_of_grid_move():
    assert family_of(_FakeMove("hist_gbm|n=100")) == "hist_gbm"
    assert family_of(_FakeMove("knn|k=3|metric=l2")) == "knn"
    assert family_of(_FakeMove("baseline")) == "baseline"


def test_cold_start_uses_catalog_prior(tmp_path):
    """With no store, ResearchMemory returns catalog priors (byte-identical to plain CaseBase)."""
    mem = ResearchMemory(store=None, fingerprint=None, warm=None)
    mv = _FakeMove("knn|k=5", prior_gain=0.05, prior_cost=2.0)
    assert mem.expected_gain(mv) == 0.05
    assert mem.expected_cost(mv) == 2.0


def test_warm_start_uses_cross_dataset_prior(tmp_path):
    """When a store has realized gains from a similar dataset, ResearchMemory uses them."""
    store = _store(tmp_path)
    fp = _fp(n=800, d=12)

    # simulate 3 past outcomes for hist_gbm on a similar dataset
    for _ in range(3):
        store.record_outcome(fp, "hist_gbm", gain=0.15, cost=3.0, device="cpu")
    store.record_outcome(fp, "logistic", gain=0.001, cost=0.3, device="cpu")  # dead end

    # build ResearchMemory for a SIMILAR dataset (same fingerprint class)
    fp2 = _fp(n=900, d=12)  # same buckets → distance ~0
    mem = ResearchMemory.build(fp2, store=store, kind="tabular", task_type="binary", device="cpu")

    mv_gbm = _FakeMove("hist_gbm|n=200", prior_gain=0.01, prior_cost=1.0)
    mv_log = _FakeMove("logistic|C=1.0", prior_gain=0.01, prior_cost=0.5)
    mv_cold = _FakeMove("svm|C=1.0", prior_gain=0.01, prior_cost=1.0)

    # hist_gbm should use the warm prior (~0.15), NOT the cold catalog prior (0.01)
    assert mem.expected_gain(mv_gbm) > 0.1
    # logistic is a dead end → below cold prior
    assert mem.expected_gain(mv_log) < mv_cold.prior_gain
    # svm has no history → cold catalog prior
    assert mem.expected_gain(mv_cold) == 0.01


def test_in_run_realized_dominates(tmp_path):
    """Once a move runs in THIS run, its in-run realized history takes precedence over warm-start."""
    store = _store(tmp_path)
    fp = _fp()
    store.record_outcome(fp, "hist_gbm", gain=0.20, cost=3.0)
    mem = ResearchMemory.build(fp, store=store, kind="tabular", task_type="binary")

    mv = _FakeMove("hist_gbm|n=200", prior_gain=0.01, prior_cost=1.0)
    assert mem.expected_gain(mv) > 0.1  # warm prior

    # simulate an in-run observation (the loop calls casebase.record)
    mem.record("hist_gbm|n=200", 0.0)  # realized 0 this run
    assert mem.expected_gain(mv) == 0.0  # in-run realized dominates


def test_observe_writes_durable(tmp_path):
    """observe() writes the outcome to the durable store with device, and it persists."""
    store = _store(tmp_path)
    fp = _fp()
    mem = ResearchMemory(store=store, fingerprint=fp, device="cuda")
    mem.observe("torch_mlp", 0.12, 5.0)

    # read back from store
    outcomes = casebase_store._read_jsonl(str(tmp_path / "outcomes.jsonl"))
    assert len(outcomes) == 1
    assert outcomes[0]["family"] == "torch_mlp"
    assert outcomes[0]["gain"] == 0.12
    assert outcomes[0]["device"] == "cuda"
    assert outcomes[0]["fp_key"] == fp["key"]


def test_device_recorded_in_outcome(tmp_path):
    """CaseBaseStore.record_outcome includes the device field and it shows up in summary."""
    store = _store(tmp_path)
    fp = _fp()
    store.record_outcome(fp, "knn", 0.05, 1.0, device="cpu")
    store.record_outcome(fp, "torch_mlp", 0.15, 4.0, device="cuda")
    store.record_outcome(fp, "torch_cnn", 0.12, 6.0, device="cuda")

    s = store.summary()
    assert s["by_device"]["cpu"] == 1
    assert s["by_device"]["cuda"] == 2
    assert s["n_outcomes"] == 3
    assert s["n_datasets"] == 1


def test_negative_memory_avoids_dead_end(tmp_path):
    """A family that repeatedly dead-ends on similar data shows up in warm_start 'avoid' and is deprioritized."""
    store = _store(tmp_path)
    fp = _fp()
    # 5 dead ends for 'logistic' → strong negative signal
    for _ in range(5):
        store.record_outcome(fp, "logistic", gain=0.0, cost=0.5, device="cpu")

    mem = ResearchMemory.build(fp, store=store, kind="tabular", task_type="binary")
    assert "logistic" in mem.avoid

    mv = _FakeMove("logistic|C=0.1", prior_gain=0.05, prior_cost=0.5)
    # avoid family → gain strictly below cold prior
    assert mem.expected_gain(mv) < 0.05


def test_durable_across_invocations(tmp_path):
    """Destroy and recreate the store pointing at same files → data persists."""
    store = _store(tmp_path)
    fp = _fp()
    store.record_outcome(fp, "hist_gbm", gain=0.10, cost=2.0, device="cpu")
    del store

    # fresh store instance, same files
    store2 = _store(tmp_path)
    mem = ResearchMemory.build(fp, store=store2, kind="tabular", task_type="binary")
    assert mem.n_neighbors >= 1
    assert "hist_gbm" in mem.warm_families


def test_cross_dataset_transfer(tmp_path):
    """Outcomes from dataset A (1000 rows, 10 features, binary) transfer to dataset B (800 rows, 8 features,
    binary) because they share modality+n_classes+balance bucket (fingerprint distance < max_distance)."""
    store = _store(tmp_path)
    fp_a = _fp(n=1000, d=10, nc=2)
    store.record_outcome(fp_a, "extra_trees", gain=0.25, cost=3.0, device="cpu")
    store.record_outcome(fp_a, "svm", gain=0.0, cost=2.0, device="cpu")

    # Dataset B: different size (same order of magnitude) + slightly different feature count
    fp_b = _fp(n=800, d=8, nc=2)
    mem = ResearchMemory.build(fp_b, store=store, kind="tabular", task_type="binary")

    # extra_trees should transfer (positive gain) and svm should be in avoid (dead end)
    assert "extra_trees" in mem.warm_families
    assert mem.warm_families["extra_trees"]["prior_gain"] > 0.1
    assert "svm" in mem.avoid


def test_no_cross_modality_transfer(tmp_path):
    """A tabular outcome does NOT warm-start a vision dataset when max_distance is strict (modality
    mismatch alone contributes 3.0, plus other coordinate diffs pushes above a tight threshold)."""
    store = _store(tmp_path)
    fp_tab = _fp(kind="tabular", n=1000, d=10, nc=2)
    store.record_outcome(fp_tab, "hist_gbm", gain=0.20, cost=2.0)

    fp_vis = casebase_store.fingerprint(
        {"n_rows": 1000, "n_features": 512, "n_classes": 10, "kind": "vision"},
        kind="vision", task_type="multiclass")
    # Use a strict max_distance (3.0) so cross-modality (distance>3.0) is blocked
    mem = ResearchMemory.build(fp_vis, store=store, kind="vision", task_type="multiclass",
                               max_distance=3.0)
    # modality alone costs 3.0, plus other diffs → distance exceeds threshold
    assert "hist_gbm" not in mem.warm_families


def test_default_off_no_writes(tmp_path):
    """With memory_store=None, ResearchMemory.observe is a no-op and no files are created."""
    mem = ResearchMemory(store=None, fingerprint=None, device="cpu")
    mem.observe("hist_gbm", 0.15, 3.0)
    assert not os.path.exists(str(tmp_path / "outcomes.jsonl"))


def test_warm_summary(tmp_path):
    """warm_summary() returns introspectable metadata about what this run inherited."""
    store = _store(tmp_path)
    fp = _fp()
    store.record_outcome(fp, "hist_gbm", gain=0.15, cost=2.0, device="cuda")

    mem = ResearchMemory.build(fp, store=store, kind="tabular", task_type="binary", device="cuda")
    s = mem.warm_summary()
    assert s["device"] == "cuda"
    assert s["n_neighbors"] >= 1
    assert "hist_gbm" in s["warm_families"]
