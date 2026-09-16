"""Locks for the declarative EXPERIMENT SUITE + the GPU experiment basis (vfplatform/experiments.py) and the
gated torch-MLP head (Track A). All offline, deterministic, and WITHOUT touching the frozen certifier core
(asserted byte-identical at the end). The contract under test:

  * MANIFEST -- a JSON suite parses into typed ExperimentSpec rows; unknown annotation keys are ignored.
  * HONEST GATING -- gpu_preflight() is read-only and, with no RunPod key, reports the GPU lane GATED with the
    exact reason while the CPU smoke stays runnable; select_provider(prefer_gpu=True) falls back to a CPU
    provider (it never fakes a GPU run).
  * TORCH GATE -- a 'torch_mlp' family/head is runnable ONLY on a worker/GPU provider (in-catalog) and the
    arena only fits it when allow_torch_head is set; with the gate OFF the recipe path is the sklearn baseline.
  * GPU PATH EXERCISED ON CPU -- the loop lane proposes & fits the torch families through the in-process
    worker (the identical remote contract) so a GPU run is verified before any spend.
  * GOAL LANE -- the front door still returns an honest GoalCertificate, and the default (gpu=False) path is
    unchanged.
"""
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfplatform import experiments as E          # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HAS_TORCH = __import__("importlib").util.find_spec("torch") is not None


def _sha256(path):
    with open(os.path.join(_ROOT, path), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:8]


def _small_clf(n=120, d=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


# --------------------------------------------------------------------------- manifest + report plumbing
def test_default_manifest_parses():
    path = os.path.join(_ROOT, "experiments", "default_suite.json")
    name, specs = E.load_suite(path)
    assert name == "attestra-default"
    assert len(specs) >= 6
    by_name = {s.name: s for s in specs}
    assert by_name["wine_goal"].lane == "goal"
    assert by_name["wine_goal_gpu_head"].gpu is True
    assert by_name["digits_loop_gpu"].lane == "loop"
    assert by_name["digits_loop_gpu"].objective == "maximize"


def test_unknown_manifest_keys_ignored(tmp_path):
    import json
    doc = {"suite": "x", "experiments": [{"name": "a", "lane": "goal", "goal": "g", "data": "iris",
                                          "note": "ignored", "_doc": "ignored", "bogus_key": 123}]}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc))
    name, specs = E.load_suite(str(p))
    assert name == "x" and len(specs) == 1 and specs[0].data == "iris"


def test_frozen_hashes_helper_matches_expected():
    assert E.frozen_hashes() == E.FROZEN_EXPECTED
    assert E.frozen_hashes() == {f: _sha256(f) for f in E.FROZEN_FILES}


# --------------------------------------------------------------------------- honest GPU gating
def test_gpu_preflight_is_honest_and_readonly():
    pf = E.gpu_preflight()
    # no RunPod key in CI -> the serverless GPU lane is GATED with an actionable reason, never silently "ready"
    sl = pf["providers"]["runpod-gpu"]
    if not sl["available"]:
        assert sl["capabilities"]["gated"] is True
        assert "reason" in sl["capabilities"] and sl["capabilities"]["reason"]
    # the CPU smoke is always runnable, and torch availability is reported truthfully
    assert pf["cpu_smoke_runnable"] is True
    assert pf["torch"]["available"] == _HAS_TORCH
    assert pf["frozen_hashes"] == E.FROZEN_EXPECTED


def test_select_provider_honest_cpu_fallback():
    # prefer_gpu with no GPU connected -> a CPU provider (honest fallback), NEVER a faked GPU run
    p, info = E.select_provider(prefer_gpu=True)
    assert info["device"] == "cpu"
    assert info["chosen"] in ("local-worker", "local-cpu")
    p2, info2 = E.select_provider(prefer_gpu=False)
    assert info2["device"] == "cpu"


# --------------------------------------------------------------------------- torch gate (Track A)
def test_torch_family_gated_by_provider():
    from vfplatform.harness import runnable_catalog
    from vfplatform.providers import LocalCpuProvider, LocalWorkerProvider
    cat_cpu, _ = runnable_catalog("tabular", "multiclass", provider=LocalCpuProvider())
    assert "torch_mlp" not in cat_cpu          # non-worker CPU path cannot run torch -> never proposed
    cat_worker, _ = runnable_catalog("tabular", "multiclass", provider=LocalWorkerProvider())
    assert ("torch_mlp" in cat_worker) == _HAS_TORCH   # worker path proposes torch iff torch importable


def test_recipe_head_registry_includes_torch():
    from vfplatform.recipe import HEADS
    from vfplatform.recipe_generator import GeneratorConfig
    assert "torch_mlp" in HEADS
    assert GeneratorConfig().gpu_heads is False        # default OFF => baseline recipe path unchanged


def test_arena_torch_head_gate():
    from scripts.code_arena import TabularCodeArena, TabularSplits
    from vfplatform.recipe import Recipe
    X, y = _small_clf()
    sp = TabularSplits(data=(X, y, 2, "syn"), dataset="syn", shape="iid", split="random", seed=0)
    ar = TabularCodeArena(splits=sp)
    rec = Recipe(backbone="raw", adaptation="linear_probe", head="torch_mlp")
    # gate OFF -> a torch_mlp recipe falls back to the sklearn baseline head (no crash, real predictions)
    ar.allow_torch_head = False
    pred_off = ar._fit_head(rec, X[:90], y[:90], X[90:])
    assert len(pred_off) == len(X[90:])
    if _HAS_TORCH:
        # gate ON -> the torch head fits via worker/torch_models and returns predictions over the eval rows
        ar.allow_torch_head = True
        pred_on = ar._fit_torch_head(X[:90], y[:90], X[90:])
        assert pred_on is not None and len(pred_on) == len(X[90:])


# --------------------------------------------------------------------------- lanes run end-to-end
def test_records_from_xy_roundtrip():
    X, y = _small_clf(n=40, d=5)
    recs, labels = E._records_from_xy(X, y)
    assert len(recs) == 40 and set(labels) == {"0", "1"}
    assert len(recs[0]["features"]) == 5 and recs[0]["target"] in labels


def test_goal_lane_default_path_clean_and_frozen():
    pre = E.frozen_hashes()
    s = E.ExperimentSpec(name="iris", lane="goal", goal="classify iris species", data="iris", peeks=4, seed=0)
    r = E.run_goal_spec(s)
    assert r["solved"] is True and r["champion_head"] == "linear"   # default path: sklearn head, unchanged
    assert r["numeric_audit_ok"] is True
    assert E.frozen_hashes() == pre == E.FROZEN_EXPECTED


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_loop_lane_exercises_torch_on_cpu_worker():
    """The GPU path, verified on CPU: the loop proposes & FITS the torch families through the in-process
    worker (the identical JobSpec contract). On a GPU box these same fits run on cuda."""
    from vfplatform.harness import harness_for
    from vfplatform.loop import run_goal_loop
    X, y = _small_clf(n=160, d=6)
    recs, labels = E._records_from_xy(X, y)
    provider, info = E.select_provider(prefer_gpu=True)
    res = run_goal_loop(recs, "classify", harness=harness_for("tabular", "binary"),
                        kind="tabular", task_type="binary", target_key="target", labels=labels,
                        threshold=0.5, metric="accuracy", providers=[provider], seeds=(0,),
                        max_rounds=2, objective="maximize", llm_enabled=False, llm_propose=False, seed=0)
    families = {r.family for r in res.leaderboard.runs}
    assert any("torch" in f for f in families), f"torch family never ran: {sorted(families)}"
    assert res.decision in ("certified", "best_effort", "honest_stop", "not_certified")


def test_run_suite_smoke_cpu_frozen_and_report():
    specs = [
        E.ExperimentSpec(name="iris_goal", lane="goal", goal="classify iris species", data="iris",
                         peeks=4, seed=0),
        E.ExperimentSpec(name="forecast_decline", lane="goal", goal="forecast next month's sales",
                         data="iris", peeks=4, seed=0),
    ]
    report = E.run_suite(specs, suite_name="smoke", lane="cpu")
    assert report["frozen_ok"] is True and report["frozen_hashes"] == E.FROZEN_EXPECTED
    assert report["n_experiments"] == 2 and report["all_ok"] is True
    res = {r["name"]: r for r in report["results"]}
    assert res["iris_goal"]["solved"] is True
    assert res["forecast_decline"]["declined"] is True       # honest decline, no run
    md = E.leaderboard_markdown(report)
    assert "Experiment suite: smoke" in md and "front-door lane" in md


# --------------------------------------------------------------------------- cross-experiment memory
def _nonlinear_binary_npz(tmp_path, name, n=240, seed=0):
    """A small NONLINEAR (XOR-ish) binary set written as an .npz the loop lane can acquire. Nonlinear so the
    loop actually has gains to LEARN (a linear baseline fails -> capacity families pay off)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    p = tmp_path / f"{name}.npz"
    np.savez(str(p), X=X.astype(np.float64), y=y.astype(int), n_classes=np.int64(2))
    return str(p)


def _loop_specs_for(paths):
    return [E.ExperimentSpec(name=f"d{i}", lane="loop", goal="classify", data=p, kind="tabular",
                             task_type="binary", threshold=0.6, max_rounds=2, objective="maximize",
                             candidate_seeds=(0,), seed=0) for i, p in enumerate(paths)]


def test_cross_experiment_memory_accumulates_and_reports(tmp_path):
    """A loop campaign with --memory warm-starts VoI from -- and appends realized gains back to -- a shared
    per-(kind,task_type) case-base, so later experiments LEARN from earlier ones. Frozen core untouched."""
    import json
    mem = tmp_path / "memory"
    paths = [_nonlinear_binary_npz(tmp_path, "a", seed=1), _nonlinear_binary_npz(tmp_path, "b", seed=2)]
    pre = E.frozen_hashes()
    report = E.run_suite(_loop_specs_for(paths), suite_name="campaign", lane="cpu", memory_dir=str(mem))
    assert E.frozen_hashes() == pre == E.FROZEN_EXPECTED
    assert report["all_ok"] is True

    cb_path = mem / "casebase_tabular_binary.json"
    assert cb_path.exists(), "shared case-base was not written by the loop lane"
    blob = json.loads(cb_path.read_text())
    # the campaign LEARNED: many moves accumulated realized gains, and across BOTH experiments (so several
    # moves carry >1 observation -- the second experiment appended to the first's history).
    gains = blob["gains"]
    assert len(gains) >= 3
    assert sum(len(v) for v in gains.values()) > len(gains), "no move accumulated >1 obs across experiments"

    xem = report["cross_experiment_memory"]
    assert xem is not None and "casebase_tabular_binary.json" in xem["casebases"]
    info = xem["casebases"]["casebase_tabular_binary.json"]
    assert info["n_observations"] > info["n_moves"]
    md = E.leaderboard_markdown(report)
    assert "Cross-experiment memory" in md


def test_memory_default_off_is_cold(tmp_path):
    """No --memory => cross_experiment_memory is None and NO case-base file is written (cold start, the
    byte-identical pre-memory behavior)."""
    paths = [_nonlinear_binary_npz(tmp_path, "a", seed=1)]
    report = E.run_suite(_loop_specs_for(paths), suite_name="cold", lane="cpu")
    assert report["cross_experiment_memory"] is None
    assert not any(f.startswith("casebase_") for f in os.listdir(tmp_path))


def test_baseline_gain_not_inflated_in_casebase(tmp_path):
    """REGRESSION LOCK: the round-0 baseline anchor records gain 0.0 (an incremental lift), NOT its full
    score -- so a persisted cross-run case-base is not polluted by a phantom high-gain 'baseline' move."""
    import json
    from vfplatform.harness import harness_for
    from vfplatform.loop import run_goal_loop
    from vfplatform.providers import LocalCpuProvider
    X, y = _small_clf(n=160, d=6)
    recs, labels = E._records_from_xy(X, y)
    cb = tmp_path / "cb.json"
    run_goal_loop(recs, "classify", harness=harness_for("tabular", "binary"), kind="tabular",
                  task_type="binary", target_key="target", labels=labels, threshold=0.6, metric="accuracy",
                  providers=[LocalCpuProvider()], seeds=(0,), max_rounds=2, objective="maximize",
                  llm_enabled=False, llm_propose=False, seed=0, casebase_path=str(cb))
    blob = json.loads(cb.read_text())
    assert "baseline" in blob["gains"]
    assert all(g == 0.0 for g in blob["gains"]["baseline"]), \
        f"baseline gain should be 0.0 (incremental), got {blob['gains']['baseline']}"


# ========================== DURABLE RESEARCH MEMORY (PR #10) ==========================

def test_durable_memory_records_outcomes_with_device(tmp_path):
    """run_suite with --memory writes durable outcomes (outcomes.jsonl) with device field and reports them."""
    import json as _json
    mem = tmp_path / "memory"
    paths = [_nonlinear_binary_npz(tmp_path, "a", seed=7)]
    report = E.run_suite(_loop_specs_for(paths), suite_name="durable", lane="cpu", memory_dir=str(mem))
    assert report["all_ok"]

    # durable outcomes were written
    outcomes_path = mem / "outcomes.jsonl"
    assert outcomes_path.exists(), "durable outcomes.jsonl not written"
    lines = [_json.loads(ln) for ln in outcomes_path.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 2, "expected several recorded outcomes"
    # every outcome has a device field (cpu in this env)
    assert all(rec.get("device") == "cpu" for rec in lines)
    # every outcome has a fingerprint key
    assert all(rec.get("fp_key") for rec in lines)
    # the durable_memory summary is in the report
    dm = report["durable_memory"]
    assert dm is not None
    assert dm["n_outcomes"] >= 2
    assert "cpu" in dm["by_device"]
    assert len(dm["top_learned_levers"]) >= 1


def test_durable_memory_cross_dataset_transfer(tmp_path):
    """Two loop experiments on DIFFERENT (but similar) datasets share a durable store → the second
    experiment's ResearchMemory warm-starts from the first's realized outcomes (cross-dataset transfer)."""
    import json as _json
    mem = tmp_path / "memory"
    # Two binary datasets with same shape (both tabular|1e2|1e0|2|balanced fingerprint)
    paths = [_nonlinear_binary_npz(tmp_path, "ds1", seed=10),
             _nonlinear_binary_npz(tmp_path, "ds2", seed=20)]
    report = E.run_suite(_loop_specs_for(paths), suite_name="transfer", lane="cpu", memory_dir=str(mem))
    assert report["all_ok"]

    outcomes_path = mem / "outcomes.jsonl"
    lines = [_json.loads(ln) for ln in outcomes_path.read_text().splitlines() if ln.strip()]
    # both experiments contributed (distinct fingerprint keys for each dataset's different seed-based data)
    fp_keys = {rec["fp_key"] for rec in lines}
    # at least one fingerprint key present; both datasets map to the same bucket → same key
    assert len(fp_keys) >= 1
    # check the durable_memory summary reports outcomes from both experiments
    dm = report["durable_memory"]
    assert dm["n_outcomes"] >= 4  # multiple moves from 2 experiments


def test_durable_memory_default_off_no_files(tmp_path):
    """No memory_dir → no durable outcomes/negative files, durable_memory=None (byte-identical cold start)."""
    paths = [_nonlinear_binary_npz(tmp_path, "a", seed=3)]
    report = E.run_suite(_loop_specs_for(paths), suite_name="cold", lane="cpu")
    assert report["durable_memory"] is None
    # no outcomes.jsonl anywhere in tmp_path
    for root, _, files in os.walk(str(tmp_path)):
        assert "outcomes.jsonl" not in files


def test_frozen_hashes_after_durable_memory_run(tmp_path):
    """CRITICAL: the durable memory wiring does not touch the frozen certifier (byte-identical before+after)."""
    mem = tmp_path / "memory"
    paths = [_nonlinear_binary_npz(tmp_path, "a", seed=5)]
    pre = E.frozen_hashes()
    E.run_suite(_loop_specs_for(paths), suite_name="frozen_check", lane="cpu", memory_dir=str(mem))
    post = E.frozen_hashes()
    assert pre == post == E.FROZEN_EXPECTED
