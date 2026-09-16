"""Declarative experiment suite + orchestrator -- ONE manifest, ONE command, CPU now / GPU when provisioned.

This is the experiment BASIS for GPU runs: a JSON manifest enumerates a matrix of experiments (datasets x
goals x seeds x budgets), and `run_suite` executes every one through Attestra's real engines under the FROZEN
certifier, aggregating one machine-checkable report (a leaderboard + per-experiment certificate summary). It is
the reproducible campaign harness the autoresearcher runs the day a GPU is connected -- and it is verifiable on
CPU BEFORE any GPU spend.

Two lanes, ONE frozen certifier (vectorforge/science.py + vfplatform/sealed.py, byte-identical across lanes):

  lane="goal"  -- the autonomous /goal FRONT DOOR (vfplatform.goal_solver.solve): a free-text goal + a dataset
                  pointer -> a GoalCertificate (certified champion, honest non-promotion, or honest decline).
                  Pure CPU + deterministic. With `gpu: true` it also lets the front door recruit a torch-MLP
                  head (the same torch model code the GPU worker runs) as a first-class candidate.

  lane="loop"  -- the PROVIDER-AWARE harness engine (vfplatform.loop.run_goal_loop), which PROPOSES the GPU
                  torch families (torch_mlp / torch_cnn) whenever the selected provider can run them. On a GPU
                  box (RunPod key + endpoint, or a runpodctl pod) the torch fits run on cuda; with no GPU it
                  honestly falls back to the IN-PROCESS worker (LocalWorkerProvider) so the IDENTICAL remote
                  contract (JobSpec -> worker -> predictions -> local frozen certify) is exercised on CPU.

NOTHING here spends money on its own. Provider selection PROBES read-only (RunPod `myself`/health GraphQL +
a torch/cuda import check); a GPU job runs only when a real endpoint/pod is configured AND a loop-lane
experiment is requested. `gpu_preflight()` reports readiness + an honest per-hour cost estimate without
submitting anything.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_FILES = ("vectorforge/science.py", "vfplatform/sealed.py")
FROZEN_EXPECTED = {"vectorforge/science.py": "b564fba2", "vfplatform/sealed.py": "30ad6245"}


def frozen_hashes() -> Dict[str, str]:
    """The live sha256 (first 8 hex) of the FROZEN certifier core -- asserted byte-identical around a suite."""
    return {f: hashlib.sha256(open(os.path.join(_ROOT, f), "rb").read()).hexdigest()[:8] for f in FROZEN_FILES}


# ============================================================================ declarative experiment spec
@dataclass
class ExperimentSpec:
    """ONE experiment in a suite. `lane` selects the engine; the remaining fields are the knobs that engine
    needs. A manifest is a list of these (see experiments/*.json and load_suite)."""
    name: str
    lane: str = "goal"                 # "goal" (front door) | "loop" (provider-aware harness, GPU families)
    goal: str = ""                     # the free-text goal handed to the engine
    data: str = ""                     # dataset pointer (sklearn name / openml://id / npz / csv path)
    seed: int = 0
    peeks: int = 8                     # goal lane: sealed-peek budget
    gpu: bool = False                  # goal lane: allow the torch-MLP head candidate
    use_literature: bool = False       # goal lane: ground discovery in the literature scout (slower)
    # loop lane only ------------------------------------------------------------------------------------
    task_type: str = "multiclass"      # binary | multiclass | regression
    kind: str = "tabular"
    threshold: float = 0.5             # certification bar (theta)
    metric: Optional[str] = None       # None -> accuracy (clf) defaulted by the engine
    candidate_seeds: Tuple[int, ...] = (0, 1)
    max_rounds: int = 6
    objective: str = "maximize"        # loop lane: "maximize" explores capacity (incl. torch) for the best
    #                                    model -- the GPU-worthy regime; "certify" stops at the first model
    #                                    that clears theta (cheap, may never reach torch).
    note: str = ""

    def to_dict(self) -> dict:
        d = dict(vars(self))
        d["candidate_seeds"] = list(self.candidate_seeds)
        return d


def load_suite(path: str) -> Tuple[str, List[ExperimentSpec]]:
    """Load a suite manifest: {"suite": <name>, "experiments": [ {spec}, ... ]}. Unknown keys are ignored so
    a manifest can carry human-readable annotations alongside the typed fields."""
    with open(path) as fh:
        doc = json.load(fh)
    fields = set(ExperimentSpec.__dataclass_fields__.keys())
    specs: List[ExperimentSpec] = []
    for raw in doc.get("experiments", []):
        kw = {k: v for k, v in raw.items() if k in fields}
        if "candidate_seeds" in kw and kw["candidate_seeds"] is not None:
            kw["candidate_seeds"] = tuple(kw["candidate_seeds"])
        specs.append(ExperimentSpec(**kw))
    return doc.get("suite", os.path.basename(path)), specs


# ============================================================================ provider selection + preflight
def _torch_info() -> dict:
    if importlib.util.find_spec("torch") is None:
        return {"available": False, "version": None, "cuda": False}
    import torch
    return {"available": True, "version": str(torch.__version__),
            "cuda": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0}


def _weights_manifest() -> dict:
    path = os.path.join(_ROOT, "docs", "GPU_WEIGHTS_MANIFEST.json")
    if not os.path.exists(path):
        return {"present": False}
    try:
        with open(path) as fh:
            man = json.load(fh)
    except Exception:  # noqa: BLE001
        return {"present": True, "readable": False}
    total = man.get("total_gb")
    n = len(man.get("weights", man.get("files", []))) if isinstance(man, dict) else None
    return {"present": True, "readable": True, "total_gb": total, "n_entries": n}


def gpu_preflight() -> dict:
    """READ-ONLY GPU readiness report (NO spend, NO job submission). Surfaces, for each GPU lane, whether it
    is runnable today and -- if not -- the exact, honest reason + the cheapest unblock. Safe to run anywhere."""
    from .providers import RunPodProvider, RunPodPodProvider, LocalWorkerProvider, LocalCpuProvider
    from ._runpod import resolve_runpod_key

    key = resolve_runpod_key()
    serverless = RunPodProvider()
    serverless_caps = serverless.capabilities()
    serverless_probe = None
    if serverless.available():
        try:
            serverless_probe = serverless.probe()           # read-only health() GraphQL, no spend
        except Exception as ex:  # noqa: BLE001
            serverless_probe = {"available": False, "reason": f"probe failed: {str(ex)[:120]}"}

    pod = RunPodPodProvider()
    local_worker = LocalWorkerProvider()
    torch = _torch_info()

    # the local in-process worker can run the GPU torch families if torch imports (device-swaps cpu<->cuda)
    local_worker_torch = torch["available"]
    gpu_ready = bool((serverless_probe or {}).get("available")) or bool(pod.available())

    return {
        "ready_for_gpu": gpu_ready,
        "torch": torch,
        "weights_manifest": _weights_manifest(),
        "secrets": {
            "RUNPOD_API_KEY_or_keyfile": bool(key),
            "RUNPOD_ENDPOINT_ID": bool(os.environ.get("RUNPOD_ENDPOINT_ID")),
        },
        "providers": {
            "runpod-gpu": {"available": serverless.available(), "capabilities": serverless_caps,
                           "probe": serverless_probe},
            "runpod-pod": {"available": pod.available(), "capabilities": pod.capabilities()},
            "local-worker": {"available": local_worker.available(),
                             "capabilities": local_worker.capabilities(),
                             "runs_torch_families": local_worker_torch},
            "local-cpu": {"available": True, "capabilities": LocalCpuProvider().capabilities()},
        },
        "cpu_smoke_runnable": True,   # the loop lane always runs on local-worker (CPU) when no GPU is present
        "frozen_hashes": frozen_hashes(),
    }


def select_provider(prefer_gpu: bool) -> Tuple[object, dict]:
    """Pick the provider for a loop-lane run with HONEST fallback. prefer_gpu tries the real GPU providers
    first (serverless endpoint, then a runpodctl pod) and falls back to the in-process worker (CPU, same
    contract) when neither is connected -- it NEVER fakes a GPU run. Returns (provider, info)."""
    from .providers import RunPodProvider, RunPodPodProvider, LocalWorkerProvider, LocalCpuProvider

    chain: List[Tuple[object, str]] = []
    if prefer_gpu:
        sl = RunPodProvider()
        try:
            sl_ok = sl.available() and sl.probe().get("available", False)
        except Exception:  # noqa: BLE001
            sl_ok = False
        if sl_ok:
            chain.append((sl, "runpod serverless endpoint reachable"))
        pod = RunPodPodProvider()
        if pod.available():
            chain.append((pod, "runpod pod available"))
        # in-process worker exercises the IDENTICAL remote contract on CPU (torch families device-swap to cpu)
        chain.append((LocalWorkerProvider(), "no GPU connected -> in-process worker on CPU (honest fallback)"))
    else:
        chain.append((LocalWorkerProvider(), "CPU smoke via in-process worker (same JobSpec contract)"))
    chain.append((LocalCpuProvider(), "local CPU thread pool"))

    provider, reason = chain[0]
    info = {"chosen": provider.name, "reason": reason, "prefer_gpu": prefer_gpu,
            "device": provider.capabilities().get("device", "cpu"),
            "gated": provider.capabilities().get("gated", False),
            "considered": [p.name for p, _ in chain]}
    return provider, info


# ============================================================================ lane runners
def _records_from_xy(X: np.ndarray, y: np.ndarray) -> Tuple[List[dict], List[str]]:
    """(X, y) -> the loop's record format: {"features": {e0..eN: float}, "target": "<int>"}. Labels are the
    sorted distinct string targets (classification); the loop encodes them deterministically."""
    X = np.asarray(X, dtype=float)
    width = len(str(X.shape[1]))
    recs = [{"features": {f"e{str(k).zfill(width)}": float(v) for k, v in enumerate(row)},
             "target": str(int(yi)), "rid": i} for i, (row, yi) in enumerate(zip(X, y))]
    labels = sorted({r["target"] for r in recs}, key=lambda s: int(s))
    return recs, labels


def run_goal_spec(spec: ExperimentSpec, memory_store=None) -> dict:
    """Run a GOAL-lane experiment through the autonomous front door and summarize its GoalCertificate."""
    from . import goal_solver
    t0 = time.time()
    cert = goal_solver.solve(spec.goal, spec.data, peeks=spec.peeks, seed=spec.seed,
                             use_literature=spec.use_literature, gpu=spec.gpu,
                             memory_store=memory_store)
    champ = cert.champion_recipe or {}
    return {
        "name": spec.name, "lane": "goal", "data": spec.data, "goal": spec.goal,
        "elapsed_s": round(time.time() - t0, 2),
        "solved": cert.solved, "improved": cert.improved, "refused": cert.refused, "declined": cert.declined,
        "decline_reason": cert.decline_reason,
        "champion": cert.champion, "champion_head": champ.get("head"),
        "champion_has_code": bool(champ.get("code_patch")),
        "theta_floor": cert.theta_floor, "pooled_sealed_lb": cert.pooled_sealed_lb,
        "peeks_used": cert.peeks_used, "stop_reason": cert.stop_reason,
        "numeric_audit_ok": bool(cert.numeric_audit.get("single_source_of_truth", {}).get("agreement", False)
                                 and cert.numeric_audit.get("firewall_held", False)),
        "device": "cpu",
    }


def run_loop_spec(spec: ExperimentSpec, provider, provider_info: dict,
                  casebase_path: Optional[str] = None,
                  memory_store=None) -> dict:
    """Run a LOOP-lane experiment through the provider-aware harness engine. With a GPU provider the torch
    families (torch_mlp / torch_cnn) train on cuda; otherwise they device-swap to the in-process CPU worker.
    The frozen certifier scores either way.

    casebase_path: opt-in flat per-(kind,task_type) case-base (a JSON file). The loop warm-starts VoI from
    realized (val-gain / cost) and appends this experiment's outcomes back.

    memory_store: opt-in DURABLE, DATASET-AWARE memory (a casebase_store.CaseBaseStore). When given, the loop
    fingerprints this dataset by meta-features, warm-starts VoI from the per-family realized gains of the most
    SIMILAR past datasets (cross-dataset transfer), and records this run's outcomes (with device) back to the
    shared store. A campaign of experiments learns which families pay off, transfers that knowledge to NEW
    datasets, and tags every outcome by device (so GPU campaign gains are distinguishable and compound).
    casebase_path and memory_store compose: casebase_path is the flat in-run persistence, memory_store adds
    the richer cross-dataset tier on top. Default None == cold start == byte-identical to pre-memory."""
    from . import goal_solver
    from .harness import harness_for
    from .loop import run_goal_loop

    t0 = time.time()
    X, y, n_classes, task_hint = goal_solver.acquire(spec.data)
    records, labels = _records_from_xy(X, y)
    harness = harness_for(spec.kind, spec.task_type)
    res = run_goal_loop(
        records, spec.goal, harness=harness, kind=spec.kind, task_type=spec.task_type,
        target_key="target", labels=(labels if spec.task_type != "regression" else None),
        threshold=spec.threshold,
        metric=(spec.metric or ("accuracy" if spec.task_type != "regression" else "tolerance")),
        providers=[provider], seeds=spec.candidate_seeds, max_rounds=spec.max_rounds,
        objective=spec.objective, llm_enabled=False, llm_propose=False, seed=spec.seed,
        casebase_path=casebase_path, memory_store=memory_store)

    champ = res.winner
    champ_family = champ.family if champ is not None else None
    cert = res.certificate or {}
    families_ran = sorted({r.family for r in res.leaderboard.runs}) if res.leaderboard is not None else []
    return {
        "name": spec.name, "lane": "loop", "data": spec.data, "goal": spec.goal,
        "elapsed_s": round(time.time() - t0, 2),
        "decision": res.decision, "certified": bool(cert.get("certified", False)),
        "champion_family": champ_family,
        "champion_val": (round(float(champ.val_score), 4) if champ is not None else None),
        "sealed_lower_bound": cert.get("lower_bound"), "sealed_observed": cert.get("observed"),
        "n_sealed": res.n_test, "provider": res.provider,
        "device": provider_info.get("device", "cpu"),
        # the evidence that the GPU path was actually exercised (not just available): every family the engine
        # proposed AND fit this run, and whether any was a torch family (cuda on a GPU box, CPU worker here).
        "families_ran": families_ran, "n_runs": len(families_ran),
        "torch_ran": any("torch" in f for f in families_ran),
        "compute_report": res.compute_report,
    }


# ============================================================================ suite driver + report
def _casebase_path_for(memory_dir: str, spec: ExperimentSpec) -> str:
    """Per-(kind, task_type) case-base file inside the campaign memory dir. Moves are named by family, which
    is shared across DATASETS of the same modality+task -- so binary-tabular experiments warm-start each other
    while a regression campaign keeps its own (separate) realized-gain history (no cross-task contamination)."""
    return os.path.join(memory_dir, f"casebase_{spec.kind}_{spec.task_type}.json")


def _memory_summary(memory_dir: str) -> dict:
    """Read-only fold over the campaign memory: which moves accumulated realized val-gains, and the strongest
    learned levers. Pure reporting -- never feeds back into a run by itself."""
    out = {"memory_dir": memory_dir, "casebases": {}}
    if not os.path.isdir(memory_dir):
        return out
    for fn in sorted(os.listdir(memory_dir)):
        if not fn.startswith("casebase_") or not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(memory_dir, fn)) as fh:
                blob = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        gains = blob.get("gains", {}) if isinstance(blob, dict) else {}
        avg = {m: round(sum(v) / len(v), 4) for m, v in gains.items() if v}
        top = sorted(avg.items(), key=lambda kv: -kv[1])[:8]
        out["casebases"][fn] = {
            "n_moves": len(gains),
            "n_observations": sum(len(v) for v in gains.values()),
            "top_learned_moves": top,
        }
    return out


def run_suite(specs: List[ExperimentSpec], *, suite_name: str = "suite", lane: str = "auto",
              only: Optional[str] = None, out_path: Optional[str] = None,
              memory_dir: Optional[str] = None, on_event=None) -> dict:
    """Run every spec under the FROZEN certifier (asserted byte-identical before AND after) and return one
    aggregate report. `lane`: "cpu" forces CPU; "gpu" prefers GPU; "auto" prefers GPU and falls back honestly.
    `only` runs just the named experiment.

    memory_dir: opt-in DURABLE LEARNING SUBSTRATE for the loop lane. When set, every loop experiment in the
    campaign (a) warm-starts its flat per-(kind,task_type) VoI case-base from prior experiments in the same
    campaign, AND (b) fingerprints the dataset by meta-features and warm-starts from the realized gains of
    the most SIMILAR past datasets in a shared durable store (casebase_store.CaseBaseStore). A campaign thus
    learns which families pay off, transfers that knowledge to NEW datasets, and records every outcome
    tagged by device (so GPU campaign gains compound and are visible). Sequential within a suite =>
    deterministic and race-free. Default None == every run cold-starts (byte-identical to pre-memory)."""
    from . import casebase_store as _cs

    pre = frozen_hashes()
    assert pre == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED before suite: {pre} != {FROZEN_EXPECTED}"

    prefer_gpu = lane in ("gpu", "auto")
    provider, provider_info = select_provider(prefer_gpu)
    preflight = gpu_preflight()
    durable_store = None
    if memory_dir:
        os.makedirs(memory_dir, exist_ok=True)
        durable_store = _cs.CaseBaseStore(
            outcome_path=os.path.join(memory_dir, "outcomes.jsonl"),
            negative_path=os.path.join(memory_dir, "negative.jsonl"))

    chosen = [s for s in specs if (only is None or s.name == only)]
    results: List[dict] = []
    for s in chosen:
        if on_event:
            on_event({"event": "experiment_start", "name": s.name, "lane": s.lane})
        try:
            if s.lane == "loop":
                cb = _casebase_path_for(memory_dir, s) if memory_dir else None
                r = run_loop_spec(s, provider, provider_info, casebase_path=cb,
                                  memory_store=durable_store)
            else:
                r = run_goal_spec(s, memory_store=durable_store)
            r["ok"] = True
        except Exception as ex:  # noqa: BLE001  one bad experiment never aborts the campaign
            r = {"name": s.name, "lane": s.lane, "data": s.data, "ok": False,
                 "error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        results.append(r)
        if on_event:
            on_event({"event": "experiment_done", "name": s.name, "result": r})

    post = frozen_hashes()
    assert post == FROZEN_EXPECTED, f"FROZEN CERTIFIER CHANGED during suite: {post} != {FROZEN_EXPECTED}"

    report = {
        "suite": suite_name, "lane_requested": lane,
        "provider": provider_info, "preflight": preflight,
        "n_experiments": len(chosen), "results": results,
        "frozen_hashes": post, "frozen_ok": post == FROZEN_EXPECTED,
        "all_ok": all(r.get("ok") for r in results),
        "cross_experiment_memory": (_memory_summary(memory_dir) if memory_dir else None),
        "durable_memory": (durable_store.summary() if durable_store else None),
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        report["written_to"] = out_path
    return report


def leaderboard_markdown(report: dict) -> str:
    """A compact human leaderboard for the suite report (printed by the CLI, embeddable in a PR)."""
    lines = [f"# Experiment suite: {report['suite']}",
             "",
             f"- lane requested: `{report['lane_requested']}`  |  provider: "
             f"`{report['provider']['chosen']}` ({report['provider']['device']}) -- {report['provider']['reason']}",
             f"- GPU ready: **{report['preflight']['ready_for_gpu']}**  |  "
             f"torch={report['preflight']['torch']['available']} "
             f"cuda={report['preflight']['torch']['cuda']}",
             f"- frozen certifier byte-identical: **{report['frozen_ok']}** {report['frozen_hashes']}",
             ""]
    goal_rows = [r for r in report["results"] if r.get("lane") == "goal"]
    loop_rows = [r for r in report["results"] if r.get("lane") == "loop"]
    if goal_rows:
        lines += ["## /goal front-door lane", "",
                  "| experiment | data | verdict | champion (head) | theta | sealed_lb | peeks | audit |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in goal_rows:
            if not r.get("ok"):
                lines.append(f"| {r['name']} | {r.get('data', '')} | ERROR | {r.get('error', '')} | | | | |")
                continue
            verdict = ("declined" if r["declined"] else "refused" if r["refused"]
                       else "solved" if r["solved"] else "not-solved")
            head = r.get("champion_head") or ("code" if r.get("champion_has_code") else "-")
            lines.append(f"| {r['name']} | {r['data']} | {verdict} | {r.get('champion', '')} ({head}) | "
                         f"{r.get('theta_floor')} | {r.get('pooled_sealed_lb')} | {r.get('peeks_used')} | "
                         f"{r.get('numeric_audit_ok')} |")
        lines.append("")
    if loop_rows:
        lines += ["## GPU loop lane (torch families: torch_mlp / torch_cnn)", "",
                  "| experiment | data | decision | certified | champion family | sealed_lb | "
                  "torch fit | device |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in loop_rows:
            if not r.get("ok"):
                lines.append(f"| {r['name']} | {r.get('data', '')} | ERROR | {r.get('error', '')} | | | | |")
                continue
            lines.append(f"| {r['name']} | {r['data']} | {r.get('decision')} | {r.get('certified')} | "
                         f"{r.get('champion_family')} | {r.get('sealed_lower_bound')} | "
                         f"{r.get('torch_ran')} | {r.get('device')} |")
        lines.append("")
    mem = report.get("cross_experiment_memory")
    if mem and mem.get("casebases"):
        lines += ["## Cross-experiment memory (loop lane warm-starts VoI from prior experiments)", "",
                  f"memory dir: `{mem['memory_dir']}`", ""]
        for fn, info in mem["casebases"].items():
            top = ", ".join(f"`{m}`={g:+.3f}" for m, g in info["top_learned_moves"]) or "(none yet)"
            lines += [f"- **{fn}**: {info['n_moves']} moves, {info['n_observations']} observations learned",
                      f"  - top realized-gain levers: {top}"]
        lines.append("")
    return "\n".join(lines)
