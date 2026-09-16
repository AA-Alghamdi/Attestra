"""Execution substrate: ONE backend-agnostic Executor.run(Job) -> RunResult seam.

Implements design 02 (`frontier/core/design/02_execution_substrate.md`) as a single module
(the assignment constrains the CORE execution work to this one file; the design's per-concern
files — job/fitspec/probe/backends/executor/sandbox2 — are realized here as cohesive sections).

# === WIRING =================================================================================
#
# WHAT THIS IS
#   The Phase-0 spine runs untrusted candidate code out-of-process via
#   `frontier.sandbox.run_program(...)`, which hardcodes a sklearn `est.fit/est.predict` child
#   and CPU-only execution. This module generalizes that into ONE declarative seam:
#
#       Executor.run(Job) -> RunResult
#
#   A `Job` bundles (Program, train/eval arrays, task kind, a chosen BackendSpec, ResourceLimits,
#   seed, label vocab). A `BackendSpec` declaratively names (a) the entrypoint the authored code
#   must define ("build_estimator" for sklearn, "build_module" for torch), (b) the import names
#   whose availability defines runnability, (c) the child-side runner source. An `Executor`
#   decides WHERE the child runs: `LocalSubprocessExecutor` (this host, exactly the Phase-0
#   Popen+rlimits+timeout+killpg behavior) or `RemotePodExecutor` (a GPU pod, identical bundle +
#   runner + OK/ERR contract). GPU is therefore a SUBSTRATE SWAP, not a rewrite.
#
# HOW THE INTEGRATOR COMPOSES IT INTO THE SPINE
#   The engine today calls `frontier.sandbox.run_program(prog, Xtr, ytr, Xev, kind=...)`. The
#   integrator (core/orchestrator.py, design 04) instead calls the back-compat shim here:
#
#       from frontier.core.execution import run_program, SKLEARN_SPEC, LocalSubprocessExecutor
#       res = run_program(prog, Xtr, ytr, Xev, kind=kind, backend=SKLEARN_SPEC)   # == Phase-0
#
#   With no `backend` kwarg the result is BIT-FOR-BIT identical to `frontier.sandbox.run_program`
#   because the sklearn child runner here is `frontier.sandbox._RUNNER` imported VERBATIM and the
#   CLI invocation is the same 4-arg form. `frontier/sandbox.py` is NOT edited; `test_spine.py`
#   stays green. To run a torch-tagged Program the integrator passes `backend=TORCH_SPEC`; on this
#   machine (no torch) that returns `error_kind="backend_unavailable"` and the arm declines
#   HONESTLY — it is never silently rerouted to sklearn (a different model => a different
#   certificate => a faked result). To move to GPU the integrator injects a `RemotePodExecutor`
#   built from a vfplatform provider; nothing else changes.
#
# FIREWALL (preserved across local and remote, sklearn and torch)
#   The child returns ONLY predictions (an object .npy: string labels for clf, floats for reg) plus
#   a typed status line. It computes NO metric. The trusted parent scores via certify.score_val and
#   certifies the winner via certify.certify_on_sealed -> vectorforge.science / vfplatform.sealed.
#   The certifier cannot tell a torch winner from a sklearn one — that is the point. This module
#   imports neither torch nor an LLM in the parent; torch lives only inside the (out-of-process)
#   torch child runner string and behind the capability probe.
#
# CONCURRENCY IS OUT OF SCOPE HERE (by design 02 §5)
#   `Executor.run` is a single blocking call, reentrant, sharing no module-level mutable state
#   (each call gets its own TemporaryDirectory / pod). A separate portfolio module owns bounded
#   concurrency, GPU/CPU slot accounting, and early-kill; it consumes this single-job seam.
#
# HONEST DEGRADATION (design 02 §6, and the project's "never fake a GPU run" rule)
#   `probe_backend` runs the import check in a THROWAWAY subprocess (a segfaulting torch build
#   cannot crash the parent) and records the TRUE device. Locally: sklearn runnable, torch not
#   ("missing modules: ['torch']"). A CPU-torch run is runnable but `meta.json["device"]=="cpu"`
#   so a "GPU run" claim is always checkable. The probe never imports heavy deps into the parent.
# ===========================================================================================
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

# Repo root on sys.path so `frontier.*`, `vectorforge.*`, `vfplatform.*` import the same way the
# Phase-0 modules do (mirrors frontier/certify.py and frontier/tests/test_spine.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier.program import Program, RunResult  # noqa: E402  (frozen Phase-0 types)
from frontier import sandbox as _phase0_sandbox    # noqa: E402  (for the VERBATIM sklearn runner)


# ============================================================================================
# Section 1 — declarative job + resource + fit specs (design 02 §1, §2, §7.1)
# ============================================================================================

@dataclass(frozen=True)
class ResourceLimits:
    """Resource envelope for one child run. POSIX rlimits + a parent-enforced wall timeout.

    `gpu_mem_fraction` is honored only on the torch/CUDA path (via
    `torch.cuda.set_per_process_memory_fraction` inside the child); `None` means no cap. On the
    sklearn path it is ignored, keeping the Phase-0 contract intact.
    """
    wall_seconds: float = 60.0
    cpu_seconds: int = 55
    address_mb: int = 4096
    gpu_mem_fraction: Optional[float] = None


@dataclass(frozen=True)
class FitSpec:
    """Unified declarative training intent consumed by BOTH backends (design 02 §7.1).

    A proposer/LLM writes ONE training intent; the substrate maps it to whichever backend runs.
    sklearn ignores the neural-only fields; torch ignores `sklearn_overrides`. The defaults are a
    documented literature-standard recipe (AdamW + cosine + early stop + AMP-on-CUDA) used as a
    FALLBACK only — explicitly NOT tuned to any target benchmark (scientific-integrity rule).

    The child CLAMPS candidate-supplied values (epochs/batch_size) so an over-ambitious recipe
    cannot blow the wall budget into a `timeout`; the early-stop snapshot means best-so-far weights
    are used even if training is cut short.
    """
    seed: int = 0
    early_stop: bool = True
    # neural-only (ignored by sklearn)
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 1e-4
    amp: bool = True
    patience: int = 10
    grad_clip: Optional[float] = 1.0
    # sklearn-only (ignored by torch)
    sklearn_overrides: dict = field(default_factory=dict)


BackendTag = str  # "sklearn" | "torch"


@dataclass(frozen=True)
class BackendSpec:
    """Declarative description of one execution backend. Lives in the trusted parent.

    The parent uses this only to (a) decide if the backend is runnable here (probe), (b) pick
    resource defaults, (c) stamp the child invocation with `tag`. It does NOT import the backend's
    heavy deps (torch); all heavy work is child-side.
    """
    tag: BackendTag
    entrypoint: str                          # symbol the authored code must define
    requires: tuple                          # import names whose availability == runnability
    optional: tuple = ()                     # e.g. ("cuda",): prefers a GPU, can run CPU
    cuda_required: bool = False              # if True, no-CUDA => not runnable (declines)
    defaults: Optional[ResourceLimits] = None
    note: str = ""


@dataclass
class Job:
    """Everything one child run needs. Serialized to a temp dir identically local and remote."""
    program: Program
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    kind: str                                # "classification" | "regression"
    backend: BackendSpec
    limits: ResourceLimits
    seed: int = 0
    labels: Optional[tuple] = None           # sorted class-label vocab for clf, else None
    fit_spec: Optional[FitSpec] = None       # None => backend default (a documented fallback)


# ============================================================================================
# Section 2 — capability probe: detect backends, degrade honestly (design 02 §6)
# ============================================================================================

@dataclass(frozen=True)
class Capability:
    """Result of a real, out-of-process probe. `reason` is the honest decline explanation."""
    tag: BackendTag
    runnable: bool
    cuda: bool = False
    devices: tuple = ()                      # e.g. ("NVIDIA A100",)
    reason: str = ""
    interpreter: str = ""


# Cache per (interpreter, tag) so repeated runs in one process do not re-spawn the probe.
_PROBE_CACHE: dict = {}


def probe_backend(spec: BackendSpec, *, python: str = sys.executable,
                  use_cache: bool = True) -> Capability:
    """Answer "can this backend run here, right now?" without importing heavy deps in-parent.

    Runs the import check in a THROWAWAY subprocess so a segfaulting/broken backend build cannot
    crash the parent, and records the TRUE device (so a GPU claim is always checkable). Caches per
    (interpreter, tag). Never fakes: a missing module yields `runnable=False` with the honest
    reason, and the engine declines that arm rather than substituting another backend.
    """
    key = (python, spec.tag)
    if use_cache and key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    src = (
        "import json\n"
        "import importlib.util as _u\n"
        f"req={list(spec.requires)}; opt={list(spec.optional)}\n"
        "miss=[m for m in req if _u.find_spec(m) is None]\n"
        "cuda=False; devs=[]\n"
        "if not miss and 'cuda' in opt:\n"
        "    try:\n"
        "        import torch\n"
        "        cuda=bool(torch.cuda.is_available())\n"
        "        devs=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]\n"
        "    except Exception:\n"
        "        pass\n"
        "print(json.dumps({'miss':miss,'cuda':cuda,'devs':devs}))\n"
    )
    try:
        out = subprocess.run([python, "-c", src], capture_output=True, text=True, timeout=60)
        line = (out.stdout or "").strip().splitlines()
        if not line:
            cap = Capability(spec.tag, runnable=False,
                             reason=f"probe produced no output (stderr: {(out.stderr or '')[:120]})",
                             interpreter=python)
            if use_cache:
                _PROBE_CACHE[key] = cap
            return cap
        info = json.loads(line[-1])
    except Exception as e:  # noqa: BLE001 — probe failure is itself an honest "not runnable"
        cap = Capability(spec.tag, runnable=False, reason=f"probe failed: {e}",
                         interpreter=python)
        if use_cache:
            _PROBE_CACHE[key] = cap
        return cap

    if info["miss"]:
        cap = Capability(spec.tag, runnable=False,
                         reason=f"missing modules: {info['miss']}", interpreter=python)
    elif spec.cuda_required and not info["cuda"]:
        cap = Capability(spec.tag, runnable=False, reason="CUDA required but unavailable",
                         interpreter=python)
    else:
        cap = Capability(spec.tag, runnable=True, cuda=bool(info["cuda"]),
                         devices=tuple(info["devs"]), interpreter=python)
    if use_cache:
        _PROBE_CACHE[key] = cap
    return cap


# ============================================================================================
# Section 3 — child runner sources (parent <-> child contract, design 02 §2, §3)
# ============================================================================================
#
# SKLEARN: reuse the FROZEN Phase-0 runner VERBATIM. Importing the module-level string guarantees
# byte-for-byte identical child behavior (and identical predictions) to frontier.sandbox. The
# sklearn child takes the Phase-0 4-arg CLI: `runner.py job.npz candidate.py preds.npy <kind>`.
_SKLEARN_CHILD_RUNNER = _phase0_sandbox._RUNNER

# TORCH: a self-contained nn.Module train/eval child. Runs ONLY in the out-of-process child; the
# parent never imports torch. Extended 6-arg CLI: `runner.py job.npz candidate.py preds.npy
# <kind> <backend_tag> meta.json`. Properties enforced (design 02 §3):
#   - returns predictions in the EXACT format the sklearn child returns (string labels / floats),
#     so certify.py is backend-agnostic;
#   - computes NO metric (early-stop loss is a TRAINING signal on a TRAIN-internal slice only);
#   - NaN/Inf loss or predictions => typed `diverged` failure, never silently-zeroed predictions;
#   - CUDA OOM => typed `cuda_oom` (distinct from host `oom`);
#   - clamps candidate epochs/batch_size to a sane ceiling so a recipe cannot blow the wall budget;
#   - records the TRUE device + epochs_ran + best early-stop loss into meta.json (honest provenance).
_TORCH_CHILD_RUNNER = r'''
import sys, json
import numpy as np

job, codep, outp, kind, tag, metap = sys.argv[1:7]

def emit(line):
    print(line)
    sys.stdout.flush()

# Last line of defense behind the parent's probe (e.g. a pod image missing torch).
try:
    import torch
    import torch.nn as nn
except Exception as e:
    emit("ERR:import:torch not importable in child: %s" % (str(e)[:160],)); sys.exit(0)

try:
    d = np.load(job, allow_pickle=True)
    Xtr = d["Xtr"].astype("float32"); ytr = d["ytr"]; Xev = d["Xev"].astype("float32")
    seed = int(d["seed"][()]) if d["seed"].shape == () else int(d["seed"])
    labels = [str(x) for x in d["labels"]] if d["labels"].size else None
    spec_json = str(d["fit_spec"][()]) if d["fit_spec"].size else "{}"
    fit = json.loads(spec_json) if spec_json else {}
except Exception as e:
    emit("ERR:other:could not load job: %s" % (str(e)[:160],)); sys.exit(0)

# Deterministic seeding for every torch source of nondeterminism (design 02 §7.3).
torch.manual_seed(seed); np.random.seed(seed)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
try:
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True; cudnn.benchmark = False
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
frac = fit.get("gpu_mem_fraction", None)
if device == "cuda" and frac is not None:
    # RLIMIT_AS is unreliable with CUDA (the driver maps a huge virtual address space); GPU memory
    # is bounded here by set_per_process_memory_fraction + the container cgroup, host RAM by
    # RLIMIT_DATA where available. This substitution is documented, not hidden.
    try:
        torch.cuda.set_per_process_memory_fraction(float(frac))
    except Exception:
        pass

# Target encoding kept child-internal; predictions decoded back to the SAME dtype/labels the
# sklearn path returns, so certify.py stays backend-agnostic.
if kind == "classification":
    if labels is None:
        labels = sorted({str(v) for v in ytr})
    lab2idx = {l: i for i, l in enumerate(labels)}
    try:
        y = torch.tensor([lab2idx[str(v)] for v in ytr], dtype=torch.long)
    except KeyError as e:
        emit("ERR:other:unknown class label in train: %s" % (str(e)[:120],)); sys.exit(0)
    n_out = len(labels); loss_fn = nn.CrossEntropyLoss()
else:
    y = torch.tensor(ytr.astype("float32")).view(-1, 1)
    n_out = 1; loss_fn = nn.MSELoss()

# Build the authored module (untrusted code; only this child ever executes it).
ns = {}
try:
    exec(compile(open(codep).read(), "<candidate>", "exec"), ns)
except Exception as e:
    emit("ERR:build:compile/exec failed: %s" % (str(e)[:160],)); sys.exit(0)
if "build_module" not in ns or not callable(ns["build_module"]):
    emit("ERR:build:no callable build_module(n_features,n_outputs,kind)"); sys.exit(0)
try:
    model = ns["build_module"](Xtr.shape[1], n_out, kind).to(device)
except Exception as e:
    emit("ERR:build:%s: %s" % (type(e).__name__, str(e)[:160])); sys.exit(0)

# Optional candidate-supplied overrides, CLAMPED to a sane ceiling (cannot blow the wall budget).
try:
    cand = ns.get("fit_spec", lambda: {})()
    if isinstance(cand, dict):
        fit.update(cand)
except Exception:
    pass
epochs   = int(max(1, min(int(fit.get("epochs", 100)), 500)))
bs       = int(max(1, min(int(fit.get("batch_size", 256)), 4096)))
lr       = float(fit.get("lr", 1e-3))
wd       = float(fit.get("weight_decay", 1e-4))
patience = int(max(1, fit.get("patience", 10)))
clip     = fit.get("grad_clip", 1.0)
use_amp  = bool(fit.get("amp", True)) and device == "cuda"

opt_name = str(fit.get("optimizer", "adamw")).lower()
if opt_name == "sgd":
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
elif opt_name == "adam":
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
else:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

sched_name = str(fit.get("scheduler", "cosine")).lower()
if sched_name == "cosine":
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
elif sched_name == "onecycle":
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs))
else:
    sched = None

try:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=False)

Xtr_t = torch.tensor(Xtr)
gen = torch.Generator().manual_seed(seed)
n = len(y)
perm = torch.randperm(n, generator=gen)
n_es = max(1, int(0.1 * n)) if n > 1 else 0
es_idx = perm[:n_es]; tr_idx = perm[n_es:] if n_es < n else perm

def eval_loss(idx):
    if len(idx) == 0:
        return float("inf")
    model.eval()
    with torch.no_grad():
        out = model(Xtr_t[idx].to(device))
        return float(loss_fn(out, y[idx].to(device)).item())

best_state = None; best_es = float("inf"); bad = 0; epoch = 0
try:
    for epoch in range(epochs):
        model.train()
        eg = torch.Generator().manual_seed(seed + epoch)
        order = tr_idx[torch.randperm(len(tr_idx), generator=eg)]
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            xb = Xtr_t[b].to(device); yb = y[b].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(xb); l = loss_fn(out, yb)
            if not torch.isfinite(l):
                emit("ERR:diverged:non-finite loss at epoch %d" % epoch); sys.exit(0)
            scaler.scale(l).backward()
            if clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            scaler.step(opt); scaler.update()
        if sched is not None:
            sched.step()
        es = eval_loss(es_idx)
        if es < best_es - 1e-4:
            best_es = es; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Eval in batches so a huge eval split never OOMs.
    model.eval()
    Xev_t = torch.tensor(Xev)
    outs = []
    with torch.no_grad():
        for s in range(0, len(Xev_t), max(1, bs)):
            outs.append(model(Xev_t[s:s + bs].to(device)).cpu())
    logits = torch.cat(outs, dim=0) if outs else torch.zeros((0, n_out))

    if kind == "classification":
        pred_idx = logits.argmax(1).numpy()
        preds = np.array([labels[int(i)] for i in pred_idx], dtype=object)
        finite_ok = bool(np.isfinite(logits.numpy()).all())
    else:
        arr = logits.view(-1).numpy().astype(float)
        preds = arr.astype(object)
        finite_ok = bool(np.isfinite(arr).all())
    if not finite_ok:
        emit("ERR:diverged:non-finite predictions"); sys.exit(0)

    np.save(outp, np.asarray(preds, dtype=object), allow_pickle=True)
    try:
        json.dump({"device": device, "epochs_ran": int(epoch + 1),
                   "best_es": (None if best_es == float("inf") else float(best_es)),
                   "cuda": bool(device == "cuda")}, open(metap, "w"))
    except Exception:
        pass
    emit("OK")
except torch.cuda.OutOfMemoryError:
    emit("ERR:cuda_oom:CUDA out of memory")
except MemoryError:
    emit("ERR:oom:host MemoryError")
except Exception as e:
    emit("ERR:fit:%s: %s" % (type(e).__name__, str(e)[:160]))
'''


# ============================================================================================
# Section 4 — backend handles (design 02 §1)
# ============================================================================================

SKLEARN_SPEC = BackendSpec(
    tag="sklearn",
    entrypoint="build_estimator",
    requires=("numpy", "sklearn"),
    defaults=ResourceLimits(wall_seconds=60.0, cpu_seconds=55, address_mb=4096),
    note="CPU only; the Phase-0 verified path (sklearn child runner is byte-identical).",
)

TORCH_SPEC = BackendSpec(
    tag="torch",
    entrypoint="build_module",
    requires=("numpy", "torch"),
    optional=("cuda",),
    cuda_required=False,
    defaults=ResourceLimits(wall_seconds=900.0, cpu_seconds=900, address_mb=16384,
                            gpu_mem_fraction=0.9),
    note="needs importable torch; CUDA optional (CPU torch allowed but logged as slow).",
)


@runtime_checkable
class Backend(Protocol):
    """Parent-side handle: spec + a runtime probe + the child-side runner source."""
    spec: BackendSpec

    def available(self) -> Capability: ...
    def child_runner_source(self) -> str: ...
    def child_argv_tail(self, paths: "_BundlePaths", job: Job) -> list: ...


@dataclass
class SklearnBackend:
    """The Phase-0 verified path. Byte-identical child runner; 4-arg CLI."""
    spec: BackendSpec = SKLEARN_SPEC

    def available(self) -> Capability:
        return probe_backend(self.spec)

    def child_runner_source(self) -> str:
        return _SKLEARN_CHILD_RUNNER

    def child_argv_tail(self, paths: "_BundlePaths", job: Job) -> list:
        # Phase-0 CLI exactly: job.npz candidate.py preds.npy <kind>
        return [paths.job, paths.code, paths.preds, job.kind]


@dataclass
class TorchBackend:
    """Gated neural path. Declines honestly when torch is absent; 6-arg CLI with meta.json."""
    spec: BackendSpec = TORCH_SPEC

    def available(self) -> Capability:
        return probe_backend(self.spec)

    def child_runner_source(self) -> str:
        return _TORCH_CHILD_RUNNER

    def child_argv_tail(self, paths: "_BundlePaths", job: Job) -> list:
        return [paths.job, paths.code, paths.preds, job.kind, job.backend.tag, paths.meta]


def backend_for(spec: BackendSpec) -> Backend:
    """Map a BackendSpec to its parent-side handle (the only spec->handle dispatch point)."""
    if spec.tag == "sklearn":
        return SklearnBackend(spec)
    if spec.tag == "torch":
        return TorchBackend(spec)
    raise ValueError(f"no backend handle for tag {spec.tag!r}")


# ============================================================================================
# Section 5 — on-disk bundle (identical local and remote, design 02 §2, §4)
# ============================================================================================

@dataclass
class _BundlePaths:
    """Absolute paths of the files one child run reads/writes inside its temp dir."""
    root: str
    job: str
    code: str
    runner: str
    preds: str
    meta: str


def serialize_job_to_dir(job: Job, root: str) -> _BundlePaths:
    """Write the (job.npz, candidate.py, runner.py) bundle into `root`.

    SKLEARN: writes EXACTLY the Phase-0 npz (Xtr float / ytr object / Xev float) so the verbatim
    runner sees identical bytes => identical predictions. TORCH: adds kind/seed/labels/fit_spec so
    the child can deterministically encode/decode labels and consume the unified FitSpec.
    """
    paths = _BundlePaths(
        root=root,
        job=os.path.join(root, "job.npz"),
        code=os.path.join(root, "candidate.py"),
        runner=os.path.join(root, "runner.py"),
        preds=os.path.join(root, "preds.npy"),
        meta=os.path.join(root, "meta.json"),
    )
    Xtr = np.asarray(job.X_train, dtype=float)
    ytr = np.asarray(job.y_train, dtype=object)
    Xev = np.asarray(job.X_eval, dtype=float)
    if job.backend.tag == "sklearn":
        # Byte-compatible with frontier.sandbox.run_program (same keys, same dtypes, no extras).
        np.savez(paths.job, Xtr=Xtr, ytr=ytr, Xev=Xev)
    else:
        spec = job.fit_spec or FitSpec(seed=job.seed)
        spec_dict = {
            "epochs": spec.epochs, "batch_size": spec.batch_size, "lr": spec.lr,
            "optimizer": spec.optimizer, "scheduler": spec.scheduler,
            "weight_decay": spec.weight_decay, "amp": spec.amp, "patience": spec.patience,
            "grad_clip": spec.grad_clip,
            "gpu_mem_fraction": job.limits.gpu_mem_fraction,
        }
        labels_arr = np.asarray(list(job.labels), dtype=object) if job.labels else np.asarray([], dtype=object)
        np.savez(paths.job, Xtr=Xtr, ytr=ytr, Xev=Xev,
                 kind=np.asarray(job.kind), seed=np.asarray(int(job.seed)),
                 labels=labels_arr, fit_spec=np.asarray(json.dumps(spec_dict)))
    with open(paths.code, "w") as f:
        f.write(job.program.code)
    with open(paths.runner, "w") as f:
        f.write(backend_for(job.backend).child_runner_source())
    return paths


def _parse_status_line(out: str) -> str:
    """The last non-empty stdout line is the OK/ERR source of truth (Phase-0 convention)."""
    s = (out or "").strip()
    return s.splitlines()[-1] if s else ""


def parse_runner_result(job: Job, *, status: str, stderr: str, returncode: Optional[int],
                        paths: _BundlePaths, wall: float) -> RunResult:
    """Turn (status line, stderr, returncode, preds.npy, meta.json) into a RunResult.

    IDENTICAL parser local and remote — the firewall and error taxonomy are wire-agnostic.
    On success attaches the real device from meta.json into RunResult.error (used as a provenance
    breadcrumb field; ok=True so it is not an error) so a GPU claim is always checkable.
    """
    if status == "OK" and os.path.exists(paths.preds):
        preds = np.load(paths.preds, allow_pickle=True)
        device = ""
        try:
            if os.path.exists(paths.meta):
                meta = json.load(open(paths.meta))
                device = str(meta.get("device", ""))
        except Exception:
            pass
        # `error` doubles as a provenance note on success (ok=True). Keeps RunResult shape frozen.
        note = f"device={device}" if device else ""
        return RunResult(job.program.id, ok=True, preds=list(preds), error=note,
                         wall_seconds=wall)

    if status.startswith("ERR:"):
        parts = status.split(":", 2)
        error_kind = parts[1].strip() if len(parts) > 1 else "other"
        message = parts[2].strip() if len(parts) > 2 else status
        return RunResult(job.program.id, ok=False, error=message,
                         error_kind=error_kind, wall_seconds=wall)

    # No status / killed by rlimit (e.g. RLIMIT_CPU) -> stderr tail (Phase-0 behavior).
    tail = (stderr or "").strip().splitlines()[-1] if (stderr or "").strip() else "no output"
    kind_guess = "cpu" if returncode and returncode < 0 else "other"
    return RunResult(job.program.id, ok=False, error=tail[:200],
                     error_kind=kind_guess, wall_seconds=wall)


# ============================================================================================
# Section 6 — executors: WHERE the child runs (design 02 §4)
# ============================================================================================

@runtime_checkable
class Executor(Protocol):
    """The substrate seam. A single blocking, reentrant, state-free run-one-job call."""
    def run(self, job: Job) -> RunResult: ...


def _preexec(cpu_seconds: int, address_mb: int, *, skip_address: bool):
    """Set resource limits in the child before exec (POSIX only). Mirrors frontier.sandbox._preexec.

    `skip_address` is set for the CUDA path (RLIMIT_AS spuriously OOMs on the GPU driver's large
    virtual mappings). On the sklearn path it is False, preserving Phase-0 behavior exactly.
    """
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    except (ValueError, OSError):
        pass
    if address_mb and not skip_address and sys.platform != "darwin":
        try:
            soft = address_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        except (ValueError, OSError):
            pass


class LocalSubprocessExecutor:
    """Run the child on THIS host. Generalizes frontier.sandbox.run_program over backends.

    For the sklearn backend the bundle, the verbatim runner, the 4-arg CLI, the rlimits, the wall
    timeout, the killpg-on-timeout, and the OK/ERR parser are all Phase-0 behavior — so the result
    matches `frontier.sandbox.run_program` bit-for-bit.
    """

    def run(self, job: Job) -> RunResult:
        t0 = time.time()
        # Parent-side capability gate: an unrunnable backend declines HONESTLY here, before any
        # child is spawned. Never silently rerouted to another backend.
        cap = probe_backend(job.backend)
        if not cap.runnable:
            return RunResult(job.program.id, ok=False,
                             error=f"backend '{job.backend.tag}' unavailable here: {cap.reason}",
                             error_kind="backend_unavailable", wall_seconds=time.time() - t0)

        skip_address = job.backend.tag == "torch" and cap.cuda  # RLIMIT_AS unreliable with CUDA
        with tempfile.TemporaryDirectory(prefix="frontier_sbx_") as d:
            paths = serialize_job_to_dir(job, d)
            argv = [sys.executable, paths.runner] + backend_for(job.backend).child_argv_tail(paths, job)
            posix = os.name == "posix"
            try:
                proc = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    start_new_session=posix,
                    preexec_fn=(lambda: _preexec(job.limits.cpu_seconds, job.limits.address_mb,
                                                 skip_address=skip_address)) if posix else None,
                )
            except Exception as e:  # noqa: BLE001
                return RunResult(job.program.id, ok=False, error=f"spawn failed: {e}",
                                 error_kind="other", wall_seconds=time.time() - t0)

            try:
                out, err = proc.communicate(timeout=job.limits.wall_seconds)
            except subprocess.TimeoutExpired:
                if posix:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                proc.communicate()
                return RunResult(job.program.id, ok=False,
                                 error=f"wall timeout {job.limits.wall_seconds}s",
                                 error_kind="timeout", wall_seconds=time.time() - t0)

            wall = time.time() - t0
            status = _parse_status_line(out)
            return parse_runner_result(job, status=status, stderr=err,
                                       returncode=proc.returncode, paths=paths, wall=wall)


class RemotePodExecutor:
    """Run the SAME bundle/runner/CLI on a GPU pod. GPU is a substrate SWAP, not a rewrite.

    Depends only on a minimal DUCK-TYPED provider interface (the verb set vfplatform's RunPod /
    Prime Intellect providers already expose), NOT on those modules — so this file compiles and
    the sklearn path runs with no torch and no pod present:

        provider.acquire(*, gpu, image) -> handle
        provider.push(handle, local_dir, remote)            # upload the bundle dir
        provider.exec(handle, cmd, timeout) -> {"status": "OK"|"ERR..."|line, "stdout","stderr","returncode"}
        provider.pull(handle, remote, local)                # download a file
        provider.release(handle)                            # honors the $5/day cap

    Firewall preserved across the wire: only preds.npy + the status line + meta.json come back;
    the candidate's code/weights/tensors never return to the parent. Infra failures are typed
    `error_kind="remote"` (separable from the candidate's own `fit`/`build`/`diverged` faults, so
    the portfolio can retry infra flakes but never retry a bad candidate).
    """

    def __init__(self, provider, *, image: str, gpu: str = "A100",
                 idle_release_s: float = 120.0, remote_dir: str = "/work"):
        self.provider = provider
        self.image = image
        self.gpu = gpu
        self.idle_release_s = idle_release_s
        self.remote_dir = remote_dir

    def run(self, job: Job) -> RunResult:
        t0 = time.time()
        # We trust the pod image to carry the backend; the parent still cannot probe the remote
        # interpreter cheaply, so the child's defensive import is the gate (typed `import` error).
        with tempfile.TemporaryDirectory(prefix="frontier_pod_") as d:
            paths = serialize_job_to_dir(job, d)
            try:
                h = self.provider.acquire(gpu=self.gpu, image=self.image)
            except Exception as e:  # noqa: BLE001
                return RunResult(job.program.id, ok=False, error=f"acquire: {e}",
                                 error_kind="remote", wall_seconds=time.time() - t0)
            try:
                self.provider.push(h, d, remote=self.remote_dir)
                tail = backend_for(job.backend).child_argv_tail(paths, job)
                # Rewrite local absolute paths to remote basenames (the bundle lands in remote_dir).
                tail = [os.path.basename(a) if os.path.isabs(str(a)) else a for a in tail]
                rd = self.remote_dir
                cmd = (f"cd {rd} && timeout {job.limits.wall_seconds} "
                       f"python runner.py " + " ".join(str(a) for a in tail))
                status_obj = self.provider.exec(h, cmd, timeout=job.limits.wall_seconds + 30)
                # Pull artifacts back (best-effort; absence => parsed as a non-OK status below).
                try:
                    self.provider.pull(h, remote=f"{rd}/preds.npy", local=paths.preds)
                except Exception:
                    pass
                try:
                    self.provider.pull(h, remote=f"{rd}/meta.json", local=paths.meta)
                except Exception:
                    pass
                wall = time.time() - t0
                status_line = self._status_from(status_obj)
                rc = status_obj.get("returncode") if isinstance(status_obj, dict) else None
                stderr = status_obj.get("stderr", "") if isinstance(status_obj, dict) else ""
                return parse_runner_result(job, status=status_line, stderr=stderr,
                                           returncode=rc, paths=paths, wall=wall)
            except TimeoutError:
                return RunResult(job.program.id, ok=False, error="pod wall timeout",
                                 error_kind="timeout", wall_seconds=time.time() - t0)
            except Exception as e:  # noqa: BLE001 — infra failure, NOT the candidate's fault
                return RunResult(job.program.id, ok=False, error=f"remote: {e}",
                                 error_kind="remote", wall_seconds=time.time() - t0)
            finally:
                # Honor the cost cap: release in finally so an idle pod never keeps billing.
                try:
                    self.provider.release(h)
                except Exception:
                    pass

    @staticmethod
    def _status_from(status_obj) -> str:
        """Extract the OK/ERR line from a provider exec result (dict or raw stdout string)."""
        if isinstance(status_obj, dict):
            if isinstance(status_obj.get("status"), str) and \
                    (status_obj["status"] == "OK" or status_obj["status"].startswith("ERR:")):
                return status_obj["status"]
            return _parse_status_line(status_obj.get("stdout", ""))
        return _parse_status_line(str(status_obj))


# ============================================================================================
# Section 7 — cost-aware backend selection (design 02 §7.2)
# ============================================================================================

@dataclass(frozen=True)
class EstimatedCost:
    seconds: float
    dollars: float


def cost_model(job: Job, capability: Capability, *, gpu_dollars_per_hour: float = 2.0) -> EstimatedCost:
    """Crude, auditable estimate of one run's wall-time and dollar cost.

    PROPOSAL-time convenience only — NEVER touches promotion (whichever backend runs, the SAME
    sealed certificate decides). The portfolio uses this to prefer the local CPU sklearn path for
    small problems and escalate to a GPU pod only past a break-even and within the $5/day cap.
    Estimates are logged so the choice is auditable; they are intentionally conservative, not tuned.
    """
    n, d = len(job.X_train), (job.X_train.shape[1] if job.X_train.ndim > 1 else 1)
    work = float(n) * float(d)
    if job.backend.tag == "sklearn":
        # CPU-bound; ~1e7 feature-ops/sec as a deliberately rough floor.
        secs = max(0.05, work / 1e7)
        return EstimatedCost(seconds=secs, dollars=0.0)
    # torch: epochs * work, GPU ~50x throughput when CUDA present, else CPU-torch (slow).
    epochs = (job.fit_spec.epochs if job.fit_spec else 100)
    tput = 5e8 if capability.cuda else 5e6
    secs = max(0.5, epochs * work / tput)
    dollars = (secs / 3600.0) * gpu_dollars_per_hour if capability.cuda else 0.0
    return EstimatedCost(seconds=secs, dollars=dollars)


# ============================================================================================
# Section 8 — back-compat entrypoint the integrator calls (design 02 §1 "dispatch")
# ============================================================================================

def run_program(program: Program, X_train, y_train, X_eval, *, kind: str,
                wall_seconds: float = 60.0, cpu_seconds: int = 55, address_mb: int = 4096,
                backend: BackendSpec = SKLEARN_SPEC,
                executor: Optional[Executor] = None,
                seed: int = 0, labels: Optional[tuple] = None,
                fit_spec: Optional[FitSpec] = None) -> RunResult:
    """Backend-agnostic generalization of frontier.sandbox.run_program.

    With no `backend`/`executor` (the defaults) this is BIT-FOR-BIT identical to the frozen
    Phase-0 sklearn path (verified in the test): same npz, same verbatim runner, same CLI, same
    rlimits/timeout/killpg, same OK/ERR parser, same RunResult. Pass `backend=TORCH_SPEC` for the
    gated neural path (declines honestly when torch is absent). Inject a `RemotePodExecutor` for
    GPU. The default executor is `LocalSubprocessExecutor`.
    """
    # gpu_mem_fraction comes from the backend defaults so the torch path bounds GPU memory; the
    # sklearn path ignores it (Phase-0 contract unchanged).
    gpu_frac = getattr(backend.defaults, "gpu_mem_fraction", None)
    limits = ResourceLimits(wall_seconds=wall_seconds, cpu_seconds=cpu_seconds,
                            address_mb=address_mb, gpu_mem_fraction=gpu_frac)
    if labels is None and kind == "classification":
        labels = tuple(sorted({str(v) for v in np.asarray(y_train).tolist()}))
    job = Job(program=program, X_train=np.asarray(X_train), y_train=np.asarray(y_train),
              X_eval=np.asarray(X_eval), kind=kind, backend=backend, limits=limits,
              seed=seed, labels=labels,
              fit_spec=(fit_spec or (FitSpec(seed=seed) if backend.tag == "torch" else None)))
    ex = executor or LocalSubprocessExecutor()
    return ex.run(job)


__all__ = [
    "ResourceLimits", "FitSpec", "BackendSpec", "Job",
    "Capability", "probe_backend",
    "SKLEARN_SPEC", "TORCH_SPEC", "Backend", "SklearnBackend", "TorchBackend", "backend_for",
    "serialize_job_to_dir", "parse_runner_result",
    "Executor", "LocalSubprocessExecutor", "RemotePodExecutor",
    "EstimatedCost", "cost_model",
    "run_program",
]
