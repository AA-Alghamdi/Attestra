# 02 - Execution Substrate: one backend-agnostic contract (sklearn now, torch/GPU next)

Status: implementation-ready design. Builds on the FROZEN Phase-0 spine
(`frontier/{program,task,certify,sandbox,engine}.py`, `frontier/CONTRACT.md`). Adds NEW modules
under `frontier/core/`; does not edit Phase-0 files. References the provider contract in
`vfplatform/` conceptually only (those modules are in-flight; nothing here imports them at module
load, so this design compiles and the sklearn path runs without torch or any pod).

## 0. Problem statement (from the audit)

`frontier/sandbox.py` runs untrusted candidate code out-of-process, fits a sklearn-like estimator,
predicts the eval split, and returns predictions only (the numeric firewall). It is real and
verified. But the substrate is sklearn-on-CPU only: `_RUNNER` hardcodes `est.fit / est.predict`,
the parent ships `(Xtr, ytr, Xev)` as plain `.npz`, and there is no notion of a backend. The
`vfplatform` torch harness and pod providers are real code that the loop never invokes, so
"architectures and code and GPU" is aspirational.

This document defines ONE execution contract so that:
1. The Phase-0 sklearn path is preserved bit-for-bit (the certifier sees the same predictions).
2. A neural/torch path runs the SAME propose -> sandbox -> predictions -> certify-on-sealed loop,
   gated on the backend actually being importable (and CUDA optionally present).
3. A local CPU run and a Prime Intellect / RunPod GPU run are a substrate SWAP behind the same
   runner contract, not a rewrite.
4. The system degrades honestly: if a backend or a GPU is unavailable it logs that and declines
   that arm. It never fakes a GPU run or silently downgrades torch to numpy.

Non-goals: this module does not author neural architectures (that is the proposer/agentic loop),
does not compute any metric (the parent does, via `certify.py` -> `science.py`), and does not own
network isolation (that is the container/pod the runner is dropped into).

---

## 1. The Backend abstraction

A backend is "how a Program's authored code becomes predictions on an eval split." sklearn and
torch differ in: what the authored code must define, how the child trains it, and where it runs.
We capture that in a small, declarative `Backend` descriptor plus a child-side `run` entrypoint.
The descriptor lives in the PARENT (trusted). The child gets a backend TAG (a string), imports the
matching child-side runner, and executes it. The parent never imports torch.

```python
# frontier/core/backends.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Callable

BackendTag = str   # "sklearn" | "torch"

@dataclass(frozen=True)
class BackendSpec:
    """Declarative description of one execution backend. Lives in the trusted parent.

    The parent uses this only to: (a) decide if the backend is runnable here
    (probe), (b) pick resource defaults, (c) stamp the child invocation with `tag`.
    It does NOT import the backend's heavy deps (torch). All heavy work is child-side.
    """
    tag: BackendTag
    # contract the authored code must satisfy, surfaced to proposers/LLM and used by the
    # child to know what symbol to look for. sklearn: "build_estimator"; torch: "build_module".
    entrypoint: str
    # import names whose importability defines availability (probed in a subprocess).
    requires: tuple[str, ...]
    # optional capabilities; "cuda" means "prefers a GPU but can run CPU".
    optional: tuple[str, ...] = ()
    # default resource envelope (overridable per-run).
    defaults: "ResourceLimits" = None
    # human note for honest-degradation logs.
    note: str = ""

class Backend(Protocol):
    """Parent-side handle. Thin: spec + a probe + a way to build the child invocation.
    The actual fit/predict is in the child runner selected by `spec.tag`."""
    spec: BackendSpec
    def available(self) -> "Capability": ...        # runtime probe (see §6)
    def child_runner_source(self) -> str: ...        # returns the child-side runner text
```

### SklearnBackend (works today)

Wraps the existing `_RUNNER` essentially unchanged. `entrypoint="build_estimator"`,
`requires=("sklearn","numpy")`, no `cuda`. `available()` returns a `Capability` that is always
present in the local jax-env-311 interpreter (sklearn/scipy/numpy installed; no torch).

```python
# frontier/core/sklearn_backend.py
SKLEARN_SPEC = BackendSpec(
    tag="sklearn",
    entrypoint="build_estimator",
    requires=("numpy", "sklearn"),
    defaults=ResourceLimits(wall_seconds=60.0, cpu_seconds=55, address_mb=4096),
    note="CPU only; the Phase-0 verified path.",
)

class SklearnBackend:
    spec = SKLEARN_SPEC
    def available(self) -> Capability:
        return probe_backend(self.spec)          # §6
    def child_runner_source(self) -> str:
        return _SKLEARN_CHILD_RUNNER             # the fit/predict body, §2
```

### TorchBackend (gated)

`entrypoint="build_module"`, `requires=("torch","numpy")`, `optional=("cuda",)`. `available()`
probes torch importability and CUDA in a throwaway subprocess; if torch is absent the backend
reports unavailable (locally true), so any torch-tagged Program is declined honestly, never run as
sklearn.

```python
# frontier/core/torch_backend.py
TORCH_SPEC = BackendSpec(
    tag="torch",
    entrypoint="build_module",
    requires=("numpy", "torch"),
    optional=("cuda",),
    defaults=ResourceLimits(wall_seconds=900.0, cpu_seconds=900, address_mb=16384,
                            gpu_mem_fraction=0.9),
    note="needs importable torch; CUDA optional (CPU torch allowed but logged as slow).",
)

class TorchBackend:
    spec = TORCH_SPEC
    def available(self) -> Capability:
        return probe_backend(self.spec)          # importable torch? cuda.is_available()?
    def child_runner_source(self) -> str:
        return _TORCH_CHILD_RUNNER               # nn.Module train/eval body, §3
```

### How dispatch preserves the Phase-0 sklearn path

The generalization is: `run_program` keeps its EXACT current signature (back-compat), and gains a
keyword `backend: BackendSpec = SKLEARN_SPEC`. With no `backend` passed, behavior is identical to
today: same `.npz` job, same child runner body, same `RunResult`. The only internal change is that
the child runner text is selected by `backend.tag` and the backend tag is passed as a CLI arg to
the child. There is no behavioral diff on the sklearn path, which keeps `test_spine.py` green.

```python
# frontier/core/sandbox2.py  (NEW; frontier/sandbox.py stays frozen and re-exports run_program)
def run_program(program, X_train, y_train, X_eval, *, kind,
                wall_seconds=60.0, cpu_seconds=55, address_mb=4096,
                backend: BackendSpec = SKLEARN_SPEC,
                executor: "Executor | None" = None) -> RunResult:
    limits = ResourceLimits(wall_seconds, cpu_seconds, address_mb,
                            gpu_mem_fraction=getattr(backend.defaults, "gpu_mem_fraction", None))
    cap = probe_backend(backend)                       # §6
    if not cap.runnable:
        return RunResult(program.id, ok=False,
                         error=f"backend '{backend.tag}' unavailable here: {cap.reason}",
                         error_kind="backend_unavailable", wall_seconds=0.0)
    job = Job(program=program, X_train=X_train, y_train=y_train, X_eval=X_eval,
              kind=kind, backend=backend, limits=limits)
    ex = executor or LocalSubprocessExecutor()         # §4: swap for RemotePodExecutor
    return ex.run(job)
```

`frontier/sandbox.py` is not edited; the new path lives in `frontier/core/sandbox2.py`, and the
engine is pointed at it via dependency injection. The frozen `run_program` remains as the
verified reference and is what `test_spine.py` exercises.

---

## 2. The runner protocol (parent <-> child)

The contract between parent and child is a small set of files in a temp dir, exactly mirroring the
current sandbox, plus a backend tag and a richer job (train/val/sealed are all just "eval feature
arrays" from the runner's view; the parent decides which split to send).

### What the parent passes in

A `Job` serialized to disk. The features are float arrays; targets are object-typed (so string
class labels survive). For torch we additionally need to tell the child the task kind and the
label vocabulary (so the child can map class strings to integer indices deterministically and map
back on predict, keeping the firewall: the child returns the SAME string/float predictions the
sklearn path returns).

```python
# frontier/core/job.py
@dataclass
class ResourceLimits:
    wall_seconds: float = 60.0
    cpu_seconds: int = 55
    address_mb: int = 4096
    gpu_mem_fraction: float | None = None   # torch pod path only; None = no cap

@dataclass
class Job:
    program: Program
    X_train: "np.ndarray"
    y_train: "np.ndarray"
    X_eval: "np.ndarray"
    kind: str                 # "classification" | "regression"
    backend: BackendSpec
    limits: ResourceLimits
    seed: int = 0             # deterministic seeding across backends (§7)
    labels: tuple | None = None   # sorted class-label vocab for clf (else None)
```

On disk (one temp dir per run, auto-cleaned):
- `job.npz` : `Xtr` (float), `ytr` (object), `Xev` (float), plus `kind`, `seed`, `labels` arrays.
- `candidate.py` : `program.code` verbatim (untrusted).
- `runner.py` : the TRUSTED child runner body for this backend (`backend.child_runner_source()`).
- `preds.npy` : child writes predictions here on success.
- `meta.json` : child writes a small typed-status record (kind, message, device used, train
  steps, peak mem) on exit, so the parent can log honest device/resource facts without parsing
  free-text. The `OK` / `ERR:<kind>:<msg>` stdout line remains the source of truth for ok/fail.

Child invocation: `python runner.py job.npz candidate.py preds.npy <kind> <backend_tag>
meta.json`. The backend tag is an explicit CLI arg so the same launcher serves both backends.

### How the child detects/imports the backend

The child runner file is already backend-specific (the parent picked it by tag), so it does not
guess. But it still imports defensively and emits a typed error if the import fails (this is the
last line of defense behind the parent's probe, e.g. a pod image that is missing torch):

```python
# top of _TORCH_CHILD_RUNNER
try:
    import torch
except Exception as e:
    print("ERR:import:torch not importable in child: %s" % (str(e)[:160],)); sys.exit(0)
```

The sklearn child runner is byte-identical to the current `_RUNNER` (it already handles
build/compile/fit errors with the existing taxonomy).

### Resource limits

Reuses the current `_preexec`:
- `RLIMIT_CPU` set to `limits.cpu_seconds` (POSIX). Kills runaway compute.
- Wall-clock timeout enforced by the parent `proc.communicate(timeout=...)`, then
  `os.killpg(SIGKILL)` on the new session so child threads/subprocesses die too.
- `RLIMIT_AS` best-effort, skipped on macOS (it breaks interpreter startup) and skipped for the
  torch path even on Linux when CUDA is used, because RLIMIT_AS counts the GPU driver's large
  virtual mappings and would spuriously OOM. For torch we instead rely on (a) `gpu_mem_fraction`
  via `torch.cuda.set_per_process_memory_fraction` inside the child, and (b) the pod's container
  memory cgroup. This is documented, not hidden: the design comment in the torch runner says
  "RLIMIT_AS is unreliable with CUDA; GPU memory is bounded by set_per_process_memory_fraction +
  container cgroup, host RAM by RLIMIT_DATA where available."

### Typed error taxonomy (extends `RunResult.error_kind`)

Current: `"timeout"|"import"|"fit"|"build"|"oom"|"cpu"|"other"`. Add:
- `"backend_unavailable"` : parent probe says this backend cannot run here (torch missing, no
  CUDA when the spec required it). Distinct from `"import"` (which means the import failed inside
  a child that was expected to have it, e.g. a broken pod image).
- `"cuda_oom"` : child caught `torch.cuda.OutOfMemoryError`. Distinct from CPU `"oom"`
  (`MemoryError`), because the remedy differs (smaller batch / model vs more host RAM).
- `"diverged"` : training produced NaN/Inf loss or NaN predictions. The child detects this
  (`torch.isfinite` on loss each step; finite-check on predictions) and reports it rather than
  writing garbage predictions. Mirrors the repo rule "when a solve returns NaN/Inf, replace
  with zeros and count the event" - here we do not write zeros silently; we fail typed so the
  proposer learns the recipe diverged.
- `"remote"` : the remote executor failed for an infrastructure reason (pod unreachable, upload
  failed, result fetch failed) as opposed to the candidate's own code failing. Keeps "the
  candidate is bad" separable from "the cloud flaked," which matters for retries (§8 risks).

All of these flow into `recent_errors` in the engine context and so condition the next round's
proposals exactly as today.

---

## 3. The neural train/eval protocol (TorchBackend, child-internal)

The authored code defines `build_module(n_features, n_outputs, kind) -> nn.Module` (and may
define an optional `fit_spec()` returning training hyperparameters; see §7). The child owns the
train loop so that authored code stays small and the firewall holds (the child, not the candidate,
decides nothing about the metric and returns only predictions). Everything below runs inside the
sandbox child; the parent sees only `preds.npy` + typed status.

```python
# _TORCH_CHILD_RUNNER (sketch; runs in the isolated child)
import sys, json, numpy as np
job, codep, outp, kind, tag, metap = sys.argv[1:7]
try:
    import torch, torch.nn as nn
except Exception as e:
    print("ERR:import:%s" % (str(e)[:160],)); sys.exit(0)

d = np.load(job, allow_pickle=True)
Xtr, ytr, Xev = d["Xtr"].astype("float32"), d["ytr"], d["Xev"].astype("float32")
seed = int(d["seed"]); labels = list(d["labels"]) if d["labels"].size else None

# deterministic seeding across the whole child (§7)
torch.manual_seed(seed); np.random.seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda" and "gpu_mem_fraction" available: torch.cuda.set_per_process_memory_fraction(frac)

# target encoding (kept child-internal; predictions are decoded back to the SAME dtype/labels
# the sklearn path returns, so certify.py is backend-agnostic)
if kind == "classification":
    lab2idx = {l: i for i, l in enumerate(labels)}
    y = torch.tensor([lab2idx[str(v)] for v in ytr], dtype=torch.long)
    n_out, loss_fn = len(labels), nn.CrossEntropyLoss()
else:
    y = torch.tensor(ytr.astype("float32")).view(-1, 1)
    n_out, loss_fn = 1, nn.MSELoss()

ns = {}
try:
    exec(compile(open(codep).read(), "<candidate>", "exec"), ns)
    model = ns["build_module"](Xtr.shape[1], n_out, kind).to(device)
except KeyError:
    print("ERR:build:no callable build_module(n_features,n_outputs,kind)"); sys.exit(0)
except Exception as e:
    print("ERR:build:%s: %s" % (type(e).__name__, str(e)[:160])); sys.exit(0)

spec = ns.get("fit_spec", lambda: {})()       # optional; defaults below
opt   = make_optimizer(model, spec)            # adamw, lr from spec or default
sched = make_scheduler(opt, spec)              # cosine/onecycle/none
epochs, bs, patience = spec.get(...defaults...)
scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda" and spec.get("amp", True)))

# train/val split INSIDE train (held-out for early stopping; NOT the certifier's sealed split)
idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
n_in_val = max(1, int(0.1 * len(y)))
tr_idx, es_idx = idx[n_in_val:], idx[:n_in_val]

best_state, best_es, bad = None, float("inf"), 0
try:
    for epoch in range(epochs):
        model.train()
        for xb, yb in batches(Xtr[tr_idx], y[tr_idx], bs, shuffle=True, seed=seed+epoch):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(xb.to(device)); l = loss_fn(out, yb.to(device))
            if not torch.isfinite(l):
                print("ERR:diverged:non-finite loss at epoch %d" % epoch); sys.exit(0)
            scaler.scale(l).backward(); scaler.step(opt); scaler.update()
        if sched: sched.step()
        es = eval_loss(model, Xtr[es_idx], y[es_idx], loss_fn, device)   # early-stop signal
        if es < best_es - 1e-4: best_es, best_state, bad = es, snapshot(model), 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state: model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = batched_forward(model, Xev, bs, device)       # never OOM on a huge eval
    if kind == "classification":
        pred_idx = logits.argmax(1).cpu().numpy()
        preds = np.array([labels[i] for i in pred_idx], dtype=object)   # decode -> SAME labels
    else:
        preds = logits.view(-1).cpu().numpy().astype(float)
    if not np.all(np.isfinite(preds.astype(float) if kind=="regression" else np.arange(len(preds)))):
        print("ERR:diverged:non-finite predictions"); sys.exit(0)
    np.save(outp, np.asarray(preds, dtype=object), allow_pickle=True)
    json.dump({"device": device, "epochs_ran": epoch+1, "best_es": float(best_es)}, open(metap,"w"))
    print("OK")
except torch.cuda.OutOfMemoryError:
    print("ERR:cuda_oom:CUDA out of memory")
except MemoryError:
    print("ERR:oom:host MemoryError")
except Exception as e:
    print("ERR:fit:%s: %s" % (type(e).__name__, str(e)[:160]))
```

Key firewall and integrity properties:
- The child returns predictions in the EXACT format the sklearn child returns (string labels for
  clf, floats for reg), so `certify.score_val` and `certify_on_sealed` are unchanged and
  backend-agnostic. The certifier cannot tell a torch winner from a sklearn one - that is the
  point.
- No metric is ever computed in the child. The internal early-stop loss is a TRAINING signal, not
  a reported number, and is computed on a child-internal slice of the TRAIN data only, never on
  the parent's val or sealed split. (The child only ever receives one eval feature array and no
  eval targets.)
- NaN/Inf are typed failures, never silently-zeroed predictions, so a diverged recipe is visible
  to the proposer rather than scoring as chance.
- Default `fit_spec` (when the candidate omits it) is a sane literature-standard recipe (AdamW,
  cosine schedule, early stopping, AMP on CUDA), documented as a fallback, not a tuned answer.

---

## 4. Remote execution: GPU as a substrate swap

The `Executor` is the seam. `LocalSubprocessExecutor` is exactly today's behavior (Popen + rlimits
+ timeout + killpg). `RemotePodExecutor` implements the SAME interface against a Prime Intellect /
RunPod pod. The engine and the backends are unchanged; only which executor is injected changes.

```python
# frontier/core/executor.py
class Executor(Protocol):
    def run(self, job: Job) -> RunResult: ...

class LocalSubprocessExecutor:
    """Today's sandbox.run_program internals, generalized over backend.child_runner_source()."""
    def run(self, job: Job) -> RunResult:
        # write job.npz / candidate.py / runner.py / meta.json into a TemporaryDirectory,
        # Popen([python, runner, ...args, job.backend.tag]) with start_new_session + _preexec,
        # communicate(timeout=job.limits.wall_seconds), killpg on timeout,
        # parse OK / ERR:<kind>:<msg>, load preds.npy, attach meta.json device facts.
        ...

class RemotePodExecutor:
    """The SAME contract, on a GPU pod. References the vfplatform provider contract conceptually:
    a provider exposes acquire()->handle, push(handle, local->remote), exec(handle, cmd)->status,
    pull(handle, remote->local), release(handle). We do NOT import those modules here; we depend
    only on this minimal duck-typed interface so this file compiles with no torch and no pod."""
    def __init__(self, provider, *, image: str, gpu: str = "A100",
                 idle_release_s: float = 120.0):
        self.provider = provider          # injected; satisfies the duck interface above
        ...
    def run(self, job: Job) -> RunResult:
        bundle = serialize_job_to_dir(job)            # SAME on-disk layout as local
        try:
            h = self.provider.acquire(gpu=self.gpu, image=self.image)
        except Exception as e:
            return RunResult(job.program.id, ok=False, error=f"acquire: {e}",
                             error_kind="remote", wall_seconds=0.0)
        try:
            self.provider.push(h, bundle, remote="/work")
            # remote command is the IDENTICAL runner invocation, just `python` on the pod.
            cmd = f"cd /work && timeout {job.limits.wall_seconds} python runner.py " \
                  f"job.npz candidate.py preds.npy {job.kind} {job.backend.tag} meta.json"
            status = self.provider.exec(h, cmd, timeout=job.limits.wall_seconds + 30)
            self.provider.pull(h, remote="/work/preds.npy", local=bundle.preds_path)
            self.provider.pull(h, remote="/work/meta.json", local=bundle.meta_path)
            return parse_runner_result(job, status, bundle)   # SAME parser as local
        except RemoteTimeout:
            return RunResult(job.program.id, ok=False, error="pod wall timeout",
                             error_kind="timeout", wall_seconds=...)
        except Exception as e:
            return RunResult(job.program.id, ok=False, error=f"remote: {e}",
                             error_kind="remote", wall_seconds=...)
        finally:
            self.provider.release(h)       # honor the $5/day cap; idle pods are released
```

Why this is a swap, not a rewrite:
- The on-disk bundle, the child `runner.py`, the CLI arg order, and the OK/ERR parser are
  identical local and remote. The only difference is whether `python runner.py ...` runs via Popen
  on this host or via `provider.exec` on the pod.
- The firewall is preserved across the wire: only `preds.npy` + the status line + `meta.json`
  come back. The candidate's code, weights, and any intermediate tensors never return to the
  parent; the parent still computes every number.
- The provider interface is the minimal verb set that the real `vfplatform/_runpod.py` /
  `providers.py` already implement (acquire/push/exec/pull/release). We depend on the SHAPE, not
  the modules, so this design compiles and the sklearn path runs with no pod and no torch present.
- Cost/safety: `release` in a `finally` plus `idle_release_s` honors the deployment's $5/day spend
  cap recorded in project memory. The GPU lane being "not yet wired" is exactly what this seam
  wires, behind the capability probe (§6) so it is off until a provider is injected.

---

## 5. Resource / parallelism: composing with a portfolio executor

Concurrency is OUTSIDE the substrate: the substrate exposes a single blocking `Executor.run(job)
-> RunResult`. A separate `frontier/core/portfolio.py` (its own module, designed elsewhere) runs
many jobs concurrently with bounded resources. The substrate makes this easy because each run is a
fully isolated process (local) or a separate pod (remote) with no shared mutable state.

```python
# sketch of the seam the portfolio module consumes (substrate side stays single-job)
class BoundedExecutor:
    """Wraps any Executor with a concurrency cap + per-backend resource accounting.
    Lives in the portfolio module; shown here only to fix the contract."""
    def __init__(self, inner: Executor, max_concurrency: int,
                 *, gpu_slots: int = 1, cpu_slots: int = os.cpu_count()):
        ...
    def submit(self, job: Job) -> "Future[RunResult]": ...
```

Contract the substrate guarantees so the portfolio can schedule safely:
- `Executor.run` is reentrant and shares no global mutable state (each call gets its own temp dir
  / pod). The Phase-0 JIT-globals contract does not apply here (no JAX in the substrate), but the
  same discipline holds: no module-level mutable singletons in the child runner.
- Resource accounting is per-backend: sklearn jobs consume a CPU slot; torch-CUDA jobs consume a
  GPU slot. The portfolio uses two semaphores so a flood of torch jobs cannot starve sklearn jobs
  and the pod count stays within the cost cap. `BackendSpec.optional` containing `"cuda"` is the
  signal that a job needs a GPU slot.
- Early-kill of losing arms (ROADMAP Phase 6) composes cleanly: cancelling a `Future` triggers
  `proc.kill()` / `provider.release(h)` so a killed arm frees its slot immediately and stops
  spending.
- Bounded concurrency default is conservative: `max_concurrency = min(cpu_count, gpu_slots +
  cpu_budget)`; remote GPU concurrency defaults to 1 until the cost cap is explicitly raised.

This keeps the substrate single-responsibility (run one job safely, return predictions) and lets
the portfolio own scheduling, fairness, and cost.

---

## 6. Capability probe: detect backends, degrade honestly

The probe answers "can this backend actually run here, right now?" without importing heavy deps
into the parent and without faking. It runs the import check in a throwaway subprocess (so a
segfaulting torch build cannot crash the parent) and caches the result per (interpreter, backend).

```python
# frontier/core/probe.py
@dataclass(frozen=True)
class Capability:
    tag: BackendTag
    runnable: bool
    cuda: bool = False
    devices: tuple = ()        # e.g. ("NVIDIA A100",)
    reason: str = ""           # honest explanation when not runnable
    interpreter: str = ""      # which python was probed

def probe_backend(spec: BackendSpec, *, python=sys.executable) -> Capability:
    # one-shot child that imports spec.requires and reports cuda; never imported in-parent.
    src = (
        "import json,sys\n"
        f"req={list(spec.requires)}; opt={list(spec.optional)}\n"
        "miss=[m for m in req if __import__('importlib').util.find_spec(m) is None]\n"
        "cuda=False; devs=[]\n"
        "if not miss and 'cuda' in opt:\n"
        "    try:\n"
        "        import torch; cuda=torch.cuda.is_available()\n"
        "        devs=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]\n"
        "    except Exception: pass\n"
        "print(json.dumps({'miss':miss,'cuda':cuda,'devs':devs}))\n"
    )
    try:
        out = subprocess.run([python, "-c", src], capture_output=True, text=True, timeout=30)
        info = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:
        return Capability(spec.tag, runnable=False, reason=f"probe failed: {e}", interpreter=python)
    if info["miss"]:
        return Capability(spec.tag, runnable=False,
                          reason=f"missing modules: {info['miss']}", interpreter=python)
    cuda_required = ("cuda" in spec.optional) and ("cuda_required" in spec.optional)
    if cuda_required and not info["cuda"]:
        return Capability(spec.tag, runnable=False, reason="CUDA required but unavailable",
                          interpreter=python)
    return Capability(spec.tag, runnable=True, cuda=info["cuda"],
                      devices=tuple(info["devs"]), interpreter=python)
```

Honest-degradation rules (these are load-bearing, per the audit's "real-but-gated" requirement):
- Locally (jax-env-311: sklearn yes, torch no), `probe_backend(SKLEARN_SPEC).runnable == True` and
  `probe_backend(TORCH_SPEC).runnable == False` with `reason="missing modules: ['torch']"`. A
  torch-tagged Program returns `error_kind="backend_unavailable"`; the engine logs it and that arm
  declines. It is NEVER silently rerouted to sklearn (different model => different certificate =>
  that would be a faked result).
- A torch backend with `cuda=False` (CPU torch) is runnable but the executor logs `device=cpu` and
  the meta record carries it, so a "GPU run" claim can always be checked against the recorded
  device. The system never reports a GPU run that did not touch a GPU.
- The probe result and the per-run `meta.json["device"]` are both surfaced in `EngineResult`
  history, so the research artifact (ROADMAP Phase 9) can state exactly where each candidate ran.

---

## 7. Innovation

### 7.1 Unified declarative fit-spec (one training contract for both backends)

A single `FitSpec` dataclass that both backends consume, so a proposer (or the LLM) writes ONE
declarative training intent and the substrate maps it to whichever backend runs. sklearn ignores
the neural-only fields; torch ignores the sklearn-only ones. This means the proposer reasons about
"train this for ~N effort with early stopping" once, not per-framework.

```python
# frontier/core/fitspec.py
@dataclass(frozen=True)
class FitSpec:
    # shared
    seed: int = 0
    early_stop: bool = True
    # neural-only (ignored by sklearn)
    epochs: int = 100; batch_size: int = 256; lr: float = 1e-3
    optimizer: str = "adamw"; scheduler: str = "cosine"; weight_decay: float = 1e-4
    amp: bool = True; patience: int = 10; grad_clip: float | None = 1.0
    # sklearn-only (ignored by torch) - e.g. n_iter for iterative estimators
    sklearn_overrides: dict = field(default_factory=dict)
```

The authored code may expose `fit_spec() -> dict` to override defaults; the child validates and
clamps it (epochs/batch bounded so a candidate cannot request a 10-hour run that blows the wall
timeout into a `timeout` failure). The default spec is a documented literature-standard fallback,
explicitly NOT tuned to any target benchmark (per the scientific-integrity rule).

### 7.2 Cost-aware backend selection

When a Program is backend-agnostic (e.g. an MLP that both an sklearn `MLPClassifier` and a torch
module could realize), the substrate can pick the cheapest runnable backend that meets a SLA. A
tiny `cost_model(job, capability) -> EstimatedCost(seconds, dollars)` lets the portfolio prefer
the local CPU sklearn path for small `n` and only escalate to a GPU pod when `n*d` or the model's
parameter count crosses a break-even (and only within the $5/day cap). Selection is a PROPOSAL-time
convenience and never touches promotion: whichever backend runs, the SAME sealed certificate
decides. Cost estimates are logged so the choice is auditable.

### 7.3 Deterministic seeding across backends (reproducibility)

One `seed` flows from `EngineConfig` -> `Job.seed` -> the child, which seeds EVERY source of
nondeterminism for its backend: sklearn (`random_state` injected where the estimator accepts it;
`numpy` global seed otherwise), torch (`torch.manual_seed`, `numpy.random.seed`,
`torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic=True`,
`DataLoader` worker seeding, the train/early-stop split RNG). This gives the Phase-7 "seed-
controlled re-execution to confirm reproducibility" oracle a real hook: re-running the winner with
the same seed must reproduce the sealed predictions bit-for-bit on CPU and within a documented
tolerance on CUDA (where some kernels are nondeterministic even with the flag; that tolerance is
recorded honestly rather than claimed as exact). The certificate records the seed and the device
so a reviewer can reproduce from a clean checkout.

---

## 8. Risks + mitigations

| Risk | Why it bites | Mitigation |
|---|---|---|
| Silent backend downgrade (torch-tagged Program quietly run as sklearn) | Would produce a certificate for a DIFFERENT model than was proposed - a faked result | Probe gates every run; unavailable backend returns `backend_unavailable` and declines that arm; the child runner is backend-specific and never substitutes a model |
| Fake-GPU claim | A "GPU run" that actually ran CPU torch overstates capability | `meta.json["device"]` records the real device; probe records `cuda`/`devices`; both surfaced in history and the report; a GPU claim is always checkable |
| RLIMIT_AS spuriously OOMs CUDA jobs | GPU driver maps huge virtual address space | Skip RLIMIT_AS for the CUDA path; bound GPU memory via `set_per_process_memory_fraction` + container cgroup; document the substitution in the runner comment |
| Diverged training scoring as chance | NaN/Inf predictions silently treated as valid would corrupt selection | Child finite-checks loss each step and predictions before save; emits `error_kind="diverged"` instead of writing garbage; proposer sees the typed failure |
| Remote pod flakiness conflated with bad candidates | Infra failure would wrongly mark a good candidate as failed and mislead the proposer | Distinct `error_kind="remote"`; the portfolio retries `remote` failures (bounded) but NEVER retries `fit`/`build`/`diverged` (those are the candidate's fault) |
| Runaway pod cost | GPU pods bill per minute; the deployment has a $5/day cap | `provider.release` in `finally`; `idle_release_s`; remote GPU concurrency defaults to 1; cost model logs estimated/actual spend; the GPU lane stays gated until a provider is explicitly injected |
| Wall-timeout via over-ambitious epochs | A candidate requests epochs that exceed the wall budget and dies as `timeout`, losing the partial result | Child clamps `FitSpec.epochs/batch_size` to fit the wall budget; early-stop snapshot means the best-so-far weights are used even if training is cut short |
| Breaking the frozen Phase-0 path | Any behavioral diff fails `test_spine.py` and the audit's invariants | New code lives in `frontier/core/`; `frontier/sandbox.py` stays frozen; sklearn child runner is byte-identical; back-compat `run_program` signature; the engine is pointed at the new path by injection, with `test_spine.py` re-run green before and after |
| Pickle/`allow_pickle` surface on remote pulls | Loading attacker-influenced `.npy` with `allow_pickle=True` is an RCE vector if the pod is compromised | Predictions are written/read as object arrays we control; on the remote path, validate dtype/shape against the expected eval length before trusting; longer-term, switch the wire format to a non-pickle codec (typed JSON / arrow) for predictions, which are just strings or floats |

---

## 9. Build order (incremental, each step ships green)

1. `frontier/core/{job,fitspec,probe,backends,sklearn_backend}.py` + `executor.py`
   (`LocalSubprocessExecutor`) + `sandbox2.py`. Wire `SklearnBackend` through the new path; confirm
   it reproduces `frontier/sandbox.run_program` outputs bit-for-bit; re-run `test_spine.py` (7/7).
2. `torch_backend.py` + `_TORCH_CHILD_RUNNER`. Locally, probe reports torch unavailable; add a unit
   test asserting `backend_unavailable` and honest decline (no torch needed to test the gate).
3. `RemotePodExecutor` against a duck-typed fake provider in tests (push/exec/pull/release stubbed);
   prove the SAME runner contract round-trips predictions. Real provider injection is a deployment
   step behind the cost cap.
4. Cost model + seed plumbing + the Phase-7 reproduce-the-seal oracle hook.

The invariant across all four: the parent imports no torch, computes every number via
`certify.py` -> `science.py`, and the sealed test is still touched exactly once for the winner -
regardless of which backend or host produced the predictions.
