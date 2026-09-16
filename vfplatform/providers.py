"""Compute provider abstraction -- the EXECUTE-fan-out dispatch layer (CPU now, GPU/RunPod gated).

A Provider runs a list of deterministic Jobs (a callable + its spec) and returns results. LocalCpuProvider
runs now on the laptop (thread pool). RunPodProvider is GATED on RUNPOD_API_KEY: with no key it reports
itself unavailable and `map` raises ResourceGated -- an HONEST decline, never a fake GPU run. The real
RunPod submit/poll/fetch is deferred until the user connects credentials; this is explicit, not hidden.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits  # cap each fit's native BLAS/OpenMP pool to avoid oversubscription
from dataclasses import dataclass, field
from typing import Any, Callable

_VF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ResourceGated(Exception):
    """Raised when a provider needs credentials/hardware that are not connected (honest stop)."""


def classify_transient(status_or_exc) -> bool:
    """Decide whether a failure is worth retrying (TRUE) or should fail fast (FALSE).

    TRANSIENT (retryable): network timeouts, ConnectionError, HTTP 429 (rate limit), HTTP 5xx
    (server-side). PERSISTENT (fail fast): HTTP 401/403/404/400 and other client errors -- retrying
    a bad key or a missing endpoint just burns the timeout budget.

    Accepts either an int HTTP status code OR an exception instance, so both the poll loop (which sees
    status codes / raise_for_status) and the request layer (which sees socket exceptions) can call it.
    """
    # int HTTP status path
    if isinstance(status_or_exc, bool):
        return False
    if isinstance(status_or_exc, int):
        code = status_or_exc
        if code == 429:
            return True
        if 500 <= code <= 599:
            return True
        return False
    # exception path
    exc = status_or_exc
    # requests is optional at import time of this module; resolve lazily for the type checks.
    try:
        import requests
        if isinstance(exc, requests.exceptions.Timeout):
            return True
        if isinstance(exc, requests.exceptions.ConnectionError):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            resp = getattr(exc, "response", None)
            code = getattr(resp, "status_code", None)
            if isinstance(code, int):
                return classify_transient(code)
            return False
        if isinstance(exc, requests.exceptions.RequestException):
            # other requests-layer errors (e.g. ChunkedEncodingError) are typically transient network hiccups
            return True
    except Exception:  # noqa: BLE001  requests not importable -> fall through to the stdlib checks below
        pass
    # stdlib / generic fallbacks
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    # an HTTPError-like object that carries a status code (duck-typed) -> classify by code
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(code, int):
        return classify_transient(code)
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return classify_transient(code)
    return False


@dataclass
class Job:
    """A deterministic unit of work: an id + the kwargs that fully specify it (for replay)."""
    job_id: str
    spec: dict = field(default_factory=dict)


class Provider:
    name = "abstract"
    execution_mode = "inprocess"   # "inprocess" = run a local fn(job); "worker" = run a serializable JobSpec

    def available(self) -> bool:
        raise NotImplementedError

    def capabilities(self) -> dict:
        raise NotImplementedError

    def map(self, fn: Callable[[Job], Any], jobs):
        """Run fn over jobs and return results in order. May raise ResourceGated if unavailable."""
        raise NotImplementedError


class LocalCpuProvider(Provider):
    name = "local-cpu"

    def __init__(self, max_workers=None):
        self.max_workers = max_workers or min(8, (os.cpu_count() or 2))

    def available(self) -> bool:
        return True

    def capabilities(self) -> dict:
        return {"device": "cpu", "max_workers": self.max_workers, "gated": False,
                "cost_per_hour_usd": 0.0}

    def map(self, fn, jobs):
        jobs = list(jobs)
        if not jobs:
            return []
        # sklearn fits release the GIL in their C/numpy inner loops, so a thread pool gives real overlap
        # for small laptop fan-outs without the pickling cost of processes. PIN each fit's native BLAS/OpenMP
        # team to 1 thread: without this, hist_gbm (no n_jobs knob) spawns an OpenMP team per fit and the
        # concurrent fits oversubscribe the cores (a 569-row hist_gbm fit measured ~27s vs ~0.2s capped --
        # the whole ~34s/run regression). This changes only HOW MANY threads a fit uses, never its numerical
        # output: the selected winner, val_score, certificate (checks=1), and sealed peek count are unchanged.
        def _one(job):
            with threadpool_limits(limits=1):
                return fn(job)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(_one, jobs))


class LocalWorkerProvider(Provider):
    """Runs the SERIALIZABLE JobSpec through the worker contract IN-PROCESS (no GPU, no network, no spend).
    Identical contract to RunPodProvider, so the whole remote path (JobSpec -> worker -> predictions ->
    local certify) is validated on CPU before any GPU spend. sklearn families only (no local torch)."""
    name = "local-worker"
    execution_mode = "worker"

    def __init__(self):
        self._handler = None

    def _h(self):
        if self._handler is None:
            import importlib.util
            import sys
            wdir = os.path.join(_VF, "worker")
            if wdir not in sys.path:        # so the handler's `from torch_models import ...` resolves in-process
                sys.path.insert(0, wdir)    # (the GPU worker has it on path via the image; the in-process path did not)
            path = os.path.join(wdir, "handler.py")
            spec = importlib.util.spec_from_file_location("vf_worker_handler", path)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            self._handler = m
        return self._handler

    def available(self) -> bool:
        return True

    def capabilities(self) -> dict:
        return {"device": "cpu", "worker": "local", "gated": False, "cost_per_hour_usd": 0.0}

    def run_jobspec(self, jobspec: dict) -> dict:
        return self._h().handler({"input": jobspec})

    def map(self, fn, jobs):
        return [fn(job, self.run_jobspec((job.spec or {}).get("remote", {}))) for job in jobs]


class RunPodPodProvider(Provider):
    """RunPod POD GPU provider, driven by `runpodctl` (no custom image, key-only). Provisions a stock
    pytorch pod, runs each JobSpec on the GPU via `runpodctl exec python` (an embedded script that reuses
    worker/handler.py verbatim -- no file transfer), parses the result, and the FROZEN certifier runs
    locally. Tear down with teardown(). execution_mode='worker', identical contract to the serverless path.

    NOTE: `runpodctl exec` uses SSH (a high TCP port). It works from a normal-egress machine/server, NOT
    from an HTTPS-only sandbox. Validate from your own machine; the product runs on a normal server."""
    name = "runpod-pod"
    execution_mode = "worker"

    def __init__(self, *, gpu="NVIDIA RTX A4000", image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                 cost_ceiling=0.6, api_key=None, pod_id=None, gpu_candidates=None):
        from ._runpod import resolve_runpod_key
        self.api_key = api_key or resolve_runpod_key()
        self.gpu, self.image, self.cost_ceiling = gpu, image, cost_ceiling
        # ordered cheap GPUs to try for community availability (fall through on no-capacity)
        self.gpu_candidates = gpu_candidates or [gpu, "NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000",
                                                 "NVIDIA GeForce RTX 4090"]
        self.pod_id = pod_id
        self._handler_src = None

    def available(self) -> bool:
        import shutil
        return bool(self.api_key) and shutil.which("runpodctl") is not None

    def capabilities(self) -> dict:
        if not self.available():
            return {"device": "gpu", "gpu": self.gpu, "gated": True,
                    "reason": "needs RUNPOD_API_KEY + runpodctl on PATH"}
        return {"device": "gpu", "gpu": self.gpu, "gated": False,
                "cost_per_hour_usd": RunPodProvider._GPU_COST.get(self.gpu.replace("NVIDIA ", "")
                                                                  .replace("GeForce ", ""), 0.5)}

    def _handler_source(self):
        if self._handler_src is None:
            with open(os.path.join(_VF, "worker", "handler.py")) as fh:
                self._handler_src = fh.read()
        return self._handler_src

    def build_script(self, jobspec: dict) -> str:
        """Self-contained pod script: torch_models.py + worker/handler.py (verbatim, import inlined) + the
        inlined JobSpec, printing VFRESULT. No file transfer, no custom image -- reuses the worker contract."""
        import json
        with open(os.path.join(_VF, "vfplatform", "torch_models.py")) as fh:
            torch_src = fh.read()
        # the torch classes are now defined in this script's namespace -> drop handler's module import
        handler_src = self._handler_source().replace(
            "from torch_models import TorchMLPClassifier, TorchMLPRegressor  # in the image",
            "# torch_models inlined above (TorchMLPClassifier / TorchMLPRegressor in scope)")
        return (torch_src + "\n\n" + handler_src
                + "\n\n_VF_INPUT = " + json.dumps(jobspec)
                + "\nimport json as _json\nprint('VFRESULT ' + _json.dumps(run(_VF_INPUT)))\n")

    def ensure_pod(self, *, dry_run=False):
        if self.pod_id:
            return self.pod_id
        import subprocess, re, time
        # UNIQUE name (RunPod deletes async -> a fixed name can collide with a still-deleting pod) + try a
        # few cheap GPUs for community availability + surface the REAL error instead of an opaque exit code.
        name = f"vf-pod-{int(time.time()) % 1000000}"
        gpus = self.gpu_candidates or [self.gpu]
        errors = []
        for gpu in gpus:
            cmd = (f'runpodctl create pod --name {name} --gpuType "{gpu}" --communityCloud '
                   f'--imageName "{self.image}" --cost {self.cost_ceiling} --gpuCount 1 '
                   f'--containerDiskSize 20 --startSSH --ports "22/tcp"')
            if dry_run:
                return {"dry_run": True, "cmd": cmd}
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            m = re.search(r'pod "([^"]+)" created', r.stdout or "")
            if r.returncode == 0 and m:
                self.pod_id, self.gpu = m.group(1), gpu
                return self.pod_id
            errors.append(f"{gpu}: {((r.stderr or '') + (r.stdout or '')).strip()[:160] or 'exit ' + str(r.returncode)}")
        raise ResourceGated(f"could not create a pod (tried {len(gpus)} GPU type(s)): " + " | ".join(errors))

    def run_jobspec(self, jobspec: dict, *, dry_run=False, exec_timeout=900, verbose=True):
        """Run one JobSpec on the pod. ONE `runpodctl exec` call -- runpodctl waits for the pod to come
        online internally (the heavy image can take minutes to pull + start sshd), then runs the script
        and returns. A long timeout lets that single wait complete instead of being chopped into retries."""
        import json, subprocess, tempfile, os as _os
        script = self.build_script(jobspec)
        if dry_run:
            return {"dry_run": True, "pod_cmd": f"runpodctl exec python <script> --pod_id {self.pod_id}",
                    "script_len": len(script)}
        self.ensure_pod()
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script); path = f.name
        if verbose:
            print(f"  [pod {self.pod_id}] running on GPU (runpodctl waits for the pod to come online, "
                  f"then runs; up to ~{exec_timeout // 60} min while the image pulls)...")
        try:
            r = subprocess.run(f"runpodctl exec python {path} --pod_id {self.pod_id}",
                               shell=True, capture_output=True, text=True, timeout=exec_timeout)
            text = (r.stdout or "") + "\n" + (r.stderr or "")
            for line in text.splitlines():
                if line.startswith("VFRESULT "):
                    return json.loads(line[len("VFRESULT "):])
            return {"error": f"no VFRESULT (pod may not have come online): {text.strip()[-220:]}"}
        except subprocess.TimeoutExpired:
            return {"error": f"exec timed out after {exec_timeout}s (image too slow / pod never online)"}
        finally:
            _os.unlink(path)

    def map(self, fn, jobs):
        if not self.available():
            raise ResourceGated(self.capabilities().get("reason", "RunPod pod provider unavailable"))
        self.ensure_pod()
        return [fn(job, self.run_jobspec((job.spec or {}).get("remote", {}))) for job in jobs]

    def teardown(self):
        if self.pod_id:
            import subprocess
            subprocess.run(f"runpodctl remove pod {self.pod_id}", shell=True, capture_output=True, text=True)
            self.pod_id = None


class RunPodProvider(Provider):
    """RunPod Serverless GPU provider. Auth via the key loader; jobs are submitted to a serverless
    ENDPOINT (built from the worker image). Honest gating: no key OR no endpoint => not runnable, and
    submission raises ResourceGated rather than faking a GPU run. Cost is real, so the loop's Checkpoint
    must approve any RunPod move before map() is reached."""
    name = "runpod-gpu"
    execution_mode = "worker"
    RUN_URL = "https://api.runpod.ai/v2/{ep}/runsync"
    STATUS_URL = "https://api.runpod.ai/v2/{ep}/status/{jid}"
    # rough per-hour on-demand estimates (USD) so the Checkpoint always sees GPU work as PAID; refine from
    # the live RunPod price API later. These are upper-ish bounds for cost-safety, not billing truth.
    _GPU_COST = {"RTX A4000": 0.4, "A4000": 0.4, "A40": 0.79, "A100": 1.99, "H100": 3.99}

    def __init__(self, api_key=None, endpoint_id=None, gpu="A40", timeout=600, workers_max=None):
        from ._runpod import resolve_runpod_key
        self.api_key = api_key or resolve_runpod_key()
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
        self.gpu = gpu
        self.timeout = timeout
        # Client-side concurrency cap for the parallel fan-out. NOTE: the endpoint's own workersMax (set at
        # the RunPod endpoint, not per-request) is the real upper bound on how many workers actually run in
        # parallel; this only caps how many jobs we submit+poll concurrently. None preserves the prior
        # behavior (min(len(jobs), 8)).
        self.workers_max = workers_max

    def available(self) -> bool:
        return bool(self.api_key and self.endpoint_id)

    def capabilities(self) -> dict:
        if not self.api_key:
            return {"device": "gpu", "gpu": self.gpu, "gated": True,
                    "reason": "no RunPod API key (set RUNPOD_API_KEY or .runpod_key)"}
        if not self.endpoint_id:
            return {"device": "gpu", "gpu": self.gpu, "gated": True,
                    "reason": "key present but no serverless endpoint: build+push the worker image, "
                              "create an endpoint, then set RUNPOD_ENDPOINT_ID (see deploy_runpod.md)"}
        return {"device": "gpu", "gpu": self.gpu, "gated": False, "endpoint": self.endpoint_id,
                "cost_per_hour_usd": self._GPU_COST.get(self.gpu, 1.0)}   # positive => Checkpoint gates it

    def health(self):
        from ._runpod import health
        return health(self.api_key)

    def health_summary(self) -> dict:
        """Read-only worker/job snapshot for THIS serverless endpoint (NO spend).

        GET https://api.runpod.ai/v2/{endpoint_id}/health with the bearer key. On HTTP 200 returns
        {"http": 200, "workers": {...}, "jobs": {...}} (the RunPod health payload). On any non-200
        returns {"http": <code>, "error": "..."}. NEVER raises: a network error (no connectivity, DNS,
        timeout) becomes {"http": 0, "error": "..."} so callers can branch on a plain dict."""
        if not self.api_key:
            return {"http": 0, "error": "no RunPod API key"}
        if not self.endpoint_id:
            return {"http": 0, "error": "no RunPod endpoint configured"}
        import requests
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/health"
        hdr = {"Authorization": f"Bearer {self.api_key}"}
        try:
            r = requests.get(url, headers=hdr, timeout=15)
        except Exception as ex:  # noqa: BLE001  honest: a network failure is http 0, never an exception
            return {"http": 0, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
        if r.status_code != 200:
            return {"http": r.status_code, "error": f"HTTP {r.status_code}: {(r.text or '')[:160]}"}
        try:
            body = r.json()
        except Exception as ex:  # noqa: BLE001
            return {"http": r.status_code, "error": f"bad JSON: {str(ex)[:120]}"}
        return {"http": 200, "workers": body.get("workers", {}) or {}, "jobs": body.get("jobs", {}) or {}}

    def probe(self) -> dict:
        """Deep, read-only availability check (NO spend). Returns
        {"available": bool, "reason": str|None, "health": <health_summary dict>}.

        This is the HONEST GPU gate: presence of a key+endpoint (available()) is necessary but not
        sufficient. A bad/expired key, an unreachable endpoint, or an endpoint whose only workers are
        unhealthy all mean we should fall back to CPU rather than fake a GPU run. A cold endpoint
        (0 ready, 0 unhealthy) is FINE: serverless cold-start spins a worker on first submit, so we
        must NOT block that."""
        if not (self.api_key and self.endpoint_id):
            return {"available": False, "reason": "RunPod not configured",
                    "health": {"http": 0, "error": "not configured"}}
        h = self.health_summary()
        code = h.get("http", 0)
        if code in (401, 403):
            return {"available": False, "reason": "RunPod auth rejected (bad or expired key)", "health": h}
        if code != 200:
            return {"available": False,
                    "reason": f"RunPod endpoint unreachable (HTTP {code})", "health": h}
        w = h.get("workers", {}) or {}
        unhealthy = int(w.get("unhealthy", 0) or 0)
        live = (int(w.get("ready", 0) or 0) + int(w.get("idle", 0) or 0)
                + int(w.get("initializing", 0) or 0) + int(w.get("running", 0) or 0))
        if unhealthy > 0 and live == 0:
            return {"available": False,
                    "reason": "RunPod workers unhealthy (0 ready); running on CPU", "health": h}
        return {"available": True, "reason": None, "health": h}

    def submit_one(self, payload: dict, timeout=None, poll=5, max_retries=3):
        """Run ONE job on the serverless endpoint with RESILIENT retry-with-backoff. Wraps the async
        /run + /status poll (_submit_once). On a TRANSIENT failure (timeout, ConnectionError, HTTP 429,
        HTTP 5xx -- see classify_transient) it retries with capped exponential backoff (sleeps 1, 2, 4s,
        up to max_retries=3 attempts), all bounded by the overall timeout budget. On a PERSISTENT failure
        (HTTP 401/403/404/400, worker error, job FAILED/CANCELLED/TIMED_OUT) it raises IMMEDIATELY -- no
        pointless retries against a bad key or a missing endpoint. Returns the worker output dict. Real spend."""
        if not self.api_key:
            raise ResourceGated("no RunPod API key")
        if not self.endpoint_id:
            raise ResourceGated("no RunPod endpoint configured (set RUNPOD_ENDPOINT_ID; see deploy_runpod.md)")
        import time
        budget = timeout or self.timeout
        deadline = time.time() + budget
        attempt = 0
        last_exc = None
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                # out of budget; surface the last transient cause if we have one
                if last_exc is not None:
                    raise RuntimeError(f"RunPod submit gave up after {attempt} attempt(s) within {budget}s: "
                                       f"{type(last_exc).__name__}: {str(last_exc)[:160]}")
                raise RuntimeError(f"RunPod submit exhausted {budget}s budget")
            try:
                return self._submit_once(payload, timeout=remaining, poll=poll)
            except ResourceGated:
                raise  # config gate, never retry
            except Exception as e:  # noqa: BLE001
                if not classify_transient(e) or attempt >= max_retries:
                    raise  # persistent failure (fail fast) OR retry budget exhausted -> surface it
                last_exc = e
                back = min(2 ** attempt, 4)        # capped exponential backoff: 1, 2, 4, 4, ...
                attempt += 1
                # never sleep past the deadline
                if deadline - time.time() <= back:
                    raise RuntimeError(f"RunPod submit out of time after {attempt} transient failure(s): "
                                       f"{type(e).__name__}: {str(e)[:160]}")
                time.sleep(back)

    def _submit_once(self, payload: dict, timeout=None, poll=5):
        """ONE async /run + /status poll until COMPLETED (no retry). Survives cold starts (the first job
        waits for a worker to spin up + pull the image, which can far exceed /runsync's internal wait).
        Returns the worker output dict, or raises on failure (the caller classifies + may retry)."""
        import requests, time
        base = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        hdr = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        r = requests.post(f"{base}/run", json={"input": payload}, headers=hdr, timeout=60)
        r.raise_for_status()
        jid = r.json().get("id")
        if not jid:
            raise RuntimeError(f"RunPod /run gave no job id: {str(r.json())[:200]}")
        deadline = time.time() + (timeout or self.timeout)
        st = "IN_QUEUE"
        while time.time() < deadline:
            sr = requests.get(f"{base}/status/{jid}", headers=hdr, timeout=30)
            sr.raise_for_status()
            s = sr.json()
            st = s.get("status")
            if st == "COMPLETED":
                out = s.get("output", {})
                if isinstance(out, dict) and out.get("error"):
                    raise RuntimeError(f"worker error: {out['error']}")
                return out
            if st in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"RunPod job {st}: {str(s)[:200]}")
            time.sleep(poll)
        raise RuntimeError(f"RunPod job not COMPLETED within {timeout or self.timeout}s (last status {st})")

    def run_jobspec(self, jobspec: dict) -> dict:
        """Run one serializable JobSpec on the endpoint (used for the winner's fit_predict_test). Real spend."""
        return self.submit_one(jobspec)

    def map(self, fn, jobs):
        """PARALLEL serverless fan-out: each job carries a serializable payload in job.spec['remote']. All
        candidates are submitted + polled CONCURRENTLY (a client thread per job), so the serverless endpoint
        runs them in parallel up to its workersMax (set workersMax>1 for real concurrency; otherwise they
        queue but submission never blocks). A single candidate's failure is isolated into an error output
        (the loop scores it FAILED and proceeds) rather than aborting the whole fan-out. fn(job, out) shapes
        each result. Order is preserved. Raises ResourceGated until an endpoint is configured."""
        if not self.available():
            raise ResourceGated(self.capabilities().get("reason", "RunPod not available"))
        import concurrent.futures as _cf

        def _one(job):
            payload = (job.spec or {}).get("remote")
            if payload is None:
                raise ValueError("RunPodProvider.map requires job.spec['remote'] (a serializable payload)")
            try:
                out = self.submit_one(payload)
            except Exception as e:  # noqa: BLE001  isolate a single bad candidate
                out = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            return fn(job, out)

        if len(jobs) <= 1:
            return [_one(j) for j in jobs]
        # cap = workers_max when set (client-side concurrency limit), else the prior default of 8.
        cap = self.workers_max if (self.workers_max and self.workers_max > 0) else 8
        with _cf.ThreadPoolExecutor(max_workers=min(len(jobs), cap)) as ex:
            return list(ex.map(_one, jobs))


class PrimeIntellectPodProvider(Provider):
    """Prime Intellect GPU Pod provider. Provisions an on-demand GPU pod via the Prime Intellect REST
    API, runs the worker handler via SSH, and tears down on completion. Same contract as RunPodPodProvider
    (execution_mode='worker', JobSpec -> worker/handler.py -> VFRESULT) so the loop and certification
    are identical. Cost is real; the Checkpoint must approve any GPU move before map() is reached.

    Lifecycle: find_best_offer -> create_pod -> wait ACTIVE -> SSH exec -> parse VFRESULT -> delete_pod.
    Auth: PRIME_INTELLECT_API_KEY env var (or .prime_intellect_key file)."""
    name = "prime-intellect-gpu"
    execution_mode = "worker"

    def __init__(self, *, api_key=None, gpu_preferences=None, max_price=5.0,
                 image="cuda_12_1_pytorch_2_2", exec_timeout=900, disk_size=50):
        from ._prime_intellect import resolve_pi_key
        self.api_key = api_key or resolve_pi_key()
        self.gpu_preferences = gpu_preferences or ["A100_80GB", "H100_80GB", "A100_40GB"]
        self.max_price = max_price
        self.image = image
        self.exec_timeout = exec_timeout
        self.disk_size = disk_size
        self.pod_id = None
        self._ssh_info = None
        self._gpu_name = None
        self._handler_src = None

    def available(self) -> bool:
        return bool(self.api_key)

    def capabilities(self) -> dict:
        if not self.api_key:
            return {"device": "gpu", "gpu": "prime-intellect", "gated": True,
                    "reason": "no Prime Intellect API key (set PRIME_INTELLECT_API_KEY or .prime_intellect_key)"}
        return {"device": "gpu", "gpu": self._gpu_name or "prime-intellect", "gated": False,
                "cost_per_hour_usd": 2.0}

    def probe(self) -> dict:
        """Check if we can provision a GPU pod (read-only). Queries availability."""
        if not self.api_key:
            return {"available": False, "reason": "Prime Intellect not configured"}
        from ._prime_intellect import find_best_offer
        offer = find_best_offer(api_key=self.api_key, gpu_preferences=self.gpu_preferences,
                                max_price=self.max_price)
        if not offer:
            return {"available": False, "reason": "no GPU available within price ceiling"}
        return {"available": True, "reason": None,
                "offer": {"gpu": offer.get("gpuType"), "price": offer.get("prices", {}).get("onDemand")}}

    def _handler_source(self):
        if self._handler_src is None:
            with open(os.path.join(_VF, "worker", "handler.py")) as fh:
                self._handler_src = fh.read()
        return self._handler_src

    def build_script(self, jobspec: dict) -> str:
        """Self-contained script: torch_models.py + worker/handler.py + inlined JobSpec -> VFRESULT."""
        import json
        with open(os.path.join(_VF, "vfplatform", "torch_models.py")) as fh:
            torch_src = fh.read()
        handler_src = self._handler_source().replace(
            "from torch_models import TorchMLPClassifier, TorchMLPRegressor  # in the image",
            "# torch_models inlined above (TorchMLPClassifier / TorchMLPRegressor in scope)")
        return (torch_src + "\n\n" + handler_src
                + "\n\n_VF_INPUT = " + json.dumps(jobspec)
                + "\nimport json as _json\nprint('VFRESULT ' + _json.dumps(run(_VF_INPUT)))\n")

    def ensure_pod(self, *, verbose=True):
        """Provision or reuse a GPU pod. Returns pod_id."""
        if self.pod_id and self._ssh_info:
            return self.pod_id
        from ._prime_intellect import find_best_offer, create_pod, wait_for_active, parse_ssh_connection
        offer = find_best_offer(api_key=self.api_key, gpu_preferences=self.gpu_preferences,
                                max_price=self.max_price)
        if not offer:
            raise ResourceGated("no Prime Intellect GPU available within price ceiling "
                                f"(tried {self.gpu_preferences}, max ${self.max_price}/hr)")
        self._gpu_name = offer.get("gpuType", "unknown")
        if verbose:
            print(f"  [prime-intellect] provisioning {self._gpu_name} "
                  f"(${offer.get('prices', {}).get('onDemand', '?')}/hr)...")
        result = create_pod(offer, image=self.image, disk_size=self.disk_size, api_key=self.api_key)
        if not result.get("ok"):
            raise ResourceGated(f"Prime Intellect create_pod failed: {result.get('error', 'unknown')}")
        self.pod_id = result.get("id")
        if not self.pod_id:
            raise ResourceGated(f"create_pod response missing id: {result}")
        if verbose:
            print(f"  [prime-intellect] pod {self.pod_id} created, waiting for ACTIVE...")
        active = wait_for_active(self.pod_id, api_key=self.api_key, timeout=600)
        if not active.get("ok"):
            self.teardown()
            raise ResourceGated(f"Prime Intellect pod failed to become ACTIVE: {active.get('error')}")
        self._ssh_info = parse_ssh_connection(active)
        if not self._ssh_info:
            self.teardown()
            raise ResourceGated("Prime Intellect pod has no SSH connection info")
        if verbose:
            host, port, user = self._ssh_info
            print(f"  [prime-intellect] pod ACTIVE at {user}@{host}:{port}")
        return self.pod_id

    def run_jobspec(self, jobspec: dict, *, verbose=True) -> dict:
        """Run one JobSpec on the pod via SSH. Same contract as RunPodPodProvider.run_jobspec."""
        import json
        import subprocess
        import tempfile
        self.ensure_pod(verbose=verbose)
        host, port, user = self._ssh_info
        script = self.build_script(jobspec)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            path = f.name
        try:
            scp_cmd = (f'scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
                       f'-P {port} {path} {user}@{host}:/tmp/vf_job.py')
            subprocess.run(scp_cmd, shell=True, capture_output=True, text=True, timeout=120)
            ssh_cmd = (f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
                       f'-p {port} {user}@{host} "python3 /tmp/vf_job.py"')
            if verbose:
                print(f"  [prime-intellect] running job on {self._gpu_name} (up to "
                      f"~{self.exec_timeout // 60} min)...")
            r = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True,
                               timeout=self.exec_timeout)
            text = (r.stdout or "") + "\n" + (r.stderr or "")
            for line in text.splitlines():
                if line.startswith("VFRESULT "):
                    return json.loads(line[len("VFRESULT "):])
            return {"error": f"no VFRESULT in output: {text.strip()[-220:]}"}
        except subprocess.TimeoutExpired:
            return {"error": f"SSH exec timed out after {self.exec_timeout}s"}
        except Exception as ex:  # noqa: BLE001
            return {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
        finally:
            os.unlink(path)

    def map(self, fn, jobs):
        if not self.available():
            raise ResourceGated(self.capabilities().get("reason", "Prime Intellect unavailable"))
        self.ensure_pod()
        return [fn(job, self.run_jobspec((job.spec or {}).get("remote", {}))) for job in jobs]

    def teardown(self):
        """Delete the pod (stops billing)."""
        if self.pod_id:
            from ._prime_intellect import delete_pod
            try:
                delete_pod(self.pod_id, api_key=self.api_key)
            except Exception:  # noqa: BLE001
                pass
            self.pod_id = None
            self._ssh_info = None


def first_available(providers):
    """Pick the first available provider (honest fallback to CPU when GPU is gated)."""
    for p in providers:
        if p.available():
            return p
    raise ResourceGated("no compute provider available")
