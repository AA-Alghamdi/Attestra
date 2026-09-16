"""Unified GPU backend: local GPU -> Prime Intellect pod -> honest degradation.

Execution priority:
  1. Local GPU (torch + CUDA/MPS) — fastest, no cost
  2. Prime Intellect GPU pod — remote, pay-per-use, needs stock
  3. Honest decline — no GPU available, caller decides fallback

The backend also provides LLM inference via Prime Intellect's
OpenAI-compatible endpoint (always available, independent of GPU pods).
"""
from __future__ import annotations

import json
import os
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------------------- config

PI_INFERENCE_BASE = "https://api.pinference.ai/api/v1"
PI_API_BASE = "https://api.primeintellect.ai/api/v1"

# Models ranked by capability for code generation tasks
_CODE_MODELS = [
    "deepseek/deepseek-r1",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen3-235b-a22b",
    "deepseek/deepseek-v3-0324",
]

# Preferred GPU types in order of cost-efficiency
_GPU_PREFERENCES = ["A100_80GB", "H100_80GB", "A100_40GB", "A6000"]


@dataclass
class GPUStatus:
    """Current GPU availability across all backends."""
    local_available: bool = False
    local_device: str = "cpu"
    local_info: str = ""
    remote_available: bool = False
    remote_offers: List[Dict] = field(default_factory=list)
    remote_provider: str = ""
    active_pods: List[Dict] = field(default_factory=list)
    inference_available: bool = False
    inference_models: List[str] = field(default_factory=list)
    api_key_set: bool = False
    error: str = ""

    @property
    def any_gpu(self) -> bool:
        return self.local_available or self.remote_available or bool(self.active_pods)

    @property
    def has_active_pod(self) -> bool:
        return len(self.active_pods) > 0

    def summary(self) -> str:
        parts = []
        if self.local_available:
            parts.append(f"local:{self.local_device}")
        if self.active_pods:
            parts.append(f"deployed:{len(self.active_pods)} active pod(s)")
        if self.remote_offers:
            n = len(self.remote_offers)
            parts.append(f"remote:{n} offers ({self.remote_provider})")
        if self.inference_available:
            parts.append(f"inference:{len(self.inference_models)} models")
        if not parts:
            parts.append("no GPU available")
        return " | ".join(parts)


@dataclass
class RemoteTrainResult:
    """Result from a remote GPU training job."""
    ok: bool
    score: float = 0.0
    model_artifact: Optional[bytes] = None
    predictions: Optional[np.ndarray] = None
    wall_seconds: float = 0.0
    cost_usd: float = 0.0
    error: str = ""
    pod_id: str = ""
    gpu_type: str = ""
    logs: str = ""


# ------------------------------------------------------------------------------- key resolution

def _resolve_pi_key(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    return os.environ.get("PRIME_INTELLECT_API_KEY")


# ------------------------------------------------------------------------------- deployed pods

def list_active_pods(*, api_key: Optional[str] = None) -> List[Dict]:
    """Return the user's already-deployed pods that are ACTIVE (usable now).

    Unlike `/availability/gpus` (rentable *offers*), this queries `/pods/` — pods
    the user has already created. This lets the backend attach to a running pod
    (e.g. a manually-deployed H100) instead of provisioning a new one, and means
    the user is not charged twice. Pods returned here are NEVER auto-deleted by
    the training path (see `submit_remote_training`).
    """
    key = _resolve_pi_key(api_key)
    if not key:
        return []
    try:
        import requests
        r = requests.get(
            f"{PI_API_BASE}/pods/",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        pods = data.get("data", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        active = [p for p in pods if isinstance(p, dict) and str(p.get("status", "")).upper() == "ACTIVE"]
        return active
    except Exception:
        return []


# ------------------------------------------------------------------------------- GPU status

def check_gpu_status(*, api_key: Optional[str] = None) -> GPUStatus:
    """Comprehensive GPU status check across all backends."""
    status = GPUStatus()
    key = _resolve_pi_key(api_key)
    status.api_key_set = bool(key)

    # 1. Local GPU
    try:
        import torch
        if torch.cuda.is_available():
            status.local_available = True
            status.local_device = "cuda"
            status.local_info = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            status.local_available = True
            status.local_device = "mps"
            status.local_info = "Apple Silicon GPU"
    except ImportError:
        status.local_info = "torch not installed"

    if not key:
        status.error = "no Prime Intellect API key"
        return status

    # 2a. Already-deployed pods (usable immediately, no new provisioning/charge)
    status.active_pods = list_active_pods(api_key=key)
    if status.active_pods:
        status.remote_provider = "prime_intellect"

    # 2b. Rentable GPU offers (stock available to provision a new pod)
    try:
        import requests
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        r = requests.get(
            f"{PI_API_BASE}/availability/gpus",
            params={"gpu_count": 1},
            headers=headers, timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            offers = data if isinstance(data, list) else data.get("data", data.get("offers", []))
            if isinstance(offers, list):
                available = [o for o in offers if o.get("stockStatus") == "Available"]
                status.remote_offers = available
                status.remote_provider = "prime_intellect"
        # remote is "available" if we either have a deployed pod OR rentable stock
        status.remote_available = bool(status.active_pods) or len(status.remote_offers) > 0
    except Exception as e:
        status.remote_available = bool(status.active_pods)
        status.error = f"remote GPU check failed: {e!r}"

    # 3. Inference
    try:
        import requests
        r = requests.get(
            f"{PI_INFERENCE_BASE}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            models = data.get("data", []) if isinstance(data, dict) else []
            status.inference_available = True
            status.inference_models = [m.get("id") for m in models if m.get("id")]
    except Exception:
        pass

    return status


# ------------------------------------------------------------------------------- LLM inference

def build_llm_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4000,
) -> Optional[Callable]:
    """Build an LLM call function using Prime Intellect inference.

    Returns a callable: (system_prompt: str, user_prompt: str) -> (text: str, usage: dict)
    or None if no API key is available.
    """
    key = _resolve_pi_key(api_key)
    if not key:
        return None

    try:
        import openai
    except ImportError:
        return None

    client = openai.OpenAI(
        api_key=key,
        base_url=PI_INFERENCE_BASE,
        timeout=120,
    )

    # Select best available model
    selected_model = model
    if not selected_model:
        # Try to find the best available model
        try:
            import requests
            r = requests.get(
                f"{PI_INFERENCE_BASE}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                available_ids = {
                    m.get("id") for m in data.get("data", []) if m.get("id")
                }
                for preferred in _CODE_MODELS:
                    if preferred in available_ids:
                        selected_model = preferred
                        break
        except Exception:
            pass
        if not selected_model:
            selected_model = "meta-llama/llama-3.3-70b-instruct"

    def llm_call(system: str, user: str) -> Tuple[str, dict]:
        resp = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "model": selected_model,
            }
        return content, usage

    return llm_call


def build_frontier_llm_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Callable[[str], str]]:
    """Build a frontier-compatible LLM client: prompt(str) -> code(str).

    The frontier engine expects a simpler interface than the full llm_call.
    """
    llm_call = build_llm_client(api_key=api_key, model=model)
    if llm_call is None:
        return None

    def frontier_client(prompt: str) -> str:
        text, _ = llm_call(
            "You are an expert ML engineer. Write clean Python code. "
            "Return ONLY the code, no markdown fences, no explanation.",
            prompt,
        )
        return text

    return frontier_client


# ------------------------------------------------------------------------------- remote GPU execution

def find_best_gpu_offer(
    *,
    api_key: Optional[str] = None,
    gpu_preferences: Optional[List[str]] = None,
    max_price_usd: Optional[float] = None,
) -> Optional[Dict]:
    """Find the cheapest available GPU pod from Prime Intellect."""
    key = _resolve_pi_key(api_key)
    if not key:
        return None

    import requests
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    prefs = gpu_preferences or _GPU_PREFERENCES

    for gpu_type in prefs:
        try:
            r = requests.get(
                f"{PI_API_BASE}/availability/gpus",
                params={"gpu_type": gpu_type, "gpu_count": 1},
                headers=headers, timeout=15,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            offers = data if isinstance(data, list) else data.get("data", data.get("offers", []))
            if not isinstance(offers, list):
                continue
            available = [o for o in offers if o.get("stockStatus") == "Available"]
            if max_price_usd:
                available = [
                    o for o in available
                    if (o.get("prices", {}).get("onDemand") or 999) <= max_price_usd
                ]
            if available:
                available.sort(key=lambda o: o.get("prices", {}).get("onDemand", 999))
                return available[0]
        except Exception:
            continue
    return None


def create_gpu_pod(
    offer: Dict,
    *,
    name: Optional[str] = None,
    image: str = "cuda_12_1_pytorch_2_2",
    disk_size: int = 50,
    api_key: Optional[str] = None,
) -> Dict:
    """Create a GPU pod from an availability offer. Returns pod info or error dict."""
    key = _resolve_pi_key(api_key)
    if not key:
        return {"ok": False, "error": "no API key"}

    import requests
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    pod_name = name or f"attestra-gpu-{int(time.time()) % 1000000}"
    body = {
        "pod": {
            "name": pod_name,
            "cloudId": offer["cloudId"],
            "gpuType": offer["gpuType"],
            "socket": offer.get("socket", "PCIe"),
            "gpuCount": offer.get("gpuCount", 1),
            "image": image,
            "diskSize": disk_size,
        },
        "provider": {"type": offer.get("provider", "hyperstack")},
    }
    dc = offer.get("dataCenter") or offer.get("dataCenterId")
    if dc:
        body["pod"]["dataCenterId"] = dc

    try:
        r = requests.post(
            f"{PI_API_BASE}/pods/", json=body,
            headers=headers, timeout=60,
        )
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    if r.status_code not in (200, 201):
        return {"ok": False, "error": f"HTTP {r.status_code}: {(r.text or '')[:300]}"}
    try:
        return {"ok": True, **r.json()}
    except Exception:
        return {"ok": False, "error": "bad JSON in create response"}


def wait_for_pod_active(
    pod_id: str,
    *,
    api_key: Optional[str] = None,
    timeout: int = 600,
    poll_interval: int = 10,
) -> Dict:
    """Poll pod until ACTIVE/FINISHED. Returns pod dict or error."""
    key = _resolve_pi_key(api_key)
    if not key:
        return {"ok": False, "error": "no API key"}

    import requests
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    deadline = time.time() + timeout
    last_status = "unknown"

    while time.time() < deadline:
        try:
            r = requests.get(
                f"{PI_API_BASE}/pods/{pod_id}",
                headers=headers, timeout=30,
            )
            if r.status_code != 200:
                time.sleep(poll_interval)
                continue
            pod = r.json()
            status = pod.get("status", "UNKNOWN")
            install = pod.get("installationStatus", "PENDING")
            last_status = f"{status}/{install}"
            if status == "ERROR":
                return {"ok": False, "error": f"pod entered ERROR: {pod.get('installationFailure', '')}"}
            if status == "TERMINATED":
                return {"ok": False, "error": "pod was terminated"}
            if status == "ACTIVE" and install == "FINISHED":
                return {"ok": True, **pod}
        except Exception:
            pass
        time.sleep(poll_interval)

    return {"ok": False, "error": f"pod not ACTIVE within {timeout}s (last: {last_status})"}


def delete_gpu_pod(pod_id: str, *, api_key: Optional[str] = None) -> Dict:
    """Terminate and delete a GPU pod."""
    key = _resolve_pi_key(api_key)
    if not key:
        return {"ok": False, "error": "no API key"}

    import requests
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        r = requests.delete(
            f"{PI_API_BASE}/pods/{pod_id}",
            headers=headers, timeout=30,
        )
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    if r.status_code not in (200, 204):
        return {"ok": False, "error": f"HTTP {r.status_code}: {(r.text or '')[:200]}"}
    return {"ok": True}


_SSH_KEY_CACHE: Optional[str] = None


def _ssh_identity_opts() -> List[str]:
    """Return ['-i', keyfile] if a private key is configured, else [].

    Precedence: PRIME_INTELLECT_SSH_KEY_FILE (a path) wins; otherwise, if
    PRIME_INTELLECT_SSH_KEY holds the key material itself, materialize it once to a
    0600 file under ~/.ssh and reuse it. Without either, return [] so ssh falls back
    to the agent / default identities -- never crash for a missing key.
    """
    global _SSH_KEY_CACHE
    path = os.environ.get("PRIME_INTELLECT_SSH_KEY_FILE")
    if path and os.path.exists(path):
        return ["-i", path]
    if _SSH_KEY_CACHE and os.path.exists(_SSH_KEY_CACHE):
        return ["-i", _SSH_KEY_CACHE]
    material = os.environ.get("PRIME_INTELLECT_SSH_KEY")
    if not material:
        return []
    try:
        ssh_dir = os.path.expanduser("~/.ssh")
        os.makedirs(ssh_dir, mode=0o700, exist_ok=True)
        keyfile = os.path.join(ssh_dir, "prime_intellect_id")
        body = material if material.endswith("\n") else material + "\n"
        with open(os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            f.write(body)
        _SSH_KEY_CACHE = keyfile
        return ["-i", keyfile]
    except OSError:
        return []


def _parse_ssh(pod_info: Dict) -> Optional[Tuple[str, int, str]]:
    """Extract (host, port, user) from pod's SSH connection info."""
    ssh = pod_info.get("sshConnection")
    if isinstance(ssh, list):
        ssh = next((s for s in ssh if s), None)
    if not ssh:
        ip = pod_info.get("ip")
        if isinstance(ip, list):
            ip = next((i for i in ip if i), None)
        if ip:
            return (ip, 22, "root")
        return None
    ssh = ssh.strip()
    # Tokenized forms: "ssh root@1.2.3.4 -p 22" OR "root@1.2.3.4 -p 22"
    # (Prime Intellect's sshConnection field uses the latter, no leading "ssh").
    if "@" in ssh and (" -p " in f" {ssh} " or ssh.startswith("ssh ")):
        parts = ssh.split()
        user_host = None
        port = 22
        for i, p in enumerate(parts):
            if p == "-p" and i + 1 < len(parts):
                try:
                    port = int(parts[i + 1])
                except ValueError:
                    pass
            elif "@" in p and not p.startswith("-"):
                user_host = p
        if user_host:
            user, host = user_host.split("@", 1)
            return (host, port, user)
    if "@" in ssh:
        user_part, host_part = ssh.split("@", 1)
        if ":" in host_part:
            host, port_s = host_part.rsplit(":", 1)
            return (host, int(port_s), user_part)
        return (host_part, 22, user_part)
    if ":" in ssh:
        host, port_s = ssh.rsplit(":", 1)
        return (host, int(port_s), "root")
    return (ssh, 22, "root")


def submit_remote_training(
    training_code: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    api_key: Optional[str] = None,
    max_price_usd: float = 5.0,
    timeout_s: int = 1800,
    gpu_preferences: Optional[List[str]] = None,
) -> RemoteTrainResult:
    """Submit a training job to a Prime Intellect GPU pod.

    Lifecycle:
      1. Find best available GPU offer
      2. Create pod
      3. Wait for ACTIVE
      4. Upload data + training script via SSH
      5. Run training
      6. Download results
      7. Delete pod (always, even on error)

    Returns RemoteTrainResult with predictions or error.
    """
    t0 = time.time()

    pod_id = None
    owns_pod = False  # only pods WE create are torn down; user-deployed pods stay alive
    gpu_type = "unknown"

    # 0. Reuse an already-deployed ACTIVE pod if one exists (no new charge, instant).
    deployed = list_active_pods(api_key=api_key)
    reused_pod: Optional[Dict] = None
    if deployed:
        reused_pod = deployed[0]
        pod_id = reused_pod.get("id") or reused_pod.get("pod_id") or reused_pod.get("podId")
        gpu_type = reused_pod.get("gpuName") or reused_pod.get("gpuType", "unknown")

    try:
        if reused_pod is not None:
            # Attach to the existing pod; it is already ACTIVE.
            active = dict(reused_pod)
            active["ok"] = True
        else:
            # 1. Find offer + provision a new pod (we own it -> we delete it).
            offer = find_best_gpu_offer(
                api_key=api_key,
                gpu_preferences=gpu_preferences,
                max_price_usd=max_price_usd,
            )
            if offer is None:
                return RemoteTrainResult(
                    ok=False, error="no GPU pods available (no deployed pod and stock empty)",
                    wall_seconds=time.time() - t0,
                )
            gpu_type = offer.get("gpuType", "unknown")

            # 2. Create pod
            pod_info = create_gpu_pod(offer, api_key=api_key)
            if not pod_info.get("ok"):
                return RemoteTrainResult(
                    ok=False, error=f"pod creation failed: {pod_info.get('error', '')}",
                    wall_seconds=time.time() - t0, gpu_type=gpu_type,
                )
            pod_id = pod_info.get("id") or pod_info.get("pod_id") or pod_info.get("podId")
            if not pod_id:
                return RemoteTrainResult(
                    ok=False, error="pod created but no ID returned",
                    wall_seconds=time.time() - t0, gpu_type=gpu_type,
                )
            owns_pod = True

            # 3. Wait for ACTIVE
            active = wait_for_pod_active(pod_id, api_key=api_key, timeout=min(timeout_s // 2, 600))
            if not active.get("ok"):
                return RemoteTrainResult(
                    ok=False, error=f"pod never became active: {active.get('error', '')}",
                    wall_seconds=time.time() - t0, pod_id=pod_id, gpu_type=gpu_type,
                )

        # 4. SSH connection
        ssh_info = _parse_ssh(active)
        if ssh_info is None:
            return RemoteTrainResult(
                ok=False, error="no SSH connection info in pod",
                wall_seconds=time.time() - t0, pod_id=pod_id, gpu_type=gpu_type,
            )

        host, port, user = ssh_info

        # 5. Upload and execute via SSH
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save data
            np.save(os.path.join(tmpdir, "X_train.npy"), X_train)
            np.save(os.path.join(tmpdir, "y_train.npy"), y_train)
            np.save(os.path.join(tmpdir, "X_val.npy"), X_val)
            np.save(os.path.join(tmpdir, "y_val.npy"), y_val)

            # Write training script
            script = _build_remote_script(training_code)
            script_path = os.path.join(tmpdir, "train.py")
            with open(script_path, "w") as f:
                f.write(script)

            # Port flag differs: ssh uses lowercase -p, scp uses uppercase -P. Keep the
            # common opts (identity + host-key + timeout) shared and append the right port flag
            # per tool, so scp does not misread the port as a 'preserve-times' flag + filename.
            common_opts = _ssh_identity_opts() + [
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=30",
            ]
            ssh_opts = common_opts + ["-p", str(port)]
            scp_opts = common_opts + ["-P", str(port)]

            # Upload files
            scp_cmd = ["scp"] + scp_opts + [
                os.path.join(tmpdir, "X_train.npy"),
                os.path.join(tmpdir, "y_train.npy"),
                os.path.join(tmpdir, "X_val.npy"),
                os.path.join(tmpdir, "y_val.npy"),
                script_path,
                f"{user}@{host}:/tmp/",
            ]
            upload = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
            if upload.returncode != 0:
                return RemoteTrainResult(
                    ok=False, error=f"SCP upload failed: {upload.stderr[:200]}",
                    wall_seconds=time.time() - t0, pod_id=pod_id, gpu_type=gpu_type,
                )

            # Run training
            train_timeout = max(timeout_s - int(time.time() - t0), 60)
            ssh_cmd = ["ssh"] + ssh_opts + [
                f"{user}@{host}",
                f"cd /tmp && python train.py",
            ]
            train_proc = subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=train_timeout,
            )
            logs = (train_proc.stdout or "") + (train_proc.stderr or "")

            if train_proc.returncode != 0:
                return RemoteTrainResult(
                    ok=False, error=f"training failed: {logs[-500:]}",
                    wall_seconds=time.time() - t0, pod_id=pod_id,
                    gpu_type=gpu_type, logs=logs,
                )

            # Download predictions
            dl_cmd = ["scp"] + scp_opts + [
                f"{user}@{host}:/tmp/predictions.npy",
                os.path.join(tmpdir, "predictions.npy"),
            ]
            dl = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=60)
            if dl.returncode != 0:
                return RemoteTrainResult(
                    ok=False, error=f"result download failed: {dl.stderr[:200]}",
                    wall_seconds=time.time() - t0, pod_id=pod_id,
                    gpu_type=gpu_type, logs=logs,
                )

            preds = np.load(os.path.join(tmpdir, "predictions.npy"))

            # Parse score from logs
            score = 0.0
            for line in logs.splitlines():
                if line.startswith("ATTESTRA_SCORE="):
                    try:
                        score = float(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass

            return RemoteTrainResult(
                ok=True, score=score, predictions=preds,
                wall_seconds=time.time() - t0, pod_id=pod_id,
                gpu_type=gpu_type, logs=logs,
            )

    finally:
        # 7. Tear down ONLY pods we provisioned. A user-deployed pod we attached
        #    to is left running so the user keeps control of (and billing for) it.
        if pod_id and owns_pod:
            try:
                delete_gpu_pod(pod_id, api_key=api_key)
            except Exception:
                pass


def _build_remote_script(training_code: str) -> str:
    """Build the full training script to run on the remote GPU pod."""
    return textwrap.dedent("""\
        import numpy as np
        import sys
        import traceback

        # Load data
        X_train = np.load("/tmp/X_train.npy")
        y_train = np.load("/tmp/y_train.npy")
        X_val = np.load("/tmp/X_val.npy")
        y_val = np.load("/tmp/y_val.npy")

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Device: {device}")
            if device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        except ImportError:
            print("torch not available, using sklearn only")

        # User training code
    """) + training_code + textwrap.dedent("""

        # Save predictions
        try:
            preds = solve(X_train, y_train, X_val)
            np.save("/tmp/predictions.npy", preds)

            # Compute score
            from sklearn.metrics import accuracy_score, r2_score
            if len(set(y_val)) <= 30:
                score = accuracy_score(y_val, preds)
            else:
                score = r2_score(y_val, preds)
            print(f"ATTESTRA_SCORE={score}")
        except Exception as e:
            traceback.print_exc()
            sys.exit(1)
    """)


# ------------------------------------------------------------------------------- unified dispatch

def dispatch_gpu_training(
    training_code: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    api_key: Optional[str] = None,
    prefer_local: bool = True,
    max_remote_price_usd: float = 5.0,
    timeout_s: int = 1800,
) -> RemoteTrainResult:
    """Dispatch a training job to the best available GPU.

    Priority: local GPU > remote Prime Intellect pod > honest decline.
    """
    # 1. Try local GPU
    if prefer_local:
        try:
            import torch
            if torch.cuda.is_available() or (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                return _run_local_gpu(
                    training_code, X_train, y_train, X_val, y_val,
                )
        except ImportError:
            pass

    # 2. Try remote GPU
    return submit_remote_training(
        training_code, X_train, y_train, X_val, y_val,
        api_key=api_key,
        max_price_usd=max_remote_price_usd,
        timeout_s=timeout_s,
    )


def _run_local_gpu(
    training_code: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> RemoteTrainResult:
    """Run training locally with GPU. Uses subprocess for isolation."""
    import subprocess
    import tempfile

    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "X_train.npy"), X_train)
        np.save(os.path.join(tmpdir, "y_train.npy"), y_train)
        np.save(os.path.join(tmpdir, "X_val.npy"), X_val)
        np.save(os.path.join(tmpdir, "y_val.npy"), y_val)

        script = _build_remote_script(training_code).replace(
            "/tmp/X_train.npy", os.path.join(tmpdir, "X_train.npy"),
        ).replace(
            "/tmp/y_train.npy", os.path.join(tmpdir, "y_train.npy"),
        ).replace(
            "/tmp/X_val.npy", os.path.join(tmpdir, "X_val.npy"),
        ).replace(
            "/tmp/y_val.npy", os.path.join(tmpdir, "y_val.npy"),
        ).replace(
            "/tmp/predictions.npy", os.path.join(tmpdir, "predictions.npy"),
        )

        script_path = os.path.join(tmpdir, "train.py")
        with open(script_path, "w") as f:
            f.write(script)

        proc = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, timeout=600,
        )
        logs = (proc.stdout or "") + (proc.stderr or "")

        if proc.returncode != 0:
            return RemoteTrainResult(
                ok=False, error=f"local GPU training failed: {logs[-500:]}",
                wall_seconds=time.time() - t0, gpu_type="local",
                logs=logs,
            )

        pred_path = os.path.join(tmpdir, "predictions.npy")
        if not os.path.exists(pred_path):
            return RemoteTrainResult(
                ok=False, error="no predictions produced",
                wall_seconds=time.time() - t0, gpu_type="local",
                logs=logs,
            )

        preds = np.load(pred_path)
        score = 0.0
        for line in logs.splitlines():
            if line.startswith("ATTESTRA_SCORE="):
                try:
                    score = float(line.split("=", 1)[1].strip())
                except ValueError:
                    pass

        return RemoteTrainResult(
            ok=True, score=score, predictions=preds,
            wall_seconds=time.time() - t0, gpu_type="local",
            logs=logs,
        )
