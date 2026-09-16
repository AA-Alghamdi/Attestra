"""Prime Intellect API client — GPU Pod management + inference key resolution.

Key resolution: explicit arg -> PRIME_INTELLECT_API_KEY env -> gitignored <repo>/.prime_intellect_key.
Pod lifecycle: availability → create → poll ACTIVE → SSH exec → delete (teardown).
Inference: OpenAI-compatible at https://api.pinference.ai/api/v1.

NOTHING here spends money on its own except create_pod + run commands on the pod.
"""
import os
import time

_VF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PI_API_BASE = "https://api.primeintellect.ai/api/v1"
PI_INFERENCE_BASE = "https://api.pinference.ai/api/v1"

# Preferred GPU types in order of cost-efficiency for short ML training jobs
_GPU_PREFERENCES = ["A100_80GB", "H100_80GB", "A100_40GB", "A6000"]
_GPU_COST = {"A100_80GB": 2.0, "H100_80GB": 3.5, "A100_40GB": 1.5, "A6000": 0.8}


def _key_file():
    return os.environ.get("VF_PI_KEY_FILE", os.path.join(_VF, ".prime_intellect_key"))


def resolve_pi_key(explicit=None):
    """Resolve Prime Intellect API key: explicit -> env -> key file."""
    if explicit:
        return explicit
    env = os.environ.get("PRIME_INTELLECT_API_KEY")
    if env:
        return env
    path = _key_file()
    try:
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#") and "PASTE_YOUR" not in s:
                    return s
    except (OSError, IOError):
        return None
    return None


def _headers(api_key):
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


# ----------------------------------------------------------------------------- GPU availability
def gpu_availability(gpu_type=None, *, api_key=None, regions=None, gpu_count=1, timeout=30):
    """Query GPU availability (read-only, no spend). Returns list of offers."""
    import requests
    key = api_key or resolve_pi_key()
    if not key:
        return {"ok": False, "error": "no Prime Intellect API key"}
    params = {"gpu_count": gpu_count}
    if gpu_type:
        params["gpu_type"] = gpu_type
    if regions:
        params["regions"] = regions
    try:
        r = requests.get(f"{PI_API_BASE}/availability/gpus", params=params,
                         headers=_headers(key), timeout=timeout)
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}: {(r.text or '')[:200]}"}
    try:
        data = r.json()
    except Exception:
        return {"ok": False, "error": "bad JSON response"}
    offers = data if isinstance(data, list) else data.get("data", data.get("offers", []))
    return {"ok": True, "offers": offers if isinstance(offers, list) else []}


def find_best_offer(*, api_key=None, gpu_preferences=None, gpu_count=1, max_price=None):
    """Find the cheapest available GPU from the preference list. Returns offer dict or None."""
    prefs = gpu_preferences or _GPU_PREFERENCES
    for gpu_type in prefs:
        result = gpu_availability(gpu_type, api_key=api_key, gpu_count=gpu_count)
        if not result.get("ok"):
            continue
        offers = result.get("offers", [])
        available = [o for o in offers if o.get("stockStatus") == "Available"]
        if max_price:
            available = [o for o in available
                         if (o.get("prices", {}).get("onDemand") or 999) <= max_price]
        if available:
            available.sort(key=lambda o: o.get("prices", {}).get("onDemand", 999))
            return available[0]
    return None


# ----------------------------------------------------------------------------- Pod lifecycle
def create_pod(offer, *, name=None, image="cuda_12_1_pytorch_2_2", disk_size=50,
               api_key=None, timeout=60):
    """Create a pod from an availability offer. Returns APIPodConfig dict or error."""
    import requests
    key = api_key or resolve_pi_key()
    if not key:
        return {"ok": False, "error": "no Prime Intellect API key"}
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
        "provider": {
            "type": offer.get("provider", "hyperstack")
        }
    }
    dc = offer.get("dataCenter") or offer.get("dataCenterId")
    if dc:
        body["pod"]["dataCenterId"] = dc
    country = offer.get("country")
    if country:
        body["pod"]["country"] = country
    security = offer.get("security")
    if security:
        body["pod"]["security"] = security
    try:
        r = requests.post(f"{PI_API_BASE}/pods/", json=body, headers=_headers(key), timeout=timeout)
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    if r.status_code not in (200, 201):
        return {"ok": False, "error": f"HTTP {r.status_code}: {(r.text or '')[:300]}"}
    try:
        return {"ok": True, **r.json()}
    except Exception:
        return {"ok": False, "error": "bad JSON in create response"}


def get_pod(pod_id, *, api_key=None, timeout=30):
    """Get pod status/details (read-only)."""
    import requests
    key = api_key or resolve_pi_key()
    if not key:
        return {"ok": False, "error": "no Prime Intellect API key"}
    try:
        r = requests.get(f"{PI_API_BASE}/pods/{pod_id}", headers=_headers(key), timeout=timeout)
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}: {(r.text or '')[:200]}"}
    try:
        return {"ok": True, **r.json()}
    except Exception:
        return {"ok": False, "error": "bad JSON"}


def delete_pod(pod_id, *, api_key=None, timeout=30):
    """Terminate and delete a pod."""
    import requests
    key = api_key or resolve_pi_key()
    if not key:
        return {"ok": False, "error": "no Prime Intellect API key"}
    try:
        r = requests.delete(f"{PI_API_BASE}/pods/{pod_id}", headers=_headers(key), timeout=timeout)
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    if r.status_code not in (200, 204):
        return {"ok": False, "error": f"HTTP {r.status_code}: {(r.text or '')[:200]}"}
    return {"ok": True}


def wait_for_active(pod_id, *, api_key=None, timeout=600, poll_interval=10):
    """Poll pod until status=ACTIVE + installationStatus=FINISHED. Returns pod dict or error."""
    deadline = time.time() + timeout
    last_status = "unknown"
    while time.time() < deadline:
        pod = get_pod(pod_id, api_key=api_key)
        if not pod.get("ok"):
            return pod
        status = pod.get("status", "UNKNOWN")
        install = pod.get("installationStatus", "PENDING")
        last_status = f"{status}/{install}"
        if status == "ERROR":
            return {"ok": False, "error": f"pod entered ERROR state: {pod.get('installationFailure', '')}"}
        if status == "TERMINATED":
            return {"ok": False, "error": "pod was terminated"}
        if status == "ACTIVE" and install == "FINISHED":
            return pod
        time.sleep(poll_interval)
    return {"ok": False, "error": f"pod not ACTIVE within {timeout}s (last: {last_status})"}


def parse_ssh_connection(pod_info):
    """Extract (host, port, user) from pod's sshConnection field.
    Handles formats: 'ssh user@host -p port' or 'user@host:port' or just 'host:port'."""
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
    if ssh.startswith("ssh "):
        parts = ssh.split()
        user_host = None
        port = 22
        for i, p in enumerate(parts):
            if p == "-p" and i + 1 < len(parts):
                port = int(parts[i + 1])
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


# ----------------------------------------------------------------------------- Inference
def inference_health(*, api_key=None, timeout=15):
    """Check inference endpoint reachability by listing models (read-only, no spend)."""
    import requests
    key = api_key or resolve_pi_key()
    if not key:
        return {"ok": False, "error": "no Prime Intellect API key"}
    try:
        r = requests.get(f"{PI_INFERENCE_BASE}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}: {(r.text or '')[:160]}"}
    try:
        data = r.json()
        models = data.get("data", []) if isinstance(data, dict) else []
        return {"ok": True, "models": [m.get("id") for m in models]}
    except Exception:
        return {"ok": True, "models": []}


def list_models(*, api_key=None, timeout=15):
    """List available inference models."""
    return inference_health(api_key=api_key, timeout=timeout)
