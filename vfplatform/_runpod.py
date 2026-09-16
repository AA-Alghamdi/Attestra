"""RunPod credential loading + read-only connectivity, and the (Checkpoint-gated) serverless REST client.

Key resolution: explicit arg -> RUNPOD_API_KEY env -> gitignored <repo>/.runpod_key (or $VF_RUNPOD_KEY_FILE).
NOTHING here spends money on its own: `health()` is a read-only GraphQL `myself` query; job submission lives
in providers.RunPodProvider and is gated by the Checkpoint.
"""
import json
import os

_VF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHQL_URL = "https://api.runpod.io/graphql"


def _key_file():
    return os.environ.get("VF_RUNPOD_KEY_FILE", os.path.join(_VF, ".runpod_key"))


def resolve_runpod_key(explicit=None):
    if explicit:
        return explicit
    env = os.environ.get("RUNPOD_API_KEY")
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


def graphql(query, *, api_key=None, timeout=30):
    """Run a GraphQL query against the RunPod API. Returns parsed JSON. Read-only by query choice."""
    import requests
    key = api_key or resolve_runpod_key()
    if not key:
        raise RuntimeError("no RunPod API key (set RUNPOD_API_KEY or .runpod_key)")
    r = requests.post(f"{GRAPHQL_URL}?api_key={key}", json={"query": query}, timeout=timeout,
                      headers={"Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()


def health(api_key=None):
    """Read-only auth + account check (NO spend). Returns {ok, account|error}."""
    try:
        data = graphql("query { myself { id email currentSpendPerHr } }", api_key=api_key)
    except Exception as ex:  # noqa: BLE001
        return {"ok": False, "error": str(ex)[:200]}
    if data.get("errors"):
        return {"ok": False, "error": str(data["errors"])[:200]}
    me = (data.get("data") or {}).get("myself") or {}
    return {"ok": bool(me), "account": {"id": me.get("id"),
                                        "email": me.get("email"),
                                        "current_spend_per_hr": me.get("currentSpendPerHr")}}


def gpu_types(api_key=None):
    """Read-only: list available GPU type ids (NO spend). Use these ids for endpoint gpuIds."""
    try:
        data = graphql("query { gpuTypes { id displayName memoryInGb secureCloud communityCloud } }",
                       api_key=api_key)
    except Exception as ex:  # noqa: BLE001
        return {"ok": False, "error": str(ex)[:200]}
    if data.get("errors"):
        return {"ok": False, "error": str(data["errors"])[:200]}
    return {"ok": True, "gpus": (data.get("data") or {}).get("gpuTypes") or []}


# ----------------------------------------------------------------------------- provisioning (mutations)
def _mutation(query, *, api_key=None, dry_run=False, timeout=60):
    if dry_run:
        return {"dry_run": True, "mutation": query}
    return graphql(query, api_key=api_key, timeout=timeout)


def find_or_create_template(*, name, image, container_disk_gb=20, env=None, api_key=None, dry_run=False):
    """Idempotent: reuse a serverless template by name, else create it (saveTemplate). Free (no GPU)."""
    existing = graphql("query { myself { podTemplates { id name imageName } } }", api_key=api_key) \
        if not dry_run else {"data": {"myself": {"podTemplates": []}}}
    for t in (((existing.get("data") or {}).get("myself") or {}).get("podTemplates") or []):
        if t.get("name") == name:
            return {"ok": True, "id": t["id"], "reused": True}
    env_str = ", ".join(f'{{ key: "{k}", value: "{v}" }}' for k, v in (env or {}).items())
    q = ('mutation { saveTemplate(input: { '
         f'name: "{name}", imageName: "{image}", isServerless: true, '
         f'containerDiskInGb: {int(container_disk_gb)}, volumeInGb: 0, dockerArgs: "", ports: "", '
         f'env: [{env_str}] '
         '}) { id name imageName } }')
    res = _mutation(q, api_key=api_key, dry_run=dry_run)
    if dry_run:
        return {"ok": True, "dry_run": True, "mutation": q}
    if res.get("errors"):
        return {"ok": False, "error": str(res["errors"])[:300]}
    t = (res.get("data") or {}).get("saveTemplate") or {}
    return {"ok": bool(t.get("id")), "id": t.get("id"), "reused": False}


def find_or_create_endpoint(*, name, template_id, gpu_ids="NVIDIA RTX A4000", workers_min=0, workers_max=2,
                            idle_timeout=5, api_key=None, dry_run=False):
    """Idempotent: reuse a serverless endpoint by name, else create it (saveEndpoint). Endpoint EXISTENCE
    is free; cost accrues only when a job runs (workers_min=0 => scale to zero)."""
    if not dry_run:
        eps = list_endpoints(api_key=api_key)
        for e in (eps.get("endpoints") or []):
            if e.get("name") == name:
                return {"ok": True, "id": e["id"], "reused": True}
    q = ('mutation { saveEndpoint(input: { '
         f'name: "{name}", templateId: "{template_id}", gpuIds: "{gpu_ids}", '
         f'workersMin: {int(workers_min)}, workersMax: {int(workers_max)}, '
         f'idleTimeout: {int(idle_timeout)}, scalerType: "QUEUE_DELAY", scalerValue: 4, '
         'locations: "", networkVolumeId: "" '
         '}) { id name } }')
    res = _mutation(q, api_key=api_key, dry_run=dry_run)
    if dry_run:
        return {"ok": True, "dry_run": True, "mutation": q}
    if res.get("errors"):
        return {"ok": False, "error": str(res["errors"])[:300]}
    e = (res.get("data") or {}).get("saveEndpoint") or {}
    return {"ok": bool(e.get("id")), "id": e.get("id"), "reused": False}


def list_endpoints(api_key=None):
    """Read-only: list serverless endpoints on the account (NO spend)."""
    try:
        data = graphql("query { myself { endpoints { id name templateId } } }", api_key=api_key)
    except Exception as ex:  # noqa: BLE001
        return {"ok": False, "error": str(ex)[:200]}
    if data.get("errors"):
        return {"ok": False, "error": str(data["errors"])[:200]}
    eps = ((data.get("data") or {}).get("myself") or {}).get("endpoints") or []
    return {"ok": True, "endpoints": eps}
