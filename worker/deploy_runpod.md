# Deploying the VectorForge GPU worker to RunPod Serverless

This is the one collaborative step (it needs Docker + a registry login + your RunPod account). After it,
GPU fan-out runs through `RunPodProvider` with no change to the loop, the certifier, or the moat.

## What runs where
- The **worker** (this dir) trains a candidate on the GPU and returns the **validation score** (fan-out)
  or the **sealed-test predictions** (winner). It NEVER certifies.
- The **frozen certifier runs locally** on the returned predictions - Ring 0 never leaves your machine.

## 0. Prereqs
- Docker (installed locally), a container registry you can push to (Docker Hub: `docker login`).
- Your RunPod key is already in `../.runpod_key` (read-only check passes: `RunPodProvider().health()`).

## 1. Build the worker image
```bash
cd vectorforge-product/worker
cp ../vfplatform/torch_models.py .            # the torch families the worker needs
docker build -t <your-dockerhub-user>/vectorforge-worker:latest .
```

## 2. Push it
```bash
docker login
docker push <your-dockerhub-user>/vectorforge-worker:latest
```

## 3. Create a serverless endpoint
- Console: RunPod -> Serverless -> New Endpoint -> Container image = `<user>/vectorforge-worker:latest`,
  pick a small GPU (e.g. A40 / RTX A4000), workers min 0 / max 1-2, then Create. Copy the **Endpoint ID**.
- (Or via the GraphQL API; the console is simplest for the first one.)

## 4. Point VectorForge at it
```bash
export RUNPOD_ENDPOINT_ID=<the endpoint id>
```
Now `RunPodProvider().available()` is True and `capabilities()["gated"]` is False.

## 5. Run a tiny validated GPU job (one approval)
```python
from vfplatform.providers import RunPodProvider
p = RunPodProvider()
out = p.submit_one({"op": "fit_val", "task_type": "binary", "family": "torch_mlp_clf|64",
                    "params": {"hidden": "64", "epochs": 30}, "seed": 0,
                    "Xtr": [[0.1,0.2],[0.9,0.8]]*50, "ytr": [0,1]*50,
                    "Xva": [[0.1,0.2],[0.9,0.8]]*20, "yva": [0,1]*20})
print(out)   # {"val_score":..., "device":"cuda", ...}  <- confirms GPU
```
This is the first real spend; the loop gates it behind the Checkpoint (approve_spend).

## Cost & safety
- Billing is per-second of worker execution. Min-workers 0 means no idle cost.
- The loop's `Checkpoint` blocks any RunPod move unless `approve_spend=True`, so GPU fan-out never runs
  without your explicit go-ahead.
