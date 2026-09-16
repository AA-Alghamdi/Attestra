# Deploying the Attestera landing page

The app: type a goal + give data (built-in dataset, paste JSON, or **upload a CSV**) + optionally your own
held-out **verification set** → the autoresearch cycle runs (CPU in-process, or **GPU parallel fan-out** on
RunPod via the toggle) → a live 2D workflow shows each step → you get the certified model + a servable
version. Selection on validation; one sealed-test peek; deploy only what is certified.

The GPU panel now exposes two more controls: a **max workers** setting (how many candidates fan out in
parallel) and a **preferred GPU type** selector (so you can pin the RunPod hardware class). A **Stop** button
lets you halt an in-flight run cleanly at any time.

## 1. Local (no Docker)
```
/Users/abdullahalghamdi/jax-env-311/bin/python webapp/server.py
# open http://127.0.0.1:8765
```
GPU toggle uses the gitignored `.runpod_key` + `.runpod_endpoint`. Free-text goals use `.anthropic_key`
(optional; falls back to deterministic inference).

## 2. One-command Docker (local / LAN)
Build once, then run. Secrets are passed as `-e` (never baked into the image).
```
docker build -f webapp/Dockerfile -t vectorforge-web .

docker run --rm -p 8765:8765 \
  -e RUNPOD_API_KEY="$(cat .runpod_key)" \
  -e RUNPOD_ENDPOINT_ID="$(cat .runpod_endpoint)" \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  vectorforge-web
# open http://localhost:8765   (the container binds 0.0.0.0 internally; map the port)
```
- Omit the RunPod env vars to run CPU-only (the GPU toggle then degrades to CPU).
- On a LAN, others reach it at `http://<your-machine-ip>:8765`.

## 3. Public internet: single VM + Docker + Caddy (turnkey)
This is the supported public recipe: `webapp/deploy/` has a `docker-compose.yml` (the app + a **Caddy**
reverse proxy) and a `Caddyfile`. Caddy gives you **automatic HTTPS** (Let's Encrypt) and an **HTTP
basic-auth** password gate; the app container is reachable **only through Caddy** (its port is internal to
the compose network, never published to the host). A **hard per-day spend cap** is built into the app so a
shared/leaked password can't drain your RunPod/Anthropic budget.

### What you provide
1. **A VM with a public IP** and Docker + the compose plugin (DigitalOcean / Hetzner / EC2 / GCP; a 1 to 2 GB
   box is plenty; the CPU lane is sklearn, GPU runs remotely on RunPod). Open inbound **80** and **443**.
2. **A domain** with an **A record → the VM's IP** (e.g. `vf.example.com`). Caddy needs this to get a cert.

### Steps (run on the VM, from the repo root)
```bash
# a) configure
cp webapp/deploy/.env.example webapp/deploy/.env
#    edit webapp/deploy/.env: set VF_DOMAIN, VF_BASIC_USER, and the spend keys you want live.

# b) make the password hash (paste the $2a$... output into VF_BASIC_HASH; escape each $ as $$ in .env)
docker run --rm caddy:2 caddy hash-password --plaintext 'choose-a-strong-password'

# c) launch (build context is the repo root so it sees vfplatform/ vectorforge/ webapp/)
docker compose --env-file webapp/deploy/.env -f webapp/deploy/docker-compose.yml up -d --build

# d) verify
docker compose -f webapp/deploy/docker-compose.yml ps          # both services healthy
curl -u USER:PASS https://vf.example.com/healthz               # {"ok":true,"spend":{...}}
```
Open `https://vf.example.com`, enter the password, and the live UI is public. To update after a code change:
`git pull && docker compose -f webapp/deploy/docker-compose.yml up -d --build`.

### The spend backstop (`webapp/spendcap.py`)
Every GPU fan-out run and every paid LLM-intake call is counted per **UTC day**; when a ceiling is reached
the app **degrades honestly** (GPU → CPU, AI-goal → deterministic) and the UI says "capped"; it never
silently overspends or fakes a certificate. Defaults (override in `.env`): **$5.00/day**, **100 GPU
runs/day**, **200 LLM calls/day**. Charges happen **only when the spend is real** (a RunPod provider is live
/ an Anthropic key is set); otherwise those lanes are free local fallbacks and aren't counted. Live counters:
`GET /healthz`. State persists in the `vf_runs` volume.

## Notes
- GPU spend: ticking the GPU box is the spend consent; the endpoint scales to zero when idle. The per-day cap
  is the hard ceiling on top of that consent.
- Secrets are never written to the image (see `.dockerignore`) and never committed (`.env` is gitignored);
  they're injected at runtime via `.env` → container env.
- The bundled stdlib server is fine behind Caddy for demo/low traffic; for heavy concurrent load put
  gunicorn/uvicorn in front (the loop itself runs each request on its own thread).
- TLS certs persist in the `caddy_data` volume, so restarts won't re-hit Let's Encrypt rate limits.
