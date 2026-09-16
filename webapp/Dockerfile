# VectorForge landing page + autoresearch loop (CPU in-process; GPU fan-out via the RunPod HTTP API).
# Light image: no torch (CPU uses sklearn; GPU runs remotely on RunPod). Secrets are passed as ENV at
# runtime (never baked into the image) -- see .dockerignore and the run command in webapp/DEPLOY.md.
FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir "numpy<2" scikit-learn requests anthropic

# the platform + frozen core + the web app (the external `ar` autoresearch pkg is NOT needed: the UI always
# passes an explicit kind, so the llm_shell/ar inference path is never imported)
COPY vfplatform /app/vfplatform
COPY vectorforge /app/vectorforge
COPY attestra /app/attestra
COPY frontier /app/frontier
COPY webapp /app/webapp

ENV VF_HOST=0.0.0.0 VF_PORT=8765 PYTHONUNBUFFERED=1
EXPOSE 8765

# RUNPOD_API_KEY + RUNPOD_ENDPOINT_ID (for the optional GPU toggle) and ANTHROPIC_API_KEY (for free-text
# goal parsing) are supplied at `docker run` time via -e, not copied in.
CMD ["python", "/app/webapp/server.py"]
