# VectorForge GPU fan-out worker (RunPod Serverless) -- LIGHT base.
# python:3.11-slim + pip torch (the default Linux torch wheel bundles CUDA, so GPU works on a RunPod GPU)
# keeps the image ~4-5GB (fits a small Docker VM disk) instead of the ~20GB runpod/pytorch devel image.
# Build context must contain handler.py and torch_models.py.
FROM python:3.11-slim

WORKDIR /app
# Pin torch to the CUDA 12.4 runtime build. The default `pip install torch` now resolves to a CUDA 13
# wheel, whose bundled runtime requires a host NVIDIA driver newer than RunPod's Ampere/Ada GPU fleet
# provides -> torch.cuda.is_available() is False and the worker silently falls back to CPU. cu124 matches
# the driver fleet (550+), so the GPU is actually used. (Verified via the handler's _cuda_diag output.)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir scikit-learn "numpy<2" runpod

COPY torch_models.py /app/torch_models.py
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
