# syntax=docker/dockerfile:1

# ── Stage 1: build ───────────────────────────────────────────────────────
# Installs Python deps (including compiling anything that needs a
# compiler) into an isolated prefix. None of this stage's layers end up
# in the final image, so build tools and pip's download/cache files never
# add to the shipped size.
FROM python:3.11-slim AS builder

WORKDIR /srv

# No build-essential needed here: torch/torchvision, opencv-python-headless,
# numpy, pandas, scipy, and ultralytics all ship as pre-built manylinux
# wheels for Python 3.11, so nothing in requirements.txt needs a compiler.
# Skipping it also avoids apt-get pulling in gcc/g++ (~300+ MB and a slow,
# memory-heavy install).

COPY requirements.txt ./requirements.txt

# Install CPU-only torch/torchvision FIRST and pinned, so that when pip
# resolves requirements.txt afterwards it sees torch already satisfied
# and does not fall back to pulling the default CUDA-enabled wheel that
# ultralytics depends on. This is the single biggest size saving (the
# CUDA wheel + its bundled NVIDIA libs run several GB heavier than CPU).
RUN pip install --no-cache-dir --user \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Strip bytecode caches and packaged docs/tests that some wheels ship,
# to shrink what gets copied into the final stage.
RUN find /root/.local -type d -name "__pycache__" -exec rm -rf {} + \
    && find /root/.local -type d -name "tests" -exec rm -rf {} + \
    && find /root/.local -type d -name "*.dist-info" -exec sh -c \
        'rm -f "$1"/RECORD "$1"/INSTALLER "$1"/direct_url.json' _ {} \;

# ── Stage 2: runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /srv

# System dependencies:
#   ffmpeg               - used by pipeline.py to re-encode output video to
#                          H.264 via subprocess
#   libgl1, libglib2.0-0 - required by opencv-python-headless / ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Bring over only the installed Python packages from the builder stage -
# no compiler, no pip cache, no source downloads.
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/root/.local/lib/python3.11/site-packages

# Backend (FastAPI) - main.py uses relative imports, so it lives in a
# package directory and is run as `app.main`
COPY app ./app
COPY model ./model

# Frontend (Flask) - proxies to the backend over localhost inside this
# same container, so BACKEND_URL stays on its default (127.0.0.1:8000)
COPY frontend/flask_app.py ./flask_app.py
COPY frontend/templates ./templates

COPY start.sh ./start.sh
RUN chmod +x ./start.sh

ENV DETECTION_DATA_DIR=/tmp/detection_jobs \
    MODELS_DIR=/tmp/detection_jobs/models \
    BACKEND_URL=http://127.0.0.1:8000 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /tmp/detection_jobs/models

# 8000 = FastAPI backend (direct API access), 5000 = Flask UI
EXPOSE 8000 5000

CMD ["./start.sh"]
