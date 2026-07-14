FROM python:3.11-slim

# ── CPU / GPU switch (build-time) ───────────────────────────────────────────
# TORCH_VARIANT=auto (default) installs PyTorch's standard PyPI wheel, which
#   bundles CUDA support and works on BOTH machine types: it auto-detects a
#   GPU at runtime (torch.cuda.is_available()) and falls back to CPU
#   correctly if there isn't one. Simplest option - one image runs anywhere.
# TORCH_VARIANT=cpu forces the CPU-only wheel instead (~600MB smaller image).
#   Use this if you know this image will only ever run on CPU-only machines
#   and want a leaner build.
#
#   docker build -t detection-app .                              # auto (default)
#   docker build --build-arg TORCH_VARIANT=cpu -t detection-app:cpu .
ARG TORCH_VARIANT=auto

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

COPY requirements.txt ./requirements.txt

# Install torch FIRST so the TORCH_VARIANT choice above wins; ultralytics
# (installed next, via requirements.txt) will see a compatible torch
# already present and won't pull in a different build on top of it.
RUN if [ "$TORCH_VARIANT" = "cpu" ]; then \
        echo "Installing CPU-only torch..." && \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu ; \
    else \
        echo "Installing default torch (CUDA-capable, auto-detects GPU at runtime)..." && \
        pip install --no-cache-dir torch torchvision ; \
    fi

RUN pip install --no-cache-dir -r requirements.txt

# Backend (FastAPI) - main.py uses relative imports, so it lives in a
# package directory and is run as `app.main`
COPY app ./app
COPY model ./model

# Frontend (Flask) - proxies to the backend over localhost inside this
# same container, so BACKEND_URL stays on its default (127.0.0.1:8000)
COPY flask_app.py ./flask_app.py
COPY templates ./templates

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
