# ── Base Image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System Dependencies ────────────────────────────────────────────────────
# ffmpeg: video re-encoding in pipeline.py
# libgl/libglib: OpenCV headless requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python Dependencies ────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Application Code ─────────────────────────────────────────────────
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY templates/ ./templates/
COPY model/ ./model/
COPY start.sh .

RUN chmod +x start.sh

# ── Create Data Directories ───────────────────────────────────────────────
RUN mkdir -p /data/jobs /data/models

# ── Environment Variables ──────────────────────────────────────────────────
ENV DETECTION_DATA_DIR=/data/jobs \
    MODELS_DIR=/data/models \
    BACKEND_URL=http://127.0.0.1:8000 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000

# ── Ports ──────────────────────────────────────────────────────────────────
# 5000 = Flask frontend (user-facing)
# 8000 = FastAPI backend (internal, but exposed for debugging)
EXPOSE 5000 8000

# ── Startup ────────────────────────────────────────────────────────────────
CMD ["./start.sh"]