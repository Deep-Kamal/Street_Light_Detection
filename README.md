# Detection Pipeline — Street Light Detection Platform

A self-hosted, Dockerised web application for GPS-referenced street light detection and generic YOLO object detection from vehicle-mounted survey video. Upload a video (and, for street light jobs, a GPS track recorded during the same drive) and get back an annotated video, a structured per-detection CSV with GPS coordinates, and a curated set of best-quality snapshot images — one per physically distinct object encountered.

Built and maintained at the **Remote Sensing Applications Centre, Uttar Pradesh (RSAC-UP)**, an autonomous organisation under the Department of Science & Technology, Government of Uttar Pradesh.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [Option 1 — Docker (recommended)](#option-1--docker-recommended)
  - [Option 2 — Manual / Local Setup](#option-2--manual--local-setup)
- [Usage](#usage)
- [REST API](#rest-api)
- [Configuration](#configuration)
- [Testing](#testing)
- [Acknowledgments](#acknowledgments)

---

## Overview

The platform is a two-tier web application:

- A **FastAPI backend** that owns detection-model registration, asynchronous job orchestration, video processing, and result delivery.
- A **Flask frontend** that renders the operator web UI and transparently proxies every request to the backend.

The primary pipeline (`model_type: streetlight`) combines YOLO object tracking with a monocular pinhole-camera distance estimate, a GPS-interpolation model, and compass-bearing calculation to convert raw video detections into a geo-referenced street light inventory. A secondary, model-agnostic pipeline (`model_type: yolo_detection` / `yolo_segmentation` / `custom`) supports any Ultralytics-compatible weights file for straightforward detection of other classes (e.g. potholes) without writing new pipeline code.

## Features

- 🚦 **GPS-referenced street light detection** — distance, elevation angle, and compass bearing computed for every detection.
- 🗺️ **Interactive detection map** — plot every logged detection from a completed job on a Leaflet/OpenStreetMap view.
- 🧩 **Multi-model registry** — register and switch between multiple detection models (street light, potholes, or any custom YOLO weights) from the same UI.
- 🎯 **Per-object best snapshot** — keeps only the closest-range frame of each tracked object, so output size scales with distinct objects, not video frames.
- 📦 **One-click downloads** — annotated video, detections CSV, snapshot ZIP, or a combined "all outputs" bundle.
- 🐳 **Single-container deployment** — both services ship together in one Docker image.

## Screenshots

**Dashboard — idle state, before any file is uploaded**

![Dashboard idle state](screenshots/dashboard-idle.png)

**Job completed — annotated video playback with GPS/compass HUD**

The bounding box, tracked object ID, confidence score, and the live GPS/compass heads-up display (`LAT`, `LON`, `DIR`, `FRAME`) are all rendered directly on the output video.

![Detection video playback](screenshots/detection-video-playback.png)

**Job completed — results panel with map reset to default extent**

![Dashboard with completed job, map reset](screenshots/dashboard-map-reset.png)

**Detections plotted on the GPS map**

Selecting **Plot on Map** re-centres the map on the survey route and drops a marker for every logged detection.

![Plotted detections on map](screenshots/plotted-map-detection.png)

## Project Structure

```
STREETLIGHT_APP/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI backend — model registry, job orchestration, REST API
│   ├── pipeline.py          # Street light detection pipeline (tracking, GPS, distance, HUD)
│   └── frontend/
│       ├── templates/       # Operator web UI (index.html)
│       └── flask_app.py     # Flask frontend — serves the UI, proxies API calls to the backend
├── jobs/                    # Per-job uploads, outputs, and job registry (created at runtime)
├── model/                   # Detection model weights (.pt), bind-mounted into the container
├── tests/                   # Test suite
├── tmp_debug/                # Scratch/debug output (not required for normal operation)
├── venv/                    # Local Python virtual environment (not committed)
├── .dockerignore
├── .gitignore
├── docker-compose.yml       # Docker Compose service definition
├── Dockerfile                # Single-image build for both backend + frontend
├── requirements.txt          # Python dependencies
└── start.sh                   # Launches both the FastAPI backend and the Flask frontend
```

## Architecture

| Component | Technology | Port | Responsibility |
|---|---|---|---|
| Backend | FastAPI (Uvicorn ASGI) | `8000` | Model registry, job orchestration, video processing, result delivery, map-data aggregation |
| Frontend | Flask | `5000` | Operator web UI, API proxy (including range-request video streaming) |
| Detection | Ultralytics YOLO | — | Object detection + persistent object tracking |

Both services run as two processes inside a single Docker container, launched together by `start.sh`.

## Prerequisites

**For Docker setup (recommended):**
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

**For manual/local setup:**
- Python 3.11+
- `pip`
- [FFmpeg](https://ffmpeg.org/download.html) (required for output video re-encoding)
- `git`
- An NVIDIA GPU + CUDA drivers (optional — speeds up inference, not required)

## Setup

### Option 1 — Docker (recommended)

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd STREETLIGHT_APP
   ```

2. **Add your detection model weights**

   Place your trained `.pt` weights file(s) inside the `model/` directory. The backend auto-registers a default "Street Light Detection" model on startup if a matching weights file is found here.

3. **Build and run using Docker Compose**
   ```bash
   docker-compose up --build -d
   ```

   Or, build and run manually with plain Docker:
   ```bash
   docker build -t detection-app .

   docker run -d -p 5000:5000 -p 8000:8000 \
     -v $(pwd)/model:/srv/model \
     -v $(pwd)/jobs:/tmp/detection_jobs \
     --name detection-app-container detection-app
   ```

4. **Open the app**

   - Operator UI: [http://localhost:5000](http://localhost:5000)
   - Backend API / health check: [http://localhost:8000/health](http://localhost:8000/health)

5. **Stop the app**
   ```bash
   docker-compose down
   ```

### Option 2 — Manual / Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd STREETLIGHT_APP
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv

   # macOS / Linux
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg** (if not already installed)
   ```bash
   # Ubuntu / Debian
   sudo apt-get update && sudo apt-get install -y ffmpeg

   # macOS (Homebrew)
   brew install ffmpeg

   # Windows (Chocolatey)
   choco install ffmpeg
   ```

5. **Add your detection model weights** to the `model/` directory.

6. **Start both services**

   Using the provided script (starts backend + frontend together):
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

   Or start each service manually in separate terminals:
   ```bash
   # Terminal 1 — backend (FastAPI)
   uvicorn app.main:app --host 0.0.0.0 --port 8000

   # Terminal 2 — frontend (Flask)
   python app/frontend/flask_app.py
   ```

7. **Open the app** at [http://localhost:5000](http://localhost:5000)

## Usage

1. Open the operator UI and confirm the correct model is marked **ACTIVE** under **Active Model** (or register a new one via **Add New**).
2. Under **Process Video**, upload your survey video and, for street light jobs, its accompanying GPS track CSV.
3. Click **Start Detection**. The **STATUS** panel will track progress from queued through to `Done (100%)`.
4. Once complete, use the **RESULTS** panel to:
   - **Play Video** — view the annotated output inline.
   - **Plot on Map** — drop every logged detection onto the GPS Detection Map.
   - Download the detections CSV, output video, snapshot ZIP, or a combined bundle.

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/models` | List all registered models |
| `POST` | `/models` | Register a new model |
| `POST` | `/models/{model_id}/activate` | Set a model as active |
| `POST` | `/process_async` | Start an async detection job (returns a `job_id` immediately) |
| `GET` | `/status/{job_id}` | Poll job status/progress and download links |
| `GET` | `/download/{job_id}/{video\|csv\|frames_zip\|all}` | Download a result artefact |
| `GET` | `/jobs/{job_id}/map-data` | GPS points for one completed job |
| `GET` | `/map/all-jobs` | Combined map points across every completed job |
| `GET` | `/health` | Health check |

## Configuration

Key environment variables (set in `docker-compose.yml`, your shell, or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8000` | Backend URL the Flask frontend proxies requests to |
| `DETECTION_DATA_DIR` | `/tmp/detection_jobs` | Directory where job uploads, outputs, and the model/job registries are persisted |

For production deployments, mount `DETECTION_DATA_DIR` and `model/` to persistent Docker volumes so job history and registered models survive container restarts.

## Testing

```bash
pytest tests/
```

## Acknowledgments

Developed at the **Remote Sensing Applications Centre, Uttar Pradesh (RSAC-UP)**, an autonomous organisation under the Department of Science & Technology, Government of Uttar Pradesh.
