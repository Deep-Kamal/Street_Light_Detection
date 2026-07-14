"""
FastAPI service for Multi-Model Detection Pipeline
(Street Light + generic YOLO/Custom).

Endpoints
---------
Models:
  GET  /models                      - List all registered models
  POST /models                      - Add a new model
  DELETE /models/{model_id}         - Remove a model
  POST /models/{model_id}/activate  - Set a model as active
  GET  /models/active               - Get currently active model

Processing:
  POST /process_async               - Start async job with active/selected model.
  GET  /status/{job_id}             - Get job status
  GET  /download/{job_id}/{type}    - Download results (automatically waits if still processing)

Map Data:
  GET  /jobs/{job_id}/map-data      - Get lat/long data from CSV for mapping
  GET  /map/all-jobs                - Get all jobs with map data

GET /health                         - Health check
"""

import os
import shutil
import uuid
import zipfile
import logging
import traceback
import json
import threading
import time  # Added for synchronous wait functionality
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Import pipelines
try:
    from .pipeline import PipelineConfig, run_pipeline, get_map_points
    STREETLIGHT_PIPELINE_AVAILABLE = True
except ImportError:
    STREETLIGHT_PIPELINE_AVAILABLE = False
    logging.warning("Street light pipeline not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multimodel_api")

# ── CONFIGURATION ──────────────────────────────────────────────────────────
BASE_DIR = os.environ.get("DETECTION_DATA_DIR", "/tmp/detection_jobs")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="Multi-Model Detection Pipeline", version="2.0.0")

# ── DATA MODELS ────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    model_id: str
    name: str
    description: Optional[str] = ""
    model_type: str  # "streetlight", "yolo_detection", "yolo_segmentation", "custom"
    model_path: str
    is_active: bool = False
    created_at: str
    config: Optional[Dict[str, Any]] = {}

class ModelCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    model_type: str  # "streetlight", "yolo_detection", "yolo_segmentation", "custom"
    model_path: Optional[str] = None  # Path to existing model file
    config: Optional[Dict[str, Any]] = {}

# ── MODEL / JOB REGISTRY (DISK-BACKED SO IT SURVIVES RESTARTS & MULTIPLE WORKERS) ──
MODELS: Dict[str, ModelInfo] = {}
JOBS: Dict[str, Dict] = {}

MODELS_REGISTRY_FILE = os.path.join(BASE_DIR, "models_registry.json")
JOBS_REGISTRY_FILE = os.path.join(BASE_DIR, "jobs_registry.json")
_MODELS_LOCK = threading.Lock()
_JOBS_LOCK = threading.Lock()


def _save_models_to_disk():
    with _MODELS_LOCK:
        try:
            tmp_path = MODELS_REGISTRY_FILE + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump({mid: m.model_dump() for mid, m in MODELS.items()}, f)
            os.replace(tmp_path, MODELS_REGISTRY_FILE)
        except Exception as e:
            logger.error("Failed to persist models registry: %s", e)


def _load_models_from_disk():
    if not os.path.exists(MODELS_REGISTRY_FILE):
        return
    with _MODELS_LOCK:
        try:
            with open(MODELS_REGISTRY_FILE, "r") as f:
                data = json.load(f)
            for mid, m in data.items():
                MODELS[mid] = ModelInfo(**m)
        except Exception as e:
            logger.error("Failed to load models registry: %s", e)


def _save_jobs_to_disk():
    with _JOBS_LOCK:
        try:
            tmp_path = JOBS_REGISTRY_FILE + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(JOBS, f, default=str)
            os.replace(tmp_path, JOBS_REGISTRY_FILE)
        except Exception as e:
            logger.error("Failed to persist jobs registry: %s", e)


def _load_jobs_from_disk():
    """Merge any jobs persisted by other workers/processes into memory."""
    if not os.path.exists(JOBS_REGISTRY_FILE):
        return
    with _JOBS_LOCK:
        try:
            with open(JOBS_REGISTRY_FILE, "r") as f:
                data = json.load(f)
            for job_id, job in data.items():
                if job_id not in JOBS or job.get("status") in ("done", "failed"):
                    JOBS[job_id] = job
        except Exception as e:
            logger.error("Failed to load jobs registry: %s", e)


def _initialize_default_models():
    """Load any previously registered models, then ensure the built-in
    defaults exist (without duplicating them on every restart)."""
    _load_models_from_disk()
    existing_names = {m.name for m in MODELS.values()}

    # Default street light model
    default_streetlight_path = os.path.join(
        os.path.dirname(__file__), "..", "model", "Street_light_Detection_DN_model_best.pt"
    )
    if os.path.exists(default_streetlight_path) and "Street Light Detection (Default)" not in existing_names:
        model_id = str(uuid.uuid4())[:8]
        MODELS[model_id] = ModelInfo(
            model_id=model_id,
            name="Street Light Detection (Default)",
            description="Default YOLO model for street light detection",
            model_type="streetlight",
            model_path=default_streetlight_path,
            is_active=len(MODELS) == 0,
            created_at=datetime.now().isoformat(),
            config={"conf": 0.25, "focal_length": 800.0, "known_object_height": 5.0, "camera_height_m": 1.5}
        )
        logger.info(f"Registered default street light model: {model_id}")

    # Safety net: if models are registered but somehow none ended up active
    if MODELS and not any(m.is_active for m in MODELS.values()):
        fallback = next(iter(MODELS.values()))
        fallback.is_active = True
        logger.info(f"No active model found on startup; activating: {fallback.name}")

    _save_models_to_disk()

# Initialize on startup
_initialize_default_models()


# ── EXCEPTION HANDLER ──────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# ── MODEL MANAGEMENT ENDPOINTS ─────────────────────────────────────────────

@app.get("/models")
def list_models():
    """List all registered models."""
    _load_models_from_disk()
    models_list = [m.model_dump() for m in MODELS.values()]
    return {
        "models": models_list,
        "total": len(models_list),
        "active_model_id": next((m.model_id for m in MODELS.values() if m.is_active), None)
    }


@app.get("/models/active")
def get_active_model():
    """Get the currently active model."""
    _load_models_from_disk()
    active = next((m for m in MODELS.values() if m.is_active), None)
    if not active:
        raise HTTPException(status_code=404, detail="No active model set")
    return active.model_dump()


@app.post("/models")
async def add_model(
    name: str = Form(...),
    description: str = Form(""),
    model_type: str = Form(...),
    model_file: Optional[UploadFile] = File(None),
    model_path: Optional[str] = Form(None),
    config_json: Optional[str] = Form("{}")
):
    """Add a new model. Can upload a file or specify an existing path."""
    _load_models_from_disk()

    valid_types = ["streetlight", "yolo_detection", "yolo_segmentation", "custom"]
    if model_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type. Must be one of: {valid_types}"
        )
    
    try:
        config = json.loads(config_json) if config_json else {}
    except json.JSONDecodeError:
        config = {}
    
    final_model_path = model_path
    if model_file:
        model_id = str(uuid.uuid4())[:8]
        file_ext = os.path.splitext(model_file.filename or ".pt")[1]
        final_model_path = os.path.join(MODELS_DIR, f"{model_id}{file_ext}")
        with open(final_model_path, "wb") as f:
            shutil.copyfileobj(model_file.file, f)
    elif not model_path:
        raise HTTPException(
            status_code=400,
            detail="Either model_file or model_path must be provided"
        )
    elif not os.path.exists(model_path):
        raise HTTPException(
            status_code=400,
            detail=f"Model file not found at: {model_path}"
        )
    else:
        model_id = str(uuid.uuid4())[:8]
    
    model_info = ModelInfo(
        model_id=model_id,
        name=name,
        description=description,
        model_type=model_type,
        model_path=final_model_path,
        is_active=len(MODELS) == 0,
        created_at=datetime.now().isoformat(),
        config=config
    )
    
    MODELS[model_id] = model_info
    _save_models_to_disk()
    logger.info(f"Added new model: {name} ({model_id})")
    
    return {
        "message": "Model added successfully",
        "model": model_info.model_dump()
    }


@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    """Remove a model from the registry."""
    _load_models_from_disk()
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = MODELS[model_id]
    was_active = model.is_active
    
    del MODELS[model_id]
    
    if was_active and MODELS:
        first_model = next(iter(MODELS.values()))
        first_model.is_active = True
        logger.info(f"Auto-activated model: {first_model.name}")

    _save_models_to_disk()
    
    return {"message": f"Model '{model.name}' deleted", "was_active": was_active}


@app.post("/models/{model_id}/activate")
def activate_model(model_id: str):
    """Set a model as the active one."""
    _load_models_from_disk()
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    for m in MODELS.values():
        m.is_active = False
    
    MODELS[model_id].is_active = True
    
    _save_models_to_disk()
    logger.info(f"Activated model: {MODELS[model_id].name}")
    
    return {
        "message": f"Model '{MODELS[model_id].name}' activated",
        "model": MODELS[model_id].model_dump()
    }


# ── JOB PROCESSING ENDPOINTS ───────────────────────────────────────────────

def _job_dir(job_id: str) -> str:
    return os.path.join(BASE_DIR, job_id)


def _save_upload(upload: UploadFile, dest_path: str):
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)


def _bundle_zip(job_id: str, result: dict) -> str:
    bundle_path = os.path.join(_job_dir(job_id), "all_outputs.zip")
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as z:
        if "output_video_path" in result and os.path.exists(result["output_video_path"]):
            z.write(result["output_video_path"], arcname="output_video.mp4")
        if "csv_path" in result and os.path.exists(result["csv_path"]):
            z.write(result["csv_path"], arcname="detections.csv")
        if "zip_path" in result and os.path.exists(result["zip_path"]):
            z.write(result["zip_path"], arcname="detected_frames.zip")
    return bundle_path


def _run_job(job_id: str, model_id: str, cfg_dict: dict):
    """Run detection job based on model type."""
    try:
        JOBS[job_id]["status"] = "running"
        model_info = MODELS.get(model_id)
        
        if not model_info:
            raise Exception(f"Model {model_id} not found")
        
        def progress(done, total):
            JOBS[job_id]["progress"] = {"done": done, "total": total}

        result = {}
        
        if model_info.model_type == "streetlight" and STREETLIGHT_PIPELINE_AVAILABLE:
            from .pipeline import PipelineConfig, run_pipeline
            cfg = PipelineConfig(
                model_path=model_info.model_path,
                video_path=cfg_dict["video_path"],
                gps_csv_path=cfg_dict["csv_path"],
                output_dir=_job_dir(job_id),
                conf=cfg_dict.get("conf", 0.25),
                focal_length=cfg_dict.get("focal_length", 800.0),
                known_object_height=cfg_dict.get("known_object_height", 5.0),
                camera_height_m=cfg_dict.get("camera_height_m", 1.5),
            )
            result = run_pipeline(cfg, progress_cb=progress)

        elif model_info.model_type in ["yolo_detection", "yolo_segmentation", "custom"]:
            from ultralytics import YOLO
            import pandas as pd
            
            model = YOLO(model_info.model_path)
            video_path = cfg_dict["video_path"]
            output_dir = _job_dir(job_id)
            
            conf = cfg_dict.get("conf", 0.25)
            iou = cfg_dict.get("iou", 0.50)
            imgsz = cfg_dict.get("imgsz", 640)
            
            results = model.predict(
                source=video_path,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                save=True,
                project=output_dir,
                name="prediction_results"
            )
            
            rows = []
            for result_index, result in enumerate(results):
                source_name = os.path.basename(result.path)
                boxes = result.boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    class_name = model.names[cls_id]
                    conf_score = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    
                    rows.append({
                        "frame": result_index,
                        "source_name": source_name,
                        "class_name": class_name,
                        "confidence": round(conf_score, 4),
                        "bbox_x1": round(float(xyxy[0]), 2),
                        "bbox_y1": round(float(xyxy[1]), 2),
                        "bbox_x2": round(float(xyxy[2]), 2),
                        "bbox_y2": round(float(xyxy[3]), 2),
                    })
            
            df = pd.DataFrame(rows)
            csv_output = os.path.join(output_dir, "detections.csv")
            df.to_csv(csv_output, index=False)
            
            result = {
                "frames_processed": len(results),
                "detection_rows": len(df),
                "unique_objects": df["class_name"].nunique() if len(df) > 0 else 0,
                "csv_path": csv_output,
                "output_video_path": "",
                "zip_path": ""
            }
        else:
            raise Exception(f"Pipeline not available for model type: {model_info.model_type}")
        
        # Bundle results
        if "csv_path" in result and result["csv_path"]:
            result["bundle_path"] = _bundle_zip(job_id, result)
        
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result
        JOBS[job_id]["model_id"] = model_id
        JOBS[job_id]["model_name"] = model_info.name
        _save_jobs_to_disk()
        
    except Exception as e:
        logger.error("Job %s failed: %s\n%s", job_id, e, traceback.format_exc())
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        _save_jobs_to_disk()


@app.post("/process_async")
async def process_async(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    gps_csv: Optional[UploadFile] = File(None),
    model_id: Optional[str] = Form(None),
    conf: Optional[float] = Form(None),
    focal_length: Optional[float] = Form(None),
    known_object_height: Optional[float] = Form(None),
    camera_height_m: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None),
    detection_csv: Optional[UploadFile] = File(None),
    approved_frames: Optional[str] = Form(None),
):
    """Start async detection job."""
    
    if model_id:
        if model_id not in MODELS:
            raise HTTPException(status_code=400, detail=f"Model ID {model_id} not found")
    else:
        active = next((m for m in MODELS.values() if m.is_active), None)
        if not active:
            raise HTTPException(status_code=400, detail="No active model. Please select or activate a model.")
        model_id = active.model_id
    
    job_id = str(uuid.uuid4())
    os.makedirs(_job_dir(job_id), exist_ok=True)

    model_info = MODELS[model_id]
    default_config = model_info.config or {}

    # gps_csv is only strictly required for "streetlight" jobs (pipeline.py's
    # PipelineConfig.gps_csv_path is a required plain str with no fallback).
    # yolo_detection / yolo_segmentation / custom never use it at all - so
    # forcing every job to upload one was rejecting those requests outright
    # with a 422 before the job ever started.
    if model_info.model_type == "streetlight" and gps_csv is None:
        raise HTTPException(
            status_code=400,
            detail="model_type 'streetlight' requires a gps_csv upload."
        )

    video_path = os.path.join(_job_dir(job_id), "input_video" + os.path.splitext(video.filename or ".mp4")[1])
    _save_upload(video, video_path)

    csv_path = None
    if gps_csv is not None:
        csv_path = os.path.join(_job_dir(job_id), "gps_track.csv")
        _save_upload(gps_csv, csv_path)

    detection_csv_path = None
    if detection_csv is not None:
        detection_csv_path = os.path.join(_job_dir(job_id), "detections_input.csv")
        _save_upload(detection_csv, detection_csv_path)

    parsed_approved_frames = None
    if approved_frames:
        try:
            parsed_approved_frames = [int(f.strip()) for f in approved_frames.split(",") if f.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="approved_frames must be a comma-separated list of integers")

    cfg_dict = {
        "video_path": video_path,
        "csv_path": csv_path,
        "conf": conf if conf is not None else default_config.get("conf", 0.25),
        "focal_length": focal_length if focal_length is not None else default_config.get("focal_length", 800.0),
        "known_object_height": known_object_height if known_object_height is not None else default_config.get("known_object_height", 5.0),
        "camera_height_m": camera_height_m if camera_height_m is not None else default_config.get("camera_height_m", 1.5),
        "iou": iou if iou is not None else default_config.get("iou", 0.50),
        "imgsz": imgsz if imgsz is not None else default_config.get("imgsz", 640),
        "detection_csv_path": detection_csv_path,
        "approved_frames": parsed_approved_frames,
    }

    JOBS[job_id] = {
        "status": "queued",
        "progress": None,
        "result": None,
        "error": None,
        "model_id": model_id,
        "model_name": model_info.name,
        "created_at": datetime.now().isoformat(),
        "video_filename": video.filename
    }
    
    _save_jobs_to_disk()
    background_tasks.add_task(_run_job, job_id, model_id, cfg_dict)

    return {"job_id": job_id, "status": "queued", "model_id": model_id, "model_name": model_info.name}


@app.get("/status/{job_id}")
def job_status(job_id: str):
    """Get job status."""
    _load_jobs_from_disk()
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "model_id": job.get("model_id"),
        "model_name": job.get("model_name"),
        "created_at": job.get("created_at")
    }
    
    if job["status"] == "done":
        result = job.get("result") or {}
        download_urls = {}
        # Only advertise a link if this model type actually produced that
        # output. yolo_detection/yolo_segmentation/custom jobs, for example,
        # never produce a video (output_video_path is ""), so showing a
        # "video" link for them always 404'd - which the player then
        # reported as the generic "no video available" error.
        if result.get("output_video_path"):
            download_urls["video"] = f"/download/{job_id}/video"
        if result.get("csv_path"):
            download_urls["csv"] = f"/download/{job_id}/csv"
        if result.get("zip_path"):
            download_urls["frames_zip"] = f"/download/{job_id}/frames_zip"
        if result.get("bundle_path"):
            download_urls["all"] = f"/download/{job_id}/all"

        response["download_urls"] = download_urls
        response["summary"] = {
            "frames_processed": job["result"].get("frames_processed", 0),
            "detection_rows": job["result"].get("detection_rows", 0),
            "unique_objects": job["result"].get("unique_objects", 0),
        }
    elif job["status"] == "failed":
        response["error"] = job["error"]
    
    return response


@app.get("/jobs")
def list_jobs():
    """List all jobs."""
    _load_jobs_from_disk()
    jobs_list = []
    for job_id, job in JOBS.items():
        jobs_list.append({
            "job_id": job_id,
            "status": job["status"],
            "model_id": job.get("model_id"),
            "model_name": job.get("model_name"),
            "created_at": job.get("created_at"),
            "video_filename": job.get("video_filename"),
            "progress": job.get("progress"),
            "summary": {
                "frames_processed": job["result"].get("frames_processed", 0) if job.get("result") else 0,
                "detection_rows": job["result"].get("detection_rows", 0) if job.get("result") else 0,
            } if job["status"] == "done" else None
        })
    
    return {"jobs": jobs_list, "total": len(jobs_list)}


# ── MAP DATA ENDPOINTS ─────────────────────────────────────────────────────

_LAT_COL_CANDIDATES = ["latitude", "lat", "gps_lat", "detected_pothole_latitude"]
_LNG_COL_CANDIDATES = ["longitude", "lng", "lon", "long", "gps_lon", "gps_lng", "detected_pothole_longitude"]
_FRAME_COL_CANDIDATES = ["frame", "frame_no", "frame_id"]


def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_map_points_for_job(job_id: str, job: dict) -> List[Dict[str, Any]]:
    """Build a list of {lat, lng, ...} points for a completed job."""
    import pandas as pd

    csv_path = job["result"].get("csv_path")
    if not csv_path or not os.path.exists(csv_path):
        return []

    # Prefer pipeline.py's get_map_points() when the CSV has the street
    # light pipeline's own schema (gps_lat/gps_lon/object_id/frame_id/
    # distance_m/is_best_snap) — it is the single source of truth for what
    # counts as a "detected frame" for that pipeline (see pipeline.py's
    # docstring), rather than duplicating that decision here via generic
    # column-guessing. Falls through to the generic logic below for any
    # other model_type's CSV schema (yolo_detection/yolo_segmentation/
    # custom), which doesn't have these columns.
    if STREETLIGHT_PIPELINE_AVAILABLE:
        try:
            header = pd.read_csv(csv_path, nrows=0).columns
        except Exception:
            header = []
        streetlight_schema = {"gps_lat", "gps_lon", "object_id", "frame_id", "distance_m", "is_best_snap"}
        if streetlight_schema.issubset(set(header)):
            sl_points = get_map_points(csv_path)
            return [
                {
                    "lat": p["lat"],
                    "lng": p["lon"],
                    "type": "streetlight",
                    "confidence": 0.0,
                    "frame": p["frame_id"],
                    "object_id": p["object_id"],
                    "distance_m": p["distance_m"],
                    "is_best_snap": p["is_best_snap"],
                }
                for p in sl_points
            ]

    det_df = pd.read_csv(csv_path)
    if det_df.empty:
        return []

    det_lat_col = _pick_col(det_df, _LAT_COL_CANDIDATES)
    det_lng_col = _pick_col(det_df, _LNG_COL_CANDIDATES)
    det_frame_col = _pick_col(det_df, _FRAME_COL_CANDIDATES)

    if det_lat_col and det_lng_col:
        points = []
        for _, row in det_df.iterrows():
            lat, lng = row.get(det_lat_col), row.get(det_lng_col)
            if pd.isna(lat) or pd.isna(lng):
                continue
            frame_val = row.get(det_frame_col, 0) if det_frame_col else 0
            points.append({
                "lat": float(lat),
                "lng": float(lng),
                "type": row.get("class_name", row.get("severity", "detection")),
                "confidence": float(row.get("confidence", row.get("confidence_score", 0)) or 0),
                "frame": int(frame_val) if pd.notna(frame_val) else 0
            })
        return points

    if det_frame_col is None:
        return []

    gps_csv_path = os.path.join(_job_dir(job_id), "gps_track.csv")
    if not os.path.exists(gps_csv_path):
        return []

    gps_df = pd.read_csv(gps_csv_path)
    gps_lat_col = _pick_col(gps_df, _LAT_COL_CANDIDATES)
    gps_lng_col = _pick_col(gps_df, _LNG_COL_CANDIDATES)
    gps_frame_col = _pick_col(gps_df, _FRAME_COL_CANDIDATES)
    if gps_lat_col is None or gps_lng_col is None or gps_frame_col is None:
        return []

    # merge_asof requires the two "on" columns to share the exact same
    # dtype. If either CSV has any missing/blank frame numbers, pandas
    # silently upgrades that column to float64 while the other stays
    # int64, and merge_asof throws "incompatible merge keys ... dtype
    # int64 and dtype float64". Force both to float64 (safe for NaN,
    # unlike int64) before sorting/merging to avoid that.
    det_sorted = det_df.copy()
    det_sorted[det_frame_col] = pd.to_numeric(det_sorted[det_frame_col], errors="coerce")
    det_sorted = det_sorted.dropna(subset=[det_frame_col])
    det_sorted[det_frame_col] = det_sorted[det_frame_col].astype("float64")
    det_sorted = det_sorted.sort_values(det_frame_col)

    gps_sorted = gps_df.copy()
    gps_sorted[gps_frame_col] = pd.to_numeric(gps_sorted[gps_frame_col], errors="coerce")
    gps_sorted = gps_sorted.dropna(subset=[gps_frame_col])
    gps_sorted[gps_frame_col] = gps_sorted[gps_frame_col].astype("float64")
    gps_sorted = gps_sorted.sort_values(gps_frame_col)

    if det_sorted.empty or gps_sorted.empty:
        return []

    merged = pd.merge_asof(
        det_sorted,
        gps_sorted,
        left_on=det_frame_col,
        right_on=gps_frame_col,
        direction="nearest",
        suffixes=("", "_gps"),
    )

    merged_lat_col = gps_lat_col + "_gps" if gps_lat_col in det_sorted.columns else gps_lat_col
    merged_lng_col = gps_lng_col + "_gps" if gps_lng_col in det_sorted.columns else gps_lng_col

    points = []
    for _, row in merged.iterrows():
        lat, lng = row.get(merged_lat_col), row.get(merged_lng_col)
        if pd.isna(lat) or pd.isna(lng):
            continue
        points.append({
            "lat": float(lat),
            "lng": float(lng),
            "type": row.get("class_name", "detection"),
            "confidence": float(row.get("confidence", row.get("confidence_score", 0)) or 0),
            "frame": int(row.get(det_frame_col, 0) or 0),
            "timestamp": row.get("timestamp", "") if "timestamp" in merged.columns else ""
        })
    return points


@app.get("/jobs/{job_id}/map-data")
def get_job_map_data(job_id: str):
    """Get lat/long data from job's CSV for mapping."""
    _load_jobs_from_disk()
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    try:
        points = _load_map_points_for_job(job_id, job)
        return {
            "job_id": job_id,
            "model_name": job.get("model_name"),
            "total_points": len(points),
            "points": points
        }
    except Exception as e:
        logger.error("Error reading map data: %s", e)
        return {"points": [], "error": str(e)}


@app.get("/map/all-jobs")
def get_all_jobs_map_data():
    """Get map data for all completed jobs - for showing all running models on map."""
    _load_jobs_from_disk()
    _load_models_from_disk()
    all_data = []
    
    for job_id, job in JOBS.items():
        if job["status"] == "done":
            try:
                points = _load_map_points_for_job(job_id, job)

                all_data.append({
                    "job_id": job_id,
                    "model_id": job.get("model_id"),
                    "model_name": job.get("model_name"),
                    "model_type": MODELS.get(job.get("model_id"), type('obj', (object,), {'model_type': 'unknown'})()).model_type if job.get("model_id") in MODELS else "unknown",
                    "status": job["status"],
                    "created_at": job.get("created_at"),
                    "total_points": len(points),
                    "points": points
                })
                
            except Exception as e:
                logger.error("Error reading map data for job %s: %s", job_id, e)
    
    running_jobs = [
        {
            "job_id": job_id,
            "model_id": job.get("model_id"),
            "model_name": job.get("model_name"),
            "status": job["status"],
            "progress": job.get("progress"),
            "created_at": job.get("created_at")
        }
        for job_id, job in JOBS.items() 
        if job["status"] in ["queued", "running"]
    ]
    
    return {
        "completed_jobs": all_data,
        "running_jobs": running_jobs,
        "total_completed": len(all_data),
        "total_running": len(running_jobs)
    }


# ── DOWNLOAD ENDPOINTS ─────────────────────────────────────────────────────

def _require_done_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"job is '{job['status']}', not ready yet")
    return job


def _wait_for_job(job_id: str, timeout: int = 600) -> Dict:
    """Wait for a job to finish if it's still running/queued.
    
    Using standard `time.sleep` inside a threadpool-safe `def` endpoint 
    prevents the API from returning a JSON error (which the browser saves 
    as video.json) if the user clicks download before the job finishes.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    
    if job["status"] in ("done", "failed"):
        return job
        
    logger.info(f"Download requested for job {job_id}, but status is '{job['status']}'. Waiting for completion...")
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(2)
        _load_jobs_from_disk()
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_id not found")
        if job["status"] in ("done", "failed"):
            return job
            
    raise HTTPException(status_code=408, detail="Timeout waiting for job to complete")


# NOTE: These download endpoints use standard `def` (not `async def`) so 
# FastAPI automatically runs them in a threadpool. This is critical because 
# `_wait_for_job` uses `time.sleep`, which would block the main async event 
# loop if the endpoint were async.

@app.get("/download/{job_id}/video")
def download_video(job_id: str, wait: bool = True):
    job = _wait_for_job(job_id) if wait else _require_done_job(job_id)
    
    if job["status"] == "failed":
        raise HTTPException(status_code=410, detail=f"Job failed: {job.get('error')}")
        
    path = job["result"].get("output_video_path", "")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video output not available")
        
    return FileResponse(
        path, 
        media_type="video/mp4", 
        filename="output_video.mp4",
        headers={"Content-Disposition": 'attachment; filename="output_video.mp4"'}
    )


@app.get("/download/{job_id}/csv")
def download_csv(job_id: str, wait: bool = True):
    job = _wait_for_job(job_id) if wait else _require_done_job(job_id)
    
    if job["status"] == "failed":
        raise HTTPException(status_code=410, detail=f"Job failed: {job.get('error')}")
        
    path = job["result"].get("csv_path", "")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="CSV output not available")
        
    return FileResponse(
        path, 
        media_type="text/csv", 
        filename="detections.csv",
        headers={"Content-Disposition": 'attachment; filename="detections.csv"'}
    )


@app.get("/download/{job_id}/frames_zip")
def download_frames_zip(job_id: str, wait: bool = True):
    job = _wait_for_job(job_id) if wait else _require_done_job(job_id)
    
    if job["status"] == "failed":
        raise HTTPException(status_code=410, detail=f"Job failed: {job.get('error')}")
        
    path = job["result"].get("zip_path", "")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Frames zip not available")
        
    return FileResponse(
        path, 
        media_type="application/zip", 
        filename="detected_frames.zip",
        headers={"Content-Disposition": 'attachment; filename="detected_frames.zip"'}
    )


@app.get("/download/{job_id}/all")
def download_all(job_id: str, wait: bool = True):
    job = _wait_for_job(job_id) if wait else _require_done_job(job_id)
    
    if job["status"] == "failed":
        raise HTTPException(status_code=410, detail=f"Job failed: {job.get('error')}")
        
    path = job["result"].get("bundle_path", "")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Bundle not available")
        
    return FileResponse(
        path, 
        media_type="application/zip", 
        filename="detection_outputs.zip",
        headers={"Content-Disposition": 'attachment; filename="detection_outputs.zip"'}
    )


# ── HEALTH CHECK ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_count": len(MODELS),
        "active_model": next((m.name for m in MODELS.values() if m.is_active), None),
        "jobs_count": len(JOBS),
        "running_jobs": sum(1 for j in JOBS.values() if j["status"] in ["queued", "running"]),
        "streetlight_pipeline": STREETLIGHT_PIPELINE_AVAILABLE,
    }