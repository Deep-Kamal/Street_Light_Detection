"""
Street Light Detection Pipeline
================================
Core processing logic extracted from the original Colab notebooks
(Street_light_Dete_with_CSV.ipynb). Wrapped as a reusable function so it
can be called from the FastAPI service (main.py) or from a CLI script.

Given:
  - a video file
  - a GPS survey CSV (columns: point_type, frame_no, elapsed_ms,
    latitude, longitude, timestamp)
  - a YOLO .pt model

Produces:
  - an annotated output video (detections + GPS/compass HUD watermark)
  - a detections CSV (one row per detected object per frame)
  - a ZIP of the single best ("closest") snapshot per tracked object
"""

import os
import csv
import math
import zipfile
import logging
import subprocess
from dataclasses import dataclass

import cv2
import pandas as pd
from scipy.interpolate import interp1d
from ultralytics import YOLO

logger = logging.getLogger("streetlight_pipeline")


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    model_path: str
    video_path: str
    gps_csv_path: str
    output_dir: str

    focal_length: float = 800.0       # px
    known_object_height: float = 5.0  # metres (street light height)
    camera_height_m: float = 1.5      # metres
    conf: float = 0.25                # YOLO confidence threshold

    @property
    def output_video_path(self) -> str:
        return os.path.join(self.output_dir, "output_video.mp4")

    @property
    def csv_path(self) -> str:
        return os.path.join(self.output_dir, "detections.csv")

    @property
    def zip_path(self) -> str:
        return os.path.join(self.output_dir, "detected_frames.zip")


# ──────────────────────────────────────────────────────────────────────────
# GPS helpers
# ──────────────────────────────────────────────────────────────────────────
def bearing_deg(lat1, lon1, lat2, lon2):
    """Compass bearing from (lat1,lon1) -> (lat2,lon2). 0-360, None if identical."""
    if lat1 == lat2 and lon1 == lon2:
        return None
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)
    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r))
    b = math.degrees(math.atan2(x, y))
    return (b + 360) % 360


def compass_label(deg):
    if deg is None:
        return "--"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    return dirs[round(deg / 45) % 8]


# ──────────────────────────────────────────────────────────────────────────
# HUD watermark (compass rose + lat/lon/dir/frame panel, bottom 20%)
# ──────────────────────────────────────────────────────────────────────────
def draw_watermark(frame, gps_lat, gps_lon, direction, dir_label, frame_id):
    h, w = frame.shape[:2]
    panel_h = max(120, int(h * 0.20))
    panel_y = h - panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.line(frame, (0, panel_y), (w, panel_y), (0, 200, 255), 2)

    compass_r = int(panel_h * 0.36)
    cx_compass = int(panel_h * 0.80)
    cy_compass = panel_y + panel_h // 2

    cv2.circle(frame, (cx_compass, cy_compass), compass_r, (0, 200, 255), 1)
    cv2.circle(frame, (cx_compass, cy_compass), compass_r + 2, (0, 80, 120), 1)

    font_c = cv2.FONT_HERSHEY_SIMPLEX
    cardinal = {0: "N", 90: "E", 180: "S", 270: "W"}
    tick_r = compass_r - 6
    lbl_r = compass_r + 14
    for angle_deg, label in cardinal.items():
        rad = math.radians(angle_deg - 90)
        tx = int(cx_compass + tick_r * math.cos(rad))
        ty = int(cy_compass + tick_r * math.sin(rad))
        ox = int(cx_compass + (compass_r + 2) * math.cos(rad))
        oy = int(cy_compass + (compass_r + 2) * math.sin(rad))
        cv2.line(frame, (ox, oy), (tx, ty), (0, 200, 255), 2)
        lx = int(cx_compass + lbl_r * math.cos(rad))
        ly = int(cy_compass + lbl_r * math.sin(rad))
        fs = max(0.35, compass_r / 80)
        col = (0, 80, 255) if label == "N" else (200, 200, 200)
        cv2.putText(frame, label, (lx - 6, ly + 5), font_c, fs, col, 1, cv2.LINE_AA)

    for angle_deg in [45, 135, 225, 315]:
        rad = math.radians(angle_deg - 90)
        ox = int(cx_compass + compass_r * math.cos(rad))
        oy = int(cy_compass + compass_r * math.sin(rad))
        ix = int(cx_compass + (compass_r - 8) * math.cos(rad))
        iy = int(cy_compass + (compass_r - 8) * math.sin(rad))
        cv2.line(frame, (ox, oy), (ix, iy), (0, 130, 180), 1)

    if direction is not None:
        needle_r = int(compass_r * 0.82)
        tail_r = int(compass_r * 0.35)
        rad = math.radians(direction - 90)
        nx = int(cx_compass + needle_r * math.cos(rad))
        ny = int(cy_compass + needle_r * math.sin(rad))
        tx = int(cx_compass - tail_r * math.cos(rad))
        ty_ = int(cy_compass - tail_r * math.sin(rad))
        cv2.arrowedLine(frame, (tx, ty_), (nx, ny), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)

    cv2.circle(frame, (cx_compass, cy_compass), 4, (255, 255, 255), -1)

    col_x = cx_compass + compass_r + 30
    label_w = 80
    val_x = col_x + label_w
    line_h = max(18, int(panel_h / 6.5))
    start_y = panel_y + int(panel_h * 0.18)
    key_col = (0, 200, 255)
    val_col = (255, 255, 255)
    hi_col = (0, 255, 140)
    fs_key = max(0.38, panel_h / 420)
    fs_val = max(0.42, panel_h / 380)

    rows = [
        ("LAT", f"{gps_lat:.6f} deg", val_col),
        ("LON", f"{gps_lon:.6f} deg", val_col),
        ("DIR", dir_label, hi_col),
        ("FRAME", f"{frame_id:07d}", (180, 180, 180)),
    ]
    for idx, (key, val, vcol) in enumerate(rows):
        y = start_y + idx * line_h
        cv2.putText(frame, key, (col_x, y), font_c, fs_key, key_col, 1, cv2.LINE_AA)
        cv2.putText(frame, val, (val_x, y), font_c, fs_val, vcol, 1, cv2.LINE_AA)

    div_x = cx_compass + compass_r + 14
    cv2.line(frame, (div_x, panel_y + 8), (div_x, h - 8), (0, 120, 160), 1)

    brand = "Street Light Survey"
    fs_b = max(0.30, panel_h / 500)
    (bw, bh), _ = cv2.getTextSize(brand, font_c, fs_b, 1)
    cv2.putText(frame, brand, (w - bw - 10, h - 6), font_c, fs_b, (0, 130, 180), 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────────────────────────────────
# GPS CSV loading
# ──────────────────────────────────────────────────────────────────────────
def load_gps(gps_csv_path):
    """Returns (frame_gps dict, lat_fn, lon_fn, frame_ts dict)."""
    gps_raw = pd.read_csv(gps_csv_path)

    interp_df = (gps_raw[gps_raw["point_type"] == "interpolated"]
                 .dropna(subset=["frame_no", "latitude", "longitude"])
                 .sort_values("frame_no")
                 .reset_index(drop=True))

    frame_gps = {
        int(row.frame_no): (row.latitude, row.longitude)
        for row in interp_df.itertuples()
    }

    rec_df = (gps_raw[gps_raw["point_type"] == "recorded"]
              .dropna(subset=["elapsed_ms", "latitude", "longitude"])
              .sort_values("elapsed_ms"))

    if len(rec_df) == 0:
        raise ValueError("GPS CSV has no 'recorded' rows with elapsed_ms/lat/lon — "
                          "cannot build fallback interpolation.")

    t = rec_df["elapsed_ms"].values
    lat = rec_df["latitude"].values
    lon = rec_df["longitude"].values
    lat_fn = interp1d(t, lat, bounds_error=False, fill_value=(lat[0], lat[-1]))
    lon_fn = interp1d(t, lon, bounds_error=False, fill_value=(lon[0], lon[-1]))

    frame_ts = {
        int(row.frame_no): row.timestamp
        for row in interp_df.itertuples()
        if pd.notna(row.frame_no)
    }

    return frame_gps, lat_fn, lon_fn, frame_ts


# ──────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────
def run_pipeline(cfg: PipelineConfig, progress_cb=None):
    """
    Runs detection + GPS overlay + CSV/ZIP export.

    progress_cb: optional callable(frame_id, total_frames) for progress reporting.

    Returns dict with paths and summary stats.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = YOLO(cfg.model_path)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # NOTE: cap.get(CAP_PROP_FRAME_WIDTH/HEIGHT) reports the video's raw
    # coded dimensions from container metadata. When the source video carries
    # a rotation flag (e.g. a portrait phone recording tagged 90 degrees),
    # OpenCV's FFmpeg backend auto-applies that rotation to the actual pixel
    # data returned by cap.read() - but CAP_PROP_FRAME_WIDTH/HEIGHT keep
    # reporting the PRE-rotation values. That mismatch was causing the
    # VideoWriter below to be initialized with swapped/wrong dimensions,
    # which is what produced the rotated-looking output video. Grabbing the
    # true size from an actual decoded frame avoids this entirely.
    ret0, probe_frame = cap.read()
    if not ret0:
        raise RuntimeError(f"Cannot read any frames from video: {cfg.video_path}")
    height, width = probe_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind so the main loop sees frame 0 again

    frame_gps, lat_fn, lon_fn, frame_ts = load_gps(cfg.gps_csv_path)

    # Write to a temp file first - cv2's "mp4v" fourcc is MPEG-4 Part 2,
    # which most browsers' video tags cannot decode (causes
    # MEDIA_ERR_SRC_NOT_SUPPORTED even though the file itself is valid).
    # Re-encode to H.264 via ffmpeg after the frame loop finishes.
    raw_video_path = os.path.join(cfg.output_dir, "_raw_output.mp4")
    writer = cv2.VideoWriter(
        raw_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (width, height),
    )

    records = []
    frame_id = 0
    prev_lat = prev_lon = None
    best_snap = {}  # obj_id -> {distance_m, snap_name, img_bytes, frame_id}

    zip_file = zipfile.ZipFile(cfg.zip_path, "w", zipfile.ZIP_DEFLATED)

    logger.info("Starting detection on %s frames ...", total)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            if frame_id in frame_gps:
                gps_lat, gps_lon = frame_gps[frame_id]
                gps_source = "interpolated_csv"
            else:
                gps_lat = float(lat_fn(elapsed_ms))
                gps_lon = float(lon_fn(elapsed_ms))
                gps_source = "recorded_interp"

            raw_ts = frame_ts.get(frame_id, "")
            if raw_ts:
                try:
                    timestamp_str = str(raw_ts).replace("T", "  ").split(".")[0]
                except Exception:
                    timestamp_str = str(raw_ts)[:19]
            else:
                secs = elapsed_ms // 1000
                timestamp_str = f"T+{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"

            if prev_lat is not None and prev_lon is not None:
                direction = bearing_deg(prev_lat, prev_lon, gps_lat, gps_lon)
            else:
                direction = None
            dir_label = f"{direction:.1f} deg ({compass_label(direction)})" if direction is not None else "--"

            results = model.track(
                frame, persist=True, conf=cfg.conf, verbose=False,
                tracker=os.path.join(os.path.dirname(__file__), "bytetrack_streetlight.yaml"),
            )
            annotated = results[0].plot()
            boxes = results[0].boxes

            annotated = draw_watermark(annotated, gps_lat, gps_lon, direction, dir_label, frame_id)

            if len(boxes) > 0:
                coords = boxes.xyxy.cpu().numpy()
                if boxes.id is not None:
                    detection_ids = boxes.id.cpu().numpy().astype(int)
                else:
                    detection_ids = list(range(len(boxes)))

                for box, obj_id in zip(coords, detection_ids):
                    x1, y1, x2, y2 = box
                    pixel_height = max(1, y2 - y1)

                    distance = (cfg.known_object_height * cfg.focal_length) / pixel_height

                    cy_pp = height / 2.0
                    elev_deg = math.degrees(math.atan2(cy_pp - y1, cfg.focal_length))

                    height_diff = cfg.known_object_height - cfg.camera_height_m
                    elev_geom_deg = math.degrees(math.atan2(height_diff, distance)) if distance > 0 else None

                    cx_box = int((x1 + x2) / 2)
                    cy_box = max(int(y1) - 10, 15)
                    cv2.putText(annotated, f"{distance:.1f}m", (cx_box, cy_box),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(annotated, f"Elev: {elev_deg:+.1f} deg", (cx_box, max(cy_box - 20, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 1, cv2.LINE_AA)
                    cv2.putText(annotated, f"({gps_lat:.5f}, {gps_lon:.5f})",
                                (int(x1), int(y2) + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
                    cv2.putText(annotated, f"Dir: {dir_label}",
                                (int(x1), int(y2) + 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

                    elev_sign = "P" if elev_deg >= 0 else "N"
                    snap_name = (f"best_obj{int(obj_id):04d}"
                                 f"_f{frame_id:07d}"
                                 f"_dist{distance:.1f}m"
                                 f"_lat{gps_lat:.5f}_lon{gps_lon:.5f}"
                                 f"_dir{int(direction) if direction is not None else 'NA'}"
                                 f"_elev{elev_sign}{abs(elev_deg):.1f}.jpg")

                    obj_key = int(obj_id)
                    if obj_key not in best_snap or distance < best_snap[obj_key]["distance_m"]:
                        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if ok:
                            best_snap[obj_key] = {
                                "distance_m": distance,
                                "snap_name": snap_name,
                                "img_bytes": buf.tobytes(),
                                "frame_id": frame_id,
                            }

                    records.append({
                        "frame_id": frame_id,
                        "object_id": obj_key,
                        "elapsed_ms": elapsed_ms,
                        "timestamp": timestamp_str,
                        "distance_m": round(float(distance), 2),
                        "direction_deg": round(direction, 2) if direction is not None else "",
                        "direction_card": compass_label(direction),
                        "elevation_angle_deg": round(elev_deg, 2),
                        "elevation_geom_deg": round(elev_geom_deg, 2) if elev_geom_deg is not None else "",
                        "camera_height_m": cfg.camera_height_m,
                        "gps_lat": round(gps_lat, 7),
                        "gps_lon": round(gps_lon, 7),
                        "gps_source": gps_source,
                        "bbox_x1": round(float(x1), 1),
                        "bbox_y1": round(float(y1), 1),
                        "bbox_x2": round(float(x2), 1),
                        "bbox_y2": round(float(y2), 1),
                        "snap_file": "",
                        "is_best_snap": False,
                    })

            writer.write(annotated)
            prev_lat, prev_lon = gps_lat, gps_lon
            frame_id += 1

            if progress_cb and frame_id % 25 == 0:
                progress_cb(frame_id, total)

        # write best snaps to zip
        best_frame_ids = {}
        for obj_key, info in best_snap.items():
            zip_file.writestr(info["snap_name"], info["img_bytes"])
            best_frame_ids[obj_key] = info["frame_id"]

        for rec in records:
            obj_key = rec["object_id"]
            if obj_key in best_snap:
                rec["snap_file"] = best_snap[obj_key]["snap_name"]
                rec["is_best_snap"] = (rec["frame_id"] == best_frame_ids[obj_key])

    finally:
        cap.release()
        writer.release()
        zip_file.close()

    # Re-encode the raw mp4v file to H.264 so it plays in browser <video> tags.
    # Falls back to the raw file (with a warning) if ffmpeg isn't available
    # or the re-encode fails, so the API still returns something.
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", raw_video_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                cfg.output_video_path,
            ],
            check=True,
            capture_output=True,
        )
        os.remove(raw_video_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(
            "ffmpeg re-encode to H.264 failed (%s); falling back to raw mp4v "
            "output, which may not play in browsers.", e
        )
        os.replace(raw_video_path, cfg.output_video_path)

    if records:
        fieldnames = list(records[0].keys())
        with open(cfg.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(records)
    else:
        # still write an empty CSV with no rows so the API always returns a file
        with open(cfg.csv_path, "w", newline="") as f:
            f.write("")

    return {
        "output_video_path": cfg.output_video_path,
        "csv_path": cfg.csv_path,
        "zip_path": cfg.zip_path,
        "frames_processed": frame_id,
        "detection_rows": len(records),
        "unique_objects": len(best_snap),
    }