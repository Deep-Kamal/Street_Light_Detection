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
  - a detections CSV (one row per detected object per frame — never a row
    for a frame with no detection, so anything built from this CSV, such as
    the map points below, is detected-frames-only by construction)
  - a ZIP of the single best ("closest") snapshot per tracked object, whose
    integrity (exactly one image per object, and it really is that object's
    minimum-distance frame) is verified once processing finishes rather than
    just assumed

Also exposes get_map_points(), a small helper that returns only the
detected-frame GPS points from a completed job's CSV — the same rows a
consumer (e.g. main.py's map-data endpoint) should plot, as opposed to the
full per-frame survey GPS track in gps_csv_path, which includes frames where
nothing was detected and should never be plotted as a "detection".
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

            results = model.track(frame, persist=True, conf=cfg.conf, verbose=False)
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

    zip_verified = _verify_zip_is_min_distance_per_object(cfg.zip_path, records, best_snap)

    return {
        "output_video_path": cfg.output_video_path,
        "csv_path": cfg.csv_path,
        "zip_path": cfg.zip_path,
        "frames_processed": frame_id,
        "detection_rows": len(records),
        "unique_objects": len(best_snap),
        "zip_verified": zip_verified,
    }


# ──────────────────────────────────────────────────────────────────────────
# ZIP integrity verification
# ──────────────────────────────────────────────────────────────────────────
def _verify_zip_is_min_distance_per_object(zip_path, records, best_snap) -> bool:
    """
    Confirms that detected_frames.zip contains exactly one image per tracked
    object, and that each image is genuinely that object's minimum-distance
    frame — rather than just trusting the best_snap bookkeeping done during
    the frame loop. Logs the result; never raises, so a verification failure
    is surfaced (via the returned bool / a warning log) without failing an
    otherwise-successful job.
    """
    if not records:
        logger.info("ZIP integrity check skipped: no detections in this job.")
        return True

    by_object = {}
    for rec in records:
        by_object.setdefault(rec["object_id"], []).append(rec["distance_m"])

    mismatches = []
    for obj_id, distances in by_object.items():
        true_min = min(distances)
        claimed = best_snap.get(obj_id, {}).get("distance_m")
        if claimed is None or round(claimed, 2) != round(true_min, 2):
            mismatches.append(obj_id)

    try:
        with zipfile.ZipFile(zip_path) as z:
            n_zip_images = len(z.namelist())
    except Exception as e:
        logger.warning("Could not open %s to verify ZIP integrity: %s", zip_path, e)
        return False

    n_objects = len(by_object)
    ok = (n_zip_images == n_objects) and not mismatches

    if ok:
        logger.info(
            "ZIP integrity verified: %d image(s), one per object, each its "
            "true minimum-distance frame.", n_zip_images
        )
    else:
        logger.warning(
            "ZIP integrity check FAILED: %d objects vs %d zip images; "
            "mismatched objects (snapshot is not their true minimum-distance "
            "frame): %s", n_objects, n_zip_images, mismatches
        )

    return ok


# ──────────────────────────────────────────────────────────────────────────
# Map points — detected frames only
# ──────────────────────────────────────────────────────────────────────────
def get_map_points(csv_path, best_only=True):
    """
    Returns [{lat, lon, object_id, frame_id, distance_m, is_best_snap}, ...]
    from a completed job's detections CSV.

    Deliberately reads detections.csv, not the survey gps_csv_path: the
    detections CSV only ever contains a row for a frame where something was
    actually detected (see the frame loop above), so every point this
    returns corresponds to a real detection. The full per-frame GPS survey
    track is not an acceptable substitute here — it includes every frame of
    the video, detected or not, and plotting it would show survey coverage
    rather than street light detections.

    best_only (default True): only return rows where is_best_snap is True —
    i.e. exactly one point per tracked object, at its closest-range frame,
    matching one-for-one with the images actually saved into
    detected_frames.zip. Pass best_only=False to get every detected frame
    instead (one point per detection row, multiple per object).
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return []

    df = pd.read_csv(csv_path)
    if df.empty or "gps_lat" not in df.columns or "gps_lon" not in df.columns:
        return []

    if best_only and "is_best_snap" in df.columns:
        df = df[df["is_best_snap"] == True]  # noqa: E712 - explicit bool compare for CSV-sourced values

    points = []
    for row in df.itertuples():
        points.append({
            "lat": float(row.gps_lat),
            "lon": float(row.gps_lon),
            "object_id": int(row.object_id),
            "frame_id": int(row.frame_id),
            "distance_m": float(row.distance_m),
            "is_best_snap": bool(row.is_best_snap),
        })
    return points