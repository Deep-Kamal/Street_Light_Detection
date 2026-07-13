"""
Flask frontend server for Multi-Model Detection Pipeline.
Serves the HTML UI and proxies all API requests to the FastAPI backend.
"""

import os
import requests
from flask import Flask, render_template, request, Response, jsonify, stream_with_context

app = Flask(__name__)

# ── CONFIGURATION ──────────────────────────────────────────────────────────
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))
# ───────────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Proxy: GET /health"""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return jsonify(r.json()), r.status_code
    except requests.ConnectionError:
        return jsonify({"status": "error", "detail": "Cannot reach backend"}), 502


# ── MODEL MANAGEMENT PROXIES ───────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def list_models():
    """Proxy: GET /models"""
    try:
        r = requests.get(f"{BACKEND_URL}/models", timeout=10)
        return jsonify(r.json()), r.status_code
    except requests.ConnectionError:
        return jsonify({"detail": "Cannot reach backend"}), 502


@app.route("/api/models", methods=["POST"])
def add_model():
    """Proxy: POST /models - handles file upload"""
    try:
        files = {}
        form_data = {}
        
        for key in request.files:
            f = request.files[key]
            if f.filename:
                files[key] = (f.filename, f.stream, f.content_type or "application/octet-stream")
        
        for key in request.form:
            val = request.form[key]
            if val:
                form_data[key] = val
        
        r = requests.post(
            f"{BACKEND_URL}/models",
            files=files,
            data=form_data,
            timeout=30
        )
        return jsonify(r.json()), r.status_code
    except requests.ConnectionError:
        return jsonify({"detail": "Cannot reach backend"}), 502


@app.route("/api/models/active", methods=["GET"])
def get_active_model():
    """Proxy: GET /models/active"""
    try:
        r = requests.get(f"{BACKEND_URL}/models/active", timeout=10)
        return jsonify(r.json()), r.status_code
    except requests.ConnectionError:
        return jsonify({"detail": "Cannot reach backend"}), 502


@app.route("/api/models/<model_id>", methods=["DELETE"])
def delete_model(model_id):
    """Proxy: DELETE /models/{model_id}"""
    try:
        r = requests.delete(f"{BACKEND_URL}/models/{model_id}", timeout=10)
        return jsonify(r.json()), r.status_code
    except requests.ConnectionError:
        return jsonify({"detail": "Cannot reach backend"}), 502


@app.route("/api/models/<model_id>/activate", methods=["POST"])
def activate_model(model_id):
    """Proxy: POST /models/{model_id}/activate"""
    try:
        r = requests.post(f"{BACKEND_URL}/models/{model_id}/activate", timeout=10)
        return jsonify(r.json()), r.status_code
    except requests.ConnectionError:
        return jsonify({"detail": "Cannot reach backend"}), 502


# ── JOB PROCESSING PROXIES ─────────────────────────────────────────────────

@app.route("/api/process", methods=["POST"])
def process_async():
    """Proxy: POST /process_async"""
    try:
        files = {}
        form_data = {}

        for key in request.files:
            f = request.files[key]
            if f.filename:
                files[key] = (f.filename, f.stream, f.content_type or "application/octet-stream")

        for key in request.form:
            val = request.form[key]
            if val:
                if key in ("conf", "focal_length", "known_object_height", "camera_height_m", "iou"):
                    try:
                        form_data[key] = float(val)
                    except ValueError:
                        form_data[key] = val
                elif key == "imgsz":
                    try:
                        form_data[key] = int(val)
                    except ValueError:
                        form_data[key] = val
                else:
                    form_data[key] = val

        r = requests.post(
            f"{BACKEND_URL}/process_async",
            files=files,
            data=form_data,
            timeout=30,
        )
        return jsonify(r.json()), r.status_code
    except requests.ConnectionError:
        return jsonify({"detail": "Cannot reach backend"}), 502
    except requests.Timeout:
        return jsonify({"detail": "Backend timeout during upload"}), 504


@app.route("/api/status/<job_id>")
def job_status(job_id):
    """Proxy: GET /status/{job_id}"""
    try:
        r = requests.get(f"{BACKEND_URL}/status/{job_id}", timeout=10)
        return jsonify(r.json()), r.status_code
    except requests.ConnectionError:
        return jsonify({"detail": "Cannot reach backend"}), 502


@app.route("/api/jobs")
def list_jobs():
    """Proxy: GET /jobs"""
    try:
        r = requests.get(f"{BACKEND_URL}/jobs", timeout=10)
        return jsonify(r.json()), r.status_code
    except requests.ConnectionError:
        return jsonify({"detail": "Cannot reach backend"}), 502


# ── MAP DATA PROXIES ───────────────────────────────────────────────────────

@app.route("/api/jobs/<job_id>/map-data")
def job_map_data(job_id):
    """Proxy: GET /jobs/{job_id}/map-data"""
    try:
        r = requests.get(f"{BACKEND_URL}/jobs/{job_id}/map-data", timeout=10)
        return jsonify(r.json()), r.status_code
    except requests.ConnectionError:
        return jsonify({"detail": "Cannot reach backend"}), 502


@app.route("/api/map/all-jobs")
def all_jobs_map_data():
    """Proxy: GET /map/all-jobs"""
    try:
        r = requests.get(f"{BACKEND_URL}/map/all-jobs", timeout=10)
        return jsonify(r.json()), r.status_code
    except requests.ConnectionError:
        return jsonify({"detail": "Cannot reach backend"}), 502


# ── DOWNLOAD PROXIES ───────────────────────────────────────────────────────

@app.route("/api/download/<job_id>/<file_type>")
def download_file(job_id, file_type):
    """Proxy: GET /download/{job_id}/{video|csv|frames_zip|all}
    
    CRITICAL FIX: If the user clicks download before the background job is 
    done, the backend (FastAPI) will now automatically pause and wait for the 
    job to finish before sending the file. We increased the timeout here in 
    Flask so the proxy doesn't kill the connection while waiting.
    """
    valid_types = {"video", "csv", "frames_zip", "all"}
    if file_type not in valid_types:
        return jsonify({"detail": f"Invalid file_type. Must be one of: {valid_types}"}), 400

    filename_map = {
        "video": "output_video.mp4",
        "csv": "detections.csv",
        "frames_zip": "detected_frames.zip",
        "all": "detection_outputs.zip",
    }
    content_type_map = {
        "video": "video/mp4",
        "csv": "text/csv",
        "frames_zip": "application/zip",
        "all": "application/zip",
    }

    # Forward the browser's Range header to the backend so the <video>
    # player on the page can seek instead of re-downloading from the start
    # every time. Harmless no-op for the other file types.
    upstream_headers = {}
    if "Range" in request.headers:
        upstream_headers["Range"] = request.headers["Range"]

    try:
        # CRITICAL FIX: Increased timeout to 1800 seconds (30 mins).
        # Previously it was 120s. If a video took longer than 2 minutes to process,
        # Flask would timeout and return a JSON error, which the browser incorrectly 
        # saved as a file named "video.json".
        r = requests.get(
            f"{BACKEND_URL}/download/{job_id}/{file_type}",
            stream=True,
            timeout=1800,
            headers=upstream_headers,
        )
        
        if r.status_code not in (200, 206):
            return jsonify(r.json()), r.status_code

        def generate():
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk

        # Video is served "inline" so it plays directly in the page's
        # <video> element; everything else still forces a download.
        disposition = "inline" if file_type == "video" else "attachment"
        response_headers = {
            "Content-Disposition": f'{disposition}; filename="{filename_map[file_type]}"'
        }
        
        # Pass through streaming/range metadata when the backend sent it, so
        # the browser knows this is seekable and how big it is.
        for header_name in ("Content-Range", "Accept-Ranges", "Content-Length"):
            if header_name in r.headers:
                response_headers[header_name] = r.headers[header_name]

        # stream_with_context keeps the request context alive for the
        # duration of the generator (needed for any streaming response),
        # and direct_passthrough tells Werkzeug to send bytes straight
        # through rather than buffering/re-encoding them - important for
        # binary video data, and for not blocking the dev server on large
        # files (see threaded=True on app.run() below).
        resp = Response(
            stream_with_context(generate()),
            status=r.status_code,
            content_type=content_type_map[file_type],
            headers=response_headers,
        )
        resp.direct_passthrough = True
        return resp
        
    except requests.ConnectionError:
        return jsonify({"detail": "Cannot reach backend"}), 502
    except requests.Timeout:
        return jsonify({"detail": "Backend timeout while waiting for processing to finish"}), 504


if __name__ == "__main__":
    print(f"Flask frontend starting on http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"Proxying API requests to FastAPI at {BACKEND_URL}")
    # threaded=True matters here: without it, Flask's dev server handles one
    # request at a time, so a video streaming through /api/download/... blocks
    # every other request (job status polling, model list, etc.) until it
    # finishes - which looks exactly like "the video won't play" from the
    # browser's side, since the page appears to freeze while the video loads.
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True, threaded=True)