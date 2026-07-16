#!/bin/bash
set -e

echo "=============================================="
echo "  Street Light Detection Pipeline Starting"
echo "=============================================="

# ── Start FastAPI Backend (Background) ─────────────────────────────────────
echo "[1/2] Starting FastAPI backend on port 8000..."
cd /app
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &
FASTAPI_PID=$!

# ── Wait for FastAPI to Be Ready ───────────────────────────────────────────
echo "[*] Waiting for FastAPI to initialize..."
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "[✓] FastAPI backend is ready (PID: $FASTAPI_PID)"
        break
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo "[✗] FastAPI failed to start within ${MAX_WAIT}s"
        exit 1
    fi
    sleep 1
done

# ── Start Flask Frontend (Foreground) ──────────────────────────────────────
echo "[2/2] Starting Flask frontend on port 5000..."
echo "=============================================="
echo "  Access the app at: http://localhost:5000"
echo "  Direct API at:     http://localhost:8000"
echo "=============================================="

cd /app/frontend
exec python flask_app.py

# ── Cleanup (runs on container stop) ──────────────────────────────────────
trap "echo 'Shutting down...'; kill $FASTAPI_PID 2>/dev/null; exit 0" SIGTERM SIGINT