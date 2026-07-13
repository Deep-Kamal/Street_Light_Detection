#!/bin/bash
set -e

# Start the FastAPI backend in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start the Flask frontend in the foreground (keeps the container alive)
python flask_app.py &
FRONTEND_PID=$!

# If either process dies, stop the container instead of hanging
wait -n $BACKEND_PID $FRONTEND_PID
exit $?
