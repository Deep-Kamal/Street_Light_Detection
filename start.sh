#!/bin/bash
set -e

echo "──────────────────────────────────────────────"
echo " Detection Pipeline - startup"
echo "──────────────────────────────────────────────"
python -c "
try:
    import torch
    if torch.cuda.is_available():
        print(f'GPU detected: {torch.cuda.get_device_name(0)} (device count: {torch.cuda.device_count()})')
        print('Jobs with device=\"auto\" (the default) will run on GPU.')
    else:
        print('No GPU detected - running on CPU.')
        print('Jobs with device=\"auto\" (the default) will run on CPU.')
except Exception as e:
    print(f'Could not determine device availability: {e}')
"
echo "──────────────────────────────────────────────"

# Start the FastAPI backend in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start the Flask frontend in the background too
python flask_app.py &
FRONTEND_PID=$!

# If either process dies, stop the container instead of hanging
wait -n $BACKEND_PID $FRONTEND_PID
exit $?
