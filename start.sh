#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "========================================="
echo "  VaaniDub — Starting All Services"
echo "========================================="

# Check prerequisites
command -v redis-server >/dev/null 2>&1 || { echo "ERROR: redis-server not installed. Run: sudo apt install redis-server"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not installed. Run: sudo apt install ffmpeg"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }

# Activate venv if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Start Redis (if not already running)
if ! redis-cli ping >/dev/null 2>&1; then
    echo "[1/4] Starting Redis..."
    redis-server --daemonize yes
else
    echo "[1/4] Redis already running"
fi

# Initialize DB if needed
if [ ! -f "./data/vaanidub.db" ]; then
    echo "[DB]  Initializing database..."
    python3 -c "
from vaanidub.config import AppConfig
from vaanidub.db.session import init_db
config = AppConfig()
config.ensure_directories()
init_db(config)
print('      Database created at ./data/vaanidub.db')
"
fi

# Start Celery worker
echo "[2/4] Starting Celery worker..."
celery -A vaanidub.worker.celery_app worker \
    --loglevel=info \
    --concurrency=2 \
    --logfile=./data/celery.log &
CELERY_PID=$!

# Start backend API
echo "[3/4] Starting backend API on :8000..."
uvicorn vaanidub.api.app:create_app \
    --factory \
    --host 0.0.0.0 \
    --port 8000 &
API_PID=$!

# Start frontend
echo "[4/4] Starting frontend on :3000..."
cd frontend && npm run dev &
FRONTEND_PID=$!
cd "$DIR"

sleep 3

echo ""
echo "========================================="
echo "  VaaniDub is running!"
echo "========================================="
echo "  Frontend:  http://localhost:3000"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Health:    http://localhost:8000/api/v1/health"
echo "========================================="
echo "  Press Ctrl+C to stop all services"
echo ""

cleanup() {
    echo ""
    echo "Shutting down VaaniDub..."
    kill $CELERY_PID $API_PID $FRONTEND_PID 2>/dev/null
    wait $CELERY_PID $API_PID $FRONTEND_PID 2>/dev/null
    echo "All services stopped."
}

trap cleanup EXIT INT TERM
wait
