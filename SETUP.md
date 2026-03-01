# VaaniDub — Production Setup Guide

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.11 |
| GPU | NVIDIA 8GB VRAM | NVIDIA 16GB+ (A100/4090) |
| RAM | 16GB | 32GB |
| CUDA | 11.8+ | 12.1 |
| Redis | 7.0+ | 7.2 |
| FFmpeg | 4.4+ | 6.0 |
| Disk | 20GB free | 50GB+ |

---

## Step 1: System Dependencies

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y \
  ffmpeg \
  redis-server \
  python3.11 python3.11-venv python3.11-dev \
  build-essential git curl

# Start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

---

## Step 2: Clone & Setup Python Environment

```bash
git clone https://github.com/mohitk-lab/vaanidub.git
cd vaanidub

python3.11 -m venv .venv
source .venv/bin/activate

# Install ALL dependencies (GPU + worker + paid APIs)
pip install -e ".[all]"

# OR install only what you need:
pip install -e ".[gpu,worker]"        # Core GPU pipeline + Celery worker
pip install -e ".[elevenlabs]"        # ElevenLabs TTS fallback (optional)
pip install -e ".[google]"            # Google Translate fallback (optional)
pip install -e ".[dev]"               # Dev/test tools
```

---

## Step 3: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your actual keys:

```bash
# REQUIRED — get from https://huggingface.co/settings/tokens
# Needed for: pyannote speaker diarization + IndicF5 TTS (gated models)
# First accept model terms at:
#   https://huggingface.co/pyannote/speaker-diarization-3.1
#   https://huggingface.co/ai4bharat/IndicF5
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OPTIONAL — paid API fallbacks
ELEVENLABS_API_KEY=sk_xxxxxxxx     # https://elevenlabs.io/app/settings
GOOGLE_TRANSLATE_API_KEY=AIza...   # Google Cloud Console
OPENAI_API_KEY=sk-...              # For Whisper API fallback
```

---

## Step 4: Initialize Database

```bash
# Creates SQLite DB at ./data/vaanidub.db
python -c "
from vaanidub.config import AppConfig
from vaanidub.db.session import init_db
config = AppConfig()
config.ensure_directories()
init_db(config)
print('Database initialized!')
"
```

---

## Step 5: Start All Services

You need **3 terminals** (or use tmux/screen):

### Terminal 1 — Backend API
```bash
cd vaanidub
source .venv/bin/activate
uvicorn vaanidub.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2 — Celery Worker
```bash
cd vaanidub
source .venv/bin/activate
celery -A vaanidub.worker.celery_app worker --loglevel=info --concurrency=2
```

### Terminal 3 — Frontend
```bash
cd vaanidub/frontend
npm install
npm run dev
```

---

## Step 6: Verify Everything Works

```bash
# Check backend health
curl http://localhost:8000/api/v1/health | python -m json.tool

# Expected output:
# {
#   "status": "ok",
#   "gpu_available": true,
#   "redis_connected": true,
#   "db_connected": true,
#   "worker_count": 2
# }

# Check frontend
open http://localhost:3000
```

---

## Quick Start (Single Command)

After setup, you can use the deploy script to start everything:

```bash
# Start all services
./start.sh
```

Create `start.sh`:
```bash
#!/bin/bash
set -e

echo "Starting Redis..."
redis-server --daemonize yes

echo "Starting Celery worker..."
cd "$(dirname "$0")"
source .venv/bin/activate
celery -A vaanidub.worker.celery_app worker --loglevel=info --concurrency=2 &
CELERY_PID=$!

echo "Starting backend API..."
uvicorn vaanidub.api.app:create_app --factory --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "Starting frontend..."
cd frontend && npm run dev &
FRONTEND_PID=$!

echo ""
echo "VaaniDub is running!"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

trap "kill $CELERY_PID $API_PID $FRONTEND_PID 2>/dev/null; redis-cli shutdown 2>/dev/null" EXIT
wait
```

---

## Dubbing Pipeline Stages

When you upload a video, it goes through 8 stages:

| # | Stage | What it does | Model/Tool |
|---|-------|-------------|------------|
| 1 | **Ingest** | Extract audio from video, detect format | FFmpeg |
| 2 | **Separate** | Split vocals from background music | Demucs (htdemucs_ft) |
| 3 | **Diarize** | Identify different speakers | PyAnnote 3.1 |
| 4 | **Transcribe** | Speech-to-text with timestamps | Faster-Whisper (large-v2) |
| 5 | **Prosody** | Analyze speech rhythm, pitch, emotion | Praat/Parselmouth |
| 6 | **Translate** | Translate text to target languages | IndicTrans2 |
| 7 | **Synthesize** | Generate dubbed audio (TTS) | IndicF5 / ElevenLabs |
| 8 | **Mixdown** | Mix dubbed audio with background music | FFmpeg + PyDub |

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Use smaller Whisper model
export PROVIDERS__STT__MODEL_SIZE=medium
# Or enable CPU offloading
export GPU__MODEL_OFFLOAD_TO_CPU=true
```

### Redis connection refused
```bash
redis-cli ping   # Should return PONG
sudo systemctl start redis-server
```

### HuggingFace 401 error
- Make sure you accepted model terms on HuggingFace for:
  - `pyannote/speaker-diarization-3.1`
  - `ai4bharat/IndicF5`
- Verify token: `huggingface-cli whoami`

### Worker not picking up jobs
```bash
# Check Celery is connected
celery -A vaanidub.worker.celery_app inspect ping
```

---

## Deploy to Production

For production deployment, see [Vercel Deployment](#vercel-frontend) for frontend and use Docker for backend:

### Frontend (Vercel)
1. Go to https://vercel.com/new
2. Import `mohitk-lab/vaanidub`
3. Set Root Directory: `frontend`
4. Set env: `NEXT_PUBLIC_API_URL=https://your-api-server.com`
5. Deploy

### Backend (Docker)
```bash
docker compose up -d   # Uses docker-compose.yml (API + Worker + Redis + PostgreSQL)
```
