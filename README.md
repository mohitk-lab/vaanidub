# VaaniDub

AI-powered regional dubbing tool for Indian languages. Takes any audio or video, auto-detects the source language, and produces dubbed output in 11 Indian languages with voice cloning, background preservation, and multi-speaker support.

## Features

- **11 Indian languages** — Hindi, Tamil, Telugu, Bengali, Marathi, Kannada, Malayalam, Gujarati, Assamese, Odia, Punjabi
- **8-stage pipeline** — Ingest, Separate, Diarize, Transcribe, Prosody, Translate, Synthesize, Mixdown
- **Voice cloning** — Preserves speaker identity using IndicF5 zero-shot TTS with ElevenLabs fallback
- **Multi-speaker** — Speaker diarization via pyannote.audio with per-speaker voice cloning
- **Background preservation** — Demucs source separation isolates vocals from music/effects
- **Prosody matching** — Pitch, energy, and speaking rate analysis for natural-sounding output
- **Provider fallback** — Open-source primary (WhisperX, IndicTrans2, IndicF5) with paid API fallbacks (OpenAI, Google, ElevenLabs)
- **REST API + CLI + Web UI** — FastAPI backend, Typer CLI, Next.js frontend
- **Async worker** — Celery + Redis for background job processing
- **Checkpoint/resume** — Pipeline saves checkpoints after each stage for recovery

## Quick Start

### Install (CPU-only, for development)

```bash
pip install -e ".[dev]"
```

### Install (with GPU support)

```bash
pip install -e ".[all]"
```

### CLI Usage

```bash
# Dub an audio file into Hindi and Tamil
vaanidub dub input.mp4 --target hi --target ta --output ./output

# Detect language of a file
vaanidub detect input.wav

# List supported languages
vaanidub languages

# Check GPU and model status
vaanidub models check
vaanidub models list

# Start API server
vaanidub serve --port 8000

# List jobs (requires running server)
vaanidub jobs list
vaanidub jobs status <job-id>
```

## Architecture

```
Input Audio/Video
       |
  [1. Ingest] ──────── Validate format, extract audio via FFmpeg
       |
  [2. Separate] ─────── Split vocals/background using Demucs
       |
  [3. Diarize] ──────── Identify speakers via pyannote.audio
       |
  [4. Transcribe] ───── Speech-to-text via WhisperX (word timestamps)
       |
  [5. Prosody] ──────── Analyze pitch, energy, speaking rate, emotion
       |
  [6. Translate] ────── IndicTrans2 primary → Google Translate fallback
       |
  [7. Synthesize] ───── IndicF5 voice cloning → ElevenLabs fallback
       |
  [8. Mixdown] ──────── Assemble vocals, normalize LUFS, mix with background
       |
  Dubbed Output (per language)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/jobs` | Create dubbing job (file upload) |
| `GET` | `/api/v1/jobs` | List jobs (pagination, status filter) |
| `GET` | `/api/v1/jobs/{id}` | Get job details and stage logs |
| `GET` | `/api/v1/jobs/{id}/output/{lang}` | Download dubbed output |
| `POST` | `/api/v1/jobs/{id}/retry` | Retry failed job |
| `DELETE` | `/api/v1/jobs/{id}` | Delete job and artifacts |
| `GET` | `/api/v1/languages` | List supported languages |
| `GET` | `/api/v1/health` | System health check |

## Docker Deployment

```bash
# Start all services (API, worker, Redis, PostgreSQL, frontend)
docker-compose up -d

# Frontend: http://localhost:3000
# API:      http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check vaanidub/ tests/

# Run formatter
ruff format vaanidub/ tests/

# Run with coverage
pytest tests/ --cov=vaanidub --cov-report=term-missing
```

## Supported Languages

| Code | Language | Native Name | Script | TTS Providers |
|------|----------|-------------|--------|---------------|
| `hi` | Hindi | हिन्दी | Devanagari | IndicF5, ElevenLabs |
| `ta` | Tamil | தமிழ் | Tamil | IndicF5, ElevenLabs |
| `te` | Telugu | తెలుగు | Telugu | IndicF5, ElevenLabs |
| `bn` | Bengali | বাংলা | Bengali | IndicF5, ElevenLabs |
| `mr` | Marathi | मराठी | Devanagari | IndicF5, ElevenLabs |
| `kn` | Kannada | ಕನ್ನಡ | Kannada | IndicF5, ElevenLabs |
| `ml` | Malayalam | മലയാളം | Malayalam | IndicF5, ElevenLabs |
| `gu` | Gujarati | ગુજરાતી | Gujarati | IndicF5, ElevenLabs |
| `as` | Assamese | অসমীয়া | Bengali | IndicF5 |
| `or` | Odia | ଓଡ଼ିଆ | Odia | IndicF5 |
| `pa` | Punjabi | ਪੰਜਾਬੀ | Gurmukhi | IndicF5 |

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required for GPU features
HF_TOKEN=hf_...                    # Hugging Face token (for pyannote, IndicTrans2)

# Optional paid API fallbacks
ELEVENLABS_API_KEY=...             # ElevenLabs voice cloning
GOOGLE_TRANSLATE_API_KEY=...       # Google Cloud Translation

# Infrastructure
DATABASE_URL=postgresql://...      # Default: sqlite:///vaanidub.db
REDIS_URL=redis://localhost:6379   # For Celery worker
```

## License

MIT
