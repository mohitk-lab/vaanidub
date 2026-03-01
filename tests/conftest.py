"""Shared test fixtures for VaaniDub."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that auto-cleans."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_audio_path(tmp_dir) -> Path:
    """Create a sample 16kHz mono WAV file (3 seconds of sine wave)."""
    path = tmp_dir / "sample.wav"
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave at 440Hz with some amplitude variation
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    sf.write(path, audio.astype(np.float32), sr)
    return path


@pytest.fixture
def sample_stereo_audio(tmp_dir) -> Path:
    """Create a sample stereo WAV file."""
    path = tmp_dir / "stereo.wav"
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.3 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 554 * t)
    audio = np.column_stack([left, right]).astype(np.float32)
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def silent_audio_path(tmp_dir) -> Path:
    """Create a silent audio file."""
    path = tmp_dir / "silent.wav"
    sr = 16000
    audio = np.zeros(sr * 2, dtype=np.float32)
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def sample_job_dir(tmp_dir) -> Path:
    """Create a sample job directory structure."""
    job_dir = tmp_dir / "test-job-001"
    job_dir.mkdir()
    for stage in ["ingest", "separate", "diarize", "transcribe",
                  "prosody", "translate", "synthesize", "mixdown"]:
        (job_dir / stage).mkdir()
    return job_dir


@pytest.fixture
def app_config():
    """Create a test AppConfig."""
    os.environ.setdefault("DATABASE_URL", "sqlite:///")
    from vaanidub.config import AppConfig
    return AppConfig(debug=True, log_level="DEBUG")
