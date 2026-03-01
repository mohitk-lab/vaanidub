"""Demo pipeline — simulates all 8 dubbing stages without ML models or ffmpeg.

Works without GPU, Redis, ffmpeg, or any heavy dependencies. Uses pure Python
(numpy + soundfile) for all audio processing. Produces real audio output
with pitch-shifted vocals to simulate language dubbing.
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog
from scipy.signal import resample as scipy_resample

from vaanidub.config import AppConfig
from vaanidub.constants import SAMPLE_RATE
from vaanidub.db.models import Job, Segment as SegmentModel, Speaker, StageLog
from vaanidub.db.session import get_session

logger = structlog.get_logger()

# Simulated translations for demo
DEMO_TRANSLATIONS = {
    "hi": "यह एक डेमो डबिंग है। वाणीडब आपकी आवाज़ को हिंदी में बदल रहा है।",
    "ta": "இது ஒரு டெமோ டப்பிங். வாணிடப் உங்கள் குரலை தமிழில் மாற்றுகிறது.",
    "te": "ఇది ఒక డెమో డబ్బింగ్. వాణీడబ్ మీ గొంతును తెలుగులోకి మారుస్తోంది.",
    "bn": "এটি একটি ডেমো ডাবিং। ভানিডাব আপনার কণ্ঠকে বাংলায় রূপান্তর করছে।",
    "mr": "हा एक डेमो डबिंग आहे. वाणीडब तुमचा आवाज मराठीत बदलत आहे.",
    "kn": "ಇದು ಡೆಮೊ ಡಬ್ಬಿಂಗ್. ವಾಣಿಡಬ್ ನಿಮ್ಮ ಧ್ವನಿಯನ್ನು ಕನ್ನಡಕ್ಕೆ ಬದಲಾಯಿಸುತ್ತಿದೆ.",
    "ml": "ഇത് ഒരു ഡെമോ ഡബ്ബിംഗ് ആണ്. വാണിഡബ് നിങ്ങളുടെ ശബ്ദം മലയാളത്തിലേക്ക് മാറ്റുന്നു.",
    "gu": "આ એક ડેમો ડબિંગ છે. વાણીડબ તમારો અવાજ ગુજરાતીમાં બદલી રહ્યું છે.",
    "as": "এইটো এটা ডেমো ডাবিং। ভানিডাব আপোনাৰ কণ্ঠক অসমীয়ালৈ সলনি কৰিছে।",
    "or": "ଏହା ଏକ ଡେମୋ ଡବିଂ। ଭାନୀଡବ ଆପଣଙ୍କ କଣ୍ଠକୁ ଓଡ଼ିଆକୁ ବଦଳାଉଛି।",
    "pa": "ਇਹ ਇੱਕ ਡੈਮੋ ਡਬਿੰਗ ਹੈ। ਵਾਣੀਡੱਬ ਤੁਹਾਡੀ ਆਵਾਜ਼ ਨੂੰ ਪੰਜਾਬੀ ਵਿੱਚ ਬਦਲ ਰਿਹਾ ਹੈ।",
}

# Pitch shift factor per language (>1 = higher pitch, <1 = lower)
PITCH_FACTORS = {
    "hi": 1.08, "ta": 0.94, "te": 1.12, "bn": 0.92, "mr": 1.06,
    "kn": 0.88, "ml": 1.15, "gu": 0.97, "as": 1.10, "or": 0.93, "pa": 1.05,
}


def run_demo_pipeline(job_id: str) -> None:
    """Run the full demo pipeline for a job. Call from a background thread."""
    log = logger.bind(job_id=job_id)
    log.info("demo_pipeline_start")

    config = AppConfig()
    config.ensure_directories()

    session = get_session()
    job = session.query(Job).filter(Job.id == job_id).first()
    if not job:
        log.error("job_not_found")
        return

    try:
        job.status = "processing"
        job.started_at = datetime.utcnow()
        session.commit()

        target_languages = json.loads(job.target_languages)
        input_path = Path(job.input_file_path)
        job_dir = input_path.parent.parent

        # ── Stage 1: Ingest ──
        _update_stage(session, job, "ingest", 1, 0.05)
        ingest_dir = job_dir / "ingest"
        ingest_dir.mkdir(parents=True, exist_ok=True)
        raw_audio = ingest_dir / "raw_audio.wav"

        _load_audio_to_wav(input_path, raw_audio)
        audio_data, sr = sf.read(raw_audio)
        duration = len(audio_data) / sr
        _log_stage(session, job_id, "ingest", 1, "completed", round(duration, 1))

        # ── Stage 2: Separate ──
        _update_stage(session, job, "separate", 2, 0.15)
        sep_dir = job_dir / "separate"
        sep_dir.mkdir(parents=True, exist_ok=True)
        vocals_path = sep_dir / "vocals.wav"
        background_path = sep_dir / "background.wav"

        # Demo: copy audio as vocals, create quiet background
        shutil.copy2(raw_audio, vocals_path)
        _create_quiet_background(raw_audio, background_path)
        time.sleep(1)
        _log_stage(session, job_id, "separate", 2, "completed", 1.0)

        # ── Stage 3: Diarize ──
        _update_stage(session, job, "diarize", 3, 0.25)
        time.sleep(0.5)

        # Demo: create one speaker with the full audio
        speaker = Speaker(
            job_id=job_id,
            speaker_label="SPEAKER_00",
            reference_clip_path=str(vocals_path),
            total_duration_sec=duration,
        )
        session.add(speaker)
        session.commit()
        _log_stage(session, job_id, "diarize", 3, "completed", 0.5)

        # ── Stage 4: Transcribe ──
        _update_stage(session, job, "transcribe", 4, 0.40)
        time.sleep(0.5)

        # Demo: create segments (split into ~5s chunks)
        chunk_dur = 5.0
        segments = []
        idx = 0
        t = 0.0
        while t < duration:
            end_t = min(t + chunk_dur, duration)
            seg = SegmentModel(
                job_id=job_id,
                speaker_id=speaker.id,
                segment_index=idx,
                start_time=t,
                end_time=end_t,
                duration=end_t - t,
                original_text=f"[Demo speech segment {idx + 1}]",
                source_language="en",
            )
            session.add(seg)
            segments.append(seg)
            idx += 1
            t = end_t
        session.commit()

        job.source_language = "en"
        job.source_language_confidence = 0.95
        session.commit()
        _log_stage(session, job_id, "transcribe", 4, "completed", 0.5)

        # ── Stage 5: Prosody ──
        _update_stage(session, job, "prosody", 5, 0.50)
        time.sleep(0.3)
        for seg in segments:
            seg.emotion = "neutral"
            seg.avg_pitch = 120.0
            seg.speaking_rate = 3.5
            seg.energy = 0.6
        session.commit()
        _log_stage(session, job_id, "prosody", 5, "completed", 0.3)

        # ── Stage 6: Translate ──
        _update_stage(session, job, "translate", 6, 0.60)
        time.sleep(0.5)
        for seg in segments:
            translations = {}
            for lang in target_languages:
                translations[lang] = DEMO_TRANSLATIONS.get(lang, f"[{lang} translation]")
            seg.translations = json.dumps(translations)
        session.commit()
        _log_stage(session, job_id, "translate", 6, "completed", 0.5)

        # ── Stage 7: Synthesize ──
        _update_stage(session, job, "synthesize", 7, 0.75)
        synth_dir = job_dir / "synthesize"
        synth_dir.mkdir(parents=True, exist_ok=True)

        for lang in target_languages:
            pitch_factor = PITCH_FACTORS.get(lang, 1.0)
            for seg in segments:
                seg_audio_path = synth_dir / f"seg{seg.segment_index}_{lang}.wav"
                _extract_segment_pitched(
                    audio_data, sr, seg_audio_path,
                    seg.start_time, seg.end_time, pitch_factor
                )
                paths = json.loads(seg.dubbed_audio_paths) if seg.dubbed_audio_paths else {}
                paths[lang] = str(seg_audio_path)
                seg.dubbed_audio_paths = json.dumps(paths)
            session.commit()

        _log_stage(session, job_id, "synthesize", 7, "completed", 1.0)

        # ── Stage 8: Mixdown ──
        _update_stage(session, job, "mixdown", 8, 0.90)
        mixdown_dir = job_dir / "mixdown"
        mixdown_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}
        for lang in target_languages:
            pitched_full = mixdown_dir / f"dubbed_{lang}.wav"
            _pitch_shift_full(audio_data, sr, pitched_full, PITCH_FACTORS.get(lang, 1.0))
            output_paths[lang] = str(pitched_full)

        _log_stage(session, job_id, "mixdown", 8, "completed", 1.0)

        # ── Complete ──
        job.status = "completed"
        job.progress = 1.0
        job.current_stage = "completed"
        job.completed_at = datetime.utcnow()
        job.duration_seconds = duration
        job.output_paths = json.dumps(output_paths)
        session.commit()

        log.info("demo_pipeline_complete", outputs=list(output_paths.keys()))

    except Exception as e:
        log.error("demo_pipeline_failed", error=str(e))
        job.status = "failed"
        job.error_message = str(e)[:1000]
        job.error_stage = job.current_stage
        session.commit()
    finally:
        session.close()


def _update_stage(session, job: Job, stage: str, num: int, progress: float) -> None:
    """Update job's current stage and progress."""
    job.current_stage = stage
    job.progress = progress
    session.commit()
    logger.info("demo_stage", stage=stage, progress=progress)


def _log_stage(session, job_id: str, stage: str, num: int, status: str, dur: float) -> None:
    """Write a StageLog record."""
    log = StageLog(
        job_id=job_id,
        stage_name=stage,
        stage_number=num,
        status=status,
        provider_used="demo",
        duration_sec=dur,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )
    session.add(log)
    session.commit()


def _load_audio_to_wav(input_path: Path, output_path: Path) -> None:
    """Load any soundfile-supported audio and save as 16kHz mono WAV."""
    try:
        audio, sr = sf.read(input_path)
    except Exception:
        raise RuntimeError(f"Cannot read audio file: {input_path}")

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        new_length = int(len(audio) * SAMPLE_RATE / sr)
        audio = scipy_resample(audio, new_length)

    sf.write(output_path, audio.astype(np.float32), SAMPLE_RATE)


def _create_quiet_background(source_wav: Path, output_path: Path) -> None:
    """Create a quiet version of audio to simulate background track."""
    audio, sr = sf.read(source_wav)
    quiet = audio * 0.05
    sf.write(output_path, quiet.astype(np.float32), sr)


def _extract_segment_pitched(
    audio_data: np.ndarray, sr: int, output_path: Path,
    start: float, end: float, pitch_factor: float,
) -> None:
    """Extract a segment from audio and pitch-shift it (pure Python)."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = audio_data[start_sample:end_sample].copy()

    # Pitch shift by resampling: resample to change pitch, then resample back to original length
    if pitch_factor != 1.0 and len(segment) > 100:
        stretched_len = int(len(segment) / pitch_factor)
        stretched = scipy_resample(segment, stretched_len)
        # Resample back to original duration to keep timing
        pitched = scipy_resample(stretched, len(segment))
    else:
        pitched = segment

    sf.write(output_path, pitched.astype(np.float32), sr)


def _pitch_shift_full(
    audio_data: np.ndarray, sr: int, output_path: Path, pitch_factor: float,
) -> None:
    """Pitch-shift entire audio (pure Python via scipy resample)."""
    if pitch_factor != 1.0 and len(audio_data) > 100:
        stretched_len = int(len(audio_data) / pitch_factor)
        stretched = scipy_resample(audio_data, stretched_len)
        pitched = scipy_resample(stretched, len(audio_data))
    else:
        pitched = audio_data.copy()

    sf.write(output_path, pitched.astype(np.float32), sr)
