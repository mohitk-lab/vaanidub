"""Pipeline context — carries state between stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class SpeakerInfo:
    """Information about a detected speaker."""
    label: str                          # "SPEAKER_00"
    segments: list[dict]                # List of {start, end, text, ...}
    reference_clip_path: Path | None = None
    total_duration_sec: float = 0.0


@dataclass
class Segment:
    """A single speech segment from the transcript."""
    index: int
    speaker_label: str
    start_time: float
    end_time: float
    text: str = ""
    translations: dict[str, str] = field(default_factory=dict)
    emotion: str = "neutral"
    avg_pitch: float = 0.0
    speaking_rate: float = 0.0
    energy: float = 0.0
    dubbed_audio_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class PipelineContext:
    """Carries state through the entire dubbing pipeline."""

    job_id: str
    job_dir: Path
    target_languages: list[str]
    source_language: str | None = None
    source_language_confidence: float = 0.0

    # Input
    input_file_path: Path | None = None
    input_type: str = ""                    # "audio" or "video"

    # Stage 1: Ingest
    raw_audio_path: Path | None = None
    media_metadata: dict = field(default_factory=dict)

    # Stage 2: Source separation
    vocals_path: Path | None = None
    background_path: Path | None = None

    # Stage 3-4: Diarization + Transcription
    speakers: dict[str, SpeakerInfo] = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)

    # Stage 5: Prosody (stored in segments)
    # Stage 6: Translation (stored in segments)
    # Stage 7: Synthesis (stored in segments)

    # Stage 8: Final output
    final_output_paths: dict[str, Path] = field(default_factory=dict)

    # Tracking
    current_stage: int = 0
    stage_timings: dict[str, float] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)

    # Progress callback
    on_progress: Callable[[str, int, str], None] | None = None

    def report_progress(self, stage: str, percent: int, message: str = "") -> None:
        """Report progress to callback if registered."""
        if self.on_progress:
            self.on_progress(stage, percent, message)

    def stage_dir(self, stage_name: str) -> Path:
        """Get the directory for a specific stage's artifacts."""
        d = self.job_dir / stage_name
        d.mkdir(parents=True, exist_ok=True)
        return d
