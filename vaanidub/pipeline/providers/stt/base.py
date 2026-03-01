"""Abstract base for Speech-to-Text providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TranscriptionSegment:
    """A single transcribed segment with word-level detail."""
    start: float
    end: float
    text: str
    speaker: str = ""
    words: list[dict] = field(default_factory=list)  # [{word, start, end, confidence}]
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """Full transcription result."""
    segments: list[TranscriptionSegment]
    language: str
    language_confidence: float
    provider_name: str


class STTProvider(ABC):
    """Abstract interface for STT providers."""

    name: str

    @abstractmethod
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file with word-level timestamps and speaker diarization."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify provider is operational."""
        ...
