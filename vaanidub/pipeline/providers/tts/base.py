"""Abstract base for Text-to-Speech / voice cloning providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TTSRequest:
    """Request to synthesize speech with voice cloning."""
    text: str
    target_language: str
    reference_audio_path: Path       # Speaker reference clip for voice cloning
    reference_text: str              # Text spoken in reference audio
    target_duration_sec: float       # Desired output duration
    speaking_rate: float = 1.0
    pitch_shift_semitones: float = 0.0
    emotion_hint: str = "neutral"


@dataclass
class TTSResult:
    """Result of TTS synthesis."""
    audio_path: Path
    actual_duration_sec: float
    sample_rate: int
    provider_name: str


class TTSProvider(ABC):
    """Abstract interface for TTS / voice cloning providers."""

    name: str
    supported_languages: list[str]

    @abstractmethod
    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """Generate speech from text with voice cloning."""
        ...

    @abstractmethod
    def supports_language(self, language_code: str) -> bool:
        """Check if this provider supports the given language."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...
