"""pyannote.audio diarization provider — identify who speaks when."""

from dataclasses import dataclass
from pathlib import Path

import structlog

from vaanidub.exceptions import ProviderError

logger = structlog.get_logger()


@dataclass
class DiarizationSegment:
    """A single speaker turn."""
    start: float
    end: float
    speaker: str
    duration: float = 0.0

    def __post_init__(self):
        self.duration = self.end - self.start


@dataclass
class DiarizationResult:
    """Full diarization result."""
    segments: list[DiarizationSegment]
    speakers: list[str]
    provider_name: str


class PyAnnoteProvider:
    """Speaker diarization using pyannote.audio."""

    name = "pyannote"

    def __init__(
        self,
        hf_token: str = "",
        device: str = "cuda",
        min_speakers: int = 1,
        max_speakers: int = 10,
    ):
        self.hf_token = hf_token
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return

        from pyannote.audio import Pipeline

        if not self.hf_token:
            raise ProviderError(
                self.name,
                "HuggingFace token required for pyannote (set HF_TOKEN)"
            )

        logger.info("loading_pyannote")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token,
        )
        self._pipeline.to(self.device)
        logger.info("pyannote_loaded")

    async def diarize(
        self,
        audio_path: Path,
        num_speakers: int | None = None,
    ) -> DiarizationResult:
        """Run speaker diarization on audio file."""
        self._ensure_pipeline()

        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        else:
            kwargs["min_speakers"] = self.min_speakers
            kwargs["max_speakers"] = self.max_speakers

        logger.info("pyannote_diarize_start", audio=str(audio_path))
        diarization = self._pipeline(str(audio_path), **kwargs)

        segments = []
        speakers_set = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizationSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
            ))
            speakers_set.add(speaker)

        speakers = sorted(speakers_set)

        logger.info(
            "pyannote_diarize_complete",
            segments=len(segments),
            speakers=len(speakers),
        )

        return DiarizationResult(
            segments=segments,
            speakers=speakers,
            provider_name=self.name,
        )

    async def health_check(self) -> bool:
        try:
            self._ensure_pipeline()
            return self._pipeline is not None
        except Exception:
            return False
