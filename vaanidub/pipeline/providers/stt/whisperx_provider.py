"""WhisperX STT provider — transcription with word-level timestamps."""

from pathlib import Path

import structlog

from vaanidub.pipeline.providers.stt.base import (
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = structlog.get_logger()


class WhisperXProvider(STTProvider):
    """Speech-to-text using WhisperX (faster-whisper + alignment)."""

    name = "whisperx"

    def __init__(
        self,
        model_size: str = "large-v2",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import whisperx
        self._model = whisperx.load_model(
            self.model_size, self.device, compute_type=self.compute_type
        )

    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        import whisperx

        self._ensure_model()

        audio = whisperx.load_audio(str(audio_path))

        # Transcribe
        result = self._model.transcribe(audio, batch_size=self.batch_size, language=language)

        detected_lang = result.get("language", language or "en")
        lang_confidence = result.get("language_probability", 0.0)

        # Word-level alignment
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=detected_lang, device=self.device
            )
            result = whisperx.align(
                result["segments"], align_model, align_metadata,
                audio, self.device, return_char_alignments=False,
            )
        except Exception as e:
            logger.warning("whisperx_alignment_failed", error=str(e))

        # Build segments
        segments = []
        for seg in result.get("segments", []):
            words = []
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                    "confidence": w.get("score", 0),
                })

            segments.append(TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg.get("text", "").strip(),
                speaker=seg.get("speaker", ""),
                words=words,
                confidence=seg.get("avg_logprob", 0),
            ))

        return TranscriptionResult(
            segments=segments,
            language=detected_lang,
            language_confidence=lang_confidence,
            provider_name=self.name,
        )

    async def health_check(self) -> bool:
        try:
            self._ensure_model()
            return self._model is not None
        except Exception:
            return False
