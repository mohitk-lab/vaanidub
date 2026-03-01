"""OpenAI Whisper API STT provider — cloud-based fallback."""

from pathlib import Path

import structlog

from vaanidub.exceptions import ProviderError
from vaanidub.pipeline.providers.stt.base import (
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = structlog.get_logger()


class OpenAIWhisperProvider(STTProvider):
    """Speech-to-text using OpenAI Whisper API (paid fallback)."""

    name = "openai_whisper_api"

    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderError(self.name, "OpenAI API key is required")
        self.api_key = api_key

    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        import httpx

        async with httpx.AsyncClient(timeout=300) as client:
            with open(audio_path, "rb") as f:
                data = {"model": "whisper-1", "response_format": "verbose_json"}
                if language:
                    data["language"] = language

                resp = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data=data,
                    files={"file": (audio_path.name, f, "audio/wav")},
                )

            if resp.status_code != 200:
                raise ProviderError(self.name, f"API error {resp.status_code}: {resp.text[:300]}")

            result = resp.json()

        detected_lang = result.get("language", language or "en")

        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptionSegment(
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                text=seg.get("text", "").strip(),
                confidence=seg.get("avg_logprob", 0),
            ))

        return TranscriptionResult(
            segments=segments,
            language=detected_lang,
            language_confidence=0.9,  # API doesn't return confidence
            provider_name=self.name,
        )

    async def health_check(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return resp.status_code == 200
        except Exception:
            return False
