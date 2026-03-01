"""ElevenLabs TTS provider — paid API fallback for voice cloning."""

from pathlib import Path

import structlog

from vaanidub.exceptions import ProviderError
from vaanidub.pipeline.providers.tts.base import TTSProvider, TTSRequest, TTSResult

logger = structlog.get_logger()

# ElevenLabs supported Indian languages
ELEVENLABS_LANGUAGES = ["hi", "ta", "te", "bn", "mr", "kn", "ml", "gu"]


class ElevenLabsProvider(TTSProvider):
    """Voice cloning TTS using ElevenLabs API."""

    name = "elevenlabs"
    supported_languages = ELEVENLABS_LANGUAGES

    def __init__(self, api_key: str, model: str = "eleven_multilingual_v2"):
        if not api_key:
            raise ProviderError(self.name, "ElevenLabs API key is required")
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.elevenlabs.io/v1"
        self._voice_cache: dict[str, str] = {}  # ref_path -> voice_id

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        import httpx
        import soundfile as sf

        ref_key = str(request.reference_audio_path)

        async with httpx.AsyncClient(timeout=120) as client:
            # Get or create cloned voice
            if ref_key not in self._voice_cache:
                voice_id = await self._create_voice_clone(client, request.reference_audio_path)
                self._voice_cache[ref_key] = voice_id

            voice_id = self._voice_cache[ref_key]

            # Generate speech
            resp = await client.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": request.text,
                    "model_id": self.model,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.5,
                        "use_speaker_boost": True,
                    },
                },
            )

            if resp.status_code != 200:
                raise ProviderError(
                    self.name, f"TTS failed: {resp.status_code} {resp.text[:200]}"
                )

            # Save as MP3 first, then convert to WAV
            output_mp3 = request.reference_audio_path.parent / f"el_out_{id(request)}.mp3"
            output_wav = request.reference_audio_path.parent / f"el_out_{id(request)}.wav"

            output_mp3.write_bytes(resp.content)

            # Convert MP3 to WAV using pydub
            from pydub import AudioSegment
            audio_seg = AudioSegment.from_mp3(str(output_mp3))
            audio_seg.export(str(output_wav), format="wav")
            output_mp3.unlink()  # Clean up MP3

            # Get duration
            audio_data, sr = sf.read(output_wav)
            actual_duration = len(audio_data) / sr

        return TTSResult(
            audio_path=output_wav,
            actual_duration_sec=actual_duration,
            sample_rate=sr,
            provider_name=self.name,
        )

    async def _create_voice_clone(self, client, ref_audio_path: Path) -> str:
        """Upload reference audio to create a cloned voice on ElevenLabs."""
        with open(ref_audio_path, "rb") as f:
            resp = await client.post(
                f"{self.base_url}/voices/add",
                headers={"xi-api-key": self.api_key},
                data={
                    "name": f"vaanidub_{ref_audio_path.stem}",
                    "description": "Auto-created by VaaniDub for dubbing",
                },
                files={"files": (ref_audio_path.name, f, "audio/wav")},
            )

        if resp.status_code != 200:
            raise ProviderError(
                self.name, f"Voice clone creation failed: {resp.status_code} {resp.text[:200]}"
            )

        voice_id = resp.json()["voice_id"]
        logger.info("elevenlabs_voice_created", voice_id=voice_id)
        return voice_id

    async def cleanup_voices(self) -> None:
        """Delete all cloned voices created during this session."""
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            for ref_key, voice_id in self._voice_cache.items():
                try:
                    await client.delete(
                        f"{self.base_url}/voices/{voice_id}",
                        headers={"xi-api-key": self.api_key},
                    )
                    logger.debug("elevenlabs_voice_deleted", voice_id=voice_id)
                except Exception as e:
                    logger.warning("elevenlabs_voice_delete_failed", error=str(e))

        self._voice_cache.clear()

    def supports_language(self, language_code: str) -> bool:
        return language_code in ELEVENLABS_LANGUAGES

    async def health_check(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.base_url}/voices",
                    headers={"xi-api-key": self.api_key},
                )
                return resp.status_code == 200
        except Exception:
            return False
