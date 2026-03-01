"""Stage 7: Voice cloning TTS — synthesize dubbed speech per speaker using IndicF5 / ElevenLabs."""

from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

from vaanidub.constants import MAX_TIME_STRETCH
from vaanidub.exceptions import AllProvidersFailed, StageError
from vaanidub.pipeline.base import PipelineStage
from vaanidub.pipeline.context import PipelineContext

logger = structlog.get_logger()


class SynthesizeStage(PipelineStage):
    name = "synthesize"
    stage_number = 7
    timeout_seconds = 1800  # 30 min for long content

    def __init__(self, device: str = "cuda", elevenlabs_api_key: str = "", hf_token: str = ""):
        self.device = device
        self.elevenlabs_api_key = elevenlabs_api_key
        self.hf_token = hf_token
        self._indicf5_model = None

    def _load_indicf5(self):
        """Lazy-load IndicF5 model."""
        if self._indicf5_model is not None:
            return
        from transformers import AutoModel
        self._indicf5_model = AutoModel.from_pretrained(
            "ai4bharat/IndicF5",
            trust_remote_code=True,
        )

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.segments:
            raise StageError(self.name, "No segments to synthesize")
        if not ctx.speakers:
            raise StageError(self.name, "No speaker info available")

        stage_dir = ctx.stage_dir("synthesize")
        total_segments = len(ctx.segments) * len(ctx.target_languages)
        processed = 0

        for target_lang in ctx.target_languages:
            lang_dir = stage_dir / target_lang
            lang_dir.mkdir(parents=True, exist_ok=True)

            for seg in ctx.segments:
                translated_text = seg.translations.get(target_lang, "")
                if not translated_text:
                    processed += 1
                    continue

                # Get speaker reference clip
                speaker = ctx.speakers.get(seg.speaker_label)
                ref_audio_path = speaker.reference_clip_path if speaker else None

                if ref_audio_path is None or not ref_audio_path.exists():
                    logger.warning("no_reference_clip", speaker=seg.speaker_label)
                    ref_audio_path = ctx.vocals_path  # Fallback to full vocals

                # Get reference text (original text of the reference clip segment)
                ref_text = self._get_reference_text(ctx, seg.speaker_label)

                # Generate dubbed audio
                output_path = lang_dir / f"seg_{seg.index:04d}.wav"
                try:
                    await self._synthesize_indicf5(
                        text=translated_text,
                        ref_audio_path=ref_audio_path,
                        ref_text=ref_text,
                        output_path=output_path,
                    )
                except Exception as e:
                    logger.warning("indicf5_failed", error=str(e), segment=seg.index)
                    try:
                        await self._synthesize_elevenlabs(
                            text=translated_text,
                            ref_audio_path=ref_audio_path,
                            target_lang=target_lang,
                            output_path=output_path,
                        )
                    except Exception as e2:
                        raise AllProvidersFailed("tts", e2) from e2

                # Time-stretch to match original segment duration
                if output_path.exists():
                    self._match_duration(output_path, seg.duration)

                seg.dubbed_audio_paths[target_lang] = output_path
                processed += 1

                if processed % 5 == 0 or processed == total_segments:
                    pct = int(100 * processed / total_segments)
                    ctx.report_progress("synthesize", pct,
                                        f"Synthesized {processed}/{total_segments}")

        ctx.report_progress("synthesize", 100, "Synthesis complete")
        logger.info("synthesis_complete", total_segments=total_segments)
        return ctx

    async def validate_output(self, ctx: PipelineContext) -> bool:
        for target_lang in ctx.target_languages:
            dubbed = sum(
                1 for s in ctx.segments
                if s.dubbed_audio_paths.get(target_lang) and
                s.dubbed_audio_paths[target_lang].exists()
            )
            if dubbed == 0:
                logger.error("no_dubbed_segments", target=target_lang)
                return False
        return True

    async def _synthesize_indicf5(
        self, text: str, ref_audio_path: Path, ref_text: str, output_path: Path
    ) -> None:
        """Generate speech using IndicF5 with voice cloning."""
        self._load_indicf5()

        audio = self._indicf5_model(
            text,
            ref_audio_path=str(ref_audio_path),
            ref_text=ref_text,
        )

        # IndicF5 returns numpy array
        if isinstance(audio, np.ndarray):
            sf.write(output_path, audio, 24000)
        elif isinstance(audio, tuple):
            # Some models return (audio, sample_rate)
            sf.write(output_path, audio[0], audio[1])
        else:
            raise StageError(self.name, f"Unexpected IndicF5 output type: {type(audio)}")

    async def _synthesize_elevenlabs(
        self, text: str, ref_audio_path: Path, target_lang: str, output_path: Path
    ) -> None:
        """Fallback: generate speech using ElevenLabs API with voice cloning."""
        if not self.elevenlabs_api_key:
            raise StageError(self.name, "ElevenLabs API key not configured")

        import httpx

        # Upload voice clone
        async with httpx.AsyncClient(timeout=120) as client:
            # Add voice from reference clip
            with open(ref_audio_path, "rb") as f:
                voice_resp = await client.post(
                    "https://api.elevenlabs.io/v1/voices/add",
                    headers={"xi-api-key": self.elevenlabs_api_key},
                    data={"name": f"vaanidub_clone_{target_lang}"},
                    files={"files": ("ref.wav", f, "audio/wav")},
                )
                voice_resp.raise_for_status()
                voice_id = voice_resp.json()["voice_id"]

            # Generate speech
            tts_resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": self.elevenlabs_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                },
            )
            tts_resp.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(tts_resp.content)

            # Cleanup: delete the cloned voice
            await client.delete(
                f"https://api.elevenlabs.io/v1/voices/{voice_id}",
                headers={"xi-api-key": self.elevenlabs_api_key},
            )

    def _match_duration(self, audio_path: Path, target_duration: float) -> None:
        """Time-stretch audio to match target segment duration."""
        audio, sr = sf.read(audio_path)
        current_duration = len(audio) / sr

        if current_duration <= 0 or target_duration <= 0:
            return

        ratio = current_duration / target_duration

        # Only stretch if ratio is significant and within bounds
        if abs(ratio - 1.0) < 0.05:
            return  # Close enough, skip

        if ratio > MAX_TIME_STRETCH:
            # Too much stretching needed, cap it
            ratio = MAX_TIME_STRETCH
            logger.warning("duration_mismatch_capped", ratio=ratio, target=target_duration)

        if ratio < 1.0 / MAX_TIME_STRETCH:
            ratio = 1.0 / MAX_TIME_STRETCH

        try:
            import pyrubberband as pyrb
            stretched = pyrb.time_stretch(audio, sr, ratio)
            sf.write(audio_path, stretched, sr)
        except ImportError:
            # Fallback: simple resampling (lower quality)
            from scipy.signal import resample
            target_samples = int(len(audio) / ratio)
            stretched = resample(audio, target_samples)
            sf.write(audio_path, stretched, sr)

    def _get_reference_text(self, ctx: PipelineContext, speaker_label: str) -> str:
        """Get the original text for the speaker's reference clip."""
        speaker = ctx.speakers.get(speaker_label)
        if not speaker or not speaker.segments:
            return ""

        # Find the longest segment (which was likely used as reference)
        longest = max(speaker.segments, key=lambda s: s["duration"])

        # Find matching transcript segment
        for seg in ctx.segments:
            if (seg.speaker_label == speaker_label and
                    abs(seg.start_time - longest["start"]) < 0.5):
                return seg.text
        return ""
