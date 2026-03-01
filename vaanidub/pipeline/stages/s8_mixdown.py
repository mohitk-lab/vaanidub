"""Stage 8: Audio mixing & mastering — combine dubbed vocals with original background."""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

from vaanidub.constants import TARGET_LUFS
from vaanidub.exceptions import StageError
from vaanidub.pipeline.base import PipelineStage
from vaanidub.pipeline.context import PipelineContext

logger = structlog.get_logger()


class MixdownStage(PipelineStage):
    name = "mixdown"
    stage_number = 8
    timeout_seconds = 300

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.segments:
            raise StageError(self.name, "No segments available")
        if ctx.background_path is None:
            raise StageError(self.name, "No background audio from separation stage")

        stage_dir = ctx.stage_dir("mixdown")
        bg_audio, bg_sr = sf.read(ctx.background_path)
        total_duration = ctx.media_metadata.get("duration", len(bg_audio) / bg_sr)

        for lang_idx, target_lang in enumerate(ctx.target_languages):
            ctx.report_progress(
                "mixdown",
                int(100 * lang_idx / len(ctx.target_languages)),
                f"Mixing {target_lang} output",
            )

            # Build the dubbed vocal track by placing segments on timeline
            dubbed_vocals = self._assemble_vocal_track(
                ctx.segments, target_lang, total_duration, bg_sr
            )

            # Resample background to match if needed
            bg_track = bg_audio
            if len(bg_track) != len(dubbed_vocals):
                # Pad or trim to match
                if len(bg_track) < len(dubbed_vocals):
                    bg_track = np.pad(bg_track, (0, len(dubbed_vocals) - len(bg_track)))
                else:
                    bg_track = bg_track[:len(dubbed_vocals)]

            # Mix vocals and background
            # Vocals at full volume, background at reduced level (standard dubbing mix)
            mixed = dubbed_vocals * 1.0 + bg_track * 0.4

            # Normalize to prevent clipping
            peak = np.max(np.abs(mixed))
            if peak > 0.95:
                mixed = mixed * (0.95 / peak)

            # Apply LUFS normalization
            mixed = self._normalize_lufs(mixed, bg_sr)

            # Write mixed audio
            mixed_audio_path = stage_dir / f"dubbed_{target_lang}.wav"
            sf.write(mixed_audio_path, mixed, bg_sr)

            # If input was video, mux the dubbed audio into the original video
            if ctx.input_type == "video" and ctx.input_file_path:
                output_path = stage_dir / f"dubbed_{target_lang}.mp4"
                self._mux_audio_into_video(
                    ctx.input_file_path, mixed_audio_path, output_path
                )
                ctx.final_output_paths[target_lang] = output_path
            else:
                ctx.final_output_paths[target_lang] = mixed_audio_path

        ctx.report_progress("mixdown", 100, "Mixdown complete")
        logger.info(
            "mixdown_complete",
            outputs={k: str(v) for k, v in ctx.final_output_paths.items()},
        )
        return ctx

    async def validate_output(self, ctx: PipelineContext) -> bool:
        for target_lang in ctx.target_languages:
            path = ctx.final_output_paths.get(target_lang)
            if path is None or not path.exists():
                logger.error("output_missing", target=target_lang)
                return False
            if path.stat().st_size < 1000:
                logger.error("output_too_small", target=target_lang)
                return False
        return True

    def _assemble_vocal_track(
        self,
        segments: list,
        target_lang: str,
        total_duration: float,
        sample_rate: int,
    ) -> np.ndarray:
        """Place dubbed segments onto a timeline, filling gaps with silence."""
        total_samples = int(total_duration * sample_rate)
        track = np.zeros(total_samples, dtype=np.float32)

        for seg in segments:
            dubbed_path = seg.dubbed_audio_paths.get(target_lang)
            if dubbed_path is None or not dubbed_path.exists():
                continue

            seg_audio, seg_sr = sf.read(dubbed_path)

            # Resample if sample rates differ
            if seg_sr != sample_rate:
                from scipy.signal import resample
                new_len = int(len(seg_audio) * sample_rate / seg_sr)
                seg_audio = resample(seg_audio, new_len)

            # Place on timeline
            start_sample = int(seg.start_time * sample_rate)
            end_sample = start_sample + len(seg_audio)

            if end_sample > total_samples:
                seg_audio = seg_audio[:total_samples - start_sample]
                end_sample = total_samples

            if start_sample < total_samples:
                # Cross-fade with existing content (5ms)
                fade_samples = min(int(0.005 * sample_rate), len(seg_audio))
                if fade_samples > 0:
                    fade_in = np.linspace(0, 1, fade_samples)
                    seg_audio[:fade_samples] *= fade_in

                track[start_sample:start_sample + len(seg_audio)] = seg_audio

        return track

    def _normalize_lufs(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize audio to target LUFS level."""
        # Simple RMS-based approximation of LUFS normalization
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-8:
            return audio

        # Target RMS corresponding to ~TARGET_LUFS
        # LUFS ≈ 20 * log10(RMS) - 0.691 (simplified)
        target_rms = 10 ** ((TARGET_LUFS + 0.691) / 20)
        gain = target_rms / rms

        # Limit gain to prevent excessive amplification
        gain = min(gain, 10.0)

        normalized = audio * gain

        # Final limiter
        peak = np.max(np.abs(normalized))
        if peak > 0.98:
            normalized = normalized * (0.98 / peak)

        return normalized

    def _mux_audio_into_video(
        self, video_path: Path, audio_path: Path, output_path: Path
    ) -> None:
        """Replace the audio track in a video file using ffmpeg."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",         # Copy video stream as-is
            "-map", "0:v:0",        # Video from first input
            "-map", "1:a:0",        # Audio from second input
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise StageError(self.name, f"ffmpeg mux failed: {result.stderr[:500]}")
