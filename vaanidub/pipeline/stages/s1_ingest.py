"""Stage 1: Media ingestion — extract audio from video, validate, convert to 16kHz mono WAV."""

import json
import subprocess
from pathlib import Path

import structlog

from vaanidub.constants import AUDIO_FORMATS, SAMPLE_RATE, VIDEO_FORMATS
from vaanidub.exceptions import MediaValidationError
from vaanidub.pipeline.base import PipelineStage
from vaanidub.pipeline.context import PipelineContext

logger = structlog.get_logger()


class IngestStage(PipelineStage):
    name = "ingest"
    stage_number = 1

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        input_path = ctx.input_file_path
        if input_path is None or not input_path.exists():
            raise MediaValidationError(f"Input file not found: {input_path}")

        suffix = input_path.suffix.lower()
        if suffix not in AUDIO_FORMATS and suffix not in VIDEO_FORMATS:
            raise MediaValidationError(
                f"Unsupported format: {suffix}. "
                f"Supported: {AUDIO_FORMATS | VIDEO_FORMATS}"
            )

        ctx.input_type = "video" if suffix in VIDEO_FORMATS else "audio"

        # Probe media metadata with ffprobe
        ctx.media_metadata = self._probe_media(input_path)
        ctx.report_progress("ingest", 30, "Probed media metadata")

        duration = ctx.media_metadata.get("duration", 0)
        ctx.media_metadata["duration"] = duration
        logger.info("media_probed", input_type=ctx.input_type, duration=duration)

        # Extract/convert audio to 16kHz mono WAV
        stage_dir = ctx.stage_dir("ingest")
        raw_audio_path = stage_dir / "raw_audio.wav"

        self._extract_audio(input_path, raw_audio_path)
        ctx.raw_audio_path = raw_audio_path
        ctx.report_progress("ingest", 100, "Audio extracted")

        return ctx

    async def validate_output(self, ctx: PipelineContext) -> bool:
        if ctx.raw_audio_path is None or not ctx.raw_audio_path.exists():
            return False
        # Check file is not empty
        if ctx.raw_audio_path.stat().st_size < 1000:
            logger.error("extracted_audio_too_small", size=ctx.raw_audio_path.stat().st_size)
            return False
        return True

    def _probe_media(self, path: Path) -> dict:
        """Get media metadata using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise MediaValidationError(f"ffprobe failed: {result.stderr}")

        data = json.loads(result.stdout)
        fmt = data.get("format", {})

        return {
            "duration": float(fmt.get("duration", 0)),
            "size_bytes": int(fmt.get("size", 0)),
            "format_name": fmt.get("format_name", ""),
            "streams": len(data.get("streams", [])),
            "has_video": any(s["codec_type"] == "video" for s in data.get("streams", [])),
            "has_audio": any(s["codec_type"] == "audio" for s in data.get("streams", [])),
        }

    def _extract_audio(self, input_path: Path, output_path: Path) -> None:
        """Extract audio and convert to 16kHz mono WAV using ffmpeg."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn",                          # No video
            "-acodec", "pcm_s16le",         # 16-bit PCM
            "-ar", str(SAMPLE_RATE),        # 16kHz
            "-ac", "1",                     # Mono
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise MediaValidationError(f"ffmpeg audio extraction failed: {result.stderr[:500]}")
