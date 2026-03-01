"""Stage 2: Audio source separation — split vocals from background using Demucs."""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

from vaanidub.exceptions import StageError
from vaanidub.pipeline.base import PipelineStage
from vaanidub.pipeline.context import PipelineContext

logger = structlog.get_logger()


class SeparateStage(PipelineStage):
    name = "separate"
    stage_number = 2
    timeout_seconds = 600

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.raw_audio_path is None:
            raise StageError(self.name, "No raw audio from previous stage")

        stage_dir = ctx.stage_dir("separate")
        ctx.report_progress("separate", 10, "Starting vocal separation with Demucs")

        # Run Demucs to separate vocals from background
        self._run_demucs(ctx.raw_audio_path, stage_dir)

        # Demucs outputs to a subdirectory structure
        vocals_path, bg_path = self._find_demucs_outputs(stage_dir, ctx.raw_audio_path.stem)

        ctx.vocals_path = vocals_path
        ctx.background_path = bg_path
        ctx.report_progress("separate", 100, "Vocal separation complete")

        logger.info(
            "separation_complete",
            vocals_size=vocals_path.stat().st_size,
            bg_size=bg_path.stat().st_size,
        )
        return ctx

    async def validate_output(self, ctx: PipelineContext) -> bool:
        if ctx.vocals_path is None or not ctx.vocals_path.exists():
            return False
        if ctx.background_path is None or not ctx.background_path.exists():
            return False

        # Check vocals are not silent
        audio, sr = sf.read(ctx.vocals_path)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            logger.error("vocals_track_silent", rms=rms)
            return False

        return True

    def _run_demucs(self, audio_path: Path, output_dir: Path) -> None:
        """Run Demucs source separation."""
        cmd = [
            "python", "-m", "demucs",
            "--two-stems", "vocals",
            "--name", "htdemucs_ft",
            "-o", str(output_dir),
            str(audio_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout_seconds
        )
        if result.returncode != 0:
            raise StageError(self.name, f"Demucs failed: {result.stderr[:500]}")

    def _find_demucs_outputs(self, output_dir: Path, stem: str) -> tuple[Path, Path]:
        """Find vocals.wav and no_vocals.wav in Demucs output directory."""
        # Demucs outputs to: output_dir/htdemucs_ft/stem/vocals.wav, no_vocals.wav
        demucs_dir = output_dir / "htdemucs_ft" / stem

        if not demucs_dir.exists():
            # Try alternative directory structures
            for d in output_dir.rglob("vocals.wav"):
                demucs_dir = d.parent
                break
            else:
                raise StageError(self.name, f"Demucs output not found in {output_dir}")

        vocals_path = demucs_dir / "vocals.wav"
        bg_path = demucs_dir / "no_vocals.wav"

        if not vocals_path.exists():
            raise StageError(self.name, "vocals.wav not found in Demucs output")
        if not bg_path.exists():
            raise StageError(self.name, "no_vocals.wav not found in Demucs output")

        return vocals_path, bg_path
