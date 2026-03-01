"""Demucs audio source separation provider — separate vocals from background."""

import subprocess
from pathlib import Path

import structlog

from vaanidub.exceptions import ProviderError

logger = structlog.get_logger()


class DemucsProvider:
    """Audio source separation using Meta's Demucs."""

    name = "demucs"

    def __init__(self, model: str = "htdemucs_ft", device: str = "cuda"):
        self.model = model
        self.device = device

    async def separate(
        self,
        audio_path: Path,
        output_dir: Path,
        two_stems: str = "vocals",
    ) -> tuple[Path, Path]:
        """
        Separate audio into vocals and background.

        Returns:
            Tuple of (vocals_path, background_path)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "demucs",
            "--two-stems", two_stems,
            "--name", self.model,
            "-o", str(output_dir),
        ]

        if self.device == "cpu":
            cmd.append("--device=cpu")

        cmd.append(str(audio_path))

        logger.info("demucs_start", model=self.model, input=str(audio_path))

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            raise ProviderError(self.name, f"Demucs failed: {result.stderr[:500]}")

        # Find output files
        stem = audio_path.stem
        demucs_dir = output_dir / self.model / stem

        # Check alternative paths
        if not demucs_dir.exists():
            # Search for vocals.wav in output directory
            for candidate in output_dir.rglob("vocals.wav"):
                demucs_dir = candidate.parent
                break
            else:
                raise ProviderError(self.name, f"Output not found in {output_dir}")

        vocals_path = demucs_dir / "vocals.wav"
        bg_path = demucs_dir / "no_vocals.wav"

        if not vocals_path.exists():
            raise ProviderError(self.name, "vocals.wav not found")
        if not bg_path.exists():
            raise ProviderError(self.name, "no_vocals.wav not found")

        logger.info(
            "demucs_complete",
            vocals_size=vocals_path.stat().st_size,
            bg_size=bg_path.stat().st_size,
        )

        return vocals_path, bg_path

    async def health_check(self) -> bool:
        """Check if Demucs is available."""
        try:
            result = subprocess.run(
                ["python", "-m", "demucs", "--help"],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False
