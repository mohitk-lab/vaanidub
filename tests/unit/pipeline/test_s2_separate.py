"""Tests for Stage 2: Audio Source Separation."""

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from vaanidub.exceptions import StageError
from vaanidub.pipeline.context import PipelineContext
from vaanidub.pipeline.stages.s2_separate import SeparateStage


class TestSeparateStageMetadata:
    def test_stage_metadata(self):
        stage = SeparateStage()
        assert stage.name == "separate"
        assert stage.stage_number == 2


class TestSeparateExecute:
    @pytest.mark.asyncio
    async def test_rejects_missing_audio(self, sample_job_dir):
        stage = SeparateStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.raw_audio_path = None
        with pytest.raises(StageError, match="No raw audio"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_calls_demucs_subprocess(self, sample_job_dir, sample_audio_path):
        """Verify Demucs CLI is called with correct arguments."""
        stage = SeparateStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.raw_audio_path = sample_audio_path

        stage_dir = ctx.stage_dir("separate")

        # Create expected Demucs output structure
        demucs_out = stage_dir / "htdemucs_ft" / sample_audio_path.stem
        demucs_out.mkdir(parents=True)
        vocals = demucs_out / "vocals.wav"
        bg = demucs_out / "no_vocals.wav"
        # Write real audio files
        sr = 16000
        audio = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, endpoint=False))
        sf.write(vocals, audio.astype(np.float32), sr)
        sf.write(bg, audio.astype(np.float32), sr)

        mock_run = MagicMock(return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ))

        with patch("subprocess.run", mock_run):
            ctx = await stage.execute(ctx)

        # Verify subprocess was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "demucs" in call_args  # ["python", "-m", "demucs", ...]
        assert "--two-stems" in call_args
        assert "vocals" in call_args

        assert ctx.vocals_path == vocals
        assert ctx.background_path == bg

    @pytest.mark.asyncio
    async def test_raises_on_demucs_failure(self, sample_job_dir, sample_audio_path):
        """Demucs returncode != 0 should raise StageError."""
        stage = SeparateStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.raw_audio_path = sample_audio_path

        mock_run = MagicMock(return_value=subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="CUDA out of memory"
        ))

        with patch("subprocess.run", mock_run):
            with pytest.raises(StageError, match="Demucs failed"):
                await stage.execute(ctx)


class TestSeparateFindOutputs:
    def test_finds_demucs_outputs(self, tmp_dir):
        """Standard Demucs output directory structure."""
        stage = SeparateStage()
        demucs_dir = tmp_dir / "htdemucs_ft" / "mysong"
        demucs_dir.mkdir(parents=True)
        (demucs_dir / "vocals.wav").write_bytes(b"fake")
        (demucs_dir / "no_vocals.wav").write_bytes(b"fake")

        vocals, bg = stage._find_demucs_outputs(tmp_dir, "mysong")
        assert vocals.name == "vocals.wav"
        assert bg.name == "no_vocals.wav"

    def test_finds_outputs_in_alternative_dirs(self, tmp_dir):
        """Fallback rglob search when standard dir doesn't exist."""
        stage = SeparateStage()
        # Put files in an unexpected location
        alt_dir = tmp_dir / "some" / "other" / "path"
        alt_dir.mkdir(parents=True)
        (alt_dir / "vocals.wav").write_bytes(b"fake")
        (alt_dir / "no_vocals.wav").write_bytes(b"fake")

        vocals, bg = stage._find_demucs_outputs(tmp_dir, "nonexistent_stem")
        assert vocals.name == "vocals.wav"
        assert bg.name == "no_vocals.wav"

    def test_raises_when_no_outputs_found(self, tmp_dir):
        stage = SeparateStage()
        with pytest.raises(StageError, match="Demucs output not found"):
            stage._find_demucs_outputs(tmp_dir, "missing")

    def test_raises_when_vocals_missing(self, tmp_dir):
        """Directory exists but vocals.wav is missing."""
        stage = SeparateStage()
        demucs_dir = tmp_dir / "htdemucs_ft" / "test"
        demucs_dir.mkdir(parents=True)
        # Only create no_vocals, not vocals
        (demucs_dir / "no_vocals.wav").write_bytes(b"fake")

        with pytest.raises(StageError, match="vocals.wav not found"):
            stage._find_demucs_outputs(tmp_dir, "test")


class TestSeparateValidateOutput:
    @pytest.mark.asyncio
    async def test_validate_output_valid(self, tmp_dir, sample_audio_path):
        stage = SeparateStage()
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.background_path = sample_audio_path

        assert await stage.validate_output(ctx) is True

    @pytest.mark.asyncio
    async def test_validate_output_silent_vocals(self, tmp_dir, silent_audio_path):
        stage = SeparateStage()
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.vocals_path = silent_audio_path
        ctx.background_path = silent_audio_path

        assert await stage.validate_output(ctx) is False

    @pytest.mark.asyncio
    async def test_validate_output_missing_files(self, tmp_dir):
        stage = SeparateStage()
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.vocals_path = None
        ctx.background_path = None

        assert await stage.validate_output(ctx) is False

    @pytest.mark.asyncio
    async def test_validate_output_nonexistent_path(self, tmp_dir):
        stage = SeparateStage()
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.vocals_path = tmp_dir / "nonexistent.wav"
        ctx.background_path = tmp_dir / "nonexistent_bg.wav"

        assert await stage.validate_output(ctx) is False
