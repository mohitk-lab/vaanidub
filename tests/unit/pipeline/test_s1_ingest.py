"""Tests for Stage 1: Media Ingestion."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vaanidub.exceptions import MediaValidationError
from vaanidub.pipeline.context import PipelineContext
from vaanidub.pipeline.stages.s1_ingest import IngestStage


class TestIngestStage:
    def test_stage_metadata(self):
        stage = IngestStage()
        assert stage.name == "ingest"
        assert stage.stage_number == 1

    @pytest.mark.asyncio
    async def test_rejects_missing_file(self, sample_job_dir):
        stage = IngestStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir,
            target_languages=["hi"],
            input_file_path=Path("/nonexistent/file.mp4"),
        )
        with pytest.raises(MediaValidationError, match="not found"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_rejects_unsupported_format(self, sample_job_dir, tmp_dir):
        bad_file = tmp_dir / "file.xyz"
        bad_file.write_text("not a media file")

        stage = IngestStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir,
            target_languages=["hi"],
            input_file_path=bad_file,
        )
        with pytest.raises(MediaValidationError, match="Unsupported format"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_detects_audio_type(self, sample_audio_path, sample_job_dir):
        stage = IngestStage()

        # Mock ffprobe and ffmpeg
        mock_probe = MagicMock(return_value=subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout='{"format":{"duration":"3.0","size":"96000","format_name":"wav"},"streams":[{"codec_type":"audio"}]}',
        ))
        mock_extract = MagicMock(return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        ))

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir,
            target_languages=["hi"],
            input_file_path=sample_audio_path,
        )

        side_effects = [mock_probe.return_value, mock_extract.return_value]
        with patch("subprocess.run", side_effect=side_effects):
            ctx = await stage.execute(ctx)

        assert ctx.input_type == "audio"
        assert ctx.media_metadata["duration"] == 3.0

    @pytest.mark.asyncio
    async def test_validate_output_empty_file(self, tmp_dir):
        stage = IngestStage()
        empty_file = tmp_dir / "empty.wav"
        empty_file.write_bytes(b"tiny")

        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.raw_audio_path = empty_file

        result = await stage.validate_output(ctx)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_output_no_file(self, tmp_dir):
        stage = IngestStage()
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.raw_audio_path = None
        assert await stage.validate_output(ctx) is False
