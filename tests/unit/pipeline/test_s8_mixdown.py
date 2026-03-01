"""Tests for Stage 8: Mixdown."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from vaanidub.exceptions import StageError
from vaanidub.pipeline.context import PipelineContext, Segment
from vaanidub.pipeline.stages.s8_mixdown import MixdownStage


class TestMixdownStageMetadata:
    def test_stage_metadata(self):
        stage = MixdownStage()
        assert stage.name == "mixdown"
        assert stage.stage_number == 8


class TestMixdownExecute:
    @pytest.mark.asyncio
    async def test_rejects_no_segments(self, sample_job_dir):
        stage = MixdownStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.segments = []
        ctx.background_path = Path("/some/bg.wav")
        with pytest.raises(StageError, match="No segments"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_rejects_no_background(self, sample_job_dir):
        stage = MixdownStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.segments = [Segment(index=0, speaker_label="S0", start_time=0, end_time=1)]
        ctx.background_path = None
        with pytest.raises(StageError, match="No background audio"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_audio_output_mix(self, sample_job_dir, tmp_dir):
        """Verify a mixed audio file is produced for each target language."""
        stage = MixdownStage()

        # Create background audio
        sr = 16000
        duration = 3.0
        bg_audio = 0.1 * np.random.randn(int(sr * duration)).astype(np.float32)
        bg_path = tmp_dir / "bg.wav"
        sf.write(bg_path, bg_audio, sr)

        # Create a dubbed segment audio
        seg_audio = 0.3 * np.sin(
            2 * np.pi * 440 * np.linspace(0, 1, sr, endpoint=False)
        ).astype(np.float32)
        seg_path = tmp_dir / "seg_0000.wav"
        sf.write(seg_path, seg_audio, sr)

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"],
            input_type="audio",
        )
        ctx.background_path = bg_path
        ctx.media_metadata = {"duration": duration}
        ctx.segments = [
            Segment(
                index=0, speaker_label="S0", start_time=0.0, end_time=1.0,
                text="hello", dubbed_audio_paths={"hi": seg_path},
            ),
        ]

        ctx = await stage.execute(ctx)

        assert "hi" in ctx.final_output_paths
        output_path = ctx.final_output_paths["hi"]
        assert output_path.exists()
        assert output_path.stat().st_size > 100

    @pytest.mark.asyncio
    async def test_video_mux(self, sample_job_dir, tmp_dir):
        """Verify ffmpeg mux is called for video input."""
        stage = MixdownStage()

        sr = 16000
        duration = 2.0
        bg_audio = 0.1 * np.random.randn(int(sr * duration)).astype(np.float32)
        bg_path = tmp_dir / "bg.wav"
        sf.write(bg_path, bg_audio, sr)

        video_path = tmp_dir / "input.mp4"
        video_path.write_bytes(b"fake video content")

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"],
            input_type="video", input_file_path=video_path,
        )
        ctx.background_path = bg_path
        ctx.media_metadata = {"duration": duration}
        ctx.segments = []  # no segments, just background

        # We need segments to get past the check, add an empty one
        seg_audio = np.zeros(sr, dtype=np.float32)
        seg_path = tmp_dir / "seg_0000.wav"
        sf.write(seg_path, seg_audio, sr)
        ctx.segments = [
            Segment(
                index=0, speaker_label="S0", start_time=0.0, end_time=1.0,
                text="test", dubbed_audio_paths={"hi": seg_path},
            ),
        ]

        mock_run = MagicMock(return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ))

        with patch("subprocess.run", mock_run):
            ctx = await stage.execute(ctx)

        # Verify ffmpeg was called for muxing
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "ffmpeg" in call_args
        assert "-c:v" in call_args
        assert "copy" in call_args


class TestMixdownAssembleVocalTrack:
    def test_assemble_vocal_track(self, tmp_dir):
        """Verify segments are placed on the correct timeline positions."""
        stage = MixdownStage()
        sr = 16000

        # Create two segment audio files
        seg0 = 0.5 * np.ones(sr, dtype=np.float32)  # 1 second
        seg1 = 0.3 * np.ones(sr, dtype=np.float32)  # 1 second

        seg0_path = tmp_dir / "seg0.wav"
        seg1_path = tmp_dir / "seg1.wav"
        sf.write(seg0_path, seg0, sr)
        sf.write(seg1_path, seg1, sr)

        segments = [
            Segment(index=0, speaker_label="S0", start_time=0.0, end_time=1.0,
                    dubbed_audio_paths={"hi": seg0_path}),
            Segment(index=1, speaker_label="S0", start_time=2.0, end_time=3.0,
                    dubbed_audio_paths={"hi": seg1_path}),
        ]

        track = stage._assemble_vocal_track(segments, "hi", 4.0, sr)

        assert len(track) == 4 * sr
        # Segment 0 should be at samples 0..sr (with fade-in)
        # Gap at samples sr..2*sr should be ~0
        assert np.max(np.abs(track[sr + 100:2 * sr - 100])) < 0.01
        # Segment 1 should be at samples 2*sr..3*sr
        assert np.max(np.abs(track[2 * sr + 100:3 * sr])) > 0.1

    def test_assemble_missing_dubbed_path(self, tmp_dir):
        """Segments without dubbed audio are skipped."""
        stage = MixdownStage()
        sr = 16000

        segments = [
            Segment(index=0, speaker_label="S0", start_time=0.0, end_time=1.0,
                    dubbed_audio_paths={}),
        ]

        track = stage._assemble_vocal_track(segments, "hi", 2.0, sr)
        # Entire track should be silent
        assert np.max(np.abs(track)) == 0.0


class TestMixdownNormalizeLufs:
    def test_normalize_lufs_silent(self):
        """Silent audio should be returned unchanged."""
        stage = MixdownStage()
        audio = np.zeros(16000, dtype=np.float32)
        result = stage._normalize_lufs(audio, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_normalize_lufs_normal(self):
        """Non-silent audio should have gain applied."""
        stage = MixdownStage()
        audio = 0.001 * np.ones(16000, dtype=np.float32)  # Very quiet
        result = stage._normalize_lufs(audio, 16000)
        # Result should be louder than input
        assert np.sqrt(np.mean(result ** 2)) > np.sqrt(np.mean(audio ** 2))

    def test_normalize_lufs_no_clipping(self):
        """Normalized audio should not clip past 0.98."""
        stage = MixdownStage()
        audio = 0.5 * np.ones(16000, dtype=np.float32)
        result = stage._normalize_lufs(audio, 16000)
        assert np.max(np.abs(result)) <= 0.98 + 1e-6


class TestMixdownValidateOutput:
    @pytest.mark.asyncio
    async def test_validate_output_valid(self, tmp_dir):
        stage = MixdownStage()
        # Create a realistic output file
        out = tmp_dir / "dubbed_hi.wav"
        sf.write(out, np.random.randn(16000).astype(np.float32), 16000)

        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.final_output_paths = {"hi": out}
        assert await stage.validate_output(ctx) is True

    @pytest.mark.asyncio
    async def test_validate_output_missing(self, tmp_dir):
        stage = MixdownStage()
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.final_output_paths = {}
        assert await stage.validate_output(ctx) is False

    @pytest.mark.asyncio
    async def test_validate_output_too_small(self, tmp_dir):
        stage = MixdownStage()
        small_file = tmp_dir / "tiny.wav"
        small_file.write_bytes(b"x" * 100)

        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.final_output_paths = {"hi": small_file}
        assert await stage.validate_output(ctx) is False
