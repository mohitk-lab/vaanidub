"""Tests for Stage 5: Prosody Analysis."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vaanidub.exceptions import StageError
from vaanidub.pipeline.context import PipelineContext, Segment
from vaanidub.pipeline.stages.s5_prosody import ProsodyStage


class TestProsodyStageMetadata:
    def test_stage_metadata(self):
        stage = ProsodyStage()
        assert stage.name == "prosody"
        assert stage.stage_number == 5


class TestProsodyExecute:
    @pytest.mark.asyncio
    async def test_rejects_missing_vocals(self, sample_job_dir):
        stage = ProsodyStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = None
        ctx.segments = [Segment(index=0, speaker_label="S0", start_time=0, end_time=1)]
        with pytest.raises(StageError, match="No vocals or segments"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_rejects_no_segments(self, sample_job_dir, sample_audio_path):
        stage = ProsodyStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.segments = []
        with pytest.raises(StageError, match="No vocals or segments"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_energy_calculation(self, sample_job_dir, sample_audio_path):
        """Verify RMS energy is computed for segments."""
        stage = ProsodyStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0.0, end_time=2.0,
                    text="hello world test phrase"),
        ]

        # Mock parselmouth and librosa since they may not be installed
        mock_snd = MagicMock()
        mock_pitch_obj = MagicMock()
        mock_pitch_obj.selected_array = {"frequency": np.array([200.0, 210.0, 190.0])}
        mock_snd.to_pitch.return_value = mock_pitch_obj

        mock_parselmouth = MagicMock()
        mock_parselmouth.Sound.return_value = mock_snd

        with patch.dict("sys.modules", {"parselmouth": mock_parselmouth, "librosa": MagicMock()}):
            ctx = await stage.execute(ctx)

        assert ctx.segments[0].energy > 0.0

    @pytest.mark.asyncio
    async def test_speaking_rate_calculation(self, sample_job_dir, sample_audio_path):
        """Verify speaking rate (words/sec) is computed."""
        stage = ProsodyStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0.0, end_time=2.0,
                    text="one two three four five six"),
        ]

        mock_snd = MagicMock()
        mock_pitch = MagicMock()
        mock_pitch.selected_array = {"frequency": np.array([200.0])}
        mock_snd.to_pitch.return_value = mock_pitch

        mock_parselmouth = MagicMock()
        mock_parselmouth.Sound.return_value = mock_snd

        with patch.dict("sys.modules", {"parselmouth": mock_parselmouth, "librosa": MagicMock()}):
            ctx = await stage.execute(ctx)

        # 6 words / 2.0 seconds = 3.0 words/sec
        assert ctx.segments[0].speaking_rate == pytest.approx(3.0, rel=0.1)

    @pytest.mark.asyncio
    async def test_skips_short_segments(self, sample_job_dir, sample_audio_path):
        """Segments < 100ms should be skipped (no prosody data)."""
        stage = ProsodyStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0.0, end_time=0.05,
                    text="hi"),
        ]

        mock_parselmouth = MagicMock()
        with patch.dict("sys.modules", {"parselmouth": mock_parselmouth, "librosa": MagicMock()}):
            ctx = await stage.execute(ctx)

        # Should remain at default values since segment was skipped
        assert ctx.segments[0].energy == 0.0
        assert ctx.segments[0].avg_pitch == 0.0

    @pytest.mark.asyncio
    async def test_pitch_extraction_error_handled(self, sample_job_dir, sample_audio_path):
        """If parselmouth raises, pitch should default to 0.0."""
        stage = ProsodyStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0.0, end_time=2.0,
                    text="test words here"),
        ]

        mock_parselmouth = MagicMock()
        mock_parselmouth.Sound.side_effect = RuntimeError("Praat error")

        with patch.dict("sys.modules", {"parselmouth": mock_parselmouth, "librosa": MagicMock()}):
            ctx = await stage.execute(ctx)

        assert ctx.segments[0].avg_pitch == 0.0
        # Energy should still be computed since it doesn't use parselmouth
        assert ctx.segments[0].energy > 0.0


class TestProsodyEmotionClassification:
    def test_emotion_neutral_zeros(self):
        stage = ProsodyStage()
        assert stage._classify_emotion(0, 0, 0) == "neutral"

    def test_emotion_excited(self):
        stage = ProsodyStage()
        assert stage._classify_emotion(300, 0.08, 4.0) == "excited"

    def test_emotion_happy(self):
        stage = ProsodyStage()
        assert stage._classify_emotion(220, 0.04, 2.0) == "happy"

    def test_emotion_sad(self):
        stage = ProsodyStage()
        assert stage._classify_emotion(100, 0.01, 1.5) == "sad"

    def test_emotion_angry(self):
        stage = ProsodyStage()
        assert stage._classify_emotion(180, 0.08, 4.0) == "angry"

    def test_emotion_neutral_moderate(self):
        stage = ProsodyStage()
        assert stage._classify_emotion(180, 0.03, 2.5) == "neutral"


class TestProsodyValidateOutput:
    @pytest.mark.asyncio
    async def test_validate_output_always_true(self, tmp_dir):
        """Prosody validation is best-effort, always returns True."""
        stage = ProsodyStage()
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1)
        ]
        assert await stage.validate_output(ctx) is True

    @pytest.mark.asyncio
    async def test_validate_output_empty_segments(self, tmp_dir):
        stage = ProsodyStage()
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.segments = []
        # Still returns True — prosody is nice-to-have
        assert await stage.validate_output(ctx) is True
