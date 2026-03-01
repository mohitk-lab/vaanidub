"""Tests for Stage 4: Speech-to-Text Transcription."""

from unittest.mock import MagicMock, patch

import pytest

from vaanidub.exceptions import StageError
from vaanidub.pipeline.context import PipelineContext, SpeakerInfo
from vaanidub.pipeline.stages.s4_transcribe import TranscribeStage


class TestTranscribeStageMetadata:
    def test_stage_metadata(self):
        stage = TranscribeStage(device="cpu")
        assert stage.name == "transcribe"
        assert stage.stage_number == 4


class TestTranscribeExecute:
    def _make_mock_whisperx(self, segments=None, language="en", probability=0.95):
        """Create a mock whisperx module."""
        mock_wx = MagicMock()

        # Mock load_model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": language,
            "language_probability": probability,
            "segments": segments or [
                {"start": 0.0, "end": 2.0, "text": "Hello world", "speaker": "SPEAKER_00"},
                {"start": 2.5, "end": 4.0, "text": "How are you", "speaker": "SPEAKER_00"},
            ],
        }
        mock_wx.load_model.return_value = mock_model
        mock_wx.load_audio.return_value = MagicMock()  # numpy array

        # Mock alignment — just pass through the segments
        mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_wx.align.return_value = {
            "segments": segments or [
                {"start": 0.0, "end": 2.0, "text": "Hello world", "speaker": "SPEAKER_00"},
                {"start": 2.5, "end": 4.0, "text": "How are you", "speaker": "SPEAKER_00"},
            ],
        }
        mock_wx.assign_word_speakers.return_value = {
            "segments": segments or [
                {"start": 0.0, "end": 2.0, "text": "Hello world", "speaker": "SPEAKER_00"},
                {"start": 2.5, "end": 4.0, "text": "How are you", "speaker": "SPEAKER_00"},
            ],
        }

        return mock_wx

    @pytest.mark.asyncio
    async def test_rejects_missing_vocals(self, sample_job_dir):
        stage = TranscribeStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = None
        with pytest.raises(StageError, match="No vocals audio"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_transcription_populates_segments(self, sample_job_dir, sample_audio_path):
        """Verify segments are created from whisperx output."""
        stage = TranscribeStage(device="cpu")
        mock_wx = self._make_mock_whisperx()

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path

        with patch.dict("sys.modules", {"whisperx": mock_wx}):
            # Need to reset the cached model
            stage._model = None
            ctx = await stage.execute(ctx)

        assert len(ctx.segments) == 2
        assert ctx.segments[0].text == "Hello world"
        assert ctx.segments[1].text == "How are you"
        assert ctx.segments[0].start_time == 0.0
        assert ctx.segments[0].end_time == 2.0

    @pytest.mark.asyncio
    async def test_language_detection(self, sample_job_dir, sample_audio_path):
        """Verify source language is detected."""
        stage = TranscribeStage(device="cpu")
        mock_wx = self._make_mock_whisperx(language="hi", probability=0.88)

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["ta"]
        )
        ctx.vocals_path = sample_audio_path

        with patch.dict("sys.modules", {"whisperx": mock_wx}):
            stage._model = None
            ctx = await stage.execute(ctx)

        assert ctx.source_language == "hi"
        assert ctx.source_language_confidence == 0.88

    @pytest.mark.asyncio
    async def test_speaker_assignment_with_diarization(self, sample_job_dir, sample_audio_path):
        """When speakers are pre-populated, whisperx.assign_word_speakers should be called."""
        stage = TranscribeStage(device="cpu")
        mock_wx = self._make_mock_whisperx()

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.speakers = {
            "SPEAKER_00": SpeakerInfo(
                label="SPEAKER_00",
                segments=[{"start": 0.0, "end": 4.0, "duration": 4.0}],
            )
        }

        # We need pandas for _build_diarize_segments
        mock_pd = MagicMock()
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.to_dict.return_value = [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}
        ]
        mock_pd.DataFrame.return_value = mock_df

        with patch.dict("sys.modules", {"whisperx": mock_wx, "pandas": mock_pd}):
            stage._model = None
            ctx = await stage.execute(ctx)

        mock_wx.assign_word_speakers.assert_called_once()

    @pytest.mark.asyncio
    async def test_speaker_assignment_without_diarization(self, sample_job_dir, sample_audio_path):
        """When no speakers, assignment should be skipped."""
        stage = TranscribeStage(device="cpu")
        mock_wx = self._make_mock_whisperx()

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.speakers = {}  # No diarization

        with patch.dict("sys.modules", {"whisperx": mock_wx}):
            stage._model = None
            ctx = await stage.execute(ctx)

        mock_wx.assign_word_speakers.assert_not_called()


class TestTranscribeValidateOutput:
    @pytest.mark.asyncio
    async def test_validate_output_valid(self, tmp_dir):
        from vaanidub.pipeline.context import Segment
        stage = TranscribeStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.source_language = "en"
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1, text="Hello"),
        ]
        assert await stage.validate_output(ctx) is True

    @pytest.mark.asyncio
    async def test_validate_output_no_segments(self, tmp_dir):
        stage = TranscribeStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.source_language = "en"
        ctx.segments = []
        assert await stage.validate_output(ctx) is False

    @pytest.mark.asyncio
    async def test_validate_output_no_language(self, tmp_dir):
        from vaanidub.pipeline.context import Segment
        stage = TranscribeStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.source_language = None
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1, text="Hello"),
        ]
        assert await stage.validate_output(ctx) is False

    @pytest.mark.asyncio
    async def test_validate_output_all_empty_text(self, tmp_dir):
        from vaanidub.pipeline.context import Segment
        stage = TranscribeStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.source_language = "en"
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1, text=""),
            Segment(index=1, speaker_label="S0", start_time=1, end_time=2, text=""),
        ]
        assert await stage.validate_output(ctx) is False
