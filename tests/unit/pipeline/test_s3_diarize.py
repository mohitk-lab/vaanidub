"""Tests for Stage 3: Speaker Diarization."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from vaanidub.exceptions import StageError
from vaanidub.pipeline.context import PipelineContext, SpeakerInfo
from vaanidub.pipeline.stages.s3_diarize import DiarizeStage


@dataclass
class MockTurn:
    """Mimics pyannote Segment/Turn with start/end."""
    start: float
    end: float


class MockDiarization:
    """Mimics pyannote Annotation object."""
    def __init__(self, tracks: list[tuple[MockTurn, str, str]]):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        for turn, _, speaker in self._tracks:
            yield turn, None, speaker


class TestDiarizeStageMetadata:
    def test_stage_metadata(self):
        stage = DiarizeStage()
        assert stage.name == "diarize"
        assert stage.stage_number == 3


class TestDiarizeExecute:
    @pytest.mark.asyncio
    async def test_rejects_missing_vocals(self, sample_job_dir):
        stage = DiarizeStage()
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = None
        with pytest.raises(StageError, match="No vocals audio"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_speaker_info_populated(self, sample_job_dir, sample_audio_path):
        """Verify speakers dict is populated with correct SpeakerInfo."""
        stage = DiarizeStage()

        # Create mock diarization result with 2 speakers
        diarization = MockDiarization([
            (MockTurn(0.0, 5.5), None, "SPEAKER_00"),
            (MockTurn(6.0, 8.0), None, "SPEAKER_01"),
            (MockTurn(8.5, 12.0), None, "SPEAKER_00"),
        ])

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = diarization

        stage._pipeline = mock_pipeline

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path

        ctx = await stage.execute(ctx)

        assert "SPEAKER_00" in ctx.speakers
        assert "SPEAKER_01" in ctx.speakers
        assert len(ctx.speakers) == 2

        spk0 = ctx.speakers["SPEAKER_00"]
        assert spk0.label == "SPEAKER_00"
        assert len(spk0.segments) == 2

    @pytest.mark.asyncio
    async def test_raises_on_no_speakers(self, sample_job_dir, sample_audio_path):
        """Empty diarization result should raise StageError."""
        stage = DiarizeStage()

        diarization = MockDiarization([])
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = diarization
        stage._pipeline = mock_pipeline

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path

        with pytest.raises(StageError, match="No speakers detected"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_reference_clips_extracted(self, sample_job_dir, sample_audio_path):
        """Each speaker should have a reference clip path."""
        stage = DiarizeStage()

        diarization = MockDiarization([
            (MockTurn(0.0, 2.5), None, "SPEAKER_00"),
        ])

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = diarization
        stage._pipeline = mock_pipeline

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path

        ctx = await stage.execute(ctx)

        spk = ctx.speakers["SPEAKER_00"]
        assert spk.reference_clip_path is not None
        assert spk.reference_clip_path.exists()


class TestDiarizeExtractReferenceClip:
    def test_extract_reference_clip_longest(self, tmp_dir, sample_audio_path):
        """Should pick the longest segment as reference clip."""
        stage = DiarizeStage()
        audio, sr = sf.read(sample_audio_path)

        segments = [
            {"start": 0.0, "end": 1.0, "duration": 1.0},   # short
            {"start": 1.0, "end": 2.5, "duration": 1.5},   # longer
        ]

        result = stage._extract_reference_clip(audio, sr, segments, "SPK", tmp_dir)
        assert result is not None
        assert result.exists()
        # Should have created ref_SPK.wav
        assert result.name == "ref_SPK.wav"

    def test_extract_reference_clip_skips_quiet(self, tmp_dir):
        """Segments with very low RMS should be skipped; the non-silent one is chosen."""
        stage = DiarizeStage()
        sr = 16000
        # Create audio that is silent for first 6s, then has content for 10s
        silence = np.zeros(int(6 * sr), dtype=np.float32)
        tone = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 10, int(10 * sr)))
        audio = np.concatenate([silence, tone]).astype(np.float32)

        segments = [
            {"start": 0.0, "end": 6.0, "duration": 6.0},   # silent — should be skipped
            {"start": 6.0, "end": 16.0, "duration": 10.0},  # has audio, long enough
        ]

        result = stage._extract_reference_clip(audio, sr, segments, "SPK", tmp_dir)
        assert result is not None
        # The first segment (longest sorted by duration=10.0 vs 6.0) is the non-silent one
        # since sorted by descending duration, 10s segment is tried first
        clip_audio, _ = sf.read(result)
        rms = np.sqrt(np.mean(clip_audio ** 2))
        assert rms > 1e-4

    def test_extract_reference_clip_fallback_short(self, tmp_dir, sample_audio_path):
        """When all segments are shorter than MIN_REFERENCE_CLIP_SEC, use longest anyway."""
        stage = DiarizeStage()
        audio, sr = sf.read(sample_audio_path)

        segments = [
            {"start": 0.0, "end": 0.5, "duration": 0.5},   # very short
        ]

        result = stage._extract_reference_clip(audio, sr, segments, "SPK", tmp_dir)
        # Should still create a reference clip (fallback)
        assert result is not None
        assert result.exists()


class TestDiarizeValidateOutput:
    @pytest.mark.asyncio
    async def test_validate_output_valid(self, tmp_dir):
        stage = DiarizeStage()
        ref = tmp_dir / "ref.wav"
        sf.write(ref, np.zeros(1000, dtype=np.float32), 16000)

        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.speakers = {
            "SPEAKER_00": SpeakerInfo(
                label="SPEAKER_00", segments=[], reference_clip_path=ref
            )
        }
        assert await stage.validate_output(ctx) is True

    @pytest.mark.asyncio
    async def test_validate_output_no_speakers(self, tmp_dir):
        stage = DiarizeStage()
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.speakers = {}
        assert await stage.validate_output(ctx) is False

    @pytest.mark.asyncio
    async def test_validate_output_no_reference_clips(self, tmp_dir):
        stage = DiarizeStage()
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.speakers = {
            "SPEAKER_00": SpeakerInfo(
                label="SPEAKER_00", segments=[], reference_clip_path=None
            )
        }
        assert await stage.validate_output(ctx) is False
