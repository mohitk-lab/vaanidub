"""Tests for Stage 7: Voice Cloning TTS Synthesis."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import soundfile as sf

from vaanidub.exceptions import AllProvidersFailed, StageError
from vaanidub.pipeline.context import PipelineContext, Segment, SpeakerInfo
from vaanidub.pipeline.stages.s7_synthesize import SynthesizeStage


class TestSynthesizeStageMetadata:
    def test_stage_metadata(self):
        stage = SynthesizeStage(device="cpu")
        assert stage.name == "synthesize"
        assert stage.stage_number == 7


class TestSynthesizeExecute:
    @pytest.mark.asyncio
    async def test_rejects_no_segments(self, sample_job_dir):
        stage = SynthesizeStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.segments = []
        ctx.speakers = {"S0": SpeakerInfo(label="S0", segments=[])}
        with pytest.raises(StageError, match="No segments to synthesize"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_rejects_no_speakers(self, sample_job_dir):
        stage = SynthesizeStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.segments = [Segment(index=0, speaker_label="S0", start_time=0, end_time=1)]
        ctx.speakers = {}
        with pytest.raises(StageError, match="No speaker info"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_indicf5_success(self, sample_job_dir, sample_audio_path):
        """Mock IndicF5 model to verify audio files are written."""
        stage = SynthesizeStage(device="cpu")

        # Create a reference clip
        ref_clip = sample_audio_path

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.segments = [
            Segment(index=0, speaker_label="SPEAKER_00", start_time=0.0, end_time=1.0,
                    text="Hello", translations={"hi": "नमस्ते"}),
        ]
        ctx.speakers = {
            "SPEAKER_00": SpeakerInfo(
                label="SPEAKER_00",
                segments=[{"start": 0.0, "end": 1.0, "duration": 1.0}],
                reference_clip_path=ref_clip,
            )
        }

        # Mock _synthesize_indicf5 to write a real audio file
        async def mock_synth(text, ref_audio_path, ref_text, output_path):
            audio = 0.3 * np.random.randn(24000).astype(np.float32)
            sf.write(output_path, audio, 24000)

        with patch.object(stage, "_synthesize_indicf5", side_effect=mock_synth):
            ctx = await stage.execute(ctx)

        assert "hi" in ctx.segments[0].dubbed_audio_paths
        assert ctx.segments[0].dubbed_audio_paths["hi"].exists()

    @pytest.mark.asyncio
    async def test_indicf5_fail_elevenlabs_fallback(self, sample_job_dir, sample_audio_path):
        """When IndicF5 fails, ElevenLabs should be used."""
        stage = SynthesizeStage(device="cpu", elevenlabs_api_key="test-key")

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.segments = [
            Segment(index=0, speaker_label="SPEAKER_00", start_time=0.0, end_time=1.0,
                    text="Hello", translations={"hi": "नमस्ते"}),
        ]
        ctx.speakers = {
            "SPEAKER_00": SpeakerInfo(
                label="SPEAKER_00",
                segments=[{"start": 0.0, "end": 1.0, "duration": 1.0}],
                reference_clip_path=sample_audio_path,
            )
        }

        async def mock_indicf5_fail(*args, **kwargs):
            raise RuntimeError("Model failed")

        async def mock_elevenlabs(text, ref_audio_path, target_lang, output_path):
            audio = 0.3 * np.random.randn(24000).astype(np.float32)
            sf.write(output_path, audio, 24000)

        with patch.object(stage, "_synthesize_indicf5", side_effect=mock_indicf5_fail), \
             patch.object(stage, "_synthesize_elevenlabs", side_effect=mock_elevenlabs):
            ctx = await stage.execute(ctx)

        assert ctx.segments[0].dubbed_audio_paths["hi"].exists()

    @pytest.mark.asyncio
    async def test_both_providers_fail(self, sample_job_dir, sample_audio_path):
        stage = SynthesizeStage(device="cpu", elevenlabs_api_key="test-key")

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.segments = [
            Segment(index=0, speaker_label="SPEAKER_00", start_time=0.0, end_time=1.0,
                    text="Hello", translations={"hi": "नमस्ते"}),
        ]
        ctx.speakers = {
            "SPEAKER_00": SpeakerInfo(
                label="SPEAKER_00",
                segments=[{"start": 0.0, "end": 1.0, "duration": 1.0}],
                reference_clip_path=sample_audio_path,
            )
        }

        async def mock_fail(*args, **kwargs):
            raise RuntimeError("Failed")

        with patch.object(stage, "_synthesize_indicf5", side_effect=mock_fail), \
             patch.object(stage, "_synthesize_elevenlabs", side_effect=mock_fail):
            with pytest.raises(AllProvidersFailed):
                await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_empty_translation_skipped(self, sample_job_dir, sample_audio_path):
        """Segments with empty translations should be skipped."""
        stage = SynthesizeStage(device="cpu")

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.vocals_path = sample_audio_path
        ctx.segments = [
            Segment(index=0, speaker_label="SPEAKER_00", start_time=0.0, end_time=1.0,
                    text="", translations={"hi": ""}),
        ]
        ctx.speakers = {
            "SPEAKER_00": SpeakerInfo(
                label="SPEAKER_00",
                segments=[{"start": 0.0, "end": 1.0, "duration": 1.0}],
                reference_clip_path=sample_audio_path,
            )
        }

        mock_synth = AsyncMock()
        with patch.object(stage, "_synthesize_indicf5", mock_synth):
            ctx = await stage.execute(ctx)

        # Should not have called synthesis for empty translation
        mock_synth.assert_not_called()


class TestSynthesizeMatchDuration:
    def test_match_duration_no_stretch_needed(self, tmp_dir):
        """Ratio ~1.0 should not trigger stretching."""
        stage = SynthesizeStage(device="cpu")
        sr = 16000
        audio = 0.3 * np.sin(
            2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False)
        ).astype(np.float32)
        path = tmp_dir / "test.wav"
        sf.write(path, audio, sr)

        # Target duration is close to current (1.0 sec)
        stage._match_duration(path, 1.02)

        # Should be unchanged (no stretch)
        result_audio, result_sr = sf.read(path)
        assert len(result_audio) == len(audio)

    def test_match_duration_stretch(self, tmp_dir):
        """When ratio > threshold, time stretch should be applied."""
        stage = SynthesizeStage(device="cpu")
        sr = 16000
        audio = 0.3 * np.ones(sr, dtype=np.float32)  # 1 second
        path = tmp_dir / "test.wav"
        sf.write(path, audio, sr)

        # Target 2.0 seconds — ratio = 1.0/2.0 = 0.5
        # This triggers scipy fallback (pyrubberband not installed)
        stage._match_duration(path, 2.0)

        result_audio, _ = sf.read(path)
        # Should be stretched (longer than original)
        assert len(result_audio) > sr


class TestSynthesizeGetReferenceText:
    def test_get_reference_text(self):
        """Should return text of the transcript segment matching longest speaker segment."""
        stage = SynthesizeStage(device="cpu")

        ctx = PipelineContext(
            job_id="test", job_dir=Path("/tmp"), target_languages=["hi"]
        )
        ctx.speakers = {
            "SPEAKER_00": SpeakerInfo(
                label="SPEAKER_00",
                segments=[
                    {"start": 0.0, "end": 2.0, "duration": 2.0},
                    {"start": 5.0, "end": 10.0, "duration": 5.0},  # longest
                ],
            )
        }
        ctx.segments = [
            Segment(index=0, speaker_label="SPEAKER_00", start_time=0.0, end_time=2.0,
                    text="Short segment"),
            Segment(index=1, speaker_label="SPEAKER_00", start_time=5.0, end_time=10.0,
                    text="This is the longest segment text"),
        ]

        result = stage._get_reference_text(ctx, "SPEAKER_00")
        assert result == "This is the longest segment text"

    def test_get_reference_text_no_speaker(self):
        stage = SynthesizeStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=Path("/tmp"), target_languages=["hi"]
        )
        ctx.speakers = {}
        ctx.segments = []

        result = stage._get_reference_text(ctx, "UNKNOWN")
        assert result == ""

    def test_get_reference_text_no_matching_transcript(self):
        stage = SynthesizeStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=Path("/tmp"), target_languages=["hi"]
        )
        ctx.speakers = {
            "SPEAKER_00": SpeakerInfo(
                label="SPEAKER_00",
                segments=[{"start": 100.0, "end": 105.0, "duration": 5.0}],
            )
        }
        ctx.segments = [
            Segment(index=0, speaker_label="SPEAKER_00", start_time=0.0, end_time=2.0,
                    text="no match"),
        ]

        result = stage._get_reference_text(ctx, "SPEAKER_00")
        assert result == ""


class TestSynthesizeValidateOutput:
    @pytest.mark.asyncio
    async def test_validate_output_valid(self, tmp_dir):
        stage = SynthesizeStage(device="cpu")
        dubbed = tmp_dir / "seg_0000.wav"
        sf.write(dubbed, np.random.randn(16000).astype(np.float32), 16000)

        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1,
                    dubbed_audio_paths={"hi": dubbed}),
        ]
        assert await stage.validate_output(ctx) is True

    @pytest.mark.asyncio
    async def test_validate_output_no_dubbed(self, tmp_dir):
        stage = SynthesizeStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1,
                    dubbed_audio_paths={}),
        ]
        assert await stage.validate_output(ctx) is False
