"""Integration test — full pipeline E2E with mock providers.

This test verifies the orchestrator correctly chains all 8 stages
using mocked AI providers (no GPU required).
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vaanidub.config import AppConfig, StorageConfig
from vaanidub.pipeline.context import PipelineContext, Segment, SpeakerInfo
from vaanidub.pipeline.orchestrator import PipelineOrchestrator


@pytest.fixture
def mock_config(tmp_dir):
    """Config with tmp directories."""
    return AppConfig(
        storage=StorageConfig(
            base_path=tmp_dir / "jobs",
            temp_path=tmp_dir / "tmp",
        ),
        debug=True,
    )


@pytest.fixture
def e2e_input_audio(tmp_dir) -> Path:
    """Create a realistic-ish test audio file."""
    path = tmp_dir / "test_input.wav"
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Simulate speech-like audio with varying amplitude
    audio = np.zeros_like(t)
    # Speaker 1: 0-2s
    audio[:2*sr] = 0.3 * np.sin(2 * np.pi * 200 * t[:2*sr])
    # Silence: 2-2.5s
    # Speaker 2: 2.5-5s
    audio[int(2.5*sr):] = 0.25 * np.sin(2 * np.pi * 300 * t[:int(2.5*sr)])

    sf.write(path, audio.astype(np.float32), sr)
    return path


class TestPipelineE2E:
    """End-to-end pipeline test with mocked AI components."""

    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self, tmp_dir, mock_config, e2e_input_audio):
        """Verify all 8 stages chain correctly with mocked providers."""
        mock_config.ensure_directories()

        job_dir = mock_config.storage.base_path / "e2e-test"
        job_dir.mkdir(parents=True, exist_ok=True)

        ctx = PipelineContext(
            job_id="e2e-test",
            job_dir=job_dir,
            target_languages=["hi"],
            input_file_path=e2e_input_audio,
        )

        # We'll create the orchestrator and mock each stage individually
        # This tests the orchestrator's ability to chain stages correctly

        orchestrator = PipelineOrchestrator(mock_config)

        # Replace all stages with mock stages that set expected context
        from vaanidub.pipeline.base import PipelineStage

        class MockIngest(PipelineStage):
            name = "ingest"
            stage_number = 1

            async def execute(self, ctx):
                ctx.input_type = "audio"
                ctx.raw_audio_path = e2e_input_audio
                ctx.media_metadata = {"duration": 5.0}
                return ctx

            async def validate_output(self, ctx):
                return ctx.raw_audio_path is not None

        class MockSeparate(PipelineStage):
            name = "separate"
            stage_number = 2

            async def execute(self, ctx):
                # Create mock vocals and background files
                vocals = ctx.stage_dir("separate") / "vocals.wav"
                bg = ctx.stage_dir("separate") / "background.wav"

                audio, sr = sf.read(ctx.raw_audio_path)
                sf.write(vocals, audio, sr)
                sf.write(bg, audio * 0.1, sr)

                ctx.vocals_path = vocals
                ctx.background_path = bg
                return ctx

            async def validate_output(self, ctx):
                return ctx.vocals_path is not None

        class MockDiarize(PipelineStage):
            name = "diarize"
            stage_number = 3

            async def execute(self, ctx):
                # Create mock reference clips
                ref_dir = ctx.stage_dir("diarize")
                ref_clip = ref_dir / "ref_SPEAKER_00.wav"

                audio, sr = sf.read(ctx.vocals_path)
                sf.write(ref_clip, audio[:sr*2], sr)

                ctx.speakers = {
                    "SPEAKER_00": SpeakerInfo(
                        label="SPEAKER_00",
                        segments=[{"start": 0, "end": 2, "duration": 2}],
                        reference_clip_path=ref_clip,
                        total_duration_sec=2.0,
                    ),
                    "SPEAKER_01": SpeakerInfo(
                        label="SPEAKER_01",
                        segments=[{"start": 2.5, "end": 5, "duration": 2.5}],
                        reference_clip_path=ref_clip,
                        total_duration_sec=2.5,
                    ),
                }
                return ctx

            async def validate_output(self, ctx):
                return len(ctx.speakers) > 0

        class MockTranscribe(PipelineStage):
            name = "transcribe"
            stage_number = 4

            async def execute(self, ctx):
                ctx.source_language = "en"
                ctx.source_language_confidence = 0.95
                ctx.segments = [
                    Segment(index=0, speaker_label="SPEAKER_00",
                            start_time=0, end_time=2, text="Hello world"),
                    Segment(index=1, speaker_label="SPEAKER_01",
                            start_time=2.5, end_time=5, text="How are you"),
                ]
                return ctx

            async def validate_output(self, ctx):
                return len(ctx.segments) > 0

        class MockProsody(PipelineStage):
            name = "prosody"
            stage_number = 5

            async def execute(self, ctx):
                for seg in ctx.segments:
                    seg.avg_pitch = 200.0
                    seg.energy = 0.03
                    seg.speaking_rate = 2.5
                    seg.emotion = "neutral"
                return ctx

            async def validate_output(self, ctx):
                return True

        class MockTranslate(PipelineStage):
            name = "translate"
            stage_number = 6

            async def execute(self, ctx):
                for seg in ctx.segments:
                    seg.translations["hi"] = f"Hindi: {seg.text}"
                return ctx

            async def validate_output(self, ctx):
                return all(seg.translations.get("hi") for seg in ctx.segments)

        class MockSynthesize(PipelineStage):
            name = "synthesize"
            stage_number = 7

            async def execute(self, ctx):
                synth_dir = ctx.stage_dir("synthesize") / "hi"
                synth_dir.mkdir(parents=True, exist_ok=True)

                for seg in ctx.segments:
                    out = synth_dir / f"seg_{seg.index:04d}.wav"
                    # Create a short audio file
                    dur = seg.duration
                    audio = 0.2 * np.sin(
                        2 * np.pi * 300 * np.linspace(0, dur, int(16000 * dur))
                    )
                    sf.write(out, audio.astype(np.float32), 16000)
                    seg.dubbed_audio_paths["hi"] = out
                return ctx

            async def validate_output(self, ctx):
                return all(
                    seg.dubbed_audio_paths.get("hi")
                    for seg in ctx.segments
                )

        class MockMixdown(PipelineStage):
            name = "mixdown"
            stage_number = 8

            async def execute(self, ctx):
                out_path = ctx.stage_dir("mixdown") / "dubbed_hi.wav"
                # Just copy background as mock output
                import shutil
                shutil.copy2(ctx.background_path, out_path)
                ctx.final_output_paths["hi"] = out_path
                return ctx

            async def validate_output(self, ctx):
                return all(p.exists() for p in ctx.final_output_paths.values())

        # Replace stages
        orchestrator.stages = [
            MockIngest(),
            MockSeparate(),
            MockDiarize(),
            MockTranscribe(),
            MockProsody(),
            MockTranslate(),
            MockSynthesize(),
            MockMixdown(),
        ]

        # Run full pipeline
        result = await orchestrator.run(ctx)

        # Verify final state
        assert result.source_language == "en"
        assert len(result.segments) == 2
        assert len(result.speakers) == 2
        assert "hi" in result.final_output_paths
        assert result.final_output_paths["hi"].exists()
        assert len(result.stage_timings) == 8

        # Verify checkpoint was saved
        checkpoint = PipelineOrchestrator.load_checkpoint(job_dir)
        assert checkpoint is not None
        assert checkpoint["job_id"] == "e2e-test"
        assert checkpoint["current_stage"] == 8
