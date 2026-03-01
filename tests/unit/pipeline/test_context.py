"""Tests for PipelineContext."""

from vaanidub.pipeline.context import PipelineContext, Segment, SpeakerInfo


class TestPipelineContext:
    def test_create_context(self, tmp_dir):
        ctx = PipelineContext(
            job_id="test-123",
            job_dir=tmp_dir,
            target_languages=["hi", "ta"],
        )
        assert ctx.job_id == "test-123"
        assert ctx.target_languages == ["hi", "ta"]
        assert ctx.current_stage == 0
        assert ctx.segments == []
        assert ctx.speakers == {}

    def test_stage_dir_creates_directory(self, tmp_dir):
        ctx = PipelineContext(job_id="test", job_dir=tmp_dir, target_languages=["hi"])
        stage_dir = ctx.stage_dir("ingest")
        assert stage_dir.exists()
        assert stage_dir == tmp_dir / "ingest"

    def test_report_progress_with_callback(self, tmp_dir):
        calls = []

        def on_progress(stage, pct, msg):
            calls.append((stage, pct, msg))

        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"],
            on_progress=on_progress,
        )
        ctx.report_progress("translate", 50, "halfway")
        assert len(calls) == 1
        assert calls[0] == ("translate", 50, "halfway")

    def test_report_progress_without_callback(self, tmp_dir):
        ctx = PipelineContext(job_id="test", job_dir=tmp_dir, target_languages=["hi"])
        # Should not raise
        ctx.report_progress("translate", 50, "test")


class TestSegment:
    def test_segment_duration(self):
        seg = Segment(index=0, speaker_label="SPEAKER_00", start_time=1.5, end_time=4.0)
        assert seg.duration == 2.5

    def test_segment_defaults(self):
        seg = Segment(index=0, speaker_label="SPEAKER_00", start_time=0, end_time=1)
        assert seg.text == ""
        assert seg.emotion == "neutral"
        assert seg.translations == {}
        assert seg.dubbed_audio_paths == {}


class TestSpeakerInfo:
    def test_speaker_info(self):
        info = SpeakerInfo(
            label="SPEAKER_00",
            segments=[{"start": 0, "end": 5, "duration": 5}],
            total_duration_sec=5.0,
        )
        assert info.label == "SPEAKER_00"
        assert len(info.segments) == 1
        assert info.reference_clip_path is None
