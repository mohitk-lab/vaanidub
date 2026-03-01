"""Tests for provider base classes."""

from pathlib import Path

from vaanidub.pipeline.providers.stt.base import TranscriptionSegment
from vaanidub.pipeline.providers.translation.base import TranslationResult
from vaanidub.pipeline.providers.tts.base import TTSRequest, TTSResult


class TestTranscriptionSegment:
    def test_creation(self):
        seg = TranscriptionSegment(start=1.0, end=2.5, text="hello world")
        assert seg.start == 1.0
        assert seg.end == 2.5
        assert seg.text == "hello world"
        assert seg.speaker == ""
        assert seg.words == []

    def test_with_words(self):
        seg = TranscriptionSegment(
            start=0, end=1,
            text="hi there",
            words=[{"word": "hi", "start": 0, "end": 0.4}],
        )
        assert len(seg.words) == 1


class TestTranslationResult:
    def test_creation(self):
        r = TranslationResult(
            text="नमस्ते",
            source_language="en",
            target_language="hi",
            confidence=0.9,
            provider_name="indictrans2",
        )
        assert r.text == "नमस्ते"
        assert r.source_language == "en"
        assert r.target_language == "hi"


class TestTTSRequest:
    def test_creation(self):
        req = TTSRequest(
            text="नमस्ते दुनिया",
            target_language="hi",
            reference_audio_path=Path("/tmp/ref.wav"),
            reference_text="Hello world",
            target_duration_sec=2.5,
        )
        assert req.text == "नमस्ते दुनिया"
        assert req.target_language == "hi"
        assert req.speaking_rate == 1.0
        assert req.emotion_hint == "neutral"


class TestTTSResult:
    def test_creation(self):
        r = TTSResult(
            audio_path=Path("/tmp/out.wav"),
            actual_duration_sec=2.3,
            sample_rate=24000,
            provider_name="indicf5",
        )
        assert r.sample_rate == 24000
        assert r.provider_name == "indicf5"
