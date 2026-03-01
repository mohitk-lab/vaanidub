"""Tests for constants and language definitions."""

from vaanidub.constants import (
    AUDIO_FORMATS,
    LANGUAGE_CODES,
    LANGUAGES,
    STAGE_NAMES,
    SUPPORTED_FORMATS,
    VIDEO_FORMATS,
    JobStatus,
)


class TestLanguages:
    def test_all_11_languages_defined(self):
        assert len(LANGUAGES) == 11

    def test_required_languages_present(self):
        required = {"hi", "ta", "te", "bn", "mr", "kn", "ml", "gu", "as", "or", "pa"}
        assert required == LANGUAGE_CODES

    def test_language_info_complete(self):
        for code, info in LANGUAGES.items():
            assert info.code == code
            assert info.name
            assert info.native_name
            assert info.script
            assert info.indictrans2_code
            assert len(info.tts_providers) > 0

    def test_hindi_details(self):
        hi = LANGUAGES["hi"]
        assert hi.name == "Hindi"
        assert hi.native_name == "हिन्दी"
        assert hi.script == "Devanagari"
        assert hi.indictrans2_code == "hin_Deva"
        assert "indicf5" in hi.tts_providers

    def test_all_have_indicf5(self):
        for code, info in LANGUAGES.items():
            assert "indicf5" in info.tts_providers, f"{code} missing indicf5"


class TestFormats:
    def test_audio_formats(self):
        assert ".mp3" in AUDIO_FORMATS
        assert ".wav" in AUDIO_FORMATS
        assert ".flac" in AUDIO_FORMATS

    def test_video_formats(self):
        assert ".mp4" in VIDEO_FORMATS
        assert ".mkv" in VIDEO_FORMATS
        assert ".avi" in VIDEO_FORMATS

    def test_supported_is_union(self):
        assert SUPPORTED_FORMATS == AUDIO_FORMATS | VIDEO_FORMATS


class TestStages:
    def test_eight_stages(self):
        assert len(STAGE_NAMES) == 8

    def test_stage_order(self):
        assert STAGE_NAMES[0] == "ingest"
        assert STAGE_NAMES[-1] == "mixdown"
        assert "synthesize" in STAGE_NAMES


class TestJobStatus:
    def test_status_values(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
