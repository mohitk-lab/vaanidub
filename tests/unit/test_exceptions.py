"""Tests for exception hierarchy."""

from vaanidub.exceptions import (
    AllProvidersFailed,
    MediaValidationError,
    PipelineError,
    ProviderError,
    StageError,
    StageValidationError,
    UnsupportedLanguageError,
    VaaniDubError,
)


class TestExceptionHierarchy:
    def test_base_exception(self):
        e = VaaniDubError("test error", details={"key": "val"})
        assert str(e) == "test error"
        assert e.message == "test error"
        assert e.details == {"key": "val"}

    def test_pipeline_error_inherits(self):
        assert issubclass(PipelineError, VaaniDubError)

    def test_stage_error(self):
        e = StageError("translate", "Model not found", retriable=False)
        assert "translate" in str(e)
        assert e.stage_name == "translate"
        assert e.retriable is False

    def test_stage_validation_error(self):
        e = StageValidationError("synthesize")
        assert e.stage_name == "synthesize"
        assert e.retriable is True

    def test_provider_error(self):
        e = ProviderError("indicf5", "GPU OOM")
        assert "indicf5" in str(e)
        assert e.provider_name == "indicf5"

    def test_all_providers_failed(self):
        inner = ValueError("connection timeout")
        e = AllProvidersFailed("tts", inner)
        assert "tts" in str(e)
        assert e.provider_type == "tts"
        assert e.last_error is inner

    def test_unsupported_language(self):
        e = UnsupportedLanguageError("xx")
        assert "xx" in str(e)
        assert e.language_code == "xx"

    def test_media_validation(self):
        e = MediaValidationError("corrupt file")
        assert isinstance(e, VaaniDubError)
