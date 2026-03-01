"""Exception hierarchy for VaaniDub."""


class VaaniDubError(Exception):
    """Base exception for all VaaniDub errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class PipelineError(VaaniDubError):
    """Error during pipeline execution."""
    pass


class StageError(PipelineError):
    """Error within a specific pipeline stage."""

    def __init__(self, stage_name: str, message: str, retriable: bool = True, **kwargs):
        super().__init__(f"[{stage_name}] {message}", **kwargs)
        self.stage_name = stage_name
        self.retriable = retriable


class StageValidationError(StageError):
    """Stage output did not pass validation."""

    def __init__(self, stage_name: str, message: str = "Output validation failed"):
        super().__init__(stage_name, message, retriable=True)


class ProviderError(VaaniDubError):
    """Error from a specific provider (STT, TTS, translation)."""

    def __init__(self, provider_name: str, message: str, **kwargs):
        super().__init__(f"[{provider_name}] {message}", **kwargs)
        self.provider_name = provider_name


class AllProvidersFailed(PipelineError):  # noqa: N818
    """All providers in a fallback chain failed."""

    def __init__(self, provider_type: str, last_error: Exception | None = None):
        msg = f"All {provider_type} providers failed"
        if last_error:
            msg += f". Last error: {last_error}"
        super().__init__(msg)
        self.provider_type = provider_type
        self.last_error = last_error


class MediaValidationError(VaaniDubError):
    """Input media file is invalid or unsupported."""
    pass


class GPUMemoryError(VaaniDubError):
    """Insufficient GPU memory."""
    pass


class ModelNotLoadedError(VaaniDubError):
    """A required AI model is not loaded or available."""
    pass


class UnsupportedLanguageError(VaaniDubError):
    """Requested language is not supported."""

    def __init__(self, language_code: str):
        super().__init__(f"Language '{language_code}' is not supported")
        self.language_code = language_code
