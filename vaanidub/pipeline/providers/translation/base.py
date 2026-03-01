"""Abstract base for translation providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """Result of a translation."""
    text: str
    source_language: str
    target_language: str
    confidence: float
    provider_name: str


class TranslationProvider(ABC):
    """Abstract interface for translation providers."""

    name: str
    supported_language_pairs: list[tuple[str, str]]

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Translate text from source to target language."""
        ...

    @abstractmethod
    async def translate_batch(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[TranslationResult]:
        """Translate a batch of texts."""
        ...

    @abstractmethod
    def supports_pair(self, source: str, target: str) -> bool:
        """Check if this provider supports the given language pair."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...
