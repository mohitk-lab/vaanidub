"""Google Cloud Translation API provider — paid fallback."""

import structlog

from vaanidub.exceptions import ProviderError
from vaanidub.pipeline.providers.translation.base import TranslationProvider, TranslationResult

logger = structlog.get_logger()


class GoogleTranslateProvider(TranslationProvider):
    """Translation using Google Cloud Translation API v2."""

    name = "google_translate"
    supported_language_pairs = []  # Supports virtually all pairs

    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderError(self.name, "Google Translate API key is required")
        self.api_key = api_key
        self.base_url = "https://translation.googleapis.com/language/translate/v2"

    async def translate(
        self, text: str, source_language: str, target_language: str
    ) -> TranslationResult:
        results = await self.translate_batch([text], source_language, target_language)
        return results[0]

    async def translate_batch(
        self, texts: list[str], source_language: str, target_language: str
    ) -> list[TranslationResult]:
        import httpx

        results = []
        batch_size = 100  # Google API batch limit

        async with httpx.AsyncClient(timeout=60) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                resp = await client.post(self.base_url, json={
                    "q": batch,
                    "source": source_language,
                    "target": target_language,
                    "key": self.api_key,
                    "format": "text",
                })

                if resp.status_code != 200:
                    raise ProviderError(
                        self.name,
                        f"Google API error {resp.status_code}: {resp.text[:300]}"
                    )

                data = resp.json()
                translations = data.get("data", {}).get("translations", [])

                for j, t in enumerate(translations):
                    results.append(TranslationResult(
                        text=t["translatedText"],
                        source_language=source_language,
                        target_language=target_language,
                        confidence=0.8,
                        provider_name=self.name,
                    ))

        return results

    def supports_pair(self, source: str, target: str) -> bool:
        # Google supports essentially all language pairs
        return True

    async def health_check(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.base_url}/languages",
                    params={"key": self.api_key},
                )
                return resp.status_code == 200
        except Exception:
            return False
