"""Stage 6: Translation — translate segments using IndicTrans2 with Google fallback."""

import structlog

from vaanidub.constants import LANGUAGES
from vaanidub.exceptions import AllProvidersFailed, StageError
from vaanidub.pipeline.base import PipelineStage
from vaanidub.pipeline.context import PipelineContext

logger = structlog.get_logger()


class TranslateStage(PipelineStage):
    name = "translate"
    stage_number = 6
    timeout_seconds = 300

    def __init__(self, device: str = "cuda", google_api_key: str = ""):
        self.device = device
        self.google_api_key = google_api_key
        self._indictrans_model = None
        self._indictrans_tokenizer = None

    def _load_indictrans(self):
        """Lazy-load IndicTrans2 model."""
        if self._indictrans_model is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
        self._indictrans_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._indictrans_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._indictrans_model = self._indictrans_model.to(self.device)
        if self.device.startswith("cuda"):
            self._indictrans_model = self._indictrans_model.half()

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.segments:
            raise StageError(self.name, "No segments to translate")
        if not ctx.target_languages:
            raise StageError(self.name, "No target languages specified")

        ctx.report_progress("translate", 5, "Starting translation")

        # Collect texts to translate
        texts = [seg.text for seg in ctx.segments if seg.text]

        for lang_idx, target_lang in enumerate(ctx.target_languages):
            lang_info = LANGUAGES.get(target_lang)
            if lang_info is None:
                logger.warning("unsupported_target_language", lang=target_lang)
                continue

            lang_label = lang_info.name
            ctx.report_progress(
                "translate",
                5 + int(90 * lang_idx / len(ctx.target_languages)),
                f"Translating to {lang_label}",
            )

            try:
                translated = await self._translate_with_indictrans(
                    texts, ctx.source_language or "en", lang_info.indictrans2_code
                )
            except Exception as e:
                logger.warning("indictrans_failed", error=str(e), target=target_lang)
                try:
                    translated = await self._translate_with_google(
                        texts, ctx.source_language or "en", target_lang
                    )
                except Exception as e2:
                    raise AllProvidersFailed("translation", e2) from e2

            # Assign translations to segments
            text_idx = 0
            for seg in ctx.segments:
                if seg.text:
                    seg.translations[target_lang] = translated[text_idx]
                    text_idx += 1
                else:
                    seg.translations[target_lang] = ""

        ctx.report_progress("translate", 100, "Translation complete")
        logger.info(
            "translation_complete",
            languages=ctx.target_languages,
            segments=len(ctx.segments),
        )
        return ctx

    async def validate_output(self, ctx: PipelineContext) -> bool:
        for target_lang in ctx.target_languages:
            translated_count = sum(
                1 for s in ctx.segments if s.translations.get(target_lang)
            )
            if translated_count == 0:
                logger.error("no_translations", target=target_lang)
                return False
        return True

    async def _translate_with_indictrans(
        self, texts: list[str], source_lang: str, target_indic_code: str
    ) -> list[str]:
        """Translate using IndicTrans2."""
        self._load_indictrans()

        # IndicTrans2 expects specific source language codes
        self._map_to_indictrans_code(source_lang)

        translated = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Prepare inputs with language tags
            tagged_batch = [
                f"<2{target_indic_code}> {text}" for text in batch
            ]
            inputs = self._indictrans_tokenizer(
                tagged_batch, return_tensors="pt", padding=True, truncation=True,
                max_length=256,
            ).to(self.device)

            outputs = self._indictrans_model.generate(
                **inputs, max_length=256, num_beams=5
            )
            decoded = self._indictrans_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            translated.extend(decoded)

        return translated

    async def _translate_with_google(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """Fallback: translate using Google Cloud Translation API."""
        if not self.google_api_key:
            raise StageError(self.name, "Google Translate API key not configured")

        import httpx

        url = "https://translation.googleapis.com/language/translate/v2"
        translated = []

        # Google API accepts batches of up to 128 texts
        batch_size = 100
        async with httpx.AsyncClient(timeout=60) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = await client.post(url, json={
                    "q": batch,
                    "source": source_lang,
                    "target": target_lang,
                    "key": self.google_api_key,
                    "format": "text",
                })
                resp.raise_for_status()
                data = resp.json()
                for t in data["data"]["translations"]:
                    translated.append(t["translatedText"])

        return translated

    def _map_to_indictrans_code(self, lang_code: str) -> str:
        """Map ISO 639-1 code to IndicTrans2 source code."""
        mapping = {
            "en": "eng_Latn",
            "hi": "hin_Deva",
            "ta": "tam_Taml",
            "te": "tel_Telu",
            "bn": "ben_Beng",
            "mr": "mar_Deva",
            "kn": "kan_Knda",
            "ml": "mal_Mlym",
            "gu": "guj_Gujr",
            "as": "asm_Beng",
            "or": "ory_Orya",
            "pa": "pan_Guru",
        }
        return mapping.get(lang_code, "eng_Latn")
