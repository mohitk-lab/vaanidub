"""IndicTrans2 translation provider — AI4Bharat's Indian language translator."""

import structlog

from vaanidub.exceptions import ProviderError
from vaanidub.pipeline.providers.translation.base import TranslationProvider, TranslationResult

logger = structlog.get_logger()

# IndicTrans2 language code mapping (ISO 639-1 -> Flores-200)
INDICTRANS_CODES = {
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


class IndicTrans2Provider(TranslationProvider):
    """Translation using AI4Bharat IndicTrans2 (200M distilled)."""

    name = "indictrans2"
    supported_language_pairs = [
        (src, tgt) for src in INDICTRANS_CODES for tgt in INDICTRANS_CODES if src != tgt
    ]

    def __init__(self, device: str = "cuda", batch_size: int = 32, max_length: int = 256):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
        logger.info("loading_indictrans2", model=model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = self._model.to(self.device)
        if self.device.startswith("cuda"):
            self._model = self._model.half()

        logger.info("indictrans2_loaded")

    async def translate(
        self, text: str, source_language: str, target_language: str
    ) -> TranslationResult:
        results = await self.translate_batch([text], source_language, target_language)
        return results[0]

    async def translate_batch(
        self, texts: list[str], source_language: str, target_language: str
    ) -> list[TranslationResult]:
        self._ensure_model()

        src_code = INDICTRANS_CODES.get(source_language)
        tgt_code = INDICTRANS_CODES.get(target_language)

        if not src_code or not tgt_code:
            raise ProviderError(
                self.name,
                f"Unsupported language pair: {source_language} -> {target_language}"
            )

        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Tag each input with target language
            tagged = [f"<2{tgt_code}> {t}" for t in batch]

            inputs = self._tokenizer(
                tagged,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            outputs = self._model.generate(**inputs, max_length=self.max_length, num_beams=5)
            decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for original, translated in zip(batch, decoded):
                results.append(TranslationResult(
                    text=translated.strip(),
                    source_language=source_language,
                    target_language=target_language,
                    confidence=0.85,  # IndicTrans2 doesn't expose confidence
                    provider_name=self.name,
                ))

        return results

    def supports_pair(self, source: str, target: str) -> bool:
        return source in INDICTRANS_CODES and target in INDICTRANS_CODES

    async def health_check(self) -> bool:
        try:
            self._ensure_model()
            return self._model is not None
        except Exception:
            return False
