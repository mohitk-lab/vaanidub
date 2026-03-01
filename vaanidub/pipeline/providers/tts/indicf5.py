"""IndicF5 TTS provider — AI4Bharat's voice cloning for 11 Indian languages."""


import numpy as np
import soundfile as sf
import structlog

from vaanidub.exceptions import ProviderError
from vaanidub.pipeline.providers.tts.base import TTSProvider, TTSRequest, TTSResult

logger = structlog.get_logger()

INDICF5_LANGUAGES = ["hi", "ta", "te", "bn", "mr", "kn", "ml", "gu", "as", "or", "pa"]


class IndicF5Provider(TTSProvider):
    """Zero-shot voice cloning TTS using AI4Bharat IndicF5.

    Supports 11 Indian languages. Takes reference audio + text to clone voice.
    """

    name = "indicf5"
    supported_languages = INDICF5_LANGUAGES

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return

        from transformers import AutoModel

        logger.info("loading_indicf5")
        self._model = AutoModel.from_pretrained(
            "ai4bharat/IndicF5",
            trust_remote_code=True,
        )
        logger.info("indicf5_loaded")

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        self._ensure_model()

        if not request.reference_audio_path.exists():
            raise ProviderError(
                self.name,
                f"Reference audio not found: {request.reference_audio_path}",
            )

        logger.debug(
            "indicf5_synthesize",
            text_len=len(request.text),
            target_lang=request.target_language,
            ref_audio=str(request.reference_audio_path),
        )

        # Call IndicF5 model
        try:
            audio_output = self._model(
                request.text,
                ref_audio_path=str(request.reference_audio_path),
                ref_text=request.reference_text,
            )
        except Exception as e:
            raise ProviderError(self.name, f"IndicF5 inference failed: {e}") from e

        # Handle output format — IndicF5 returns numpy array at 24kHz
        sample_rate = 24000
        if isinstance(audio_output, tuple):
            audio_data, sample_rate = audio_output[0], audio_output[1]
        elif isinstance(audio_output, np.ndarray):
            audio_data = audio_output
        else:
            # Try to convert to numpy
            audio_data = np.array(audio_output, dtype=np.float32)

        # Normalize to prevent clipping
        peak = np.max(np.abs(audio_data))
        if peak > 0.95:
            audio_data = audio_data * (0.95 / peak)

        # Write output
        output_path = request.reference_audio_path.parent / f"indicf5_out_{id(request)}.wav"
        sf.write(output_path, audio_data, sample_rate)

        actual_duration = len(audio_data) / sample_rate

        return TTSResult(
            audio_path=output_path,
            actual_duration_sec=actual_duration,
            sample_rate=sample_rate,
            provider_name=self.name,
        )

    def supports_language(self, language_code: str) -> bool:
        return language_code in INDICF5_LANGUAGES

    async def health_check(self) -> bool:
        try:
            self._ensure_model()
            return self._model is not None
        except Exception:
            return False
