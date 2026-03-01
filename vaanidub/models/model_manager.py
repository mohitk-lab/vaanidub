"""AI model lifecycle manager — download, load, unload, GPU memory tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class ModelInfo:
    """Metadata about a registered AI model."""
    name: str
    model_type: str          # stt, diarization, separation, translation, tts
    gpu_vram_mb: int
    cpu_ram_mb: int
    download_size_gb: float
    description: str


# Registry of all models used in the pipeline
MODEL_REGISTRY: dict[str, ModelInfo] = {
    "faster_whisper_large_v2": ModelInfo(
        name="faster_whisper_large_v2",
        model_type="stt",
        gpu_vram_mb=4000,
        cpu_ram_mb=8000,
        download_size_gb=3.0,
        description="Faster Whisper large-v2 for speech recognition",
    ),
    "pyannote_diarization": ModelInfo(
        name="pyannote_diarization",
        model_type="diarization",
        gpu_vram_mb=1500,
        cpu_ram_mb=3000,
        download_size_gb=0.5,
        description="pyannote speaker diarization 3.1",
    ),
    "demucs_htdemucs_ft": ModelInfo(
        name="demucs_htdemucs_ft",
        model_type="separation",
        gpu_vram_mb=2000,
        cpu_ram_mb=4000,
        download_size_gb=0.3,
        description="Demucs vocal separation (htdemucs_ft)",
    ),
    "indictrans2_en_indic_200m": ModelInfo(
        name="indictrans2_en_indic_200m",
        model_type="translation",
        gpu_vram_mb=1500,
        cpu_ram_mb=2000,
        download_size_gb=0.8,
        description="IndicTrans2 English to Indian languages (200M distilled)",
    ),
    "indicf5": ModelInfo(
        name="indicf5",
        model_type="tts",
        gpu_vram_mb=4000,
        cpu_ram_mb=8000,
        download_size_gb=2.5,
        description="IndicF5 zero-shot voice cloning TTS for 11 Indian languages",
    ),
}


class ModelManager:
    """Manages AI model lifecycle including GPU memory."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "vaanidub" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, Any] = {}

    def list_models(self) -> list[dict]:
        """List all registered models with their status."""
        result = []
        for name, info in MODEL_REGISTRY.items():
            result.append({
                "name": name,
                "type": info.model_type,
                "description": info.description,
                "gpu_vram_mb": info.gpu_vram_mb,
                "download_size_gb": info.download_size_gb,
                "loaded": name in self._loaded,
            })
        return result

    def get_total_download_size(self) -> float:
        """Get total download size of all models in GB."""
        return sum(m.download_size_gb for m in MODEL_REGISTRY.values())

    def get_gpu_requirements(self) -> dict:
        """Get GPU VRAM requirements summary."""
        max_vram = max(m.gpu_vram_mb for m in MODEL_REGISTRY.values())
        total_vram = sum(m.gpu_vram_mb for m in MODEL_REGISTRY.values())
        return {
            "min_vram_mb": max_vram,  # Pipeline runs sequentially
            "total_if_concurrent_mb": total_vram,
            "recommendation": f"Minimum {max_vram}MB VRAM (sequential), "
                              f"{total_vram}MB for concurrent loading",
        }

    def check_gpu(self) -> dict:
        """Check GPU availability and memory."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                return {
                    "available": True,
                    "device_name": props.name,
                    "total_vram_mb": total_mem // (1024 * 1024),
                    "free_vram_mb": free_mem // (1024 * 1024),
                    "cuda_version": torch.version.cuda,
                }
            else:
                return {"available": False, "reason": "CUDA not available"}
        except ImportError:
            return {"available": False, "reason": "PyTorch not installed"}

    async def download_model(self, model_name: str, hf_token: str = "") -> None:
        """Download a specific model."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        info = MODEL_REGISTRY[model_name]
        logger.info("downloading_model", model=model_name, size_gb=info.download_size_gb)

        if model_name == "faster_whisper_large_v2":
            from faster_whisper import WhisperModel
            WhisperModel("large-v2", device="cpu", compute_type="int8")

        elif model_name == "pyannote_diarization":
            from pyannote.audio import Pipeline
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )

        elif model_name == "indictrans2_en_indic_200m":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            AutoTokenizer.from_pretrained(
                "ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True
            )
            AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True
            )

        elif model_name == "indicf5":
            from transformers import AutoModel
            AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

        elif model_name == "demucs_htdemucs_ft":
            # Demucs downloads on first use
            pass

        logger.info("model_downloaded", model=model_name)

    async def download_all(self, hf_token: str = "") -> None:
        """Download all required models."""
        for name in MODEL_REGISTRY:
            await self.download_model(name, hf_token=hf_token)
