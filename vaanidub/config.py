"""Application configuration via pydantic-settings with layered loading."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class STTConfig(BaseSettings):
    model_size: str = "large-v2"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True


class DiarizationConfig(BaseSettings):
    pipeline: str = "pyannote/speaker-diarization-3.1"
    device: str = "cuda"
    hf_token: str = ""
    min_speakers: int = 1
    max_speakers: int = 10


class SeparationConfig(BaseSettings):
    model: str = "htdemucs_ft"
    device: str = "cuda"
    two_stems: str = "vocals"


class TranslationConfig(BaseSettings):
    model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M"
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 256


class IndicF5Config(BaseSettings):
    model_name: str = "ai4bharat/IndicF5"
    device: str = "cuda"
    reference_clip_min_sec: float = 5.0
    reference_clip_max_sec: float = 15.0


class ElevenLabsConfig(BaseSettings):
    api_key: str = ""
    model: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75


class TTSConfig(BaseSettings):
    primary: str = "indicf5"
    fallback: str = "elevenlabs"
    indicf5: IndicF5Config = IndicF5Config()
    elevenlabs: ElevenLabsConfig = ElevenLabsConfig()


class ProvidersConfig(BaseSettings):
    stt: STTConfig = STTConfig()
    diarization: DiarizationConfig = DiarizationConfig()
    separation: SeparationConfig = SeparationConfig()
    translation: TranslationConfig = TranslationConfig()
    tts: TTSConfig = TTSConfig()


class DatabaseConfig(BaseSettings):
    url: str = "sqlite:///./data/vaanidub.db"
    pool_size: int = 5
    max_overflow: int = 10


class RedisConfig(BaseSettings):
    url: str = "redis://localhost:6379/0"


class StorageConfig(BaseSettings):
    base_path: Path = Path("./data/jobs")
    temp_path: Path = Path("./data/tmp")
    cleanup_after_days: int = 30


class APIConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]


class QualityConfig(BaseSettings):
    min_acceptable_score: float = 60.0
    voice_similarity_threshold: float = 0.55
    timing_tolerance_percent: float = 20.0
    auto_escalate_to_paid: bool = False
    run_quality_check: bool = True


class GPUConfig(BaseSettings):
    device: str = "cuda:0"
    max_vram_usage_percent: int = 90
    model_offload_to_cpu: bool = True


class AppConfig(BaseSettings):
    """Root application configuration. Loads from .env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    # Core
    app_name: str = "vaanidub"
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "console"
    max_concurrent_jobs: int = 2
    max_upload_size_mb: int = 2048
    max_audio_duration_sec: int = 7200  # 2 hours

    # HuggingFace token (needed for gated models)
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # Paid API keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    google_translate_api_key: str = Field(default="", alias="GOOGLE_TRANSLATE_API_KEY")

    # Sub-configs
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    storage: StorageConfig = StorageConfig()
    api: APIConfig = APIConfig()
    providers: ProvidersConfig = ProvidersConfig()
    quality: QualityConfig = QualityConfig()
    gpu: GPUConfig = GPUConfig()

    def resolve_secrets(self) -> None:
        """Propagate top-level API keys into provider configs."""
        providers = self.providers
        if self.hf_token and not providers.diarization.hf_token:
            new_diar = providers.diarization.model_copy(update={"hf_token": self.hf_token})
            providers = providers.model_copy(update={"diarization": new_diar})
        if self.elevenlabs_api_key and not providers.tts.elevenlabs.api_key:
            new_el = providers.tts.elevenlabs.model_copy(
                update={"api_key": self.elevenlabs_api_key}
            )
            new_tts = providers.tts.model_copy(update={"elevenlabs": new_el})
            providers = providers.model_copy(update={"tts": new_tts})
        self.__dict__["providers"] = providers

    def ensure_directories(self) -> None:
        """Create required data directories."""
        self.storage.base_path.mkdir(parents=True, exist_ok=True)
        self.storage.temp_path.mkdir(parents=True, exist_ok=True)
