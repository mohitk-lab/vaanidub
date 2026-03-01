"""Language definitions, supported formats, and pipeline constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageInfo:
    code: str
    name: str
    native_name: str
    script: str
    indictrans2_code: str
    tts_providers: tuple[str, ...]


# All supported Indian languages
LANGUAGES: dict[str, LanguageInfo] = {
    "hi": LanguageInfo("hi", "Hindi", "हिन्दी", "Devanagari", "hin_Deva",
                       ("indicf5", "elevenlabs")),
    "ta": LanguageInfo("ta", "Tamil", "தமிழ்", "Tamil", "tam_Taml",
                       ("indicf5", "elevenlabs")),
    "te": LanguageInfo("te", "Telugu", "తెలుగు", "Telugu", "tel_Telu",
                       ("indicf5", "elevenlabs")),
    "bn": LanguageInfo("bn", "Bengali", "বাংলা", "Bengali", "ben_Beng",
                       ("indicf5", "elevenlabs")),
    "mr": LanguageInfo("mr", "Marathi", "मराठी", "Devanagari", "mar_Deva",
                       ("indicf5", "elevenlabs")),
    "kn": LanguageInfo("kn", "Kannada", "ಕನ್ನಡ", "Kannada", "kan_Knda",
                       ("indicf5", "elevenlabs")),
    "ml": LanguageInfo("ml", "Malayalam", "മലയാളം", "Malayalam", "mal_Mlym",
                       ("indicf5", "elevenlabs")),
    "gu": LanguageInfo("gu", "Gujarati", "ગુજરાતી", "Gujarati", "guj_Gujr",
                       ("indicf5", "elevenlabs")),
    "as": LanguageInfo("as", "Assamese", "অসমীয়া", "Bengali", "asm_Beng",
                       ("indicf5",)),
    "or": LanguageInfo("or", "Odia", "ଓଡ଼ିଆ", "Odia", "ory_Orya",
                       ("indicf5",)),
    "pa": LanguageInfo("pa", "Punjabi", "ਪੰਜਾਬੀ", "Gurmukhi", "pan_Guru",
                       ("indicf5",)),
}

LANGUAGE_CODES = set(LANGUAGES.keys())

# Supported input formats
AUDIO_FORMATS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}
VIDEO_FORMATS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv", ".wmv"}
SUPPORTED_FORMATS = AUDIO_FORMATS | VIDEO_FORMATS

# Pipeline stage names
STAGE_NAMES = (
    "ingest",
    "separate",
    "diarize",
    "transcribe",
    "prosody",
    "translate",
    "synthesize",
    "mixdown",
)

# Job statuses
class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Audio processing defaults
SAMPLE_RATE = 16000           # 16kHz for STT pipeline
TARGET_LUFS = -16.0           # Broadcast standard for spoken word
CHUNK_DURATION_SEC = 300      # 5 minutes per processing chunk
MAX_TIME_STRETCH = 1.3        # Max speedup for duration matching
MIN_REFERENCE_CLIP_SEC = 5    # Min speaker reference clip for voice cloning
MAX_REFERENCE_CLIP_SEC = 15   # Max speaker reference clip
