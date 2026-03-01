"""Per-stage output validators for quality assurance."""

from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

logger = structlog.get_logger()


def validate_audio_not_silent(audio_path: Path, min_rms: float = 1e-5) -> bool:
    """Check that an audio file is not silent."""
    try:
        audio, sr = sf.read(audio_path)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        return bool(rms >= min_rms)
    except Exception:
        return False


def validate_audio_no_clipping(audio_path: Path, threshold: float = 0.99) -> bool:
    """Check that audio doesn't have excessive clipping."""
    try:
        audio, sr = sf.read(audio_path)
        clipping_ratio = float(np.mean(np.abs(audio) > threshold))
        return bool(clipping_ratio < 0.01)  # Less than 1% clipping
    except Exception:
        return False


def validate_duration_match(
    original_duration: float,
    dubbed_duration: float,
    tolerance_percent: float = 30.0,
) -> bool:
    """Check that dubbed audio duration is within tolerance of original."""
    if original_duration <= 0:
        return True
    deviation = abs(dubbed_duration - original_duration) / original_duration * 100
    return deviation <= tolerance_percent


def validate_text_in_script(text: str, target_language: str) -> bool:
    """Verify translated text is in the expected script for the target language."""
    from vaanidub.constants import LANGUAGES

    lang_info = LANGUAGES.get(target_language)
    if lang_info is None:
        return True  # Can't validate unknown language

    # Simple Unicode range checks for Indian scripts
    script_ranges = {
        "Devanagari": (0x0900, 0x097F),
        "Tamil": (0x0B80, 0x0BFF),
        "Telugu": (0x0C00, 0x0C7F),
        "Bengali": (0x0980, 0x09FF),
        "Kannada": (0x0C80, 0x0CFF),
        "Malayalam": (0x0D00, 0x0D7F),
        "Gujarati": (0x0A80, 0x0AFF),
        "Gurmukhi": (0x0A00, 0x0A7F),
        "Odia": (0x0B00, 0x0B7F),
    }

    script = lang_info.script
    if script not in script_ranges:
        return True

    low, high = script_ranges[script]

    # Check if at least 30% of non-space characters are in the expected script
    chars = [c for c in text if not c.isspace() and not c.isascii()]
    if not chars:
        return False

    in_script = sum(1 for c in chars if low <= ord(c) <= high)
    return (in_script / len(chars)) >= 0.3 if chars else True
