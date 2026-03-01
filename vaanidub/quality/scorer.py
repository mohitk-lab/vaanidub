"""Quality scoring for dubbed output — voice similarity, timing, audio quality."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

logger = structlog.get_logger()


@dataclass
class QualityReport:
    """Quality assessment report for a dubbed output."""
    target_language: str
    audio_quality_score: float      # 0-100 (SNR-based)
    voice_similarity_score: float   # 0-100 (speaker embedding cosine similarity)
    timing_accuracy_score: float    # 0-100 (how close dubbed matches original timing)
    overall_score: float            # Weighted average

    @staticmethod
    def compute(
        audio_quality: float,
        voice_similarity: float,
        timing_accuracy: float,
        target_language: str = "",
    ) -> "QualityReport":
        overall = (
            audio_quality * 0.25
            + voice_similarity * 0.40  # Voice matching is most important
            + timing_accuracy * 0.35
        )
        return QualityReport(
            target_language=target_language,
            audio_quality_score=round(audio_quality, 1),
            voice_similarity_score=round(voice_similarity, 1),
            timing_accuracy_score=round(timing_accuracy, 1),
            overall_score=round(overall, 1),
        )


class QualityScorer:
    """Computes quality scores for dubbed output."""

    def score_audio_quality(self, audio_path: Path) -> float:
        """Score audio quality based on SNR and basic metrics (0-100)."""
        try:
            audio, sr = sf.read(audio_path)
            rms = np.sqrt(np.mean(audio ** 2))

            # Estimate SNR: signal vs noise floor
            # Higher RMS relative to noise = better quality
            if rms < 1e-8:
                return 0.0

            # Simple SNR estimation
            peak = np.max(np.abs(audio))
            noise_floor = np.percentile(np.abs(audio), 5)

            if noise_floor < 1e-8:
                snr_db = 60.0
            else:
                snr_db = 20 * np.log10(peak / noise_floor)

            # Check for clipping
            clipping_ratio = np.mean(np.abs(audio) > 0.99)
            clipping_penalty = clipping_ratio * 50

            # Map SNR to 0-100 score
            score = min(100, max(0, snr_db * 2 - clipping_penalty))
            return score

        except Exception as e:
            logger.warning("audio_quality_scoring_failed", error=str(e))
            return 50.0  # Default neutral score

    def score_timing_accuracy(
        self,
        original_durations: list[float],
        dubbed_durations: list[float],
        tolerance_percent: float = 20.0,
    ) -> float:
        """Score how well dubbed segments match original timing (0-100)."""
        if not original_durations or not dubbed_durations:
            return 50.0

        min_len = min(len(original_durations), len(dubbed_durations))
        scores = []

        for i in range(min_len):
            orig = original_durations[i]
            dub = dubbed_durations[i]

            if orig <= 0:
                continue

            deviation_pct = abs(dub - orig) / orig * 100

            if deviation_pct <= tolerance_percent:
                # Within tolerance = full score
                score = 100 - (deviation_pct / tolerance_percent * 20)
            else:
                # Beyond tolerance, penalize heavily
                score = max(0, 80 - (deviation_pct - tolerance_percent) * 2)

            scores.append(score)

        return np.mean(scores) if scores else 50.0

    def score_voice_similarity(
        self,
        original_audio_path: Path,
        dubbed_audio_path: Path,
    ) -> float:
        """Score voice similarity using basic spectral comparison (0-100).

        For production, use pyannote speaker embeddings for true cosine similarity.
        This is a simplified spectral correlation approach.
        """
        try:
            import librosa

            orig, sr1 = librosa.load(str(original_audio_path), sr=16000, duration=10)
            dub, sr2 = librosa.load(str(dubbed_audio_path), sr=16000, duration=10)

            # Compare MFCCs (Mel-frequency cepstral coefficients)
            orig_mfcc = librosa.feature.mfcc(y=orig, sr=16000, n_mfcc=13)
            dub_mfcc = librosa.feature.mfcc(y=dub, sr=16000, n_mfcc=13)

            # Compute mean MFCC per coefficient
            orig_mean = np.mean(orig_mfcc, axis=1)
            dub_mean = np.mean(dub_mfcc, axis=1)

            # Cosine similarity
            dot = np.dot(orig_mean, dub_mean)
            norm = np.linalg.norm(orig_mean) * np.linalg.norm(dub_mean)

            if norm == 0:
                return 50.0

            similarity = dot / norm  # -1 to 1
            # Map to 0-100
            score = (similarity + 1) * 50
            return min(100, max(0, score))

        except Exception as e:
            logger.warning("voice_similarity_scoring_failed", error=str(e))
            return 50.0
