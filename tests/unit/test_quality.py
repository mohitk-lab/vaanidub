"""Tests for quality scoring module."""

import numpy as np
import soundfile as sf

from vaanidub.quality.scorer import QualityReport, QualityScorer
from vaanidub.quality.validators import (
    validate_audio_no_clipping,
    validate_audio_not_silent,
    validate_duration_match,
    validate_text_in_script,
)


class TestQualityReport:
    def test_compute_overall_score(self):
        report = QualityReport.compute(
            audio_quality=80,
            voice_similarity=90,
            timing_accuracy=70,
            target_language="hi",
        )
        assert report.target_language == "hi"
        assert report.audio_quality_score == 80
        assert report.voice_similarity_score == 90
        assert report.timing_accuracy_score == 70
        # overall = 80*0.25 + 90*0.40 + 70*0.35 = 20 + 36 + 24.5 = 80.5
        assert report.overall_score == 80.5

    def test_perfect_score(self):
        report = QualityReport.compute(100, 100, 100)
        assert report.overall_score == 100.0

    def test_zero_score(self):
        report = QualityReport.compute(0, 0, 0)
        assert report.overall_score == 0.0


class TestQualityScorer:
    def test_audio_quality_normal(self, sample_audio_path):
        scorer = QualityScorer()
        score = scorer.score_audio_quality(sample_audio_path)
        assert 0 <= score <= 100
        assert score > 30  # Normal audio should score decently

    def test_audio_quality_silent(self, silent_audio_path):
        scorer = QualityScorer()
        score = scorer.score_audio_quality(silent_audio_path)
        assert score == 0.0

    def test_timing_accuracy_perfect(self):
        scorer = QualityScorer()
        originals = [2.0, 3.0, 1.5]
        dubbed = [2.0, 3.0, 1.5]
        score = scorer.score_timing_accuracy(originals, dubbed)
        assert score >= 95

    def test_timing_accuracy_within_tolerance(self):
        scorer = QualityScorer()
        originals = [2.0, 3.0, 1.5]
        dubbed = [2.2, 3.3, 1.65]  # 10% deviation
        score = scorer.score_timing_accuracy(originals, dubbed, tolerance_percent=20)
        assert score >= 70

    def test_timing_accuracy_way_off(self):
        scorer = QualityScorer()
        originals = [2.0, 3.0]
        dubbed = [5.0, 8.0]  # Way over
        score = scorer.score_timing_accuracy(originals, dubbed)
        assert score < 50

    def test_timing_accuracy_empty(self):
        scorer = QualityScorer()
        assert scorer.score_timing_accuracy([], []) == 50.0


class TestValidators:
    def test_audio_not_silent_with_audio(self, sample_audio_path):
        assert validate_audio_not_silent(sample_audio_path) is True

    def test_audio_not_silent_with_silence(self, silent_audio_path):
        assert validate_audio_not_silent(silent_audio_path) is False

    def test_audio_no_clipping_normal(self, sample_audio_path):
        assert validate_audio_no_clipping(sample_audio_path) is True

    def test_audio_no_clipping_clipped(self, tmp_dir):
        path = tmp_dir / "clipped.wav"
        audio = np.ones(16000, dtype=np.float32)  # All max values
        sf.write(path, audio, 16000)
        assert validate_audio_no_clipping(path) is False

    def test_duration_match_within_tolerance(self):
        assert validate_duration_match(10.0, 11.5, tolerance_percent=20) is True

    def test_duration_match_outside_tolerance(self):
        assert validate_duration_match(10.0, 14.0, tolerance_percent=20) is False

    def test_duration_match_zero_original(self):
        assert validate_duration_match(0.0, 5.0) is True

    def test_text_in_script_hindi(self):
        assert validate_text_in_script("यह हिंदी में है", "hi") is True

    def test_text_in_script_tamil(self):
        assert validate_text_in_script("இது தமிழில் உள்ளது", "ta") is True

    def test_text_in_script_wrong_script(self):
        # Hindi text checked against Tamil
        assert validate_text_in_script("यह हिंदी में है", "ta") is False

    def test_text_in_script_english_only(self):
        # Pure ASCII should fail script validation for Indian languages
        assert validate_text_in_script("This is English", "hi") is False
