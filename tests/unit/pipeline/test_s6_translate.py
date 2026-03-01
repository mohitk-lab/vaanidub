"""Tests for Stage 6: Translation."""

from unittest.mock import patch

import pytest

from vaanidub.exceptions import AllProvidersFailed, StageError
from vaanidub.pipeline.context import PipelineContext, Segment
from vaanidub.pipeline.stages.s6_translate import TranslateStage


class TestTranslateStageMetadata:
    def test_stage_metadata(self):
        stage = TranslateStage(device="cpu")
        assert stage.name == "translate"
        assert stage.stage_number == 6


class TestTranslateExecute:
    @pytest.mark.asyncio
    async def test_rejects_no_segments(self, sample_job_dir):
        stage = TranslateStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"]
        )
        ctx.segments = []
        with pytest.raises(StageError, match="No segments to translate"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_rejects_no_target_languages(self, sample_job_dir):
        stage = TranslateStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=[]
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1, text="hello")
        ]
        with pytest.raises(StageError, match="No target languages"):
            await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_indictrans_success(self, sample_job_dir):
        """Mock IndicTrans2 model to verify translations are stored."""
        stage = TranslateStage(device="cpu")

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"],
            source_language="en",
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1, text="Hello world"),
            Segment(index=1, speaker_label="S0", start_time=1, end_time=2, text="How are you"),
        ]

        # Mock the IndicTrans2 translation method
        async def mock_translate(texts, src, tgt):
            return [f"translated_{t}" for t in texts]

        with patch.object(stage, "_translate_with_indictrans", side_effect=mock_translate):
            ctx = await stage.execute(ctx)

        assert ctx.segments[0].translations["hi"] == "translated_Hello world"
        assert ctx.segments[1].translations["hi"] == "translated_How are you"

    @pytest.mark.asyncio
    async def test_indictrans_failure_google_fallback(self, sample_job_dir):
        """When IndicTrans2 fails, Google Translate should be used."""
        stage = TranslateStage(device="cpu", google_api_key="test-key")

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"],
            source_language="en",
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1, text="Hello"),
        ]

        async def mock_indictrans_fail(texts, src, tgt):
            raise RuntimeError("Model load failed")

        async def mock_google_translate(texts, src, tgt):
            return [f"google_{t}" for t in texts]

        with patch.object(stage, "_translate_with_indictrans", side_effect=mock_indictrans_fail), \
             patch.object(stage, "_translate_with_google", side_effect=mock_google_translate):
            ctx = await stage.execute(ctx)

        assert ctx.segments[0].translations["hi"] == "google_Hello"

    @pytest.mark.asyncio
    async def test_both_providers_fail(self, sample_job_dir):
        """AllProvidersFailed when both IndicTrans2 and Google fail."""
        stage = TranslateStage(device="cpu", google_api_key="test-key")

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"],
            source_language="en",
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1, text="Hello"),
        ]

        async def mock_fail(texts, src, tgt):
            raise RuntimeError("Failed")

        with patch.object(stage, "_translate_with_indictrans", side_effect=mock_fail), \
             patch.object(stage, "_translate_with_google", side_effect=mock_fail):
            with pytest.raises(AllProvidersFailed):
                await stage.execute(ctx)

    @pytest.mark.asyncio
    async def test_empty_segments_get_empty_translation(self, sample_job_dir):
        """Segments with no text should get empty string translations."""
        stage = TranslateStage(device="cpu")

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["hi"],
            source_language="en",
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1, text=""),
            Segment(index=1, speaker_label="S0", start_time=1, end_time=2, text="Hello"),
        ]

        async def mock_translate(texts, src, tgt):
            return [f"translated_{t}" for t in texts]

        with patch.object(stage, "_translate_with_indictrans", side_effect=mock_translate):
            ctx = await stage.execute(ctx)

        assert ctx.segments[0].translations["hi"] == ""
        assert ctx.segments[1].translations["hi"] == "translated_Hello"

    @pytest.mark.asyncio
    async def test_unsupported_target_language_skipped(self, sample_job_dir):
        """Languages not in LANGUAGES dict should be skipped."""
        stage = TranslateStage(device="cpu")

        ctx = PipelineContext(
            job_id="test", job_dir=sample_job_dir, target_languages=["xx"],
            source_language="en",
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1, text="Hello"),
        ]

        async def mock_translate(texts, src, tgt):
            return [f"translated_{t}" for t in texts]

        with patch.object(stage, "_translate_with_indictrans", side_effect=mock_translate):
            ctx = await stage.execute(ctx)

        # "xx" should be skipped — no translation for it
        assert "xx" not in ctx.segments[0].translations


class TestTranslateMapCodes:
    def test_map_english(self):
        stage = TranslateStage(device="cpu")
        assert stage._map_to_indictrans_code("en") == "eng_Latn"

    def test_map_hindi(self):
        stage = TranslateStage(device="cpu")
        assert stage._map_to_indictrans_code("hi") == "hin_Deva"

    def test_map_tamil(self):
        stage = TranslateStage(device="cpu")
        assert stage._map_to_indictrans_code("ta") == "tam_Taml"

    def test_map_unknown_defaults_to_english(self):
        stage = TranslateStage(device="cpu")
        assert stage._map_to_indictrans_code("zz") == "eng_Latn"

    def test_all_languages_mapped(self):
        stage = TranslateStage(device="cpu")
        expected = {
            "en": "eng_Latn", "hi": "hin_Deva", "ta": "tam_Taml",
            "te": "tel_Telu", "bn": "ben_Beng", "mr": "mar_Deva",
            "kn": "kan_Knda", "ml": "mal_Mlym", "gu": "guj_Gujr",
            "as": "asm_Beng", "or": "ory_Orya", "pa": "pan_Guru",
        }
        for code, expected_code in expected.items():
            assert stage._map_to_indictrans_code(code) == expected_code


class TestTranslateValidateOutput:
    @pytest.mark.asyncio
    async def test_validate_output_valid(self, tmp_dir):
        stage = TranslateStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1,
                    text="Hello", translations={"hi": "नमस्ते"}),
        ]
        assert await stage.validate_output(ctx) is True

    @pytest.mark.asyncio
    async def test_validate_output_missing_lang(self, tmp_dir):
        stage = TranslateStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi", "ta"]
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1,
                    text="Hello", translations={"hi": "नमस्ते"}),
        ]
        # "ta" is missing translations
        assert await stage.validate_output(ctx) is False

    @pytest.mark.asyncio
    async def test_validate_output_no_translations(self, tmp_dir):
        stage = TranslateStage(device="cpu")
        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )
        ctx.segments = [
            Segment(index=0, speaker_label="S0", start_time=0, end_time=1, text="Hello"),
        ]
        assert await stage.validate_output(ctx) is False
