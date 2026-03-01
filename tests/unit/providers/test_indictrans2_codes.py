"""Tests for IndicTrans2 language code mapping."""

from vaanidub.pipeline.providers.translation.indictrans2 import INDICTRANS_CODES


class TestIndicTransCodes:
    def test_all_11_languages_mapped(self):
        expected = {"en", "hi", "ta", "te", "bn", "mr", "kn", "ml", "gu", "as", "or", "pa"}
        assert set(INDICTRANS_CODES.keys()) == expected

    def test_hindi_code(self):
        assert INDICTRANS_CODES["hi"] == "hin_Deva"

    def test_tamil_code(self):
        assert INDICTRANS_CODES["ta"] == "tam_Taml"

    def test_english_code(self):
        assert INDICTRANS_CODES["en"] == "eng_Latn"

    def test_all_codes_are_flores200_format(self):
        """Flores-200 codes are xxx_Yyyy format."""
        for lang, code in INDICTRANS_CODES.items():
            parts = code.split("_")
            assert len(parts) == 2, f"Invalid code format for {lang}: {code}"
            assert len(parts[0]) == 3, f"Language part should be 3 chars for {lang}"
            assert parts[1][0].isupper(), f"Script should start uppercase for {lang}"
