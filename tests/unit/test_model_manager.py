"""Tests for model manager."""

from vaanidub.models.model_manager import MODEL_REGISTRY, ModelManager


class TestModelRegistry:
    def test_all_required_models_registered(self):
        required = {
            "faster_whisper_large_v2",
            "pyannote_diarization",
            "demucs_htdemucs_ft",
            "indictrans2_en_indic_200m",
            "indicf5",
        }
        assert required == set(MODEL_REGISTRY.keys())

    def test_model_info_complete(self):
        for name, info in MODEL_REGISTRY.items():
            assert info.name == name
            assert info.model_type in ("stt", "diarization", "separation", "translation", "tts")
            assert info.gpu_vram_mb > 0
            assert info.download_size_gb > 0
            assert info.description


class TestModelManager:
    def test_list_models(self):
        mgr = ModelManager()
        models = mgr.list_models()
        assert len(models) == len(MODEL_REGISTRY)
        for m in models:
            assert "name" in m
            assert "type" in m
            assert "loaded" in m
            assert m["loaded"] is False

    def test_total_download_size(self):
        mgr = ModelManager()
        total = mgr.get_total_download_size()
        assert total > 5  # At least 5GB total

    def test_gpu_requirements(self):
        mgr = ModelManager()
        reqs = mgr.get_gpu_requirements()
        assert reqs["min_vram_mb"] > 0
        assert reqs["total_if_concurrent_mb"] > reqs["min_vram_mb"]
        assert "recommendation" in reqs

    def test_cache_dir_creation(self, tmp_dir):
        cache = tmp_dir / "model_cache"
        ModelManager(cache_dir=cache)
        assert cache.exists()
