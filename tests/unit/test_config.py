"""Tests for configuration loading."""



class TestAppConfig:
    def test_default_config_loads(self):
        from vaanidub.config import AppConfig
        config = AppConfig()
        assert config.app_name == "vaanidub"
        assert config.debug is False
        assert config.max_concurrent_jobs == 2

    def test_config_providers_defaults(self):
        from vaanidub.config import AppConfig
        config = AppConfig()
        assert config.providers.stt.model_size == "large-v2"
        assert config.providers.tts.primary == "indicf5"
        assert config.providers.tts.fallback == "elevenlabs"

    def test_resolve_secrets(self):
        from vaanidub.config import AppConfig
        config = AppConfig(hf_token="test_token", elevenlabs_api_key="el_key")
        config.resolve_secrets()
        assert config.providers.diarization.hf_token == "test_token"
        assert config.providers.tts.elevenlabs.api_key == "el_key"

    def test_ensure_directories(self, tmp_dir):
        from vaanidub.config import AppConfig, StorageConfig
        config = AppConfig(
            storage=StorageConfig(
                base_path=tmp_dir / "jobs",
                temp_path=tmp_dir / "tmp",
            )
        )
        config.ensure_directories()
        assert (tmp_dir / "jobs").exists()
        assert (tmp_dir / "tmp").exists()

    def test_quality_config_defaults(self):
        from vaanidub.config import AppConfig
        config = AppConfig()
        assert config.quality.min_acceptable_score == 60.0
        assert config.quality.voice_similarity_threshold == 0.55
        assert config.quality.timing_tolerance_percent == 20.0
