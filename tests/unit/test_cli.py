"""Tests for CLI commands."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from vaanidub.cli.main import app

runner = CliRunner()


class TestLanguagesCommand:
    def test_languages_command(self):
        result = runner.invoke(app, ["languages"])
        assert result.exit_code == 0
        assert "Hindi" in result.output
        assert "Tamil" in result.output

    def test_languages_shows_all_codes(self):
        result = runner.invoke(app, ["languages"])
        assert result.exit_code == 0
        for code in ["hi", "ta", "te", "bn", "mr", "kn", "ml", "gu", "pa"]:
            assert code in result.output


class TestModelsListCommand:
    def test_models_list_command(self):
        """Mock ModelManager to avoid GPU checks."""
        mock_manager_instance = MagicMock()
        mock_manager_instance.list_models.return_value = [
            {
                "name": "faster_whisper_large_v2",
                "type": "stt",
                "gpu_vram_mb": 3000,
                "download_size_gb": 3.1,
                "description": "Speech-to-text",
            }
        ]
        mock_manager_instance.check_gpu.return_value = {
            "available": False,
            "reason": "test env",
        }
        mock_manager_instance.get_gpu_requirements.return_value = {"min_vram_mb": 4000}

        mock_cls = MagicMock(return_value=mock_manager_instance)

        # ModelManager is imported inside the function, so patch the module attribute
        with patch("vaanidub.models.model_manager.ModelManager", mock_cls):
            result = runner.invoke(app, ["models", "list"])

        assert result.exit_code == 0
        assert "faster_whisper_large_v2" in result.output


class TestDubCommand:
    def test_dub_missing_file(self):
        result = runner.invoke(app, ["dub", "/nonexistent/file.mp4", "--target", "hi"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_dub_unsupported_language(self, tmp_dir):
        test_file = tmp_dir / "test.mp4"
        test_file.write_bytes(b"fake video")

        result = runner.invoke(app, ["dub", str(test_file), "--target", "xx"])
        assert result.exit_code == 1
        assert "Unsupported language" in result.output


class TestDetectCommand:
    def test_detect_missing_file(self):
        result = runner.invoke(app, ["detect", "/nonexistent/file.wav"])
        assert result.exit_code == 1
        assert "not found" in result.output


class TestHelpOutput:
    def test_help_output(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "AI-powered" in result.output

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer with no_args_is_help returns exit code 0 with help text
        # Some versions may return 2 (usage error); both are acceptable
        assert "AI-powered" in result.output or "Usage" in result.output


class TestJobsCommand:
    def test_jobs_list_connection_error(self):
        """When server is not running, should show connection error."""
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            result = runner.invoke(app, ["jobs", "list", "--api-url", "http://localhost:9999"])

        assert result.exit_code == 1
        assert "Cannot connect" in result.output

    def test_jobs_list_success(self):
        """Mock successful API response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "abc12345-6789",
                "status": "completed",
                "progress": 100,
                "target_languages": ["hi", "ta"],
                "created_at": "2026-03-01T12:00:00",
            }
        ]

        with patch("httpx.get", return_value=mock_resp):
            result = runner.invoke(app, ["jobs", "list"])

        assert result.exit_code == 0
        assert "abc12345" in result.output

    def test_jobs_status_not_found(self):
        """Mock 404 response for missing job."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("httpx.get", return_value=mock_resp):
            result = runner.invoke(app, ["jobs", "status", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output
