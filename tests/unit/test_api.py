"""Tests for FastAPI API endpoints."""

import pytest


@pytest.fixture
def api_client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient

    from vaanidub.api.app import create_app
    from vaanidub.config import AppConfig, DatabaseConfig, StorageConfig

    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///"),
        storage=StorageConfig(
            base_path="/tmp/vaanidub_test/jobs",
            temp_path="/tmp/vaanidub_test/tmp",
        ),
    )
    config.ensure_directories()

    app = create_app(config)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, api_client):
        resp = api_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert data["version"] == "0.1.0"


class TestLanguagesEndpoint:
    def test_list_languages(self, api_client):
        resp = api_client.get("/api/v1/languages")
        assert resp.status_code == 200
        data = resp.json()
        assert "languages" in data
        assert len(data["languages"]) == 11

        # Check Hindi is present
        hindi = next(
            lang for lang in data["languages"] if lang["code"] == "hi"
        )
        assert hindi["name"] == "Hindi"
        assert hindi["native_name"] == "हिन्दी"
        assert "indicf5" in hindi["tts_providers"]

    def test_all_languages_have_required_fields(self, api_client):
        resp = api_client.get("/api/v1/languages")
        for lang in resp.json()["languages"]:
            assert "code" in lang
            assert "name" in lang
            assert "native_name" in lang
            assert "script" in lang
            assert "tts_providers" in lang


class TestJobsEndpoint:
    def test_list_jobs_empty(self, api_client):
        resp = api_client.get("/api/v1/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_get_nonexistent_job(self, api_client):
        resp = api_client.get("/api/v1/jobs/nonexistent")
        assert resp.status_code == 404
