"""Pydantic schemas for API request/response models."""

from datetime import datetime

from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    target_languages: list[str] = Field(..., min_length=1, description="Target language codes")
    source_language: str | None = Field(
        None, description="Source language (auto-detect if omitted)"
    )
    webhook_url: str | None = Field(None, description="Callback URL on completion")


class JobResponse(BaseModel):
    job_id: str
    status: str
    current_stage: str | None = None
    progress: float = 0.0
    source_language: str | None = None
    source_language_confidence: float | None = None
    target_languages: list[str]
    speakers_detected: int | None = None
    segments_count: int | None = None
    duration_seconds: float | None = None
    error_message: str | None = None
    output_paths: dict[str, str] | None = None
    stage_timings: dict[str, float] | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int
    page: int
    per_page: int


class StageStatus(BaseModel):
    name: str
    status: str
    duration_sec: float | None = None
    provider_used: str | None = None


class JobDetailResponse(JobResponse):
    stages: list[StageStatus] = []


class LanguageResponse(BaseModel):
    code: str
    name: str
    native_name: str
    script: str
    tts_providers: list[str]


class LanguageListResponse(BaseModel):
    languages: list[LanguageResponse]


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    gpu_device: str | None = None
    gpu_free_vram_mb: int | None = None
    redis_connected: bool
    db_connected: bool
    worker_count: int = 0


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
