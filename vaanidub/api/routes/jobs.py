"""Job management API routes."""

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from vaanidub.api.deps import get_config, get_db
from vaanidub.api.schemas import JobDetailResponse, JobListResponse, JobResponse
from vaanidub.config import AppConfig
from vaanidub.constants import LANGUAGES, SUPPORTED_FORMATS
from vaanidub.db.models import Job

router = APIRouter(tags=["jobs"])


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(
    file: UploadFile = File(...),
    target_languages: str = Form(..., description='JSON array: ["hi","ta"]'),
    source_language: str | None = Form(None),
    config: AppConfig = Depends(get_config),
    db: Session = Depends(get_db),
):
    """Create a new dubbing job."""
    # Parse target languages
    try:
        targets = json.loads(target_languages)
    except json.JSONDecodeError:
        raise HTTPException(400, "target_languages must be a valid JSON array")

    if not isinstance(targets, list) or not targets:
        raise HTTPException(400, "target_languages must be a non-empty array")

    # Validate languages
    for lang in targets:
        if lang not in LANGUAGES:
            raise HTTPException(400, f"Unsupported language: {lang}")

    # Validate file extension
    if file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise HTTPException(400, f"Unsupported format: {suffix}")

    # Save uploaded file
    job_id = str(uuid.uuid4())[:8]
    job_dir = config.storage.base_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input" / (file.filename or "input")
    input_path.parent.mkdir(exist_ok=True)

    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create DB record
    job = Job(
        id=job_id,
        status="pending",
        input_file_path=str(input_path),
        input_type="video" if suffix in {".mp4", ".mkv", ".avi", ".webm", ".mov"} else "audio",
        source_language=source_language,
        target_languages=json.dumps(targets),
    )
    db.add(job)
    db.commit()

    # Queue the job for processing
    try:
        from vaanidub.worker.tasks import process_dubbing_job
        process_dubbing_job.delay(job_id)
    except Exception:
        # Celery/Redis not available — fall back to demo pipeline in background thread
        import threading
        from vaanidub.demo_pipeline import run_demo_pipeline
        thread = threading.Thread(target=run_demo_pipeline, args=(job_id,), daemon=True)
        thread.start()

    return JobResponse(
        job_id=job_id,
        status="pending",
        target_languages=targets,
        source_language=source_language,
    )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: str | None = None,
    page: int = 1,
    per_page: int = 20,
    db: Session = Depends(get_db),
):
    """List dubbing jobs with pagination."""
    query = db.query(Job)
    if status:
        query = query.filter(Job.status == status)

    total = query.count()
    jobs = query.order_by(Job.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()

    return JobListResponse(
        jobs=[
            JobResponse(
                job_id=j.id,
                status=j.status,
                current_stage=j.current_stage,
                progress=j.progress or 0,
                target_languages=json.loads(j.target_languages) if j.target_languages else [],
                source_language=j.source_language,
                duration_seconds=j.duration_seconds,
                created_at=j.created_at,
                completed_at=j.completed_at,
            )
            for j in jobs
        ],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job(job_id: str, db: Session = Depends(get_db)):
    """Get detailed job status."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")

    stages = []
    for log in job.stage_logs:
        stages.append({
            "name": log.stage_name,
            "status": log.status,
            "duration_sec": log.duration_sec,
            "provider_used": log.provider_used,
        })

    return JobDetailResponse(
        job_id=job.id,
        status=job.status,
        current_stage=job.current_stage,
        progress=job.progress or 0,
        target_languages=json.loads(job.target_languages) if job.target_languages else [],
        source_language=job.source_language,
        duration_seconds=job.duration_seconds,
        error_message=job.error_message,
        output_paths=json.loads(job.output_paths) if job.output_paths else None,
        created_at=job.created_at,
        completed_at=job.completed_at,
        segments_count=len(job.segments),
        speakers_detected=len(job.speakers),
        stages=stages,
    )


@router.get("/jobs/{job_id}/output/{lang}")
async def download_output(
    job_id: str,
    lang: str,
    db: Session = Depends(get_db),
):
    """Download the dubbed output file for a specific language."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")

    if job.status != "completed":
        raise HTTPException(400, f"Job is not completed (status: {job.status})")

    output_paths = json.loads(job.output_paths) if job.output_paths else {}
    path = output_paths.get(lang)
    if not path or not Path(path).exists():
        raise HTTPException(404, f"Output not found for language: {lang}")

    return FileResponse(
        path=path,
        filename=Path(path).name,
        media_type="application/octet-stream",
    )


@router.post("/jobs/{job_id}/retry", response_model=JobResponse)
async def retry_job(
    job_id: str,
    from_stage: str | None = None,
    db: Session = Depends(get_db),
):
    """Retry a failed job from the failed stage."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")

    if job.status not in ("failed", "cancelled"):
        raise HTTPException(400, f"Can only retry failed/cancelled jobs (status: {job.status})")

    job.status = "pending"
    job.error_message = None
    job.retry_count = (job.retry_count or 0) + 1
    db.commit()

    try:
        from vaanidub.worker.tasks import process_dubbing_job
        stage_num = 1
        if from_stage:
            from vaanidub.constants import STAGE_NAMES
            if from_stage in STAGE_NAMES:
                stage_num = STAGE_NAMES.index(from_stage) + 1
        process_dubbing_job.delay(job_id, start_from_stage=stage_num)
    except Exception:
        pass

    return JobResponse(
        job_id=job.id,
        status="pending",
        target_languages=json.loads(job.target_languages) if job.target_languages else [],
    )


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, db: Session = Depends(get_db)):
    """Delete a job and its artifacts."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")

    # Delete artifacts from disk
    import shutil
    job_dir = Path(job.input_file_path).parent.parent
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    db.delete(job)
    db.commit()
