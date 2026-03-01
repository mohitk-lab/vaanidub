"""Celery tasks wrapping the dubbing pipeline."""

import json
from datetime import datetime
from pathlib import Path

import structlog

from vaanidub.config import AppConfig
from vaanidub.db.models import Job
from vaanidub.db.session import get_session, init_db
from vaanidub.worker.celery_app import celery

logger = structlog.get_logger()


@celery.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def process_dubbing_job(self, job_id: str, start_from_stage: int = 1):
    """Process a dubbing job through the full pipeline."""
    config = AppConfig()
    config.resolve_secrets()
    config.ensure_directories()
    init_db(config)

    session = get_session()
    job = session.query(Job).filter(Job.id == job_id).first()

    if not job:
        logger.error("job_not_found", job_id=job_id)
        return

    try:
        # Update job status
        job.status = "processing"
        job.started_at = datetime.utcnow()
        session.commit()

        # Build pipeline context
        from vaanidub.pipeline.context import PipelineContext
        from vaanidub.pipeline.orchestrator import PipelineOrchestrator

        target_languages = json.loads(job.target_languages)
        job_dir = Path(job.input_file_path).parent.parent

        ctx = PipelineContext(
            job_id=job_id,
            job_dir=job_dir,
            target_languages=target_languages,
            source_language=job.source_language,
            input_file_path=Path(job.input_file_path),
        )

        # Progress callback to update DB
        def on_progress(stage: str, percent: int, message: str = ""):
            job.current_stage = stage
            job.progress = percent / 100.0
            session.commit()
            logger.info("job_progress", job_id=job_id, stage=stage, percent=percent)

        ctx.on_progress = on_progress

        # Run pipeline
        import asyncio
        orchestrator = PipelineOrchestrator(config)
        ctx = asyncio.get_event_loop().run_until_complete(
            orchestrator.run(ctx, start_from_stage=start_from_stage)
        )

        # Update job with results
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.source_language = ctx.source_language
        job.duration_seconds = ctx.media_metadata.get("duration", 0)
        job.output_paths = json.dumps({k: str(v) for k, v in ctx.final_output_paths.items()})
        session.commit()

        logger.info("job_completed", job_id=job_id)

    except Exception as e:
        logger.error("job_failed", job_id=job_id, error=str(e))
        job.status = "failed"
        job.error_message = str(e)[:1000]
        job.error_stage = job.current_stage
        session.commit()

        # Retry if applicable
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

    finally:
        session.close()
