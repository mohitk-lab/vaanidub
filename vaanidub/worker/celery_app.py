"""Celery application configuration."""

from celery import Celery

from vaanidub.config import AppConfig
from vaanidub.logging_config import setup_logging

config = AppConfig()
setup_logging(config.log_level, getattr(config, "log_format", "json"))

celery = Celery(
    "vaanidub",
    broker=config.redis.url,
    backend=config.redis.url,
    include=["vaanidub.worker.tasks"],
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,            # Hard kill after 1 hour
    task_soft_time_limit=3000,       # Soft limit at 50 minutes
    worker_concurrency=1,            # 1 per GPU
    worker_prefetch_multiplier=1,    # Don't prefetch extra tasks
    task_acks_late=True,             # Ack after completion
    task_reject_on_worker_lost=True, # Re-queue if worker crashes
)
