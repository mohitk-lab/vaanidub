"""SQLAlchemy ORM models for job tracking."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=_uuid)
    status = Column(String(20), nullable=False, default="pending", index=True)
    current_stage = Column(String(30))
    progress = Column(Float, default=0.0)

    # Input
    input_file_path = Column(Text, nullable=False)
    input_type = Column(String(10))  # "audio" or "video"
    source_language = Column(String(10))
    source_language_confidence = Column(Float)
    target_languages = Column(Text, nullable=False)  # JSON array stored as text
    duration_seconds = Column(Float)

    # Output
    output_paths = Column(Text)  # JSON dict stored as text

    # Error
    error_message = Column(Text)
    error_stage = Column(String(30))
    retry_count = Column(SmallInteger, default=0)

    # Config snapshot
    pipeline_config = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Relationships
    speakers = relationship("Speaker", back_populates="job", cascade="all, delete-orphan")
    segments = relationship("Segment", back_populates="job", cascade="all, delete-orphan")
    stage_logs = relationship("StageLog", back_populates="job", cascade="all, delete-orphan")


class Speaker(Base):
    __tablename__ = "speakers"

    id = Column(String(36), primary_key=True, default=_uuid)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
                    index=True)
    speaker_label = Column(String(50), nullable=False)
    reference_clip_path = Column(Text)
    total_duration_sec = Column(Float)

    job = relationship("Job", back_populates="speakers")


class Segment(Base):
    __tablename__ = "segments"

    id = Column(String(36), primary_key=True, default=_uuid)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    speaker_id = Column(String(36), ForeignKey("speakers.id", ondelete="SET NULL"))
    segment_index = Column(Integer, nullable=False)

    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)

    original_text = Column(Text)
    source_language = Column(String(10))
    translations = Column(Text)  # JSON dict

    # Prosody
    emotion = Column(String(20))
    avg_pitch = Column(Float)
    speaking_rate = Column(Float)
    energy = Column(Float)

    # Synthesis tracking
    dubbed_audio_paths = Column(Text)  # JSON dict

    job = relationship("Job", back_populates="segments")

    __table_args__ = (
        Index("idx_segments_job_order", "job_id", "segment_index"),
    )


class StageLog(Base):
    __tablename__ = "stage_logs"

    id = Column(String(36), primary_key=True, default=_uuid)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
                    index=True)
    stage_name = Column(String(30), nullable=False)
    stage_number = Column(SmallInteger, nullable=False)
    status = Column(String(20), nullable=False)
    provider_used = Column(String(50))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_sec = Column(Float)
    error_message = Column(Text)
    retry_attempt = Column(SmallInteger, default=0)
    metadata_json = Column(Text)  # Additional stage-specific data

    job = relationship("Job", back_populates="stage_logs")
