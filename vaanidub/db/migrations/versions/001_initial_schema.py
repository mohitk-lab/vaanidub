"""Initial schema — jobs, speakers, segments, stage_logs.

Revision ID: 001
Revises:
Create Date: 2026-03-01
"""

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending", index=True),
        sa.Column("current_stage", sa.String(30)),
        sa.Column("progress", sa.Float, server_default="0.0"),
        sa.Column("input_file_path", sa.Text, nullable=False),
        sa.Column("input_type", sa.String(10)),
        sa.Column("source_language", sa.String(10)),
        sa.Column("source_language_confidence", sa.Float),
        sa.Column("target_languages", sa.Text, nullable=False),
        sa.Column("duration_seconds", sa.Float),
        sa.Column("output_paths", sa.Text),
        sa.Column("error_message", sa.Text),
        sa.Column("error_stage", sa.String(30)),
        sa.Column("retry_count", sa.SmallInteger, server_default="0"),
        sa.Column("pipeline_config", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("started_at", sa.DateTime),
        sa.Column("completed_at", sa.DateTime),
    )

    op.create_table(
        "speakers",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(36),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("speaker_label", sa.String(50), nullable=False),
        sa.Column("reference_clip_path", sa.Text),
        sa.Column("total_duration_sec", sa.Float),
    )

    op.create_table(
        "segments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(36),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "speaker_id",
            sa.String(36),
            sa.ForeignKey("speakers.id", ondelete="SET NULL"),
        ),
        sa.Column("segment_index", sa.Integer, nullable=False),
        sa.Column("start_time", sa.Float, nullable=False),
        sa.Column("end_time", sa.Float, nullable=False),
        sa.Column("duration", sa.Float, nullable=False),
        sa.Column("original_text", sa.Text),
        sa.Column("source_language", sa.String(10)),
        sa.Column("translations", sa.Text),
        sa.Column("emotion", sa.String(20)),
        sa.Column("avg_pitch", sa.Float),
        sa.Column("speaking_rate", sa.Float),
        sa.Column("energy", sa.Float),
        sa.Column("dubbed_audio_paths", sa.Text),
    )

    op.create_index("idx_segments_job_order", "segments", ["job_id", "segment_index"])

    op.create_table(
        "stage_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(36),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("stage_name", sa.String(30), nullable=False),
        sa.Column("stage_number", sa.SmallInteger, nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("provider_used", sa.String(50)),
        sa.Column("started_at", sa.DateTime),
        sa.Column("completed_at", sa.DateTime),
        sa.Column("duration_sec", sa.Float),
        sa.Column("error_message", sa.Text),
        sa.Column("retry_attempt", sa.SmallInteger, server_default="0"),
        sa.Column("metadata_json", sa.Text),
    )


def downgrade() -> None:
    op.drop_table("stage_logs")
    op.drop_index("idx_segments_job_order", table_name="segments")
    op.drop_table("segments")
    op.drop_table("speakers")
    op.drop_table("jobs")
