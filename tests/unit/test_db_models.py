"""Tests for database models."""

from vaanidub.db.models import Base, Job, Segment, Speaker, StageLog


class TestJobModel:
    def test_job_column_defaults_defined(self):
        """Verify column defaults are set in the schema (applied on DB flush)."""
        status_col = Job.__table__.columns["status"]
        progress_col = Job.__table__.columns["progress"]
        retry_col = Job.__table__.columns["retry_count"]
        assert status_col.default.arg == "pending"
        assert progress_col.default.arg == 0.0
        assert retry_col.default.arg == 0

    def test_job_table_name(self):
        assert Job.__tablename__ == "jobs"


class TestSpeakerModel:
    def test_speaker_table_name(self):
        assert Speaker.__tablename__ == "speakers"


class TestSegmentModel:
    def test_segment_table_name(self):
        assert Segment.__tablename__ == "segments"


class TestStageLogModel:
    def test_stagelog_table_name(self):
        assert StageLog.__tablename__ == "stage_logs"


class TestBaseMetadata:
    def test_all_tables_defined(self):
        table_names = set(Base.metadata.tables.keys())
        assert "jobs" in table_names
        assert "speakers" in table_names
        assert "segments" in table_names
        assert "stage_logs" in table_names
