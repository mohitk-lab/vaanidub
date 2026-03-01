"""Tests for Pipeline Orchestrator."""

import json

import pytest

from vaanidub.exceptions import PipelineError, StageError
from vaanidub.pipeline.base import PipelineStage
from vaanidub.pipeline.context import PipelineContext
from vaanidub.pipeline.orchestrator import PipelineOrchestrator


class MockStage(PipelineStage):
    """A mock pipeline stage for testing."""

    def __init__(self, name: str, number: int, should_fail: bool = False):
        self.name = name
        self.stage_number = number
        self.should_fail = should_fail
        self.executed = False

    async def execute(self, ctx):
        self.executed = True
        if self.should_fail:
            raise StageError(self.name, "Mock failure", retriable=False)
        return ctx

    async def validate_output(self, ctx):
        return True


class TestPipelineOrchestrator:
    @pytest.mark.asyncio
    async def test_runs_all_stages(self, tmp_dir, app_config):
        orchestrator = PipelineOrchestrator(app_config)
        stages = [
            MockStage("stage1", 1),
            MockStage("stage2", 2),
            MockStage("stage3", 3),
        ]
        orchestrator.stages = stages

        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )

        await orchestrator.run(ctx)
        assert all(s.executed for s in stages)

    @pytest.mark.asyncio
    async def test_skips_stages_before_start(self, tmp_dir, app_config):
        orchestrator = PipelineOrchestrator(app_config)
        stages = [
            MockStage("stage1", 1),
            MockStage("stage2", 2),
            MockStage("stage3", 3),
        ]
        orchestrator.stages = stages

        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )

        await orchestrator.run(ctx, start_from_stage=2)
        assert stages[0].executed is False
        assert stages[1].executed is True
        assert stages[2].executed is True

    @pytest.mark.asyncio
    async def test_stops_on_failure(self, tmp_dir, app_config):
        orchestrator = PipelineOrchestrator(app_config)
        stages = [
            MockStage("stage1", 1),
            MockStage("stage2", 2, should_fail=True),
            MockStage("stage3", 3),
        ]
        orchestrator.stages = stages

        ctx = PipelineContext(
            job_id="test", job_dir=tmp_dir, target_languages=["hi"]
        )

        with pytest.raises(PipelineError, match="stage2"):
            await orchestrator.run(ctx)

        assert stages[0].executed is True
        assert stages[1].executed is True
        assert stages[2].executed is False

    @pytest.mark.asyncio
    async def test_saves_checkpoint(self, tmp_dir, app_config):
        orchestrator = PipelineOrchestrator(app_config)
        orchestrator.stages = [MockStage("stage1", 1)]

        ctx = PipelineContext(
            job_id="test-cp", job_dir=tmp_dir, target_languages=["hi", "ta"]
        )

        await orchestrator.run(ctx)

        checkpoint_path = tmp_dir / "checkpoint.json"
        assert checkpoint_path.exists()

        data = json.loads(checkpoint_path.read_text())
        assert data["job_id"] == "test-cp"
        assert data["target_languages"] == ["hi", "ta"]

    def test_load_checkpoint_exists(self, tmp_dir):
        checkpoint_path = tmp_dir / "checkpoint.json"
        checkpoint_path.write_text(json.dumps({"job_id": "abc", "current_stage": 3}))

        result = PipelineOrchestrator.load_checkpoint(tmp_dir)
        assert result is not None
        assert result["job_id"] == "abc"

    def test_load_checkpoint_missing(self, tmp_dir):
        result = PipelineOrchestrator.load_checkpoint(tmp_dir)
        assert result is None
