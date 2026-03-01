"""Abstract base class for all pipeline stages."""

import time
from abc import ABC, abstractmethod

import structlog

from vaanidub.pipeline.context import PipelineContext

logger = structlog.get_logger()


class PipelineStage(ABC):
    """
    Abstract base class for a dubbing pipeline stage.

    Each stage reads from PipelineContext, performs work, writes results
    back to PipelineContext, and validates its output.
    """

    name: str
    stage_number: int
    retry_count: int = 3
    timeout_seconds: int = 600

    @abstractmethod
    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this pipeline stage and return updated context."""
        ...

    @abstractmethod
    async def validate_output(self, ctx: PipelineContext) -> bool:
        """Validate that this stage produced acceptable output."""
        ...

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Run the stage with timing and logging."""
        log = logger.bind(stage=self.name, stage_number=self.stage_number, job_id=ctx.job_id)
        log.info("stage_start")

        start = time.monotonic()
        ctx = await self.execute(ctx)
        elapsed = time.monotonic() - start

        ctx.stage_timings[self.name] = elapsed
        log.info("stage_complete", duration_sec=round(elapsed, 2))

        if not await self.validate_output(ctx):
            log.error("stage_validation_failed")
            from vaanidub.exceptions import StageValidationError
            raise StageValidationError(self.name)

        return ctx

    def rollback(self, ctx: PipelineContext) -> None:
        """Optional cleanup if stage fails. Override in subclasses."""
        pass
