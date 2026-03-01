"""Pipeline orchestrator — coordinates all 8 stages with checkpoint/resume support."""

import json
import time
from pathlib import Path

import structlog

from vaanidub.config import AppConfig
from vaanidub.exceptions import PipelineError, StageError
from vaanidub.pipeline.base import PipelineStage
from vaanidub.pipeline.context import PipelineContext
from vaanidub.pipeline.stages.s1_ingest import IngestStage
from vaanidub.pipeline.stages.s2_separate import SeparateStage
from vaanidub.pipeline.stages.s3_diarize import DiarizeStage
from vaanidub.pipeline.stages.s4_transcribe import TranscribeStage
from vaanidub.pipeline.stages.s5_prosody import ProsodyStage
from vaanidub.pipeline.stages.s6_translate import TranslateStage
from vaanidub.pipeline.stages.s7_synthesize import SynthesizeStage
from vaanidub.pipeline.stages.s8_mixdown import MixdownStage

logger = structlog.get_logger()


class PipelineOrchestrator:
    """Runs the full dubbing pipeline with checkpoint/resume capability."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.stages: list[PipelineStage] = self._build_stages(config)

    def _build_stages(self, config: AppConfig) -> list[PipelineStage]:
        """Build all pipeline stages from config."""
        return [
            IngestStage(),
            SeparateStage(),
            DiarizeStage(
                hf_token=config.hf_token,
                device=config.gpu.device,
            ),
            TranscribeStage(
                model_size=config.providers.stt.model_size,
                device=config.gpu.device,
                compute_type=config.providers.stt.compute_type,
                hf_token=config.hf_token,
            ),
            ProsodyStage(),
            TranslateStage(
                device=config.gpu.device,
                google_api_key=config.google_translate_api_key,
            ),
            SynthesizeStage(
                device=config.gpu.device,
                elevenlabs_api_key=config.elevenlabs_api_key,
                hf_token=config.hf_token,
            ),
            MixdownStage(),
        ]

    async def run(
        self,
        ctx: PipelineContext,
        start_from_stage: int = 1,
    ) -> PipelineContext:
        """
        Execute the full pipeline.

        Args:
            ctx: Pipeline context with input file and target languages.
            start_from_stage: Resume from this stage number (1-8).
        """
        log = logger.bind(job_id=ctx.job_id)
        log.info("pipeline_start", target_languages=ctx.target_languages,
                 start_stage=start_from_stage)

        total_start = time.monotonic()

        for stage in self.stages:
            if stage.stage_number < start_from_stage:
                log.info("stage_skipped", stage=stage.name,
                         reason="before start_from_stage")
                continue

            ctx.current_stage = stage.stage_number

            # Run with retries
            last_error = None
            for attempt in range(1, stage.retry_count + 1):
                try:
                    ctx = await stage.run(ctx)
                    last_error = None
                    break
                except StageError as e:
                    last_error = e
                    if not e.retriable or attempt == stage.retry_count:
                        break
                    log.warning(
                        "stage_retry",
                        stage=stage.name,
                        attempt=attempt,
                        error=str(e),
                    )
                except Exception as e:
                    last_error = e
                    break

            if last_error is not None:
                ctx.errors.append({
                    "stage": stage.name,
                    "error": str(last_error),
                })
                log.error("pipeline_failed", stage=stage.name, error=str(last_error))
                raise PipelineError(
                    f"Pipeline failed at stage '{stage.name}': {last_error}"
                ) from last_error

            # Save checkpoint after each successful stage
            self._save_checkpoint(ctx)

        total_elapsed = time.monotonic() - total_start
        log.info(
            "pipeline_complete",
            total_duration_sec=round(total_elapsed, 2),
            stage_timings=ctx.stage_timings,
            outputs=list(ctx.final_output_paths.keys()),
        )
        return ctx

    def _save_checkpoint(self, ctx: PipelineContext) -> None:
        """Save pipeline state checkpoint for resume capability."""
        checkpoint_path = ctx.job_dir / "checkpoint.json"
        checkpoint = {
            "job_id": ctx.job_id,
            "current_stage": ctx.current_stage,
            "source_language": ctx.source_language,
            "target_languages": ctx.target_languages,
            "stage_timings": ctx.stage_timings,
            "segments_count": len(ctx.segments),
            "speakers_count": len(ctx.speakers),
            "final_outputs": {k: str(v) for k, v in ctx.final_output_paths.items()},
        }
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2))

    @staticmethod
    def load_checkpoint(job_dir: Path) -> dict | None:
        """Load checkpoint from a previous run."""
        checkpoint_path = job_dir / "checkpoint.json"
        if checkpoint_path.exists():
            return json.loads(checkpoint_path.read_text())
        return None
