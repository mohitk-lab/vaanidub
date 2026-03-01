"""Stage 3: Speaker diarization — identify who speaks when using pyannote.audio."""

from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

from vaanidub.constants import MAX_REFERENCE_CLIP_SEC, MIN_REFERENCE_CLIP_SEC
from vaanidub.exceptions import StageError
from vaanidub.pipeline.base import PipelineStage
from vaanidub.pipeline.context import PipelineContext, SpeakerInfo

logger = structlog.get_logger()


class DiarizeStage(PipelineStage):
    name = "diarize"
    stage_number = 3
    timeout_seconds = 300

    def __init__(self, hf_token: str = "", device: str = "cuda"):
        self.hf_token = hf_token
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load pyannote diarization pipeline."""
        if self._pipeline is not None:
            return

        from pyannote.audio import Pipeline

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token,
        )
        self._pipeline.to(self.device)

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.vocals_path is None:
            raise StageError(self.name, "No vocals audio from previous stage")

        self._load_pipeline()
        ctx.report_progress("diarize", 10, "Running speaker diarization")

        # Run diarization
        diarization = self._pipeline(str(ctx.vocals_path))

        # Collect speaker info
        speaker_segments: dict[str, list[dict]] = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append({
                "start": turn.start,
                "end": turn.end,
                "duration": turn.end - turn.start,
            })

        if not speaker_segments:
            raise StageError(self.name, "No speakers detected in audio")

        ctx.report_progress("diarize", 60, f"Found {len(speaker_segments)} speakers")

        # Build SpeakerInfo and extract reference clips
        stage_dir = ctx.stage_dir("diarize")
        audio_data, sr = sf.read(ctx.vocals_path)

        for speaker_label, segments in speaker_segments.items():
            total_dur = sum(s["duration"] for s in segments)

            # Find best reference clip: longest clean segment within bounds
            ref_clip_path = self._extract_reference_clip(
                audio_data, sr, segments, speaker_label, stage_dir
            )

            ctx.speakers[speaker_label] = SpeakerInfo(
                label=speaker_label,
                segments=segments,
                reference_clip_path=ref_clip_path,
                total_duration_sec=total_dur,
            )

        ctx.report_progress("diarize", 100, "Diarization complete")
        logger.info(
            "diarization_complete",
            speakers=len(ctx.speakers),
            details={k: round(v.total_duration_sec, 1) for k, v in ctx.speakers.items()},
        )
        return ctx

    async def validate_output(self, ctx: PipelineContext) -> bool:
        if not ctx.speakers:
            return False
        # Check at least one speaker has reference clip
        return any(s.reference_clip_path is not None for s in ctx.speakers.values())

    def _extract_reference_clip(
        self,
        audio: np.ndarray,
        sr: int,
        segments: list[dict],
        speaker_label: str,
        output_dir: Path,
    ) -> Path | None:
        """Extract the best reference clip for voice cloning."""
        # Sort segments by duration descending
        sorted_segs = sorted(segments, key=lambda s: s["duration"], reverse=True)

        for seg in sorted_segs:
            dur = seg["duration"]
            if dur < MIN_REFERENCE_CLIP_SEC:
                continue

            start_sample = int(seg["start"] * sr)
            # Cap at MAX_REFERENCE_CLIP_SEC
            end_time = min(seg["start"] + MAX_REFERENCE_CLIP_SEC, seg["end"])
            end_sample = int(end_time * sr)

            clip = audio[start_sample:end_sample]

            # Check clip is not too quiet
            rms = np.sqrt(np.mean(clip ** 2))
            if rms < 1e-4:
                continue

            clip_path = output_dir / f"ref_{speaker_label}.wav"
            sf.write(clip_path, clip, sr)
            return clip_path

        # Fallback: use longest segment even if short
        if sorted_segs:
            seg = sorted_segs[0]
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            clip = audio[start_sample:end_sample]
            clip_path = output_dir / f"ref_{speaker_label}.wav"
            sf.write(clip_path, clip, sr)
            return clip_path

        return None
