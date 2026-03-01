"""Stage 5: Prosody analysis — extract pitch, pace, energy, and emotion per segment."""

import numpy as np
import soundfile as sf
import structlog

from vaanidub.exceptions import StageError
from vaanidub.pipeline.base import PipelineStage
from vaanidub.pipeline.context import PipelineContext

logger = structlog.get_logger()


class ProsodyStage(PipelineStage):
    name = "prosody"
    stage_number = 5
    timeout_seconds = 120

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.vocals_path is None or not ctx.segments:
            raise StageError(self.name, "No vocals or segments from previous stages")

        import parselmouth

        ctx.report_progress("prosody", 10, "Analyzing prosody features")

        # Load full vocals audio
        audio, sr = sf.read(ctx.vocals_path)

        total = len(ctx.segments)
        for i, segment in enumerate(ctx.segments):
            # Extract segment audio
            start_sample = int(segment.start_time * sr)
            end_sample = int(segment.end_time * sr)
            seg_audio = audio[start_sample:end_sample]

            if len(seg_audio) < sr * 0.1:  # Skip very short segments (<100ms)
                continue

            # Pitch analysis using Parselmouth (Praat)
            try:
                snd = parselmouth.Sound(seg_audio, sampling_frequency=sr)
                pitch_obj = snd.to_pitch()
                pitch_values = pitch_obj.selected_array["frequency"]
                pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced

                if len(pitch_values) > 0:
                    segment.avg_pitch = float(np.mean(pitch_values))
            except Exception:
                segment.avg_pitch = 0.0

            # Energy (RMS)
            segment.energy = float(np.sqrt(np.mean(seg_audio ** 2)))

            # Speaking rate (approximate: words per second)
            word_count = len(segment.text.split()) if segment.text else 0
            duration = segment.duration
            if duration > 0:
                segment.speaking_rate = word_count / duration

            # Emotion classification (simple heuristic based on pitch + energy)
            segment.emotion = self._classify_emotion(
                segment.avg_pitch, segment.energy, segment.speaking_rate
            )

            if (i + 1) % 10 == 0 or i == total - 1:
                pct = 10 + int(90 * (i + 1) / total)
                ctx.report_progress("prosody", pct, f"Analyzed {i+1}/{total} segments")

        ctx.report_progress("prosody", 100, "Prosody analysis complete")
        logger.info("prosody_complete", segments_analyzed=total)
        return ctx

    async def validate_output(self, ctx: PipelineContext) -> bool:
        # Check at least some segments have prosody data
        analyzed = sum(1 for s in ctx.segments if s.avg_pitch > 0 or s.energy > 0)
        if analyzed == 0 and len(ctx.segments) > 0:
            logger.warning("no_prosody_data_extracted")
            # Don't fail hard — prosody is nice-to-have
        return True

    def _classify_emotion(self, pitch: float, energy: float, rate: float) -> str:
        """Simple rule-based emotion classification.

        This is a rough heuristic. For production, consider a trained classifier.
        """
        if pitch == 0 and energy == 0:
            return "neutral"

        # High pitch + high energy + fast = excited/happy
        if pitch > 250 and energy > 0.05 and rate > 3.0:
            return "excited"

        # High pitch + high energy = happy
        if pitch > 200 and energy > 0.03:
            return "happy"

        # Low pitch + low energy + slow = sad
        if pitch < 150 and energy < 0.02 and rate < 2.0:
            return "sad"

        # High energy + fast = angry
        if energy > 0.06 and rate > 3.5:
            return "angry"

        return "neutral"
