"""Stage 4: Speech-to-Text — transcribe with word timestamps using WhisperX / faster-whisper."""

import structlog

from vaanidub.exceptions import StageError
from vaanidub.pipeline.base import PipelineStage
from vaanidub.pipeline.context import PipelineContext, Segment

logger = structlog.get_logger()


class TranscribeStage(PipelineStage):
    name = "transcribe"
    stage_number = 4
    timeout_seconds = 600

    def __init__(self, model_size: str = "large-v2", device: str = "cuda",
                 compute_type: str = "float16", hf_token: str = ""):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.hf_token = hf_token
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        import whisperx
        self._model = whisperx.load_model(
            self.model_size,
            self.device,
            compute_type=self.compute_type,
        )

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.vocals_path is None:
            raise StageError(self.name, "No vocals audio from previous stage")

        import whisperx

        self._load_model()
        ctx.report_progress("transcribe", 10, "Loading audio for transcription")

        # Load audio
        audio = whisperx.load_audio(str(ctx.vocals_path))

        # Transcribe
        ctx.report_progress("transcribe", 20, "Transcribing speech")
        result = self._model.transcribe(audio, batch_size=16)

        # Language detection
        detected_lang = result.get("language", "en")
        ctx.source_language = detected_lang
        ctx.source_language_confidence = result.get("language_probability", 0.0)
        logger.info("language_detected", language=detected_lang,
                     confidence=ctx.source_language_confidence)

        ctx.report_progress("transcribe", 50, f"Detected language: {detected_lang}")

        # Align whisper output for word-level timestamps
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_lang, device=self.device
        )
        result = whisperx.align(
            result["segments"], align_model, align_metadata,
            audio, self.device, return_char_alignments=False
        )

        ctx.report_progress("transcribe", 70, "Aligning word timestamps")

        # Assign speakers using diarization from Stage 3
        if ctx.speakers:
            diarize_result = self._build_diarize_segments(ctx)
            result = whisperx.assign_word_speakers(diarize_result, result)

        # Build segment objects
        ctx.segments = []
        for i, seg in enumerate(result.get("segments", [])):
            speaker = seg.get("speaker", "SPEAKER_00")
            ctx.segments.append(Segment(
                index=i,
                speaker_label=speaker,
                start_time=seg["start"],
                end_time=seg["end"],
                text=seg.get("text", "").strip(),
            ))

        ctx.report_progress("transcribe", 100, f"Transcribed {len(ctx.segments)} segments")
        logger.info("transcription_complete", segments=len(ctx.segments),
                     language=detected_lang)
        return ctx

    async def validate_output(self, ctx: PipelineContext) -> bool:
        if not ctx.segments:
            logger.error("no_segments_transcribed")
            return False
        if ctx.source_language is None:
            return False
        # Check at least some segments have text
        non_empty = sum(1 for s in ctx.segments if s.text)
        if non_empty == 0:
            logger.error("all_segments_empty")
            return False
        return True

    def _build_diarize_segments(self, ctx: PipelineContext) -> dict:
        """Convert our SpeakerInfo into the format WhisperX expects for speaker assignment."""
        import pandas as pd

        rows = []
        for speaker_label, speaker_info in ctx.speakers.items():
            for seg in speaker_info.segments:
                rows.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": speaker_label,
                })

        df = pd.DataFrame(rows)
        return {"segments": df.to_dict("records")} if not df.empty else {"segments": []}
