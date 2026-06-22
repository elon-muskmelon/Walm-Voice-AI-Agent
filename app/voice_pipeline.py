"""
Voice pipeline: LLM → text segmentation → TTS → audio bytes.

Runs two concurrent async tasks:
  1. llm_task  — streams Gemini tokens into the TextSegmenter
  2. tts_task  — consumes segmented text from TTS, yields audio bytes directly

Audio bytes are forwarded to the caller's *send_bytes* callback without
any intermediate queue or artificial delay.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.config import settings
from app.gemini_streamer import GeminiStreamer
from app.language_router import reply_language_label
from app.protocol import MSG_AUDIO, MSG_TEXT, ROLE_LLM
from app.text_segmenter import TextSegmenter
from app.tts_base import TTSStreamer

logger = logging.getLogger("voice_pipeline")


@dataclass
class PipelineMetrics:
    start: float = field(default_factory=time.monotonic)
    first_text_at: float | None = None
    first_audio_at: float | None = None
    llm_end_at: float | None = None
    audio_chunks: int = 0
    audio_bytes: int = 0
    detected_language: str = ""

    def log(self) -> None:
        elapsed = time.monotonic() - self.start
        if self.first_text_at:
            logger.info("time_to_first_text=%.3f", self.first_text_at - self.start)
        if self.first_audio_at:
            logger.info("time_to_first_audio=%.3f", self.first_audio_at - self.start)
        if self.llm_end_at:
            logger.info("time_to_llm_end=%.3f", self.llm_end_at - self.start)
        if self.detected_language:
            logger.info("detected_language=%s", self.detected_language)
        logger.info("tts_audio_chunks=%d", self.audio_chunks)
        logger.info("tts_audio_bytes=%d", self.audio_bytes)
        logger.info("total_pipeline_time=%.3f", elapsed)


class VoicePipeline:
    def __init__(
        self,
        gemini: GeminiStreamer,
        tts: TTSStreamer,
    ) -> None:
        self.gemini = gemini
        self.tts = tts

    def _audio_format(self) -> str:
        if settings.tts_backend == "custom":
            return f"audio/pcm;rate={settings.tts_sample_rate};encoding=s16le"
        return "audio/mpeg"

    async def run(
        self,
        user_text: str,
        send_json: Callable[[dict], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        cancel_event: asyncio.Event,
        *,
        language: str = "hindi",
        speaker_id: str = "159",
        speed: float = 1.05,
    ) -> None:
        metrics = PipelineMetrics(detected_language=language)
        full_text: list[str] = []
        reply_lang = reply_language_label(language)

        segmenter = TextSegmenter(
            min_chars=settings.tts_min_chars,
            timeout_ms=settings.tts_segment_timeout_ms,
        )

        # ── Task 1: stream LLM tokens → text segmenter ────────────────────
        async def llm_task() -> None:
            try:
                async for chunk in self.gemini.stream(
                    user_text, reply_language=reply_lang
                ):
                    if cancel_event.is_set():
                        break
                    if metrics.first_text_at is None:
                        metrics.first_text_at = time.monotonic()
                    full_text.append(chunk)
                    await send_json(
                        {"type": MSG_TEXT, "role": ROLE_LLM, "text": chunk, "partial": True}
                    )
                    await segmenter.push(chunk)
            except Exception:
                logger.exception("LLM task failed")
            finally:
                metrics.llm_end_at = time.monotonic()
                await segmenter.close()

        # ── Task 2: consume segments → TTS → forward audio bytes ──────────
        async def tts_task() -> None:
            try:
                async for audio_chunk in self.tts.stream(
                    segmenter.segments(),
                    cancel_event,
                    language=language,
                    speaker_id=speaker_id,
                    speed=speed,
                ):
                    if cancel_event.is_set():
                        break
                    if metrics.first_audio_at is None:
                        metrics.first_audio_at = time.monotonic()
                        await send_json(
                            {
                                "type": MSG_AUDIO,
                                "event": "start",
                                "format": self._audio_format(),
                            }
                        )
                    metrics.audio_chunks += 1
                    metrics.audio_bytes += len(audio_chunk)
                    await send_bytes(audio_chunk)
            except Exception:
                logger.exception("TTS task failed")

        await asyncio.gather(
            asyncio.create_task(llm_task()),
            asyncio.create_task(tts_task()),
        )

        await send_json({"type": MSG_AUDIO, "event": "end"})

        if full_text:
            await send_json(
                {"type": MSG_TEXT, "role": ROLE_LLM, "text": "".join(full_text)}
            )

        metrics.log()
