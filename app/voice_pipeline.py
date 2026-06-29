"""
Voice pipeline: LLM → TTS → audio bytes.

Single mode (default): collect full LLM reply → one TTS request → stream PCM.
Segment mode: stream LLM into TextSegmenter → one TTS request per segment.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.config import settings
from app.debug_probe import probe
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
    tts_request_at: float | None = None
    tts_first_chunk_at: float | None = None
    audio_chunks: int = 0
    audio_bytes: int = 0
    tts_request_count: int = 0
    detected_language: str = ""

    def log(self) -> None:
        elapsed = time.monotonic() - self.start
        if self.first_text_at:
            logger.info("time_to_first_text=%.3f", self.first_text_at - self.start)
        if self.first_audio_at:
            logger.info("time_to_first_audio=%.3f", self.first_audio_at - self.start)
        if self.llm_end_at:
            logger.info("time_to_llm_end=%.3f", self.llm_end_at - self.start)
        if self.tts_request_at:
            logger.info("time_to_tts_request=%.3f", self.tts_request_at - self.start)
        if self.tts_first_chunk_at:
            logger.info("time_to_tts_first_chunk=%.3f", self.tts_first_chunk_at - self.start)
        if self.detected_language:
            logger.info("detected_language=%s", self.detected_language)
        logger.info("tts_request_count=%d", self.tts_request_count)
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

    async def _collect_llm(
        self,
        user_text: str,
        reply_lang: str,
        send_json: Callable[[dict], Awaitable[None]],
        cancel_event: asyncio.Event,
        metrics: PipelineMetrics,
    ) -> str:
        full_text: list[str] = []
        async for chunk in self.gemini.stream(user_text, reply_language=reply_lang):
            if cancel_event.is_set():
                break
            if metrics.first_text_at is None:
                metrics.first_text_at = time.monotonic()
            full_text.append(chunk)
            await send_json(
                {"type": MSG_TEXT, "role": ROLE_LLM, "text": chunk, "partial": True}
            )
        metrics.llm_end_at = time.monotonic()
        return "".join(full_text).strip()

    async def _forward_tts(
        self,
        audio_iter,
        cancel_event: asyncio.Event,
        send_json: Callable[[dict], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        metrics: PipelineMetrics,
    ) -> None:
        buffer_mode = settings.tts_buffer_before_play
        client_buffer = settings.tts_client_buffer_until_end
        pending: list[bytes] = []
        audio_started = False

        async for audio_chunk in audio_iter:
            if cancel_event.is_set():
                break
            if metrics.tts_first_chunk_at is None:
                metrics.tts_first_chunk_at = time.monotonic()
                # region agent log
                probe(
                    "B",
                    "app/voice_pipeline.py:tts_first_chunk",
                    "First PCM chunk from TTS stream",
                    {
                        "bytes": len(audio_chunk),
                        "buffer_mode": buffer_mode,
                        "profile": settings.tts_latency_profile,
                    },
                )
                # endregion
            if buffer_mode:
                pending.append(audio_chunk)
                continue

            if not audio_started:
                audio_started = True
                metrics.first_audio_at = time.monotonic()
                # region agent log
                probe(
                    "D",
                    "app/voice_pipeline.py:first_audio_out",
                    "First audio sent to websocket",
                    {"bytes": len(audio_chunk)},
                )
                # endregion
                await send_json(
                    {
                        "type": MSG_AUDIO,
                        "event": "start",
                        "format": self._audio_format(),
                        "buffer_until_end": client_buffer,
                    }
                )
            metrics.audio_chunks += 1
            metrics.audio_bytes += len(audio_chunk)
            await send_bytes(audio_chunk)

        if cancel_event.is_set():
            return

        if buffer_mode and pending:
            metrics.first_audio_at = time.monotonic()
            # region agent log
            probe(
                "A",
                "app/voice_pipeline.py:buffered_audio_out",
                "Sending full buffered audio after TTS complete",
                {"chunks": len(pending), "bytes": sum(len(c) for c in pending)},
            )
            # endregion
            await send_json(
                {
                    "type": MSG_AUDIO,
                    "event": "start",
                    "format": self._audio_format(),
                    "buffer_until_end": True,
                }
            )
            for chunk in pending:
                metrics.audio_chunks += 1
                metrics.audio_bytes += len(chunk)
                await send_bytes(chunk)

    async def _run_single_mode(
        self,
        user_text: str,
        send_json: Callable[[dict], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        cancel_event: asyncio.Event,
        metrics: PipelineMetrics,
        *,
        language: str,
        speaker_id: str,
        speed: float,
    ) -> list[str]:
        reply_lang = reply_language_label(language)
        reply = await self._collect_llm(
            user_text, reply_lang, send_json, cancel_event, metrics
        )
        if cancel_event.is_set() or not reply:
            return [reply] if reply else []

        self.tts.request_count = 0
        metrics.tts_request_at = time.monotonic()
        # region agent log
        probe(
            "A",
            "app/voice_pipeline.py:tts_request",
            "Starting single-shot TTS",
            {"reply_chars": len(reply), "profile": settings.tts_latency_profile},
        )
        # endregion

        async def _audio():
            async for chunk in self.tts.stream_text(
                reply,
                cancel_event,
                language=language,
                speaker_id=speaker_id,
                speed=speed,
            ):
                yield chunk

        await self._forward_tts(
            _audio(), cancel_event, send_json, send_bytes, metrics
        )
        metrics.tts_request_count = self.tts.request_count
        return [reply]

    async def _run_segment_mode(
        self,
        user_text: str,
        send_json: Callable[[dict], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        cancel_event: asyncio.Event,
        metrics: PipelineMetrics,
        *,
        language: str,
        speaker_id: str,
        speed: float,
    ) -> list[str]:
        reply_lang = reply_language_label(language)
        full_text: list[str] = []

        segmenter = TextSegmenter(
            min_chars=settings.tts_min_chars,
            ideal_max_chars=settings.tts_ideal_max_chars,
            hard_max_chars=settings.tts_hard_max_chars,
            timeout_ms=settings.tts_segment_timeout_ms,
        )

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
                        {
                            "type": MSG_TEXT,
                            "role": ROLE_LLM,
                            "text": chunk,
                            "partial": True,
                        }
                    )
                    await segmenter.push(chunk)
            except Exception:
                logger.exception("LLM task failed")
            finally:
                metrics.llm_end_at = time.monotonic()
                await segmenter.close()

        async def tts_task() -> None:
            self.tts.request_count = 0
            await self._forward_tts(
                self.tts.stream(
                    segmenter.segments(),
                    cancel_event,
                    language=language,
                    speaker_id=speaker_id,
                    speed=speed,
                ),
                cancel_event,
                send_json,
                send_bytes,
                metrics,
            )
            metrics.tts_request_count = self.tts.request_count

        await asyncio.gather(
            asyncio.create_task(llm_task()),
            asyncio.create_task(tts_task()),
        )
        return full_text

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

        if settings.tts_mode == "single":
            full_text = await self._run_single_mode(
                user_text,
                send_json,
                send_bytes,
                cancel_event,
                metrics,
                language=language,
                speaker_id=speaker_id,
                speed=speed,
            )
        else:
            full_text = await self._run_segment_mode(
                user_text,
                send_json,
                send_bytes,
                cancel_event,
                metrics,
                language=language,
                speaker_id=speaker_id,
                speed=speed,
            )

        await send_json({"type": MSG_AUDIO, "event": "end"})

        if full_text:
            await send_json(
                {
                    "type": MSG_TEXT,
                    "role": ROLE_LLM,
                    "text": "".join(full_text),
                }
            )

        metrics.log()
