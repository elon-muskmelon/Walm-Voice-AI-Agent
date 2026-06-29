"""
FastAPI application entry point.

WebSocket flow per client:
  browser ──PCM──► VAD ──utterance──► STT commit ──text──► VoicePipeline
                                                            ├─ LLM tokens ──► browser
                                                            └─ TTS audio  ──► browser
"""
from __future__ import annotations

import asyncio
import logging
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.websockets import WebSocketState

from app.config import settings
from app.elevenlabs_stt_streamer import (
    ElevenLabsSTTSession,
    STTEvent,
    STTEventType,
)
from app.gemini_streamer import GeminiStreamer
from app.language_router import resolve_voice
from app.protocol import (
    MSG_AUDIO,
    MSG_START,
    MSG_STOP,
    MSG_TEXT,
    ROLE_STT_FINAL,
    ROLE_STT_PARTIAL,
    dumps,
    parse,
)
from app.tts_base import TTSStreamer
from app.tts_factory import create_tts_streamer
from app.utterance_segmenter import SegmenterConfig, UtteranceSegmenter
from app.vad import VADConfig, get_shared_vad, init_shared_vad
from app.voice_pipeline import VoicePipeline

AUDIO_QUEUE_MAX = 200

logger = logging.getLogger("main")

# ── Shared singletons (created in lifespan) ───────────────────────────────────
_tts: Optional[TTSStreamer] = None


# ── Application lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    vad_config = VADConfig(
        sample_rate=settings.audio_sample_rate,
        threshold=settings.vad_speech_threshold,
    )
    model_path = Path(settings.vad_model_path) if settings.vad_model_path else None
    init_shared_vad(vad_config, model_path)

    global _tts
    _tts = create_tts_streamer()

    yield


app = FastAPI(lifespan=lifespan)


# ── Security headers ──────────────────────────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-eval'; "
            "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' ws: wss:; "
            "media-src 'self' blob:; "
            "img-src 'self' data:"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)


# ── Static files & index ──────────────────────────────────────────────────────
WEB_DIR = Path(__file__).resolve().parent.parent / "client" / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


# ── Per-session state ─────────────────────────────────────────────────────────
class SessionState:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=AUDIO_QUEUE_MAX
        )
        self.utterance_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.mic_active = False
        self.closed = False
        # Serialise STT commits so we never send two simultaneous commits
        self.commit_lock = asyncio.Lock()
        self.voice_task: Optional[asyncio.Task] = None
        self.voice_cancel: Optional[asyncio.Event] = None
        self._last_partial = ""

    async def send_json(self, payload: dict) -> None:
        if self.closed or self.ws.application_state != WebSocketState.CONNECTED:
            return
        try:
            await self.ws.send_text(dumps(payload))
        except Exception:
            pass


# ── Debug WAV helper ──────────────────────────────────────────────────────────
def _save_debug_wav(pcm_bytes: bytes, path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(settings.audio_sample_rate)
        wav.writeframes(pcm_bytes)


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    state = SessionState(ws)

    assert _tts is not None
    gemini = GeminiStreamer()
    pipeline = VoicePipeline(gemini, _tts)

    segmenter_config = SegmenterConfig(
        sample_rate=settings.audio_sample_rate,
        frame_ms=settings.vad_frame_ms,
        pre_speech_ms=settings.vad_pre_speech_ms,
        min_speech_ms=settings.vad_min_speech_ms,
        end_silence_ms=settings.vad_end_silence_ms,
        max_utterance_ms=settings.vad_max_utterance_ms,
        start_trigger_frames=settings.vad_start_trigger_frames,
        speech_threshold=settings.vad_speech_threshold,
        trailing_silence_keep_ms=settings.vad_trailing_silence_keep_ms,
    )
    segmenter = UtteranceSegmenter(segmenter_config)
    frame_bytes = segmenter.frame_bytes
    frame_buffer = bytearray()

    # ── STT session ───────────────────────────────────────────────────────
    stt_session = ElevenLabsSTTSession(
        api_key=settings.elevenlabs_api_key,
        model_id=settings.elevenlabs_stt_model_id,
        sample_rate=settings.audio_sample_rate,
        language_code=settings.elevenlabs_stt_language_code or None,
    )
    try:
        await stt_session.connect()
    except Exception as exc:
        await state.send_json({"type": "error", "message": f"STT connect failed: {exc}"})
        await ws.close()
        return

    # ── Pipeline helpers ──────────────────────────────────────────────────
    async def _cancel_pipeline(barge_in: bool = False) -> None:
        if state.voice_task and not state.voice_task.done():
            if state.voice_cancel:
                state.voice_cancel.set()
            if barge_in:
                await state.send_json({"type": MSG_AUDIO, "event": "barge"})
            try:
                await state.voice_task
            except (asyncio.CancelledError, Exception):
                pass
        state.voice_task = None
        state.voice_cancel = None

    async def _start_pipeline(text: str, stt_language: str | None = None) -> None:
        await _cancel_pipeline(barge_in=False)
        cancel_ev = asyncio.Event()
        state.voice_cancel = cancel_ev

        language, speaker_id, speed = resolve_voice(
            text,
            stt_language,
            default_language=settings.tts_default_language,
            default_speaker=settings.tts_default_speaker,
        )
        logger.info(
            "voice_route language=%s speaker=%s speed=%.2f stt_lang=%r transcript=%r",
            language,
            speaker_id,
            speed,
            stt_language,
            text[:80],
        )

        async def _send_audio(chunk: bytes) -> None:
            if state.closed or state.ws.application_state != WebSocketState.CONNECTED:
                return
            await state.ws.send_bytes(chunk)

        async def _run() -> None:
            try:
                await pipeline.run(
                    text,
                    state.send_json,
                    _send_audio,
                    cancel_ev,
                    language=language,
                    speaker_id=speaker_id,
                    speed=speed,
                )
            finally:
                if not state.closed:
                    await state.send_json({"type": "ready"})

        state.voice_task = asyncio.create_task(_run())

    # ── STT event forwarder ───────────────────────────────────────────────
    async def _stt_event_forwarder() -> None:
        try:
            while not state.closed:
                event: STTEvent = await stt_session.next_event()
                if event.type == STTEventType.PARTIAL and event.text:
                    if event.text != state._last_partial:
                        state._last_partial = event.text
                        await state.send_json(
                            {"type": MSG_TEXT, "role": ROLE_STT_PARTIAL, "text": event.text}
                        )
                elif event.type == STTEventType.ERROR and event.error:
                    if event.error != "session_closed":
                        logger.warning("stt_error=%s", event.error)
        except asyncio.CancelledError:
            pass

    # ── Utterance processor ───────────────────────────────────────────────
    async def _process_utterance(utterance: bytes) -> None:
        endpoint_ts = time.monotonic()
        await state.send_json({"type": "vad", "event": "endpoint"})

        if settings.vad_debug_save:
            debug_dir = Path(settings.vad_debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            _save_debug_wav(utterance, debug_dir / f"utt_{int(time.time() * 1000)}.wav")

        async with state.commit_lock:
            commit_start = time.monotonic()
            try:
                committed = await asyncio.wait_for(
                    stt_session.commit(),
                    timeout=settings.stt_commit_timeout_s,
                )
            except asyncio.TimeoutError:
                await state.send_json({"type": "error", "message": "STT commit timed out"})
                return
            except Exception as exc:
                await state.send_json({"type": "error", "message": f"STT commit failed: {exc}"})
                return

            logger.info(
                "stt_commit_seconds=%.3f vad_to_commit=%.3f",
                time.monotonic() - commit_start,
                time.monotonic() - endpoint_ts,
            )

        state._last_partial = ""
        text = committed.text.strip()
        if not text:
            await state.send_json({"type": "ready"})
            return

        await state.send_json(
            {
                "type": MSG_TEXT,
                "role": ROLE_STT_FINAL,
                "text": text,
                "language": committed.language_code,
            }
        )
        await _start_pipeline(text, committed.language_code)

    async def _utterance_worker() -> None:
        try:
            while True:
                utterance = await state.utterance_queue.get()
                if utterance is None:
                    break
                if state.voice_task and not state.voice_task.done():
                    logger.info("utterance_dropped voice_pipeline_busy")
                    continue
                await _process_utterance(utterance)
        except Exception as exc:
            await state.send_json({"type": "error", "message": f"STT error: {exc}"})

    # ── VAD frame processor ───────────────────────────────────────────────
    async def _frame_processor() -> None:
        try:
            while True:
                pcm = await state.audio_queue.get()
                if pcm is None:
                    break
                if not state.mic_active:
                    frame_buffer.clear()
                    segmenter.reset()
                    continue

                frame_buffer.extend(pcm)
                while len(frame_buffer) >= frame_bytes:
                    frame = bytes(frame_buffer[:frame_bytes])
                    del frame_buffer[:frame_bytes]
                    prob = get_shared_vad().speech_probability(frame)
                    prev_vad_state = segmenter.state
                    utterance = segmenter.process_frame(frame, prob)

                    # On speech start: barge-in if assistant is speaking
                    if prev_vad_state == "IDLE" and segmenter.state == "SPEECH":
                        if state.voice_task and not state.voice_task.done():
                            asyncio.create_task(_cancel_pipeline(barge_in=True))
                        state._last_partial = ""
                        if stt_session.is_connected:
                            try:
                                pre = list(segmenter.pre_buffer)
                                for pre_frame in pre[:-1]:
                                    await stt_session.send_pcm(pre_frame)
                            except Exception as exc:
                                logger.warning("stt pre-buffer send failed: %s", exc)

                    # Forward live speech frames to STT (skip while assistant pipeline runs)
                    pipeline_busy = (
                        state.voice_task is not None and not state.voice_task.done()
                    )
                    if (
                        segmenter.state == "SPEECH"
                        and stt_session.is_connected
                        and not pipeline_busy
                    ):
                        try:
                            await stt_session.send_pcm(frame)
                        except Exception as exc:
                            logger.warning("stt send_pcm failed: %s", exc)

                    if utterance:
                        state.mic_active = False
                        frame_buffer.clear()
                        segmenter.reset()
                        await state.utterance_queue.put(utterance)
        except Exception as exc:
            await state.send_json({"type": "error", "message": f"VAD error: {exc}"})

    # ── Start background tasks ─────────────────────────────────────────────
    frame_task = asyncio.create_task(_frame_processor())
    utterance_task = asyncio.create_task(_utterance_worker())
    stt_task = asyncio.create_task(_stt_event_forwarder())

    # ── Main receive loop ──────────────────────────────────────────────────
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                payload = parse(message["text"])
                msg_type = payload.get("type")

                if msg_type == MSG_START:
                    state.mic_active = True
                    if not stt_session.is_connected:
                        try:
                            await stt_session.connect()
                        except Exception as exc:
                            logger.error("STT reconnect failed: %s", exc)
                    await state.send_json({"type": "ready"})

                elif msg_type == MSG_STOP:
                    state.mic_active = False
                    frame_buffer.clear()
                    segmenter.reset()

            if "bytes" in message and message["bytes"]:
                if not state.mic_active:
                    continue
                try:
                    state.audio_queue.put_nowait(message["bytes"])
                except asyncio.QueueFull:
                    logger.warning("audio_queue full, dropping chunk")

    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        state.closed = True
        await _cancel_pipeline()
        await stt_session.close()
        await state.audio_queue.put(None)
        await state.utterance_queue.put(None)
        stt_task.cancel()
        for task in (frame_task, utterance_task, stt_task):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
