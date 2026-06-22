import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlencode

import websockets


class STTEventType(str, Enum):
    SESSION_STARTED = "session_started"
    PARTIAL = "partial"
    COMMITTED = "committed"
    ERROR = "error"


@dataclass
class CommittedTranscript:
    text: str
    language_code: Optional[str] = None


@dataclass
class STTEvent:
    type: STTEventType
    text: str = ""
    language_code: Optional[str] = None
    error: Optional[str] = None


class ElevenLabsSTTSession:
    """ElevenLabs Scribe v2 Realtime WebSocket session (manual commit)."""

    def __init__(
        self,
        api_key: str,
        model_id: str,
        sample_rate: int,
        language_code: Optional[str] = None,
        base_url: str = "wss://api.elevenlabs.io",
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.language_code = language_code or None
        self.base_url = base_url.rstrip("/")
        self.logger = logging.getLogger("elevenlabs_stt_streamer")
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._events: asyncio.Queue[STTEvent] = asyncio.Queue()
        self._commit_future: Optional[asyncio.Future[CommittedTranscript]] = None
        self._connected = asyncio.Event()
        self._closed = False

    @property
    def is_connected(self) -> bool:
        if self._ws is None or self._closed:
            return False
        if self._reader_task and self._reader_task.done():
            return False
        import websockets.protocol
        return self._ws.state == websockets.protocol.State.OPEN and self._connected.is_set()

    async def connect(self) -> None:
        if self.is_connected:
            return
        if self._ws is not None:
            await self.close()
            self._closed = False
            self._connected.clear()
            self._events = asyncio.Queue()
        params = {
            "model_id": self.model_id,
            "audio_format": "pcm_16000",
            "commit_strategy": "manual",
        }
        if self.language_code:
            params["language_code"] = self.language_code
        url = f"{self.base_url}/v1/speech-to-text/realtime?{urlencode(params)}"
        headers = {"xi-api-key": self.api_key}
        self._ws = await websockets.connect(url, additional_headers=headers, max_queue=None)
        self._reader_task = asyncio.create_task(self._reader_loop())
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=10.0)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RuntimeError("ElevenLabs STT session did not start in time") from exc

    async def send_pcm(self, pcm_bytes: bytes, *, commit: bool = False) -> None:
        if not self._ws:
            raise RuntimeError("STT session not connected")
        payload: dict = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm_bytes).decode("ascii"),
            "commit": commit,
            "sample_rate": self.sample_rate,
        }
        await self._ws.send(json.dumps(payload))

    async def commit(self) -> CommittedTranscript:
        loop = asyncio.get_running_loop()
        if self._commit_future and not self._commit_future.done():
            raise RuntimeError("Commit already in progress")
        self._commit_future = loop.create_future()
        await self.send_pcm(b"", commit=True)
        try:
            return await asyncio.wait_for(self._commit_future, timeout=30.0)
        finally:
            self._commit_future = None

    async def close(self) -> None:
        self._closed = True
        await self._events.put(
            STTEvent(type=STTEventType.ERROR, error="session_closed")
        )
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue
                data = json.loads(message)
                await self._handle_message(data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.exception("ElevenLabs STT reader error: %s", exc)
            await self._events.put(
                STTEvent(type=STTEventType.ERROR, error=str(exc))
            )
        finally:
            if self._commit_future and not self._commit_future.done():
                self._commit_future.set_exception(
                    RuntimeError("STT connection closed before commit")
                )

    async def _handle_message(self, data: dict) -> None:
        msg_type = data.get("message_type", "")
        if msg_type == "session_started":
            self._connected.set()
            await self._events.put(STTEvent(type=STTEventType.SESSION_STARTED))
            return
        if msg_type == "partial_transcript":
            text = (data.get("text") or "").strip()
            if text:
                await self._events.put(
                    STTEvent(type=STTEventType.PARTIAL, text=text)
                )
            return
        if msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
            text = (data.get("text") or "").strip()
            language = data.get("language_code")
            committed = CommittedTranscript(text=text, language_code=language)
            await self._events.put(
                STTEvent(
                    type=STTEventType.COMMITTED,
                    text=text,
                    language_code=language,
                )
            )
            if self._commit_future and not self._commit_future.done():
                self._commit_future.set_result(committed)
            return
        if msg_type in (
            "error",
            "auth_error",
            "quota_exceeded",
            "commit_throttled",
            "unaccepted_terms",
            "rate_limited",
            "queue_overflow",
            "resource_exhausted",
        ):
            error = data.get("error") or msg_type
            await self._events.put(STTEvent(type=STTEventType.ERROR, error=error))
            if self._commit_future and not self._commit_future.done():
                self._commit_future.set_exception(RuntimeError(error))

    async def next_event(self) -> STTEvent:
        return await self._events.get()
