"""
GPU TTS inference sidecar.

Run: uvicorn tts_server.main:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tts_server import config, engine

logger = logging.getLogger("tts_server")

_inference_lock = asyncio.Lock()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "hindi"
    speaker_id: str = "159"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Warming up TTS engine...")
    await asyncio.to_thread(engine.load)
    logger.info("TTS server ready")
    yield


app = FastAPI(title="Custom TTS Sidecar", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if engine.is_loaded() else "loading",
        "model": config.MODEL_ID,
        "sample_rate": config.SAMPLE_RATE,
    }


@app.post("/v1/tts/stream")
async def tts_stream(body: TTSRequest) -> StreamingResponse:
    if not engine.is_loaded():
        raise HTTPException(status_code=503, detail="TTS engine not ready")

    async def pcm_stream():
        async with _inference_lock:
            loop = asyncio.get_running_loop()
            out_q: asyncio.Queue[bytes | None] = asyncio.Queue()
            thread_q: queue.Queue[bytes | None] = queue.Queue()

            def _bridge() -> None:
                while True:
                    item = thread_q.get()
                    loop.call_soon_threadsafe(out_q.put_nowait, item)
                    if item is None:
                        break

            bridge = threading.Thread(target=_bridge, daemon=True)
            bridge.start()

            def _generate() -> None:
                try:
                    # Stream incrementally but coalesce tiny deltas into stable packet sizes.
                    target_bytes = max(1, int(config.SAMPLE_RATE * config.PCM_CHUNK_MS / 1000) * 2)
                    pending = bytearray()
                    gen_start = time.monotonic()
                    last_http_emit = gen_start
                    http_emit_count = 0
                    for pcm_chunk in engine.generate_audio_stream(
                        body.text,
                        language=body.language,
                        speaker_id=body.speaker_id,
                    ):
                        pending.extend(pcm_chunk)
                        while len(pending) >= target_bytes:
                            now = time.monotonic()
                            http_emit_count += 1
                            if http_emit_count == 1:
                                engine.probe(
                                    "C",
                                    "tts_server/main.py:first_http_chunk",
                                    "First HTTP PCM packet",
                                    {
                                        "bytes": target_bytes,
                                        "ms_since_gen_start": round(
                                            (now - gen_start) * 1000, 1
                                        ),
                                    },
                                )
                            elif http_emit_count <= 5:
                                engine.probe(
                                    "A",
                                    "tts_server/main.py:http_chunk",
                                    "HTTP PCM packet",
                                    {
                                        "emit": http_emit_count,
                                        "gap_ms": round((now - last_http_emit) * 1000, 1),
                                        "bytes": target_bytes,
                                    },
                                )
                            last_http_emit = now
                            thread_q.put(bytes(pending[:target_bytes]))
                            del pending[:target_bytes]
                    if pending:
                        thread_q.put(bytes(pending))
                except Exception:
                    logger.exception("TTS generation failed")
                finally:
                    thread_q.put(None)

            gen_thread = threading.Thread(target=_generate, daemon=True)
            gen_thread.start()

            first_emit = True
            while True:
                chunk = await out_q.get()
                if chunk is None:
                    break
                if first_emit:
                    first_emit = False
                yield chunk
                await asyncio.sleep(0)

            gen_thread.join(timeout=0.1)
            bridge.join(timeout=0.1)

    return StreamingResponse(
        pcm_stream(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Sample-Rate": str(config.SAMPLE_RATE),
            "X-Audio-Encoding": "s16le",
            "X-Audio-Channels": "1",
        },
    )
