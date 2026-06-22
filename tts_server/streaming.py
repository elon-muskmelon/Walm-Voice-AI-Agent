"""PCM chunking utilities for streaming audio to clients."""
from __future__ import annotations

import numpy as np

from tts_server.config import PCM_CHUNK_MS, SAMPLE_RATE


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def iter_pcm_chunks(pcm_bytes: bytes, chunk_ms: int = PCM_CHUNK_MS) -> list[bytes]:
    """Split Int16 PCM into fixed-duration chunks for incremental delivery."""
    if not pcm_bytes:
        return []
    samples_per_chunk = max(1, int(SAMPLE_RATE * chunk_ms / 1000))
    bytes_per_chunk = samples_per_chunk * 2
    chunks: list[bytes] = []
    for offset in range(0, len(pcm_bytes), bytes_per_chunk):
        chunks.append(pcm_bytes[offset : offset + bytes_per_chunk])
    return chunks
