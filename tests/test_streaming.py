"""Unit tests for PCM streaming utilities."""

import numpy as np

from tts_server.streaming import float32_to_pcm16_bytes, iter_pcm_chunks


def test_float32_to_pcm16_bytes():
    audio = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    pcm = float32_to_pcm16_bytes(audio)
    assert len(pcm) == 6
    samples = np.frombuffer(pcm, dtype=np.int16)
    assert samples[1] == 32767
    assert samples[2] == -32767


def test_iter_pcm_chunks():
    pcm = b"\x00\x01" * 100
    chunks = iter_pcm_chunks(pcm, chunk_ms=100)
    assert len(chunks) >= 1
    assert sum(len(c) for c in chunks) == len(pcm)
