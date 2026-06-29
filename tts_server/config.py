"""TTS GPU server configuration."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load repo-root .env so sidecar shares config with the main app.
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

MODEL_ID = os.getenv("TTS_MODEL_ID", "Mevearth2/Quantized-Merged-TTS")
MAX_SEQ_LENGTH = int(os.getenv("TTS_MAX_SEQ_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.getenv("TTS_MAX_NEW_TOKENS", "384"))
# ~14–18 audio tokens/char observed at runtime; budget must exceed speech so EOS stops first.
AUDIO_TOKENS_PER_CHAR = int(os.getenv("TTS_AUDIO_TOKENS_PER_CHAR", "18"))
AUDIO_TOKENS_MIN = int(os.getenv("TTS_AUDIO_TOKENS_MIN", "96"))
# Extra non-audio tokens allowed before speech tokens begin.
AUDIO_TOKEN_PREAMBLE = int(os.getenv("TTS_AUDIO_TOKEN_PREAMBLE", "32"))
LOAD_IN_4BIT = os.getenv("TTS_LOAD_IN_4BIT", "false").lower() in ("1", "true", "yes")
HF_TOKEN = os.getenv("HF_TOKEN") or None
SAMPLE_RATE = 24000
HOST = os.getenv("TTS_HOST", "0.0.0.0")
PORT = int(os.getenv("TTS_PORT", "8100"))
PCM_CHUNK_MS = int(os.getenv("TTS_PCM_CHUNK_MS", "400"))
# cumulative = decode all frames so far, emit only new samples (fluent).
# buffered = single decode after full generation (best quality, highest latency).
DECODE_MODE = os.getenv("TTS_DECODE_MODE", "cumulative").lower()
# Emit after N new SNAC frames (cumulative mode only).
STREAM_DECODE_FRAMES = int(os.getenv("TTS_STREAM_DECODE_FRAMES", "4"))

TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.4"))
TOP_P = float(os.getenv("TTS_TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("TTS_REPETITION_PENALTY", "1.05"))
