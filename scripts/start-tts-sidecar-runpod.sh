#!/usr/bin/env bash
# Run TTS sidecar on RunPod / Linux GPU (bf16, streaming decode).
set -euo pipefail
cd "$(dirname "$0")/.."

export TTS_LOAD_IN_4BIT="${TTS_LOAD_IN_4BIT:-false}"
export TTS_DECODE_MODE="${TTS_DECODE_MODE:-cumulative}"
export TTS_STREAM_DECODE_FRAMES="${TTS_STREAM_DECODE_FRAMES:-2}"
export TTS_PCM_CHUNK_MS="${TTS_PCM_CHUNK_MS:-200}"
export TTS_HOST="${TTS_HOST:-0.0.0.0}"
export TTS_PORT="${TTS_PORT:-8100}"

echo "TTS sidecar (RunPod) 4bit=$TTS_LOAD_IN_4BIT decode=$TTS_DECODE_MODE port=$TTS_PORT"
exec uvicorn tts_server.main:app --host "$TTS_HOST" --port "$TTS_PORT"
