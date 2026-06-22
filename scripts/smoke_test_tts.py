#!/usr/bin/env python3
"""
Standalone smoke test for the custom TTS model.

Usage (from repo root, with CUDA + tts_server deps installed):
  python scripts/smoke_test_tts.py --text "नमस्ते, यह एक परीक्षण है।"

Requires tts_server/engine.py dependencies (snac, torch, transformers==4.53.1).
Install dev deps for WAV output: pip install -r scripts/requirements-dev.txt
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf

from tts_server import engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test custom TTS engine")
    parser.add_argument(
        "--text",
        default="नमस्ते, यह एक परीक्षण है।",
        help="Utterance to synthesize",
    )
    parser.add_argument("--language", default="hindi")
    parser.add_argument("--speaker", default="159")
    parser.add_argument("--speed", type=float, default=1.05)
    parser.add_argument("--output", default="smoke_test.wav")
    args = parser.parse_args()

    print("Loading TTS engine...")
    try:
        engine.load()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print(f"Generating: {args.text!r}")
    audio = engine.generate_audio(
        args.text,
        language=args.language,
        speaker_id=args.speaker,
        speed=args.speed,
    )
    if audio is None:
        print("FAILED: no audio generated", file=sys.stderr)
        sys.exit(1)

    sf.write(args.output, audio, 24000)
    print(f"OK: wrote {args.output} ({len(audio) / 24000:.2f}s)")


if __name__ == "__main__":
    main()
