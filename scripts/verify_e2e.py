#!/usr/bin/env python3
"""Health check and optional TTS latency benchmark."""
from __future__ import annotations

import sys
import time

import httpx

TTS_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8100"
MAIN_URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
BENCHMARK = "--benchmark" in sys.argv


def check(name: str, url: str) -> bool:
    try:
        r = httpx.get(url, timeout=5.0)
        print(f"{name}: {r.status_code} {r.text[:120]}")
        return r.status_code == 200
    except httpx.HTTPError as exc:
        print(f"{name}: FAIL ({exc})")
        return False


def benchmark_tts_first_byte() -> None:
    url = f"{TTS_URL.rstrip('/')}/v1/tts/stream"
    payload = {
        "text": "Hello, this is a short latency test.",
        "language": "english",
        "speaker_id": "159",
    }
    print(f"TTS benchmark POST {url}")
    start = time.perf_counter()
    first_byte_at: float | None = None
    total_bytes = 0
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0)) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    if chunk and first_byte_at is None:
                        first_byte_at = time.perf_counter()
                    total_bytes += len(chunk)
    except httpx.HTTPError as exc:
        print(f"TTS benchmark FAIL ({exc})")
        return

    end = time.perf_counter()
    if first_byte_at is None:
        print("TTS benchmark: no audio received")
        return
    print(
        f"TTS benchmark: first_byte={first_byte_at - start:.2f}s "
        f"total={end - start:.2f}s bytes={total_bytes}"
    )


def main() -> None:
    ok_tts = check("TTS /health", f"{TTS_URL.rstrip('/')}/health")
    ok_main = check("Main /", MAIN_URL)
    if not ok_tts:
        print("Start TTS sidecar first: .\\scripts\\start-tts-sidecar.ps1")
        sys.exit(1)
    if not ok_main:
        print("Start main app: .\\scripts\\start-main-app.ps1")
        sys.exit(1)
    print("OK: both services reachable")
    if BENCHMARK:
        benchmark_tts_first_byte()


if __name__ == "__main__":
    main()
