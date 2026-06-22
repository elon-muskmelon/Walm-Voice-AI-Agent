#!/usr/bin/env python3
"""Quick health check for TTS sidecar and main app."""
from __future__ import annotations

import sys

import httpx

TTS_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8100"
MAIN_URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"


def check(name: str, url: str) -> bool:
    try:
        r = httpx.get(url, timeout=5.0)
        print(f"{name}: {r.status_code} {r.text[:120]}")
        return r.status_code == 200
    except httpx.HTTPError as exc:
        print(f"{name}: FAIL ({exc})")
        return False


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


if __name__ == "__main__":
    main()
