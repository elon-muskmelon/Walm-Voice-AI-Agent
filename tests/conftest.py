"""Pytest fixtures and env defaults for tests."""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("ELEVENLABS_API_KEY", "test-key")
