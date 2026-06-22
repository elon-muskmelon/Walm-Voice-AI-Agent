"""
Async Gemini streaming wrapper using the google-genai SDK.

Key design decisions for low-latency voice:
  - thinking_budget=0  : disables chain-of-thought on thinking models
                         (gemini-2.5-flash-lite etc.) so first token
                         arrives in ~300–700ms instead of several seconds.
  - history capped at  : last MAX_HISTORY_TURNS exchanges to prevent
    MAX_HISTORY_TURNS    growing context from increasing latency.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger("gemini_streamer")

MAX_HISTORY_TURNS = 10   # keep last N user/model pairs = 2*N messages


class GeminiStreamer:
    """Stateful async wrapper around Gemini that preserves conversation history."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model
        self._history: list[types.Content] = []
        self._system_instruction = settings.system_prompt

    def reset_history(self) -> None:
        """Clear conversation context (e.g. on new session)."""
        self._history = []

    def _trim_history(self) -> None:
        """Keep only the last MAX_HISTORY_TURNS turn-pairs."""
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    async def stream(
        self,
        user_text: str,
        *,
        reply_language: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream LLM response tokens for *user_text*.
        Appends the exchange to conversation history automatically.
        """
        self._history.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )
        self._trim_history()

        system_instruction = self._system_instruction
        if reply_language:
            system_instruction = (
                f"{self._system_instruction} "
                f"The user is speaking {reply_language}. Reply in {reply_language}."
            )

        collected: list[str] = []
        try:
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=self._history,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    # Disable chain-of-thought on thinking models.
                    # Without this, gemini-2.5-* burns thinking tokens silently
                    # before the first output token, causing variable latency.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            ):
                if chunk.text:
                    collected.append(chunk.text)
                    yield chunk.text

        except Exception:
            logger.exception("Gemini stream failed")
            # Remove the user turn we already appended so history stays consistent
            if self._history and self._history[-1].role == "user":
                self._history.pop()
            raise

        # Record the full model reply
        if collected:
            self._history.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text="".join(collected))],
                )
            )
