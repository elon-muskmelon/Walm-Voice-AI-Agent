import asyncio
import re
import time
from typing import Iterable


class TextSegmenter:
    def __init__(
        self,
        min_chars: int = 35,
        ideal_max_chars: int = 120,
        hard_max_chars: int = 180,
        timeout_ms: int = 450,
        strong_punct: str = ".?!।\n",
        soft_punct: str = ",;:—-",
        ack_phrases: Iterable[str] | None = None,
    ) -> None:
        self.min_chars = min_chars
        self.ideal_max_chars = ideal_max_chars
        self.hard_max_chars = hard_max_chars
        self.timeout_s = timeout_ms / 1000.0
        self.strong_punct = strong_punct
        self.soft_punct = soft_punct
        self.ack_phrases = {p.strip() for p in (ack_phrases or self._default_acks())}
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._buffer = ""

    async def push(self, text: str) -> None:
        await self._queue.put(text)

    async def close(self) -> None:
        await self._queue.put(None)

    async def segments(self):
        last_input_at = time.monotonic()
        while True:
            timeout = None
            if len(self._buffer) >= self.min_chars:
                timeout = self.timeout_s
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout)
            except asyncio.TimeoutError:
                for segment in self._drain_segments(force=False, timeout=True):
                    yield segment
                continue

            if item is None:
                for segment in self._drain_segments(force=True, timeout=True):
                    yield segment
                break

            last_input_at = time.monotonic()
            self._buffer += item
            for segment in self._drain_segments(force=False, timeout=False):
                yield segment

            if len(self._buffer) >= self.min_chars:
                if time.monotonic() - last_input_at >= self.timeout_s:
                    for segment in self._drain_segments(force=False, timeout=True):
                        yield segment

    def _drain_segments(self, force: bool, timeout: bool) -> list[str]:
        segments: list[str] = []
        while True:
            segment = self._pick_segment(force=force, timeout=timeout)
            if not segment:
                break
            segments.append(segment)
        return segments

    def _pick_segment(self, force: bool, timeout: bool) -> str | None:
        buffer = self._buffer.lstrip()
        if not buffer:
            self._buffer = ""
            return None

        ack = self._maybe_ack(buffer, force or timeout)
        if ack:
            return ack

        length = len(buffer)
        if length < self.min_chars and not force and not timeout:
            self._buffer = buffer
            return None

        cut = self._find_boundary(buffer, self.strong_punct, self.ideal_max_chars)
        if cut >= self.min_chars:
            return self._consume(buffer, cut)

        if length >= self.ideal_max_chars:
            cut = self._find_boundary(buffer, self.soft_punct, self.ideal_max_chars)
            if cut >= self.min_chars:
                return self._consume(buffer, cut)

        if length >= self.hard_max_chars:
            cut = self._find_boundary(buffer, self.strong_punct, self.hard_max_chars)
            if cut >= self.min_chars:
                return self._consume(buffer, cut)
            cut = self._find_boundary(buffer, self.soft_punct, self.hard_max_chars)
            if cut >= self.min_chars:
                return self._consume(buffer, cut)
            return self._consume(buffer, self._cut_at_space(buffer, self.hard_max_chars))

        if timeout or force:
            cut = self._find_boundary(buffer, self.strong_punct, length)
            if cut >= self.min_chars:
                return self._consume(buffer, cut)
            cut = self._find_boundary(buffer, self.soft_punct, length)
            if cut >= self.min_chars:
                return self._consume(buffer, cut)
            if force:
                return self._consume(buffer, self._cut_at_space(buffer, length))

        self._buffer = buffer
        return None

    def _consume(self, buffer: str, cut: int) -> str:
        segment = buffer[:cut].strip()
        remainder = buffer[cut:].lstrip()
        self._buffer = remainder
        return segment

    def _cut_at_space(self, buffer: str, limit: int) -> int:
        if limit >= len(buffer):
            return len(buffer)
        space_index = buffer.rfind(" ", 0, limit)
        return space_index if space_index > 0 else limit

    def _find_boundary(self, buffer: str, punct: str, limit: int) -> int:
        if not punct:
            return -1
        max_len = min(limit, len(buffer))
        pattern = f"[{re.escape(punct)}]"
        last = -1
        for match in re.finditer(pattern, buffer[:max_len]):
            last = match.end()
        return last

    def _maybe_ack(self, buffer: str, allow: bool) -> str | None:
        if not allow:
            return None
        stripped = buffer.strip()
        for phrase in self.ack_phrases:
            if stripped == phrase:
                return self._consume(buffer, len(buffer))
            if stripped.startswith(phrase) and len(stripped) <= self.min_chars:
                return self._consume(buffer, len(buffer))
        return None

    @staticmethod
    def _default_acks() -> list[str]:
        return [
            "Okay.",
            "Got it.",
            "Sure.",
            "One moment.",
            "Let me check.",
        ]
