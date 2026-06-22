"""Unit tests for LLM text segmentation."""

import asyncio

import pytest

from app.text_segmenter import TextSegmenter


@pytest.mark.asyncio
async def test_flush_on_min_chars_and_punctuation():
    seg = TextSegmenter(min_chars=6, timeout_ms=500)
    task = asyncio.create_task(_collect(seg))

    await seg.push("Hello.")
    await seg.close()
    segments = await task

    assert segments == ["Hello."]


@pytest.mark.asyncio
async def test_flush_on_timeout():
    seg = TextSegmenter(min_chars=4, timeout_ms=50)
    task = asyncio.create_task(_collect(seg))

    await seg.push("Hello")
    await asyncio.sleep(0.08)
    await seg.close()
    segments = await task

    assert any("Hello" in s for s in segments)


async def _collect(segmenter: TextSegmenter) -> list[str]:
    out: list[str] = []
    async for segment in segmenter.segments():
        out.append(segment)
    return out
