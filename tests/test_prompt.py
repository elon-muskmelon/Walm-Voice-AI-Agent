"""Unit tests for TTS prompt formatting."""

from tts_server.prompt import (
    AUDIO_START_ID,
    END_OF_SPEECH_ID,
    START_OF_AI_ID,
    START_OF_HUMAN_ID,
    START_OF_SPEECH_ID,
    START_OF_TEXT_ID,
    build_eval_prompt,
)


class _FakeTokenizer:
    def decode(self, ids: list[int]) -> str:
        return f"<{ids[0]}>"


def test_build_eval_prompt_structure():
    prompt = build_eval_prompt(
        _FakeTokenizer(),
        utterance="नमस्ते",
        language="hindi",
        speaker_id="159",
    )
    assert f"<{START_OF_HUMAN_ID}>" in prompt
    assert f"<{START_OF_TEXT_ID}>" in prompt
    assert "hindi159: नमस्ते" in prompt
    assert f"<{START_OF_AI_ID}>" in prompt
    assert f"<{START_OF_SPEECH_ID}>" in prompt


def test_special_token_ids():
    assert AUDIO_START_ID == 128256 + 10
    assert END_OF_SPEECH_ID == 128256 + 2
