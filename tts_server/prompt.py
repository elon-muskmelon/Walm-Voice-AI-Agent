"""Orpheus / snorTTS prompt formatting and special token constants."""
from __future__ import annotations

TOKENISER_LENGTH = 128256

START_OF_TEXT_ID = 128000
END_OF_TEXT_ID = 128009
START_OF_SPEECH_ID = TOKENISER_LENGTH + 1
END_OF_SPEECH_ID = TOKENISER_LENGTH + 2
START_OF_HUMAN_ID = TOKENISER_LENGTH + 3
END_OF_HUMAN_ID = TOKENISER_LENGTH + 4
START_OF_AI_ID = TOKENISER_LENGTH + 5
END_OF_AI_ID = TOKENISER_LENGTH + 6
PAD_TOKEN_ID = TOKENISER_LENGTH + 7
AUDIO_START_ID = TOKENISER_LENGTH + 10


def build_eval_prompt(
    tokenizer,
    *,
    utterance: str,
    language: str,
    speaker_id: str,
) -> str:
    """Build inference prompt for Orpheus-style TTS (language + speaker prefix)."""
    start_of_human = tokenizer.decode([START_OF_HUMAN_ID])
    start_of_text = tokenizer.decode([START_OF_TEXT_ID])
    end_of_text = tokenizer.decode([END_OF_TEXT_ID])
    end_of_human = tokenizer.decode([END_OF_HUMAN_ID])
    start_of_ai = tokenizer.decode([START_OF_AI_ID])
    start_of_speech = tokenizer.decode([START_OF_SPEECH_ID])

    body = f"{language}{speaker_id}: {utterance}"
    return (
        f"{start_of_human}{start_of_text}{body}{end_of_text}"
        f"{end_of_human}{start_of_ai}{start_of_speech}"
    )
