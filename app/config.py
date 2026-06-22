"""
Centralised configuration loaded from environment variables / .env file.
Uses pydantic-settings so all values are typed and validated on startup.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── API keys ──────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    # ── LLM ──────────────────────────────────────────────────────────────────
    gemini_model: str = "gemini-2.0-flash-lite"
    system_prompt: str = (
        "You are a real-time voice assistant. "
        "Reply in short, natural spoken sentences (5–15 words each). "
        "Never use markdown, bullet points, emojis, or long paragraphs. "
        "Be concise and conversational. "
        "Always reply in the same language the user spoke."
    )

    # ── TTS backend ───────────────────────────────────────────────────────────
    tts_backend: Literal["elevenlabs", "custom"] = "custom"
    tts_service_url: str = "http://localhost:8100"
    tts_default_language: str = "hindi"
    tts_default_speaker: str = "159"
    tts_sample_rate: int = 24000

    # ── ElevenLabs TTS ────────────────────────────────────────────────────────
    elevenlabs_model_id: str = "eleven_flash_v2_5"
    elevenlabs_tts_output_format: str = "mp3_44100_128"
    # Chunk-length schedule controls how aggressively EL buffers before flushing
    elevenlabs_tts_chunk_schedule: List[int] = [50, 120, 160]

    # ── ElevenLabs STT ────────────────────────────────────────────────────────
    elevenlabs_stt_model_id: str = "scribe_v2_realtime"
    elevenlabs_stt_language_code: str = ""   # empty = auto-detect

    # ── Audio / VAD ───────────────────────────────────────────────────────────
    audio_sample_rate: int = 16000
    vad_frame_ms: int = 30
    vad_pre_speech_ms: int = 300
    vad_min_speech_ms: int = 300
    vad_end_silence_ms: int = 400
    vad_max_utterance_ms: int = 25000
    vad_start_trigger_frames: int = 2
    vad_speech_threshold: float = 0.5
    vad_trailing_silence_keep_ms: int = 200
    vad_model_path: str = ""            # path to Silero ONNX; empty → WebRTC

    # Debug WAV dumps
    vad_debug_save: bool = False
    vad_debug_dir: str = "debug_utterances"

    # ── TTS text segmentation ─────────────────────────────────────────────────
    tts_min_chars: int = 6
    tts_segment_timeout_ms: int = 100

    # ── STT commit ────────────────────────────────────────────────────────────
    stt_commit_timeout_s: float = 15.0

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("gemini_api_key", "elevenlabs_api_key")
    @classmethod
    def _required_api_keys(cls, v: str, info) -> str:
        if not v:
            raise ValueError(f"{info.field_name} must be set")
        return v

    @model_validator(mode="after")
    def _validate_tts_backend(self) -> "Settings":
        if self.tts_backend == "elevenlabs" and not self.elevenlabs_voice_id:
            raise ValueError(
                "elevenlabs_voice_id must be set when TTS_BACKEND=elevenlabs"
            )
        if self.tts_backend == "custom" and not self.tts_service_url:
            raise ValueError("tts_service_url must be set when TTS_BACKEND=custom")
        return self


settings = Settings()
