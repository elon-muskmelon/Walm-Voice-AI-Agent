import collections
import logging
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class SegmenterConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    pre_speech_ms: int = 300
    min_speech_ms: int = 300
    end_silence_ms: int = 600
    max_utterance_ms: int = 25000
    start_trigger_frames: int = 2
    speech_threshold: float = 0.5
    trailing_silence_keep_ms: int = 200


class UtteranceSegmenter:
    def __init__(self, config: SegmenterConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("vad")
        self.frame_bytes = int(
            self.config.sample_rate * (self.config.frame_ms / 1000.0) * 2
        )
        self.pre_frames = max(1, int(self.config.pre_speech_ms / self.config.frame_ms))
        self.trailing_keep_frames = max(
            1, int(self.config.trailing_silence_keep_ms / self.config.frame_ms)
        )
        self.reset()

    def reset(self) -> None:
        self.state = "IDLE"
        self.pre_buffer: Deque[bytes] = collections.deque(maxlen=self.pre_frames)
        self.utterance: list[bytes] = []
        self.consecutive_speech = 0
        self.trailing_silence_ms = 0
        self.speech_ms = 0
        self.utterance_ms = 0

    def process_frame(self, frame: bytes, speech_prob: float) -> Optional[bytes]:
        is_speech = speech_prob >= self.config.speech_threshold

        if self.state == "IDLE":
            self.pre_buffer.append(frame)
            if is_speech:
                self.consecutive_speech += 1
            else:
                self.consecutive_speech = 0

            if self.consecutive_speech >= self.config.start_trigger_frames:
                self.state = "SPEECH"
                self.utterance = list(self.pre_buffer)
                self.utterance.append(frame)
                self.trailing_silence_ms = 0
                self.speech_ms = self.consecutive_speech * self.config.frame_ms
                self.utterance_ms = len(self.utterance) * self.config.frame_ms
                self.logger.info("vad_speech_start")
            return None

        self.utterance.append(frame)
        self.utterance_ms += self.config.frame_ms

        if is_speech:
            self.trailing_silence_ms = 0
            self.speech_ms += self.config.frame_ms
        else:
            self.trailing_silence_ms += self.config.frame_ms

        if self.trailing_silence_ms >= self.config.end_silence_ms:
            return self._finalize(ended_by_silence=True)

        if self.utterance_ms >= self.config.max_utterance_ms:
            return self._finalize(ended_by_silence=False)

        return None

    def _finalize(self, ended_by_silence: bool) -> Optional[bytes]:
        if self.speech_ms < self.config.min_speech_ms:
            self.logger.info("vad_reject_short speech_ms=%d", self.speech_ms)
            self.reset()
            return None

        if ended_by_silence and self.trailing_keep_frames > 0:
            trim_frames = max(0, int(self.trailing_silence_ms / self.config.frame_ms) - self.trailing_keep_frames)
            if trim_frames > 0:
                self.utterance = self.utterance[:-trim_frames]

        audio = b"".join(self.utterance)
        self.logger.info(
            "vad_endpoint utterance_ms=%d speech_ms=%d silence_ms=%d",
            self.utterance_ms,
            self.speech_ms,
            self.trailing_silence_ms,
        )
        self.reset()
        return audio
