import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


_shared_vad = None


@dataclass
class VADConfig:
    sample_rate: int = 16000
    threshold: float = 0.5


class SileroVAD:
    def __init__(self, model_path: Path, config: VADConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("vad")
        self._model_path = model_path
        self._session = None
        self._h = None
        self._c = None
        self._inputs = {}

    def load(self) -> None:
        import onnxruntime as ort

        if not self._model_path.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {self._model_path}")

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )
        self._inputs = {inp.name: inp for inp in self._session.get_inputs()}
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def speech_probability(self, pcm_int16: bytes) -> float:
        if self._session is None:
            raise RuntimeError("VAD session not loaded")
        audio = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        audio = audio.reshape(1, -1)

        feed = {"input": audio}
        if "h" in self._inputs and self._h is not None:
            feed["h"] = self._h
        if "c" in self._inputs and self._c is not None:
            feed["c"] = self._c
        if "sr" in self._inputs:
            feed["sr"] = np.array([self.config.sample_rate], dtype=np.int64)

        outputs = self._session.run(None, feed)
        prob = float(outputs[0].reshape(-1)[0])

        if len(outputs) >= 3:
            self._h = outputs[1]
            self._c = outputs[2]

        return prob


class WebRtcVAD:
    def __init__(self, config: VADConfig, mode: int = 2) -> None:
        import webrtcvad

        self.config = config
        self._vad = webrtcvad.Vad(mode)

    def speech_probability(self, pcm_int16: bytes) -> float:
        is_speech = self._vad.is_speech(pcm_int16, self.config.sample_rate)
        return 1.0 if is_speech else 0.0


class VADFactory:
    def __init__(self, config: VADConfig, model_path: Optional[Path]) -> None:
        self.config = config
        self.model_path = model_path
        self.logger = logging.getLogger("vad")

    def build(self):
        if self.model_path:
            try:
                vad = SileroVAD(self.model_path, self.config)
                vad.load()
                self.logger.info("VAD backend=silero_onnx model=%s", self.model_path)
                return vad
            except Exception as exc:
                self.logger.warning("Silero VAD init failed, falling back: %s", exc)
        vad = WebRtcVAD(self.config)
        self.logger.info("VAD backend=webrtc")
        return vad


def init_shared_vad(config: VADConfig, model_path: Optional[Path]) -> None:
    global _shared_vad
    if _shared_vad is not None:
        return
    _shared_vad = VADFactory(config, model_path).build()


def get_shared_vad():
    if _shared_vad is None:
        raise RuntimeError("Shared VAD not initialized")
    return _shared_vad
