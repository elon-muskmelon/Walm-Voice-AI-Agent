"""GPU TTS engine: transformers model + SNAC decode."""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from typing import Optional

import numpy as np
import torch
from transformers.generation.streamers import BaseStreamer

from tts_server import config
from tts_server.prompt import (
    AUDIO_START_ID,
    END_OF_SPEECH_ID,
    build_eval_prompt,
)
from tts_server.streaming import float32_to_pcm16_bytes

logger = logging.getLogger("tts_engine")

_model = None
_tokenizer = None
_snac_model = None
_device: Optional[str] = None


def _device_for_inference() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def load() -> None:
    """Load model, tokenizer, and SNAC decoder once at startup."""
    global _model, _tokenizer, _snac_model

    if _model is not None:
        return

    device = _device_for_inference()
    logger.info(
        "Loading TTS engine model=%s 4bit=%s device=%s",
        config.MODEL_ID,
        config.LOAD_IN_4BIT,
        device,
    )

    from snac import SNAC

    _load_with_transformers(device)

    logger.info("Loading SNAC decoder")
    _snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    if device == "cuda":
        _snac_model = _snac_model.to(device)
    _snac_model.eval()

    logger.info("TTS engine ready on %s", device)


def _configure_tokenizer(tokenizer) -> None:
    pad_token = tokenizer.decode([128256 + 7])
    tokenizer.pad_token = pad_token
    tokenizer.padding_side = "left"


def _load_with_transformers(device: str) -> None:
    global _model, _tokenizer

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Using transformers %s loader", transformers.__version__)

    load_kwargs: dict = {}
    if config.HF_TOKEN:
        load_kwargs["token"] = config.HF_TOKEN

    _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, **load_kwargs)
    _configure_tokenizer(_tokenizer)

    model_kwargs: dict = {
        **load_kwargs,
        "low_cpu_mem_usage": True,
        "device_map": {"": 0} if device == "cuda" else None,
    }
    if config.LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    if device == "cuda":
        torch.cuda.empty_cache()

    _model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, **model_kwargs)
    _model.eval()


def is_loaded() -> bool:
    return _model is not None


def _audio_ids_to_codes(audio_ids: list[int]) -> list[list[int]] | None:
    """Convert raw audio token ids to SNAC codebooks (full frames only)."""
    if not audio_ids:
        return None

    clean: list[int] = []
    frame_count = (len(audio_ids) + 6) // 7
    for i in range(frame_count):
        for j in range(7):
            idx = 7 * i + j
            if idx < len(audio_ids):
                clean.append(audio_ids[idx] - AUDIO_START_ID)

    codes: list[list[int]] = [[], [], []]
    code_frames = len(clean) // 7
    for i in range(code_frames):
        base = 7 * i
        if base + 6 >= len(clean):
            break
        codes[0].append(clean[base])
        codes[1].append(clean[base + 1] - 4096)
        codes[2].append(clean[base + 2] - (2 * 4096))
        codes[2].append(clean[base + 3] - (3 * 4096))
        codes[1].append(clean[base + 4] - (4 * 4096))
        codes[2].append(clean[base + 5] - (5 * 4096))
        codes[2].append(clean[base + 6] - (6 * 4096))

    return codes if codes[0] else None


def _decode_codes(codes: list[list[int]]) -> np.ndarray | None:
    """Decode SNAC codebooks to float32 mono waveform."""
    if _snac_model is None or not codes or not codes[0]:
        return None

    device = _device_for_inference()
    code_tensors = [
        torch.tensor(codes[0]).unsqueeze(0),
        torch.tensor(codes[1]).unsqueeze(0),
        torch.tensor(codes[2]).unsqueeze(0),
    ]
    if device == "cuda":
        code_tensors = [t.to(device) for t in code_tensors]

    try:
        with torch.inference_mode():
            audio = _snac_model.decode(code_tensors)
        return audio.detach().squeeze().to("cpu").numpy().astype(np.float32)
    except Exception:
        logger.exception("SNAC decode failed")
        return None


def _apply_speed(waveform: np.ndarray, speed: float) -> np.ndarray:
    if not speed or abs(speed - 1.0) <= 1e-4:
        return waveform
    try:
        import torchaudio

        tensor = torch.from_numpy(waveform).unsqueeze(0)
        sped, _ = torchaudio.sox_effects.apply_effects_tensor(
            tensor,
            config.SAMPLE_RATE,
            effects=[["tempo", f"{speed}"]],
        )
        return sped.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        logger.warning("Speed adjustment failed; returning original audio")
        return waveform


def _prepare_generation(text: str, language: str, speaker_id: str) -> tuple | None:
    if _model is None or _tokenizer is None or _snac_model is None:
        raise RuntimeError("TTS engine not loaded — call load() first")

    model_device = next(_model.parameters()).device
    prompt = build_eval_prompt(
        _tokenizer,
        utterance=text.strip(),
        language=language.lower(),
        speaker_id=speaker_id,
    )
    inputs = _tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    max_tokens = config.MAX_SEQ_LENGTH - inputs.input_ids.shape[1]
    if max_tokens <= 0:
        logger.error("Prompt too long for max_seq_length=%d", config.MAX_SEQ_LENGTH)
        return None

    gen_kwargs = {
        "input_ids": inputs.input_ids.to(model_device),
        "attention_mask": inputs.attention_mask.to(model_device),
        "max_new_tokens": max_tokens,
        "do_sample": True,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "repetition_penalty": config.REPETITION_PENALTY,
        "eos_token_id": END_OF_SPEECH_ID,
    }
    return gen_kwargs, model_device


class _SnacFrameStreamer(BaseStreamer):
    """Decode SNAC frames incrementally as audio tokens arrive from generate()."""

    def __init__(self) -> None:
        super().__init__()
        self._audio_ids: list[int] = []
        self._decoded_samples = 0
        self._chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue()

    def _process_token(self, tid: int) -> None:
        if tid >= AUDIO_START_ID and tid != END_OF_SPEECH_ID:
            self._audio_ids.append(tid)
            if len(self._audio_ids) % 7 == 0:
                self._emit_delta()

    def put(self, value: torch.Tensor) -> None:
        if value.ndim == 2:
            for row in value:
                for tid in row.tolist():
                    self._process_token(int(tid))
        else:
            for tid in value.tolist():
                self._process_token(int(tid))

    def end(self) -> None:
        self._chunk_queue.put(None)

    def iter_chunks(self) -> Iterator[np.ndarray]:
        while True:
            chunk = self._chunk_queue.get()
            if chunk is None:
                break
            yield chunk

    def _emit_delta(self) -> None:
        codes = _audio_ids_to_codes(self._audio_ids)
        if not codes:
            return
        waveform = _decode_codes(codes)
        if waveform is None or len(waveform) <= self._decoded_samples:
            return
        delta = waveform[self._decoded_samples :]
        self._decoded_samples = len(waveform)
        if len(delta) > 0:
            self._chunk_queue.put(delta)


def generate_audio_stream(
    text: str,
    *,
    language: str = "hindi",
    speaker_id: str = "159",
) -> Iterator[bytes]:
    """Yield Int16 PCM bytes incrementally during token generation."""
    prepared = _prepare_generation(text, language, speaker_id)
    if prepared is None:
        return
    gen_kwargs, _model_device = prepared

    streamer = _SnacFrameStreamer()
    gen_kwargs["streamer"] = streamer

    error_box: list[BaseException] = []

    def _run_generate() -> None:
        try:
            with torch.inference_mode():
                _model.generate(**gen_kwargs)
        except BaseException as exc:
            error_box.append(exc)
            streamer.end()

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()

    for chunk in streamer.iter_chunks():
        yield float32_to_pcm16_bytes(chunk)

    thread.join()
    if error_box:
        raise error_box[0]


def generate_audio(
    text: str,
    *,
    language: str = "hindi",
    speaker_id: str = "159",
    speed: float = 1.0,
) -> np.ndarray | None:
    """Generate full speech waveform (float32 mono @ 24 kHz) for *text*."""
    pcm = b"".join(
        generate_audio_stream(
            text,
            language=language,
            speaker_id=speaker_id,
        )
    )
    if not pcm:
        logger.warning("No audio generated for text=%r", text[:80])
        return None

    waveform = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    return _apply_speed(waveform, speed)
