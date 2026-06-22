import json
from typing import Any, Dict


def dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True)


def parse(text: str) -> Dict[str, Any]:
    return json.loads(text)


MSG_START = "start"
MSG_STOP = "stop"
MSG_TEXT = "text"
MSG_AUDIO = "audio"

ROLE_STT_PARTIAL = "stt_partial"
ROLE_STT_FINAL = "stt_final"
ROLE_LLM = "llm"
