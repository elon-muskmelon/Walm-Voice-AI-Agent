"""List Gemini models that support generateContent."""
from __future__ import annotations

import os

import dotenv
from google import genai


def main() -> None:
    dotenv.load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set in the environment.")

    client = genai.Client(api_key=api_key)
    for model in client.models.list():
        methods = getattr(model, "supported_generation_methods", None) or []
        if "generateContent" in methods or "generate_content" in str(methods).lower():
            print(model.name)


if __name__ == "__main__":
    main()
