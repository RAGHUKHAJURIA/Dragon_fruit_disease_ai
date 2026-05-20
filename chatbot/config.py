"""Configuration helpers for chatbot and LLM features.

Loads environment variables from a local `.env` when present and
provides small accessors for sensitive settings.
"""
from __future__ import annotations
import os
from typing import Optional

try:
    # soft dependency — only used during development if available
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


def get_openai_api_key() -> Optional[str]:
    return get_env("OPENAI_API_KEY")


def get_llm_model() -> str:
    return get_env("GEMINI_MODEL", "gemini-1.5-flash")


def get_gemini_api_key() -> Optional[str]:
    return get_env("GEMINI_API_KEY")
