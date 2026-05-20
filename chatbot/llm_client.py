"""Lightweight LLM client wrapper with OpenAI fallback and safe local fallback.

This module tries to call OpenAI's Chat Completions API when an
`OPENAI_API_KEY` environment variable is present. If not configured or if
the network call fails, it returns a deterministic fallback summary to
avoid breaking the application.
"""
from __future__ import annotations
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

    def generate_text(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a short text completion for the given prompt.

        If no API key is configured, returns a safe local summary derived
        from the prompt to ensure the caller always receives a usable
        response.
        """
        if not self.api_key:
            # Simple deterministic fallback: return first 2 sentences from prompt
            try:
                lines = prompt.strip().split('\n')
                summary = ' '.join(lines[:4])
                return (summary[:max_tokens*2] + '...') if len(summary) > max_tokens else summary
            except Exception:
                return "(LLM unavailable)"

        # Try to call OpenAI REST API using requests (avoid hard dependency)
        try:
            import requests

            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning("LLM request failed: %s", e)
            # Fallback deterministic behavior
            try:
                lines = prompt.strip().split('\n')
                summary = ' '.join(lines[:4])
                return (summary[:max_tokens*2] + '...') if len(summary) > max_tokens else summary
            except Exception:
                return "(LLM error)"


_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def generate_text(prompt: str, max_tokens: int = 256) -> str:
    return get_default_client().generate_text(prompt, max_tokens=max_tokens)
