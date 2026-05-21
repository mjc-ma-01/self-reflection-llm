"""OpenAI GPT-5 query helpers for Reflector data generation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class OpenAIConfig:
    model: str = os.getenv("OPENAI_MODEL", "gpt-5")
    api_key: str | None = os.getenv("OPENAI_API_KEY")
    base_url: str | None = os.getenv("OPENAI_BASE_URL")
    timeout: float = float(os.getenv("OPENAI_TIMEOUT", "120"))
    max_retries: int = int(os.getenv("OPENAI_MAX_RETRIES", "2"))


class GPT5Client:
    def __init__(self, config: OpenAIConfig | None = None) -> None:
        self.config = config or OpenAIConfig()
        if not self.config.api_key:
            raise ValueError("OPENAI_API_KEY is required for Reflector data generation.")
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    def json_completion(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                payload = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                }
                try:
                    response = self.client.chat.completions.create(**payload)
                except TypeError:
                    payload["max_tokens"] = payload.pop("max_completion_tokens")
                    response = self.client.chat.completions.create(**payload)
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"OpenAI GPT-5 request failed: {last_error}") from last_error


def require_text(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"GPT-5 JSON response missing non-empty {key!r}: {obj}")
    return value.strip()
