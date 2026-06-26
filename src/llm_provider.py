from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import requests


ProviderName = Literal["Local Mock LLM", "OpenAI", "Hugging Face"]


@dataclass
class LLMResponse:
    provider: str
    used_real_model: bool
    text: str
    error: str | None = None


class LLMProvider:
    """Optional real LLM bridge with a reliable offline fallback."""

    def __init__(self, provider: ProviderName = "Local Mock LLM") -> None:
        self.provider = provider
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-5.5")
        self.huggingface_model = os.getenv("HF_MODEL", "google/flan-t5-base")

    def generate(self, prompt: str, fallback: str) -> LLMResponse:
        if self.provider == "OpenAI":
            return self._openai(prompt, fallback)
        if self.provider == "Hugging Face":
            return self._huggingface(prompt, fallback)
        return LLMResponse(provider="Local Mock LLM", used_real_model=False, text=fallback)

    def status(self) -> dict[str, str]:
        return {
            "selected_provider": self.provider,
            "openai_key": "available" if os.getenv("OPENAI_API_KEY") else "not configured",
            "openai_model": self.openai_model,
            "huggingface_token": "available" if os.getenv("HF_API_TOKEN") else "not configured",
            "huggingface_model": self.huggingface_model,
        }

    def _openai(self, prompt: str, fallback: str) -> LLMResponse:
        if not os.getenv("OPENAI_API_KEY"):
            return LLMResponse("OpenAI", False, fallback, "OPENAI_API_KEY is not configured.")
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.responses.create(
                model=self.openai_model,
                instructions="You are a concise software engineering project assistant. Return practical, structured output.",
                input=prompt,
            )
            return LLMResponse("OpenAI", True, response.output_text.strip() or fallback)
        except Exception as exc:  # pragma: no cover - depends on external API availability
            return LLMResponse("OpenAI", False, fallback, str(exc))

    def _huggingface(self, prompt: str, fallback: str) -> LLMResponse:
        token = os.getenv("HF_API_TOKEN")
        if not token:
            return LLMResponse("Hugging Face", False, fallback, "HF_API_TOKEN is not configured.")
        try:
            endpoint = f"https://api-inference.huggingface.co/models/{self.huggingface_model}"
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {token}"},
                json={"inputs": prompt, "parameters": {"max_new_tokens": 320}},
                timeout=35,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list) and payload:
                text = payload[0].get("generated_text", fallback)
            elif isinstance(payload, dict):
                text = payload.get("generated_text", fallback)
            else:
                text = fallback
            return LLMResponse("Hugging Face", True, str(text).strip() or fallback)
        except Exception as exc:  # pragma: no cover - depends on external API availability
            return LLMResponse("Hugging Face", False, fallback, str(exc))
