from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import requests


ProviderName = Literal["Local Mock LLM", "OpenAI", "Hugging Face", "Gemini"]
PROVIDER_ALIASES = {
    "local": "Local Mock LLM",
    "mock": "Local Mock LLM",
    "local mock llm": "Local Mock LLM",
    "openai": "OpenAI",
    "huggingface": "Hugging Face",
    "hugging face": "Hugging Face",
    "hf": "Hugging Face",
    "gemini": "Gemini",
    "google": "Gemini",
}


@dataclass
class LLMResponse:
    provider: str
    used_real_model: bool
    text: str
    error: str | None = None


class LLMProvider:
    """Optional real LLM bridge with a reliable offline fallback."""

    def __init__(self, provider: ProviderName | str = "Local Mock LLM") -> None:
        self.provider = normalize_provider(provider)
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-5.5")
        self.huggingface_model = os.getenv("HF_MODEL", "google/flan-t5-base")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, prompt: str, fallback: str) -> LLMResponse:
        if self.provider == "OpenAI":
            return self._openai(prompt, fallback)
        if self.provider == "Hugging Face":
            return self._huggingface(prompt, fallback)
        if self.provider == "Gemini":
            return self._gemini(prompt, fallback)
        return LLMResponse(provider="Local Mock LLM", used_real_model=False, text=fallback)

    def status(self) -> dict[str, str]:
        return {
            "selected_provider": self.provider,
            "openai_key": "available" if os.getenv("OPENAI_API_KEY") else "not configured",
            "openai_model": self.openai_model,
            "huggingface_token": "available" if os.getenv("HF_API_TOKEN") else "not configured",
            "huggingface_model": self.huggingface_model,
            "gemini_key": "available" if os.getenv("GEMINI_API_KEY") else "not configured",
            "gemini_model": self.gemini_model,
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

    def _gemini(self, prompt: str, fallback: str) -> LLMResponse:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return LLMResponse("Gemini", False, fallback, "GEMINI_API_KEY is not configured.")
        try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
            response = requests.post(
                endpoint,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=35,
            )
            response.raise_for_status()
            payload = response.json()
            candidates = payload.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text = "\n".join(str(part.get("text", "")) for part in parts).strip()
            return LLMResponse("Gemini", True, text or fallback)
        except Exception as exc:  # pragma: no cover - depends on external API availability
            return LLMResponse("Gemini", False, fallback, str(exc))


def normalize_provider(provider: str | None) -> ProviderName:
    cleaned = (provider or "Local Mock LLM").strip().lower()
    return PROVIDER_ALIASES.get(cleaned, "Local Mock LLM")  # type: ignore[return-value]
