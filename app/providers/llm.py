import json
from collections.abc import Iterator
from functools import lru_cache
from typing import Protocol

import httpx

from app.core.config import settings


class LLM(Protocol):
    def generate(self, system: str, prompt: str, images: list[str] | None = None) -> str: ...

    def stream(
        self, system: str, prompt: str, images: list[str] | None = None
    ) -> Iterator[str]: ...


class OllamaLLM:
    """Calls a local Ollama server. 100% open-source/local generation."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.llm_model

    def _messages(self, system: str, prompt: str, images: list[str] | None) -> list[dict]:
        user_msg: dict = {"role": "user", "content": prompt}
        if images:
            # Ollama multimodal models accept base64 images on the message.
            user_msg["images"] = images
        return [
            {"role": "system", "content": system},
            user_msg,
        ]

    def generate(self, system: str, prompt: str, images: list[str] | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": self._messages(system, prompt, images),
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
            "options": {"temperature": 0.1},
        }
        # Generous timeout: the first call cold-loads the model into (V)RAM,
        # which is slow on modest GPUs before generation even starts.
        with httpx.Client(timeout=600) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"Ollama {resp.status_code}: {resp.text[:600]}")
            return resp.json()["message"]["content"]

    def stream(
        self, system: str, prompt: str, images: list[str] | None = None
    ) -> Iterator[str]:
        """Yields content tokens as Ollama produces them (newline-delimited JSON)."""
        payload = {
            "model": self.model,
            "messages": self._messages(system, prompt, images),
            "stream": True,
            "keep_alive": settings.ollama_keep_alive,
            "options": {"temperature": 0.1},
        }
        with httpx.Client(timeout=600) as client:
            with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"Ollama {resp.status_code}: {resp.read().decode()[:600]}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break


@lru_cache
def get_llm() -> LLM:
    return OllamaLLM()
