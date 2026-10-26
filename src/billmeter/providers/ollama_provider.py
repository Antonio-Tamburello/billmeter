"""Local models through Ollama (native /api/chat endpoint, no SDK needed).

Requires a model with vision support: pick one from https://ollama.com/search?c=vision
and download it with `ollama pull <model>`.
Ollama does not read PDFs, so pages are rendered to images before sending.
The `format` field constrains the output to our JSON Schema.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from ..documents import Part
from .base import Provider, ProviderConfigError, ProviderResponse


class OllamaProvider(Provider):
    name = "ollama"
    pdf_native = False
    local = True

    def generate(
        self, parts: list[Part], prompt: str, schema: dict[str, Any], source: Path | None = None
    ) -> ProviderResponse:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not host.startswith("http"):
            host = f"http://{host}"

        options: dict[str, Any] = {"num_predict": self.settings.max_tokens}
        if self.settings.temperature is not None:
            options["temperature"] = self.settings.temperature

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": [p.b64 for p in parts]}],
            "format": schema,
            "stream": False,
            "options": options,
        }
        try:
            response = httpx.post(f"{host}/api/chat", json=payload, timeout=self.settings.timeout_s)
        except httpx.ConnectError as exc:
            raise ProviderConfigError(f"Cannot reach Ollama at {host}. Is `ollama serve` running?") from exc
        if response.status_code == 404:
            raise ProviderConfigError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        response.raise_for_status()
        body = response.json()
        return ProviderResponse(
            body.get("message", {}).get("content", ""),
            body.get("prompt_eval_count"),
            body.get("eval_count"),
        )
