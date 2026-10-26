"""OpenAI and every OpenAI-compatible API (OpenRouter, Mistral, LM Studio, vLLM, ...).

The same Chat Completions code works for all of them: only the base URL and the
API key change. The presets below cover common services; "compat" lets you
point at any other server through environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..documents import Part
from .base import GenerationSettings, Provider, ProviderResponse, require_env, require_package


@dataclass(frozen=True)
class Preset:
    base_url: Optional[str]           # None = official OpenAI endpoint
    key_envs: tuple[str, ...]          # env vars holding the API key
    pdf_native: bool                   # accepts PDF files directly?
    response_format: str               # "json_schema" | "json_object" | "none"
    base_url_env: Optional[str] = None  # env var that overrides base_url
    key_optional: bool = False         # local servers usually need no key
    local: bool = False                # runs on this machine


PRESETS: dict[str, Preset] = {
    "openai": Preset(None, ("OPENAI_API_KEY",), pdf_native=True, response_format="json_schema"),
    "openrouter": Preset("https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",),
                         pdf_native=False, response_format="json_schema"),
    "mistral": Preset("https://api.mistral.ai/v1", ("MISTRAL_API_KEY",),
                      pdf_native=False, response_format="json_object"),
    "lmstudio": Preset("http://localhost:1234/v1", ("LMSTUDIO_API_KEY",), pdf_native=False,
                       response_format="json_schema", base_url_env="LMSTUDIO_BASE_URL", key_optional=True,
                       local=True),
    "compat": Preset(None, ("OPENAI_COMPAT_API_KEY",), pdf_native=False,
                     response_format="json_object", base_url_env="OPENAI_COMPAT_BASE_URL", key_optional=True),
}


class OpenAICompatibleProvider(Provider):
    def __init__(self, preset_name: str, model: str, settings: GenerationSettings | None = None):
        super().__init__(model, settings)
        self.name = preset_name
        self.preset = PRESETS[preset_name]
        self.pdf_native = self.preset.pdf_native
        self.local = self.preset.local

    def _client(self):
        openai = require_package("openai", "openai")
        base_url = os.environ.get(self.preset.base_url_env or "", "") or self.preset.base_url
        if not base_url and self.preset.base_url_env:
            base_url = require_env(self.preset.base_url_env)  # "compat" has no default URL
        if self.preset.key_optional:
            api_key = next((os.environ[k] for k in self.preset.key_envs if os.environ.get(k)), "not-needed")
        else:
            api_key = require_env(*self.preset.key_envs)
        return openai.OpenAI(api_key=api_key, base_url=base_url, timeout=self.settings.timeout_s)

    def generate(
        self, parts: list[Part], prompt: str, schema: dict[str, Any], source: Path | None = None
    ) -> ProviderResponse:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for i, part in enumerate(parts):
            data_url = f"data:{part.mime};base64,{part.b64}"
            if part.mime == "application/pdf":
                content.append({"type": "file", "file": {"filename": f"bill-{i}.pdf", "file_data": data_url}})
            else:
                content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})

        kwargs: dict[str, Any] = {}
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        # Official OpenAI renamed the parameter; most compatible servers still use the old name.
        tokens_param = "max_completion_tokens" if self.name == "openai" else "max_tokens"
        kwargs[tokens_param] = self.settings.max_tokens

        fmt = os.environ.get("OPENAI_RESPONSE_FORMAT", self.preset.response_format)
        if fmt == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "bill_data", "schema": schema},
            }
        elif fmt == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        completion = self._client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            **kwargs,
        )
        usage = completion.usage
        return ProviderResponse(
            completion.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
