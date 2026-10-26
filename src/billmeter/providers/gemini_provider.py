"""Google Gemini (Gemini API with an API key from Google AI Studio)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..documents import Part
from .base import Provider, ProviderResponse, require_env, require_package


class GeminiProvider(Provider):
    name = "gemini"
    pdf_native = True

    def generate(
        self, parts: list[Part], prompt: str, schema: dict[str, Any], source: Path | None = None
    ) -> ProviderResponse:
        genai = require_package("google.genai", "gemini")
        types = genai.types
        client = genai.Client(api_key=require_env("GEMINI_API_KEY", "GOOGLE_API_KEY"))

        contents: list[Any] = [types.Part.from_bytes(data=p.data, mime_type=p.mime) for p in parts]
        contents.append(prompt)

        config = types.GenerateContentConfig(
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_tokens,
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        response = client.models.generate_content(model=self.model, contents=contents, config=config)
        usage = response.usage_metadata
        return ProviderResponse(
            response.text or "",
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )
