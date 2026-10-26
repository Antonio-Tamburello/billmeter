"""Claude (Anthropic API).

Structured output via a single tool whose input schema is our JSON Schema.
`tool_choice` is "auto", not the tool itself: current models reject a forced
tool choice with a 400, and the prompt asks for the tool by name instead.
(`output_config.format` is the other documented route, but it compiles the
schema into a grammar and our 17 nullable fields exceed its union limit.)
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from ..documents import Part
from .base import Provider, ProviderResponse, require_env, require_package

TOOL_NAME = "record_bill_data"


class AnthropicProvider(Provider):
    name = "anthropic"
    pdf_native = True

    def generate(
        self, parts: list[Part], prompt: str, schema: dict[str, Any], source: Path | None = None
    ) -> ProviderResponse:
        anthropic = require_package("anthropic", "anthropic")
        api_key = require_env("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key, timeout=self.settings.timeout_s)

        content: list[dict[str, Any]] = []
        for part in parts:
            block_type = "document" if part.mime == "application/pdf" else "image"
            content.append({
                "type": block_type,
                "source": {"type": "base64", "media_type": part.mime, "data": part.b64},
            })
        content.append({"type": "text", "text": f"{prompt}\n\nReturn the result by calling the `{TOOL_NAME}` tool."})

        kwargs: dict[str, Any] = {}
        if self.settings.temperature is not None and _accepts(client.messages.create, "temperature"):
            # Recent SDK/model versions dropped sampling parameters: send it only if supported.
            kwargs["temperature"] = self.settings.temperature
        if self.settings.effort and _accepts(client.messages.create, "output_config"):
            # Thinking is on by default on current models and its tokens come out of
            # max_tokens: an extraction that needs no reasoning can be cut off mid-thought.
            kwargs["output_config"] = {"effort": self.settings.effort}

        message = client.messages.create(
            model=self.model,
            max_tokens=self.settings.max_tokens,
            messages=[{"role": "user", "content": content}],
            tools=[{
                "name": TOOL_NAME,
                "description": "Record the data extracted from the utility bill.",
                "input_schema": schema,
            }],
            tool_choice={"type": "auto"},
            **kwargs,
        )

        # The tool call is the answer; if the model replied in prose instead,
        # fall back to its text (extractor.py knows how to find JSON in it).
        tool_input = next((b.input for b in message.content if b.type == "tool_use"), None)
        text = json.dumps(tool_input) if tool_input is not None else "".join(
            b.text for b in message.content if b.type == "text")
        return ProviderResponse(text, message.usage.input_tokens, message.usage.output_tokens)


def _accepts(function, name: str) -> bool:
    params = inspect.signature(function).parameters
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())
