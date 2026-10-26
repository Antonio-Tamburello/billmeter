"""The single interface every AI provider implements.

The rest of the code (extractor, eval) only talks to `Provider.generate`, so
adding a new AI means adding one small class here, nothing else.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..documents import Part


class ProviderConfigError(RuntimeError):
    """Misconfiguration (missing key, missing SDK): retrying will not help."""


@dataclass
class ProviderResponse:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class GenerationSettings:
    temperature: Optional[float] = 0.0  # None = don't send it (some reasoning models reject it)
    max_tokens: int = 2048
    #: Reasoning depth for models that think before answering. Extraction needs no
    #: reasoning, and thinking tokens are billed as output and spend max_tokens, so a
    #: low value is both cheaper and safer. None = leave the model's own default.
    effort: Optional[str] = None
    timeout_s: float = 120.0


class Provider(ABC):
    #: name used in the "provider:model" spec, e.g. "anthropic"
    name: str = ""
    #: True if the API reads PDF files directly; otherwise pages are sent as images
    pdf_native: bool = False
    #: True if the model runs on this machine: one request at a time is faster than several
    local: bool = False

    def __init__(self, model: str, settings: GenerationSettings | None = None):
        self.model = model
        self.settings = settings or GenerationSettings()

    @property
    def spec(self) -> str:
        return f"{self.name}:{self.model}"

    @abstractmethod
    def generate(
        self, parts: list[Part], prompt: str, schema: dict[str, Any], source: Path | None = None
    ) -> ProviderResponse:
        """Send the document parts + prompt, return the model's JSON as text.

        `source` is the original file path; real providers ignore it (only the fake
        provider used by the tests reads it).
        """


def require_env(*names: str) -> str:
    """Return the first environment variable that is set, or explain what to set."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise ProviderConfigError(f"Missing API key: set {' or '.join(names)} in your .env file")


def require_package(module: str, extra: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:
        raise ProviderConfigError(
            f"Package for this provider is not installed. Run: uv sync --extra {extra}"
        ) from exc
