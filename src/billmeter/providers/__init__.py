"""Provider registry: turn a "provider:model" string into a Provider instance.

The format is always `provider:model`. Everything after the FIRST ':' is the
model name, so model names that contain ':' work too (common with Ollama tags).
BillMeter does not choose a model for you: use any model your provider offers.

    anthropic:<model>
    openai:<model>
    gemini:<model>
    ollama:<model>[:<tag>]        local, no key
    openrouter:<vendor>/<model>
    mistral:<model>
    lmstudio:<model>              local, no key
    compat:<model>                any OpenAI-compatible server (see .env.example)
"""

from __future__ import annotations

from collections.abc import Callable

from .anthropic_provider import AnthropicProvider
from .base import GenerationSettings, Provider, ProviderConfigError, ProviderResponse
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .openai_provider import PRESETS, OpenAICompatibleProvider

#: Each entry builds a Provider from (model, settings).
NATIVE: dict[str, Callable[..., Provider]] = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}

PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "ollama": "OLLAMA_HOST (optional)",
    "lmstudio": "LMSTUDIO_BASE_URL (optional)",
    "compat": "OPENAI_COMPAT_BASE_URL",
}

# Where each provider lists its current models (model names change often).
MODEL_LISTS = {
    "anthropic": "https://docs.claude.com/en/docs/about-claude/models/overview",
    "openai": "https://platform.openai.com/docs/models",
    "gemini": "https://ai.google.dev/gemini-api/docs/models",
    "ollama": "https://ollama.com/search?c=vision",
    "openrouter": "https://openrouter.ai/models",
    "mistral": "https://docs.mistral.ai/getting-started/models/",
    "lmstudio": "the model loaded in LM Studio",
    "compat": "your server's documentation",
}


def available_providers() -> list[str]:
    return sorted(set(NATIVE) | set(PRESETS))


def get_provider(spec: str, settings: GenerationSettings | None = None) -> Provider:
    if ":" not in spec:
        raise ValueError(f"Model must be 'provider:model', got '{spec}'. Providers: {available_providers()}")
    provider_name, model = spec.split(":", 1)
    if not model:
        raise ValueError(f"Missing model name after '{provider_name}:'. See: {MODEL_LISTS.get(provider_name, '')}")
    if provider_name in NATIVE:
        return NATIVE[provider_name](model, settings)
    if provider_name in PRESETS:
        return OpenAICompatibleProvider(provider_name, model, settings)
    raise ValueError(f"Unknown provider '{provider_name}'. Available: {available_providers()}")


__all__ = [
    "GenerationSettings", "MODEL_LISTS", "Provider", "ProviderConfigError", "ProviderResponse",
    "PROVIDER_ENV", "available_providers", "get_provider",
]
