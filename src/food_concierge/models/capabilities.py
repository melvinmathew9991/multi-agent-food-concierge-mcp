"""What each provider's endpoint supports, so the router asks for what it can deliver.

Values are provisional: they follow each provider's documentation for the
endpoint the router uses (OpenAI-compatible for Groq, Gemini and Ollama), and
the Phase 1 model profile measures them per model before ADR-0006 fixes them.
Where documentation and practice may differ, the safer choice is listed: an
unsupported parameter is a 400, and a 400 does not fall back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from food_concierge.config import ProviderName

StructuredMethod = Literal["function_calling", "json_schema", "json_mode"]


@dataclass(frozen=True)
class ProviderCapabilities:
    tools: bool
    # How ``with_structured_output`` asks this endpoint for typed output; None uses the model's default.
    structured_method: StructuredMethod | None
    vision: bool  # the endpoint accepts images (a vision model must still be configured)
    streaming: bool
    usage: bool  # replies report token usage
    seed: bool  # the endpoint accepts a sampling seed


CAPABILITIES: dict[ProviderName, ProviderCapabilities] = {
    # JSON-schema response formats exist on only some Groq and Gemini models; tool calling works on all of them.
    "groq": ProviderCapabilities(
        tools=True, structured_method="function_calling", vision=True, streaming=True, usage=True, seed=True
    ),
    "gemini": ProviderCapabilities(
        tools=True, structured_method="function_calling", vision=True, streaming=True, usage=True, seed=False
    ),
    # Ollama constrains decoding to the schema, which small local models need more than tool calling.
    "ollama": ProviderCapabilities(
        tools=True, structured_method="json_schema", vision=True, streaming=True, usage=True, seed=True
    ),
    "openai": ProviderCapabilities(
        tools=True, structured_method="json_schema", vision=True, streaming=True, usage=True, seed=True
    ),
    # Converse has no seed parameter.
    "bedrock": ProviderCapabilities(
        tools=True, structured_method="function_calling", vision=True, streaming=True, usage=True, seed=False
    ),
    "fake": ProviderCapabilities(
        tools=True, structured_method=None, vision=True, streaming=False, usage=True, seed=False
    ),
}
