"""Describe a user's photo of a dish: validate, downscale, describe, cache (architecture §4, #53).

- The upload is validated and re-encoded without metadata (``models/images.py``), and shrunk to
  ``vision_image_px`` on its longer side, the catalog's size, so a phone photo costs no more tokens
  than a catalog one. Only the catalog path resized before.
- It is described with the catalog's name-free prompt, the same text the image queries are scored on.
- Descriptions are kept in memory by the upload's SHA-256, the vision model and the prompt version and
  size, for ``cache_ttl_hours``, so sending the same photo again costs nothing. The photo itself is never
  kept. The cache is bounded (``vision_cache_entries``) and lives in this process only.
- If the model cannot describe it (a cap, a quota, an outage), the result carries no text and asks the
  user to describe the dish in words: a photo is a shortcut, never the only way in.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from food_concierge.config import Settings
from food_concierge.errors import InputValidationError, ProviderError
from food_concierge.ingestion.descriptions import PROMPT_VERSION, Variant, describe_data_uri, prompt_for
from food_concierge.models.images import clean_image
from food_concierge.models.router import get_chat_model, has_model
from food_concierge.models.usage import PHOTO_LIMIT

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

DESCRIBE_IN_WORDS = "The photo could not be described right now. Please describe the dish in words instead."


class PhotoDescription(BaseModel):
    """What the vision model saw; ``text`` is None when it could not say, and ``notice`` tells the user why."""

    model_config = ConfigDict(frozen=True)

    sha256: str
    text: str | None
    notice: str | None = None
    cached: bool = False
    latency_ms: int = 0


class VisionService:
    """Describes photos with one vision model; thread-safe."""

    def __init__(
        self,
        model: Runnable[LanguageModelInput, BaseMessage],
        model_id: str,
        settings: Settings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._model = model
        self._settings = settings
        self._clock = clock
        self._version = f"{model_id}|prompt {PROMPT_VERSION}|{settings.vision_image_px} px"
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def describe(self, data: bytes) -> PhotoDescription:
        """Describe one uploaded photo; raises ``InputValidationError`` for a file that is not an acceptable image."""
        if len(data) > self._settings.max_image_bytes:  # before hashing or decoding anything
            raise InputValidationError(f"The image is larger than {self._settings.max_image_mb:g} MB.")
        sha256 = hashlib.sha256(data).hexdigest()
        key = f"{sha256}|{self._version}"
        found = self._cached(key)
        if found is not None:
            return PhotoDescription(sha256=sha256, text=found, cached=True)
        image = clean_image(data, max_bytes=self._settings.max_image_bytes, max_px=self._settings.vision_image_px)
        try:
            text, latency_ms = describe_data_uri(self._model, image.data_uri, prompt_for(Variant.NAME_FREE, ""))
        except ProviderError as error:
            # The photo budget's refusal already asks for words; other failures get the general request.
            notice = PHOTO_LIMIT if error.message == PHOTO_LIMIT else DESCRIBE_IN_WORDS
            logger.warning("photo not described", extra={"error_code": error.code})
            return PhotoDescription(sha256=sha256, text=None, notice=notice)
        if not text:
            return PhotoDescription(sha256=sha256, text=None, notice=DESCRIBE_IN_WORDS, latency_ms=latency_ms)
        self._store(key, text)
        return PhotoDescription(sha256=sha256, text=text, latency_ms=latency_ms)

    def _cached(self, key: str) -> str | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            stored_at, text = entry
            if self._clock() - stored_at >= self._settings.cache_ttl_hours * 3600:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return text

    def _store(self, key: str, text: str) -> None:
        with self._lock:
            self._cache[key] = (self._clock(), text)
            self._cache.move_to_end(key)
            while len(self._cache) > self._settings.vision_cache_entries:
                self._cache.popitem(last=False)


def build_vision_service(settings: Settings) -> VisionService:
    """The service over the configured vision chain (Groq only, ADR-0007); cached under its first model."""
    model = get_chat_model("vision", settings)  # raises ConfigError when no provider has a vision model
    first = next(p for p in settings.provider_chain if has_model(settings, p, "vision"))
    return VisionService(model, settings.require_model_id(first, "vision"), settings)
