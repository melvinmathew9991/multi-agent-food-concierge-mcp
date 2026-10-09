"""Photo descriptions from a vision model, cached by image hash (Phase 2 plan, item 7; audit A7).

Two variants per photo:

- **name-free**: what the photo shows, without the dish's name. It feeds the allergen tagger as the third source
  (``may_contain``) and stands in for a user's photo in Phase 3's image queries.
- **named**: the same, told the dish's name, for comparing the two in Phase 3.

Descriptions are generated offline by ``scripts/describe_photos.py`` and cached in
``data/processed/image_descriptions.jsonl``, one JSON object per line. The cache is tracked, so the build, CI and
the deployed app never need a vision model. A description is regenerated only when its prompt version changes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from food_concierge.ingestion.photos import make_thumbnail
from food_concierge.ingestion.schemas import Photo
from food_concierge.models.images import clean_image

DESCRIPTIONS_FILE = "image_descriptions.jsonl"  # under Settings.processed_dir
PROMPT_VERSION = 1
DESCRIBE_PX = 1024  # at 480 px the model mistook a paneer curry for meat stew; 1024 px costs no more time


class Variant(StrEnum):
    NAME_FREE = "name_free"
    NAMED = "named"


_TASK = (
    "Describe the food in this photo for someone who cannot see it, in two or three sentences: what kind of dish "
    "it is, how it is served, and every ingredient you can actually see. Mention only what is visible."
)
PROMPTS: dict[Variant, str] = {
    Variant.NAME_FREE: _TASK + " Do not name the dish or guess its name, even if you recognise it.",
    Variant.NAMED: "This photo shows {dish}. " + _TASK,
}


class ImageDescription(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant: Variant
    model: str = Field(min_length=1)
    prompt_version: int = Field(ge=1)
    text: str = Field(min_length=1)
    latency_ms: int = Field(ge=0)


DescriptionKey = tuple[str, Variant]


def read_descriptions(path: Path) -> dict[DescriptionKey, ImageDescription]:
    """The cache, keyed by (image SHA-256, variant); a later line replaces an earlier one."""
    if not path.exists():
        return {}
    cache: dict[DescriptionKey, ImageDescription] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            description = ImageDescription.model_validate_json(line)
            cache[(description.sha256, description.variant)] = description
    return cache


def write_descriptions(path: Path, descriptions: Iterable[ImageDescription]) -> None:
    """Rewrite the cache sorted by hash and variant, so it diffs cleanly."""
    ordered = sorted(descriptions, key=lambda d: (d.sha256, d.variant.value))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(d.model_dump(mode="json"), ensure_ascii=False) for d in ordered]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def is_current(cache: Mapping[DescriptionKey, ImageDescription], sha256: str, variant: Variant) -> bool:
    found = cache.get((sha256, variant))
    return found is not None and found.prompt_version == PROMPT_VERSION


def vision_texts(photos: Iterable[Photo], cache: Mapping[DescriptionKey, ImageDescription]) -> dict[str, str]:
    """item_id → the name-free description of its photo: the allergen tagger's vision source."""
    texts: dict[str, str] = {}
    for photo in photos:
        found = cache.get((photo.sha256, Variant.NAME_FREE))
        if found is not None:
            texts[photo.item_id] = found.text
    return texts


def prompt_for(variant: Variant, dish: str) -> str:
    return PROMPTS[variant].format(dish=dish)


def describe_missing(
    photos: Iterable[tuple[str, str]],
    load_image: Callable[[str], bytes],
    model: BaseChatModel,
    model_name: str,
    path: Path,
    *,
    limit: int | None = None,
) -> list[ImageDescription]:
    """Describe every (SHA-256, dish name) photo that lacks a current description, saving after each photo.

    A photo used by two dishes is described once, named after the first dish. ``limit`` caps the number of photos
    described in this run. Returns the new descriptions; an interrupted run resumes where it stopped.
    """
    cache = read_descriptions(path)
    names: dict[str, str] = {}
    for sha256, dish in photos:
        names.setdefault(sha256, dish)
    new: list[ImageDescription] = []
    todo = [sha for sha in names if not all(is_current(cache, sha, variant) for variant in Variant)]
    for sha256 in todo[:limit]:
        image = load_image(sha256)
        for variant in Variant:
            if is_current(cache, sha256, variant):
                continue
            text, latency_ms = describe(model, image, prompt_for(variant, names[sha256]))
            description = ImageDescription(
                sha256=sha256,
                variant=variant,
                model=model_name,
                prompt_version=PROMPT_VERSION,
                text=text,
                latency_ms=latency_ms,
            )
            cache[(sha256, variant)] = description
            new.append(description)
        write_descriptions(path, cache.values())
    return new


def describe(model: BaseChatModel, image: bytes, prompt: str) -> tuple[str, int]:
    """One description of ``image`` (any original), sent at ``DESCRIBE_PX`` without metadata; text and latency."""
    resized = make_thumbnail(image, DESCRIBE_PX)
    data_uri = clean_image(resized, max_bytes=len(resized) + 1).data_uri
    message = HumanMessage(
        content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_uri}}]
    )
    started = time.perf_counter()
    reply = model.invoke([message])
    latency_ms = round((time.perf_counter() - started) * 1000)
    return " ".join(reply.text.split()), latency_ms
