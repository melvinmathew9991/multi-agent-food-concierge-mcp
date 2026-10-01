"""Images are re-encoded without metadata before they leave the process.

Phone photos carry EXIF: GPS position, device and time. Re-encoding the
pixels drops it, together with XMP, ICC and comments. The orientation EXIF
holds is applied to the pixels first, so the image still looks the same.

Every chat model the router builds cleans images in its messages before
sending them (``clean_message_images``), whatever the caller passed. Only
inline images (data URIs and base64 blocks) are cleaned; an http(s) URL is
fetched by the provider, not by us, and passes through unchanged.
"""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from PIL import Image, ImageOps, UnidentifiedImageError

from food_concierge.errors import InputValidationError

# Pillow format name -> MIME type. Formats with animation or scripting (GIF, SVG) are refused.
MIME_TYPES: dict[str, str] = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
# Large enough for any phone camera; refuses decompression bombs before pixels are decoded.
MAX_PIXELS = 40_000_000


@dataclass(frozen=True)
class CleanImage:
    data: bytes
    mime_type: str
    width: int
    height: int

    @property
    def data_uri(self) -> str:
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode('ascii')}"


def clean_image(data: bytes, *, max_bytes: int) -> CleanImage:
    """Validate an uploaded image and re-encode it without metadata, in its own format."""
    if len(data) > max_bytes:
        raise InputValidationError(f"The image is larger than {max_bytes // (1024 * 1024)} MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            source = Image.open(io.BytesIO(data))
            image_format = source.format or ""
            if image_format not in MIME_TYPES:
                raise InputValidationError("Only JPEG, PNG and WebP images are accepted.")
            if source.width * source.height > MAX_PIXELS:
                raise InputValidationError("The image has too many pixels.")
            pixels = ImageOps.exif_transpose(source)
    except InputValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning) as exc:
        raise InputValidationError("The file is not a readable image.") from exc

    if pixels.mode == "P":
        # Palette transparency lives in metadata, which is about to be dropped; move it into the pixels.
        pixels = pixels.convert("RGBA" if "transparency" in pixels.info else "RGB")
    if image_format == "JPEG" and pixels.mode not in ("RGB", "L"):
        pixels = pixels.convert("RGB")
    # A fresh image holds pixels only: no EXIF, XMP, ICC profile or text chunks to carry over.
    clean = Image.new(pixels.mode, pixels.size)
    clean.paste(pixels)
    buffer = io.BytesIO()
    options: dict[str, Any] = {"quality": 90} if image_format in ("JPEG", "WEBP") else {}
    clean.save(buffer, format=image_format, **options)
    return CleanImage(buffer.getvalue(), MIME_TYPES[image_format], clean.width, clean.height)


def _decode_base64(text: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InputValidationError("The image data is not valid base64.") from exc


def _clean_data_uri(uri: str, max_bytes: int) -> str:
    header, _, payload = uri.partition(",")
    if not header.endswith(";base64"):
        raise InputValidationError("Inline images must be base64 data URIs.")
    return clean_image(_decode_base64(payload), max_bytes=max_bytes).data_uri


def _clean_block(block: Any, max_bytes: int) -> Any:
    if not isinstance(block, dict):
        return block
    kind = block.get("type")
    if kind == "image_url":
        # OpenAI shape: {"type": "image_url", "image_url": {"url": ...}} (or the URL as a bare string).
        holder = block.get("image_url")
        url = holder.get("url") if isinstance(holder, dict) else holder
        if isinstance(url, str) and url.startswith("data:"):
            cleaned = _clean_data_uri(url, max_bytes)
            new_holder = {**holder, "url": cleaned} if isinstance(holder, dict) else cleaned
            return {**block, "image_url": new_holder}
        return block
    if kind == "image":
        # LangChain standard blocks: v1 {"base64": ...} and v0.3 {"source_type": "base64", "data": ...}.
        key = "base64" if "base64" in block else "data" if block.get("source_type") == "base64" else None
        if key is not None:
            image = clean_image(_decode_base64(str(block[key])), max_bytes=max_bytes)
            return {**block, key: base64.b64encode(image.data).decode("ascii"), "mime_type": image.mime_type}
        url = block.get("url")
        if isinstance(url, str) and url.startswith("data:"):
            return {**block, "url": _clean_data_uri(url, max_bytes)}
    return block


def clean_message_images(messages: Sequence[BaseMessage], *, max_bytes: int) -> list[BaseMessage]:
    """Messages with every inline image cleaned; messages without images are returned as they are."""
    cleaned: list[BaseMessage] = []
    for message in messages:
        if isinstance(message.content, list):
            content = [_clean_block(block, max_bytes) for block in message.content]
            if content != message.content:
                message = message.model_copy(update={"content": content})
        cleaned.append(message)
    return cleaned
