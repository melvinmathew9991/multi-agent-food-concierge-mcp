"""Images leave the process without metadata, in an accepted format and size."""

import base64
import io
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image, PngImagePlugin

from food_concierge import errors
from food_concierge.models import images
from food_concierge.models.images import clean_image, clean_message_images

MB = 1024 * 1024
DEVICE = "SecretPhone X"


def _jpeg_with_exif(size: tuple[int, int] = (40, 20), orientation: int = 6) -> bytes:
    exif = Image.Exif()
    exif[0x010F] = DEVICE  # Make
    exif[0x0112] = orientation  # 6: rotate 90 degrees to display
    exif.get_ifd(0x8825)[2] = (12.0, 58.0, 30.0)  # GPSLatitude
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def _encode(image: Image.Image, image_format: str, **options: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()


def test_jpeg_metadata_is_dropped_and_orientation_kept() -> None:
    source = _jpeg_with_exif()
    assert DEVICE.encode() in source
    assert Image.open(io.BytesIO(source)).getexif().get_ifd(0x8825)

    cleaned = clean_image(source, max_bytes=MB)

    reopened = Image.open(io.BytesIO(cleaned.data))
    assert DEVICE.encode() not in cleaned.data
    assert not reopened.getexif()
    assert "exif" not in reopened.info
    assert (cleaned.width, cleaned.height) == (20, 40)  # rotated as the EXIF orientation said
    assert cleaned.mime_type == "image/jpeg"
    assert cleaned.data_uri.startswith("data:image/jpeg;base64,")


def test_png_text_chunks_are_dropped_and_palette_transparency_kept() -> None:
    palette = Image.new("P", (8, 8), 0)
    palette.putpalette([255, 0, 0, 0, 255, 0])
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "priya@example.com")
    source = _encode(palette, "PNG", pnginfo=info, transparency=0)

    cleaned = clean_image(source, max_bytes=MB)

    reopened = Image.open(io.BytesIO(cleaned.data))
    assert b"priya@example.com" not in cleaned.data
    assert reopened.mode == "RGBA"
    assert reopened.getpixel((0, 0)) == (255, 0, 0, 0)
    assert cleaned.mime_type == "image/png"


def test_webp_and_cmyk_jpeg_are_accepted() -> None:
    webp = clean_image(_encode(Image.new("RGB", (4, 4), "blue"), "WEBP"), max_bytes=MB)
    cmyk = clean_image(_encode(Image.new("CMYK", (4, 4)), "JPEG"), max_bytes=MB)

    assert webp.mime_type == "image/webp"
    assert Image.open(io.BytesIO(cmyk.data)).mode == "RGB"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"x" * (MB + 1), "larger than 1 MB"),
        (_encode(Image.new("RGB", (4, 4)), "GIF"), "Only JPEG, PNG and WebP"),
        (b"not an image", "not a readable image"),
    ],
    ids=["too-large", "gif", "garbage"],
)
def test_unacceptable_files_are_refused(data: bytes, message: str) -> None:
    with pytest.raises(errors.InputValidationError, match=message):
        clean_image(data, max_bytes=MB)


def test_too_many_pixels_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images, "MAX_PIXELS", 10)

    with pytest.raises(errors.InputValidationError, match="too many pixels"):
        clean_image(_encode(Image.new("RGB", (4, 4)), "PNG"), max_bytes=MB)


def test_decompression_bombs_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(errors.InputValidationError, match="not a readable image"):
        clean_image(_encode(Image.new("RGB", (4, 4)), "PNG"), max_bytes=MB)


# ---------------------------------------------------------------------------
# Image blocks in messages
# ---------------------------------------------------------------------------

JPEG = _jpeg_with_exif()
JPEG_B64 = base64.b64encode(JPEG).decode()
JPEG_URI = f"data:image/jpeg;base64,{JPEG_B64}"


def _decoded(text: str) -> bytes:
    return base64.b64decode(text.split(",", 1)[-1])


@pytest.mark.parametrize(
    ("block", "read"),
    [
        ({"type": "image_url", "image_url": {"url": JPEG_URI, "detail": "low"}}, lambda b: b["image_url"]["url"]),
        ({"type": "image_url", "image_url": JPEG_URI}, lambda b: b["image_url"]),
        ({"type": "image", "base64": JPEG_B64, "mime_type": "image/jpeg"}, lambda b: b["base64"]),
        ({"type": "image", "source_type": "base64", "data": JPEG_B64, "mime_type": "image/jpeg"}, lambda b: b["data"]),
        ({"type": "image", "url": JPEG_URI}, lambda b: b["url"]),
    ],
    ids=["openai", "openai-bare-url", "standard-v1", "standard-v0", "standard-data-url"],
)
def test_inline_images_in_messages_are_cleaned(block: dict[str, Any], read: Any) -> None:
    message = HumanMessage(content=[{"type": "text", "text": "what is this?"}, block])

    (cleaned,) = clean_message_images([message], max_bytes=MB)

    assert isinstance(cleaned.content, list)
    sent = cleaned.content[1]
    assert isinstance(sent, dict)
    assert DEVICE.encode() not in _decoded(read(sent))
    assert cleaned.content[0] == {"type": "text", "text": "what is this?"}
    assert DEVICE.encode() in _decoded(read(block))  # the caller's message is not modified
    if sent.get("detail") or isinstance(sent.get("image_url"), dict):
        assert sent["image_url"]["detail"] == "low"


def test_messages_without_inline_images_pass_through_unchanged() -> None:
    remote = HumanMessage(content=[{"type": "image_url", "image_url": {"url": "https://example.org/dal.jpg"}}])
    standard_remote = HumanMessage(content=[{"type": "image", "url": "https://example.org/dal.jpg"}])
    text = HumanMessage("plain text")
    mixed = AIMessage(content=["a bare string block", {"type": "text", "text": "hi"}])

    messages = [remote, standard_remote, text, mixed]
    cleaned = clean_message_images(messages, max_bytes=MB)

    assert all(after is before for after, before in zip(cleaned, messages, strict=True))


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg,rawbytes"}},
        {"type": "image", "base64": "not base64!"},
    ],
    ids=["not-base64-uri", "bad-base64"],
)
def test_malformed_inline_images_are_refused(block: dict[str, Any]) -> None:
    with pytest.raises(errors.InputValidationError):
        clean_message_images([HumanMessage(content=[block])], max_bytes=MB)
