"""User photos: validated, shrunk to the catalog's size, described once, and answered in words when they cannot be."""

import base64
import io

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image
from pydantic import SecretStr

from food_concierge import errors
from food_concierge.config import Settings
from food_concierge.ingestion.descriptions import Variant, prompt_for
from food_concierge.models.fake import ScriptedChatModel
from food_concierge.models.usage import PHOTO_LIMIT
from food_concierge.services.vision import DESCRIBE_IN_WORDS, VisionService, build_vision_service

HOUR = 3600.0


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def jpeg(width: int = 4000, height: int = 3000, color: str = "orange") -> bytes:
    exif = Image.Exif()
    exif[0x010F] = "SecretPhone X"
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def sent_image(model: ScriptedChatModel, call: int = 0) -> Image.Image:
    (message,) = model.calls[call]
    assert isinstance(message, HumanMessage)
    image_block = message.content[1]
    assert isinstance(image_block, dict)
    data = base64.b64decode(image_block["image_url"]["url"].split(",", 1)[1])
    return Image.open(io.BytesIO(data))


def service(*replies: AIMessage | Exception, clock: Clock | None = None, **overrides: object) -> VisionService:
    settings = Settings(_env_file=None).model_copy(update=overrides)
    model = ScriptedChatModel(script=list(replies))
    return VisionService(model, "vision-model", settings, clock=clock or Clock())


def model_of(vision: VisionService) -> ScriptedChatModel:
    model = vision._model
    assert isinstance(model, ScriptedChatModel)
    return model


def test_a_phone_photo_is_shrunk_cleaned_and_described_name_free() -> None:
    vision = service(AIMessage(content="  A bowl of\n orange curry. "))

    result = vision.describe(jpeg())

    assert result.text == "A bowl of orange curry."
    assert result.notice is None
    assert not result.cached
    sent = sent_image(model_of(vision))
    assert sent.size == (1024, 768)
    assert not sent.getexif()
    (message,) = model_of(vision).calls[0]
    assert message.content[0] == {"type": "text", "text": prompt_for(Variant.NAME_FREE, "")}


def test_the_size_is_a_setting() -> None:
    vision = service(AIMessage(content="curry"), vision_image_px=768)

    vision.describe(jpeg())

    assert sent_image(model_of(vision)).size == (768, 576)


def test_the_same_photo_is_described_once_until_the_ttl_ends() -> None:
    clock = Clock()
    vision = service(AIMessage(content="first"), AIMessage(content="second"), clock=clock, cache_ttl_hours=2)
    photo = jpeg()

    assert vision.describe(photo).text == "first"
    clock.now += 2 * HOUR - 1
    again = vision.describe(photo)
    clock.now += 1
    expired = vision.describe(photo)

    assert (again.text, again.cached) == ("first", True)
    assert (expired.text, expired.cached) == ("second", False)
    assert len(model_of(vision).calls) == 2


def test_the_cache_is_bounded_least_recently_used_first() -> None:
    vision = service(*(AIMessage(content=f"dish {i}") for i in range(4)), vision_cache_entries=2)
    red, green, blue = (jpeg(64, 64, color) for color in ("red", "green", "blue"))

    vision.describe(red)
    vision.describe(green)
    vision.describe(red)  # cached; green is now the oldest
    vision.describe(blue)  # evicts green

    assert vision.describe(red).cached
    assert not vision.describe(green).cached
    assert len(model_of(vision).calls) == 4


@pytest.mark.parametrize(
    ("failure", "notice"),
    [
        (errors.ProviderRateLimitedError(provider="groq"), DESCRIBE_IN_WORDS),
        (errors.BudgetExceededError(PHOTO_LIMIT, provider="groq"), PHOTO_LIMIT),
    ],
)
def test_a_failed_description_asks_for_words_and_is_not_cached(failure: Exception, notice: str) -> None:
    vision = service(failure, AIMessage(content="curry"))
    photo = jpeg()

    result = vision.describe(photo)

    assert (result.text, result.notice) == (None, notice)
    assert vision.describe(photo).text == "curry"


def test_a_blank_description_asks_for_words() -> None:
    vision = service(AIMessage(content=" \n "), AIMessage(content="curry"))

    result = vision.describe(jpeg())

    assert (result.text, result.notice) == (None, DESCRIBE_IN_WORDS)
    assert vision.describe(jpeg()).text == "curry"


def test_unacceptable_uploads_are_refused_before_the_model() -> None:
    vision = service(max_image_mb=0.01)

    with pytest.raises(errors.InputValidationError, match="larger than"):
        vision.describe(b"x" * 20_000)
    with pytest.raises(errors.InputValidationError, match="not a readable image"):
        vision.describe(b"not an image")
    assert model_of(vision).calls == []


def test_built_from_settings_over_the_vision_chain() -> None:
    settings = Settings(_env_file=None, groq_api_key=SecretStr("gsk-test"), groq_vision_model="groq-vision")

    vision = build_vision_service(settings)

    assert vision._version.startswith("groq-vision|")
    with pytest.raises(errors.ConfigError):
        build_vision_service(Settings(_env_file=None, groq_vision_model=""))
