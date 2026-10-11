"""Image queries from altered catalog photos (Phase 3 plan §2, audits A4 and A7).

A user's photo is never one of the catalog's, so a catalog photo searched as itself leaks the answer. Each query
here alters a catalog photo with 2-3 seeded transforms, describes it as production would (name-free), and searches
with that text. Three rungs show how much each shortcut flatters search:

- ``L0``: the named description of the unaltered photo (Phase 2's local model): what the original approach measured;
- ``L1``: the name-free description of the unaltered photo: the sanity check;
- ``L2``: the name-free description of the altered photo: the reported lower bound.

The altered images are not committed. Each query records its transforms and the altered image's SHA-256, so anyone
with the photos regenerates them; a different SHA-256 (another Pillow or JPEG encoder) is logged as a warning.
"""

from __future__ import annotations

import io
import json
import logging
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, NamedTuple

from PIL import Image, ImageEnhance, ImageFilter
from pydantic import BaseModel, ConfigDict, Field

from food_concierge.evaluation.dataset import Split
from food_concierge.ingestion.descriptions import DESCRIBE_PX, DescriptionKey, ImageDescription, Variant
from food_concierge.ingestion.photos import make_thumbnail, sha256

log = logging.getLogger(__name__)

TransformName = Literal["crop", "rotate", "colour", "blur", "backdrop", "jpeg"]
Rung = Literal["L0", "L1", "L2"]
ORDER: tuple[TransformName, ...] = ("crop", "rotate", "colour", "blur", "backdrop", "jpeg")  # applied in this order
BACKDROPS = ((236, 232, 224), (40, 44, 52), (178, 34, 34), (90, 120, 80), (250, 250, 250))


class Transform(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: TransformName
    params: dict[str, float]


class RungDescription(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    model: str
    prompt_version: int
    latency_ms: int = Field(ge=0)
    tokens: int = Field(default=0, ge=0)


class ImageQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^i\d{3}$")
    photo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dish_keys: tuple[str, ...]  # every dish whose items use this photo: the relevant items
    split: Split
    transforms: tuple[Transform, ...]
    altered_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rungs: dict[Rung, RungDescription] = {}


class Described(NamedTuple):
    text: str
    latency_ms: int
    tokens: int


def draw_transforms(photo_sha256: str, seed: int) -> tuple[Transform, ...]:
    """Two or three distinct transforms with parameters, the same for the same photo and seed."""
    rng = random.Random(f"{seed}:{photo_sha256}")  # noqa: S311 - reproducible alterations, not security
    names = rng.sample(ORDER, rng.choice((2, 3)))
    params: dict[TransformName, Callable[[], dict[str, float]]] = {
        "crop": lambda: {"area": rng.uniform(0.6, 0.9), "x": rng.random(), "y": rng.random()},
        "rotate": lambda: {"degrees": rng.uniform(-25, 25)},
        "colour": lambda: {
            "brightness": rng.uniform(0.7, 1.3),
            "contrast": rng.uniform(0.7, 1.3),
            "saturation": rng.uniform(0.6, 1.4),
        },
        "blur": lambda: {"radius": rng.uniform(1.0, 3.0)},
        "backdrop": lambda: {"scale": rng.uniform(0.55, 0.8), "colour": float(rng.randrange(len(BACKDROPS)))},
        "jpeg": lambda: {"quality": float(rng.randint(30, 60))},
    }
    drawn = {name: params[name]() for name in names}
    return tuple(Transform(name=name, params=_rounded(drawn[name])) for name in ORDER if name in drawn)


def _rounded(params: Mapping[str, float]) -> dict[str, float]:
    return {key: round(value, 4) for key, value in params.items()}


def apply_transforms(original: bytes, transforms: Sequence[Transform]) -> bytes:
    """The altered photo at describe size: JPEG when re-compression is one of the transforms, else lossless PNG."""
    with Image.open(io.BytesIO(make_thumbnail(original, DESCRIBE_PX))) as opened:
        image = opened.convert("RGB")
    quality: int | None = None
    for transform in transforms:
        p = transform.params
        if transform.name == "crop":
            width, height = image.size
            side = p["area"] ** 0.5
            w, h = round(width * side), round(height * side)
            left, top = round((width - w) * p["x"]), round((height - h) * p["y"])
            image = image.crop((left, top, left + w, top + h))
        elif transform.name == "rotate":
            image = image.rotate(p["degrees"], resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(128,) * 3)
        elif transform.name == "colour":
            image = ImageEnhance.Brightness(image).enhance(p["brightness"])
            image = ImageEnhance.Contrast(image).enhance(p["contrast"])
            image = ImageEnhance.Color(image).enhance(p["saturation"])
        elif transform.name == "blur":
            image = image.filter(ImageFilter.GaussianBlur(p["radius"]))
        elif transform.name == "backdrop":
            # True background replacement needs segmentation; shrinking onto a new canvas approximates a new setting.
            canvas = Image.new("RGB", image.size, BACKDROPS[int(p["colour"])])
            small = image.resize((round(image.width * p["scale"]), round(image.height * p["scale"])))
            canvas.paste(small, ((image.width - small.width) // 2, (image.height - small.height) // 2))
            image = canvas
        else:
            quality = int(p["quality"])
    out = io.BytesIO()
    if quality is None:
        image.save(out, format="PNG", optimize=False)
    else:
        image.save(out, format="JPEG", quality=quality)
    return out.getvalue()


def split_by_dish(dish_keys: Iterable[str], seed: int) -> dict[str, Split]:
    """Half the dishes to dev and half to test, so a dish never appears on both sides."""
    keys = sorted(set(dish_keys))
    random.Random(f"{seed}:images").shuffle(keys)  # noqa: S311 - reproducible split, not security
    return {key: "dev" if position % 2 == 0 else "test" for position, key in enumerate(keys)}


def build_queries(
    photos: Mapping[str, Sequence[str]],
    load_image: Callable[[str], bytes],
    named: Mapping[DescriptionKey, ImageDescription],
    seed: int,
    existing: Mapping[str, ImageQuery] | None = None,
    unaltered: Mapping[str, RungDescription] | None = None,
) -> list[ImageQuery]:
    """One query per photo (SHA-256 → dish keys), with L0 from the local cache; rungs already described are kept.

    ``unaltered`` holds name-free descriptions of unaltered photos already made by the hosted model (the Phase 2
    vision measurement), reused as L1 instead of paying for them again.
    """
    existing = existing or {}
    unaltered = unaltered or {}
    sides = split_by_dish((keys[0] for keys in photos.values()), seed)
    queries = []
    for number, sha in enumerate(sorted(photos), start=1):
        transforms = draw_transforms(sha, seed)
        previous = existing.get(sha)
        rungs = dict(previous.rungs) if previous else {}
        local = named.get((sha, Variant.NAMED))
        if "L1" not in rungs and sha in unaltered:
            rungs["L1"] = unaltered[sha]
        if local is not None:
            rungs["L0"] = RungDescription(
                text=local.text, model=local.model, prompt_version=local.prompt_version, latency_ms=local.latency_ms
            )
        queries.append(
            ImageQuery(
                id=f"i{number:03d}",
                photo_sha256=sha,
                dish_keys=tuple(photos[sha]),
                split=sides[photos[sha][0]],
                transforms=transforms,
                altered_sha256=(
                    previous.altered_sha256 if previous else sha256(apply_transforms(load_image(sha), transforms))
                ),
                rungs=rungs,
            )
        )
    return queries


def describe_queries(
    queries: Sequence[ImageQuery],
    load_image: Callable[[str], bytes],
    describe: Callable[[bytes], Described],
    model: str,
    prompt_version: int,
    path: Path,
    limit: int | None = None,
) -> list[ImageQuery]:
    """Fill the missing L1 (unaltered) and L2 (altered) descriptions, saving after each photo; resumable."""
    done = list(queries)
    todo = [n for n, q in enumerate(done) if not {"L1", "L2"} <= q.rungs.keys()]
    for position in todo[:limit]:
        query = done[position]
        original = load_image(query.photo_sha256)
        rungs = dict(query.rungs)
        if "L1" not in rungs:
            rungs["L1"] = _rung(describe(original), model, prompt_version)
        if "L2" not in rungs:
            altered = apply_transforms(original, query.transforms)
            if sha256(altered) != query.altered_sha256:
                log.warning("altered_image_differs query=%s (a different image encoder?)", query.id)
            rungs["L2"] = _rung(describe(altered), model, prompt_version)
        done[position] = query.model_copy(update={"rungs": rungs})
        write_queries(path, done)
    return done


def _rung(described: Described, model: str, prompt_version: int) -> RungDescription:
    return RungDescription(
        text=described.text,
        model=model,
        prompt_version=prompt_version,
        latency_ms=described.latency_ms,
        tokens=described.tokens,
    )


def read_queries(path: Path) -> dict[str, ImageQuery]:
    """Queries by photo SHA-256; an absent file is an empty set."""
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {q.photo_sha256: q for q in (ImageQuery.model_validate_json(line) for line in lines)}


def write_queries(path: Path, queries: Iterable[ImageQuery]) -> None:
    """One query per line in id order, LF line ends, so the file diffs cleanly."""
    lines = [json.dumps(q.model_dump(mode="json"), ensure_ascii=False) for q in sorted(queries, key=lambda q: q.id)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")
