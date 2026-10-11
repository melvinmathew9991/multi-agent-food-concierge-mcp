"""Image queries: seeded transforms, the split by dish, building rungs and resumable description; offline."""

import io
import logging
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from food_concierge.evaluation.image_queries import (
    ORDER,
    Described,
    ImageQuery,
    RungDescription,
    Transform,
    apply_transforms,
    build_queries,
    describe_queries,
    draw_transforms,
    read_queries,
    split_by_dish,
    write_queries,
)
from food_concierge.ingestion.descriptions import ImageDescription, Variant
from food_concierge.ingestion.photos import sha256

SHA_A = "a" * 64
SHA_B = "b" * 64


def photo(colour: tuple[int, int, int] = (200, 120, 40), size: tuple[int, int] = (640, 480)) -> bytes:
    image = Image.new("RGB", size, colour)
    image.paste((20, 160, 60), (200, 150, 440, 330))  # a "dish" in the middle, so crops and blur change pixels
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=95)
    return out.getvalue()


def open_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def test_transforms_are_seeded_distinct_and_in_canonical_order() -> None:
    drawn = draw_transforms(SHA_A, seed=7)

    assert drawn == draw_transforms(SHA_A, seed=7)
    assert drawn != draw_transforms(SHA_A, seed=8)
    assert 2 <= len(drawn) <= 3
    names = [t.name for t in drawn]
    assert names == sorted(names, key=ORDER.index)
    assert len(set(names)) == len(names)


@pytest.mark.parametrize(
    ("transform", "check"),
    [
        (Transform(name="crop", params={"area": 0.64, "x": 0.0, "y": 1.0}), lambda im: im.size == (512, 384)),
        (Transform(name="rotate", params={"degrees": 20.0}), lambda im: im.size[0] > 640),
        (
            Transform(name="backdrop", params={"scale": 0.5, "colour": 1.0}),
            lambda im: im.getpixel((2, 2)) == (40, 44, 52),
        ),
        (Transform(name="jpeg", params={"quality": 30.0}), lambda im: im.format == "JPEG"),
        (
            Transform(name="colour", params={"brightness": 1.3, "contrast": 1.0, "saturation": 1.0}),
            lambda im: im.getpixel((5, 5))[0] > 200,
        ),
        (Transform(name="blur", params={"radius": 3.0}), lambda im: im.format == "PNG"),
    ],
)
def test_each_transform_alters_the_photo(transform: Transform, check: Callable[[Image.Image], bool]) -> None:
    original = photo()
    altered = apply_transforms(original, [transform])

    assert altered != apply_transforms(original, [])
    assert check(open_image(altered))
    assert apply_transforms(original, [transform]) == altered  # deterministic, so its SHA-256 can be recorded


def test_split_by_dish_halves_the_dishes() -> None:
    sides = split_by_dish(["a", "b", "c", "d", "a"], seed=7)

    assert sorted(sides.values()).count("dev") == 2
    assert sides == split_by_dish(["d", "c", "b", "a"], seed=7)


def local(sha: str, variant: Variant, text: str) -> ImageDescription:
    return ImageDescription(
        sha256=sha, variant=variant, model="qwen2.5vl:7b", prompt_version=1, text=text, latency_ms=10
    )


def test_build_fills_cached_rungs_and_keeps_what_was_described() -> None:
    photos = {SHA_A: ["masala-dosa"], SHA_B: ["dal-makhani", "dal-makhani-2"]}
    named = {(SHA_A, Variant.NAMED): local(SHA_A, Variant.NAMED, "A masala dosa.")}
    hosted = {SHA_B: RungDescription(text="Lentils.", model="groq-model", prompt_version=1, latency_ms=5, tokens=9)}

    queries = build_queries(photos, lambda _: photo(), named, seed=7, unaltered=hosted)

    first, second = queries
    assert (first.id, second.id) == ("i001", "i002")
    assert first.rungs["L0"].text == "A masala dosa."
    assert "L1" not in first.rungs
    assert second.rungs["L1"].model == "groq-model"
    assert second.dish_keys == ("dal-makhani", "dal-makhani-2")
    assert first.altered_sha256 == sha256(apply_transforms(photo(), first.transforms))

    described = first.model_copy(update={"rungs": first.rungs | {"L2": hosted[SHA_B]}})
    only_a = {SHA_A: photos[SHA_A]}
    again = build_queries(only_a, lambda _: b"never read", named, seed=7, existing={SHA_A: described})
    assert again[0].rungs["L2"] == hosted[SHA_B]
    assert again[0].altered_sha256 == first.altered_sha256


def test_describe_fills_missing_rungs_resumably(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "image_queries.jsonl"
    queries = build_queries({SHA_A: ["x"], SHA_B: ["y"]}, lambda _: photo(), {}, seed=7)
    calls: list[int] = []

    def describe(image: bytes) -> Described:
        calls.append(len(image))
        return Described(f"description {len(calls)}", 100, 1900)

    done = describe_queries(queries, lambda _: photo(), describe, "groq-model", 1, path, limit=1)

    assert len(calls) == 2  # L1 and L2 for the first photo only
    assert set(done[0].rungs) == {"L1", "L2"}
    assert done[0].rungs["L2"].tokens == 1900
    assert not done[1].rungs
    assert read_queries(path)[SHA_A] == done[0]

    tampered = [done[0], done[1].model_copy(update={"altered_sha256": "0" * 64})]
    with caplog.at_level(logging.WARNING):
        finished = describe_queries(tampered, lambda _: photo(), describe, "groq-model", 1, path)
    assert len(calls) == 4
    assert "altered_image_differs query=i002" in caplog.text
    assert all({"L1", "L2"} <= q.rungs.keys() for q in finished)


def test_queries_round_trip_and_an_absent_file_is_empty(tmp_path: Path) -> None:
    query = ImageQuery(
        id="i001", photo_sha256=SHA_A, dish_keys=("x",), split="dev", transforms=(), altered_sha256=SHA_B
    )
    path = tmp_path / "q.jsonl"
    write_queries(path, [query])

    assert read_queries(path) == {SHA_A: query}
    assert read_queries(tmp_path / "missing.jsonl") == {}
