"""Photo descriptions: the cache, the generation loop, and the build using them as the vision allergen source."""

import base64
import csv
import io
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

from food_concierge.config import REPO_ROOT, Settings, get_settings
from food_concierge.ingestion import descriptions
from food_concierge.ingestion.build import build_catalog, source_files
from food_concierge.ingestion.descriptions import (
    DESCRIBE_PX,
    DESCRIPTIONS_FILE,
    ImageDescription,
    Variant,
    describe_missing,
    read_descriptions,
    write_descriptions,
)
from food_concierge.ingestion.schemas import PHOTO_COLUMNS
from food_concierge.models.fake import ScriptedChatModel

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "catalog"
LEXICON = REPO_ROOT / "data" / "lexicon" / "allergens.yaml"
SHA_A, SHA_B = "a" * 64, "b" * 64


def jpeg(width: int = 2000, height: int = 1500) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (200, 120, 40)).save(out, format="JPEG", exif=b"Exif\x00\x00GPS")
    return out.getvalue()


def entry(sha256: str = SHA_A, variant: Variant = Variant.NAME_FREE, text: str = "A curry.") -> ImageDescription:
    return ImageDescription(
        sha256=sha256, variant=variant, model="m", prompt_version=1, text=text, latency_ms=5
    )  # fmt: skip


def test_cache_round_trips_sorted_and_later_lines_win(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    assert read_descriptions(path) == {}

    write_descriptions(path, [entry(SHA_B), entry(SHA_A, Variant.NAMED), entry(SHA_A)])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + entry(SHA_B, text="Rice.").model_dump_json() + "\n")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [line[12:13] for line in lines[:3]] == ["a", "a", "b"]  # sorted by hash, then variant
    assert lines[0].index('"name_free"') > 0
    assert read_descriptions(path)[(SHA_B, Variant.NAME_FREE)].text == "Rice."


def reply(text: str) -> AIMessage:
    return AIMessage(content=text)


def test_describe_missing_describes_each_photo_once_and_resumes(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    model = ScriptedChatModel(script=[reply("Orange  curry\nwith cream."), reply("Paneer curry."), reply("Bread.")])
    loads: list[str] = []

    def load(sha256: str) -> bytes:
        loads.append(sha256)
        return jpeg()

    photos = [(SHA_A, "Paneer Butter Masala"), (SHA_A, "Paneer Makhani"), (SHA_B, "Butter Naan")]
    new = describe_missing(photos, load, model, "qwen", path, limit=1)
    again = describe_missing(photos, load, ScriptedChatModel(script=[reply("Naan."), reply("Butter Naan.")]), "q", path)

    assert [(d.sha256, d.variant, d.text) for d in new] == [
        (SHA_A, Variant.NAME_FREE, "Orange curry with cream."),  # whitespace collapsed
        (SHA_A, Variant.NAMED, "Paneer curry."),
    ]
    assert [d.sha256 for d in again] == [SHA_B, SHA_B]  # the earlier run's photo is not described again
    assert loads == [SHA_A, SHA_B]
    name_free, named = (call[0] for call in model.calls[:2])
    assert isinstance(name_free, HumanMessage)
    assert isinstance(named, HumanMessage)
    assert "Paneer" not in name_free.content[0]["text"]  # type: ignore[index]
    assert "This photo shows Paneer Butter Masala." in named.content[0]["text"]  # type: ignore[index]
    sent = named.content[1]["image_url"]["url"]  # type: ignore[index]
    with Image.open(io.BytesIO(base64.b64decode(sent.split(",", 1)[1]))) as image:
        assert max(image.size) == DESCRIBE_PX
        assert "exif" not in image.info
    assert len(read_descriptions(path)) == 4


def test_a_new_prompt_version_regenerates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "d.jsonl"
    write_descriptions(path, [entry(), entry(variant=Variant.NAMED)])
    monkeypatch.setattr(descriptions, "PROMPT_VERSION", 2)

    new = describe_missing(
        [(SHA_A, "Dish")], lambda _: jpeg(), ScriptedChatModel(script=[reply("x"), reply("y")]), "m", path
    )

    assert [d.prompt_version for d in new] == [2, 2]


def test_only_the_stale_variant_is_regenerated(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    write_descriptions(path, [entry()])  # name-free is current; named is missing
    model = ScriptedChatModel(script=[reply("Named.")])

    new = describe_missing([(SHA_A, "Dish")], lambda _: jpeg(), model, "m", path)

    assert [d.variant for d in new] == [Variant.NAMED]
    assert len(model.calls) == 1


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(FIXTURE, tmp_path / "raw")
    (tmp_path / "lexicon").mkdir()
    shutil.copy(LEXICON, tmp_path / "lexicon" / "allergens.yaml")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    return tmp_path


def test_build_uses_name_free_descriptions_as_the_vision_source(data_dir: Path) -> None:
    with (data_dir / "raw" / "attributions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PHOTO_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "fx003",
                "file_page_url": "https://commons.wikimedia.org/wiki/File:Tikka.jpg",
                "file_url": "https://upload.wikimedia.org/a/b/Tikka.jpg",
                "author": "Cook",
                "licence": "CC0 1.0",
                "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "sha256": SHA_A,
                "width": 900,
                "height": 600,
                "bytes": 1000,
            }
        )
    settings = Settings()
    cache = settings.processed_dir / DESCRIPTIONS_FILE
    write_descriptions(
        cache,
        [
            entry(text="Grilled chicken pieces sprinkled with sesame seeds."),
            entry(variant=Variant.NAMED, text="Chicken tikka with peanuts."),  # the named variant is not a source
            entry(SHA_B, text="An unused photo with cheese."),  # not in the catalog: not stored
        ],
    )

    first = build_catalog(settings)
    with closing(sqlite3.connect(first.output)) as db:
        vision = db.execute("SELECT item_id, allergen, level FROM allergen_tags WHERE source = 'vision'").fetchall()
        stored = db.execute("SELECT sha256, variant FROM descriptions ORDER BY variant").fetchall()
    write_descriptions(cache, [entry(text="Plain grilled chicken.")])

    assert vision == [("fx003", "sesame", "may_contain")]
    assert stored == [(SHA_A, "name_free"), (SHA_A, "named")]
    assert source_files(settings)[-1] == cache
    assert build_catalog(settings).rebuilt  # descriptions are an input
