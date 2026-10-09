"""Commons photos: licence filtering, paced access, hash pinning, thumbnails, approvals and attributions."""

import csv
import importlib.util
import io
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
import respx
from PIL import Image

from food_concierge.config import REPO_ROOT
from food_concierge.errors import CatalogValidationError, PhotoSourceError
from food_concierge.ingestion.loader import ATTRIBUTIONS_FILE, load_catalog
from food_concierge.ingestion.photos import (
    API_URL,
    NONE,
    USER_AGENT,
    Candidate,
    CommonsClient,
    Licence,
    attribution,
    classify_licence,
    ensure_cached,
    make_thumbnail,
    original_path,
    parse_candidates,
    plain_text,
    read_approvals,
    review_rows,
    sha256,
    thumbnail_path,
)
from food_concierge.ingestion.schemas import PHOTO_COLUMNS, Photo

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "catalog"
FILE_URL = "https://upload.wikimedia.org/wikipedia/commons/a/a4/Dosa.jpg"


def jpeg(width: int = 900, height: int = 600) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (200, 120, 40)).save(out, format="JPEG")
    return out.getvalue()


def page(index: int, title: str, **overrides: Any) -> dict[str, Any]:
    meta = {
        "License": {"value": "cc-by-sa-4.0"},
        "LicenseShortName": {"value": "CC BY-SA 4.0"},
        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"},
        "Artist": {"value": '<a href="//commons.wikimedia.org/wiki/User:Cook">Cook &amp; Co</a>'},
        "Restrictions": {"value": ""},
    }
    meta |= overrides.pop("meta", {})
    info = {
        "url": f"https://upload.wikimedia.org/wikipedia/commons/x/{title}.jpg?utm_source=commons.wikimedia.org",
        "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{title}.jpg",
        "thumburl": f"https://thumb.wikimedia.org/x/330px-{title}.jpg?utm_content=thumbnail",
        "width": 1200,
        "height": 800,
        "size": 123456,
        "mime": "image/jpeg",
        "extmetadata": meta,
    } | overrides
    return {"index": index, "title": f"File:{title}.jpg", "imageinfo": [info]}


def photo(**changes: Any) -> Photo:
    data = jpeg()
    base: dict[str, Any] = {
        "item_id": "fx005",
        "file_page_url": "https://commons.wikimedia.org/wiki/File:Dosa.jpg",
        "file_url": FILE_URL,
        "author": "Cook",
        "licence": "CC BY-SA 4.0",
        "licence_url": "https://creativecommons.org/licenses/by-sa/4.0",
        "sha256": sha256(data),
        "width": 900,
        "height": 600,
        "bytes": len(data),
    }
    return Photo.model_validate(base | changes)


class FakeClient(CommonsClient):
    def __init__(self, files: dict[str, bytes] | None = None, results: list[Candidate] | None = None) -> None:
        super().__init__(http=httpx.Client(), min_interval_s=0)
        self.files = files or {}
        self.results = results or []
        self.downloads: list[str] = []

    def search(self, query: str, limit: int = 10) -> list[Candidate]:
        return self.results

    def download(self, url: str) -> bytes:
        self.downloads.append(url)
        return self.files[url]


@pytest.mark.parametrize(
    ("code", "url", "expected"),
    [
        ("cc-by-sa-4.0", "https://cc/by-sa", Licence("CC BY-SA 4.0", "https://cc/by-sa")),
        ("cc-by-2.0", "https://cc/by", Licence("CC BY 2.0", "https://cc/by")),
        ("cc-by-sa-3.0-de", "https://cc/de", Licence("CC BY-SA 3.0", "https://cc/de")),
        ("CC BY 4.0", "https://cc/by4", Licence("CC BY 4.0", "https://cc/by4")),
        ("cc0", None, Licence("CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/")),
        ("cc0", "http://cc/zero", Licence("CC0 1.0", "https://cc/zero")),  # Commons still serves some http links
        ("pd", None, Licence("Public domain", "")),
        ("pd-old-70", "https://pd", Licence("Public domain", "https://pd")),
        ("cc-by-nc-2.0", "https://cc/nc", None),
        ("cc-by-nd-4.0", "https://cc/nd", None),
        ("cc-by-nc-sa-4.0", "https://cc/nc-sa", None),
        ("gfdl", "https://gfdl", None),
        ("cc-by-sa-4.0", None, None),  # a CC licence must come with its URL
        ("", None, None),
        (None, None, None),
    ],
)
def test_licences(code: str | None, url: str | None, expected: Licence | None) -> None:
    assert classify_licence(code, url) == expected


def test_plain_text_strips_markup() -> None:
    assert plain_text('<a href="x">Cook &amp; Co</a>  <span>Own work</span>') == "Cook & Co Own work"


def test_parse_candidates_filters_and_orders() -> None:
    payload = {
        "query": {
            "pages": {
                "1": page(2, "Second"),
                "2": page(1, "First", meta={"License": {"value": "cc0"}, "LicenseUrl": {"value": ""}}),
                "3": page(3, "NonCommercial", meta={"License": {"value": "cc-by-nc-4.0"}}),
                "4": page(4, "Small", width=300),
                "5": page(5, "Svg", mime="image/svg+xml"),
                "6": page(6, "NonFree", meta={"NonFree": {"value": "true"}}),
                "7": page(7, "Trademark", meta={"Restrictions": {"value": "trademarked"}}),
                "8": page(8, "Anonymous", meta={"Artist": {"value": ""}, "Credit": {"value": ""}}),
                "9": page(9, "CreditOnly", meta={"Artist": {"value": ""}, "Credit": {"value": "Studio"}}),
            }
        }
    }

    candidates = parse_candidates(payload)

    assert [c.title for c in candidates] == ["File:First.jpg", "File:Second.jpg", "File:CreditOnly.jpg"]
    first = candidates[0]
    assert first.licence.name == "CC0 1.0"
    assert first.file_url == "https://upload.wikimedia.org/wikipedia/commons/x/First.jpg"  # tracking removed
    assert first.thumb_url == "https://thumb.wikimedia.org/x/330px-First.jpg"
    assert first.author == "Cook & Co"
    assert candidates[2].author == "Studio"
    assert parse_candidates({}) == []


@respx.mock
def test_client_identifies_itself_paces_and_searches() -> None:
    route = respx.get(API_URL).mock(return_value=httpx.Response(200, json={"query": {"pages": {"1": page(1, "A")}}}))
    now = [100.0]
    slept: list[float] = []
    client = CommonsClient(clock=lambda: now[0], sleep=slept.append)

    assert [c.title for c in client.search("Masala Dosa")] == ["File:A.jpg"]
    now[0] += 0.25
    client.search("Idli")

    request = route.calls[0].request
    assert request.headers["User-Agent"] == USER_AGENT
    assert request.url.params["gsrsearch"] == "Masala Dosa filetype:bitmap"
    assert slept == [pytest.approx(0.75)]


@respx.mock
def test_client_errors_become_photo_source_errors() -> None:
    respx.get(API_URL).mock(return_value=httpx.Response(503))
    client = CommonsClient(min_interval_s=0)

    with pytest.raises(PhotoSourceError, match="request failed"):
        client.search("Dosa")
    with pytest.raises(PhotoSourceError, match="outside Wikimedia"):
        client.download("https://example.com/dosa.jpg")


@respx.mock
def test_download() -> None:
    respx.get(FILE_URL).mock(return_value=httpx.Response(200, content=b"bytes"))

    assert CommonsClient(min_interval_s=0).download(FILE_URL) == b"bytes"


def test_thumbnail_is_small_rgb_webp() -> None:
    out = io.BytesIO()
    Image.new("RGBA", (1600, 900), (0, 0, 0, 0)).save(out, format="PNG")

    with Image.open(io.BytesIO(make_thumbnail(out.getvalue()))) as thumb:
        assert thumb.format == "WEBP"
        assert thumb.mode == "RGB"
        assert max(thumb.size) == 480


def test_thumbnail_accepts_large_catalog_originals_without_loosening_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)  # stands in for the 179 MP default
    monkeypatch.setattr("food_concierge.ingestion.photos.CATALOG_MAX_PIXELS", 2_000_000)

    thumbnail = make_thumbnail(jpeg(1600, 1200))
    assert Image.MAX_IMAGE_PIXELS == 1_000

    monkeypatch.undo()
    with Image.open(io.BytesIO(thumbnail)) as thumb:
        assert max(thumb.size) == 480


def test_ensure_cached_downloads_verifies_and_reuses(tmp_path: Path) -> None:
    data = jpeg()
    p = photo()
    client = FakeClient(files={FILE_URL: data})

    assert ensure_cached(p, client, tmp_path) == original_path(tmp_path, p)
    ensure_cached(p, client, tmp_path)

    assert client.downloads == [FILE_URL]  # the second call reads the cache
    assert original_path(tmp_path, p).read_bytes() == data
    assert thumbnail_path(tmp_path, p).exists()


def test_hash_mismatch_is_refused_and_not_cached(tmp_path: Path) -> None:
    p = photo(sha256="0" * 64)
    client = FakeClient(files={FILE_URL: jpeg()})

    with pytest.raises(PhotoSourceError, match="no longer matches"):
        ensure_cached(p, client, tmp_path)
    assert not original_path(tmp_path, p).exists()


def test_review_rows_end_with_none() -> None:
    candidate = parse_candidates({"query": {"pages": {"1": page(1, "A")}}})[0]

    rows = review_rows("fx005", "Masala Dosa", [candidate])

    assert [r["candidate"] for r in rows] == ["1", NONE]
    assert rows[0]["licence"] == "CC BY-SA 4.0"
    assert rows[1]["file_url"] == ""


def test_read_approvals() -> None:
    def row(item_id: str, candidate: str, approved: str = "") -> dict[str, str]:
        return {"item_id": item_id, "candidate": candidate, "approved": approved}

    approvals = read_approvals(
        [
            row("a", "1", "yes"),
            row("a", NONE),
            row("b", "1"),
            row("b", NONE, "YES"),
            row("c", "1"),
            row("d", "1", "yes"),
            row("d", "2", "yes"),
        ]
    )

    assert set(approvals.chosen) == {"a"}
    assert approvals.no_photo == {"b"}
    assert approvals.unreviewed == {"c"}
    assert approvals.problems == ("d: 2 rows approved; approve exactly one",)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"licence": "CC BY-NC 4.0"}, "String should match pattern"),
        ({"file_url": "https://example.com/a.jpg"}, "String should match pattern"),
        ({"file_url": "https://upload.wikimedia.org/a/b/Dosa.gif"}, "String should match pattern"),
        ({"sha256": "abc"}, "String should match pattern"),
        ({"licence": "CC BY 4.0", "licence_url": ""}, "needs a licence_url"),
    ],
)
def test_photo_schema(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        photo(**changes)
    assert photo(licence="Public domain", licence_url="").licence == "Public domain"
    upper = photo(file_url="https://upload.wikimedia.org/a/b/Roti_in_clay_oven.JPG")  # Commons keeps the case
    assert original_path(Path("cache"), upper).suffix == ".jpg"


def write_attributions(raw_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (raw_dir / ATTRIBUTIONS_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PHOTO_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_loader_reads_attributions(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    write_attributions(tmp_path, [photo().model_dump(mode="json")])

    assert load_catalog(tmp_path).catalog.photos == (photo(),)
    assert load_catalog(FIXTURE).catalog.photos == ()  # no attributions.csv: no photos


def test_loader_rejects_bad_attributions(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    good = photo().model_dump(mode="json")
    write_attributions(tmp_path, [good, good, good | {"item_id": "nowhere"}, good | {"licence": "CC BY-ND 4.0"}])

    with pytest.raises(CatalogValidationError) as caught:
        load_catalog(tmp_path)

    found = {(i.line, i.column, i.message.split(":")[0]) for i in caught.value.issues}
    assert found == {
        (3, "item_id", "fx005 has two photos"),
        (4, "item_id", "no menu item nowhere"),
        (5, "licence", "String should match pattern '^(CC0 1\\.0|Public domain|CC BY(-SA)? \\d\\.\\d)$'"),
    }


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_photos", REPO_ROOT / "scripts" / "fetch_photos.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["fetch_photos"] = module
    spec.loader.exec_module(module)
    return module


def test_candidates_keep_earlier_choices_and_render_a_safe_page(script: ModuleType, tmp_path: Path) -> None:
    items = load_catalog(FIXTURE).catalog.items[:2]
    hostile = page(1, "A", meta={"Artist": {"value": "</script><script>alert(1)</script>"}})
    client = FakeClient(results=parse_candidates({"query": {"pages": {"1": hostile}}}))
    review = tmp_path / "photo_review.csv"

    rows = script.find_candidates(items, client, review, per_dish=5)
    rows[0]["approved"] = "yes"
    script.write_csv(review, list(rows[0]), rows)
    rows = script.find_candidates(items, client, review, per_dish=5)
    page_html = script.render_review_page(rows)

    assert [(r["item_id"], r["candidate"], r["approved"]) for r in rows] == [
        ("fx001", "1", "yes"),
        ("fx001", NONE, ""),
        ("fx002", "1", ""),
        ("fx002", NONE, ""),
    ]
    assert "<script>alert(1)" not in page_html
    assert "checked" in page_html
    assert "No suitable photo" in page_html


def test_review_page_is_rebuilt_from_the_sheet_and_remembers_clicks(script: ModuleType, tmp_path: Path) -> None:
    rows = review_rows("fx005", "Masala Dosa", [])

    assert script.write_review_page([], tmp_path / "cache") == 1
    assert script.write_review_page(rows, tmp_path / "cache") == 0

    page_html = (tmp_path / "cache" / "photo_review.html").read_text(encoding="utf-8")
    assert "Masala Dosa (fx005)" in page_html
    assert "localStorage.setItem" in page_html


def test_approve_writes_attributions_and_reuses_unchanged(script: ModuleType, tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, tmp_path / "raw")
    raw = tmp_path / "raw"
    items = load_catalog(raw).catalog.items
    data = jpeg()
    candidate = parse_candidates({"query": {"pages": {"1": page(1, "Dosa")}}})[0]
    rows = review_rows("fx005", "Masala Dosa", [candidate]) + review_rows("fx006", "Idli Sambar", [])
    rows[0]["approved"] = "yes"
    rows[2]["approved"] = "yes"  # fx006: no suitable photo
    script.write_csv(raw / "photo_review.csv", list(rows[0]), rows)
    client = FakeClient(files={candidate.file_url: data})

    assert script.apply_approvals(items, raw / "photo_review.csv", raw / ATTRIBUTIONS_FILE, client, tmp_path) == 0
    assert script.apply_approvals(items, raw / "photo_review.csv", raw / ATTRIBUTIONS_FILE, client, tmp_path) == 0

    [attributed] = load_catalog(raw).catalog.photos
    assert (attributed.item_id, attributed.sha256, attributed.author) == ("fx005", sha256(data), "Cook & Co")
    assert client.downloads == [candidate.file_url]  # the second run keeps the existing attribution
    assert script.sync([attributed], client, tmp_path) == 0
    assert thumbnail_path(tmp_path, attributed).exists()


class QueryClient(FakeClient):
    def __init__(self, answers: dict[str, list[Candidate]]) -> None:
        super().__init__()
        self.answers = answers
        self.queries: list[str] = []

    def search(self, query: str, limit: int = 10) -> list[Candidate]:
        self.queries.append(query)
        return self.answers.get(query, [])


def test_search_falls_back_to_shorter_names(script: ModuleType) -> None:
    candidate = parse_candidates({"query": {"pages": {"1": page(1, "A")}}})[0]
    client = QueryClient({"Dim Sum": [candidate], "Moilee": [candidate]})

    assert script.search_dish(client, "Steamed Veg Dim Sum", 5) == [candidate]
    assert script.search_dish(client, "Prawn Moilee", 5) == [candidate]  # a distinctive single word is tried
    assert script.search_dish(client, "Peanut Noodle Salad", 5) == []  # but never a bare "Salad"
    assert client.queries == [
        "Steamed Veg Dim Sum", "Veg Dim Sum", "Dim Sum",
        "Prawn Moilee", "Moilee",
        "Peanut Noodle Salad", "Noodle Salad",
    ]  # fmt: skip


def test_only_re_searches_the_named_dishes(script: ModuleType, tmp_path: Path) -> None:
    items = load_catalog(FIXTURE).catalog.items[:2]
    candidate = parse_candidates({"query": {"pages": {"1": page(1, "A")}}})[0]
    review = tmp_path / "photo_review.csv"
    script.find_candidates(items, FakeClient(), review, per_dish=5)

    rows = script.find_candidates(items, FakeClient(results=[candidate]), review, per_dish=5, only={"fx002"})

    assert [(r["item_id"], r["candidate"]) for r in rows] == [("fx001", NONE), ("fx002", "1"), ("fx002", NONE)]


def test_approve_refuses_two_choices_for_one_dish(script: ModuleType, tmp_path: Path) -> None:
    candidate = parse_candidates({"query": {"pages": {"1": page(1, "Dosa")}}})[0]
    rows = review_rows("fx005", "Masala Dosa", [candidate])
    for row in rows:
        row["approved"] = "yes"
    script.write_csv(tmp_path / "review.csv", list(rows[0]), rows)

    assert script.apply_approvals([], tmp_path / "review.csv", tmp_path / "a.csv", FakeClient(), tmp_path) == 1
    assert not (tmp_path / "a.csv").exists()


def test_approve_checks_every_row_before_downloading(script: ModuleType, tmp_path: Path) -> None:
    good, bad = (parse_candidates({"query": {"pages": {"1": page(1, name)}}})[0] for name in ("Dosa", "Idli"))
    rows = review_rows("fx005", "Masala Dosa", [good]) + review_rows("fx006", "Idli Sambar", [bad])
    rows[0]["approved"] = rows[2]["approved"] = "yes"
    rows[2]["licence_url"] = "http://creativecommons.org/licenses/by-sa/4.0/"  # an older sheet: upgraded, not refused
    rows[0]["file_url"] = "https://example.com/elsewhere.jpg"
    script.write_csv(tmp_path / "review.csv", list(rows[0]), rows)
    client = FakeClient()

    assert script.apply_approvals([], tmp_path / "review.csv", tmp_path / "a.csv", client, tmp_path) == 1
    assert client.downloads == []
    assert not (tmp_path / "a.csv").exists()


def test_attribution_upgrades_http_licence_links() -> None:
    candidate = parse_candidates({"query": {"pages": {"1": page(1, "Dosa")}}})[0]
    [row, _] = review_rows("fx005", "Masala Dosa", [candidate])
    row["licence_url"] = "http://creativecommons.org/publicdomain/zero/1.0/deed.en"

    assert attribution("fx005", row, "0" * 64, 1).licence_url.startswith("https://creativecommons.org/")
