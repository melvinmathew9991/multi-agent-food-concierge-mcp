"""Catalog photos from Wikimedia Commons (Phase 2 plan, item 2).

Only CC0, public-domain, CC BY and CC BY-SA files are kept; NC, ND, GFDL-only, non-free and unknown licences are
refused. The owner approves one candidate per dish (or none). Approved files are recorded in ``attributions.csv``
with author, licence and SHA-256, and fetched into ``Settings.photo_cache_dir``, never into the repository. A file
whose bytes no longer match its pinned hash is refused.

Commons API etiquette: a descriptive User-Agent naming the project, and at most one request per second.
"""

from __future__ import annotations

import hashlib
import html
import io
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from food_concierge import __version__
from food_concierge.errors import PhotoSourceError
from food_concierge.ingestion.schemas import Photo

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    f"food-concierge-catalog/{__version__} (https://github.com/melvinmathew9991/multi-agent-food-concierge-mcp)"
)
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
MIN_WIDTH = 480
THUMBNAIL_PX = 480
CATALOG_MAX_PIXELS = 250_000_000  # the largest approved original is 200 MP
NONE = "none"  # the review-sheet candidate meaning "no suitable photo for this dish"

REVIEW_COLUMNS = (
    "item_id", "dish", "candidate", "title", "file_page_url", "file_url", "thumb_url",
    "width", "height", "bytes", "licence", "licence_url", "author", "approved",
)  # fmt: skip

_CC_LICENCE = re.compile(r"cc-by(-sa)?-(\d\.\d)(-[a-z]{2,})?")
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Licence:
    name: str  # as stored in attributions.csv: "CC BY-SA 4.0", "CC BY 2.0", "CC0 1.0", "Public domain"
    url: str


def https(url: str) -> str:
    """Commons still gives some licence links as ``http://``; every licence host serves https."""
    return "https://" + url.removeprefix("http://") if url.startswith("http://") else url


def classify_licence(code: str | None, url: str | None) -> Licence | None:
    """Map a Commons ``License`` code to an accepted licence, or None when the file must be refused."""
    url = https(url) if url else url
    key = re.sub(r"\s+", "-", (code or "").strip().lower())
    if key in {"cc0", "cc0-1.0", "cc-zero"}:
        return Licence("CC0 1.0", url or "https://creativecommons.org/publicdomain/zero/1.0/")
    if key == "pd" or key.startswith(("pd-", "public-domain")):
        return Licence("Public domain", url or "")
    match = _CC_LICENCE.fullmatch(key)
    if match and url:
        return Licence(f"CC BY{'-SA' if match.group(1) else ''} {match.group(2)}", url)
    return None


def _meta(extmetadata: Mapping[str, Any], field: str) -> str:
    value = extmetadata.get(field, {}).get("value", "")
    return str(value) if value is not None else ""


def _clean_url(url: str) -> str:
    # Commons appends tracking parameters (utm_source=...) to file URLs; the attribution keeps the plain URL.
    return url.split("?", 1)[0]


def plain_text(markup: str) -> str:
    """Commons credits are HTML; keep the visible text."""
    return " ".join(html.unescape(_TAG.sub(" ", markup)).split())


@dataclass(frozen=True)
class Candidate:
    title: str
    file_page_url: str
    file_url: str
    thumb_url: str
    width: int
    height: int
    bytes: int
    licence: Licence
    author: str


def parse_candidates(payload: Mapping[str, Any], min_width: int = MIN_WIDTH) -> list[Candidate]:
    """Usable files from a ``generator=search&prop=imageinfo`` response, in search-rank order."""
    pages = sorted(payload.get("query", {}).get("pages", {}).values(), key=lambda page: page.get("index", 0))
    candidates = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        licence = classify_licence(_meta(meta, "License") or _meta(meta, "LicenseShortName"), _meta(meta, "LicenseUrl"))
        author = plain_text(_meta(meta, "Artist")) or plain_text(_meta(meta, "Credit"))
        usable = (
            licence is not None
            and info.get("mime") in ALLOWED_MIME
            and info.get("width", 0) >= min_width
            and _meta(meta, "NonFree").lower() != "true"
            and not _meta(meta, "Restrictions")  # trademarks, personality rights and the like
            and bool(author)
        )
        if usable and licence is not None:
            candidates.append(
                Candidate(
                    title=page["title"],
                    file_page_url=_clean_url(info["descriptionurl"]),
                    file_url=_clean_url(info["url"]),
                    thumb_url=_clean_url(info.get("thumburl", info["url"])),
                    width=info["width"],
                    height=info["height"],
                    bytes=info["size"],
                    licence=licence,
                    author=author[:200],
                )
            )
    return candidates


class CommonsClient:
    """Paced, identified access to the Commons API and file servers."""

    def __init__(
        self,
        http: httpx.Client | None = None,
        min_interval_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http or httpx.Client(timeout=30.0, follow_redirects=True)
        self._http.headers["User-Agent"] = USER_AGENT
        self._min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def _get(self, url: str, params: Mapping[str, str | int] | None = None) -> httpx.Response:
        if self._last is not None:
            wait = self._min_interval_s - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        try:
            response = self._http.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise PhotoSourceError(f"Wikimedia Commons request failed: {exc}") from exc
        finally:
            self._last = self._clock()
        return response

    def search(self, query: str, limit: int = 10) -> list[Candidate]:
        params: dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 320,
        }
        return parse_candidates(self._get(API_URL, params).json())

    def download(self, url: str) -> bytes:
        if not url.startswith("https://upload.wikimedia.org/"):
            raise PhotoSourceError(f"Refusing to download from outside Wikimedia's file servers: {url}")
        return self._get(url).content


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def original_path(cache_dir: Path, photo: Photo) -> Path:
    return cache_dir / "originals" / f"{photo.sha256}{Path(photo.file_url).suffix.lower()}"


def thumbnail_path(cache_dir: Path, photo: Photo) -> Path:
    return cache_dir / "thumbs" / f"{photo.sha256}.webp"


def _write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(path)


def make_thumbnail(data: bytes, max_px: int = THUMBNAIL_PX) -> bytes:
    """A WebP no larger than ``max_px`` on either side, RGB, without metadata.

    Commons originals reach 200 MP, past Pillow's bomb limit. These files are owner-approved and hash-pinned, so the
    limit is raised to ``CATALOG_MAX_PIXELS`` here only (offline, single-threaded), and JPEGs decode at reduced scale.
    Uploads keep the default limit (``models/images.py``).
    """
    default, Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS, CATALOG_MAX_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.draft("RGB", (max_px, max_px))
            image.thumbnail((max_px, max_px))
            rgb = image.convert("RGB")
    finally:
        Image.MAX_IMAGE_PIXELS = default
    out = io.BytesIO()
    rgb.save(out, format="WEBP", quality=82, method=6)
    return out.getvalue()


def ensure_cached(photo: Photo, client: CommonsClient, cache_dir: Path) -> Path:
    """Fetch the photo into the cache if missing, verify its hash, and build its thumbnail."""
    original = original_path(cache_dir, photo)
    data = original.read_bytes() if original.exists() else client.download(photo.file_url)
    if sha256(data) != photo.sha256:
        original.unlink(missing_ok=True)
        raise PhotoSourceError(f"{photo.item_id}: {photo.file_url} no longer matches its pinned SHA-256")
    if not original.exists():
        _write_atomically(original, data)
    thumbnail = thumbnail_path(cache_dir, photo)
    if not thumbnail.exists():
        _write_atomically(thumbnail, make_thumbnail(data))
    return original


def review_rows(item_id: str, dish: str, candidates: Iterable[Candidate]) -> list[dict[str, str]]:
    """Review-sheet rows for one dish: its candidates, then a "none" row for "no suitable photo"."""
    rows = [
        {
            "item_id": item_id,
            "dish": dish,
            "candidate": str(rank),
            "title": c.title,
            "file_page_url": c.file_page_url,
            "file_url": c.file_url,
            "thumb_url": c.thumb_url,
            "width": str(c.width),
            "height": str(c.height),
            "bytes": str(c.bytes),
            "licence": c.licence.name,
            "licence_url": c.licence.url,
            "author": c.author,
            "approved": "",
        }
        for rank, c in enumerate(candidates, start=1)
    ]
    rows.append(dict.fromkeys(REVIEW_COLUMNS, "") | {"item_id": item_id, "dish": dish, "candidate": NONE})
    return rows


@dataclass(frozen=True)
class Approvals:
    chosen: Mapping[str, Mapping[str, str]]  # item_id → the approved candidate row
    no_photo: frozenset[str]  # dishes the owner marked as having no suitable photo
    unreviewed: frozenset[str]
    problems: tuple[str, ...]  # dishes with more than one approval


def read_approvals(rows: Iterable[Mapping[str, str]]) -> Approvals:
    """Collect the owner's choices: exactly one row per dish marked ``yes`` in ``approved``."""
    marked: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        picks = marked.setdefault(row["item_id"], [])
        if row["approved"].strip().lower() == "yes":
            picks.append(row)
    chosen: dict[str, Mapping[str, str]] = {}
    no_photo: set[str] = set()
    problems = []
    for item_id, picks in marked.items():
        if len(picks) > 1:
            problems.append(f"{item_id}: {len(picks)} rows approved; approve exactly one")
        elif picks and picks[0]["candidate"] == NONE:
            no_photo.add(item_id)
        elif picks:
            chosen[item_id] = picks[0]
    unreviewed = frozenset(marked) - set(chosen) - no_photo - {p.split(":")[0] for p in problems}
    return Approvals(chosen, frozenset(no_photo), unreviewed, tuple(problems))


def attribution(item_id: str, row: Mapping[str, str], digest: str, size: int) -> Photo:
    """The attribution for an approved review-sheet row; raises ValidationError when the row cannot be one."""
    return Photo(
        item_id=item_id,
        file_page_url=row["file_page_url"],
        file_url=row["file_url"],
        author=row["author"],
        licence=row["licence"],
        licence_url=https(row["licence_url"]),
        sha256=digest,
        width=int(row["width"]),
        height=int(row["height"]),
        bytes=size,
    )


def approve(item_id: str, row: Mapping[str, str], client: CommonsClient, cache_dir: Path) -> Photo:
    """Download an approved candidate, pin its hash, cache it and return its attribution."""
    data = client.download(row["file_url"])
    photo = attribution(item_id, row, sha256(data), len(data))
    _write_atomically(original_path(cache_dir, photo), data)
    return photo
