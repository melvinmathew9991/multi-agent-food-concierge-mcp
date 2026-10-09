"""Find, approve and fetch catalog photos from Wikimedia Commons (Phase 2 plan, item 2). Never run in CI.

1. Find candidates for every dish (about 150 paced API requests, a few minutes):

       python scripts/fetch_photos.py candidates

   This writes data/raw/photo_review.csv and a local page, photo_review.html, in the photo cache. Open the page,
   choose one photo per dish (or "no suitable photo"), and download the filled sheet over data/raw/photo_review.csv.
   Re-running keeps choices already made. The page remembers clicks across reloads; to rebuild it from the sheet
   without searching Commons again:

       python scripts/fetch_photos.py review

2. Turn the choices into attributions, downloading each approved file and pinning its SHA-256:

       python scripts/fetch_photos.py approve

3. On another machine, or after clearing the cache, fetch every attributed photo and verify it:

       python scripts/fetch_photos.py sync
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

from food_concierge.config import get_settings
from food_concierge.ingestion.loader import ATTRIBUTIONS_FILE, load_catalog
from food_concierge.ingestion.photos import (
    NONE,
    REVIEW_COLUMNS,
    Candidate,
    CommonsClient,
    approve,
    ensure_cached,
    read_approvals,
    review_rows,
)
from food_concierge.ingestion.schemas import PHOTO_COLUMNS, MenuItem, Photo

REVIEW_FILE = "photo_review.csv"
REVIEW_PAGE = "photo_review.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _key(row: dict[str, str]) -> tuple[str, str]:
    return row["item_id"], row["file_url"] or row["candidate"]


def search_dish(client: CommonsClient, name: str, per_dish: int) -> list[Candidate]:
    """Search the full name, then without its leading words ("Steamed Veg Dim Sum" → "Dim Sum").

    A query shorter than two words is tried only when its word is distinctive (six letters or more), so a name
    never shrinks to "Sum" or "Salad".
    """
    words = name.split()
    for start in range(len(words)):
        rest = words[start:]
        if start and len(rest) < 2 and len(rest[0]) < 6:
            break
        found = client.search(" ".join(rest))[:per_dish]
        if found:
            return found
    return []


def find_candidates(
    items: Sequence[MenuItem],
    client: CommonsClient,
    review: Path,
    per_dish: int,
    only: Collection[str] | None = None,
) -> list[dict[str, str]]:
    """Rewrite the review sheet, keeping choices already made; ``only`` re-searches just those dishes."""
    existing: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(review):
        existing.setdefault(row["item_id"], []).append(row)
    approved = {_key(row): row["approved"] for rows in existing.values() for row in rows}
    rows: list[dict[str, str]] = []
    for item in items:
        if only is not None and item.item_id not in only and item.item_id in existing:
            rows.extend(existing[item.item_id])
            continue
        for row in review_rows(item.item_id, item.name, search_dish(client, item.name, per_dish)):
            row["approved"] = approved.get(_key(row), "")
            rows.append(row)
    write_csv(review, REVIEW_COLUMNS, rows)
    return rows


def render_review_page(rows: Sequence[dict[str, str]]) -> str:
    """A self-contained page: pick one photo per dish, then download the filled review sheet."""
    dishes: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        dishes.setdefault(row["item_id"], []).append(row)
    sections = []
    for item_id, options in dishes.items():
        choices = []
        for row in options:
            checked = " checked" if row["approved"].strip().lower() == "yes" else ""
            value = html.escape(row["candidate"])
            if row["candidate"] == NONE:
                body = "<span class='none'>No suitable photo</span>"
            else:
                body = (
                    f"<img src='{html.escape(row['thumb_url'])}' alt='' loading='lazy'>"
                    f"<small>{html.escape(row['licence'])} · {html.escape(row['author'])} · "
                    f"{html.escape(row['width'])} x {html.escape(row['height'])} · "
                    f"<a href='{html.escape(row['file_page_url'])}' target='_blank' rel='noopener'>"
                    "file page</a></small>"
                )
            choices.append(
                f"<label><input type='radio' name='{html.escape(item_id)}' value='{value}'{checked}>{body}</label>"
            )
        title = html.escape(f"{options[0]['dish']} ({item_id})")
        sections.append(f"<fieldset><legend>{title}</legend>{''.join(choices)}</fieldset>")
    data = json.dumps(list(rows)).replace("</", "<\\/")  # Commons text is untrusted: keep it out of the markup
    columns = json.dumps(list(REVIEW_COLUMNS))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Photo review</title>
<style>
body {{ font: 15px/1.4 system-ui, sans-serif; margin: 16px; background: #fff; color: #111; }}
header {{ position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #ddd; }}
fieldset {{ border: 1px solid #ddd; margin: 12px 0; display: flex; flex-wrap: wrap; gap: 12px; }}
label {{ display: flex; flex-direction: column; width: 220px; gap: 4px; cursor: pointer; }}
img {{ width: 220px; height: 165px; object-fit: cover; background: #eee; }}
.none {{ display: grid; place-items: center; width: 220px; height: 165px; background: #f4f4f4; }}
small {{ color: #555; overflow-wrap: anywhere; }}
</style></head><body>
<header><strong>Photo review</strong> · <span id="count"></span>
<button id="download">Download photo_review.csv</button></header>
{"".join(sections)}
<script>
const rows = {data};
const columns = {columns};
const STORE = "photo_review_choices";  // survives a reload; the downloaded sheet is still the record
const load = () => {{ try {{ return JSON.parse(localStorage.getItem(STORE)) || {{}}; }} catch {{ return {{}}; }} }};
const choices = load();
for (const [id, value] of Object.entries(choices)) {{
  const input = document.querySelector(`input[name="${{CSS.escape(id)}}"][value="${{CSS.escape(value)}}"]`);
  if (input && !document.querySelector(`input[name="${{CSS.escape(id)}}"]:checked`)) input.checked = true;
}}
const count = () => {{
  const dishes = new Set(rows.map(r => r.item_id));
  const done = [...dishes].filter(id => document.querySelector(`input[name="${{CSS.escape(id)}}"]:checked`)).length;
  document.getElementById("count").textContent = `${{done}} of ${{dishes.size}} dishes chosen`;
}};
document.addEventListener("change", event => {{
  choices[event.target.name] = event.target.value;
  try {{ localStorage.setItem(STORE, JSON.stringify(choices)); }} catch {{}}
  count();
}});
count();
document.getElementById("download").addEventListener("click", () => {{
  const quote = v => '"' + String(v).replaceAll('"', '""') + '"';
  const lines = [columns.join(",")];
  for (const row of rows) {{
    const picked = document.querySelector(`input[name="${{CSS.escape(row.item_id)}}"]:checked`);
    const approved = picked && picked.value === row.candidate ? "yes" : "";
    lines.push(columns.map(c => quote(c === "approved" ? approved : row[c])).join(","));
  }}
  const blob = new Blob([lines.join("\\n") + "\\n"], {{ type: "text/csv" }});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "photo_review.csv";
  link.click();
}});
</script></body></html>
"""


def write_review_page(rows: Sequence[dict[str, str]], cache_dir: Path) -> int:
    if not rows:
        print(f"No review sheet yet: run `candidates` first to write data/raw/{REVIEW_FILE}.")
        return 1
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / REVIEW_PAGE).write_text(render_review_page(rows), encoding="utf-8")
    print(f"Review page: {cache_dir / REVIEW_PAGE}")
    return 0


def apply_approvals(
    items: Sequence[MenuItem], review: Path, attributions: Path, client: CommonsClient, cache_dir: Path
) -> int:
    approvals = read_approvals(read_csv(review))
    for problem in approvals.problems:
        print(problem)
    if approvals.problems:
        return 1
    existing = {row["item_id"]: row for row in read_csv(attributions)}
    photos: list[Photo] = []
    for item in items:
        row = approvals.chosen.get(item.item_id)
        if row is None:
            continue
        kept = existing.get(item.item_id)
        photo = Photo.model_validate(kept) if kept and kept["file_url"] == row["file_url"] else None
        photos.append(photo or approve(item.item_id, row, client, cache_dir))
    write_csv(attributions, PHOTO_COLUMNS, (photo.model_dump(mode="json") for photo in photos))
    print(
        f"{len(photos)} photos attributed, {len(approvals.no_photo)} dishes without a suitable photo, "
        f"{len(approvals.unreviewed)} not reviewed yet."
    )
    return 0


def sync(photos: Sequence[Photo], client: CommonsClient, cache_dir: Path) -> int:
    for photo in photos:
        ensure_cached(photo, client, cache_dir)
    print(f"{len(photos)} photos verified in {cache_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["candidates", "review", "approve", "sync"])
    parser.add_argument("--per-dish", type=int, default=5, help="candidates kept per dish (default 5)")
    parser.add_argument("--only", nargs="+", metavar="ITEM_ID", help="candidates: re-search only these dishes")
    args = parser.parse_args(argv)

    settings = get_settings()
    raw, cache = settings.raw_dir, settings.photo_cache_dir
    if args.command == "review":
        return write_review_page(read_csv(raw / REVIEW_FILE), cache)
    loaded = load_catalog(raw)
    client = CommonsClient()
    if args.command == "candidates":
        rows = find_candidates(loaded.catalog.items, client, raw / REVIEW_FILE, args.per_dish, args.only)
        found = {row["item_id"] for row in rows if row["candidate"] != NONE}
        print(f"{len(found)} of {len(loaded.catalog.items)} dishes have candidates.")
        return write_review_page(rows, cache)
    if args.command == "approve":
        return apply_approvals(loaded.catalog.items, raw / REVIEW_FILE, raw / ATTRIBUTIONS_FILE, client, cache)
    return sync(loaded.catalog.photos, client, cache)


if __name__ == "__main__":
    sys.exit(main())
