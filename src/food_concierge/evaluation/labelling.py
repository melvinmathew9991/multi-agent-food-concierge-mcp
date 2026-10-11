"""Pool candidates for the owner's relevance grades, render the blind grading page, and import the grades.

Pooling (TREC-style, Phase 3 plan §1): each query's pool is the union of the top ``POOL_DEPTH`` items from every
candidate stack, so no stack is scored on items nobody judged. The page shows each pool shuffled, with no hint of
which stack found an item. Items of a query's target dish are graded 3 by rule and left off the page.
"""

from __future__ import annotations

import csv
import html
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from food_concierge.errors import EvaluationError
from food_concierge.evaluation.dataset import EXACT, RetrievalDataset, TextQuery
from food_concierge.ingestion.schemas import MenuItem
from food_concierge.services.retrieval import RetrievalConfig, Retriever
from food_concierge.storage.catalog_db import Filters

POOL_DEPTH = 10


@dataclass(frozen=True)
class PoolStack:
    config: RetrievalConfig
    variant: str = "prefix"  # query embedding: with or without the BGE prefix
    filtered: bool = True  # the original approach (C0) searches without hard filters


# Every candidate in the plan, plus a rerank depth of 10: pooling more stacks only widens what gets judged.
POOL_STACKS: dict[str, PoolStack] = {
    "bm25": PoolStack(RetrievalConfig(mode="bm25", k=POOL_DEPTH)),
    "dense": PoolStack(RetrievalConfig(mode="dense", k=POOL_DEPTH)),
    "dense_no_prefix": PoolStack(RetrievalConfig(mode="dense", k=POOL_DEPTH), variant="no_prefix"),
    "dense_unfiltered": PoolStack(RetrievalConfig(mode="dense", k=POOL_DEPTH), filtered=False),
    "hybrid": PoolStack(RetrievalConfig(mode="hybrid", k=POOL_DEPTH)),
    "hybrid_rerank10": PoolStack(RetrievalConfig(mode="hybrid", k=POOL_DEPTH, rerank_depth=10)),
    "hybrid_rerank20": PoolStack(RetrievalConfig(mode="hybrid", k=POOL_DEPTH, rerank_depth=20)),
}


@dataclass(frozen=True)
class PoolEntry:
    query_id: str
    to_grade: tuple[str, ...]  # shuffled; what the owner grades
    targets: tuple[str, ...]  # graded 3 by rule


def build_pool(
    queries: Sequence[TextQuery],
    retrievers: Mapping[str, Retriever],
    dish_keys: Mapping[str, str],
    seed: int,
    stacks: Mapping[str, PoolStack] = POOL_STACKS,
) -> list[PoolEntry]:
    entries = []
    for query in queries:
        found: set[str] = set()
        for stack in stacks.values():
            filters = query.filters if stack.filtered else Filters()
            found |= {hit.item_id for hit in retrievers[stack.variant].search(query.text, filters, stack.config).hits}
        targets = {item for item, key in dish_keys.items() if query.target is not None and key == query.target}
        to_grade = sorted(found - targets - query.grades.keys())
        random.Random(f"{seed}:{query.id}").shuffle(to_grade)  # noqa: S311 - reproducible order, not security
        entries.append(PoolEntry(query.id, tuple(to_grade), tuple(sorted(targets))))
    return entries


def apply_grades(
    dataset: RetrievalDataset, pool: Sequence[PoolEntry], rows: Iterable[Mapping[str, str]]
) -> RetrievalDataset:
    """Merge the downloaded grades into the dataset; every pooled item must have one."""
    graded = {(row["query_id"], row["item_id"]): int(row["grade"]) for row in rows if row.get("grade", "") != ""}
    by_id = {entry.query_id: entry for entry in pool}
    missing = sum(1 for entry in pool for item in entry.to_grade if (entry.query_id, item) not in graded)
    if missing:
        raise EvaluationError(f"{missing} pooled item(s) have no grade yet; finish grading before importing.")
    queries = []
    for query in dataset.queries:
        entry = by_id.get(query.id)
        if entry is None:
            queries.append(query)
            continue
        grades = dict(query.grades) | dict.fromkeys(entry.targets, EXACT)
        grades |= {item: graded[(query.id, item)] for item in entry.to_grade}
        queries.append(query.model_copy(update={"grades": dict(sorted(grades.items()))}))
    return RetrievalDataset.model_validate(dataset.model_dump() | {"queries": queries})


def read_grades(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_dataset(dataset: RetrievalDataset, path: Path) -> None:
    data = dataset.model_dump(mode="json", exclude_defaults=True)
    data["queries"] = [query.model_dump(mode="json", exclude_defaults=True) for query in dataset.queries]
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")


GRADE_LABELS = {
    3: "3 · exactly what was asked",
    2: "2 · relevant",
    1: "1 · related, but wrong on something that matters",
    0: "0 · not relevant",
}


def render_page(pool: Sequence[PoolEntry], queries: Mapping[str, TextQuery], items: Mapping[str, MenuItem]) -> str:
    """A self-contained page: grade each pooled dish 0-3, then download grades.csv."""
    cards = []
    for entry in pool:
        if not entry.to_grade:
            continue
        query = queries[entry.query_id]
        note = "<p class='note'>Grade the fit to the words only; the filters are checked separately.</p>"
        rows = []
        for item_id in entry.to_grade:
            item = items[item_id]
            name = html.escape(f"{entry.query_id}:{item_id}")
            radios = "".join(
                f"<label><input type='radio' name='{name}' value='{grade}'>{html.escape(label)}</label>"
                for grade, label in GRADE_LABELS.items()
            )
            rows.append(
                f"<div class='dish'><strong>{html.escape(item.name)}</strong> "
                f"<small>{html.escape(item.cuisine.value.replace('_', ' '))} · {html.escape(item.category.value)} · "
                f"{html.escape(item.diet.value.replace('_', '-'))}</small>"
                f"<p>{html.escape(item.description)}</p>"
                f"<p class='ing'>{html.escape(', '.join(item.ingredients))}</p>"
                f"<div class='grades'>{radios}</div></div>"
            )
        cards.append(
            f"<section><h2>{html.escape(entry.query_id)} · “{html.escape(query.text)}”</h2>"
            f"{note if query.filters != Filters() else ''}{''.join(rows)}</section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Query grading</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;margin:0;background:#fafaf7;color:#1f1f1f}}
header{{position:sticky;top:0;background:#1f1f1f;color:#fff;padding:10px 16px;
  display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
main{{max-width:880px;margin:0 auto;padding:16px}}
section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:0 0 18px}}
h2{{font-size:17px;margin:4px 0 10px}}
.dish{{border-top:1px solid #eee;padding:10px 0}}.dish p{{margin:4px 0}}
.ing{{color:#666;font-size:13px}}.note{{color:#8a5a00;font-size:13px}}
.grades{{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:6px}}
.grades label{{cursor:pointer}}
button{{font:inherit;padding:4px 10px}}
</style></head><body>
<header><strong>Query grading</strong><span id="count"></span>
<button id="download">Download grades.csv</button></header>
<main>
<p>Grade how well each dish answers the query. Dishes appear in a random order, and the system that found each one is
hidden. Choices are kept in this browser across reloads, but the downloaded file is the record.</p>
{"".join(cards)}
</main>
<script>
const STORE = "retrieval_grades";
const load = () => {{ try {{ return JSON.parse(localStorage.getItem(STORE)) || {{}}; }} catch {{ return {{}}; }} }};
const grades = load();
const radios = [...document.querySelectorAll("input[type=radio]")];
const names = [...new Set(radios.map(r => r.name))];
const count = () => {{
  const done = names.filter(n => n in grades).length;
  document.getElementById("count").textContent = `${{done}} of ${{names.length}} graded`;
}};
radios.forEach(r => {{
  if (grades[r.name] === r.value) r.checked = true;
  r.addEventListener("change", () => {{
    grades[r.name] = r.value;
    try {{ localStorage.setItem(STORE, JSON.stringify(grades)); }} catch {{}}
    count();
  }});
}});
document.getElementById("download").addEventListener("click", () => {{
  const lines = ["query_id,item_id,grade", ...names.map(n => n.replace(":", ",") + "," + (grades[n] ?? ""))];
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], {{type: "text/csv"}}));
  link.download = "grades.csv";
  link.click();
}});
count();
</script></body></html>
"""
