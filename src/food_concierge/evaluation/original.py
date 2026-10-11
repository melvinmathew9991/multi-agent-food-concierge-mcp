"""The original approach re-implemented as baseline C0 (Phase 3 plan §3; baseline defects #4, #5 and #10).

The original searched the raw request with dense retrieval, took the top 5 with no hard filters, and asked an LLM
one Yes/No question per dish; the dishes answered "yes" were the result. Constraints reached the model only as
words in the request, and it judged them against the restaurant's declared allergens, which are often blank.

This re-implements the method on this catalog with this project's embedder and a free model (the original's paid
models are off by default, rules §5), so it is not a replay of the original. Every reply is cached with the dishes
retrieved, so C0 is scored offline and the same verdicts are read by two parsers:

- ``strict``: the original's check, lower-cased and stripped of spaces only, then ``== "yes"`` (defect #4);
- ``fixed``: the first word, ignoring case and punctuation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from food_concierge.evaluation.dataset import TextQuery
from food_concierge.evaluation.metrics import GroundTruth
from food_concierge.evaluation.runner import QueryResult, score_query
from food_concierge.ingestion.schemas import MenuItem
from food_concierge.services.retrieval import RetrievalConfig
from food_concierge.storage.catalog_db import Filters

PROMPT_VERSION = 1
SEARCH = RetrievalConfig(mode="dense", k=5)  # the original's top 5, searched with no hard filters

_FIRST_WORD = re.compile(r"\W*(yes|no)\b", re.IGNORECASE)


def strict_yes(reply: str) -> bool:
    """The original parser: ``"Yes."`` and ``"Yes\\n"`` are rejected (defect #4)."""
    return reply.lower().strip(" ") == "yes"


def fixed_yes(reply: str) -> bool:
    match = _FIRST_WORD.match(reply)
    return match is not None and match.group(1).lower() == "yes"


PARSERS: dict[str, Callable[[str], bool]] = {"strict": strict_yes, "fixed": fixed_yes}


def _words(value: str) -> str:
    return value.replace("_", " ")


def constraint_phrase(filters: Filters) -> str:
    """The query's filters in words, as a user of the original typed them into the request."""
    parts = []
    if filters.diet is not None:
        parts.append(_words(filters.diet.value))
    if filters.eggless:
        parts.append("eggless")
    if filters.exclude_allergens:
        parts.append("no " + ", ".join(_words(a.value) for a in sorted(filters.exclude_allergens)))
    if filters.max_kcal is not None:
        parts.append(f"under {filters.max_kcal} kcal")
    if filters.max_price_inr is not None:
        parts.append(f"under ₹{filters.max_price_inr}")
    if filters.cuisines:
        parts.append(" or ".join(_words(c.value) for c in sorted(filters.cuisines)) + " cuisine")
    return ", ".join(parts)


def request_text(query: TextQuery) -> str:
    """The raw request C0 searches and judges: the query text with its constraints as words, never as filters."""
    phrase = constraint_phrase(query.filters)
    return f"{query.text}, {phrase}" if phrase else query.text


def dish_card(item: MenuItem) -> str:
    """What the original showed the model: the menu row, with the declared allergens only (blank when unlabelled)."""
    if item.label_allergens is None:
        allergens = ""
    else:
        allergens = ", ".join(_words(a.value) for a in sorted(item.label_allergens)) or "none"
    return (
        f"Name: {item.name}\nDescription: {item.description}\nIngredients: {', '.join(item.ingredients)}\n"
        f"Diet: {_words(item.diet.value)}\nAllergens: {allergens}\nCalories: {item.kcal} kcal\n"
        f"Price: ₹{item.price_inr}"
    )


def relevance_prompt(request: str, item: MenuItem) -> str:
    return (
        "You check whether a dish suits a customer's request.\n\n"
        f"Request: {request}\n\nDish:\n{dish_card(item)}\n\n"
        "Is this dish relevant to the request? Answer Yes or No."
    )


class Judged(NamedTuple):
    reply: str
    tokens: int


class OriginalRun(BaseModel):
    """One query's retrieval and verdicts; a changed request makes it stale, so it is asked again."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str = Field(pattern=r"^t\d{3}$")
    request: str
    model: str
    prompt_version: int
    retrieved: tuple[str, ...]  # dense top 5, in rank order
    replies: dict[str, str]  # item_id → the model's raw reply
    tokens: int = Field(default=0, ge=0)

    def ranked(self, parser: Callable[[str], bool]) -> tuple[str, ...]:
        return tuple(item for item in self.retrieved if parser(self.replies[item]))


def _current(run: OriginalRun | None, query: TextQuery, model: str) -> bool:
    return (
        run is not None
        and run.request == request_text(query)
        and run.model == model
        and run.prompt_version == PROMPT_VERSION
    )


def collect(
    queries: Sequence[TextQuery],
    search: Callable[[str], Sequence[str]],
    judge: Callable[[str, str], Judged],
    model: str,
    path: Path,
    limit: int | None = None,
) -> dict[str, OriginalRun]:
    """Ask the missing or stale queries (``judge(request, item_id)``), saving after each one; resumable."""
    runs = read_runs(path)
    todo = [query for query in queries if not _current(runs.get(query.id), query, model)]
    for query in todo[:limit]:
        request = request_text(query)
        retrieved = tuple(search(request))
        judged = {item: judge(request, item) for item in retrieved}
        runs[query.id] = OriginalRun(
            query_id=query.id,
            request=request,
            model=model,
            prompt_version=PROMPT_VERSION,
            retrieved=retrieved,
            replies={item: verdict.reply for item, verdict in judged.items()},
            tokens=sum(verdict.tokens for verdict in judged.values()),
        )
        write_runs(path, runs.values())
    return runs


def missing(queries: Iterable[TextQuery], runs: Mapping[str, OriginalRun], model: str) -> list[str]:
    return [query.id for query in queries if not _current(runs.get(query.id), query, model)]


def replay(
    queries: Sequence[TextQuery], runs: Mapping[str, OriginalRun], truth: GroundTruth
) -> dict[str, list[QueryResult]]:
    """Score C0 from the cache under each parser; ``empty`` marks a query that got no dish at all (defect #5)."""
    results: dict[str, list[QueryResult]] = {}
    for name, parser in PARSERS.items():
        rows = []
        for query in queries:
            ranked = runs[query.id].ranked(parser)
            metrics = score_query(query, ranked, truth) | {"empty": float(not ranked)}
            rows.append(QueryResult(query.id, query.type, ranked, metrics, ()))
        results[f"C0_{name}"] = rows
    return results


def read_runs(path: Path) -> dict[str, OriginalRun]:
    """Runs by query id; an absent file is an empty cache."""
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {run.query_id: run for run in (OriginalRun.model_validate_json(line) for line in lines)}


def write_runs(path: Path, runs: Iterable[OriginalRun]) -> None:
    """One run per line in query order, LF line ends, so the file diffs cleanly."""
    ordered = sorted(runs, key=lambda run: run.query_id)
    lines = [json.dumps(run.model_dump(mode="json"), ensure_ascii=False) for run in ordered]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")
