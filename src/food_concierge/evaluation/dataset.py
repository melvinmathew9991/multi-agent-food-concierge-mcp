"""The text-query dataset (``eval/datasets/retrieval.yaml``): schema, dev/test split and the test-split lock.

The test split is touched once, after ADR-0001's decision. Its content is fixed by a SHA-256 recorded in the
file, and ``select`` refuses it unless the caller says the run is final, so the decision cannot be tuned on it.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from food_concierge.errors import EvaluationError
from food_concierge.storage.catalog_db import Filters

QueryType = Literal["known_item", "described", "vague", "attribute", "colloquial", "constrained"]
Split = Literal["dev", "test"]
Grade = int  # 0 not relevant, 1 related but wrong on something that matters, 2 relevant, 3 exactly what was asked

RELEVANT = 2  # Success@k counts an item graded at least this
EXACT = 3  # Recall@k and MRR count items graded this, for known-item and described queries
TARGETED_TYPES = frozenset({"known_item", "described"})


class TextQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^t\d{3}$")
    text: str = Field(min_length=1, max_length=500)
    type: QueryType
    author: Literal["drafted", "owner"]  # owner queries are written without looking at the catalog
    group: str | None = None  # near-duplicate queries share a group, and so a split
    split: Split | None = None
    filters: Filters = Filters()
    target: str | None = None  # dish_key of the dish a known-item or described query names; its items grade 3
    grades: dict[str, Grade] = {}

    @model_validator(mode="after")
    def _grades_in_range(self) -> TextQuery:
        bad = sorted(item for item, grade in self.grades.items() if not 0 <= grade <= EXACT)
        if bad:
            raise ValueError(f"grades must be 0-3 ({', '.join(bad)})")
        if self.target is not None and self.type not in TARGETED_TYPES:
            raise ValueError("only known-item and described queries name a target dish")
        return self

    @property
    def unit(self) -> str:
        return self.group or self.id

    @property
    def exact(self) -> frozenset[str]:
        return frozenset(item for item, grade in self.grades.items() if grade == EXACT)


class RetrievalDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    test_sha256: str | None = None
    queries: tuple[TextQuery, ...]

    @model_validator(mode="after")
    def _consistent(self) -> RetrievalDataset:
        ids = [query.id for query in self.queries]
        duplicates = sorted({query_id for query_id in ids if ids.count(query_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate query ids: {', '.join(duplicates)}")
        splits: dict[str, set[Split | None]] = defaultdict(set)
        for query in self.queries:
            splits[query.unit].add(query.split)
        split_groups = sorted(unit for unit, seen in splits.items() if len(seen) > 1)
        if split_groups:
            raise ValueError(f"a group spans both splits: {', '.join(split_groups)}")
        if self.test_sha256 is not None and self.test_sha256 != split_digest(self.queries):
            raise ValueError("the test split changed after it was locked")
        return self


def split_digest(queries: Sequence[TextQuery]) -> str:
    """SHA-256 over the test queries, everything included, in id order."""
    test = sorted((q for q in queries if q.split == "test"), key=lambda q: q.id)
    payload = json.dumps([q.model_dump(mode="json") for q in test], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def assign_splits(queries: Sequence[TextQuery], seed: int) -> tuple[TextQuery, ...]:
    """Half of each query type's units (groups, or single queries) to dev and half to test, by seeded shuffle."""
    units: dict[str, list[str]] = defaultdict(list)
    for query in queries:
        if query.unit not in units[query.type]:
            units[query.type].append(query.unit)
    side: dict[str, Split] = {}
    for query_type in sorted(units):
        shuffled = sorted(units[query_type])
        random.Random(f"{seed}:{query_type}").shuffle(shuffled)  # noqa: S311 - reproducible split, not security
        side |= {unit: "dev" if position % 2 == 0 else "test" for position, unit in enumerate(shuffled)}
    return tuple(query.model_copy(update={"split": side[query.unit]}) for query in queries)


def lock(dataset: RetrievalDataset) -> RetrievalDataset:
    if any(query.split is None for query in dataset.queries):
        raise EvaluationError("Assign every query a split before locking the test split.")
    return dataset.model_copy(update={"test_sha256": split_digest(dataset.queries)})


def load_dataset(path: Path) -> RetrievalDataset:
    try:
        return RetrievalDataset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise EvaluationError(f"The dataset {path.name} is missing or invalid: {exc}") from exc


def select(dataset: RetrievalDataset, split: Split, final: bool = False) -> tuple[TextQuery, ...]:
    if split == "test":
        if dataset.test_sha256 is None:
            raise EvaluationError("The test split is not locked yet.")
        if not final:
            raise EvaluationError("The test split runs once, after ADR-0001; pass final=True (--final) for that run.")
    return tuple(query for query in dataset.queries if query.split == split)
