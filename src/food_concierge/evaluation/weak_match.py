"""How well a signal tells "the menu has nothing like this" from a real match (#143), on held-out queries.

``eval/datasets/weak_match.yaml`` labels queries on or off the menu. ``measure`` runs each through the search the
tools serve and records what a signal could use: the best dish's cosine similarity, the margin from the best to the
tenth, whether a query word is on the menu, and the cross-encoder's score of the best of the ten for the query.
Signals call a match weak when a value falls below a cut-off; ``calibrate`` picks each cut-off on the dev split,
the one that flags the most off-menu queries while calling at most ``MAX_FALSE_WEAK`` of the on-menu ones weak, and
``score`` reports it on the test split, which played no part in choosing it.

Precision is the share of weak flags that are off the menu; recall the share of off-menu queries flagged; the
false-weak rate the share of on-menu queries flagged, which would tell a diner the menu has nothing when it does.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from food_concierge.evaluation.bootstrap import Interval, bootstrap_mean
from food_concierge.evaluation.record import RunRecord
from food_concierge.models.embeddings import Reranker
from food_concierge.services.constraints import Constraints
from food_concierge.services.relevance import MenuWords
from food_concierge.services.search import MAX_RESULTS, SearchService

Group = Literal["named", "category", "vague", "near_miss", "off_topic"]
Split = Literal["dev", "test"]

MAX_FALSE_WEAK = 0.05  # at most 1 in 20 on-menu queries may be told the menu has nothing like them
SEED = 143


class WeakQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1)
    on_menu: bool
    group: Group
    seen: bool = False  # shaped the signal before this set existed; always dev


class WeakMatchSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    queries: tuple[WeakQuery, ...]


def load_weak_match_set(path: Path) -> WeakMatchSet:
    data = WeakMatchSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    texts = [q.query for q in data.queries]
    if len(set(texts)) != len(texts):
        raise ValueError("a query appears twice in the weak-match set")
    return data


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_of(queries: Sequence[WeakQuery]) -> dict[str, Split]:
    """Seen queries go to dev; the rest of each group is ordered by SHA-256 and the first half goes to dev."""
    splits: dict[str, Split] = {q.query: "dev" for q in queries if q.seen}
    groups: dict[str, list[WeakQuery]] = {}
    for q in queries:
        groups.setdefault(q.group, []).append(q)
    for members in groups.values():
        dev_quota = (len(members) + 1) // 2 - sum(q.seen for q in members)
        for rank, q in enumerate(sorted((q for q in members if not q.seen), key=lambda q: _digest(q.query))):
            splits[q.query] = "dev" if rank < dev_quota else "test"
    return splits


class FeatureRow(BaseModel):
    """What the serving search returned for one query, and the values a signal could use."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    on_menu: bool
    group: Group
    split: Split
    returned: int  # dishes returned (up to MAX_RESULTS); 0 when the query's own constraints exclude every dish
    top_score: float | None  # best cosine similarity
    margin: float | None  # best minus the last returned (the tenth when ten came back)
    menu_word: bool  # a query word, other than a function word, appears on the menu
    cross_encoder: float | None  # the reranker's best score over the returned dishes; None without a reranker
    served_weak: bool  # what search itself said, with the production cut-off
    top_dish: str | None


class WeakMatchRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record: RunRecord
    dataset_sha256: str
    dataset_version: int
    weak_score: float | None  # the production cut-off the search ran with
    rows: tuple[FeatureRow, ...]


def measure(
    dataset: WeakMatchSet,
    search: SearchService,
    texts: Mapping[str, str],
    words: MenuWords,
    reranker: Reranker | None,
) -> list[FeatureRow]:
    """Each query through ``search`` as a client calls it: no constraints but those its own words state."""
    splits = split_of(dataset.queries)
    rows = []
    for q in dataset.queries:
        found = search.search(q.query, Constraints(), MAX_RESULTS)
        scores = [dish.score for dish in found.dishes]
        cross = None
        if reranker is not None and found.dishes:
            cross = max(reranker.score(q.query, [texts[dish.item_id] for dish in found.dishes]))
        rows.append(
            FeatureRow(
                query=q.query,
                on_menu=q.on_menu,
                group=q.group,
                split=splits[q.query],
                returned=len(scores),
                top_score=scores[0] if scores else None,
                margin=scores[0] - scores[-1] if scores else None,
                menu_word=words.mentions(q.query),
                cross_encoder=cross,
                served_weak=found.weak_match,
                top_dish=found.dishes[0].name if found.dishes else None,
            )
        )
    return rows


@dataclass(frozen=True)
class Signal:
    name: str
    description: str
    value: Callable[[FeatureRow], float | None]
    word_veto: bool = False  # a query word on the menu means "not weak", whatever the value

    def weak(self, row: FeatureRow, cutoff: float) -> bool:
        value = self.value(row)
        if value is None or (self.word_veto and row.menu_word):
            return False
        return value < cutoff


SIGNALS = (
    Signal(
        "cosine_no_menu_word",
        "best cosine below the cut-off and no query word on the menu (the #135 rule)",
        lambda r: r.top_score,
        word_veto=True,
    ),
    Signal("cosine", "best cosine below the cut-off", lambda r: r.top_score),
    Signal("margin", "best cosine minus the tenth below the cut-off", lambda r: r.margin),
    Signal("cross_encoder", "the reranker's best score over the ten below the cut-off", lambda r: r.cross_encoder),
)


@dataclass(frozen=True)
class Scores:
    cutoff: float
    flagged: int
    precision: float  # of the flagged, the share off the menu; 1.0 when nothing is flagged
    recall: Interval  # off-menu queries flagged
    false_weak: Interval  # on-menu queries flagged
    by_group: dict[str, float]  # share of each group flagged


def _scorable(rows: Sequence[FeatureRow]) -> list[FeatureRow]:
    return [r for r in rows if r.returned]


def _rates(signal: Signal, rows: Sequence[FeatureRow], cutoff: float) -> tuple[float, float]:
    off = [r for r in rows if not r.on_menu]
    on = [r for r in rows if r.on_menu]
    recall = sum(signal.weak(r, cutoff) for r in off) / len(off) if off else 0.0
    false_weak = sum(signal.weak(r, cutoff) for r in on) / len(on) if on else 0.0
    return recall, false_weak


def calibrate(signal: Signal, rows: Sequence[FeatureRow], max_false_weak: float = MAX_FALSE_WEAK) -> float | None:
    """The cut-off with the best recall on ``rows`` within the false-weak budget; ties go to fewer false weaks.

    Candidates are the midpoints between consecutive observed values. ``None`` when the signal has no values.
    """
    values = sorted({v for r in _scorable(rows) if (v := signal.value(r)) is not None})
    if not values:
        return None
    candidates = [values[0] - 1.0] + [(a + b) / 2 for a, b in pairwise(values)]
    best: tuple[float, float, float] | None = None  # (recall, -false_weak, -cutoff)
    chosen = candidates[0]
    for cutoff in candidates:
        recall, false_weak = _rates(signal, _scorable(rows), cutoff)
        if false_weak > max_false_weak:
            continue
        key = (recall, -false_weak, -cutoff)
        if best is None or key > best:
            best, chosen = key, cutoff
    return chosen


def score(signal: Signal, rows: Sequence[FeatureRow], cutoff: float, seed: int = SEED) -> Scores:
    usable = _scorable(rows)
    off = [float(signal.weak(r, cutoff)) for r in usable if not r.on_menu]
    on = [float(signal.weak(r, cutoff)) for r in usable if r.on_menu]
    flagged = [r for r in usable if signal.weak(r, cutoff)]
    groups: dict[str, list[FeatureRow]] = {}
    for r in usable:
        groups.setdefault(r.group, []).append(r)
    return Scores(
        cutoff=cutoff,
        flagged=len(flagged),
        precision=sum(not r.on_menu for r in flagged) / len(flagged) if flagged else 1.0,
        recall=bootstrap_mean(off, seed),
        false_weak=bootstrap_mean(on, seed),
        by_group={g: sum(signal.weak(r, cutoff) for r in rs) / len(rs) for g, rs in sorted(groups.items())},
    )


@dataclass(frozen=True)
class SignalReport:
    signal: Signal
    calibrated: bool  # False for the production rule at its fixed cut-off
    dev: Scores
    test: Scores


def report(run: WeakMatchRun) -> list[SignalReport]:
    """The production rule at its own cut-off, then each signal calibrated on dev; both scored on dev and test."""
    dev = [r for r in run.rows if r.split == "dev"]
    test = [r for r in run.rows if r.split == "test"]
    reports = []
    if run.weak_score is not None:
        production = SIGNALS[0]
        reports.append(
            SignalReport(
                production, False, score(production, dev, run.weak_score), score(production, test, run.weak_score)
            )
        )
    for signal in SIGNALS:
        cutoff = calibrate(signal, dev)
        if cutoff is not None:
            reports.append(SignalReport(signal, True, score(signal, dev, cutoff), score(signal, test, cutoff)))
    return reports


def _pct(interval: Interval) -> str:
    return f"{interval.mean:.0%} [{interval.low:.0%}, {interval.high:.0%}]"


def render(run: WeakMatchRun, reports: Sequence[SignalReport]) -> str:
    counts = {s: sum(r.split == s for r in run.rows) for s in ("dev", "test")}
    empty = [r.query for r in run.rows if not r.returned]
    lines = [
        "# Weak-match signals on held-out queries (#143)",
        "",
        f"Dataset `eval/datasets/weak_match.yaml` v{run.dataset_version}: {len(run.rows)} queries "
        f"({sum(r.on_menu for r in run.rows)} on the menu), dev {counts['dev']}, test {counts['test']}. "
        f"Commit `{run.record.git_sha[:7]}`. Cut-offs are chosen on dev for the most off-menu queries flagged with at "
        f"most {MAX_FALSE_WEAK:.0%} of on-menu queries flagged; test is reported once with 95% bootstrap intervals.",
        "",
    ]
    if empty:
        lines += [f"Left out (the query's own constraints exclude every dish): {', '.join(empty)}.", ""]
    lines += [
        "| Signal | Cut-off | Test precision | Test recall | Test false-weak | Dev recall | Dev false-weak |",
        "|---|---|---|---|---|---|---|",
    ]
    for rep in reports:
        name = f"{rep.signal.name} ({'calibrated' if rep.calibrated else 'production'})"
        lines.append(
            f"| {name} | {rep.test.cutoff:.3f} | {rep.test.precision:.0%} ({rep.test.flagged} flagged) | "
            f"{_pct(rep.test.recall)} | {_pct(rep.test.false_weak)} | {rep.dev.recall.mean:.0%} | "
            f"{rep.dev.false_weak.mean:.0%} |"
        )
    groups = sorted({r.group for r in run.rows})
    lines += [
        "",
        "Share of each test group flagged:",
        "",
        "| Signal | " + " | ".join(groups) + " |",
        "|---|" + "---|" * len(groups),
    ]
    for rep in reports:
        name = f"{rep.signal.name} ({'calibrated' if rep.calibrated else 'production'})"
        lines.append(f"| {name} | " + " | ".join(f"{rep.test.by_group.get(g, 0.0):.0%}" for g in groups) + " |")
    lines += [
        "",
        "What each signal sees, by group (all splits; min / median / max):",
        "",
        "| Group | Queries | Best cosine | Margin | Cross-encoder | Query word on the menu |",
        "|---|---|---|---|---|---|",
    ]
    for group in groups:
        rows = [r for r in run.rows if r.group == group]
        lines.append(
            f"| {group} | {len(rows)} | {_spread([r.top_score for r in rows])} | {_spread([r.margin for r in rows])} | "
            f"{_spread([r.cross_encoder for r in rows])} | {sum(r.menu_word for r in rows)} |"
        )
    lines += ["", "Signals:", ""]
    lines += [f"- `{s.name}`: {s.description}." for s in SIGNALS]
    return "\n".join(lines) + "\n"


def _spread(values: Sequence[float | None]) -> str:
    known = sorted(v for v in values if v is not None)
    if not known:
        return "n/a"
    return f"{known[0]:.2f} / {median(known):.2f} / {known[-1]:.2f}"
