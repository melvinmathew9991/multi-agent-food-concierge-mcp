"""The vision measurement script: sampling, scoring checks and the summary behind ADR-0007's numbers."""

import importlib.util
import sys
from types import ModuleType
from typing import Any

from food_concierge.config import REPO_ROOT, get_settings
from food_concierge.ingestion.allergens import load_lexicon
from food_concierge.ingestion.descriptions import (
    DESCRIPTIONS_FILE,
    DescriptionKey,
    ImageDescription,
    Variant,
    read_descriptions,
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vision_measurement", REPO_ROOT / "scripts" / "vision_measurement.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vision_measurement"] = module
    spec.loader.exec_module(module)
    return module


vm = _load()


def _sha(n: int) -> str:
    return f"{n:064x}"


def _cache(texts: dict[str, str], latency_ms: int = 1000) -> dict[DescriptionKey, ImageDescription]:
    cache = {}
    for sha, text in texts.items():
        for variant in Variant:
            cache[(sha, variant)] = ImageDescription(
                sha256=sha, variant=variant, model="m", prompt_version=1, text=text, latency_ms=latency_ms
            )
    return cache


def _entry(**scores: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "sha256": _sha(1),
        "item_id": "x01",
        "dish": "Dish",
        "description": "text",
        "allergen_terms": {},
        "dish_type": "correct",
        "seen": [],
        "not_seen": [],
        "not_visible": [],
        "note": "",
    }
    return entry | scores


def test_the_sample_is_seeded_sized_and_sorted() -> None:
    shas = [_sha(n) for n in range(100)]

    first = vm.draw(shas)

    assert first == vm.draw(list(reversed(shas)))
    assert len(first) == vm.SAMPLE_SIZE
    assert first == sorted(first)
    assert vm.draw(shas[:5]) == shas[:5]


def test_allergen_terms_are_the_taggers_vision_evidence() -> None:
    terms = vm.allergen_terms("Rice topped with sesame seeds and a fried egg, beside a tomato.", load_lexicon())

    assert terms == {"sesame": ["sesame"], "egg": ["eggs"]}


def test_resampling_keeps_scores_unless_the_description_changed() -> None:
    cache = _cache({_sha(1): "A bowl of paneer.", _sha(2): "Fish curry."})
    dishes = {_sha(1): ("a01", "Paneer"), _sha(2): ("a02", "Fish"), _sha(3): ("a03", "Undescribed")}
    previous = {
        _sha(1): _entry(sha256=_sha(1), description="A bowl of paneer.", dish_type="wrong", note="kept"),
        _sha(2): _entry(sha256=_sha(2), description="An older text.", dish_type="wrong"),
    }

    entries = vm.build_entries(cache, dishes, load_lexicon(), previous)

    assert [e["item_id"] for e in entries] == ["a01", "a02"]
    assert entries[0]["dish_type"] == "wrong"
    assert entries[0]["note"] == "kept"
    assert entries[0]["allergen_terms"] == {"paneer": ["milk"]}
    assert entries[1]["dish_type"] is None
    assert entries[1]["allergen_terms"] == {"fish": ["fish"]}


def test_a_report_needs_every_entry_scored_and_current() -> None:
    cache = _cache({_sha(1): "text"})
    good = _entry(allergen_terms={"egg": ["eggs"]}, not_visible=["egg"])

    assert vm.problems([good], cache) == []
    found = vm.problems(
        [_entry(dish_type=None), _entry(not_visible=["fish"]), _entry(description="old"), _entry(sha256=_sha(9))],
        cache,
    )
    assert len(found) == 4
    assert "dish_type" in found[0]
    assert "fish" in found[1]
    assert "changed" in found[2]
    assert "changed" in found[3]


def test_the_summary_counts_types_ingredients_and_false_allergen_terms() -> None:
    entries = [
        _entry(seen=["a", "b", "c"], not_seen=["d"], allergen_terms={"fish": ["fish"]}, not_visible=["fish"]),
        _entry(dish_type="partial", seen=["a"], allergen_terms={"egg": ["eggs"]}),
        _entry(dish_type="wrong"),
        _entry(seen=["a", "b"]),
    ]

    summary = vm.summarise(entries)

    assert summary["dish_type"]["counts"] == {"correct": 2, "partial": 1, "wrong": 1}
    assert summary["dish_type"]["correct"]["rate"] == 0.5
    assert summary["dish_type"]["correct_or_partial"]["k"] == 3
    assert summary["ingredients"]["precision"] == {"rate": 0.857, "k": 6, "n": 7, "ci95": vm.wilson(6, 7)}
    assert summary["ingredients"]["descriptions_with_an_unseen_ingredient"]["k"] == 1
    assert summary["allergen_terms"]["not_visible"]["rate"] == 0.5
    assert summary["allergen_terms"]["descriptions_with_a_false_term"]["k"] == 1


def test_rates_and_percentiles_handle_empty_input() -> None:
    assert vm.rate(0, 0) == {"rate": None, "k": 0, "n": 0, "ci95": (0.0, 0.0)}
    assert vm.percentile([], 0.5) is None
    assert vm.percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert vm.percentile(list(range(1, 101)), 0.95) == 95
    low, high = vm.wilson(23, 30)
    assert low < 23 / 30 < high


def test_latency_is_per_variant_over_the_whole_cache() -> None:
    timing = vm.latency(_cache({_sha(1): "a", _sha(2): "b"}, latency_ms=3000))

    assert timing == {
        "name_free": {"n": 2, "p50_ms": 3000, "p95_ms": 3000},
        "named": {"n": 2, "p50_ms": 3000, "p95_ms": 3000},
    }


def test_render_says_whether_the_owner_has_reviewed() -> None:
    summary = vm.summarise([_entry(seen=["a"])])
    timing = vm.latency(_cache({_sha(1): "a"}))

    assert "reviewed by the owner" in vm.render(summary, timing, reviewed=True)
    text = vm.render(summary, timing, reviewed=False)
    assert "NOT YET REVIEWED" in text
    assert "Allergen terms whose food is not in the photo: n/a" in text


def test_the_review_page_escapes_model_text() -> None:
    page = vm.review_page([_entry(description="<script>x</script>", allergen_terms={"egg": ["eggs"]})])

    assert "<script>x" not in page
    assert "&lt;script&gt;" in page
    assert f"thumbs/{_sha(1)}.webp" in page
    assert "egg → eggs" in page


def test_the_committed_sample_is_complete_and_matches_the_cache() -> None:
    dataset = vm.read_dataset(vm.DATASET)
    cache = read_descriptions(get_settings().processed_dir / DESCRIPTIONS_FILE)

    assert len(dataset["entries"]) == vm.SAMPLE_SIZE
    assert vm.problems(dataset["entries"], cache) == []
    terms = load_lexicon()
    assert all(e["allergen_terms"] == vm.allergen_terms(e["description"], terms) for e in dataset["entries"])
