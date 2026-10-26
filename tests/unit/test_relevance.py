"""The weak-match rule: a low best score and no query word on the menu, both needed."""

import pytest

from food_concierge.services.relevance import FUNCTION_WORDS, WEAK_SCORES, MenuWords, is_weak

WORDS = MenuWords(["Paneer Butter Masala. Paneer in a tomato and butter gravy.", "Rogan Josh. Lamb curry."])


@pytest.mark.parametrize(
    ("query", "mentions"),
    [("rogan josh", True), ("PANEER please", True), ("asdfghjkl", False), ("what is the weather", False)],
)
def test_the_menu_mentions_a_query_word_other_than_a_function_word(query: str, mentions: bool) -> None:
    assert WORDS.mentions(query) is mentions


def test_a_function_word_on_the_menu_doesnt_count() -> None:
    assert {"the", "and", "in", "a"} <= FUNCTION_WORDS
    assert not WORDS.mentions("the and in a")


def test_a_match_is_weak_only_when_both_signals_agree() -> None:
    cutoff = WEAK_SCORES["BAAI/bge-small-en-v1.5"]
    assert is_weak("asdfghjkl", 0.52, cutoff, WORDS)
    assert not is_weak("rogan josh", 0.534, cutoff, WORDS)  # low score, but the menu has the word
    assert not is_weak("asdfghjkl", 0.6, cutoff, WORDS)  # no menu word, but close in meaning
