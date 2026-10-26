"""Whether a search's best match is weak, so the menu may have nothing like the query.

Search always returns the nearest dishes, so the list alone can't tell "nothing fits" from a match. A match is weak
when both signals agree:

- the best dish's cosine similarity is below the embedding model's ``WEAK_SCORES`` entry (dense search only: other
  stacks score on other scales, and a model without an entry never reports a weak match);
- no word of the query, other than a common function word, appears in any dish's search text.

Neither is enough alone: "rogan josh" finds its dish at 0.534 while "qwerty uiop" scores 0.543, and good matches
often share no word with the query ("something to cool me down on a very hot day"). Measured on 2026-10-26 with
``BAAI/bge-small-en-v1.5``: the 62 drafted queries' best scores start at 0.540 and none is weak; eight off-menu
probes ("asdfghjkl", "what is the weather", "book me a taxi to the airport") score 0.434-0.543 and all are weak.
Provisional until the owner's grades (#42) calibrate it; the PR gate fails if a dev query is weak.
"""

from __future__ import annotations

from collections.abc import Iterable

from food_concierge.services.retrieval import tokenize

WEAK_SCORES = {"BAAI/bge-small-en-v1.5": 0.55}  # cosine cut-off per embedding model, measured as above
FUNCTION_WORDS = frozenset(
    tokenize(
        "a an and any anything are as at be by can do for from get give have i in is it me my of on or please some "
        "something that the this to want what with you"
    )
)
WEAK_MATCH = "Even the closest dishes are a weak match; the menu may have nothing like this."


class MenuWords:
    """Every word in the dishes' search text."""

    def __init__(self, texts: Iterable[str]) -> None:
        self._words = frozenset(word for text in texts for word in tokenize(text))

    def mentions(self, query: str) -> bool:
        """Whether a word of the query, other than a function word, appears on the menu."""
        return any(word in self._words for word in tokenize(query) if word not in FUNCTION_WORDS)


def is_weak(query: str, best_score: float, cutoff: float, words: MenuWords) -> bool:
    return best_score < cutoff and not words.mentions(query)
