"""Write the reference vectors the nightly embedding drift check compares against.

Run once when the embedding model or query prefix changes on purpose, and commit the result:

    python scripts/build_embedding_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

from food_concierge.config import REPO_ROOT, Settings
from food_concierge.models.embeddings import FastEmbedder

REFERENCE = REPO_ROOT / "tests" / "fixtures" / "embedding_reference.json"

# Short, varied texts: a query, dishes with and without allergen words, and a non-food sentence.
DOCUMENTS = [
    "Vegan chickpea curry with coconut milk and basmati rice",
    "Chocolate lava cake with vanilla ice cream",
    "Grilled salmon with lemon butter and asparagus",
    "Peanut noodles with tofu and spring onions",
    "The train to the airport leaves every twenty minutes",
]
QUERY = "spicy vegetarian dinner without nuts"


def build(path: Path = REFERENCE) -> None:
    embedder = FastEmbedder(Settings(_env_file=None))
    reference = {
        "fingerprint": embedder.fingerprint.model_dump(),
        "documents": DOCUMENTS,
        "query": QUERY,
        "document_vectors": [[round(float(x), 6) for x in row] for row in embedder.embed_documents(DOCUMENTS)],
        "query_vector": [round(float(x), 6) for x in embedder.embed_query(QUERY)],
    }
    path.write_text(json.dumps(reference, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
