"""Nightly: the real embedding model must still produce the committed reference vectors.

fastembed cannot pin a model revision, so a silent upstream change would otherwise
surface only as worse search results and stale evaluation caches.
"""

import json

import numpy as np
import pytest

from food_concierge.config import REPO_ROOT, Settings
from food_concierge.models.embeddings import FastEmbedder

pytestmark = pytest.mark.model_download

MIN_COSINE = 0.999


def test_embeddings_match_the_committed_reference() -> None:
    reference = json.loads((REPO_ROOT / "tests" / "fixtures" / "embedding_reference.json").read_text("utf-8"))
    embedder = FastEmbedder(Settings(_env_file=None))

    documents = embedder.embed_documents(reference["documents"])
    query = embedder.embed_query(reference["query"])

    cosines = np.sum(documents * np.asarray(reference["document_vectors"]), axis=1)
    fingerprint = embedder.fingerprint.model_dump()
    assert cosines.min() >= MIN_COSINE, f"document vectors drifted: {cosines.round(4)}; now {fingerprint}"
    query_cosine = float(query @ np.asarray(reference["query_vector"]))
    assert query_cosine >= MIN_COSINE, f"query vector drifted: {query_cosine:.4f}; now {fingerprint}"
    # Same vectors but a new snapshot or file hash still makes every built index refuse to load.
    assert fingerprint == reference["fingerprint"], f"the model files changed upstream; now {fingerprint}"
