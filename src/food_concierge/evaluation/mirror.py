"""Mirror a results file to Langfuse (Phase 3 plan §7): each query a dataset item, each candidate an experiment run.

The scores are the ones already computed; Langfuse only displays them. It runs only when Langfuse keys are
configured (in Phase 3, a local Docker Langfuse), and a failure is logged and swallowed (rules §4.5): the results
file is the record, the mirror is a view. Query text is synthetic, with no personal data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from food_concierge.evaluation.dataset import TextQuery
from food_concierge.evaluation.results import RetrievalResults

logger = logging.getLogger(__name__)

SCORES = ("ndcg@3", "success@3", "violation@3")


def mirror_results(client: Any, results: RetrievalResults, queries: Sequence[TextQuery]) -> bool:
    """True when every candidate was mirrored; ``client`` is a ``Langfuse`` client, or None to skip."""
    if client is None:
        return False
    try:
        from langfuse import Evaluation

        dataset = f"retrieval-{results.split}"
        client.create_dataset(name=dataset, metadata={"queries_sha256": results.queries_sha256})
        scored = {row.query_id for candidate in results.candidates for row in candidate.queries}
        items = [
            client.create_dataset_item(
                dataset_name=dataset,
                id=f"{dataset}-{query.id}",
                input={"query": query.text, "type": query.type},
                metadata={"query_id": query.id},
            )
            for query in queries
            if query.id in scored
        ]
        for candidate in results.candidates:
            rows = {row.query_id: row for row in candidate.queries}

            def task(*, item: Any, rows: Any = rows, **_: Any) -> list[str]:
                return list(rows[item.metadata["query_id"]].ranked)

            def scores(*, metadata: Any, rows: Any = rows, **_: Any) -> list[Any]:
                metrics = rows[metadata["query_id"]].metrics
                return [Evaluation(name=name, value=metrics[name]) for name in SCORES if name in metrics]

            client.run_experiment(
                name=dataset,
                run_name=f"{candidate.name} @ {results.record.git_sha[:7]}",
                data=items,
                task=task,
                evaluators=[scores],
                metadata={"candidate": candidate.name, "git_sha": results.record.git_sha},
            )
        client.flush()
    except Exception:
        logger.warning("langfuse mirror failed; the results file is unaffected", exc_info=True)
        return False
    return True
