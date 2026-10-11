"""Retrieval evaluation (Phase 3): datasets, metrics, bootstrap intervals, offline caches and the runner.

Everything here runs offline. Model outputs come from committed caches through the same ``Embedder`` and
``Reranker`` protocols the service uses, so the evaluation exercises the production ``Retriever``.
"""
