"""Model layer: chat model router, embeddings and test doubles.

The rest of the application asks this package for a model by role and receives
app-level errors (``food_concierge.errors``), never provider SDK exceptions.
"""
