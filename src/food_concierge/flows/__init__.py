"""Prefect flows for the offline pipeline (Prefect OSS, local; never imported by the API)."""

import os

# Prefect's local server sends anonymous usage analytics by default; this project sends nothing it doesn't need to.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
