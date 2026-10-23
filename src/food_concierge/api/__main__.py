"""Run the HTTP server: python -m food_concierge.api

Serves ``/health``, ``/ready``, ``/v1/search`` and MCP at ``/mcp`` on ``API_HOST``:``API_PORT``. Build the catalog
and indexes first (``python -m food_concierge.flows build``).
"""

from __future__ import annotations

import uvicorn

from food_concierge.api.app import create_app
from food_concierge.config import get_settings
from food_concierge.logging_setup import configure_logging
from food_concierge.mcp_server.services import build_services


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(create_app(build_services(settings)), host=settings.api_host, port=settings.api_port, log_config=None)


if __name__ == "__main__":
    main()
