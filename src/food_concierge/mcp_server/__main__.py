"""Run the MCP server on stdio, for Claude Desktop and MCP Inspector.

python -m food_concierge.mcp_server

Build the catalog and indexes first (``python -m food_concierge.flows build``). stdout carries the protocol, so
logs go to stderr.
"""

from __future__ import annotations

from food_concierge.config import get_settings
from food_concierge.logging_setup import configure_logging
from food_concierge.mcp_server.server import create_server
from food_concierge.mcp_server.services import build_services


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    create_server(build_services(settings)).run("stdio")


if __name__ == "__main__":
    main()
