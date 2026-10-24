"""The layering rule of ADR-0002, read from the source: services know no protocol, the MCP server knows no web
framework, and the agent (Phase 5) reaches services only through MCP.

Imports under ``if TYPE_CHECKING:`` are type hints, not dependencies, so they are allowed.
"""

import ast

import pytest

from food_concierge.config import REPO_ROOT

PACKAGE = REPO_ROOT / "src" / "food_concierge"
PROTOCOLS = ("mcp", "mcp_types", "fastapi", "starlette", "uvicorn", "food_concierge.mcp_server", "food_concierge.api")
WEB = ("fastapi", "starlette", "uvicorn", "food_concierge.api")
AGENT_FRAMEWORKS = ("langchain", "langchain_core", "langgraph", "food_concierge.agent")

FORBIDDEN: dict[str, tuple[str, ...]] = {
    "services": PROTOCOLS + AGENT_FRAMEWORKS,
    "storage": PROTOCOLS + AGENT_FRAMEWORKS,
    "security": (*PROTOCOLS, *AGENT_FRAMEWORKS, "food_concierge.services"),
    "mcp_server": (*WEB, "food_concierge.agent"),
    "agent": ("food_concierge.services", "food_concierge.storage"),
}


def _is_type_checking(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def runtime_imports(source: str) -> set[str]:
    """Every module a file imports when it runs, leaving out ``if TYPE_CHECKING:`` blocks."""
    found: set[str] = set()

    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.If):
                if not _is_type_checking(node):
                    visit(node.body)
                visit(node.orelse)
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    found.update(alias.name for alias in child.names)
                elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                    found.add(child.module)

    visit(ast.parse(source).body)
    return found


def _breaks(module: str, forbidden: tuple[str, ...]) -> bool:
    return any(module == name or module.startswith(name + ".") for name in forbidden)


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_each_layer_imports_only_what_it_may(layer: str) -> None:
    broken = {
        f"{path.relative_to(PACKAGE).as_posix()}: {module}"
        for path in sorted((PACKAGE / layer).rglob("*.py"))
        for module in runtime_imports(path.read_text(encoding="utf-8"))
        if _breaks(module, FORBIDDEN[layer])
    }
    assert not broken, sorted(broken)


def test_the_services_layer_exists_and_is_checked() -> None:
    assert len(list((PACKAGE / "services").glob("*.py"))) >= 6


def test_type_checking_imports_are_left_out_and_others_found() -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "import typing\n"
        "import fastapi\n"
        "if TYPE_CHECKING:\n    from mcp import Client\nelse:\n    import uvicorn\n"
        "if typing.TYPE_CHECKING:\n    import langgraph\n"
        "if True:\n    import starlette.types\n"
        "def f():\n    from langchain_core import messages\n"
        "from . import sibling\n"
    )
    assert runtime_imports(source) == {"typing", "fastapi", "uvicorn", "starlette.types", "langchain_core"}
    assert _breaks("mcp.server", ("mcp",))
    assert not _breaks("mcp_types", ("mcp",))
