# 0002. MCP-first layering: services behind MCP tools

- Status: accepted
- Date: 2026-10-24
- Requirements: PRD §6 (the MCP server usable by the agent and by external MCP clients; F8 constraints enforced in tools), §7 (safety: zero constraint violations; zero cost)

## Context

Phase 4 built the domain services (catalog, allergens, nutrition, constraints, search, vision), an MCP server over them (#111) and an HTTP app that serves MCP and a retrieval-only `/v1/search` (#113). Phase 5 adds the LangGraph agents. The question is how the agents reach the services, and what each layer may depend on.

Two requirements pull on it. External MCP clients (Claude Desktop, Inspector, other agents) must get the same tools, with the same safety checks, as our own agent. And allergen and diet constraints must hold however a tool is reached: an LLM can't be the safety layer, since in the original data 20 of 50 sampled items had empty allergen labels and 14 of 50 image descriptions mentioned allergen-relevant ingredients missing from the metadata.

**Evidence from Phase 4.**
- Search filters in SQL, then re-checks every result against the constraints. A property test runs 25 random constraint sets through an in-memory MCP session and finds no violating dish; another shows, over 100 random sets and all 150 dishes, that the constraint check passes a dish exactly when search would return it. `/v1/search` and the MCP tool share one `SearchService`, so they share that check.
- Tests reach the server through `mcp.Client(server)` in-process (no network), and the stdio and HTTP transports are smoke-tested from outside the process (#118, #122). The schemas clients see are pinned in `tests/fixtures/mcp_schema.json`.
- Cost of the protocol: 200 constrained searches on the real catalog and embedder (owner's laptop, 2026-10-24) took p50 7.8 ms called directly and p50 9.7 ms through an in-process MCP client, about 2 ms more per call. One agent turn is expected to make at most a few tool calls next to LLM calls of about a second, so the protocol costs well under 1% of a turn.

## Options

1. **The agent imports the services directly**, wrapped as LangChain tools; the MCP server is a separate adapter for outside clients. Fastest and simplest for the agent. But there are two tool surfaces to keep in step, and the one external clients use isn't the one we test hardest: a check added to one can be missed in the other.
2. **MCP-first:** services (plain Python, no protocol or agent framework) ← MCP server (the one tool surface: schemas, scopes, errors) ← agent, as an MCP client (`langchain-mcp-adapters` over the in-process or HTTP transport). The HTTP app is a second adapter over the same services, for retrieval-only search and health checks, never a second tool implementation.
3. **Tools defined in the agent framework**, with MCP generated from them. Ties the domain to LangChain's tool types, and puts framework imports in the layer that should be easiest to test.

## Decision

Option 2. Every tool the agent uses is the tool external clients use, with the same input bounds, constraint checks, scope checks and error shapes, so the safety properties are tested once and hold for both. The roughly 2 ms per call is negligible next to the model calls.

The rule is enforced by `tests/unit/test_layering.py`, which reads the imports of each package (type-only imports under `TYPE_CHECKING` excepted):

| Layer | May not import |
|---|---|
| `services/`, `storage/` | MCP, FastAPI/Starlette/uvicorn, LangChain/LangGraph, `mcp_server`, `api`, `agent` |
| `security/` | the same, and `services` |
| `mcp_server/` | FastAPI/Starlette/uvicorn, `api`, `agent` |
| `agent/` (Phase 5) | `services`, `storage`: tools only through MCP |

Two dependencies are deliberate: `services/vision.py` calls the model layer (`models.router`) to describe a photo, since describing is the service, and `api/` imports the services to serve `/v1/search` and `/ready`.

## Consequences

- One tool surface: a new tool, a new constraint or a scope change is made once, in the MCP server, and is covered by the schema snapshot.
- The agent depends on the protocol's behaviour (tool results as JSON, `isError` for domain errors), so agent tests run against the real server in-process, not against mocked tools.
- Agents run in the same process as the server on the Space; splitting them across processes later means switching the client transport to HTTP, with no change to the tools.
- Revisit if tool-call overhead becomes a measurable share of a turn (more than about 5% in Phase 6's measurements), or if an agent needs something MCP can't carry, such as streaming partial results from a tool.
