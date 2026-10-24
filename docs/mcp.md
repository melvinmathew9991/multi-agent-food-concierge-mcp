# Using the MCP server

The food concierge is an MCP server: Claude Desktop, MCP Inspector or any MCP client can search the catalog, check allergens and total a meal with it. This page covers connecting a client, what the server offers and what to do when a connection fails. How it is built is in [architecture.md](planning/architecture.md) §8 and §16.

## Before you connect

1. Install the project and build the catalog and indexes, as in the [README](../README.md#quick-start-development): `python -m food_concierge.flows build`.
2. Put provider keys (`GROQ_API_KEY`, optionally `GEMINI_API_KEY`) in the repository's `.env`. The server reads that file wherever it is started from, so keys never go into a client's config. Without a vision key every tool still works except `describe_image`, which then asks for the dish in words.
3. Start the server once from a terminal, so the embedding model is downloaded before a client is waiting for it:

   ```bash
   python -m food_concierge.mcp_server   # ready after a few seconds; Ctrl+C to stop
   ```

   On stdio, stdout carries the protocol and logs go to stderr, so seeing log lines and nothing else is normal.

## Claude Desktop

1. Open Claude Desktop, then **Settings → Developer → Edit Config**. That opens `claude_desktop_config.json` (on Windows `%APPDATA%\Claude\`, on macOS `~/Library/Application Support/Claude/`).
2. Add the server from [`examples/claude_desktop_config.json`](../examples/claude_desktop_config.json), with `command` set to the virtual environment's Python as a full path (`.venv/bin/python` on macOS and Linux):

   ```json
   {
     "mcpServers": {
       "food-concierge": {
         "command": "C:\\path\\to\\multi-agent-food-concierge-mcp\\.venv\\Scripts\\python.exe",
         "args": ["-m", "food_concierge.mcp_server"]
       }
     }
   }
   ```

   Keep any other servers already in `mcpServers`. The venv's own Python finds the installed package and the repository's `.env` and data, so no working directory or `env` block is needed.
3. Quit Claude Desktop fully (from the tray or menu bar, not only the window) and start it again. The tools appear under the tools icon in the chat box.
4. Try: *"Find me a vegan South Indian breakfast under ₹200 with no peanuts, and total two of them."* Claude should call `search_dishes` with `diet`, `cuisines`, `max_price_inr` and `exclude_allergens` as arguments, then `meal_totals`.

Over stdio the server runs as your own process, so you have every scope, `describe_image` included.

## MCP Inspector

Inspector (Node 18+) shows every tool's schema and lets you call tools, read resources and fill prompts by hand.

```bash
# stdio: Inspector starts the server itself
npx @modelcontextprotocol/inspector .venv/Scripts/python.exe -m food_concierge.mcp_server

# HTTP: start the API, then connect Inspector to it
python -m food_concierge.api
npx @modelcontextprotocol/inspector   # Transport: Streamable HTTP, URL: http://127.0.0.1:8000/mcp
```

Over HTTP a request without a token gets the public scope. To call `describe_image`, add the header `Authorization: Bearer <API_TOKEN_AGENT>` in Inspector's authentication settings.

A quick check covers: six tools, two resources, two resource templates and two prompts listed; `search_dishes` with `{"query": "dosa", "diet": "vegan", "exclude_allergens": ["peanuts"], "k": 3}` returns three dishes; `get_dish` with `{"dish_id": "nope"}` returns an error result naming the id; over HTTP without a token, `describe_image` is refused for needing the `agent` scope.

## What the server offers

### Tools

All six are read-only. Only `describe_image` reaches outside the server (a hosted vision model), so it is marked open-world and needs the `agent` scope over HTTP.

| Tool | Arguments | Returns |
|---|---|---|
| `search_dishes` | `query`; optional `diet` (`vegan`, `vegetarian`, `non_vegetarian`), `eggless`, `exclude_allergens` (plain words: "dairy", "nuts"), `max_kcal`, `max_price_inr`, `cuisines`, `include_unverified`, `exclude_ids`, `k` (1–10, default 5) | Ranked dishes that keep every constraint |
| `find_similar` | `dish_id` and the same constraints as search | Dishes like it, not the same dish at another restaurant |
| `get_dish` | `dish_id` | Ingredients, nutrition, price, restaurant, allergen tags with evidence, photo credit |
| `check_allergens` | `dish_ids` (1–20), optional `allergens` (all if left out) | Per dish: `contains`, `may_contain` (photo only), `unverified`, `clear`; and a disclaimer |
| `meal_totals` | `dish_ids` (1–20; list a dish twice to count it twice) | kcal, protein, carbs, fat, rupees, servings |
| `describe_image` | `image_base64` (at most `MAX_IMAGE_MB`) | A name-free description to search with, or a request for words |

Search filters in SQL before ranking and re-checks every result, so it never returns a dish that breaks the constraints given. A dish whose recipe can't be checked is *unverified*: it is left out whenever allergens matter, until the diner agrees and the client searches again with `include_unverified`.

### Resources and prompts

| Resource | Contents |
|---|---|
| `menu://catalog` | Every dish in short form, prices in INR |
| `menu://dishes/{dish_id}` | One dish in full |
| `menu://dishes/{dish_id}/image` | Its photo (WebP), when the thumbnail is cached on the server |
| `menu://policy/allergens` | The allergens, the words each answers to, what `contains` / `may_contain` / `unverified` mean |

| Prompt | Argument | Asks the model to |
|---|---|---|
| `plan_meal` | `constraints` | Search with the constraints as arguments, check the meal with `meal_totals` and `check_allergens`, end with the disclaimer |
| `find_similar_dish` | `description` | Search on a described or photographed dish and say why each result is similar |

The exact schemas clients see are pinned in [`tests/fixtures/mcp_schema.json`](../tests/fixtures/mcp_schema.json); a change to them fails the tests until the snapshot is regenerated (`UPDATE_MCP_SCHEMA=1 pytest tests/unit/test_mcp_server.py -k snapshot`) and reviewed.

### Errors

A domain error (an unknown dish id, an unknown allergen word, a photo too large) comes back as a tool result with `isError` and a message meant for the model to act on, such as "Error executing tool get_dish: No dish has the id nope." Anything unexpected comes back only as "Error executing tool <name>", with the detail in the server's log. Over HTTP, refused requests share one shape, `{"error": {"code", "message", "request_id"}}`: 401 for a token that matches none, 429 with `Retry-After` past a rate limit.

Allergen information is derived from recipes, labels and photos and comes with a disclaimer; it is not medical advice.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Claude Desktop shows no tools | Config not reloaded: quit Claude Desktop fully and start it again. Check the JSON is valid and `command` is a full path that exists. |
| "Server disconnected" in Claude Desktop | Read its log, `mcp-server-food-concierge.log` (Windows `%APPDATA%\Claude\logs\`, macOS `~/Library/Logs/Claude/`). A `NotReadyError` means the catalog or indexes aren't built: run `python -m food_concierge.flows build`. |
| The first start times out | The embedding model is downloading. Start the server once from a terminal and connect again. |
| The server won't start: a token setting is refused | An `API_TOKEN_*` in `.env` is shorter than 32 characters or equals another one. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. |
| `describe_image` asks for words | No vision provider is configured, the daily photo budget is spent, or the provider is down. Describe the dish instead, or add `GROQ_API_KEY`. |
| HTTP: 421 or "Invalid Host header" | MCP answers only `ALLOWED_HOSTS` (localhost by default). Add the host you connect through. |
| HTTP: `describe_image` refused for scope | Send `Authorization: Bearer <API_TOKEN_AGENT>`; a request without a token is public. |
