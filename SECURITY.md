# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability**. Don't open a public issue, pull request or discussion for a security problem.

Include what you found, how to reproduce it, and what an attacker could do with it. You should get an answer within 7 days. Once a fix is merged, the advisory is published with credit to you, unless you'd rather stay anonymous.

## Supported versions

Only the latest commit on `main`, and the deployed demo built from it, receive fixes. The project is pre-1.0 and has no maintained release branches.

## Scope

In scope:

- The MCP server and the HTTP API (`src/food_concierge/`): token scopes, rate limits, input limits, image handling, and anything that leaks a provider key, a token or user text.
- Allergen safety: any way search or `check_allergens` reports a dish as free of an allergen it contains. The disclaimer in `services/allergens.py` still applies; this is a demo, not medical advice.
- The container image (`Dockerfile`) and the CI workflows (`.github/workflows/`).

Out of scope:

- The model providers (Groq, Google, AWS) and Langfuse; report those to them.
- Rate limiting by volume on the free public demo. Its limits are documented in `README.md`.
- Findings that need a stolen token or access to the host.

## What the project already does

Secrets are scanned by gitleaks on every pull request and weekly over the full history. Dependencies are hash-pinned and audited with `pip-audit` on every pull request and nightly. The image runs as a non-root user with read-only code and data. `docs/data-handling.md` describes what user data is kept, for how long, and where.
