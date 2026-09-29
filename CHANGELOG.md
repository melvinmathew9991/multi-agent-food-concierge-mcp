# Changelog

Notable changes, grouped by delivery phase (`docs/planning/phases.md`). Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/) and are tagged at the milestones listed in the plan.

## [Unreleased]

### Phase 0: Foundation

#### Added
- Package layout (`food_concierge`), settings, application error hierarchy and key=value logging.
- CI on push and pull request: ruff lint and format, strict mypy, offline tests, pip-audit, gitleaks secret scan, and a 5 MB tracked-file limit. A weekly scheduled run scans the full history.
- Planning documents: PRD, architecture, design, engineering rules and delivery phases (revision 3), plus the baseline defect log and an audit reference index.
- Ignore rules for credentials, user data, local service state and material that is not redistributed.
- This changelog and the ADR index (`docs/adr/`).

#### Changed
- Settings follow `.env.example`: Groq by default with Gemini as fallback, one API token per scope, and Langfuse keys. A test keeps the two in sync.
- Paid providers (OpenAI, Bedrock) are refused at startup unless `ALLOW_PAID_PROVIDERS=true`.
- Unknown keys in `.env` are rejected, and blank values keep their defaults.
- The package version has a single source (`food_concierge.__version__`).

#### Security
- Log field values are quoted and escaped, so user text can't inject extra log lines or fields.
- The third-party CI action is pinned to a commit.
- `main` is protected by a repository ruleset: no force-push or deletion, and changes go through pull requests with green checks.
