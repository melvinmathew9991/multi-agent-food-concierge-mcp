# Architecture decision records

One file per decision: `NNNN-short-title.md`, numbered in order and never renumbered. A decision that changes is superseded by a new ADR, and the old one is marked, not edited away.

## Index

| ADR | Decision | Phase | Status |
|---|---|---|---|
| 0001 | Retrieval stack and vector backend, chosen from dev-split evidence | P3 | Planned |
| 0002 | MCP-first layering: services behind MCP tools | P4 | Planned |
| 0003 | LangGraph over CrewAI/AutoGen; supervisor design | P5 | Planned |
| 0004 | Default agent topology (single vs multi-agent), chosen from measurements | P6 | Planned |
| 0005 | Boundaries between MCP (agent-to-tool) and A2A (agent-to-agent) | P8 | Planned |
| [0006](0006-free-tier-models-and-fallback-order.md) | Free-tier models and fallback order, chosen from the Phase 1 model profile | P1 | Accepted |
| [0007](0007-vision-model-and-vision-allergen-tags.md) | Vision model for user photos, and what vision allergen tags may do, chosen from the Phase 2 measurement | P2 | Accepted |

## Template

```markdown
# NNNN. Title

- Status: proposed | accepted | superseded by NNNN
- Date: YYYY-MM-DD
- Requirements: PRD refs (F# in §6, categories in §7)

## Context
The problem, the constraints and the evidence available.

## Options
Each option considered, with its trade-offs.

## Decision
What was chosen and why, citing the evidence.

## Consequences
What becomes easier or harder, and what would make us revisit this decision.
```
