# Seshat

Seshat is a GenAI pipeline that turns meeting recordings into a structured knowledge base. It ingests audio or pre-formatted transcripts, extracts Architecture Decision Records, risks, agreements, and action items, and writes them to a queryable store that tracks relationships between decisions across meetings — supersessions, amendments, and conflicts.

Built as a master's thesis project.

## Documentation map

- `docs/primer.md` → Developer primer: narrative overview and end-to-end job walkthrough.
- `docs/architecture.md` → Architecture summary: key design decisions and rationale.
- `docs/seshat-sdd.md` → Solution Design Document: implementation-oriented system design.
- `docs/superpowers/specs/2026-04-21-seshat-design.md` → Full design spec and detailed contracts.
- `docs/superpowers/specs/2026-04-24-quality-gate-design.md` → Quality gate design spec (pre-commit + GHA).
- `docs/superpowers/plans/2026-04-24-quality-gate.md` → Quality gate implementation plan.
- `docs/superpowers/specs/2026-04-27-prompt-interaction-design.md` → Prompt and interaction design spec.
