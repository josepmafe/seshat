# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is an MVP — structure will be updated as components are defined. Expected layout:

```
src/          # Application source code
tests/        # pytest test suite
pyproject.toml
```
## Test Style Guide

- **Classes**: use a test class per target class, with one test method per method under test.
- **Functions**: use top-level test functions (one or more per target function).

## Code Style

- Add a blank line after closing `with` blocks before the next statement.
- Add a blank line after `if` / `try-except` blocks before the next statement, but not before the block itself.

## Notes

- Architecture and AI component decisions are documented in `docs/architecture.md`, `docs/seshat-sdd.md`, and the specs under `docs/superpowers/specs/`.
- `docs/claude-behavior.md` is a personal Claude config file, not project documentation — ignore it.
- `pyproject.toml` is the single source of truth for dependencies, tool config (ruff, pytest), and project metadata.
