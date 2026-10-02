# ADR 0001: Project-local AI development context

Status: accepted
Date: 2026-10-02

## Context

The owner requested AI-ready repo setup with Matt Pocock skills and selected
local Markdown tickets, a root AGENTS.md/glossary, and project-wide ADRs.
Historical docs diverge from current behavior.

## Decision

Use a short `AGENTS.md` with context pointers; vendor selected engineering
skills under `.agents/skills/` at a pinned revision with license/provenance.
Store specs/tickets under `.scratch/`, language in `GLOSSARY.md`, and accepted
consequential choices under `docs/adr/`.

## Alternatives and consequences

Global-only skills do not travel with the repo. The full collection adds
unrelated workflows. GitHub Issues were an alternative the owner did not choose.
Separate frontend/backend glossaries would duplicate one product's language.
Local copies make workflows inspectable but require deliberate updates.
This decision does not select an agent runtime or authorize deployment.
