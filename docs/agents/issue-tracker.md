# Issue tracker: local Markdown

Chosen by the owner on 2026-10-02. GitHub is the code remote, not this project's
configured issue tracker.

- Feature directory: `.scratch/<feature-slug>/`.
- Specification: `.scratch/<feature-slug>/spec.md`.
- One ticket per `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from
  `01`, with goal, acceptance criteria, scope, and verification.
- Record `Status: open`, `claimed`, `blocked`, or `resolved` near the top.
- A synthesized spec can record `Label: ready-for-agent`; this means ready for
  implementation planning, not authorization to implement or deploy.
- Record dependencies with `Blocked by: <ticket paths>`.
- Append conversation under `## Comments`; resolution under `## Outcome` with
  verification evidence and `Status: resolved`.

“Publish to the issue tracker” means write the local spec/ticket. “Fetch the
relevant ticket” means read the referenced file. Create files when requested work
needs them. No triage skill or GitHub label workflow is installed/configured.
