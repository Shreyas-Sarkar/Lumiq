# Investigation workspace design

Primary task: ask a question, inspect the plan, check computed evidence, then save
or revise the investigation. Existing dark-neutral tokens remain the base. A single
muted teal accent marks interactive/selected analytical content. No confidence badges.

Desktop: workspace/data navigation, investigation canvas, collapsible evidence.
Narrow screens: navigation above the canvas, tables scroll inside their region,
evidence follows results. Use native buttons, labels, details, dialog, semantic
tables, visible focus, and reduced motion. Status text comes from stored run state.

Required states: no dataset, upload rejection, profile ready, empty investigations,
planning/computing/interpreting, clarification, quota wait, partial evidence,
failed backend/session, saved completed results, legacy read-only history.

UI skills applied: anti-ui-slop (operate playbook), emil-design-eng. No paid
reference service or UI asset dependency is required.

## Homepage

Reading: product launch page for people who need inspectable answers. The previous
homepage communicated the workflow but its small, evenly weighted split layout
lacked a memorable hierarchy. Keep the existing wordmark and routes; extend the
product palette with a deep green canvas and restrained mint accent.

Design variance 7, motion intensity 5, visual density 3. Native CSS and existing
Lucide icons; self-hosted Manrope display/body typography. The hero prioritizes
one message and workspace entry. Its real interactive sample switches between
period totals, regional contributions, and sparse coverage. All values match the
checked-in synthetic CSV; no fake customer logos, performance claims, or live
inference indicators. This native data visual is the product demonstration, so
raster marketing imagery would add no useful evidence.

Entry transitions establish reading order, bar transitions explain the comparison,
and pressed controls expose sample state. Reduced motion removes those animations.
A single-column mobile layout retains the same reading order. Sample controls have
44 px touch targets, keyboard operation, visible focus, and announced results.
The sample download is byte-checked against the source fixture in a browser test.

Skills: design-taste-frontend and anti-ui-slop (new-work playbook). Verified by
browser inspection, responsive journeys, and the checks in `verification.md`.
