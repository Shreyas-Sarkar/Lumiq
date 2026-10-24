# Investigation workspace design

Primary task: ask a question, inspect the plan, check computed evidence, then save
or revise the investigation. Existing dark-neutral tokens remain the base. A single
cool accent marks links and selected analytical content. Off-white actions and
neutral surfaces share tokens with authentication and the homepage. No confidence badges.

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
product palette with a near-black canvas, off-white actions, gray borders, and a
restrained cool selected-state accent. The owner requested a closer Linear direction
after reviewing the green palette.

Design variance 6, motion intensity 4, visual density 3. Native CSS and existing
Lucide icons; self-hosted Inter throughout, with its OFL license retained
from the official Inter project (https://github.com/rsms/inter). The hero prioritizes
one message and workspace entry. Its real interactive sample switches between
period totals, regional contributions, and sparse coverage. All values match the
checked-in synthetic CSV; no fake customer logos, performance claims, or live
inference indicators. This native data visual is the product demonstration, so
raster marketing imagery would add no useful evidence.

Layered sample sheets add restrained depth. Short staggered entrance transitions
establish reading order, viewport reveals introduce sections once, and bar transitions
explain the comparison. The sample trace exposes planner/tool/analyst outputs without
spending inference quota. Animation observers and media listeners clean up on unmount,
and ordinary server-rendered content stays visible without animation JavaScript.

Pressed controls expose sample state. Reduced motion removes those animations.
A single-column mobile layout retains the same reading order. Sample controls have
44 px touch targets, keyboard operation, visible focus, and announced results.
The sample download is byte-checked against the source fixture in a browser test.

Skills: design-taste-frontend and anti-ui-slop (new-work playbook). Verified by
browser inspection, responsive journeys, and the checks in `verification.md`.

## Brand mark

The boxed initial is replaced by an original geometric evidence aperture: an open
center, two clipped corners, and an exit toward the lower right. It suggests
inspection and the final letter of Lumiq without a generic letter badge. One
`BrandMark` component supplies the mark across homepage, workspace, auth, and legacy
navigation; `app/icon.svg` carries the same geometry for browser tabs. It uses the
shared text color and remains legible without gradients or animation.
