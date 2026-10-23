# Lumiq

Lumiq answers questions about uploaded tabular data and exposes the calculations
supporting its answers.

## Language

**Dataset**: The tabular data supplied by the user for analysis.
_Avoid_: Knowledge base, document corpus.

**Chat**: A saved conversation associated with a user's questions and a dataset.

**Query**: A user's question or request for analysis.

**Computed result**: The output of a calculation against the dataset. A correct
calculation can still fail to answer the intended query.

**Insight**: An interpretation of computed evidence, distinct from the measured
result and potentially requiring qualification.

**Concept explanation**: An explanation of an analytical concept that does not
claim a new measurement from the dataset.

**Follow-up**: A suggested next question that extends an analysis.

**Investigation**: An analytical question with a plan, evidence, and conclusions
that can be revisited.

**Evidence**: The measurements and contextual facts supporting a conclusion.

**Analysis plan**: The operations and assumptions selected to answer a query.

**Finding**: A conclusion linked to supporting evidence and its limitations.

**Workspace**: The user's place for one dataset and its investigations, including
any earlier conversation.

**Investigation run**: One attempt to answer an investigation using a specific
plan and dataset version. A revised plan produces a separate run.

**Contribution**: A category's measured share of an additive change between
periods. It does not identify the business cause of that change.
