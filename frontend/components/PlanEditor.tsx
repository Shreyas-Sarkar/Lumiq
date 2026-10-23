"use client";
import { useState } from "react";
import {
  AnalysisPlan,
  AnalysisStep,
  DatasetProfile,
} from "@/types/investigations";

export default function PlanEditor({
  plan,
  profile,
  busy,
  onSubmit,
}: {
  plan: AnalysisPlan;
  profile: DatasetProfile | null;
  busy: boolean;
  onSubmit: (plan: AnalysisPlan) => void;
}) {
  const [draft, setDraft] = useState<AnalysisPlan>(() =>
    JSON.parse(JSON.stringify(plan)),
  );
  const columns =
    profile?.columns.map((c) => c.name) ||
    Array.from(
      new Set(
        plan.steps.flatMap(
          (s) =>
            [s.metric, s.group_by, s.date_column].filter(Boolean) as string[],
        ),
      ),
    );
  const change = (index: number, patch: Partial<AnalysisStep>) =>
    setDraft((old) => ({
      ...old,
      steps: old.steps.map((s, i) => (i === index ? { ...s, ...patch } : s)),
    }));
  return (
    <form
      className="plan-editor"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit({ ...draft, clarification: null, concept_answer: null });
      }}
    >
      <h2>Revise the calculation</h2>
      <p className="small-muted">
        Your previous result stays saved. These settings define the next run.
      </p>
      {draft.steps.map((step, index) => (
        <fieldset key={index} className="step-editor">
          <legend>
            Step {index + 1} · {step.operation.replaceAll("_", " ")}
          </legend>
          <div className="plan-fields">
            <label>
              Metric
              <select
                value={step.metric}
                onChange={(e) => change(index, { metric: e.target.value })}
              >
                {columns.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
            <label>
              Calculation
              <select
                value={step.aggregation || "sum"}
                onChange={(e) => change(index, { aggregation: e.target.value })}
              >
                {(step.operation === "compare_periods"
                  ? ["sum", "count"]
                  : ["sum", "mean", "count", "min", "max", "median"]
                ).map((a) => (
                  <option key={a}>{a}</option>
                ))}
              </select>
            </label>
            {(step.operation === "group_by" ||
              step.operation === "compare_periods") && (
              <label>
                Group by
                <select
                  value={step.group_by || ""}
                  required={step.operation === "group_by"}
                  onChange={(e) =>
                    change(index, { group_by: e.target.value || null })
                  }
                >
                  <option value="">No grouping</option>
                  {columns.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </label>
            )}
            {step.operation === "compare_periods" && (
              <>
                <label>
                  Date column
                  <select
                    value={step.date_column || ""}
                    required
                    onChange={(e) =>
                      change(index, { date_column: e.target.value })
                    }
                  >
                    <option value="">Choose a column</option>
                    {columns.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                </label>
                {["Previous", "Current"].map((label, p) =>
                  [0, 1].map((bound) => (
                    <label key={`${p}-${bound}`}>
                      {label} {bound === 0 ? "start" : "end"}
                      <input
                        type="date"
                        required
                        value={step.periods?.[p]?.[bound] || ""}
                        onChange={(e) => {
                          const periods = step.periods?.map((v) => [...v]) || [
                            ["", ""],
                            ["", ""],
                          ];
                          periods[p][bound] = e.target.value;
                          change(index, { periods });
                        }}
                      />
                    </label>
                  )),
                )}
              </>
            )}
          </div>
          <div className="filter-editor">
            <h3>Filters</h3>
            {step.filters?.map((filter, filterIndex) => (
              <div className="filter-row" key={filterIndex}>
                <label>
                  Column
                  <select
                    value={filter.column}
                    onChange={(e) =>
                      change(index, {
                        filters: step.filters?.map((f, i) =>
                          i === filterIndex
                            ? { ...f, column: e.target.value }
                            : f,
                        ),
                      })
                    }
                  >
                    {columns.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Operator
                  <select
                    value={filter.operator}
                    onChange={(e) =>
                      change(index, {
                        filters: step.filters?.map((f, i) =>
                          i === filterIndex
                            ? { ...f, operator: e.target.value }
                            : f,
                        ),
                      })
                    }
                  >
                    {["eq", "ne", "gt", "gte", "lt", "lte", "in"].map((o) => (
                      <option key={o} value={o}>
                        {
                          {
                            eq: "equals",
                            ne: "does not equal",
                            gt: "greater than",
                            gte: "at least",
                            lt: "less than",
                            lte: "at most",
                            in: "one of",
                          }[o]
                        }
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Value{filter.operator === "in" ? " (comma separated)" : ""}
                  <input
                    required
                    value={
                      Array.isArray(filter.value)
                        ? filter.value.join(",")
                        : String(filter.value ?? "")
                    }
                    onChange={(e) => {
                      const numeric =
                        profile?.columns.find((c) => c.name === filter.column)
                          ?.kind === "number";
                      const cast = (v: string) =>
                        numeric && v.trim() !== "" ? Number(v) : v;
                      const value =
                        filter.operator === "in"
                          ? e.target.value.split(",").map((v) => cast(v.trim()))
                          : cast(e.target.value);
                      change(index, {
                        filters: step.filters?.map((f, i) =>
                          i === filterIndex ? { ...f, value } : f,
                        ),
                      });
                    }}
                  />
                </label>
                <button
                  type="button"
                  className="text-button"
                  onClick={() =>
                    change(index, {
                      filters: step.filters?.filter(
                        (_, i) => i !== filterIndex,
                      ),
                    })
                  }
                >
                  Remove
                </button>
              </div>
            ))}
            <button
              type="button"
              className="text-button"
              disabled={(step.filters?.length || 0) >= 8}
              onClick={() =>
                change(index, {
                  filters: [
                    ...(step.filters || []),
                    { column: columns[0], operator: "eq", value: "" },
                  ],
                })
              }
            >
              Add filter
            </button>
          </div>
        </fieldset>
      ))}
      <button className="primary-button" disabled={busy}>
        Run edited plan
      </button>
    </form>
  );
}
