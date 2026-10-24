"use client";

import { useState } from "react";
import { ArrowRight, Check, GitBranch, Sigma, ScanText } from "lucide-react";

const stages = [
  { name: "Planner", caption: "Question → calculation plan", icon: GitBranch },
  {
    name: "Analytical tools",
    caption: "Plan → measured evidence",
    icon: Sigma,
  },
  { name: "Analyst", caption: "Evidence → interpretation", icon: ScanText },
];

export default function InvestigationTrace() {
  const [selected, setSelected] = useState(0);
  return (
    <div className="investigation-trace" data-reveal>
      <div className="trace-label">
        <span>Inside the sample investigation</span>
        <span>
          Click a step to inspect <ArrowRight size={13} />
        </span>
      </div>
      <ol
        className="engine-flow"
        aria-label="Explore the sample investigation workflow"
      >
        {stages.map(({ name, caption, icon: Icon }, index) => (
          <li key={name} className={selected === index ? "trace-selected" : ""}>
            <button
              type="button"
              aria-pressed={selected === index}
              aria-controls="trace-detail"
              onClick={() => setSelected(index)}
            >
              <Icon size={19} strokeWidth={1.5} aria-hidden="true" />
              <span>{name}</span>
              <small>{caption}</small>
            </button>
          </li>
        ))}
      </ol>
      <div
        id="trace-detail"
        className="trace-detail"
        aria-live="polite"
        aria-atomic="true"
      >
        <div key={selected} className="trace-detail-content">
          <div className="trace-detail-heading">
            <span>{stages[selected].name} output</span>
            <span>
              <Check size={13} /> Sample data
            </span>
          </div>
          {selected === 0 ? (
            <>
              <p>Compare the recorded revenue across two months.</p>
              <dl className="trace-facts">
                <div>
                  <dt>Metric</dt>
                  <dd>revenue</dd>
                </div>
                <div>
                  <dt>Calculation</dt>
                  <dd>Sum</dd>
                </div>
                <div>
                  <dt>Periods</dt>
                  <dd>January / February 2026</dd>
                </div>
              </dl>
            </>
          ) : selected === 1 ? (
            <>
              <p>Calculate first. Keep the evidence.</p>
              <dl className="trace-facts">
                <div>
                  <dt>January</dt>
                  <dd>300</dd>
                </div>
                <div>
                  <dt>February</dt>
                  <dd>340</dd>
                </div>
                <div>
                  <dt>Measured change</dt>
                  <dd>+40 · +13.3%</dd>
                </div>
              </dl>
            </>
          ) : (
            <>
              <p>Revenue increased in the supplied rows.</p>
              <dl className="trace-facts">
                <div>
                  <dt>Evidence</dt>
                  <dd>Period comparison</dd>
                </div>
                <div>
                  <dt>Coverage</dt>
                  <dd>Six observations</dd>
                </div>
                <div>
                  <dt>Limitation</dt>
                  <dd>Full-month coverage is unverified</dd>
                </div>
              </dl>
            </>
          )}
        </div>
      </div>
      <p className="trace-footnote">
        An illustrative trace of revenue.csv. Exploring it makes no model calls.
      </p>
    </div>
  );
}
