"use client";

import { useState } from "react";
import { ArrowUpRight, Check, FileSpreadsheet } from "lucide-react";

const views = ["The change", "The drivers", "The caveat"] as const;
type View = (typeof views)[number];

export default function SampleInvestigation() {
  const [view, setView] = useState<View>("The change");
  const index = views.indexOf(view);
  return (
    <section
      className="sample-investigation"
      aria-label="Interactive sample investigation"
    >
      <div className="sample-source">
        <FileSpreadsheet size={15} />
        <span>revenue.csv</span>
        <span>6 rows · Sample data</span>
      </div>
      <div className="sample-question">What changed in February?</div>
      <div
        className="sample-controls"
        role="group"
        aria-label="Explore the sample"
      >
        {views.map((item) => (
          <button
            key={item}
            type="button"
            aria-pressed={view === item}
            onClick={() => setView(item)}
          >
            {item}
          </button>
        ))}
      </div>
      <div className="sample-result" aria-live="polite" aria-atomic="true">
        <div key={view} className="sample-result-content">
          <div className="sample-measurement">
            <span>
              {index === 0
                ? "Revenue change"
                : index === 1
                  ? "North’s contribution"
                  : "Coverage matters"}
            </span>
            <p>
              {index === 0 ? "+40" : index === 1 ? "+60" : "6 rows"}
              <small>
                {index === 0
                  ? "+13.3%"
                  : index === 1
                    ? "of +40 net"
                    : "two months"}
              </small>
            </p>
          </div>
          {index === 0 ? (
            <figure className="sample-graphic">
              <svg
                viewBox="0 0 440 192"
                role="img"
                aria-label="January revenue 300. February revenue 340. An increase of 40."
              >
                <line
                  x1="18"
                  x2="422"
                  y1="158"
                  y2="158"
                  className="sample-axis"
                />
                <line
                  x1="18"
                  x2="422"
                  y1="81"
                  y2="81"
                  className="sample-gridline"
                />
                <g className="sample-column">
                  <rect
                    x="72"
                    y="47"
                    width="108"
                    height="111"
                    className="sample-bar-muted"
                  />
                  <text x="126" y="36" textAnchor="middle">
                    300
                  </text>
                </g>
                <g className="sample-column sample-column-later">
                  <rect
                    x="260"
                    y="32"
                    width="108"
                    height="126"
                    className="sample-bar"
                  />
                  <text x="314" y="21" textAnchor="middle">
                    340
                  </text>
                </g>
                <text
                  x="126"
                  y="184"
                  textAnchor="middle"
                  className="sample-axis-label"
                >
                  January
                </text>
                <text
                  x="314"
                  y="184"
                  textAnchor="middle"
                  className="sample-axis-label"
                >
                  February
                </text>
              </svg>
              <figcaption>Sum of revenue · January vs February 2026</figcaption>
            </figure>
          ) : index === 1 ? (
            <figure className="sample-graphic">
              <svg
                viewBox="0 0 440 192"
                role="img"
                aria-label="North adds 60. South subtracts 20. Net revenue change is plus 40."
              >
                <line
                  x1="18"
                  x2="422"
                  y1="158"
                  y2="158"
                  className="sample-axis"
                />
                <line
                  x1="18"
                  x2="422"
                  y1="81"
                  y2="81"
                  className="sample-gridline"
                />
                <path
                  d="M100 58H136 M208 38H244 M316 45H352"
                  className="sample-connector"
                />
                <g className="sample-column">
                  <rect
                    x="28"
                    y="58"
                    width="72"
                    height="100"
                    className="sample-bar-muted"
                  />
                  <text x="64" y="47" textAnchor="middle">
                    300
                  </text>
                </g>
                <g className="sample-column">
                  <rect
                    x="136"
                    y="38"
                    width="72"
                    height="20"
                    className="sample-bar"
                  />
                  <text x="172" y="27" textAnchor="middle">
                    +60
                  </text>
                </g>
                <g className="sample-column">
                  <rect
                    x="244"
                    y="38"
                    width="72"
                    height="7"
                    className="sample-bar-negative"
                  />
                  <text x="280" y="27" textAnchor="middle">
                    −20
                  </text>
                </g>
                <g className="sample-column sample-column-later">
                  <rect
                    x="352"
                    y="45"
                    width="72"
                    height="113"
                    className="sample-bar"
                  />
                  <text x="388" y="34" textAnchor="middle">
                    340
                  </text>
                </g>
                {["January", "North", "South", "February"].map((label, i) => (
                  <text
                    key={label}
                    x={64 + i * 108}
                    y="184"
                    textAnchor="middle"
                    className="sample-axis-label"
                  >
                    {label}
                  </text>
                ))}
              </svg>
              <figcaption>
                Contributions measure the change, not its business causes.
              </figcaption>
            </figure>
          ) : (
            <div className="sample-coverage">
              <div>
                <span>January 2026</span>
                <strong>3 observations</strong>
                <p>Jan 5 · Jan 15 · Jan 31</p>
              </div>
              <div>
                <span>February 2026</span>
                <strong>3 observations</strong>
                <p>Feb 5 · Feb 15 · Feb 28</p>
              </div>
              <p className="sample-coverage-note">
                A few recorded dates cannot establish complete monthly coverage.
              </p>
            </div>
          )}
          <div className="sample-reading">
            <Check size={18} aria-hidden="true" />
            <p>
              {index === 0
                ? "Revenue rose from 300 to 340 in the supplied rows."
                : index === 1
                  ? "North’s increase offsets South’s decline of 20."
                  : "Keep the result. Make the limitation visible."}
            </p>
          </div>
        </div>
      </div>
      <a className="sample-download" href="/samples/revenue.csv" download>
        Check it yourself. Download the CSV <ArrowUpRight size={14} />
      </a>
    </section>
  );
}
