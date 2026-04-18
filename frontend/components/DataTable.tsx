"use client";

interface DataTableResult {
  columns: string[];
  rows: unknown[][];
  shape: [number, number];
  truncated?: boolean;
}

interface DataTableProps {
  result: Record<string, unknown> | unknown[] | string | number | null;
  resultType: string;
}

export default function DataTable({ result, resultType }: DataTableProps) {
  // ── DataFrame ──────────────────────────────────────────────────────────────
  if (resultType === "dataframe" && result && typeof result === "object") {
    const data = result as unknown as DataTableResult;
    const { columns = [], rows = [], shape, truncated } = data;
    const totalRows = shape ? shape[0] : rows.length;

    return (
      <div>
        <div
          className="overflow-x-auto rounded-lg"
          style={{ border: "1px solid var(--border)" }}
        >
          <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "var(--surface-3)" }}>
                {columns.map((col, i) => (
                  <th
                    key={i}
                    className="px-3 py-2.5 text-left font-medium whitespace-nowrap"
                    style={{
                      color: "var(--muted)",
                      borderBottom: "1px solid var(--border)",
                      letterSpacing: "0.02em",
                    }}
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, ri) => (
                <tr
                  key={ri}
                  style={{
                    background: ri % 2 === 0 ? "var(--surface)" : "var(--surface-2)",
                    borderBottom: "1px solid var(--border)",
                  }}
                >
                  {(row as unknown[]).map((cell, ci) => (
                    <td
                      key={ci}
                      className="px-3 py-2 whitespace-nowrap font-mono"
                      style={{ color: "var(--text)" }}
                    >
                      {cell === null || cell === undefined ? (
                        <span style={{ color: "var(--muted)" }}>—</span>
                      ) : (
                        String(cell)
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {truncated && (
          <p className="text-[10px] mt-2" style={{ color: "var(--muted)" }}>
            Showing {rows.length} of {totalRows} rows
          </p>
        )}
      </div>
    );
  }

  // ── Scalar ─────────────────────────────────────────────────────────────────
  if (resultType === "scalar" && result !== null && result !== undefined) {
    // Scalar is rendered directly in AnalysisBlock for prominence — skip here
    return null;
  }

  // ── List ───────────────────────────────────────────────────────────────────
  if (resultType === "list" && Array.isArray(result)) {
    return (
      <div
        className="rounded-lg overflow-hidden"
        style={{ border: "1px solid var(--border)" }}
      >
        {(result as unknown[]).map((item, i) => (
          <div
            key={i}
            className="flex items-center gap-3 px-4 py-2.5"
            style={{
              background: i % 2 === 0 ? "var(--surface)" : "var(--surface-2)",
              borderBottom:
                i < result.length - 1 ? "1px solid var(--border)" : "none",
            }}
          >
            <span
              className="text-[10px] font-mono w-5 text-right flex-shrink-0"
              style={{ color: "var(--muted)" }}
            >
              {i + 1}
            </span>
            <span className="text-xs font-mono" style={{ color: "var(--text)" }}>
              {String(item)}
            </span>
          </div>
        ))}
      </div>
    );
  }

  return null;
}
