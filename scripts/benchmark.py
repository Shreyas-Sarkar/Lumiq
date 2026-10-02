"""Offline known-answer/resource benchmark; no inference or external services."""

import sys, time, json, resource
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import pandas as pd
from models.investigations import AnalysisPlan
from services.analysis_tools import execute_plan

root = Path(__file__).resolve().parents[1]
plan = AnalysisPlan(
    steps=[
        {
            "operation": "compare_periods",
            "metric": "revenue",
            "date_column": "date",
            "group_by": "region",
            "periods": [["2026-01-01", "2026-01-31"], ["2026-02-01", "2026-02-28"]],
        }
    ]
)
for copies in [1, 16666]:
    df = pd.concat(
        [pd.read_csv(root / "samples/revenue.csv")] * copies, ignore_index=True
    )
    start = time.perf_counter()
    e = execute_plan(df, plan)[0]
    elapsed = time.perf_counter() - start
    assert (
        e["summary"]["previous"] == 300 * copies
        and e["summary"]["current"] == 340 * copies
    )
    assert e["summary"]["change"] == 40 * copies
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024 * 1024 if sys.platform == "darwin" else 1024
    )
    print(
        json.dumps(
            {
                "rows": len(df),
                "parsed_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                "peak_process_mb": round(peak, 2),
                "calculation_ms": round(elapsed * 1000, 2),
                "known_answer": "pass",
                "model_calls": 0,
            }
        )
    )
