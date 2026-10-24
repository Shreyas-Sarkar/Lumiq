"""Bounded deterministic calculations. Evidence numbers come exclusively from Pandas."""

import hashlib
import json
import re
import pandas as pd
import numpy as np
from models.investigations import AnalysisPlan

TOOL_VERSION = "1.1"
MAX_ROWS = 100_000
MAX_MEMORY = 100 * 1024 * 1024


def clean(value):
    return json.loads(
        json.dumps(
            value,
            default=lambda x: x.item() if hasattr(x, "item") else str(x),
            allow_nan=False,
        )
    )


def dates(series):
    # Explicit ISO calendar dates; no ambiguous locale interpretation.
    strings = series.astype("string")
    valid = strings.str.match(r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$", na=False)
    return pd.to_datetime(
        strings.where(valid), format="ISO8601", errors="coerce", utc=True
    )


def profile_dataset(df):
    if len(df) > MAX_ROWS or df.memory_usage(deep=True).sum() > MAX_MEMORY:
        raise ValueError(
            "Dataset exceeds the analysis budget (100,000 rows / 100 MB parsed memory)"
        )
    columns = []
    for name in df.columns:
        s = df[name]
        c = {
            "name": str(name),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique()),
            "kind": "number" if pd.api.types.is_numeric_dtype(s) else "category",
        }
        if c["kind"] == "category":
            parsed = dates(s)
            if parsed.notna().sum() and (
                parsed.notna().sum() / max(s.notna().sum(), 1) >= 0.5
            ):
                c.update(
                    kind="date",
                    start=parsed.min().isoformat(),
                    end=parsed.max().isoformat(),
                    invalid_dates=int((s.notna() & parsed.isna()).sum()),
                )
            else:
                c["examples"] = [str(x)[:80] for x in s.dropna().unique()[:5]]
        columns.append(c)
    return {
        "version": 1,
        "rows": len(df),
        "columns": columns,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "limitations": [
            "Duplicate rows are retained.",
            "Dates use ISO format and UTC; observed coverage does not prove complete collection.",
        ],
    }


def _column(df, name, numeric=False):
    if not name or name not in df.columns:
        raise ValueError(f"Unknown column: {name}")
    if numeric and not pd.api.types.is_numeric_dtype(df[name]):
        raise ValueError(f"{name} must be numeric")
    return df[name]


def _aggregate(s, operation):
    if operation == "count":
        return int(s.count())
    v = (
        getattr(s, operation)(min_count=1)
        if operation == "sum"
        else getattr(s, operation)()
    )
    return None if pd.isna(v) or not np.isfinite(v) else float(v)


def _grouped_aggregate(data, step):
    groups = data.groupby(step.group_by, dropna=False)[step.metric]
    return (
        groups.sum(min_count=1)
        if step.aggregation == "sum"
        else groups.agg(step.aggregation)
    )


def execute_plan(df, plan: AnalysisPlan):
    profile = profile_dataset(df)
    evidence = []
    for index, step in enumerate(plan.steps):
        data = df.copy(deep=False)
        limitations = list(profile["limitations"])
        for f in step.filters:
            col = _column(data, f.column)
            if f.operator == "in":
                if not isinstance(f.value, list) or len(f.value) > 50:
                    raise ValueError("in filter requires at most 50 values")
                mask = col.isin(f.value)
            else:
                methods = {
                    "eq": "eq",
                    "ne": "ne",
                    "gt": "gt",
                    "gte": "ge",
                    "lt": "lt",
                    "lte": "le",
                }
                try:
                    mask = getattr(col, methods[f.operator])(f.value)
                except (TypeError, ValueError):
                    raise ValueError(f"Filter type incompatible with {f.column}")
            data = data.loc[mask.fillna(False)]
        numeric = step.aggregation != "count"
        metric = _column(data, step.metric, numeric=numeric)
        if metric.isna().any():
            limitations.append(
                f"{int(metric.isna().sum())} missing metric values excluded."
            )
        summary, rows, chart = {}, [], None
        if step.operation == "aggregate":
            summary = {"value": _aggregate(metric, step.aggregation)}
            rows = [{"metric": step.metric, "aggregation": step.aggregation, **summary}]
        elif step.operation == "group_by":
            _column(data, step.group_by)
            if data[step.group_by].nunique(dropna=False) > 500:
                raise ValueError("Too many groups (max 500); narrow the filters")
            groups = _grouped_aggregate(data, step).reset_index()
            groups = groups.sort_values(step.metric, ascending=not step.sort_descending)
            rows = json.loads(groups.head(step.limit).to_json(orient="records"))
            if len(groups) > step.limit:
                limitations.append(
                    f"Showing {step.limit} of {len(groups)} groups; chart/table are truncated."
                )
            chart = {"kind": "bar", "x": step.group_by, "y": step.metric, "rows": rows}
        elif step.operation == "compare_periods":
            _column(data, step.date_column)
            if step.aggregation not in ("sum", "count"):
                raise ValueError("Contributions require additive sum or count")
            if len(step.periods) != 2 or any(len(p) != 2 for p in step.periods):
                raise ValueError("Two inclusive ISO date windows are required")
            windows = []
            for period in step.periods:
                if not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) for v in period):
                    raise ValueError("Use ISO calendar dates YYYY-MM-DD")
                start, end = (
                    pd.Timestamp(period[0], tz="UTC"),
                    pd.Timestamp(period[1], tz="UTC"),
                )
                if start > end:
                    raise ValueError("Period start is after its end")
                windows.append((start, end + pd.Timedelta(days=1)))
            if max(windows[0][0], windows[1][0]) < min(windows[0][1], windows[1][1]):
                raise ValueError("Comparison periods must not overlap")
            parsed = dates(data[step.date_column])
            excluded = int(parsed.isna().sum())
            if excluded:
                limitations.append(
                    f"{excluded} rows have missing/invalid ISO dates and were excluded."
                )
            subsets = [
                data.loc[(parsed >= start) & (parsed < end)] for start, end in windows
            ]
            totals = [
                _aggregate(part[step.metric], step.aggregation) for part in subsets
            ]
            delta = (
                totals[1] - totals[0] if all(t is not None for t in totals) else None
            )
            summary = {
                "previous": totals[0],
                "current": totals[1],
                "change": delta,
                "percent_change": (
                    100 * delta / totals[0]
                    if delta is not None and totals[0] != 0
                    else None
                ),
                "previous_rows": len(subsets[0]),
                "current_rows": len(subsets[1]),
            }
            if summary["percent_change"] is None:
                limitations.append(
                    "Percentage change is undefined for an empty/zero baseline."
                )
            limitations.append(
                "Period coverage may be incomplete. Observations identify contributions, not business causes."
            )
            if step.group_by:
                _column(data, step.group_by)
                if data[step.group_by].nunique(dropna=False) > 500:
                    raise ValueError("Too many groups (max 500)")
                parts = [
                    _grouped_aggregate(p, step).rename(name)
                    for p, name in zip(subsets, ["previous", "current"])
                ]
                categories = parts[0].index.union(parts[1].index, sort=False)
                grouped = pd.concat(
                    [part.reindex(categories, fill_value=0) for part in parts], axis=1
                ).reset_index()
                limitations.append(
                    "Categories without rows in a period contribute zero; present categories "
                    "without usable metric values remain unknown."
                )
                grouped["change"] = grouped["current"] - grouped["previous"]
                grouped = grouped.sort_values("change", key=abs, ascending=False)
                rows = json.loads(grouped.head(step.limit).to_json(orient="records"))
                if len(grouped) > step.limit:
                    limitations.append(
                        f"Showing {step.limit} of {len(grouped)} contributions; summary covers all groups."
                    )
                chart = {"kind": "bar", "x": step.group_by, "y": "change", "rows": rows}
            else:
                rows = [
                    {"period": "Previous", "value": totals[0]},
                    {"period": "Current", "value": totals[1]},
                ]
                chart = {"kind": "bar", "x": "period", "y": "value", "rows": rows}
        else:
            _column(data, step.second_metric, numeric=True)
            pairs = data[[step.metric, step.second_metric]].dropna()
            if len(pairs) < 3:
                raise ValueError(
                    "Correlation requires at least three paired observations"
                )
            correlation = pairs.iloc[:, 0].corr(pairs.iloc[:, 1])
            summary = {
                "pearson_r": float(correlation) if pd.notna(correlation) else None,
                "paired_rows": len(pairs),
            }
            rows = [summary]
            limitations.append(
                "Pearson correlation measures linear association, not causation; outliers and sample selection matter."
            )
        payload = {
            "version": 1,
            "id": f"e{index + 1}",
            "tool_version": TOOL_VERSION,
            "operation": step.operation,
            "metric": step.metric,
            "aggregation": step.aggregation,
            "summary": summary,
            "rows": rows,
            "chart": chart,
            "input_rows": len(df),
            "filtered_rows": len(data),
            "filters": [f.model_dump() for f in step.filters],
            "periods": step.periods,
            "limitations": limitations,
        }
        payload["hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        evidence.append(clean(payload))
    return evidence
