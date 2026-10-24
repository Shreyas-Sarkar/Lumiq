# Demo data

These CSVs contain invented data, no personal records.

`revenue.csv`: January revenue = 300; February = 340; change = 40 (13.3333%).
North contributes +60; South contributes -20. All rows sum to 640.
January orders = 24; February = 27. Coverage is partial in both months: only
sampled days are present. The CSV cannot establish reasons for the change.

`quality-issues.csv`: 5 rows, 1 duplicate, 1 missing metric, 1 invalid date,
1 missing region. January baseline = 0, so percentage change is undefined.
Do not silently remove duplicates.
