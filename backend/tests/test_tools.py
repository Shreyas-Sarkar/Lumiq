import unittest
import pandas as pd
from services.analysis_tools import execute_plan, profile_dataset
from models.investigations import AnalysisPlan


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv("../samples/revenue.csv")

    def test_period_change_and_contributions(self):
        plan = AnalysisPlan.model_validate(
            {
                "steps": [
                    {
                        "operation": "compare_periods",
                        "metric": "revenue",
                        "date_column": "date",
                        "group_by": "region",
                        "periods": [
                            ["2026-01-01", "2026-01-31"],
                            ["2026-02-01", "2026-02-28"],
                        ],
                    }
                ]
            }
        )
        evidence = execute_plan(self.df, plan)[0]
        self.assertEqual(evidence["summary"]["previous"], 300)
        self.assertEqual(evidence["summary"]["current"], 340)
        self.assertEqual(evidence["summary"]["change"], 40)
        self.assertEqual(
            {r["region"]: r["change"] for r in evidence["rows"]},
            {"North": 60, "South": -20},
        )
        self.assertTrue(evidence["chart"])
        self.assertTrue(evidence["limitations"])

    def test_profile_quality_and_zero_baseline(self):
        df = pd.read_csv("../samples/quality-issues.csv")
        p = profile_dataset(df)
        self.assertEqual(p["duplicate_rows"], 1)
        self.assertEqual(
            next(c for c in p["columns"] if c["name"] == "date")["invalid_dates"], 1
        )
        e = execute_plan(
            df,
            AnalysisPlan(
                steps=[
                    {
                        "operation": "compare_periods",
                        "metric": "revenue",
                        "date_column": "date",
                        "periods": [
                            ["2026-01-01", "2026-01-31"],
                            ["2026-02-01", "2026-02-28"],
                        ],
                    }
                ]
            ),
        )[0]
        self.assertIsNone(e["summary"]["percent_change"])

    def test_unknown_column_and_overlapping_periods_are_rejected(self):
        for step in [
            {"operation": "aggregate", "metric": "unknown"},
            {
                "operation": "compare_periods",
                "metric": "revenue",
                "date_column": "date",
                "periods": [["2026-01-01", "2026-01-31"], ["2026-01-20", "2026-02-28"]],
            },
        ]:
            with self.assertRaises(ValueError):
                execute_plan(self.df, AnalysisPlan(steps=[step]))
