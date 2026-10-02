"""Two bounded structured-output requests per configured provider. Never prints keys."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from config import settings
from models.investigations import AnalysisPlan, Findings
from services.model_gateway import ModelGateway, InferenceUnavailable


class Budget:
    def reserve(self, *args):
        return {"allowed": True}

    def provider_failure(self, *args, **kwargs):
        pass


async def main():
    gateway = ModelGateway(settings, Budget())
    for provider in gateway.providers.copy():
        gateway.providers = [provider]
        state = {"http_attempts": 0, "provider_calls": []}

        async def save():
            pass

        try:
            plan = await gateway.complete(
                "planner",
                "Plan: compare January and February 2026 revenue by region. Columns: date (ISO dates), revenue (number), region (category). Use sum, compare_periods. Return exactly one calculation step, no clarification.",
                AnalysisPlan,
                state,
                save,
            )
            findings = await gateway.complete(
                "analyst",
                "Return qualitative findings with evidence_id e1, no numerical statements. Computed facts: North increase offsets South decline; revenue increases. No business causes established.",
                Findings,
                state,
                save,
            )
            print(
                json.dumps(
                    {
                        "provider": provider.name,
                        "model": provider.model,
                        "outcome": "valid_structured_outputs",
                        "attempts": state["http_attempts"],
                        "plan_operations": [s.operation for s in plan.steps],
                        "finding_count": len(findings.findings),
                    }
                )
            )
        except InferenceUnavailable as error:
            print(
                json.dumps(
                    {
                        "provider": provider.name,
                        "model": provider.model,
                        "outcome": "unavailable",
                        "reasons": error.reasons,
                        "attempts": state["http_attempts"],
                    }
                )
            )
    await gateway.close()


asyncio.run(main())
