import json
import unittest
from types import SimpleNamespace
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from config import Settings
from routers.investigations import router
from routers.auth_middleware import get_current_user
from services.investigation import InvestigationService
from services.model_gateway import ModelGateway
from fakes import *


class InvestigationAPITests(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()
        self.datasets = FakeDatasets()
        self.calls = []
        self.responses = [
            PLAN,
            {
                "findings": [
                    {
                        "evidence_id": "e1",
                        "observation": "North offsets the decline in South.",
                    }
                ]
            },
        ]

        def transport(request):
            self.calls.append(request.url.host)
            answer = self.responses.pop(0)
            if isinstance(answer, int):
                return httpx.Response(
                    answer,
                    json={"error": {"message": "private provider details"}},
                    headers={"retry-after": "60"},
                )
            if request.url.host == "api.groq.com":
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": json.dumps(answer)}}]},
                )
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps(answer)}]}}
                    ]
                },
            )

        cfg = Settings(
            _env_file=None, gemini_api_key="fake", llm_api_keys="fake", gemini_rpm=10
        )
        self.gateway = ModelGateway(cfg, self.store, httpx.MockTransport(transport))
        self.svc = InvestigationService(self.store, self.datasets, self.gateway)
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.state.investigations = self.svc
        app.state.dataset_manager = self.datasets
        app.state.chat_manager = FakeChats()
        self.app = app
        app.dependency_overrides[get_current_user] = lambda: {
            "id": OWNER,
            "email": "owner@example.test",
        }
        self.client = TestClient(app)

    def submit(self, **kwargs):
        body = {
            "chat_id": CHAT,
            "question": "Compare January to February revenue by region",
            "request_id": str(uuid.uuid4()),
            **kwargs,
        }
        response = self.client.post("/api/investigations", json=body)
        self.assertEqual(response.status_code, 202, response.text)
        rid = response.json()["id"]
        return self.client.get("/api/investigations/" + rid).json()

    def test_success_two_calls_table_and_chart_download(self):
        run = self.submit()
        self.assertEqual(run["state"]["status"], "complete")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(run["state"]["evidence"][0]["summary"]["change"], 40)
        self.assertTrue(run["state"]["evidence"][0]["chart"])
        csv = self.client.get(f"/api/investigations/{run['id']}/evidence/e1/download")
        self.assertEqual(csv.status_code, 200)
        self.assertIn("North,220,280,60", csv.text)
        self.assertNotIn("lease_token", run)

    def test_gemini_quota_fails_over_to_groq(self):
        self.responses = [
            429,
            PLAN,
            {
                "findings": [
                    {"evidence_id": "e1", "observation": "Revenue increased in North."}
                ]
            },
        ]
        run = self.submit()
        self.assertEqual(run["state"]["status"], "complete")
        self.assertEqual(
            self.calls,
            ["generativelanguage.googleapis.com", "api.groq.com", "api.groq.com"],
        )
        self.assertEqual(run["state"]["http_attempts"], 3)

    def test_all_provider_failure_preserves_evidence(self):
        self.responses = [PLAN, 429, 503]
        run = self.submit()
        self.assertEqual(run["state"]["status"], "interpretation_unavailable")
        self.assertEqual(run["state"]["evidence"][0]["summary"]["current"], 340)
        self.assertEqual(run["state"]["findings"], [])
        self.assertNotIn("private provider details", json.dumps(run))

    def test_clarification_is_persisted_and_answer_resumes(self):
        self.responses = [{"clarification": "Which revenue periods should I compare?"}]
        run = self.submit()
        self.assertEqual(run["state"]["status"], "needs_clarification")
        self.responses = [PLAN, {"findings": []}]
        response = self.client.post(
            f"/api/investigations/{run['id']}/resume",
            json={"clarification": "January and February 2026"},
        )
        self.assertEqual(response.status_code, 202, response.text)
        updated = self.client.get("/api/investigations/" + run["id"]).json()
        self.assertEqual(updated["state"]["status"], "complete")

    def test_ownership_and_idempotency(self):
        run = self.submit(request_id="same-request")
        duplicate = self.submit(request_id="same-request")
        self.assertEqual(run["id"], duplicate["id"])
        self.assertEqual(len(self.calls), 2)
        self.app.dependency_overrides[get_current_user] = lambda: {
            "id": str(uuid.uuid4())
        }
        self.assertEqual(
            self.client.get("/api/investigations/" + run["id"]).status_code, 404
        )
        self.assertEqual(
            self.client.get(
                f"/api/investigations/{run['id']}/evidence/e1/download"
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                "/api/investigations", params={"chat_id": CHAT}
            ).status_code,
            404,
        )

    def test_restart_reuses_computed_evidence(self):
        run = self.submit()
        raw = self.store.rows[run["id"]]
        raw["state"]["status"] = "interrupted"
        raw["state"]["findings"] = []
        self.app.state.investigations = InvestigationService(
            self.store, self.datasets, self.gateway
        )
        self.responses = [{"findings": []}]
        resumed = self.client.get("/api/investigations/" + run["id"])
        self.assertEqual(resumed.status_code, 200)
        final = self.client.get("/api/investigations/" + run["id"]).json()
        self.assertEqual(final["state"]["status"], "complete")
        self.assertEqual(len(self.calls), 3)  # only interpretation repeated

    def test_plan_edit_requires_new_revision_for_completed_run(self):
        run = self.submit()
        self.assertEqual(
            self.client.post(
                f"/api/investigations/{run['id']}/resume", json={"plan": PLAN}
            ).status_code,
            409,
        )

    def test_budget_stops_inference(self):
        self.store.blocked = {
            "gemini": {"allowed": False, "reason": "daily_quota"},
            "groq": {"allowed": False, "reason": "quota"},
        }
        run = self.submit()
        self.assertEqual(run["state"]["status"], "waiting_for_quota")
        self.assertEqual(self.calls, [])

    def test_explicit_plan_rerun_reuses_verified_computation(self):
        first = self.submit()
        self.responses = [{"findings": []}]
        second = self.submit(plan=PLAN, parent_id=first["id"])
        self.assertTrue(second["state"]["computation_reused"])
        self.assertEqual(second["state"]["http_attempts"], 1)
        self.assertEqual(second["state"]["evidence"], first["state"]["evidence"])

    def test_invalid_finding_switches_provider_without_invented_numbers(self):
        self.responses = [
            PLAN,
            {"findings": [{"evidence_id": "fake", "observation": "Revenue is 999."}]},
            {
                "findings": [
                    {"evidence_id": "e1", "observation": "North offsets South."}
                ]
            },
        ]
        run = self.submit()
        self.assertEqual(run["state"]["status"], "complete")
        self.assertEqual(self.calls[-1], "api.groq.com")
        self.assertNotIn("999", json.dumps(run["state"]["findings"]))

    def test_both_bad_plans_fail_without_synthesized_answer(self):
        self.responses = [
            {"steps": [{"operation": "aggregate", "metric": "unknown"}]},
            {"steps": [{"operation": "aggregate", "metric": "still_unknown"}]},
        ]
        run = self.submit()
        self.assertEqual(run["state"]["status"], "failed")
        self.assertEqual(run["state"]["evidence"], [])
        self.assertEqual(run["state"]["findings"], [])

    def test_changed_payload_cannot_reuse_an_idempotency_key(self):
        self.submit(request_id="same-request")
        response = self.client.post(
            "/api/investigations",
            json={
                "chat_id": CHAT,
                "question": "A different question",
                "request_id": "same-request",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.calls), 2)
