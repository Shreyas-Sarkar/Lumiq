"""Gemini native and Groq adapters with one shared, persisted attempt budget."""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import httpx
from pydantic import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class Provider:
    name: str
    model: str
    key: str
    rpm: int
    rpd: int
    tpm: int


class InferenceUnavailable(Exception):
    def __init__(self, reasons, retry_at=None):
        self.reasons = reasons
        self.retry_at = retry_at
        self.resumable = any(
            x["reason"] in ("quota", "daily_quota", "outage", "timeout", "invalid_output")
            for x in reasons
        )
        descriptions = {
            "quota": "is temporarily at capacity",
            "daily_quota": "has reached the configured daily allowance",
            "outage": "could not be reached",
            "timeout": "did not respond in time",
            "invalid_output": "returned an answer that did not pass validation",
            "authentication": "rejected its configured API key",
            "unsupported_model": "does not offer the configured model",
            "invalid_request": "rejected the request format",
            "configuration": "has no configured model provider",
            "attempt_budget_exhausted": "has used this run's model-attempt allowance",
        }
        super().__init__("; ".join(
            f"{item['provider'].capitalize()} {descriptions.get(item['reason'], 'is unavailable')}"
            for item in reasons
        ) + ".")


class ModelGateway:
    def __init__(self, settings, store, transport=None):
        self.store = store
        self.client = httpx.AsyncClient(timeout=25, transport=transport)
        self.providers = []
        if settings.gemini_api_key:
            self.providers.append(
                Provider(
                    "gemini",
                    settings.gemini_model,
                    settings.gemini_api_key,
                    settings.gemini_rpm,
                    settings.gemini_rpd,
                    settings.gemini_tpm,
                )
            )
        groq = [x.strip() for x in settings.llm_api_keys.split(",") if x.strip()]
        if groq:
            # Additional keys share organization limits; they do not multiply quota.
            self.providers.append(
                Provider(
                    "groq",
                    settings.llm_model,
                    groq[0],
                    settings.groq_rpm,
                    settings.groq_rpd,
                    settings.groq_tpm,
                )
            )

    async def close(self):
        await self.client.aclose()

    @staticmethod
    def next_day():
        pacific = datetime.now(ZoneInfo("America/Los_Angeles"))
        return (
            (
                pacific.replace(hour=0, minute=0, second=0, microsecond=0)
                + timedelta(days=1)
            )
            .astimezone(timezone.utc)
            .isoformat()
        )

    async def complete(self, stage, prompt, contract, state, checkpoint, validate=None):
        reasons, retry_times = [], []
        attempts = 0
        for provider in self.providers:
            if attempts >= 2 or state.get("http_attempts", 0) >= 6:
                break
            if len(prompt) > 24000:
                raise ValueError("Inference context exceeds the prompt budget")
            tokens = (
                len(prompt) // 3 + 2200
            )  # reserve conservatively, including max output
            budget = await asyncio.to_thread(
                self.store.reserve,
                provider.name,
                datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat(),
                provider.rpm,
                provider.rpd,
                provider.tpm,
                tokens,
            )
            if not budget.get("allowed"):
                reason = budget.get("reason", "quota")
                retry = budget.get("retry_at") or (
                    self.next_day() if reason == "daily_quota" else None
                )
                reasons.append({"provider": provider.name, "reason": reason})
                if retry:
                    retry_times.append(retry)
                continue
            state["http_attempts"] = state.get("http_attempts", 0) + 1
            attempts += 1
            state.setdefault("provider_calls", []).append(
                {
                    "stage": stage,
                    "provider": provider.name,
                    "model": provider.model,
                    "outcome": "response_unconfirmed",
                }
            )
            # Persist before inference: lost responses cannot create hidden retries.
            await checkpoint()
            try:
                schema = contract.model_json_schema()
                if provider.name == "gemini":
                    response = await self.client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{provider.model}:generateContent",
                        headers={"x-goog-api-key": provider.key},
                        json={
                            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": 0,
                                "maxOutputTokens": 2200,
                                "responseMimeType": "application/json",
                                "responseJsonSchema": schema,
                            },
                        },
                    )
                else:
                    response = await self.client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {provider.key}"},
                        json={
                            "model": provider.model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "Return one JSON object matching this schema: "
                                    + json.dumps(schema),
                                },
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0,
                            "max_tokens": 2200,
                            "response_format": {"type": "json_object"},
                        },
                    )
                if response.status_code >= 400:
                    category = (
                        "quota"
                        if response.status_code == 429
                        else "authentication"
                        if response.status_code in (401, 403)
                        else "unsupported_model"
                        if response.status_code == 404
                        else "invalid_request"
                        if response.status_code == 400
                        else "outage"
                    )
                    retry = None
                    if category == "quota":
                        seconds = self._retry_seconds(response)
                        retry = (
                            datetime.now(timezone.utc) + timedelta(seconds=seconds)
                        ).isoformat()
                        await asyncio.to_thread(
                            self.store.provider_failure, provider.name, retry_at=retry
                        )
                    elif category in ("authentication", "unsupported_model"):
                        await asyncio.to_thread(
                            self.store.provider_failure,
                            provider.name,
                            disabled=category,
                        )
                    state["provider_calls"][-1]["outcome"] = category
                    reasons.append({"provider": provider.name, "reason": category})
                    if retry:
                        retry_times.append(retry)
                    continue
                body = response.json()
                if provider.name == "gemini":
                    raw = "".join(
                        p.get("text", "")
                        for p in body.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [])
                    )
                else:
                    raw = body["choices"][0]["message"]["content"]
                result = contract.model_validate_json(raw)
                if validate:
                    validate(result)
                state["provider_calls"][-1]["outcome"] = "success"
                state["provider_calls"][-1]["usage"] = body.get(
                    "usageMetadata", body.get("usage", {})
                )
                await checkpoint()
                return result
            except (ValidationError, ValueError, KeyError, IndexError, TypeError) as error:
                issue = type(error).__name__
                if isinstance(error, ValidationError):
                    issue = "; ".join(
                        ".".join(map(str, item["loc"])) + ":" + item["type"]
                        for item in error.errors(include_input=False)[:4]
                    )
                elif str(error) in {"unknown_evidence_reference", "numerical_claim"}:
                    issue = str(error)
                state["provider_calls"][-1]["outcome"] = "invalid_output"
                state["provider_calls"][-1]["validation_issue"] = issue
                logger.warning("Model output rejected: provider=%s stage=%s issue=%s",
                               provider.name, stage, issue)
                reasons.append({"provider": provider.name, "reason": "invalid_output"})
            except httpx.TimeoutException:
                state["provider_calls"][-1]["outcome"] = "timeout"
                reasons.append({"provider": provider.name, "reason": "timeout"})
            except httpx.RequestError:
                state["provider_calls"][-1]["outcome"] = "outage"
                reasons.append({"provider": provider.name, "reason": "outage"})
        if not reasons:
            reasons = [
                {
                    "provider": "all",
                    "reason": "attempt_budget_exhausted"
                    if self.providers
                    else "configuration",
                }
            ]
        raise InferenceUnavailable(reasons, min(retry_times) if retry_times else None)

    @staticmethod
    def _retry_seconds(response):
        try:
            seconds = float(response.headers.get("retry-after", "0"))
            for d in response.json().get("error", {}).get("details", []):
                if "retryDelay" in d:
                    seconds = max(
                        seconds, float(str(d["retryDelay"]).removesuffix("s"))
                    )
            return min(max(seconds, 60), 86400)
        except (ValueError, TypeError):
            return 60
