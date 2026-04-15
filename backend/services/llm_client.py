import os
import time
import asyncio
import logging
from dataclasses import dataclass
from queue import Queue
from threading import Thread, Lock

import httpx

logger = logging.getLogger(__name__)

_STOP_SIGNAL = object()


class LLMError(Exception):
    def __init__(self, message: str, *, kind: str = "general"):
        super().__init__(message)
        self.kind = kind


@dataclass
class _QueuedLLMRequest:
    request_id: int
    messages: list[dict]
    temperature: float
    max_tokens: int
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future


class _SerializedLLMScheduler:
    """
    Process-wide serialized scheduler for all outbound LLM requests.
    Guarantees:
      - FIFO request order (single worker thread)
      - Minimum delay between HTTP calls
      - One-pass key attempts per request (no nested retries)
      - Global cooldown when all keys fail
    """

    def __init__(
        self,
        keys: list[str],
        model: str,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        min_interval_seconds: float = 0.5,
        cooldown_seconds: float = 25.0,
    ):
        self.keys = keys
        self.model = model
        self.base_url = base_url
        self.min_interval_seconds = min_interval_seconds
        self.cooldown_seconds = cooldown_seconds

        self._queue: Queue = Queue()
        self._state_lock = Lock()
        self._request_counter = 0
        self._next_key_start_index = 0
        self._last_dispatch_at = 0.0
        self._cooldown_until = 0.0
        self._closed = False

        self._client = httpx.Client(timeout=timeout_seconds)
        self._worker = Thread(
            target=self._run,
            name="llm-request-worker",
            daemon=True,
        )
        self._worker.start()

        logger.info(
            "[LLM] Scheduler initialized — keys=%d model=%s min_interval=%.2fs cooldown=%.1fs",
            len(self.keys),
            self.model,
            self.min_interval_seconds,
            self.cooldown_seconds,
        )

    def is_compatible(self, keys: list[str], model: str, base_url: str) -> bool:
        return self.keys == keys and self.model == model and self.base_url == base_url

    def enqueue(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        *,
        loop: asyncio.AbstractEventLoop,
        future: asyncio.Future,
    ) -> int:
        with self._state_lock:
            if self._closed:
                raise LLMError("LLM scheduler is closed", kind="closed")
            self._request_counter += 1
            request_id = self._request_counter

        queued = _QueuedLLMRequest(
            request_id=request_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            loop=loop,
            future=future,
        )
        self._queue.put(queued)
        logger.info("[LLM QUEUE] request queued request_id=%d", request_id)
        return request_id

    def close(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True

        self._queue.put(_STOP_SIGNAL)
        self._worker.join(timeout=5.0)
        self._client.close()
        logger.info("[LLM] Scheduler closed")

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is _STOP_SIGNAL:
                    break
                self._process_request(item)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("[LLM] Unexpected scheduler failure: %s", exc)
                if isinstance(item, _QueuedLLMRequest):
                    self._resolve_exception(
                        item,
                        LLMError("Internal LLM scheduler failure", kind="internal"),
                    )
            finally:
                self._queue.task_done()

    def _process_request(self, request: _QueuedLLMRequest):
        if request.future.cancelled():
            return

        self._wait_for_cooldown_if_needed(request_id=request.request_id)
        logger.info("[LLM QUEUE] request started request_id=%d", request.request_id)

        key_order = self._reserve_key_order()
        last_error: LLMError | None = None
        backoff = 0.5  # start small

        for key_index in key_order:
            self._apply_global_throttle()

            logger.info(
                "[LLM] using key %d request_id=%d",
                key_index + 1,
                request.request_id,
            )

            try:
                result = self._make_request(
                    api_key=self.keys[key_index],
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

                logger.info(
                    "[LLM] success request_id=%d key=%d",
                    request.request_id,
                    key_index + 1,
                )

                self._resolve_success(request, result)
                return

            except LLMError as exc:
                last_error = exc

                if exc.kind == "rate_limit":
                    logger.warning(
                        "[LLM] rate limit hit request_id=%d key=%d",
                        request.request_id,
                        key_index + 1,
                    )

                    # 🔥 CRITICAL ADDITION
                    time.sleep(backoff)
                    backoff *= 1.5  # gentle increase

                else:
                    logger.warning(
                        "[LLM] request failed request_id=%d key=%d kind=%s error=%s",
                        request.request_id,
                        key_index + 1,
                        exc.kind,
                        str(exc),
                    )
        # for key_index in key_order:
        #     self._apply_global_throttle()
        #     logger.info(
        #         "[LLM] using key %d request_id=%d",
        #         key_index + 1,
        #         request.request_id,
        #     )
        #     try:
        #         result = self._make_request(
        #             api_key=self.keys[key_index],
        #             messages=request.messages,
        #             temperature=request.temperature,
        #             max_tokens=request.max_tokens,
        #         )
        #         logger.info(
        #             "[LLM] success request_id=%d key=%d",
        #             request.request_id,
        #             key_index + 1,
        #         )
        #         self._resolve_success(request, result)
        #         return
        #     except LLMError as exc:
        #         last_error = exc
        #         if exc.kind == "rate_limit":
        #             logger.warning(
        #                 "[LLM] rate limit hit request_id=%d key=%d",
        #                 request.request_id,
        #                 key_index + 1,
        #             )
        #         else:
        #             logger.warning(
        #                 "[LLM] request failed request_id=%d key=%d kind=%s error=%s",
        #                 request.request_id,
        #                 key_index + 1,
        #                 exc.kind,
        #                 str(exc),
        #             )

        self._trigger_global_cooldown()
        self._resolve_exception(
            request,
            last_error
            or LLMError("All API keys failed for this request", kind="exhausted"),
        )

    def _reserve_key_order(self) -> list[int]:
        with self._state_lock:
            start = self._next_key_start_index
            self._next_key_start_index = (self._next_key_start_index + 1) % len(self.keys)
        return [((start + offset) % len(self.keys)) for offset in range(len(self.keys))]

    def _wait_for_cooldown_if_needed(self, *, request_id: int):
        with self._state_lock:
            remaining = self._cooldown_until - time.monotonic()
        if remaining > 0:
            logger.warning(
                "[LLM] global cooldown active request_id=%d wait=%.2fs",
                request_id,
                remaining,
            )
            time.sleep(remaining)

    def _apply_global_throttle(self):
        while True:
            with self._state_lock:
                now = time.monotonic()
                wait_for = self.min_interval_seconds - (now - self._last_dispatch_at)
                if wait_for <= 0:
                    self._last_dispatch_at = now
                    return
            time.sleep(wait_for)

    def _trigger_global_cooldown(self):
        with self._state_lock:
            next_until = time.monotonic() + self.cooldown_seconds
            self._cooldown_until = max(self._cooldown_until, next_until)
        logger.warning(
            "[LLM] global cooldown triggered duration=%.1fs",
            self.cooldown_seconds,
        )

    def _make_request(
        self,
        *,
        api_key: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise LLMError("Request timed out", kind="timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Transport error: {exc}", kind="network") from exc

        if response.status_code == 429:
            raise LLMError("rate limit exceeded", kind="rate_limit")
        if response.status_code != 200:
            snippet = response.text[:200].replace("\n", " ")
            raise LLMError(
                f"LLM API error {response.status_code}: {snippet}",
                kind="api_error",
            )

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise LLMError("Malformed LLM response payload", kind="api_error") from exc

    def _resolve_success(self, request: _QueuedLLMRequest, value: str):
        if request.future.cancelled():
            return
        try:
            request.loop.call_soon_threadsafe(self._safe_set_result, request.future, value)
        except RuntimeError as exc:
            logger.warning("[LLM] Failed to resolve request result: %s", exc)

    def _resolve_exception(self, request: _QueuedLLMRequest, error: Exception):
        if request.future.cancelled():
            return
        try:
            request.loop.call_soon_threadsafe(self._safe_set_exception, request.future, error)
        except RuntimeError as exc:
            logger.warning("[LLM] Failed to resolve request exception: %s", exc)

    @staticmethod
    def _safe_set_result(future: asyncio.Future, value: str):
        if not future.done():
            future.set_result(value)

    @staticmethod
    def _safe_set_exception(future: asyncio.Future, error: Exception):
        if not future.done():
            future.set_exception(error)


class LLMClient:
    """
    Public async facade over a global serialized request scheduler.

    Public interface remains unchanged:
      await client.complete(messages, temperature, max_tokens) -> str
      await client.complete_with_retry(messages, max_attempts) -> str
      await client.close()
    """

    _scheduler: _SerializedLLMScheduler | None = None
    _scheduler_ref_count: int = 0
    _scheduler_lock: Lock = Lock()

    def __init__(
        self,
        api_keys_str: str,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.keys: list[str] = [k.strip() for k in api_keys_str.split(",") if k.strip()]
        if not self.keys:
            raise RuntimeError(
                "No API keys configured. Set LLM_API_KEYS=key1,key2,... in your environment."
            )

        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.base_url = (
            base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        ).rstrip("/")
        self._closed = False

        with self.__class__._scheduler_lock:
            scheduler = self.__class__._scheduler
            if scheduler is None:
                scheduler = _SerializedLLMScheduler(
                    keys=self.keys,
                    model=self.model,
                    base_url=self.base_url,
                    min_interval_seconds=1.2,
                    cooldown_seconds=25.0,
                )
                self.__class__._scheduler = scheduler
            elif not scheduler.is_compatible(self.keys, self.model, self.base_url):
                logger.warning(
                    "[LLM] Existing scheduler config differs from new LLMClient config; "
                    "reusing existing global scheduler."
                )
            self.__class__._scheduler_ref_count += 1

        logger.info(
            "[LLM] Client initialized — keys=%d model=%s",
            len(self.keys),
            self.model,
        )

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> str:
        if self._closed:
            raise LLMError("LLM client is closed", kind="closed")

        with self.__class__._scheduler_lock:
            scheduler = self.__class__._scheduler
        if scheduler is None:
            raise LLMError("LLM scheduler unavailable", kind="internal")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        scheduler.enqueue(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            loop=loop,
            future=future,
        )
        return await future

    async def complete_with_retry(
        self,
        messages: list[dict],
        max_attempts: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> str:
        """
        Compatibility wrapper.
        Retry behavior is intentionally centralized in the scheduler as a
        single key-pass (no nested retry loops, no retry storms).
        """
        _ = max_attempts  # retained for API compatibility
        return await self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def close(self):
        if self._closed:
            return
        self._closed = True

        scheduler_to_close: _SerializedLLMScheduler | None = None
        with self.__class__._scheduler_lock:
            self.__class__._scheduler_ref_count = max(
                0, self.__class__._scheduler_ref_count - 1
            )
            if self.__class__._scheduler_ref_count == 0:
                scheduler_to_close = self.__class__._scheduler
                self.__class__._scheduler = None

        if scheduler_to_close is not None:
            await asyncio.to_thread(scheduler_to_close.close)
