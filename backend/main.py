import asyncio
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config import settings
from routers import upload, chat, investigations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    from db.supabase_client import supabase
    from services.dataset_manager import DatasetManager
    from services.chat_manager import ChatManager

    os.makedirs(settings.upload_dir, exist_ok=True)
    app.state.supabase_client = supabase
    app.state.settings = settings
    app.state.dataset_manager = DatasetManager(supabase, settings.upload_dir)
    app.state.chat_manager = ChatManager(supabase)
    app.state.rag_service = None
    app.state.investigations = None
    app.state.orchestrator = None
    if settings.investigations_enabled:
        from services.run_store import RunStore
        from services.model_gateway import ModelGateway
        from services.investigation import InvestigationService

        store = RunStore(supabase)
        app.state.investigations = InvestigationService(
            store, app.state.dataset_manager, ModelGateway(settings, store)
        )
    if settings.legacy_execution_enabled:
        from services.rag_service import RAGService
        from services.llm_client import LLMClient
        from services.code_generator import CodeGenerator
        from services.execution_engine import ExecutionEngine
        from services.evaluator import Evaluator
        from services.explanation import ExplanationService
        from services.log_service import ExecutionLogService
        from services.mode_classifier import ModeClassifier
        from services.cognitive_engine import CognitiveEngine
        from services.context_summarizer import ContextSummarizer
        from services.query_cache import QueryCache
        from services.orchestrator import ChatOrchestrator

        llm = LLMClient(
            settings.llm_api_keys, settings.llm_model, settings.llm_base_url
        )
        app.state.llm_client = llm
        app.state.rag_service = RAGService()
        app.state.orchestrator = ChatOrchestrator(
            chat_manager=app.state.chat_manager,
            dataset_manager=app.state.dataset_manager,
            rag_service=app.state.rag_service,
            code_generator=CodeGenerator(llm),
            execution_engine=ExecutionEngine(settings.execution_timeout_seconds),
            evaluator=Evaluator(llm, settings.max_retry_count),
            explanation_service=ExplanationService(),
            log_service=ExecutionLogService(supabase) if supabase else None,
            mode_classifier=ModeClassifier(),
            cognitive_engine=CognitiveEngine(llm),
            context_summarizer=ContextSummarizer(llm),
            query_cache=QueryCache(),
            max_retry_count=settings.max_retry_count,
        )
    recovery = (
        asyncio.create_task(app.state.investigations.recover())
        if app.state.investigations
        else None
    )
    yield
    if recovery:
        recovery.cancel()
        try:
            await recovery
        except asyncio.CancelledError:
            pass
    if app.state.investigations:
        await app.state.investigations.gateway.close()
    if getattr(app.state, "llm_client", None):
        await app.state.llm_client.close()


app = FastAPI(title="Lumiq API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(investigations.router, prefix="/api", tags=["investigations"])


@app.exception_handler(Exception)
async def error(request: Request, exc: Exception):
    logger.error("Request failed: %s", type(exc).__name__)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "A required service is unavailable. Your saved data has not been replaced. Retry, or check backend configuration."
        },
    )


@app.get("/health")
async def health(request: Request):
    checks = {}
    db = getattr(request.app.state, "supabase_client", None)
    try:
        if not db:
            raise RuntimeError()
        db.table("chat").select("id").limit(1).execute()
        checks["database"] = "ok"
        if settings.investigations_enabled:
            db.table("investigation_run").select("id").limit(1).execute()
            bucket = db.storage.get_bucket(settings.dataset_bucket)
            checks["workspace"] = (
                "ok" if not bucket.public else "error: bucket must be private"
            )
    except Exception:
        checks["database"] = "unavailable or migration missing"
    gateway = getattr(
        getattr(request.app.state, "investigations", None), "gateway", None
    )
    checks["providers"] = (
        "configured; live availability not checked"
        if gateway and gateway.providers
        else "not configured"
    )
    return {
        "version": "3.0.0",
        "status": "ok"
        if all(v == "ok" for k, v in checks.items() if k != "providers")
        and gateway
        and gateway.providers
        else "degraded",
        "checks": checks,
    }
