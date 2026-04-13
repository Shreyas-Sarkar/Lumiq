import os
from dotenv import load_dotenv

load_dotenv()

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from config import settings
from routers import upload, chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.upload_dir, exist_ok=True)

    from db.supabase_client import supabase
    app.state.supabase_client = supabase
    app.state.settings = settings

    from services.rag_service import RAGService
    from services.dataset_manager import DatasetManager
    from services.chat_manager import ChatManager
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

    app.state.rag_service = RAGService()
    app.state.dataset_manager = DatasetManager(
        supabase=supabase, upload_dir=settings.upload_dir
    )
    app.state.chat_manager = ChatManager(supabase=supabase)

    llm_client = LLMClient(
        api_keys_str=settings.llm_api_keys,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )
    app.state.llm_client = llm_client

    code_generator = CodeGenerator(llm_client=llm_client)
    execution_engine = ExecutionEngine(
        timeout_seconds=settings.execution_timeout_seconds
    )
    evaluator = Evaluator(
        llm_client=llm_client, max_retry_count=settings.max_retry_count
    )
    explanation_service = ExplanationService()
    log_service = ExecutionLogService(supabase=supabase) if supabase else None

    # New dual-mode services
    mode_classifier = ModeClassifier()
    cognitive_engine = CognitiveEngine(llm_client=llm_client)
    context_summarizer = ContextSummarizer(llm_client=llm_client)
    query_cache = QueryCache(
        ttl_seconds=3600,
        max_size=200,
    )

    app.state.orchestrator = ChatOrchestrator(
        chat_manager=app.state.chat_manager,
        dataset_manager=app.state.dataset_manager,
        rag_service=app.state.rag_service,
        code_generator=code_generator,
        execution_engine=execution_engine,
        evaluator=evaluator,
        explanation_service=explanation_service,
        log_service=log_service,
        mode_classifier=mode_classifier,
        cognitive_engine=cognitive_engine,
        context_summarizer=context_summarizer,
        query_cache=query_cache,
        max_retry_count=settings.max_retry_count,
    )

    logger.info("All services initialized (dual-mode analyst system)")
    logger.info("Using LLM model: %s via %s", settings.llm_model, settings.llm_base_url)
    yield

    if hasattr(app.state, "llm_client"):
        await app.state.llm_client.close()


app = FastAPI(title="Lumiq API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


@app.get("/health")
async def health(request: Request):
    checks = {}
    supabase = getattr(request.app.state, "supabase_client", None)

    if supabase:
        try:
            supabase.table("users").select("id").limit(1).execute()
            checks["supabase"] = "ok"
        except Exception as e:
            checks["supabase"] = f"error: {str(e)[:50]}"
    else:
        checks["supabase"] = "not configured"

    try:
        rag = getattr(request.app.state, "rag_service", None)
        if rag:
            rag._ensure_initialized()
            checks["qdrant"] = "ok"
        else:
            checks["qdrant"] = "not initialized"
    except Exception as e:
        checks["qdrant"] = f"error: {str(e)[:50]}"

    upload_dir = settings.upload_dir
    if os.path.exists(upload_dir) and os.access(upload_dir, os.W_OK):
        checks["filesystem"] = "ok"
    else:
        checks["filesystem"] = "error: upload dir not writable"

    # Cache stats
    cache = getattr(request.app.state, "orchestrator", None)
    if cache and hasattr(cache, "query_cache"):
        checks["query_cache"] = f"{len(cache.query_cache._cache)} entries"

    all_ok = all(v == "ok" for v in checks.values() if isinstance(v, str) and not v.startswith("query_cache"))
    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
        "version": "2.0.0",
    }
