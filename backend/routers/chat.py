from fastapi import APIRouter, Depends, HTTPException, Request
from models.schemas import (
    ChatCreateRequest,
    ChatResponse,
    MessageResponse,
    QueryRequest,
    QueryResponse,
    DatasetResponse,
)
from routers.auth_middleware import get_current_user
import re

router = APIRouter()

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def validate_uuid(value: str, field: str = "id"):
    if not UUID_PATTERN.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {field} format")


@router.post("/chats", response_model=ChatResponse)
async def create_chat(
    body: ChatCreateRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    chat_manager = request.app.state.chat_manager
    chat = chat_manager.create_chat(
        user_id=current_user["id"],
        email=current_user["email"],
        title=body.title,
    )
    return ChatResponse(**chat)


@router.get("/chats", response_model=list[ChatResponse])
async def list_chats(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    chat_manager = request.app.state.chat_manager
    chats = chat_manager.get_chats_by_user(user_id=current_user["id"])
    return [ChatResponse(**c) for c in chats]


@router.get("/chat/{chat_id}")
async def get_chat(
    chat_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    validate_uuid(chat_id, "chat_id")
    chat_manager = request.app.state.chat_manager
    dataset_manager = request.app.state.dataset_manager
    supabase = request.app.state.supabase_client

    try:
        chat = chat_manager.get_chat_by_id(
            chat_id=chat_id, user_id=current_user["id"]
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages = chat_manager.get_messages_by_chat(chat_id=chat_id)

    dataset = None
    try:
        # Dataset presence is determined by querying the dataset table directly;
        # chat.dataset_id was removed from the schema (circular dependency).
        dataset_data = dataset_manager.get_dataset_by_chat_id(chat_id)
        dataset = DatasetResponse(
            id=dataset_data["id"],
            chat_id=chat_id,
            filename=dataset_data["filename"],
            row_count=dataset_data["row_count"],
            column_count=dataset_data["column_count"],
            schema_info=dataset_data["schema_info"],
            sample_rows=dataset_data["sample_rows"],
            uploaded_at=dataset_data.get("uploaded_at", ""),
        )
    except ValueError:
        # No dataset for this chat — normal for new chats
        pass
    except Exception:
        # Dataset load failed but chat is still valid — degrade gracefully
        pass

    return {
        "chat": ChatResponse(**chat),
        "messages": [MessageResponse(**m) for m in messages],
        "dataset": dataset,
    }


@router.post("/chat", response_model=QueryResponse)
async def send_query(
    body: QueryRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    validate_uuid(body.chat_id, "chat_id")

    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > 1000:
        raise HTTPException(
            status_code=400, detail="Query too long (max 1000 characters)"
        )

    chat_manager = request.app.state.chat_manager
    orchestrator = request.app.state.orchestrator

    try:
        chat_manager.get_chat_by_id(
            chat_id=body.chat_id, user_id=current_user["id"]
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Chat not found")

    try:
        result = await orchestrator.handle_query(
            chat_id=body.chat_id,
            user_id=current_user["id"],
            query=query,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")
