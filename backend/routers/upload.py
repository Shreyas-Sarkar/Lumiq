import os
import uuid
import re
import hashlib
from services.analysis_tools import profile_dataset
from fastapi import APIRouter, File, Form, UploadFile, Depends, HTTPException, Request
from models.schemas import UploadResponse, DatasetResponse
from routers.auth_middleware import get_current_user

router = APIRouter()


def secure_filename(filename: str) -> str:
    filename = re.sub(r"[^\w\s\-.]", "", filename)
    filename = re.sub(r"\s+", "_", filename)
    return filename[:255]


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    request: Request,
    chat_id: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    settings = request.app.state.settings
    dataset_manager = request.app.state.dataset_manager
    rag_service = request.app.state.rag_service
    supabase = request.app.state.supabase_client

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    max_file_size_mb = min(settings.max_file_size_mb, 5)
    content = await file.read(max_file_size_mb * 1024 * 1024 + 1)
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {max_file_size_mb}MB",
        )

    try:
        chat_result = (
            supabase.table("chat")
            .select("*")
            .eq("id", chat_id)
            .eq("user_id", current_user["id"])
            .limit(1)
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Chat not found")

    if not chat_result.data:
        raise HTTPException(status_code=400, detail="Chat not found or access denied")

    chat = chat_result.data[0]

    # Guard against uploading a second dataset to the same chat.
    # dataset_id was removed from the chat table — check the dataset table directly.
    existing_dataset = (
        supabase.table("dataset").select("id").eq("chat_id", chat_id).limit(1).execute()
    )
    if existing_dataset.data:
        raise HTTPException(status_code=409, detail="This chat already has a dataset")

    dataset_id = str(uuid.uuid4())
    user_upload_dir = os.path.join(settings.upload_dir, current_user["id"])
    os.makedirs(user_upload_dir, exist_ok=True)
    safe_name = secure_filename(file.filename)
    file_path = os.path.join(user_upload_dir, f"{dataset_id}.csv")

    with open(file_path, "wb") as f:
        f.write(content)

    is_valid, reason = dataset_manager.validate_csv(file_path)
    if not is_valid:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {reason}")

    try:
        df = dataset_manager.load_csv(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))

    schema_info = dataset_manager.extract_schema(df)
    sample_rows = dataset_manager.extract_sample_rows(df)
    summary_stats = dataset_manager.extract_summary_stats(df)

    profile = None
    storage_path = file_path
    if settings.investigations_enabled:
        try:
            profile = profile_dataset(df)
            storage_path = dataset_manager.store_csv(
                content, current_user["id"], dataset_id, settings.dataset_bucket
            )
        except ValueError as error:
            os.remove(file_path)
            raise HTTPException(400, str(error))
        except Exception:
            os.remove(file_path)
            raise HTTPException(
                503,
                "Durable storage is unavailable. No dataset was saved; retry after storage is restored.",
            )

    chroma_collection_id = dataset_id

    try:
        dataset_data = dataset_manager.save_dataset_metadata(
            chat_id=chat_id,
            user_id=current_user["id"],
            filename=safe_name,
            storage_path=storage_path,
            schema_info=schema_info,
            sample_rows=sample_rows,
            summary_stats=summary_stats,
            chroma_collection_id=chroma_collection_id,
            row_count=len(df),
            column_count=len(df.columns),
            dataset_id=dataset_id,
            profile=profile,
            content_hash=hashlib.sha256(content).hexdigest(),
        )
    except Exception:
        dataset_manager.remove_csv(storage_path)
        if storage_path != file_path:
            os.remove(file_path)
        raise HTTPException(
            status_code=503,
            detail="Could not save dataset metadata. It may already exist; reload the workspace before retrying.",
        )
    if storage_path != file_path:
        os.remove(file_path)

    try:
        if settings.investigations_enabled:
            raise RuntimeError("Vector indexing is unnecessary for trusted analysis")
        rag_service.index_dataset(
            collection_id=chroma_collection_id,
            schema=schema_info,
            samples=sample_rows,
            stats=summary_stats,
        )
    except Exception as e:
        pass

    return UploadResponse(
        dataset=DatasetResponse(
            id=dataset_data["id"],
            chat_id=chat_id,
            filename=safe_name,
            row_count=len(df),
            column_count=len(df.columns),
            schema_info=schema_info,
            sample_rows=sample_rows,
            uploaded_at=dataset_data.get("uploaded_at", ""),
        ),
        message="Dataset uploaded successfully",
    )
