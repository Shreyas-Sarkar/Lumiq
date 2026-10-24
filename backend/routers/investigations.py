import csv
import io
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import Response
from models.investigations import InvestigationRequest, ResumeRequest
from routers.auth_middleware import get_current_user
from routers.chat import validate_uuid
from services.investigation import TERMINAL, grounded_findings

router = APIRouter()


def service(request):
    instance = getattr(request.app.state, "investigations", None)
    if not instance:
        raise HTTPException(
            503,
            "Investigation workspace is not configured. Apply the database migration and enable INVESTIGATIONS_ENABLED.",
        )
    return instance


def owned_chat(request, chat_id, user):
    validate_uuid(chat_id, "chat_id")
    try:
        request.app.state.chat_manager.get_chat_by_id(chat_id, user["id"])
    except ValueError:
        raise HTTPException(404, "Workspace not found")


def public(run):
    result = {
        k: v
        for k, v in run.items()
        if k not in ("lease_token", "lease_until", "user_id")
    }
    if "state" in result:
        state = {
            k: v
            for k, v in result["state"].items()
            if k not in ("interpretation_key", "interpretation_payload")
        }
        if state.get("status") == "complete" or state.get("findings"):
            state["findings"] = grounded_findings(
                state.get("evidence", []), state.get("findings", [])
            )
        result["state"] = state
    return result


@router.get("/datasets/{chat_id}/profile")
async def dataset_profile(
    chat_id: str, request: Request, user=Depends(get_current_user)
):
    owned_chat(request, chat_id, user)
    manager = request.app.state.dataset_manager
    try:
        data = manager.get_dataset_by_chat_id(chat_id, user["id"])
        if data.get("profile"):
            return data["profile"]
        from services.analysis_tools import profile_dataset

        return profile_dataset(
            manager.load_dataframe_from_storage(data["storage_path"], owner=user["id"])
        )
    except FileNotFoundError:
        raise HTTPException(
            410,
            "This legacy CSV is missing. Create a new workspace and upload it again.",
        )
    except ValueError as error:
        raise HTTPException(400, str(error))


@router.post("/investigations", status_code=202)
async def create(
    body: InvestigationRequest,
    request: Request,
    tasks: BackgroundTasks,
    user=Depends(get_current_user),
):
    owned_chat(request, body.chat_id, user)
    svc = service(request)
    try:
        run = svc.create(user["id"], body)
    except ValueError as error:
        raise HTTPException(400, str(error))
    if run["state"]["status"] not in TERMINAL:
        tasks.add_task(svc.schedule, run["id"], user["id"])
    return public(run)


@router.get("/investigations")
async def list_runs(chat_id: str, request: Request, user=Depends(get_current_user)):
    owned_chat(request, chat_id, user)
    return [public(r) for r in service(request).store.list(chat_id, user["id"])]


@router.get("/investigations/{run_id}")
async def get_run(
    run_id: str,
    request: Request,
    tasks: BackgroundTasks,
    user=Depends(get_current_user),
):
    validate_uuid(run_id, "run_id")
    svc = service(request)
    run = svc.store.get(run_id, user["id"])
    if not run:
        raise HTTPException(404, "Investigation not found")
    if run["state"]["status"] not in TERMINAL:
        tasks.add_task(svc.schedule, run_id, user["id"])
    return public(run)


@router.post("/investigations/{run_id}/resume", status_code=202)
async def resume(
    run_id: str,
    body: ResumeRequest,
    request: Request,
    tasks: BackgroundTasks,
    user=Depends(get_current_user),
):
    validate_uuid(run_id, "run_id")
    svc = service(request)
    if not svc.store.get(run_id, user["id"]):
        raise HTTPException(404, "Investigation not found")
    try:
        run = svc.resume(run_id, user["id"], body)
    except ValueError as error:
        raise HTTPException(409, str(error))
    tasks.add_task(svc.schedule, run_id, user["id"])
    return public(run)


@router.post("/investigations/{run_id}/cancel")
async def cancel(run_id: str, request: Request, user=Depends(get_current_user)):
    validate_uuid(run_id, "run_id")
    svc = service(request)
    run = svc.store.get(run_id, user["id"])
    if not run:
        raise HTTPException(404, "Investigation not found")
    if run["state"]["status"] == "complete":
        raise HTTPException(409, "Completed results are immutable")
    claimed = svc.store.claim(run_id, user["id"])
    if not claimed:
        raise HTTPException(409, "A calculation is active; cancel after it checkpoints")
    state = claimed["state"]
    if state["status"] == "complete":
        svc.store.checkpoint(claimed, state, True)
        raise HTTPException(409, "Completed results are immutable")
    state["status"] = "cancelled"
    return public(svc.store.checkpoint(claimed, state, True))


@router.get("/investigations/{run_id}/evidence/{evidence_id}/download")
async def download(
    run_id: str, evidence_id: str, request: Request, user=Depends(get_current_user)
):
    validate_uuid(run_id, "run_id")
    run = service(request).store.get(run_id, user["id"])
    if not run:
        raise HTTPException(404, "Investigation not found")
    evidence = next(
        (e for e in run["state"].get("evidence", []) if e["id"] == evidence_id), None
    )
    if not evidence:
        raise HTTPException(404, "Evidence not found")
    rows = evidence["rows"]
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        # Prevent spreadsheet formula execution when opening exported labels.
        writer.writerows(
            {
                k: (
                    "'" + v
                    if isinstance(v, str)
                    and v.startswith(("=", "+", "-", "@", "\t", "\r"))
                    else v
                )
                for k, v in row.items()
            }
            for row in rows
        )
    return Response(
        output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="evidence-{evidence_id}.csv"'
        },
    )
