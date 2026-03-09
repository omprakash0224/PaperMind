"""
Upload API — protected by Clerk JWT via the AuthUser dependency.
user_id is Clerk's userId string (e.g. "user_2NkXyz..."),
stamped into every Qdrant chunk for tenant isolation.
"""

import logging
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, status

from app.config import get_settings
from app.dependencies import AuthUser
from app.models.schemas import DocumentInfo, UploadResponse
from app.services.ingestion import ingest_document
from app.services.vectorstore import delete_document, list_documents
from app.services.storage import upload_file, is_cloud_storage_enabled

logger   = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS  = {".pdf", ".docx"}
MAX_FILE_SIZE_MB    = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

_ingestion_status: dict[str, dict] = {}


def _run_ingestion(
    file_path:   str,
    filename:    str,
    document_id: str,
    user_id:     str,   # Clerk userId string
) -> None:
    _ingestion_status[document_id]["status"] = "processing"
    try:
        result = ingest_document(file_path, filename, user_id=user_id)
        _ingestion_status[document_id].update({
            "status":       result["status"],
            "chunks_count": result["chunks_count"],
            "document_id":  result["document_id"],
        })
        logger.info(
            "Ingestion done | user_id=%s | file='%s' | status=%s",
            user_id, filename, result["status"],
        )
    except Exception as exc:
        logger.exception("Ingestion failed | user_id=%s | file='%s': %s", user_id, filename, exc)
        _ingestion_status[document_id].update({"status": "failed", "error": str(exc)})


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF or DOCX document for ingestion",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    current_user:     AuthUser,
    file: UploadFile = File(..., description="PDF or DOCX file (max 50 MB)"),
) -> UploadResponse:
    original_filename = file.filename or "upload"
    suffix = Path(original_filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    safe_stem     = Path(original_filename).stem[:64]
    document_id   = str(uuid.uuid4())
    safe_filename = f"{safe_stem}_{document_id[:8]}{suffix}"

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / safe_filename

    async with aiofiles.open(dest_path, "wb") as out_file:
        await out_file.write(content)

    storage_path = upload_file(str(dest_path), original_filename)

    if is_cloud_storage_enabled():
        try:
            dest_path.unlink()
        except OSError:
            pass

    logger.info(
        "Upload saved | user_id=%s | file='%s' | bytes=%d",
        current_user.user_id, original_filename, len(content),
    )

    _ingestion_status[document_id] = {
        "document_id":   document_id,
        "filename":      original_filename,
        "safe_filename": safe_filename,
        "status":        "queued",
        "chunks_count":  0,
        "user_id":       current_user.user_id,
    }

    background_tasks.add_task(
        _run_ingestion,
        file_path=storage_path,
        filename=original_filename,
        document_id=document_id,
        user_id=current_user.user_id,
    )

    return UploadResponse(
        document_id=document_id,
        filename=original_filename,
        chunks_count=0,
        status="queued",
    )


@router.get("/status/{document_id}", summary="Poll ingestion status")
async def get_ingestion_status(document_id: str, current_user: AuthUser) -> dict:
    record = _ingestion_status.get(document_id)
    # Return 404 for both missing and wrong-owner records (no leaking existence)
    if not record or record.get("user_id") != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No upload record found for document_id '{document_id}'.",
        )
    return record


@router.get("", response_model=list[DocumentInfo], summary="List your documents")
async def get_documents(current_user: AuthUser) -> list[DocumentInfo]:
    docs = list_documents(user_id=current_user.user_id)
    return [
        DocumentInfo(
            filename=d["filename"],
            document_id=d["document_id"],
            chunks_count=d["chunks_count"],
        )
        for d in docs
    ]


@router.delete("/{filename:path}", status_code=status.HTTP_200_OK, summary="Delete a document")
async def delete_document_endpoint(filename: str, current_user: AuthUser) -> dict:
    deleted_chunks = delete_document(filename, user_id=current_user.user_id)

    if deleted_chunks == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found.",
        )

    upload_dir    = Path(settings.UPLOAD_DIR)
    deleted_files: list[str] = []

    if upload_dir.exists():
        stem = Path(filename).stem[:64]
        for f in upload_dir.iterdir():
            if f.is_file() and f.name.startswith(stem):
                try:
                    f.unlink()
                    deleted_files.append(f.name)
                except OSError as exc:
                    logger.warning("Could not delete '%s': %s", f, exc)

    logger.info(
        "Document deleted | user_id=%s | file='%s' | chunks=%d",
        current_user.user_id, filename, deleted_chunks,
    )

    return {
        "filename":       filename,
        "deleted_chunks": deleted_chunks,
        "deleted_files":  deleted_files,
        "status":         "deleted",
    }