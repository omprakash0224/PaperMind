"""
Upload API — protected by Clerk JWT via the AuthUser dependency.

user_id is Clerk's userId string (e.g. "user_2NkXyz..."),
stamped into every Qdrant chunk for tenant isolation.

Ingestion status is persisted via status_store (Upstash Redis in production,
in-memory dict in local dev) so job state survives server restarts and
works correctly with multiple workers.
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
from app.services.status_store import set_status, get_status, update_status

logger   = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS  = {".pdf", ".docx"}
MAX_FILE_SIZE_MB    = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def _run_ingestion(
    file_path:   str,
    filename:    str,
    document_id: str,
    user_id:     str,
) -> None:
    """
    Background task: parse, chunk, embed, and store a document.

    Writes status updates to the status store at each stage so the frontend
    polling /status/{document_id} sees live progress.

    Stages:
      queued      → set at upload time (before this function is called)
      processing  → set immediately when this function starts
      completed   → set on success (includes chunks_count)
      duplicate   → set when deduplication detects a previously ingested file
      failed      → set on any exception (includes error message)
    """
    update_status(document_id, {"status": "processing"})

    try:
        result = ingest_document(file_path, filename, user_id=user_id)

        update_status(document_id, {
            "status":       result["status"],       # "completed" or "duplicate"
            "chunks_count": result["chunks_count"],
            "document_id":  result["document_id"],
        })

        logger.info(
            "Ingestion done | user_id=%s | file='%s' | status=%s",
            user_id, filename, result["status"],
        )

    except Exception as exc:
        logger.exception(
            "Ingestion failed | user_id=%s | file='%s': %s", user_id, filename, exc
        )
        update_status(document_id, {
            "status": "failed",
            "error":  str(exc),
        })


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
    """
    Accept a file upload, save it to disk (or Cloudinary), queue ingestion,
    and return immediately with HTTP 202. The client should poll
    GET /api/documents/status/{document_id} to track progress.
    """
    original_filename = file.filename or "upload"
    suffix = Path(original_filename).suffix.lower()

    # ── Validate file type ────────────────────────────────────────────────────
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # ── Read and validate file content ────────────────────────────────────────
    content = await file.read()

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB.",
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # ── Build safe filename ───────────────────────────────────────────────────
    safe_stem     = Path(original_filename).stem[:64]
    document_id   = str(uuid.uuid4())
    safe_filename = f"{safe_stem}_{document_id[:8]}{suffix}"

    # ── Save to disk ──────────────────────────────────────────────────────────
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / safe_filename

    async with aiofiles.open(dest_path, "wb") as out_file:
        await out_file.write(content)

    # ── Optionally upload to Cloudinary ───────────────────────────────────────
    storage_path = upload_file(str(dest_path), original_filename)

    # If Cloudinary is enabled, the local copy is no longer needed
    if is_cloud_storage_enabled():
        try:
            dest_path.unlink()
        except OSError:
            pass

    logger.info(
        "Upload saved | user_id=%s | file='%s' | bytes=%d",
        current_user.user_id, original_filename, len(content),
    )

    # ── Write initial status record ───────────────────────────────────────────
    # Written to Redis (prod) or in-memory dict (dev) via status_store.
    # Background task will call update_status() as it progresses.
    set_status(document_id, {
        "document_id":   document_id,
        "filename":      original_filename,
        "safe_filename": safe_filename,
        "status":        "queued",
        "chunks_count":  0,
        "user_id":       current_user.user_id,
    })

    # ── Queue background ingestion ────────────────────────────────────────────
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


@router.get(
    "/status/{document_id}",
    summary="Poll ingestion status",
)
async def get_ingestion_status(document_id: str, current_user: AuthUser) -> dict:
    """
    Return the current ingestion status for a document.

    Reads from Redis (prod) or in-memory dict (dev) via status_store.
    Returns 404 for both missing records and records belonging to other
    users — prevents leaking the existence of other users' jobs.

    Possible status values:
      queued      → upload accepted, ingestion not yet started
      processing  → parsing, chunking, embedding in progress
      completed   → all chunks stored in Qdrant successfully
      duplicate   → file was already ingested; existing chunks returned
      failed      → ingestion failed; 'error' field contains the reason
    """
    record = get_status(document_id)

    # Return 404 for missing records AND wrong-owner records.
    # Never reveal whether a document_id exists for another user.
    if not record or record.get("user_id") != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No upload record found for document_id '{document_id}'.",
        )

    return record


@router.get(
    "",
    response_model=list[DocumentInfo],
    summary="List your documents",
)
async def get_documents(current_user: AuthUser) -> list[DocumentInfo]:
    """
    Return all documents ingested by the current user.
    Reads from Qdrant — not from the status store.
    The status store only tracks in-flight jobs; Qdrant is the source of truth
    for completed ingestions.
    """
    docs = list_documents(user_id=current_user.user_id)
    return [
        DocumentInfo(
            filename=d["filename"],
            document_id=d["document_id"],
            chunks_count=d["chunks_count"],
        )
        for d in docs
    ]


@router.delete(
    "/{filename:path}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document",
)
async def delete_document_endpoint(filename: str, current_user: AuthUser) -> dict:
    """
    Delete all Qdrant chunks for a document and remove the local file if present.
    Tenant isolation is enforced — users can only delete their own documents.
    """
    deleted_chunks = delete_document(filename, user_id=current_user.user_id)

    if deleted_chunks == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found.",
        )

    # Clean up local upload file if it exists
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