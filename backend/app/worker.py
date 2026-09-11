"""
worker.py -- ARQ worker definition for PaperMind's ingestion pipeline.

Start the worker:
  arq app.worker.WorkerSettings

Or in Docker / production:
  python -m arq app.worker.WorkerSettings

ARQ will:
  - Connect to Redis using REDIS_URL
  - Pick up jobs enqueued by queue.enqueue_ingestion_job()
  - Call run_ingestion() for each job
  - Retry up to ARQ_MAX_TRIES times with exponential backoff on failure
  - Keep the job result in Redis for keep_result seconds

All ingestion code (parse, chunk, embed, upsert) runs directly in this
process -- no subprocess, no Node.js. The worker is a standard Python
process and can be scaled horizontally by running multiple instances.
"""

import logging

from arq.connections import RedisSettings

from app.config import get_settings
from app.services.status_store import update_status
from app.services.ingestion import ingest_document

logger   = logging.getLogger(__name__)
settings = get_settings()


async def run_ingestion(
    ctx: dict,
    *,
    document_id: str,
    file_path: str,
    filename: str,
    user_id: str,
) -> dict:
    """
    ARQ job function -- called by the worker for each ingestion job.

    ctx: ARQ context dict (contains 'job_id', 'job_try', 'redis', etc.).
    All other args are the job payload set in queue.enqueue_ingestion_job().

    Returns the result dict which ARQ stores in Redis for inspection.
    Raises on failure so ARQ retries the job automatically.
    """
    job_id  = ctx.get("job_id", "unknown")
    attempt = ctx.get("job_try", 1)

    logger.info(
        "Job started | job_id=%s | attempt=%d | document_id=%s | file=%s",
        job_id, attempt, document_id, filename,
    )

    update_status(document_id, {"status": "processing"})

    try:
        result = ingest_document(file_path, filename, user_id=user_id)

        update_status(document_id, {
            "status":       result["status"],
            "chunks_count": result["chunks_count"],
            "document_id":  result["document_id"],
        })

        logger.info(
            "Job done | job_id=%s | status=%s | chunks=%d",
            job_id, result["status"], result["chunks_count"],
        )
        return result

    except Exception as exc:
        logger.exception("Job failed | job_id=%s | attempt=%d: %s", job_id, attempt, exc)
        update_status(document_id, {"status": "failed", "error": str(exc)})
        raise  # re-raise so ARQ marks the job as failed and schedules a retry


async def startup(ctx: dict) -> None:
    """Called once when the worker process starts."""
    logger.info("ARQ worker starting up...")
    # Pre-warm Qdrant collection so the first job doesn't pay startup cost
    try:
        from app.services.vectorstore import ensure_collection
        ensure_collection()
        logger.info("Qdrant collection ready.")
    except Exception as exc:
        logger.warning("Qdrant warm-up failed: %s", exc)


async def shutdown(ctx: dict) -> None:
    """Called once when the worker process shuts down."""
    logger.info("ARQ worker shutting down.")


class WorkerSettings:
    """
    ARQ worker configuration.

    To start: arq app.worker.WorkerSettings
    """
    functions   = [run_ingestion]  # job functions this worker handles
    on_startup  = startup
    on_shutdown = shutdown

    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)

    max_jobs          = settings.ARQ_MAX_JOBS     # concurrent jobs (default 2)
    job_timeout       = settings.ARQ_JOB_TIMEOUT  # seconds before job is killed (default 600)
    max_tries         = settings.ARQ_MAX_TRIES    # retry attempts (default 5)
    keep_result       = 3600                      # keep result in Redis for 1 hour
    retry_jobs        = True
    # Exponential backoff: 15s, 30s, 60s, 120s, 240s between retries
    health_check_interval = 30
