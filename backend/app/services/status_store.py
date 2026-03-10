"""
Status store for document ingestion jobs.

Abstracts the storage backend so the rest of the codebase doesn't care
whether it's Redis or in-memory. Two backends:

  Production  → Upstash Redis (UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN set)
  Development → In-memory dict (no env vars needed, zero setup)

Redis key schema:
  ingestion:{document_id}  →  JSON-serialised status dict

TTL:
  24 hours — entries auto-expire after completion.
  No manual cleanup job needed. If a user polls after TTL expiry they get
  a 404, which is correct (the job is long gone).

Why Upstash over self-hosted Redis:
  • Serverless — no idle cost, pay per request
  • REST-based — works from any environment including serverless functions
  • Free tier covers typical RAG app usage comfortably
  • No TCP connection management — each call is a stateless HTTP request
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# TTL for each job status record in Redis
_JOB_TTL_SECONDS = 60 * 60 * 24  # 24 hours

# ── In-memory fallback (local dev only) ──────────────────────────────────────
# Used automatically when UPSTASH_REDIS_REST_URL is not set.
# Not suitable for production — does not survive restarts or scale horizontally.
_local_store: dict[str, dict] = {}


# ── Redis key helper ──────────────────────────────────────────────────────────

def _key(document_id: str) -> str:
    """Namespace all keys under 'ingestion:' to avoid collisions."""
    return f"ingestion:{document_id}"


# ── Redis client factory ──────────────────────────────────────────────────────

def _get_redis():
    """
    Return a configured Upstash Redis client, or None if not configured.

    The client is created fresh each call — Upstash uses HTTP/REST so there
    is no persistent connection to manage or pool. This is intentional.

    Returns None in local dev (UPSTASH_REDIS_REST_URL not set) so callers
    fall back to the in-memory store transparently.
    """
    from app.config import get_settings
    s = get_settings()

    if not s.use_redis:
        return None

    try:
        from upstash_redis import Redis
        return Redis(
            url=s.UPSTASH_REDIS_REST_URL,
            token=s.UPSTASH_REDIS_REST_TOKEN,
        )
    except ImportError:
        logger.error(
            "upstash-redis is not installed. "
            "Run: pip install upstash-redis"
        )
        raise


# ── Public API ────────────────────────────────────────────────────────────────

def set_status(document_id: str, data: dict[str, Any]) -> None:
    """
    Write (or overwrite) the full status record for a document ingestion job.

    Called once at upload time to create the initial 'queued' record,
    then again by update_status() whenever the job progresses.

    Args:
        document_id: UUID assigned to the upload — used as the Redis key suffix.
        data:        Full status dict. Must include at minimum:
                       document_id, filename, safe_filename, status,
                       chunks_count, user_id
    """
    redis = _get_redis()

    if redis:
        try:
            redis.set(_key(document_id), json.dumps(data), ex=_JOB_TTL_SECONDS)
            logger.debug(
                "Status written to Redis | document_id=%s | status=%s",
                document_id, data.get("status"),
            )
        except Exception as exc:
            # Log but don't crash — a failed status write shouldn't fail the upload
            logger.error(
                "Redis set_status failed for document_id=%s: %s", document_id, exc
            )
    else:
        _local_store[document_id] = data
        logger.debug(
            "Status written to memory | document_id=%s | status=%s",
            document_id, data.get("status"),
        )


def get_status(document_id: str) -> dict | None:
    """
    Return the status record for a document, or None if not found / expired.

    Called on every poll from the frontend (every 2 seconds until complete).
    Redis GET is O(1) — no performance concern for frequent polling.

    Returns None for:
      • document_id that was never registered
      • document_id whose TTL has expired (job completed > 24 hours ago)
      • document_id belonging to a different user (caller checks user_id)
    """
    redis = _get_redis()

    if redis:
        try:
            raw = redis.get(_key(document_id))
            if raw is None:
                return None
            # Upstash returns a string — parse it back to dict
            return json.loads(raw) if isinstance(raw, str) else raw
        except Exception as exc:
            logger.error(
                "Redis get_status failed for document_id=%s: %s", document_id, exc
            )
            return None
    else:
        return _local_store.get(document_id)


def update_status(document_id: str, updates: dict[str, Any]) -> None:
    """
    Merge `updates` into an existing status record and re-save it.

    Safe to call from background tasks — reads the current record first
    so no fields are accidentally lost (e.g. user_id, filename).

    Silently no-ops if the record doesn't exist (e.g. TTL expired between
    job start and completion — extremely unlikely but handled gracefully).

    Args:
        document_id: UUID of the upload to update.
        updates:     Partial dict to merge. Common usage:
                       {"status": "processing"}
                       {"status": "completed", "chunks_count": 42}
                       {"status": "failed", "error": "..."}
    """
    existing = get_status(document_id)

    if existing is None:
        logger.warning(
            "update_status: no record found for document_id=%s "
            "(may have expired or never been set).",
            document_id,
        )
        return

    existing.update(updates)
    set_status(document_id, existing)

    logger.debug(
        "Status updated | document_id=%s | updates=%s",
        document_id, updates,
    )


def delete_status(document_id: str) -> None:
    """
    Explicitly delete a status record before its TTL expires.

    Optional — records expire automatically after 24 hours.
    Useful if you want to clean up immediately after a user deletes a document.
    """
    redis = _get_redis()

    if redis:
        try:
            redis.delete(_key(document_id))
            logger.debug("Status deleted from Redis | document_id=%s", document_id)
        except Exception as exc:
            logger.warning(
                "Redis delete_status failed for document_id=%s: %s", document_id, exc
            )
    else:
        _local_store.pop(document_id, None)
        logger.debug("Status deleted from memory | document_id=%s", document_id)