"""
queue.py -- ARQ job enqueueing for the ingestion pipeline.

ARQ uses a connection pool (ArqRedis) built on top of aioredis.
Enqueueing is async-native -- works perfectly inside FastAPI endpoints
without threading or sync wrappers.

The worker (app/worker.py) defines the same function name that is used
here as the job name, forming the contract between producer and consumer.
"""

import logging
from typing import Optional

import arq
from arq.connections import RedisSettings

from app.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

# Module-level pool -- initialised once on first call, reused across requests.
_arq_pool: Optional[arq.ArqRedis] = None


def _redis_settings() -> RedisSettings:
    """Parse REDIS_URL into ARQ's RedisSettings object."""
    return RedisSettings.from_dsn(settings.REDIS_URL)


async def get_arq_pool() -> arq.ArqRedis:
    """
    Return a module-level ArqRedis connection pool.
    Created on first call and reused for the lifetime of the process.
    """
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await arq.create_pool(_redis_settings())
        logger.info("ARQ Redis pool created: %s", settings.REDIS_URL)
    return _arq_pool


async def enqueue_ingestion_job(
    *,
    document_id: str,
    file_path: str,
    filename: str,
    user_id: str,
) -> str:
    """
    Enqueue an ingestion job to the ARQ queue.

    Returns the ARQ job ID string.

    The function name "run_ingestion" must match the function defined
    in app/worker.py that the ARQ worker will call.
    """
    pool = await get_arq_pool()

    job = await pool.enqueue_job(
        "run_ingestion",
        document_id=document_id,
        file_path=file_path,
        filename=filename,
        user_id=user_id,
    )

    job_id = job.job_id if job else "unknown"
    logger.info(
        "Enqueued ingestion job | job_id=%s | document_id=%s | user_id=%s",
        job_id, document_id, user_id,
    )
    return job_id
