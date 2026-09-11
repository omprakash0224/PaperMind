"""
storage.py -- Cloudflare R2 storage backend (replaces Cloudinary).

Upload backend:
  Production  -> Cloudflare R2 (R2_* env vars set)
  Development -> Local disk (no env vars needed)

R2 is S3-compatible; we use boto3 with a custom endpoint_url.
Zero egress fees mean downloading for processing is always free.
"""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_KEY_PREFIX = "rag-uploads"


def is_cloud_storage_enabled() -> bool:
    """Returns True if all R2 env vars are set."""
    from app.config import get_settings
    return get_settings().use_r2


def _get_r2_client():
    """
    Build and return a boto3 S3 client configured for Cloudflare R2.

    R2 is S3-compatible but uses a custom endpoint_url:
      https://{account_id}.r2.cloudflarestorage.com

    boto3 is used because:
      - Standard S3 API -- no vendor lock-in
      - Pre-signed URLs are one-liners
      - Multipart upload available if needed later
    """
    import boto3
    from app.config import get_settings
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.R2_ENDPOINT_URL,
        aws_access_key_id=s.R2_ACCESS_KEY_ID,
        aws_secret_access_key=s.R2_SECRET_ACCESS_KEY,
        region_name="auto",  # R2 ignores region but boto3 requires it
    )


def _r2_key(user_id: str, document_id: str, filename: str) -> str:
    """
    Build an R2 object key scoped by user.

    Format: rag-uploads/{user_id}/{document_id}{.ext}

    Scoping by user_id means:
      - Files are naturally partitioned per tenant
      - Listing a user's files is a prefix scan
      - No accidental cross-tenant access via guessable filenames
    """
    suffix = Path(filename).suffix.lower()
    return f"{_KEY_PREFIX}/{user_id}/{document_id}{suffix}"


def _content_type(filename: str) -> str:
    """Return the correct MIME type for supported file types."""
    suffix = Path(filename).suffix.lower()
    return {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }.get(suffix, "application/octet-stream")


def upload_file(
    local_path: str,
    filename: str,
    *,
    user_id: str,
    document_id: str,
) -> str:
    """
    Upload file to Cloudflare R2 (prod) or keep local path (dev).
    Returns the R2 object key (prod) or the local path (dev).

    Note: signature adds `user_id` and `document_id` keyword-only args
    vs the old Cloudinary version. Callers (upload.py) already have both.
    """
    if not is_cloud_storage_enabled():
        logger.info("Cloud storage disabled -- using local path: %s", local_path)
        return local_path

    from app.config import get_settings
    s = get_settings()
    client = _get_r2_client()
    key = _r2_key(user_id, document_id, filename)

    logger.info("Uploading '%s' to R2 key '%s'...", filename, key)
    client.upload_file(
        Filename=local_path,
        Bucket=s.R2_BUCKET_NAME,
        Key=key,
        ExtraArgs={"ContentType": _content_type(filename)},
    )
    logger.info("Uploaded to R2: %s", key)
    return key  # return the object key; generate pre-signed URLs on demand


def generate_download_url(r2_key: str, expires_in: int = 3600) -> str:
    """
    Generate a pre-signed R2 URL valid for `expires_in` seconds (default 1 hour).
    Used by the ingestion worker to download the file for processing.

    Pre-signed URLs keep the bucket private while allowing time-limited access.
    """
    from app.config import get_settings
    s = get_settings()
    client = _get_r2_client()
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": s.R2_BUCKET_NAME, "Key": r2_key},
        ExpiresIn=expires_in,
    )
    logger.debug("Generated pre-signed R2 URL for key '%s' (expires=%ds).", r2_key, expires_in)
    return url


def download_for_processing(file_url_or_path: str, suffix: str) -> str:
    """
    Resolve a file reference to a local temp path for ingestion.

    Three modes:
      1. Local path (dev)         -> return as-is
      2. R2 object key (no http)  -> generate pre-signed URL, then download
      3. Direct HTTP URL          -> download directly

    This is the single entry point ingestion.py uses to get a local file,
    abstracting over all storage backends.
    """
    # Local file -- dev mode
    if not file_url_or_path.startswith("http") and os.path.exists(file_url_or_path):
        return file_url_or_path

    # R2 object key -- generate a pre-signed URL first
    download_url = file_url_or_path
    if not file_url_or_path.startswith("http"):
        logger.info("R2 key detected -- generating pre-signed URL: %s", file_url_or_path)
        download_url = generate_download_url(file_url_or_path)

    # Download from URL to a temp file
    import httpx
    logger.info("Downloading file for processing...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        response = httpx.get(download_url, timeout=120)
        response.raise_for_status()
        tmp.write(response.content)
        tmp_path = tmp.name
    logger.info("Downloaded to temp file: %s", tmp_path)
    return tmp_path


def delete_file(r2_key: str) -> None:
    """Delete an object from R2 (prod) or no-op (dev)."""
    if not is_cloud_storage_enabled():
        return  # local file cleanup is handled in upload.py

    from app.config import get_settings
    s = get_settings()
    client = _get_r2_client()
    try:
        client.delete_object(Bucket=s.R2_BUCKET_NAME, Key=r2_key)
        logger.info("Deleted from R2: %s", r2_key)
    except Exception as exc:
        logger.warning("Could not delete '%s' from R2: %s", r2_key, exc)
