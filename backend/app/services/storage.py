import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def is_cloud_storage_enabled() -> bool:
    """Returns True if all Cloudinary env vars are set."""
    from app.config import get_settings
    s = get_settings()
    return bool(s.CLOUDINARY_CLOUD_NAME and s.CLOUDINARY_API_KEY and s.CLOUDINARY_API_SECRET)


def _get_cloudinary():
    """Configure and return cloudinary module."""
    import cloudinary
    import cloudinary.uploader
    from app.config import get_settings
    s = get_settings()
    cloudinary.config(
        cloud_name=s.CLOUDINARY_CLOUD_NAME,
        api_key=s.CLOUDINARY_API_KEY,
        api_secret=s.CLOUDINARY_API_SECRET,
    )
    return cloudinary


def upload_file(local_path: str, filename: str) -> str:
    """
    Upload file to Cloudinary (prod) or keep local path (dev).
    Returns the path/URL to use for ingestion.
    """
    if not is_cloud_storage_enabled():
        logger.info("Cloud storage disabled — using local path: %s", local_path)
        return local_path

    import cloudinary.uploader
    _get_cloudinary()

    logger.info("Uploading '%s' to Cloudinary...", filename)
    result = cloudinary.uploader.upload(
        local_path,
        resource_type="raw",          # required for PDF/DOCX (non-image)
        public_id=f"rag_uploads/{Path(local_path).stem}",
        overwrite=False,
        use_filename=True,
    )
    url = result["secure_url"]
    logger.info("Uploaded to Cloudinary: %s", url)
    return url


def download_for_processing(file_url_or_path: str, suffix: str) -> str:
    """
    If file_url_or_path is a remote URL, download to a temp file and return
    the temp path. Otherwise return as-is (already a local path).
    Used so ingestion.py always works with a local file path.
    """
    if file_url_or_path.startswith("http"):
        import httpx
        logger.info("Downloading file from Cloudinary for processing...")

        # If Cloudinary is configured, generate a signed URL for authenticated access
        download_url = file_url_or_path
        if is_cloud_storage_enabled():
            try:
                import cloudinary.utils
                _get_cloudinary()
                # Extract the public_id from the URL
                # URL format: .../raw/upload/v.../rag_uploads/filename.ext
                parts = file_url_or_path.split("/raw/upload/")
                if len(parts) == 2:
                    # Remove version prefix (v1234567890/) to get public_id.ext
                    path_after_upload = parts[1]
                    # Remove version segment if present
                    segments = path_after_upload.split("/", 1)
                    if len(segments) == 2 and segments[0].startswith("v"):
                        resource_path = segments[1]
                    else:
                        resource_path = path_after_upload
                    # Remove file extension for public_id
                    public_id = str(Path(resource_path).with_suffix(""))
                    signed_url, _ = cloudinary.utils.cloudinary_url(
                        public_id,
                        resource_type="raw",
                        sign_url=True,
                        type="upload",
                    )
                    if signed_url:
                        download_url = signed_url
                        logger.info("Using signed Cloudinary URL for download.")
            except Exception as exc:
                logger.warning("Could not generate signed URL, using original: %s", exc)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            response = httpx.get(download_url)
            response.raise_for_status()
            tmp.write(response.content)
            tmp_path = tmp.name
        logger.info("Downloaded to temp file: %s", tmp_path)
        return tmp_path
    return file_url_or_path


def delete_file(filename: str) -> None:
    """Delete file from Cloudinary (prod) or local disk (dev)."""
    if not is_cloud_storage_enabled():
        return  # local file cleanup is handled in upload.py already

    import cloudinary.uploader
    _get_cloudinary()

    stem = Path(filename).stem[:64]
    public_id = f"rag_uploads/{stem}"
    try:
        cloudinary.uploader.destroy(public_id, resource_type="raw")
        logger.info("Deleted from Cloudinary: %s", public_id)
    except Exception as exc:
        logger.warning("Could not delete '%s' from Cloudinary: %s", public_id, exc)