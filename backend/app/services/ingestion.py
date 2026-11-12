import hashlib
import uuid
import logging
import random
import re
import time
from collections import deque
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    SparseVector as QdrantSparseVector,
)

from app.config import get_settings
from app.utils.parsers import parse_document
from app.services.vectorstore import (
    get_qdrant_client,
    get_sparse_encoder,
    ensure_collection,
    SPARSE_VECTOR_NAME,
)
from app.services.storage import download_for_processing
from app.services.status_store import update_status

logger   = logging.getLogger(__name__)
settings = get_settings()

# ── Token pacing ──────────────────────────────────────────────────────────────
# Gemini counts tokenizer tokens differently from tiktoken. Apply a small
# safety factor so the pacing window stays under the real TPM ceiling.
_TOKEN_SAFETY_FACTOR = 1.1

# Best-effort parsers for the retry delay hints Google puts in 429 messages:
#   "retryDelay': '23s'"  |  "Please retry in 23.4s"  |  Retry-After: 23
_RETRY_AFTER_PATTERNS = (
    r"""retryDelay["']?\s*[:=]\s*["']?(\d+(?:\.\d+)?)\s*s""",
    r"""retry in\s*(\d+(?:\.\d+)?)\s*s""",
    r"""Retry-After["']?\s*[:=]\s*["']?(\d+(?:\.\d+)?)""",
)


class _SlidingWindowLimiter:
    """
    Enforces a tokens-per-minute and requests-per-minute budget using a
    sliding 60-second window.

    Instead of sleeping a fixed interval after every batch, we only sleep
    when the next batch would actually exceed the configured limits. This
    maximises throughput while staying inside the free-tier ceiling.
    """

    def __init__(self, window: float = 60.0) -> None:
        self._window = window
        self._events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._requests: deque[float] = deque()

    def _evict(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= self._window:
            self._events.popleft()
        while self._requests and now - self._requests[0] >= self._window:
            self._requests.popleft()

    def acquire(self, tokens: int, *, tpm: int, rpm: int) -> None:
        """
        Block until a batch of `tokens` fits within both the TPM and RPM
        budgets, then record it.
        """
        tokens = max(1, tokens)

        while True:
            now = time.monotonic()
            self._evict(now)

            used_tokens = sum(t for _, t in self._events)
            waits: list[float] = []

            if used_tokens + tokens > tpm and self._events:
                waits.append(self._window - (now - self._events[0][0]))
            if len(self._requests) + 1 > rpm and self._requests:
                waits.append(self._window - (now - self._requests[0]))

            if not waits:
                break
            # Sleep until the oldest event that blocks us leaves the window.
            time.sleep(max(0.05, min(waits)))

        stamp = time.monotonic()
        self._events.append((stamp, tokens))
        self._requests.append(stamp)


def _parse_retry_after(exc: Exception) -> float | None:
    """Extract a retry delay (seconds) from a 429 error message, if present."""
    message = str(exc)
    for pattern in _RETRY_AFTER_PATTERNS:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _build_token_batches(
    docs: list[Document],
    *,
    max_tokens: int,
    max_texts: int,
) -> list[tuple[list[Document], int]]:
    """
    Group chunks into batches bounded by a token budget and a text cap.

    Returns a list of (batch_docs, estimated_tokens) tuples. A single chunk
    larger than `max_tokens` gets its own batch (the API will reject it, but
    we let the error surface rather than dropping content silently).
    """
    batches: list[tuple[list[Document], int]] = []
    current: list[Document] = []
    current_tokens = 0

    for doc in docs:
        doc_tokens = max(1, int(_tiktoken_length(doc.page_content) * _TOKEN_SAFETY_FACTOR))

        if current and (
            current_tokens + doc_tokens > max_tokens or len(current) >= max_texts
        ):
            batches.append((current, current_tokens))
            current, current_tokens = [], 0

        current.append(doc)
        current_tokens += doc_tokens

    if current:
        batches.append((current, current_tokens))

    return batches


def _tiktoken_length(text: str) -> int:
    """Token counter using cl100k_base (GPT-4 / text-embedding-3 compatible)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of file content for deduplication."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _doc_hash_exists(file_hash: str, user_id: str) -> dict | None:
    """
    Check if a document with this hash was already ingested by this user.
    Returns existing doc info dict or None.
    Scoped per user — same file uploaded by two users is stored twice (correct).
    """
    client = get_qdrant_client()

    existing_collections = [c.name for c in client.get_collections().collections]
    if settings.COLLECTION_NAME not in existing_collections:
        return None

    hash_filter = Filter(
        must=[
            FieldCondition(key="metadata.file_hash", match=MatchValue(value=file_hash)),
            FieldCondition(key="metadata.user_id",   match=MatchValue(value=user_id)),
        ]
    )

    results, _ = client.scroll(
        collection_name=settings.COLLECTION_NAME,
        scroll_filter=hash_filter,
        with_payload=True,
        with_vectors=False,
        limit=1,
    )

    if not results:
        return None

    meta: dict = results[0].payload.get("metadata", {})
    count = client.count(
        collection_name=settings.COLLECTION_NAME,
        count_filter=hash_filter,
        exact=True,
    ).count

    return {
        "document_id":  meta.get("document_id", "unknown"),
        "filename":     meta.get("source", "unknown"),
        "chunks_count": count,
        "status":       "duplicate",
    }


def _upsert_batch_with_retry(
    batch:      list[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
) -> None:
    """
    Embed a batch of chunks (dense + sparse) and upsert into Qdrant.

    Why we upsert directly instead of using vectorstore.add_documents():
      LangChain's QdrantVectorStore.add_documents() only stores the dense
      vector. To store BOTH dense and sparse vectors in a single point we
      must build PointStruct objects manually and call client.upsert().

    Dense  → Google Gemini via embed_documents()
             task_type="RETRIEVAL_DOCUMENT" is required for asymmetric retrieval
    Sparse → Pure-Python BM25 via passage_embed()
             Returns raw TF weights; Qdrant applies IDF server-side (Modifier.IDF)

    Retries with exponential backoff on Google API 429 / RESOURCE_EXHAUSTED errors,
    honouring the server's Retry-After / retryDelay hint when present.
    Non-rate-limit errors are re-raised immediately.
    """
    client = get_qdrant_client()
    sparse_encoder = get_sparse_encoder()
    texts  = [doc.page_content for doc in batch]

    for attempt in range(1, settings.EMBED_MAX_RETRIES + 1):
        try:
            # ── Dense embeddings (Google Gemini) ──────────────────────────────
            dense_vectors: list[list[float]] = embeddings.embed_documents(texts)

            # ── Sparse embeddings (Pure-Python BM25) ──────────────────────────
            # passage_embed() returns a list of SparseEmbedding objects.
            # Each has .indices (FNV-1a token hashes) and .values (TF counts).
            # BM25 is symmetric — passage_embed and query_embed are identical.
            sparse_results = sparse_encoder.passage_embed(texts)

            # ── Build PointStruct list ─────────────────────────────────────────
            points: list[PointStruct] = []
            for doc, dense_vec, sparse_res in zip(batch, dense_vectors, sparse_results):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            # "" is the default/unnamed dense vector in Qdrant
                            "": dense_vec,
                            # Named sparse vector — queried by SPARSE_VECTOR_NAME
                            SPARSE_VECTOR_NAME: QdrantSparseVector(
                                indices=sparse_res.indices,
                                values=sparse_res.values,
                            ),
                        },
                        payload={
                            # LangChain QdrantVectorStore expects these exact keys
                            "page_content": doc.page_content,
                            "metadata":     doc.metadata,
                        },
                    )
                )

            client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=points,
            )
            return  # success — exit retry loop

        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < settings.EMBED_MAX_RETRIES:
                delay = settings.EMBED_RETRY_BASE * (2 ** (attempt - 1))
                delay = min(delay, 60.0) * random.uniform(1.0, 1.5)

                retry_after = _parse_retry_after(e)
                if retry_after is not None:
                    delay = max(delay, retry_after + 1.0)

                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %.1fs...",
                    attempt, settings.EMBED_MAX_RETRIES, delay,
                )
                time.sleep(delay)
            else:
                raise


def ingest_document(
    file_path: str,
    filename: str,
    *,
    user_id: str,
    status_document_id: str | None = None,
) -> dict:
    """
    Full ingestion pipeline:
      1. Hash check (per user)  → return early if duplicate
      2. Parse PDF/DOCX         → page-level text blocks
      3. Chunk                  → RecursiveCharacterTextSplitter (tiktoken-based)
      4. Embed + store          → dense (Gemini) + sparse (BM25) upserted together
         — user_id stamped into every chunk for tenant isolation
         — chunks grouped into token-budgeted batches and paced to the
           configured TPM / RPM limits with a sliding-window limiter
         — retries with jittered exponential backoff on 429 errors
         — progress (chunks_done / chunks_total) written to the status store

    Args:
        file_path: Local path or Cloudflare R2 object key to the uploaded file.
        filename:  Original filename (used as the source metadata key).
        user_id:   Clerk userId — stamped into every chunk for tenant isolation.
        status_document_id:
            document_id of the queued job. When provided, batch progress is
            written to the status store under this key.

    Returns:
        dict with keys: document_id, filename, chunks_count, status
    """
    # ── 0. Download remote file to temp path if needed ────────────────────────
    suffix       = Path(filename).suffix.lower()
    local_path   = download_for_processing(file_path, suffix)
    cleanup_temp = (local_path != file_path)

    try:
        # ── 1. Deduplication check (scoped per user) ──────────────────────────
        file_hash = _compute_file_hash(local_path)
        existing  = _doc_hash_exists(file_hash, user_id)
        if existing:
            logger.info(
                "Duplicate detected for '%s' (hash=%s, user=%s). Skipping.",
                filename, file_hash[:12], user_id,
            )
            return existing

        # ── 2. Parse document ─────────────────────────────────────────────────
        logger.info("Parsing document: %s (user=%s)", filename, user_id)
        pages = parse_document(local_path)
        if not pages:
            raise ValueError(
                f"No extractable text found in '{filename}'. "
                "The file may be scanned/image-only."
            )

        # ── 3. Chunk ──────────────────────────────────────────────────────────
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=_tiktoken_length,
            separators=["\n\n", "\n", ". ", " "],
            is_separator_regex=False,
        )

        document_id    = str(uuid.uuid4())
        langchain_docs: list[Document] = []

        for page_data in pages:
            page_text: str = page_data["text"].strip()
            if not page_text:
                continue

            page_num: int  = page_data["metadata"].get("page_number", 0)
            chunks         = splitter.split_text(page_text)

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                langchain_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source":      filename,
                            "page":        page_num,
                            "chunk_index": i,
                            "document_id": document_id,
                            "file_hash":   file_hash,
                            "user_id":     user_id,   # ← tenant isolation key
                        },
                    )
                )

        if not langchain_docs:
            raise ValueError(
                f"Document '{filename}' produced no text chunks after splitting."
            )

        logger.info(
            "Split '%s' into %d chunks (user=%s).", filename, len(langchain_docs), user_id
        )

        # ── 4. Embed and store (dense + sparse) ───────────────────────────────
        # task_type="RETRIEVAL_DOCUMENT" is required by Gemini embedding models
        # for the document side of asymmetric retrieval.
        # The query side uses task_type="RETRIEVAL_QUERY" (set in rag_chain.py).
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            task_type="RETRIEVAL_DOCUMENT",
        )

        # Ensure the Qdrant collection exists before upserting
        ensure_collection()

        total_chunks  = len(langchain_docs)
        batches       = _build_token_batches(
            langchain_docs,
            max_tokens=settings.EMBED_MAX_TOKENS_PER_REQUEST,
            max_texts=settings.EMBED_BATCH_SIZE,
        )
        total_batches = len(batches)
        limiter       = _SlidingWindowLimiter()
        embedded      = 0
        started_at    = time.monotonic()

        logger.info(
            "Embedding %d chunks in %d token-paced batches "
            "(budget: %d TPM / %d RPM)...",
            total_chunks, total_batches,
            settings.EMBED_TPM_LIMIT, settings.EMBED_RPM_LIMIT,
        )

        if status_document_id:
            update_status(status_document_id, {
                "chunks_done":  0,
                "chunks_total": total_chunks,
            })

        for batch_num, (batch, batch_tokens) in enumerate(batches, 1):
            limiter.acquire(
                batch_tokens,
                tpm=settings.EMBED_TPM_LIMIT,
                rpm=settings.EMBED_RPM_LIMIT,
            )

            _upsert_batch_with_retry(batch, embeddings)

            embedded += len(batch)
            if status_document_id:
                update_status(status_document_id, {
                    "chunks_done":  embedded,
                    "chunks_total": total_chunks,
                })

            logger.debug(
                "Embedded batch %d/%d (%d chunks, ~%d tokens, %d/%d chunks done)",
                batch_num, total_batches, len(batch), batch_tokens,
                embedded, total_chunks,
            )

        elapsed = time.monotonic() - started_at
        logger.info(
            "Ingestion complete for '%s': %d chunks stored in %.1fs.",
            filename, total_chunks, elapsed,
        )

        return {
            "document_id":  document_id,
            "filename":     filename,
            "chunks_count": total_chunks,
            "status":       "completed",
        }

    finally:
        if cleanup_temp:
            try:
                Path(local_path).unlink()
            except OSError:
                pass