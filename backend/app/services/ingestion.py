import hashlib
import uuid
import logging
import time
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

logger   = logging.getLogger(__name__)
settings = get_settings()

# ── Rate-limit config ─────────────────────────────────────────────────────────
BATCH_SIZE  = 5    # chunks per Google embed call (free tier: 5 RPM)
BATCH_DELAY = 12   # seconds between batches (60s / 5 RPM = 12s)
MAX_RETRIES = 5
RETRY_BASE  = 15   # seconds — doubles on each retry (exponential backoff)


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

    Retries with exponential backoff on Google API 429 / RESOURCE_EXHAUSTED errors.
    Non-rate-limit errors are re-raised immediately.
    """
    delay  = RETRY_BASE
    client = get_qdrant_client()
    sparse_encoder = get_sparse_encoder()
    texts  = [doc.page_content for doc in batch]

    for attempt in range(1, MAX_RETRIES + 1):
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
            if is_rate_limit and attempt < MAX_RETRIES:
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %ds...",
                    attempt, MAX_RETRIES, delay,
                )
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                raise


def ingest_document(file_path: str, filename: str, *, user_id: str) -> dict:
    """
    Full ingestion pipeline:
      1. Hash check (per user)  → return early if duplicate
      2. Parse PDF/DOCX         → page-level text blocks
      3. Chunk                  → RecursiveCharacterTextSplitter (tiktoken-based)
      4. Embed + store          → dense (Gemini) + sparse (BM25) upserted together
         — user_id stamped into every chunk for tenant isolation
         — batched in groups of BATCH_SIZE with BATCH_DELAY between batches
         — retries with exponential backoff on 429 rate-limit errors

    Args:
        file_path: Local path or Cloudinary URL to the uploaded file.
        filename:  Original filename (used as the source metadata key).
        user_id:   Clerk userId — stamped into every chunk for tenant isolation.

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

        total_batches = (len(langchain_docs) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_num, start in enumerate(range(0, len(langchain_docs), BATCH_SIZE), 1):
            batch = langchain_docs[start : start + BATCH_SIZE]

            logger.debug(
                "Embedding batch %d/%d (chunks %d–%d)...",
                batch_num, total_batches, start, start + len(batch) - 1,
            )

            _upsert_batch_with_retry(batch, embeddings)

            # Respect free-tier RPM limit — skip delay after the last batch
            if start + BATCH_SIZE < len(langchain_docs):
                time.sleep(BATCH_DELAY)

        logger.info(
            "Ingestion complete for '%s': %d chunks stored.", filename, len(langchain_docs)
        )

        return {
            "document_id":  document_id,
            "filename":     filename,
            "chunks_count": len(langchain_docs),
            "status":       "completed",
        }

    finally:
        if cleanup_temp:
            try:
                Path(local_path).unlink()
            except OSError:
                pass