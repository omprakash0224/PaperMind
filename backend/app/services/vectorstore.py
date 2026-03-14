import logging

from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Modifier,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from app.config import get_settings
from app.services.bm25 import BM25Encoder, get_bm25_encoder

logger   = logging.getLogger(__name__)
settings = get_settings()

SPARSE_VECTOR_NAME = "text-sparse"

# ── Singleton client (module-level cache) ─────────────────────────────────────

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """
    Returns a module-level singleton QdrantClient.
    Uses Qdrant Cloud if QDRANT_URL + QDRANT_API_KEY are set, otherwise local disk.
    """
    global _qdrant_client
    if _qdrant_client is None:
        if settings.use_qdrant_cloud:
            logger.info("Initialising QdrantClient (cloud): %s", settings.QDRANT_URL)
            _qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            logger.info("Initialising QdrantClient (local): %s", settings.QDRANT_PERSIST_DIR)
            _qdrant_client = QdrantClient(path=str(settings.QDRANT_PERSIST_DIR))
    return _qdrant_client


def get_sparse_encoder() -> BM25Encoder:
    """
    Returns the pure-Python BM25 encoder singleton.

    No external dependencies — uses only Python stdlib (re, hashlib).
    Works on any Python version including 3.14.

    Qdrant handles IDF weighting server-side via Modifier.IDF on the
    sparse vector field. This encoder only supplies raw TF weights.
    """
    return get_bm25_encoder()


def _ensure_payload_indexes(client: QdrantClient) -> None:
    """
    Create payload indexes required for filtered queries.

    Qdrant Cloud requires explicit indexes on fields used in filters.
    Creating an index that already exists is a safe no-op.
    """
    for field_name in ("metadata.user_id", "metadata.source", "metadata.file_hash"):
        try:
            client.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Payload index ensured: %s (keyword)", field_name)
        except Exception as exc:
            # Index may already exist — that's fine
            if "already exists" in str(exc).lower():
                logger.debug("Payload index already exists: %s", field_name)
            else:
                logger.warning("Could not create payload index '%s': %s", field_name, exc)


def ensure_collection() -> None:
    """
    Create the Qdrant collection if it doesn't exist yet.

    Collection schema:
      • Dense vectors  → Google Gemini embeddings (cosine similarity)
      • Sparse vectors → BM25 keyword index named "text-sparse"
                         only created when ENABLE_HYBRID_SEARCH=True

    Modifier.IDF instructs Qdrant to apply inverse document frequency
    weighting at query time — correct for BM25 (client supplies TF,
    Qdrant applies IDF). This would be Modifier.NONE for SPLADE.

    ⚠️  Toggling ENABLE_HYBRID_SEARCH after a collection already exists
    will NOT retroactively add/remove the sparse index. You must delete
    the collection and re-ingest all documents to change the schema.
    """
    client   = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]

    if settings.COLLECTION_NAME not in existing:
        sparse_vectors_config = {}

        if settings.ENABLE_HYBRID_SEARCH:
            sparse_vectors_config[SPARSE_VECTOR_NAME] = SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF,
            )
            logger.info(
                "Creating Qdrant collection '%s' with dense (dim=%d) + BM25 sparse vectors.",
                settings.COLLECTION_NAME,
                settings.EMBEDDING_DIM,
            )
        else:
            logger.info(
                "Creating Qdrant collection '%s' with dense vectors only (dim=%d).",
                settings.COLLECTION_NAME,
                settings.EMBEDDING_DIM,
            )

        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config=sparse_vectors_config or None,
        )
    else:
        logger.debug("Collection '%s' already exists.", settings.COLLECTION_NAME)

    # Always ensure payload indexes exist (idempotent)
    _ensure_payload_indexes(client)


# ── Tenant isolation helper ───────────────────────────────────────────────────

def _build_tenant_filter(user_id: str, filename: str | None = None) -> Filter:
    """
    Build a Qdrant Filter scoped to a single tenant.
    Always filters by metadata.user_id. Optionally narrows to one document.
    """
    conditions = [
        FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))
    ]
    if filename is not None:
        conditions.append(
            FieldCondition(key="metadata.source", match=MatchValue(value=filename))
        )
    return Filter(must=conditions)


# ── VectorStore factory ───────────────────────────────────────────────────────

def get_langchain_vectorstore(
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
    user_id: str | None = None,
) -> QdrantVectorStore:
    """
    Returns a LangChain QdrantVectorStore wrapper around the singleton client.

    Used for dense-only retrieval fallback (ENABLE_HYBRID_SEARCH=False).
    For hybrid ingestion we bypass this and upsert via the Qdrant client
    directly so we can include both dense + sparse vectors in one call.
    """
    if embeddings is None:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    ensure_collection()

    kwargs: dict = dict(
        client=get_qdrant_client(),
        collection_name=settings.COLLECTION_NAME,
        embedding=embeddings,
    )

    if user_id:
        kwargs["filter"] = _build_tenant_filter(user_id)
        logger.debug("VectorStore scoped to tenant user_id='%s'.", user_id)

    return QdrantVectorStore(**kwargs)


# ── Document-level operations ─────────────────────────────────────────────────

def delete_document(filename: str, user_id: str) -> int:
    """
    Delete all chunks matching both filename and user_id.
    Tenant isolation is enforced — users can only delete their own documents.
    Returns the number of chunks deleted, or 0 if none found.
    """
    client        = get_qdrant_client()
    tenant_filter = _build_tenant_filter(user_id, filename)

    count_result = client.count(
        collection_name=settings.COLLECTION_NAME,
        count_filter=tenant_filter,
        exact=True,
    )
    chunk_count: int = count_result.count

    if chunk_count == 0:
        logger.warning(
            "delete_document: no chunks found for filename='%s' user_id='%s'.",
            filename, user_id,
        )
        return 0

    client.delete(
        collection_name=settings.COLLECTION_NAME,
        points_selector=tenant_filter,
    )

    logger.info(
        "Deleted %d chunks | filename='%s' | user_id='%s'.",
        chunk_count, filename, user_id,
    )
    return chunk_count


def list_documents(user_id: str) -> list[dict]:
    """
    Return a list of distinct documents for a specific tenant.
    Each entry: {filename, document_id, chunks_count}.
    Scrolls through only the tenant's chunks and groups by source filename.
    """
    client   = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]

    if settings.COLLECTION_NAME not in existing:
        return []

    tenant_filter = _build_tenant_filter(user_id)

    total = client.count(
        collection_name=settings.COLLECTION_NAME,
        count_filter=tenant_filter,
        exact=True,
    ).count

    if total == 0:
        return []

    doc_map: dict[str, dict] = {}
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=tenant_filter,
            with_payload=True,
            with_vectors=False,
            limit=100,
            offset=offset,
        )

        for record in records:
            payload = record.payload or {}
            meta: dict = payload.get("metadata", {})
            source: str = meta.get("source", "unknown")

            if source not in doc_map:
                doc_map[source] = {
                    "filename":     source,
                    "document_id":  meta.get("document_id", "unknown"),
                    "chunks_count": 0,
                }
            doc_map[source]["chunks_count"] += 1

        if offset is None:
            break

    documents = sorted(doc_map.values(), key=lambda d: d["filename"])
    logger.debug(
        "list_documents: %d distinct documents for user_id='%s'.",
        len(documents), user_id,
    )
    return documents