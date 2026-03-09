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
)

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SPARSE_VECTOR_NAME = "text-sparse"

# ── Singleton client (module-level cache) ─────────────────────────────────────

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """
    Returns a module-level singleton QdrantClient.
    Uses local on-disk persistence via qdrant_client's built-in storage.
    Uses Qdrant Cloud if QDRANT_URL and QDRANT_API_KEY are set in config.
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


def ensure_collection() -> None:
    """
    Create the Qdrant collection if it doesn't exist yet.

    Collection schema:
      • Dense vectors  → Google embedding (cosine)
      • Sparse vectors → BM25 keyword index (named "text-sparse")
                         only created when ENABLE_HYBRID_SEARCH=True

    ⚠️  If you toggle ENABLE_HYBRID_SEARCH after a collection already exists,
    the sparse index WON'T be added retroactively. You must either:
      a) Delete the collection and re-ingest all documents, or
      b) Call client.update_collection() to add the sparse vector config.
    """
    client = get_qdrant_client()
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


# ── Tenant isolation helper ───────────────────────────────────────────────────

def _build_tenant_filter(user_id: str, filename: str | None = None) -> Filter:
    """
    Build a Qdrant Filter scoped to a single tenant.

    Always filters by metadata.user_id (mandatory).
    Optionally narrows to a specific document via metadata.source.

    This matches the metadata keys stamped during ingestion:
        metadata = {
            "source":    filename,   ← matched by filename param
            "user_id":   user_id,    ← always matched
            ...
        }
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

    Args:
        embeddings: Optional pre-built embeddings instance to reuse.
        user_id:    When provided, all similarity searches on this store
                    are automatically scoped to that tenant's chunks only.
                    Pass None only during ingestion (tenant is enforced by
                    metadata stamping, not by query filter).
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
    Delete all chunks matching both *filename* and *user_id*.

    Tenant isolation is enforced — a user can only delete their own chunks
    even if another user has a document with the same filename.

    Returns the number of chunks deleted, or 0 if none were found.
    """
    client = get_qdrant_client()
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

    Scrolls through only the tenant's chunks (filtered by metadata.user_id)
    and groups them by source filename in Python.
    """
    client = get_qdrant_client()

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
            scroll_filter=tenant_filter,   # ← only this tenant's chunks
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