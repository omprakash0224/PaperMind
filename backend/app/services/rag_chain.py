import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
)

from app.config import get_settings
from app.services.vectorstore import (
    get_langchain_vectorstore,
    get_qdrant_client,
    get_sparse_encoder,       # ← pure-Python BM25 encoder
    SPARSE_VECTOR_NAME,
)

logger   = logging.getLogger(__name__)
settings = get_settings()


# ── Query rewriting prompt ────────────────────────────────────────────────────

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert at converting natural-language questions into "
            "keyword-rich search queries optimised for semantic document retrieval.\n\n"
            "Rules:\n"
            "1. Output ONLY the rewritten query — no explanation, no preamble, no quotes.\n"
            "2. Keep it concise (ideally 10-20 words).\n"
            "3. Use specific noun phrases and domain keywords.\n"
            "4. Expand abbreviations where helpful (e.g. 'FY' → 'fiscal year').\n"
            "5. Preserve all important entities: names, dates, numbers, proper nouns.\n"
            "6. Remove filler words: 'can you tell me', 'what is', 'please explain', etc.\n"
            "7. If the question is already a good search query, return it unchanged.\n\n"
            "Examples:\n"
            "  Input:  'What was the total revenue for FY2023?'\n"
            "  Output: total revenue fiscal year 2023 annual financials\n\n"
            "  Input:  'Can you explain what the company risk factors are?'\n"
            "  Output: company risk factors key risks business threats\n\n"
            "  Input:  'Who are the board members?'\n"
            "  Output: board of directors members names executive leadership\n"
        ),
    ),
    ("human", "{question}"),
])

# ── RAG answer prompt ─────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful assistant answering questions about uploaded documents.\n"
            "Use ONLY the following context to answer. If the context doesn't contain\n"
            "the answer, say \"I don't have enough information to answer this.\"\n"
            "Always cite the source document and page number.\n\n"
            "Context:\n{context}"
        ),
    ),
    ("human", "{question}"),
])


# ── Component builders ────────────────────────────────────────────────────────

def _build_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        # RETRIEVAL_QUERY is the correct task type for the query side.
        # Documents were ingested with task_type="RETRIEVAL_DOCUMENT".
        task_type="RETRIEVAL_QUERY",
    )


def _build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=0.1,
        google_api_key=settings.GOOGLE_API_KEY,
    )


# ── Query rewriting ───────────────────────────────────────────────────────────

def _rewrite_query(question: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Rewrite the user's question into a keyword-rich search query.
    Falls back to the original question on any failure — never blocks pipeline.
    """
    try:
        rewrite_chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        rewritten     = rewrite_chain.invoke({"question": question}).strip()

        if not rewritten or len(rewritten) > 500:
            logger.warning("Query rewrite returned unusable result — using original.")
            return question

        logger.info(
            "Query rewritten | original='%s' | rewritten='%s'",
            question[:80], rewritten[:80],
        )
        return rewritten

    except Exception as exc:
        logger.warning("Query rewrite failed (%s) — falling back to original.", exc)
        return question


# ── Score threshold filtering ─────────────────────────────────────────────────

def _apply_score_threshold(
    docs_and_scores: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """
    Drop chunks below MIN_SCORE_THRESHOLD.
    Returns top MAX_CONTEXT_CHUNKS survivors sorted highest-score first.
    Returns empty list if ALL chunks are below threshold — caller returns
    the "not enough information" response in that case.
    """
    threshold = settings.MIN_SCORE_THRESHOLD
    before    = len(docs_and_scores)

    filtered = [
        (doc, score)
        for doc, score in docs_and_scores
        if score >= threshold
    ]

    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[: settings.MAX_CONTEXT_CHUNKS]

    dropped = before - len(filtered)
    if dropped:
        logger.info(
            "Score threshold %.2f dropped %d/%d chunks (kept %d).",
            threshold, dropped, before, len(filtered),
        )

    return filtered


# ── Dense-only retrieval (fallback) ──────────────────────────────────────────

def _retrieve_dense(
    query:           str,
    vectorstore:     QdrantVectorStore,
    document_filter: Optional[str],
    user_id:         Optional[str] = None,
) -> list[tuple[Document, float]]:
    """
    Pure semantic (dense vector) retrieval via cosine similarity.
    Used when ENABLE_HYBRID_SEARCH=False in config.
    """
    search_kwargs: dict  = {"k": settings.RETRIEVAL_TOP_K}
    must_conditions      = []

    if user_id:
        must_conditions.append(
            FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))
        )
    if document_filter:
        must_conditions.append(
            FieldCondition(key="metadata.source", match=MatchValue(value=document_filter))
        )
    if must_conditions:
        search_kwargs["filter"] = Filter(must=must_conditions)

    return vectorstore.similarity_search_with_relevance_scores(query, **search_kwargs)


# ── Hybrid retrieval (dense + BM25 via Qdrant RRF) ───────────────────────────

def _retrieve_hybrid(
    query:           str,
    embeddings:      GoogleGenerativeAIEmbeddings,
    document_filter: Optional[str],
    user_id:         Optional[str] = None,
) -> list[tuple[Document, float]]:
    """
    Hybrid retrieval using Qdrant's native Reciprocal Rank Fusion (RRF).

    Pipeline:
      1. Dense prefetch  — embed query with Gemini, retrieve top-N by cosine similarity
      2. Sparse prefetch — encode query with pure-Python BM25, retrieve top-N by keyword score
      3. RRF fusion      — Qdrant merges both ranked lists:
                           score(d) = Σ 1 / (k + rank_i(d)),  k=60 (Qdrant default)
                           Documents appearing in both lists get a significant boost.

    BM25 sparse encoding:
      get_sparse_encoder().query_embed(query) produces (indices, values) where:
        indices = FNV-1a hashes of stemmed, stopword-filtered tokens
        values  = raw term frequency counts
      Qdrant applies IDF weighting server-side via Modifier.IDF on the collection.

    RRF scores are normalised to [0, 1] before returning so MIN_SCORE_THRESHOLD
    has consistent meaning regardless of retrieval mode (hybrid vs dense).
    """
    client = get_qdrant_client()

    # ── 1. Dense query vector (Gemini) ────────────────────────────────────────
    query_vector: list[float] = embeddings.embed_query(query)

    # ── 2. Sparse query vector (Pure-Python BM25) ─────────────────────────────
    # query_embed() returns a list with one SparseEmbedding.
    # .indices → sorted list of uint32 token hashes
    # .values  → corresponding TF weights (floats)
    sparse_encoder = get_sparse_encoder()
    sparse_result  = sparse_encoder.query_embed(query)[0]

    sparse_query = SparseVector(
        indices=sparse_result.indices,
        values=sparse_result.values,
    )

    logger.debug(
        "BM25 sparse query | unique_terms=%d | top_weight=%.2f",
        len(sparse_result.indices),
        max(sparse_result.values, default=0.0),
    )

    # ── 3. Build optional Qdrant filter ──────────────────────────────────────
    must_conditions = []
    if user_id:
        must_conditions.append(
            FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))
        )
    if document_filter:
        must_conditions.append(
            FieldCondition(key="metadata.source", match=MatchValue(value=document_filter))
        )
    qdrant_filter: Filter | None = Filter(must=must_conditions) if must_conditions else None

    prefetch_limit = settings.HYBRID_PREFETCH_LIMIT

    # ── 4. Issue hybrid query with RRF fusion ─────────────────────────────────
    results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        prefetch=[
            # Dense sub-query — cosine similarity via Gemini embeddings
            Prefetch(
                query=query_vector,
                using="",               # "" = default unnamed dense vector
                limit=prefetch_limit,
                filter=qdrant_filter,
            ),
            # Sparse sub-query — BM25 keyword match via pure-Python encoder
            Prefetch(
                query=sparse_query,     # real token indices + TF weights
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=qdrant_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=settings.RETRIEVAL_TOP_K,
        with_payload=True,
        with_vectors=False,
    )

    # ── 5. Convert ScoredPoint list → (Document, score) pairs ────────────────
    points = results.points
    if not points:
        return []

    # Normalise RRF scores to [0, 1] so MIN_SCORE_THRESHOLD is mode-agnostic.
    # Raw RRF scores are small (e.g. 0.016) — dividing by max preserves ranking
    # while making the threshold config consistent with dense-only mode.
    max_score = max(p.score for p in points) if points else 1.0
    if max_score == 0:
        max_score = 1.0

    docs_and_scores: list[tuple[Document, float]] = []
    for point in points:
        payload = point.payload or {}
        meta: dict = payload.get("metadata", {})
        doc = Document(
            page_content=payload.get("page_content", ""),
            metadata=meta,
        )
        normalised_score = point.score / max_score
        docs_and_scores.append((doc, normalised_score))

    logger.debug(
        "Hybrid retrieval | points=%d | max_raw_rrf_score=%.4f",
        len(docs_and_scores), max_score,
    )
    return docs_and_scores


# ── Context formatting ────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks into the context string injected into the LLM prompt."""
    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _docs_to_sources(docs_and_scores: list[tuple[Document, float]]) -> list[dict]:
    """Convert (Document, score) pairs into serialisable source dicts for the API response."""
    return [
        {
            "content": doc.page_content,
            "source":  doc.metadata.get("source", "unknown"),
            "page":    doc.metadata.get("page", 0),
            "score":   round(score, 4),
        }
        for doc, score in docs_and_scores
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def query(
    question:        str,
    document_filter: Optional[str] = None,
    user_id:         Optional[str] = None,
) -> dict:
    """
    Run a RAG query against the Qdrant vector store.

    Full pipeline:
        question
          → _rewrite_query          (Gemini rewrites to keyword-rich query)
          → _retrieve_hybrid        (dense Gemini + BM25 via RRF)   ← ENABLE_HYBRID_SEARCH=True
            OR _retrieve_dense      (pure cosine similarity)         ← ENABLE_HYBRID_SEARCH=False
          → _apply_score_threshold  (drop low-relevance chunks)
          → _format_docs            (render chunks → context string)
          → RAG_PROMPT + Gemini     (generate grounded answer)
          → StrOutputParser

    Key design decisions:
      • Rewritten query is used for RETRIEVAL only.
        Original question is passed to the LLM for GENERATION so the answer
        directly addresses what the user asked, not the rewritten form.
      • Score threshold runs AFTER retrieval so we always fetch RETRIEVAL_TOP_K
        candidates first, then filter — avoids empty results from tight k values.
      • RRF scores are normalised to [0, 1] so MIN_SCORE_THRESHOLD has the same
        meaning regardless of whether hybrid or dense retrieval is used.

    Returns:
        {
            "answer":          str,
            "rewritten_query": str,
            "retrieval_mode":  "hybrid" | "dense",
            "sources":         [{"content", "source", "page", "score"}, ...]
        }
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    embeddings  = _build_embeddings()
    llm         = _build_llm()
    vectorstore = get_langchain_vectorstore(embeddings)

    # ── Step 1: Rewrite query ─────────────────────────────────────────────────
    rewritten_query = _rewrite_query(question, llm)

    # ── Step 2: Retrieve ──────────────────────────────────────────────────────
    if settings.ENABLE_HYBRID_SEARCH:
        retrieval_mode = "hybrid"
        logger.info(
            "Hybrid retrieval (dense+BM25 RRF) | user_id=%s | query='%s'",
            user_id, rewritten_query[:80],
        )
        docs_and_scores = _retrieve_hybrid(
            rewritten_query, embeddings, document_filter, user_id=user_id
        )
    else:
        retrieval_mode = "dense"
        logger.info(
            "Dense-only retrieval | user_id=%s | query='%s'",
            user_id, rewritten_query[:80],
        )
        docs_and_scores = _retrieve_dense(
            rewritten_query, vectorstore, document_filter, user_id=user_id
        )

    if not docs_and_scores:
        logger.warning("No chunks retrieved — collection may be empty.")
        return {
            "answer": (
                "I don't have enough information to answer this. "
                "Please upload a relevant document first."
            ),
            "rewritten_query": rewritten_query,
            "retrieval_mode":  retrieval_mode,
            "sources":         [],
        }

    # ── Step 3: Apply score threshold ─────────────────────────────────────────
    filtered = _apply_score_threshold(docs_and_scores)

    if not filtered:
        logger.warning(
            "All %d retrieved chunks scored below threshold %.2f — no context available.",
            len(docs_and_scores), settings.MIN_SCORE_THRESHOLD,
        )
        return {
            "answer": (
                "I found some potentially related content but none was relevant enough "
                "to answer confidently. Try rephrasing your question or uploading a "
                "more specific document."
            ),
            "rewritten_query": rewritten_query,
            "retrieval_mode":  retrieval_mode,
            "sources":         [],
        }

    # ── Step 4: Format context and generate answer ────────────────────────────
    docs_only = [doc for doc, _ in filtered]
    context   = _format_docs(docs_only)

    generation_chain = (
        RunnablePassthrough()
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info(
        "Generating answer | model=%s | chunks=%d | mode=%s",
        settings.GEMINI_MODEL, len(filtered), retrieval_mode,
    )

    answer: str = generation_chain.invoke({
        "context":  context,
        "question": question,   # ← always the ORIGINAL question, not rewritten
    })

    sources = _docs_to_sources(filtered)
    logger.info(
        "Query complete | answer_len=%d | sources=%d | rewritten='%s'",
        len(answer), len(sources), rewritten_query[:60],
    )

    return {
        "answer":          answer,
        "rewritten_query": rewritten_query,
        "retrieval_mode":  retrieval_mode,
        "sources":         sources,
    }