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
    NamedVector,
    NamedSparseVector,
)

from app.config import get_settings
from app.services.vectorstore import (
    get_langchain_vectorstore,
    get_qdrant_client,
    SPARSE_VECTOR_NAME,
)

logger = logging.getLogger(__name__)
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
        rewritten = rewrite_chain.invoke({"question": question}).strip()

        if not rewritten or len(rewritten) > 500:
            logger.warning("Query rewrite returned unusable result — using original.")
            return question

        logger.info(
            "Query rewritten | original='%s' | rewritten='%s'",
            question[:80],
            rewritten[:80],
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
    Drop any (document, score) pair whose score is below MIN_SCORE_THRESHOLD.

    This prevents low-relevance chunks from polluting the LLM's context window
    with noise, which can cause hallucination or dilute the real answer.

    Returns the top MAX_CONTEXT_CHUNKS survivors, sorted highest-score first.
    If ALL chunks are below the threshold, returns an empty list — the caller
    will then return the "not enough information" response.
    """
    threshold = settings.MIN_SCORE_THRESHOLD
    before = len(docs_and_scores)

    filtered = [
        (doc, score)
        for doc, score in docs_and_scores
        if score >= threshold
    ]

    # Sort descending by score and cap at MAX_CONTEXT_CHUNKS
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
    query: str,
    vectorstore: QdrantVectorStore,
    document_filter: Optional[str],
    user_id: Optional[str] = None,
) -> list[tuple[Document, float]]:
    """
    Pure semantic (dense vector) retrieval.
    Returns [(Document, score), ...] with cosine similarity scores.
    Used when ENABLE_HYBRID_SEARCH=False.
    """
    search_kwargs: dict = {"k": settings.RETRIEVAL_TOP_K}

    must_conditions = []

    if user_id:
        must_conditions.append(
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=user_id),
            )
        )

    if document_filter:
        must_conditions.append(
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value=document_filter),
            )
        )

    if must_conditions:
        search_kwargs["filter"] = Filter(must=must_conditions)

    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        **search_kwargs,
    )
    return results


# ── Hybrid retrieval (dense + BM25 via Qdrant RRF) ───────────────────────────

def _retrieve_hybrid(
    query: str,
    embeddings: GoogleGenerativeAIEmbeddings,
    document_filter: Optional[str],
    user_id: Optional[str] = None,
) -> list[tuple[Document, float]]:
    """
    Hybrid retrieval using Qdrant's native Reciprocal Rank Fusion (RRF).

    How it works:
      1. Dense prefetch  — embed the query and retrieve top-N by cosine similarity
      2. Sparse prefetch — tokenise the query and retrieve top-N by BM25 score
      3. RRF fusion      — Qdrant merges both ranked lists using the formula:
                           score(d) = Σ  1 / (k + rank_i(d))
                           where k=60 (Qdrant default). Documents appearing in
                           both lists get a significant score boost.

    This approach catches two failure modes that pure semantic search misses:
      • Exact-term queries: "clause 4.2", "John Smith", specific IDs/codes
      • Vocabulary mismatch: document says "remuneration", user asks "salary"

    Args:
        query:           The (rewritten) search query string.
        embeddings:      Pre-built Google embeddings instance.
        document_filter: Optional filename to scope retrieval to one document.

    Returns:
        List of (Document, rrf_score) tuples. RRF scores are NOT cosine
        similarities — they are rank-based fusion scores in (0, ~0.03] range.
        The threshold filter uses these scores directly, so MIN_SCORE_THRESHOLD
        should be set low (0.001–0.01) when hybrid search is enabled.
        We normalise them to [0, 1] before returning so the threshold config
        stays consistent with pure dense search.
    """
    client = get_qdrant_client()

    # ── 1. Embed the query (dense vector) ────────────────────────────────────
    query_vector: list[float] = embeddings.embed_query(query)

    # ── 2. Build optional Qdrant filter ──────────────────────────────────────
    must_conditions = []

    if user_id:
        must_conditions.append(
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=user_id),
            )
        )

    if document_filter:
        must_conditions.append(
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value=document_filter),
            )
        )

    qdrant_filter: Filter | None = Filter(must=must_conditions) if must_conditions else None

    prefetch_limit = settings.HYBRID_PREFETCH_LIMIT

    # ── 3. Issue hybrid query with RRF fusion ─────────────────────────────────
    # Prefetch runs TWO sub-queries in parallel inside Qdrant:
    #   a) Dense: nearest neighbours by cosine similarity
    #   b) Sparse: BM25 keyword match on the "text-sparse" index
    # Then FusionQuery merges them via Reciprocal Rank Fusion.
    results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        prefetch=[
            # Dense sub-query
            Prefetch(
                query=query_vector,
                using="",             # "" = default dense vector
                limit=prefetch_limit,
                filter=qdrant_filter,
            ),
            # Sparse / BM25 sub-query
            # Qdrant tokenises `query` internally using the IDF modifier
            # configured on the "text-sparse" sparse vector field.
            Prefetch(
                query=SparseVector(
                    indices=[],       # empty = let Qdrant tokenise the text
                    values=[],
                ),
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

    # ── 4. Convert ScoredPoint list → (Document, score) pairs ────────────────
    points = results.points
    if not points:
        return []

    # Normalise RRF scores to [0, 1] so MIN_SCORE_THRESHOLD works uniformly.
    # RRF scores are small (e.g. 0.016) so we normalise by the max score.
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
        normalised_score = point.score / max_score   # → [0, 1]
        docs_and_scores.append((doc, normalised_score))

    logger.debug(
        "Hybrid retrieval returned %d points (max_raw_score=%.4f).",
        len(docs_and_scores),
        max_score,
    )
    return docs_and_scores


# ── Context formatting ────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """
    Render retrieved chunks into the context string injected into the prompt.
    Each chunk is prefixed with source + page for in-answer citations.
    """
    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _docs_to_sources(docs_and_scores: list[tuple[Document, float]]) -> list[dict]:
    """Convert (Document, score) pairs into serialisable source dicts."""
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
    question: str,
    document_filter: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    Run a RAG query against the Qdrant vector store.

    Full pipeline:
        question
          → _rewrite_query          (Gemini rewrites to keyword-rich query)
          → _retrieve_hybrid        (dense + BM25 via RRF)  ← if ENABLE_HYBRID_SEARCH
            OR _retrieve_dense      (pure cosine similarity) ← fallback
          → _apply_score_threshold  (drop low-relevance chunks)
          → _format_docs            (render chunks → context string)
          → RAG_PROMPT + Gemini     (generate grounded answer)
          → StrOutputParser

    Key design decisions:
      • Rewritten query is used for RETRIEVAL; original question for GENERATION
        so the answer directly addresses what the user asked.
      • Score threshold runs AFTER retrieval so we always fetch RETRIEVAL_TOP_K
        candidates and filter down, avoiding empty results from tight k values.
      • Hybrid RRF scores are normalised to [0,1] before threshold comparison
        so MIN_SCORE_THRESHOLD has the same meaning regardless of retrieval mode.

    Returns:
        {
            "answer":          str,
            "rewritten_query": str,
            "retrieval_mode":  "hybrid" | "dense",
            "sources": [{"content", "source", "page", "score"}, ...]
        }
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    embeddings  = _build_embeddings()
    llm         = _build_llm()
    vectorstore = get_langchain_vectorstore(embeddings)

    rewritten_query = _rewrite_query(question, llm)

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
            "sources": [],
        }

    # ── Step 3: Apply score threshold ─────────────────────────────────────────
    # Filters out chunks below MIN_SCORE_THRESHOLD and caps at MAX_CONTEXT_CHUNKS.
    filtered = _apply_score_threshold(docs_and_scores)

    if not filtered:
        logger.warning(
            "All %d retrieved chunks scored below threshold %.2f — no context available.",
            len(docs_and_scores),
            settings.MIN_SCORE_THRESHOLD,
        )
        return {
            "answer": (
                "I found some potentially related content but none was relevant enough "
                "to answer confidently. Try rephrasing your question or uploading a "
                "more specific document."
            ),
            "rewritten_query": rewritten_query,
            "retrieval_mode":  retrieval_mode,
            "sources": [],
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
        settings.GEMINI_MODEL,
        len(filtered),
        retrieval_mode,
    )
    answer: str = generation_chain.invoke({
        "context":  context,
        "question": question,       # ← always the ORIGINAL question
    })

    sources = _docs_to_sources(filtered)
    logger.info(
        "Query complete | answer_len=%d | sources=%d | rewritten='%s'",
        len(answer),
        len(sources),
        rewritten_query[:60],
    )

    return {
        "answer":          answer,
        "rewritten_query": rewritten_query,
        "retrieval_mode":  retrieval_mode,
        "sources":         sources,
    }