import logging

from fastapi import APIRouter, HTTPException, status

from app.dependencies import AuthUser
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services.rag_chain import query as rag_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about your uploaded documents",
)
async def query_documents(request: QueryRequest, current_user: AuthUser) -> QueryResponse:
    logger.info(
        "Query | user_id=%s | question='%s' | filter='%s'",
        current_user.user_id, request.question[:80], request.document_filter,
    )

    try:
        result = rag_query(
            question=request.question,
            document_filter=request.document_filter,
            user_id=current_user.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("RAG query failed | user_id=%s: %s", current_user.user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query could not be completed. Please try again.",
        )

    sources = [
        SourceChunk(
            content=chunk["content"],
            source=chunk["source"],
            page=chunk.get("page", 0),
            score=chunk.get("score"),
        )
        for chunk in result.get("sources", [])
    ]

    return QueryResponse(answer=result["answer"], sources=sources)