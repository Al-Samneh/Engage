from fastapi import APIRouter, Depends

from ..dependencies import get_rag_service
from ..schemas import RestaurantSearchRequest, RestaurantSearchResponse
from ..services.rag import RAGService

router = APIRouter()


@router.post(
    "/search",
    response_model=RestaurantSearchResponse,
    summary="Restaurant RAG search",
)
async def search_restaurants(
    payload: RestaurantSearchRequest, rag_service: RAGService = Depends(get_rag_service)
) -> RestaurantSearchResponse:
    """Run the Task 1 LangGraph workflow, persisting chat state per conversation id."""
    result = rag_service.search(payload)
    return RestaurantSearchResponse(
        trace_id=result.trace_id,
        latency_ms=result.latency_ms,
        data=result.payload,
    )

