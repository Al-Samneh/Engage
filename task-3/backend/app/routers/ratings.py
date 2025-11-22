from fastapi import APIRouter, Depends

from ..dependencies import get_rating_service
from ..schemas import RatingPredictionRequest, RatingPredictionResponse
from ..services.model import RatingModelService

router = APIRouter()


@router.post(
    "/predict",
    response_model=RatingPredictionResponse,
    summary="Predict user rating",
)
async def predict_rating(
    payload: RatingPredictionRequest,
    model_service: RatingModelService = Depends(get_rating_service),
) -> RatingPredictionResponse:
    """Proxy to the Task 2 XGBoost pipeline. Returns latency + trace metadata."""
    result = model_service.predict(payload)
    return RatingPredictionResponse(
        trace_id=result.trace_id,
        latency_ms=result.latency_ms,
        data=result.payload,
    )

