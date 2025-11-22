import logging

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_rating_service
from ..schemas import RatingPredictionRequest, RatingPredictionResponse
from ..services.model import RatingModelService

router = APIRouter()
logger = logging.getLogger(__name__)


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
    try:
        result = model_service.predict(payload)
    except HTTPException:
        # Re-raise structured FastAPI errors from the service layer unchanged.
        raise
    except Exception as exc:
        # Log full traceback plus a compact view of the request for CloudWatch debugging.
        try:
            safe_payload = payload.model_dump(mode="json")
        except Exception:
            safe_payload = {"unserializable_payload": str(payload)}
        logger.exception("Rating prediction FAILED", extra={"payload": safe_payload})
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": f"Rating prediction crashed: {exc}",
            },
        ) from exc

    return RatingPredictionResponse(
        trace_id=result.trace_id,
        latency_ms=result.latency_ms,
        data=result.payload,
    )

