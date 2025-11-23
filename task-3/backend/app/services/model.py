from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException, status

from ..config import Settings
from ..schemas import (
    RatingPredictionPayload,
    RatingPredictionRequest,
)
from .embedding import EmbeddingService


def _ensure_task2_on_path(task2_dir: Path) -> None:
    path_str = str(task2_dir)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)
    # Ensure future imports pull Task 2 modules (config, etc.), not Task 1 versions.
    for module_name in list(sys.modules.keys()):
        if module_name in {"config", "data_utils", "features", "model"}:
            module = sys.modules[module_name]
            module_file = getattr(module, "__file__", "") or ""
            if module_file and str(task2_dir) not in module_file:
                sys.modules.pop(module_name, None)


@dataclass
class RatingResult:
    payload: RatingPredictionPayload
    latency_ms: int
    trace_id: uuid.UUID


class RatingModelService:
    """Loads the Task 2 pipeline artifact and exposes a FastAPI-friendly predict method."""

    def __init__(self, settings: Settings, embedding_service: EmbeddingService) -> None:
        self._settings = settings
        self._embedding_service = embedding_service
        self._sm_client = None

        if settings.enable_local_model:
            _ensure_task2_on_path(settings.task2_dir)
            model_path = settings.task2_dir / settings.rating_model_filename
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Rating model not found at {model_path}. Train Task 2 pipeline first."
                )
            self._model = joblib.load(model_path)
            self._model_version = model_path.stem
            preprocessor = self._model.named_steps["preprocessor"]
            embed_entry = next(
                (entry for entry in preprocessor.transformers if entry[0] == "embed"), None
            )
            self._embedding_cols: List[str] = embed_entry[2] if embed_entry else []

            # Find the model step (could be 'regressor' or 'classifier' depending on training)
            model_step_name = None
            for step_name in self._model.named_steps:
                if step_name in ['regressor', 'classifier']:
                    model_step_name = step_name
                    break

            if model_step_name:
                model_step = self._model.named_steps[model_step_name]
                residuals = getattr(model_step, "best_score", None)
                if hasattr(model_step, "evals_result"):
                    pass  # placeholder for richer metrics
            else:
                residuals = None

            self._residual_std = 0.3  # fallback
        else:
            if not settings.sagemaker_endpoint_name:
                raise ValueError("ENABLE_LOCAL_MODEL=false but no SageMaker endpoint provided.")
            session_kwargs = {}
            if settings.aws_profile:
                session_kwargs["profile_name"] = settings.aws_profile
            # Use the configured AWS profile (if any) so local testing can leverage named credentials.
            boto3.setup_default_session(**session_kwargs)
            config = Config(
                region_name=settings.aws_region,
                retries={"max_attempts": 3, "mode": "standard"},
                read_timeout=settings.sagemaker_timeout_seconds,
                connect_timeout=settings.sagemaker_timeout_seconds,
            )
            self._sm_client = boto3.client(
                "sagemaker-runtime",
                region_name=settings.aws_region,
                endpoint_url=settings.aws_endpoint_url,
                config=config,
            )
            self._model = None
            self._model_version = settings.sagemaker_endpoint_name
            self._embedding_cols = []
            self._residual_std = None

    def _resolve_embedding(self, request: RatingPredictionRequest) -> List[float]:
        """
        Accept caller-provided embeddings when available, otherwise create one on the fly.
        """
        if request.embeddings and request.embeddings.review_text_embedding:
            return request.embeddings.review_text_embedding

        if request.review_text:
            try:
                return self._embedding_service.embed(request.review_text)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "code": "EMBEDDING_FAILURE",
                        "message": f"Failed to generate embedding: {exc}",
                    },
                ) from exc

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "MISSING_REVIEW_TEXT",
                "message": "Provide either review_text or review_text_embedding.",
            },
        )

    def _to_dataframe(
        self, request: RatingPredictionRequest, embedding_vector: List[float]
    ) -> pd.DataFrame:
        """
        Mirror the feature schema used during training so the sklearn pipeline can run.
        """
        rest = request.restaurant
        user = request.user
        ctx = request.review_context

        row: Dict[str, Optional[float]] = {
            "helpful_count": ctx.helpful_count,
            "age": user.age,
            "avg_rating_given": user.avg_rating_given,
            "total_reviews_written": user.total_reviews_written,
            "popularity_score": rest.popularity_score,
            "avg_price": rest.avg_price,
            "booking_lead_time_days": ctx.booking_lead_time_days,
            "popularity_7_day_avg": rest.trend_features.popularity_7_day_avg,
            "popularity_30_day_avg": rest.trend_features.popularity_30_day_avg,
            "popularity_lag_1": rest.trend_features.popularity_lag_1,
            "avg_price_7_day_avg": rest.trend_features.avg_price_7_day_avg,
            "popularity_7_day_growth": rest.trend_features.popularity_7_day_growth,
            "price_avg": rest.avg_price,
            "price_alignment_score": user.user_cuisine_match,
            "user_cuisine_match": user.user_cuisine_match,
            "dietary_conflict": user.dietary_conflict,
            "is_local_resident": 1 if user.is_local_resident else 0,
            "resto_location": rest.location,
            "resto_cuisine": rest.cuisine,
            "resto_price_bucket": rest.price_bucket,
            "home_location": user.home_location,
            "preferred_price_range": user.preferred_price_range,
            "dietary_restrictions": user.dietary_restrictions,
            "dining_frequency": user.dining_frequency,
            "season": ctx.season,
            "day_type": ctx.day_type,
            "weather_impact_category": ctx.weather_impact_category,
            "review_month": ctx.review_month,
            "review_day_of_week": ctx.review_day_of_week,
            "is_holiday": int(ctx.is_holiday),
            "resto_description": rest.description,
            "resto_amenities": ", ".join(rest.amenities),
            "resto_attributes": ", ".join(rest.attributes),
        }

        for idx, col in enumerate(self._embedding_cols):
            row[col] = embedding_vector[idx]

        return pd.DataFrame([row])

    def predict(self, request: RatingPredictionRequest) -> RatingResult:
        """
        Core inference entry point. Handles trace bookkeeping + error translation.
        """
        trace_id = uuid.uuid4()
        start = time.perf_counter()

        embedding_vector = self._resolve_embedding(request)
        if self._embedding_cols and len(embedding_vector) != len(self._embedding_cols):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "EMBED_DIM_MISMATCH",
                    "message": (
                        f"Expected {len(self._embedding_cols)} embedding values, "
                        f"received {len(embedding_vector)}."
                    ),
                },
            )

        if self._sm_client:
            pred_raw = self._predict_remote(request, embedding_vector, trace_id)
        else:
            df = self._to_dataframe(request, embedding_vector)
            try:
                pred_raw = float(self._model.predict(df)[0])
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "code": "MODEL_INFERENCE_ERROR",
                        "message": "Prediction failed. Ensure schema matches training data.",
                        "trace_id": str(trace_id),
                    },
                ) from exc

        pred = float(np.clip(pred_raw, 1.0, 5.0))
        rounded_pred = float(np.clip(np.rint(pred), 1.0, 5.0))

        ci = None
        if self._residual_std:
            lower = max(1.0, pred - 1.96 * self._residual_std)
            upper = min(5.0, pred + 1.96 * self._residual_std)
            ci = [lower, upper]

        payload = RatingPredictionPayload(
            rating_prediction=pred,
            rounded_rating=rounded_pred,
            confidence_interval=ci,
            model_version=self._model_version,
            inference_mode="remote" if self._sm_client else "local",
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        return RatingResult(payload=payload, latency_ms=latency_ms, trace_id=trace_id)

    def _predict_remote(
        self,
        request: RatingPredictionRequest,
        embedding_vector: List[float],
        trace_id: uuid.UUID,
    ) -> float:
        """
        Invoke the configured SageMaker endpoint and return the raw float prediction.
        """
        if not self._sm_client:
            raise RuntimeError("Remote inference requested but SageMaker client not initialized.")

        payload = request.model_dump(mode="json")
        payload.setdefault("embeddings", {})
        # Always send the embedding vector explicitly so SageMaker does not need to
        # run an embedding model during inference (keeps latency low and logic aligned
        # with the local flow).
        payload["embeddings"]["review_text_embedding"] = embedding_vector

        try:
            response = self._sm_client.invoke_endpoint(
                EndpointName=self._settings.sagemaker_endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload).encode("utf-8"),
            )
        except (BotoCoreError, ClientError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "SAGEMAKER_UNAVAILABLE",
                    "message": f"SageMaker invocation failed: {exc}",
                    "trace_id": str(trace_id),
                },
            ) from exc

        try:
            body = response.get("Body").read()
            result = json.loads(body)
            return float(result["rating_prediction"])
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "SAGEMAKER_BAD_RESPONSE",
                    "message": f"SageMaker returned invalid payload: {exc}",
                    "trace_id": str(trace_id),
                },
            ) from exc

