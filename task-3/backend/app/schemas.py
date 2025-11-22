from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, UUID4, conlist, constr


# ---- Shared ----


class TraceEnvelope(BaseModel):
    trace_id: UUID4
    latency_ms: int = Field(ge=0)


class ErrorResponse(TraceEnvelope):
    code: constr(strip_whitespace=True, min_length=2, max_length=64)
    message: constr(strip_whitespace=True, min_length=2, max_length=512)


# ---- Restaurant Search ----


class RestaurantMetadata(BaseModel):
    user_id: Optional[str] = Field(default=None, max_length=64)
    channel: Optional[str] = Field(default=None, max_length=32)


class RestaurantSearchRequest(BaseModel):
    question: constr(min_length=1, max_length=1000)
    conversation_id: Optional[UUID4] = None
    metadata: Optional[RestaurantMetadata] = None


class DocumentSnippet(BaseModel):
    id: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=-1, le=1)
    snippet: Optional[str] = Field(default=None, max_length=4000)
    metadata: Optional[dict] = None


class AppliedFilters(BaseModel):
    location: Optional[str] = None
    cuisine: Optional[str] = None
    price_max: Optional[int] = Field(default=None, ge=0)
    amenities: Optional[str] = None


class RestaurantSearchPayload(BaseModel):
    answer: str
    applied_filters: AppliedFilters
    documents: List[DocumentSnippet]
    fallback: bool = False


class RestaurantSearchResponse(TraceEnvelope):
    data: RestaurantSearchPayload


# ---- Rating Prediction ----


class TrendFeatures(BaseModel):
    popularity_7_day_avg: float
    popularity_30_day_avg: float
    popularity_lag_1: float
    avg_price_7_day_avg: float
    popularity_7_day_growth: float


class RestaurantInfo(BaseModel):
    location: constr(min_length=2, max_length=64)
    cuisine: constr(min_length=2, max_length=64)
    price_bucket: constr(min_length=1, max_length=32)
    description: str = ""
    amenities: conlist(str, min_length=0) = []
    attributes: conlist(str, min_length=0) = []
    avg_price: float = Field(ge=0)
    popularity_score: float = Field(ge=0)
    trend_features: TrendFeatures


class UserInfo(BaseModel):
    age: int = Field(ge=0, le=120)
    home_location: str
    preferred_price_range: str
    dietary_restrictions: str = "none"
    dining_frequency: str
    avg_rating_given: float = Field(ge=0, le=5)
    total_reviews_written: int = Field(ge=0)
    is_local_resident: bool = False
    user_cuisine_match: float = Field(ge=0, le=1)
    dietary_conflict: float = Field(ge=0, le=1)


class ReviewContext(BaseModel):
    helpful_count: int = Field(ge=0)
    season: str
    day_type: str
    weather_impact_category: str
    review_month: int = Field(ge=1, le=12)
    review_day_of_week: int = Field(ge=0, le=6)
    is_holiday: bool = False
    booking_lead_time_days: int = Field(ge=0)


class EmbeddingPayload(BaseModel):
    review_text_embedding: List[float]
    tokenizer_version: Optional[str] = None


class RatingPredictionRequest(BaseModel):
    restaurant: RestaurantInfo
    user: UserInfo
    review_context: ReviewContext
    review_text: Optional[str] = None
    embeddings: Optional[EmbeddingPayload] = None


class RatingPredictionPayload(BaseModel):
    rating_prediction: float = Field(ge=0, le=5)
    rounded_rating: float = Field(ge=1, le=5)
    confidence_interval: Optional[List[float]] = None
    model_version: str
    inference_mode: str


class RatingPredictionResponse(TraceEnvelope):
    data: RatingPredictionPayload


# ---- Health ----


class DependencyStatus(BaseModel):
    status: str
    latency_ms: Optional[int] = None
    detail: Optional[str] = None


class HealthPayload(BaseModel):
    status: str
    chroma: DependencyStatus
    rag_graph: DependencyStatus
    rating_model: DependencyStatus


class HealthResponse(TraceEnvelope):
    data: HealthPayload

