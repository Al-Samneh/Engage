"""
Central configuration for the Task 3 backend.

The API embeds Task 1 (RAG) and Task 2 (rating model) assets directly from neighboring
folders, so we compute those roots once here and expose them through a Pydantic settings
object. Keeping this module tiny makes it easy to swap env vars when deploying.
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# BASE_DIR resolves to the Engage repo root (…/Engage/). From there we can reference
# task-1/ and task-2/ regardless of where the API is launched.
BASE_DIR = Path(__file__).resolve().parents[3]
TASK1_DIR = BASE_DIR / "task-1"
TASK2_DIR = BASE_DIR / "task-2"


class Settings(BaseModel):
    """Runtime knobs for the FastAPI app plus downstream ML services."""

    api_version: str = "v1"
    api_name: str = "Restaurant Intelligence API"
    default_rate_limit_per_minute: int = 60
    rate_limit_window_seconds: int = 60
    langgraph_cache_size: int = Field(1, description="How many compiled graphs to cache")
    task1_dir: Path = TASK1_DIR
    task2_dir: Path = TASK2_DIR
    rating_model_filename: str = "best_restaurant_rating_model_xgboost.pkl"
    enable_local_model: bool = True
    sagemaker_endpoint_name: Optional[str] = None
    aws_region: str = os.environ.get("AWS_REGION", "us-east-1")
    aws_profile: Optional[str] = os.environ.get("AWS_PROFILE")
    aws_endpoint_url: Optional[str] = os.environ.get("AWS_ENDPOINT_URL")
    sagemaker_timeout_seconds: int = Field(10, ge=1, le=60)
    redis_url: str = os.environ.get("CHAT_CACHE_URL", "memory://")
    google_api_key: Optional[str] = os.environ.get("GOOGLE_API_KEY")
    # Default to a high-performance, widely available SBERT model. Can be overridden in env.
    rating_embedding_model: str = os.environ.get(
        "RATING_EMBED_MODEL", "all-mpnet-base-v2"
    )
    rating_embedding_device: str = os.environ.get("RATING_EMBED_DEVICE", "cpu")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached accessor so every dependency import does not rebuild the Settings object.
    """
    return Settings()

