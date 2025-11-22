from __future__ import annotations

from functools import lru_cache

from .config import get_settings
from .services.chat_store import ChatStore, InMemoryChatStore
from .services.embedding import EmbeddingService
from .services.model import RatingModelService
from .services.rag import RAGService


@lru_cache(maxsize=1)
def _chat_store() -> ChatStore:
    """
    Factory for the conversational memory layer. Defaults to an in-process deque, but this
    is the single place to swap in Redis or Dynamo without touching routers.
    """
    settings = get_settings()
    if settings.redis_url == "memory://":
        return InMemoryChatStore()
    # Placeholder for Redis/Dynamo integration.
    return InMemoryChatStore()


@lru_cache(maxsize=1)
def _rag_service() -> RAGService:
    """Compile the LangGraph workflow once and reuse it across requests."""
    return RAGService(settings=get_settings(), chat_store=_chat_store())


@lru_cache(maxsize=1)
def _embedding_service() -> EmbeddingService:
    """SentenceTransformer wrapper shared by both training and inference paths."""
    settings = get_settings()
    return EmbeddingService(
        model_name=settings.rating_embedding_model,
        device=settings.rating_embedding_device,
    )


@lru_cache(maxsize=1)
def _rating_service() -> RatingModelService:
    """Load the Task 2 pipeline artifact and keep it warm for future predictions."""
    return RatingModelService(
        settings=get_settings(),
        embedding_service=_embedding_service(),
    )


def get_chat_store() -> ChatStore:
    """FastAPI dependency hook."""
    return _chat_store()


def get_rag_service() -> RAGService:
    return _rag_service()


def get_rating_service() -> RatingModelService:
    return _rating_service()


def get_embedding_service() -> EmbeddingService:
    return _embedding_service()

