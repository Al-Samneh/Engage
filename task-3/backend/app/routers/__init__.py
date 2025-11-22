from fastapi import APIRouter

from .health import router as health_router
from .ratings import router as rating_router
from .restaurants import router as restaurant_router

# Single router exported to main.py. Individual modules keep their own prefixes/tags.
api_router = APIRouter()
api_router.include_router(restaurant_router, prefix="/restaurants", tags=["restaurants"])
api_router.include_router(rating_router, prefix="/ratings", tags=["ratings"])
api_router.include_router(health_router, tags=["health"])

__all__ = ["api_router"]

