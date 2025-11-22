from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .middleware import make_rate_limit_middleware
from .routers import api_router


def create_app() -> FastAPI:
    """
    Assemble the FastAPI instance with shared middleware + routers.
    Called both by uvicorn and tests, so keep side-effects here.
    """
    settings = get_settings()
    app = FastAPI(
        title=settings.api_name,
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.middleware("http")(make_rate_limit_middleware())
    app.include_router(api_router, prefix=f"/{settings.api_version}")

    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="frontend")

    @app.get("/")
    async def root_redirect():
        return RedirectResponse(url="/ui/")

    return app


app = create_app()

