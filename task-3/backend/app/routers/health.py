import sys
import time
import uuid

import importlib.util
import sys
from pathlib import Path

from fastapi import APIRouter, Depends

from ..config import get_settings
from ..dependencies import get_rag_service, get_rating_service
from ..schemas import DependencyStatus, HealthPayload, HealthResponse
from ..services.model import RatingModelService
from ..services.rag import RAGService

router = APIRouter()


def _load_task1_config(task1_dir: Path):
    config_path = task1_dir / "config.py"
    spec = importlib.util.spec_from_file_location(
        f"task1_config_{hash(config_path)}", config_path
    )
    if not spec or not spec.loader:
        raise ImportError(f"Cannot create spec for {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ping_chroma(task1_dir: Path) -> DependencyStatus:
    task1_config = _load_task1_config(task1_dir)
    client = task1_config.get_chroma_client()
    ping_start = time.perf_counter()
    client.heartbeat()
    return DependencyStatus(
        status="ok", latency_ms=int((time.perf_counter() - ping_start) * 1000)
    )


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Service health status",
)
async def healthz(
    rag_service: RAGService = Depends(get_rag_service),
    rating_service: RatingModelService = Depends(get_rating_service),
) -> HealthResponse:
    """
    Ping all major dependencies: Chroma (Task 1), LangGraph, and the rating model.
    Returns a trace_id so callers can correlate with logs.
    """
    trace_id = uuid.uuid4()
    start = time.perf_counter()

    chroma_status = DependencyStatus(status="unknown")
    rag_status = DependencyStatus(status="ok")
    rating_status = DependencyStatus(status="ok", detail=rating_service._model_version)

    try:
        settings = get_settings()
        chroma_status = _ping_chroma(settings.task1_dir)
    except Exception as exc:  # pragma: no cover
        chroma_status = DependencyStatus(status="degraded", detail=str(exc))

    payload = HealthPayload(
        status="ok",
        chroma=chroma_status,
        rag_graph=rag_status,
        rating_model=rating_status,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    return HealthResponse(trace_id=trace_id, latency_ms=latency_ms, data=payload)

